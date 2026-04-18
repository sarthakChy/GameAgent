"""
input_recorder_orchestrator.py
───────────────────────────────
Runs silently in the background while you play.
Coordinates both input logging (windows_input_recorder.py) and
OBS screen recording (obs_recorder.py) from a single hotkey.

  F9  → start input logger + OBS recording  (two short beeps)
  F10 → stop both + write meta.json          (one low beep)

Both keys are suppressed — the game never sees them.
Output per session (recordings/session_NNN/):
    session_NNN.jsonl       <- keyboard + mouse log
    session_NNN.mkv         <- OBS screen recording
    session_NNN_meta.json   <- timestamps for alignment in convert_session.py

OBS WebSocket config (from .env or CLI):
    OBS_HOST     = localhost
    OBS_PORT     = 4455
    OBS_PASSWORD = your_password

Usage
─────
  python input_recorder_orchestrator.py
  python input_recorder_orchestrator.py --output-dir C:\\recordings
  python input_recorder_orchestrator.py --no-obs          # input only, no OBS
  python input_recorder_orchestrator.py --obs-host localhost --obs-port 4455

Then alt-tab to the game and forget about the terminal.
Press Ctrl+C in the terminal (or close it) to fully exit.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import threading
import time
import winsound
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if platform.system() != "Windows":
    raise SystemExit("This orchestrator is Windows-only.")

# ── Import the recorder from the sibling file ──────────────────────────────
import importlib.util, sys

_HERE = Path(__file__).parent
_RECORDER_PATH = _HERE / "windows_input_recorder.py"
if not _RECORDER_PATH.exists():
    raise SystemExit(
        f"windows_input_recorder.py not found next to this script.\n"
        f"Expected: {_RECORDER_PATH}"
    )
_spec = importlib.util.spec_from_file_location("windows_input_recorder", _RECORDER_PATH)
_mod = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
_spec.loader.exec_module(_mod)                         # type: ignore[union-attr]
InputRecorder = _mod.InputRecorder

# ── Import OBSRecorder from sibling file ───────────────────────────────────
_OBS_RECORDER_PATH = _HERE / "obs_recorder.py"
_obs_mod = None
OBSRecorder = None

if _OBS_RECORDER_PATH.exists():
    try:
        _obs_spec = importlib.util.spec_from_file_location("obs_recorder", _OBS_RECORDER_PATH)
        _obs_mod = importlib.util.module_from_spec(_obs_spec)   # type: ignore[arg-type]
        _obs_spec.loader.exec_module(_obs_mod)                   # type: ignore[union-attr]
        OBSRecorder = _obs_mod.OBSRecorder
    except BaseException as _obs_import_err:
        print(f"[warn] Could not import obs_recorder.py: {_obs_import_err}")
        print("[warn] OBS recording will be disabled.")
else:
    print("[warn] obs_recorder.py not found — OBS recording disabled.")

# ── Minimal Win32 bindings needed for the global hotkey hook ───────────────
from ctypes import wintypes

user32   = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

if not hasattr(wintypes, "LRESULT"):
    wintypes.LRESULT = ctypes.c_ssize_t if hasattr(ctypes, "c_ssize_t") else ctypes.c_longlong

WH_KEYBOARD_LL = 13
HC_ACTION       = 0
WM_KEYDOWN      = 0x0100
WM_SYSKEYDOWN   = 0x0104
WM_KEYUP        = 0x0101
WM_SYSKEYUP     = 0x0105
WM_QUIT         = 0x0012

HOOKPROC = ctypes.WINFUNCTYPE(wintypes.LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

VK_NAME_MAP: dict[int, str] = {
    0x70: "f1",  0x71: "f2",  0x72: "f3",  0x73: "f4",
    0x74: "f5",  0x75: "f6",  0x76: "f7",  0x77: "f8",
    0x78: "f9",  0x79: "f10", 0x7A: "f11", 0x7B: "f12",
    0x1B: "esc", 0x60: "numpad_0",
}

def _vk_to_name(vk: int) -> str:
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk).lower()
    return VK_NAME_MAP.get(vk, f"vk_{vk}")


# ── Audio feedback ──────────────────────────────────────────────────────────

def _beep_start() -> None:
    """Two short high beeps — recording started."""
    threading.Thread(target=_beep_start_sync, daemon=True).start()

def _beep_start_sync() -> None:
    winsound.Beep(880, 100)
    time.sleep(0.05)
    winsound.Beep(880, 100)

def _beep_stop() -> None:
    """One longer low beep — recording stopped."""
    threading.Thread(target=lambda: winsound.Beep(440, 300), daemon=True).start()

def _beep_already_running() -> None:
    """Three rapid beeps — tried to start while already recording."""
    threading.Thread(target=_beep_already_running_sync, daemon=True).start()

def _beep_already_running_sync() -> None:
    for _ in range(3):
        winsound.Beep(660, 80)
        time.sleep(0.04)


# ── Session file naming ─────────────────────────────────────────────────────

def _next_session_dir(output_dir: Path) -> tuple[Path, str]:
    """
    Returns (session_dir, session_name) where session_name is e.g. 'session_001'.
    Creates session_dir. NNN is one higher than the highest existing session folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in output_dir.iterdir() if p.is_dir() and re.fullmatch(r"session_\d{3}", p.name))
    if not existing:
        name = "session_001"
    else:
        last = existing[-1].name
        m = re.search(r"(\d+)$", last)
        n = int(m.group(1)) + 1 if m else 1
        name = f"session_{n:03d}"
    session_dir = output_dir / name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, name


def _compute_obs_input_offset_ms(input_log_path: Path, obs_started_utc_str: str) -> float:
    """
    Compute how many ms into the input log the OBS video starts.

    Positive → input log started before OBS (trim input log start in convert_session).
    Negative → OBS started before input log (trim video start in post — rare).

    Method:
      session_start event gives wall_time_utc + elapsed_ms.
      Input zero point = session_start.wall_time_utc - session_start.elapsed_ms.
      offset_ms = (obs_first_frame_wall - input_zero_wall) in ms.
    """
    from datetime import timedelta

    session_start_wall_utc   = None
    session_start_elapsed_ms = None

    with open(input_log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event_type") == "session_start":
                session_start_wall_utc   = e.get("wall_time_utc")
                session_start_elapsed_ms = e.get("elapsed_ms", 0.0)
                break

    if session_start_wall_utc is None:
        raise ValueError("No session_start event found in input log")

    session_start_dt   = datetime.fromisoformat(session_start_wall_utc)
    input_zero_wall_dt = session_start_dt - timedelta(milliseconds=session_start_elapsed_ms)
    obs_started_dt     = datetime.fromisoformat(obs_started_utc_str)

    return round((obs_started_dt - input_zero_wall_dt).total_seconds() * 1000, 3)


# ── Orchestrator ────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(
        self,
        output_dir: Path,
        start_key: str = "f9",
        stop_key: str  = "f10",
        poll_interval_ms: int = 5,
        obs_host: str = "localhost",
        obs_port: int = 4455,
        obs_password: str = "",
        use_obs: bool = True,
    ):
        self.output_dir       = output_dir
        self.start_key        = start_key.lower()
        self.stop_key         = stop_key.lower()
        self.poll_interval_ms = poll_interval_ms
        self.obs_host         = obs_host
        self.obs_port         = obs_port
        self.obs_password     = obs_password
        # Disable OBS if module not available or explicitly disabled
        self.use_obs          = use_obs and (OBSRecorder is not None)

        self._recorder: Optional[InputRecorder] = None
        self._obs_recorder: Optional[object]    = None   # OBSRecorder instance
        self._recorder_lock   = threading.Lock()
        self._hook_handle: Optional[int] = None
        self._hook_ref: Optional[object]  = None
        self._thread_id: Optional[int]    = None
        self._quit_event      = threading.Event()

    # ── public ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until Ctrl+C or the process is killed."""
        self._install_hook()

        obs_status = f"OBS @ {self.obs_host}:{self.obs_port}" if self.use_obs else "OBS disabled"
        print(
            f"\n  Orchestrator ready.\n"
            f"  [{self.start_key.upper()}] start recording\n"
            f"  [{self.stop_key.upper()}]  stop  recording\n"
            f"  Output -> {self.output_dir}\n"
            f"  {obs_status}\n"
            f"  (Ctrl+C here to exit)\n"
        )
        try:
            self._message_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_recording()
            self._remove_hook()
            print("\nOrchestrator exited.")

    # ── hook ────────────────────────────────────────────────────────────────

    def _install_hook(self) -> None:
        self._hook_ref = HOOKPROC(self._keyboard_hook_proc)
        hinstance = kernel32.GetModuleHandleW(None)
        self._hook_handle = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._hook_ref, hinstance, 0
        )
        if not self._hook_handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def _remove_hook(self) -> None:
        if self._hook_handle:
            user32.UnhookWindowsHookEx(self._hook_handle)
            self._hook_handle = None

    def _message_loop(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        msg = _mod.MSG()  # reuse MSG from windows_input_recorder to match Win32 type exactly
        while not self._quit_event.is_set():
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == 0 or result == -1:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _quit_message_loop(self) -> None:
        if self._thread_id is not None:
            self._quit_event.set()
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    # ── hook callback ───────────────────────────────────────────────────────

    def _keyboard_hook_proc(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code != HC_ACTION:
            return user32.CallNextHookEx(self._hook_handle, n_code, w_param, l_param)

        # Handle both keydown AND keyup for hotkeys so neither reaches the game or recorder
        is_key_event = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP)
        if not is_key_event:
            return user32.CallNextHookEx(self._hook_handle, n_code, w_param, l_param)

        # Read the virtual-key code from KBDLLHOOKSTRUCT.vkCode (first DWORD)
        vk = ctypes.cast(l_param, ctypes.POINTER(ctypes.c_ulong)).contents.value
        name = _vk_to_name(vk)

        is_down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)

        if name == self.start_key:
            if is_down:
                threading.Thread(target=self._on_start_key, daemon=True).start()
            return 1   # suppress keydown AND keyup — game + recorder never see it

        if name == self.stop_key:
            if is_down:
                threading.Thread(target=self._on_stop_key, daemon=True).start()
            return 1   # suppress keydown AND keyup

        return user32.CallNextHookEx(self._hook_handle, n_code, w_param, l_param)

    # ── session control ─────────────────────────────────────────────────────

    def _on_start_key(self) -> None:
        with self._recorder_lock:
            if self._recorder is not None:
                _beep_already_running()
                print("[!] Already recording. Press F10 to stop first.")
                return

            session_dir, session_name = _next_session_dir(self.output_dir)
            print(f"\n▶  Session: {session_name}  ->  {session_dir}")

            # ── Start OBS first (it takes longer to init) ──────────────────
            obs_started_utc = None
            if self.use_obs:
                try:
                    self._obs_recorder = OBSRecorder(
                        session_dir  = session_dir,
                        session_name = session_name,
                        host         = self.obs_host,
                        port         = self.obs_port,
                        password     = self.obs_password,
                    )
                    self._obs_recorder.connect()
                    obs_started_utc = self._obs_recorder.start_recording()
                    print(f"  OBS recording started at {obs_started_utc}")
                except Exception as exc:
                    print(f"  [warn] OBS failed: {exc}")
                    print("  [warn] Continuing with input logging only.")
                    self._obs_recorder = None

            # ── Start input recorder ───────────────────────────────────────
            input_path = session_dir / f"{session_name}.jsonl"
            self._recorder = InputRecorder(
                output_path      = input_path,
                stop_key         = "__never__",
                poll_interval_ms = self.poll_interval_ms,
                suppress_keys    = {self.start_key, self.stop_key},
            )

            # ── Write meta.json with both start timestamps ─────────────────
            meta = {
                "session_name":       session_name,
                "input_log":          str(input_path),
                "obs_video":          str(session_dir / f"{session_name}.mkv") if self.use_obs else None,
                "obs_started_utc":    obs_started_utc,
                "input_started_utc":  datetime.now(timezone.utc).isoformat(),
                # offset_ms filled in after stop (needs input session_start elapsed_ms)
                "obs_input_offset_ms": None,
            }
            meta_path = session_dir / f"{session_name}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))
            self._meta_path    = meta_path
            self._session_name = session_name
            self._session_dir  = session_dir

        _beep_start()
        threading.Thread(target=self._recorder.start, daemon=True, name="RecorderLoop").start()

    def _on_stop_key(self) -> None:
        with self._recorder_lock:
            if self._recorder is None:
                return
            r = self._recorder
            obs_r = self._obs_recorder
            meta_path = getattr(self, "_meta_path", None)
            self._recorder     = None
            self._obs_recorder = None

        _beep_stop()

        # ── Stop input recorder ────────────────────────────────────────────
        r.close()
        print("■  Input recording stopped.")

        # ── Stop OBS ──────────────────────────────────────────────────────
        if obs_r is not None:
            try:
                mkv_path = obs_r.stop_recording()
                obs_r.disconnect()
                print(f"■  OBS recording stopped -> {mkv_path}")
            except Exception as exc:
                print(f"  [warn] OBS stop failed: {exc}")

        # ── Update meta.json with computed offset ──────────────────────────
        # offset = how many ms into the input log the video starts.
        # We derive it by comparing OBS first-frame wall time against the
        # input logger's zero point (session_start.wall_time_utc - elapsed_ms).
        if meta_path and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta["stopped_utc"] = datetime.now(timezone.utc).isoformat()

                # Compute offset from the input log's session_start event
                input_log_path = Path(meta.get("input_log", ""))
                obs_started_utc_str = meta.get("obs_started_utc")

                if input_log_path.exists() and obs_started_utc_str:
                    offset_ms = _compute_obs_input_offset_ms(
                        input_log_path, obs_started_utc_str
                    )
                    meta["obs_input_offset_ms"] = offset_ms
                    print(f"  OBS→input offset: {offset_ms:+.1f}ms "
                          f"({'OBS started first' if offset_ms < 0 else 'input started first'})")

                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception as exc:
                print(f"  [warn] Could not update meta.json: {exc}")

        print("■  Session complete.\n")

        # Single-session mode: after stop, exit orchestrator.
        self._quit_message_loop()

    def _stop_recording(self) -> None:
        """Called on exit to cleanly close any open session."""
        with self._recorder_lock:
            r = self._recorder
            obs_r = self._obs_recorder
            self._recorder     = None
            self._obs_recorder = None
        if r is not None:
            r.close()
        if obs_r is not None:
            try:
                obs_r.stop_recording()
                obs_r.disconnect()
            except Exception:
                pass


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Background orchestrator: coordinates input logging + OBS recording.\n"
            "Press the start key to begin a session, the stop key to end it.\n"
            "Both keys are suppressed from the game."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "recordings"),
        help="Directory to save session folders. Default: <repo>/recordings",
    )
    parser.add_argument(
        "--start-key", default="f9",
        help="Hotkey to START a session. Default: f9",
    )
    parser.add_argument(
        "--stop-key", default="f10",
        help="Hotkey to STOP a session. Default: f10",
    )
    parser.add_argument(
        "--poll-interval-ms", type=int, default=5,
        help="Input state-snapshot interval in ms. Default: 5",
    )
    # OBS args — defaults come from .env
    parser.add_argument(
        "--obs-host",
        default=os.getenv("OBS_HOST", os.getenv("OBS_SERVER_IP", "localhost")),
        help="OBS WebSocket host. Default: OBS_HOST env / localhost",
    )
    parser.add_argument(
        "--obs-port",
        type=int,
        default=int(os.getenv("OBS_PORT", "4455")),
        help="OBS WebSocket port. Default: OBS_PORT env / 4455",
    )
    parser.add_argument(
        "--obs-password",
        default=os.getenv("OBS_PASSWORD", os.getenv("OBS_PASS", "")),
        help="OBS WebSocket password. Default: OBS_PASSWORD env var",
    )
    parser.add_argument(
        "--no-obs", action="store_true",
        help="Disable OBS recording — input logging only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = Orchestrator(
        output_dir       = Path(args.output_dir),
        start_key        = args.start_key,
        stop_key         = args.stop_key,
        poll_interval_ms = args.poll_interval_ms,
        obs_host         = args.obs_host,
        obs_port         = args.obs_port,
        obs_password     = args.obs_password,
        use_obs          = not args.no_obs,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()