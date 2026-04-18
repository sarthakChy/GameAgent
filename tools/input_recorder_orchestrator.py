"""
input_recorder_orchestrator.py
───────────────────────────────
Runs silently in the background while you play.

  F9  → start a new recording session   (two short beeps)
  F10 → stop the current session         (one low beep)

Both keys are suppressed — the game never sees them.
Output files are auto-numbered: session_001.jsonl, session_002.jsonl, …

Usage
─────
  python input_recorder_orchestrator.py
  python input_recorder_orchestrator.py --output-dir C:\\recordings
  python input_recorder_orchestrator.py --start-key f9 --stop-key f10
  python input_recorder_orchestrator.py --poll-interval-ms 5

Then alt-tab to the game and forget about the terminal.
Press Ctrl+C in the terminal (or close it) to fully exit.
"""
from __future__ import annotations

import argparse
import ctypes
import platform
import re
import threading
import time
import winsound
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

def _next_session_path(output_dir: Path) -> Path:
    """
    Returns output_dir/session_NNN.jsonl where NNN is one higher than
    the highest existing session file (or 001 if none exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("session_*.jsonl"))
    if not existing:
        return output_dir / "session_001.jsonl"
    last = existing[-1].stem          # e.g. "session_007"
    m = re.search(r"(\d+)$", last)
    n = int(m.group(1)) + 1 if m else 1
    return output_dir / f"session_{n:03d}.jsonl"


# ── Orchestrator ────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(
        self,
        output_dir: Path,
        start_key: str = "f9",
        stop_key: str  = "f10",
        poll_interval_ms: int = 5,
    ):
        self.output_dir       = output_dir
        self.start_key        = start_key.lower()
        self.stop_key         = stop_key.lower()
        self.poll_interval_ms = poll_interval_ms

        self._recorder: Optional[InputRecorder] = None
        self._recorder_lock   = threading.Lock()
        self._hook_handle: Optional[int] = None
        self._hook_ref: Optional[object]  = None   # keep alive — GC would crash Python
        self._thread_id: Optional[int]    = None
        self._quit_event      = threading.Event()

    # ── public ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until Ctrl+C or the process is killed."""
        self._install_hook()
        print(
            f"\n  Orchestrator ready.\n"
            f"  [{self.start_key.upper()}] start recording\n"
            f"  [{self.stop_key.upper()}]  stop  recording\n"
            f"  Output → {self.output_dir}\n"
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
                # Already recording — warn the user
                _beep_already_running()
                print("[!] Already recording. Press F10 to stop first.")
                return

            path = _next_session_path(self.output_dir)
            print(f"\n▶  Recording started → {path.name}")
            self._recorder = InputRecorder(
                output_path     = path,
                stop_key        = "__never__",   # we handle stop ourselves
                poll_interval_ms= self.poll_interval_ms,
                suppress_keys   = {self.start_key, self.stop_key},
            )

        _beep_start()
        # Run the recorder's blocking message loop in a background thread
        threading.Thread(target=self._recorder.start, daemon=True, name="RecorderLoop").start()

    def _on_stop_key(self) -> None:
        with self._recorder_lock:
            if self._recorder is None:
                return   # nothing running, ignore silently
            r = self._recorder
            self._recorder = None

        _beep_stop()
        r.close()
        print("■  Recording stopped.\n")

    def _stop_recording(self) -> None:
        """Called on exit to cleanly close any open session."""
        with self._recorder_lock:
            r = self._recorder
            self._recorder = None
        if r is not None:
            r.close()


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Background orchestrator for windows_input_recorder.py.\n"
            "Press the start key to begin a session, the stop key to end it.\n"
            "Both keys are suppressed from the game."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory to save session JSONL files. Default: ./recordings",
    )
    parser.add_argument(
        "--start-key",
        default="f9",
        help="Hotkey to START a recording session. Default: f9",
    )
    parser.add_argument(
        "--stop-key",
        default="f10",
        help="Hotkey to STOP a recording session. Default: f10",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=5,
        help="State-snapshot interval passed to InputRecorder. Default: 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = Orchestrator(
        output_dir      = Path(args.output_dir),
        start_key       = args.start_key,
        stop_key        = args.stop_key,
        poll_interval_ms= args.poll_interval_ms,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
