"""
obs_recorder.py
───────────────
Standalone OBS recording controller using the OBS WebSocket v5 API.
Mirrors the structure of windows_input_recorder.py — can be used alone
or imported by input_recorder_orchestrator.py.

Requires:
    pip install obsws-python python-dotenv

OBS setup (Tools → WebSocket Server Settings):
    - Enable WebSocket server: ON
    - Server Port: 4455
    - Enable Authentication: ON (set a password)

.env file (in project root):
    OBS_HOST     = localhost
    OBS_PORT     = 4455
    OBS_PASSWORD = your_password

Usage (standalone):
    python obs_recorder.py --session-dir recordings/session_001
    python obs_recorder.py --session-dir recordings/session_001 --stop-after 60
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Dependency check ──────────────────────────────────────────────────────────

try:
    import obsws_python as obs
except ImportError:
    raise SystemExit(
        "obsws-python not installed.\n"
        "Run: pip install obsws-python"
    )

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env not required — env vars can be set manually


# ── Config defaults (overridden by .env) ─────────────────────────────────────

DEFAULT_HOST     = os.getenv("OBS_HOST", "localhost")
DEFAULT_PORT     = int(os.getenv("OBS_PORT", "4455"))
DEFAULT_PASSWORD = os.getenv("OBS_PASSWORD", os.getenv("OBS_PASS", ""))


# ── OBSRecorder ───────────────────────────────────────────────────────────────

class OBSRecorder:
    """
    Controls OBS recording for one session.

    Lifecycle:
        recorder = OBSRecorder(session_dir, host, port, password)
        recorder.connect()          # connect to OBS WebSocket
        recorder.start_recording()  # begin recording, returns wall-time ms
        ...
        recorder.stop_recording()   # stop, rename file, returns final .mkv path
        recorder.disconnect()

    Or use as a context manager:
        with OBSRecorder(...) as recorder:
            recorder.start_recording()
            ...
            recorder.stop_recording()
    """

    def __init__(
        self,
        session_dir: Path,
        session_name: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        password: str = DEFAULT_PASSWORD,
        connect_timeout: int = 5,
    ):
        self.session_dir  = Path(session_dir)
        self.session_name = session_name   # e.g. "session_001"
        self.host         = host
        self.port         = port
        self.password     = password
        self.connect_timeout = connect_timeout

        self._req_client: Optional[obs.ReqClient]   = None
        self._evt_client: Optional[obs.EventClient] = None

        # Timing — set when recording actually starts (RecordStateChanged event)
        self._record_started_event = threading.Event()
        self._obs_started_wall_ns:  Optional[int]  = None   # high-res wall time
        self._obs_started_wall_utc: Optional[str]  = None   # ISO string

        # Path OBS writes to (from StopRecord response)
        self._raw_output_path: Optional[Path] = None
        # Path after rename to match session name
        self.output_path: Optional[Path] = None

        self._recording = False
        self._lock = threading.Lock()

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connect both the request client and event client to OBS."""
        try:
            self._req_client = obs.ReqClient(
                host=self.host,
                port=self.port,
                password=self.password,
                timeout=self.connect_timeout,
            )
            self._evt_client = obs.EventClient(
                host=self.host,
                port=self.port,
                password=self.password,
                timeout=self.connect_timeout,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to OBS at {self.host}:{self.port}.\n"
                f"Make sure OBS is open and WebSocket server is enabled.\n"
                f"Error: {exc}"
            )

        # Register event handler for recording state changes
        self._evt_client.callback.register(self.on_record_state_changed)

    def disconnect(self) -> None:
        if self._evt_client:
            try:
                self._evt_client.disconnect()
            except Exception:
                pass
            self._evt_client = None

        if self._req_client:
            try:
                self._req_client.base_client.ws.close()
            except Exception:
                pass
            self._req_client = None

    # ── Event handler ─────────────────────────────────────────────────────────

    def on_record_state_changed(self, data) -> None:
        """
        Fired by OBS when recording state changes.
        outputState values: OBS_WEBSOCKET_OUTPUT_STARTING → STARTED → STOPPING → STOPPED
        We capture wall time at STARTED — that's when the first frame is written.
        """
        state = getattr(data, "output_state", None) or getattr(data, "outputState", None)

        if state == "OBS_WEBSOCKET_OUTPUT_STARTED":
            with self._lock:
                # Capture precise wall time the moment OBS confirms recording started
                now_ns = time.time_ns()
                self._obs_started_wall_ns  = now_ns
                self._obs_started_wall_utc = datetime.fromtimestamp(
                    now_ns / 1e9, tz=timezone.utc
                ).isoformat()
            self._record_started_event.set()

    # ── Recording control ─────────────────────────────────────────────────────

    def start_recording(self, wait_timeout: float = 10.0) -> str:
        """
        Start OBS recording.
        Blocks until OBS confirms recording has started (RecordStateChanged STARTED).
        Returns the ISO wall-time string of when the first frame was written.

        Raises TimeoutError if OBS doesn't confirm within wait_timeout seconds.
        """
        if self._req_client is None:
            raise RuntimeError("Not connected. Call connect() first.")

        with self._lock:
            self._record_started_event.clear()
            self._obs_started_wall_ns  = None
            self._obs_started_wall_utc = None

        # Check OBS isn't already recording
        status = self._req_client.get_record_status()
        if status.output_active:
            raise RuntimeError("OBS is already recording. Stop it first.")

        self._req_client.start_record()

        # Wait for the STARTED event — this is when the first frame is actually written
        if not self._record_started_event.wait(timeout=wait_timeout):
            raise TimeoutError(
                f"OBS did not confirm recording started within {wait_timeout}s.\n"
                "Check OBS output settings and disk space."
            )

        self._recording = True
        return self._obs_started_wall_utc  # type: ignore[return-value]

    def stop_recording(self) -> Path:
        """
        Stop OBS recording.
        Renames the output file to match the session name.
        Returns the final .mkv path.
        """
        if self._req_client is None:
            raise RuntimeError("Not connected.")
        if not self._recording:
            raise RuntimeError("Not currently recording.")

        resp = self._req_client.stop_record()
        self._recording = False

        # OBS returns the output path in the stop response
        raw_path_str = getattr(resp, "output_path", None)
        if raw_path_str:
            self._raw_output_path = Path(raw_path_str)
        else:
            # Fallback: scan session_dir for newest .mkv if response path missing
            candidates = sorted(
                self.session_dir.parent.glob("*.mkv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                self._raw_output_path = candidates[0]
            else:
                print("  [warn] Could not determine OBS output path — rename manually")
                self.output_path = None
                return None  # type: ignore[return-value]

        # Rename to session name and move into session dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        target = self.session_dir / f"{self.session_name}.mkv"

        # OBS may keep the file handle briefly after StopRecord returns.
        # Retry a few times before giving up on rename.
        move_error: Optional[Exception] = None
        for _ in range(10):
            try:
                shutil.move(str(self._raw_output_path), str(target))
                self.output_path = target
                print(f"  OBS output → {target}")
                move_error = None
                break
            except Exception as exc:
                move_error = exc
                time.sleep(0.5)

        if move_error is not None:
            print(f"  [warn] Could not rename OBS output: {move_error}")
            self.output_path = self._raw_output_path

        return self.output_path

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_start_wall_ns(self) -> Optional[int]:
        """High-resolution wall-clock nanoseconds when OBS started writing frames."""
        return self._obs_started_wall_ns

    def get_start_wall_utc(self) -> Optional[str]:
        """ISO UTC string of when OBS started writing frames."""
        return self._obs_started_wall_utc

    def get_obs_version(self) -> str:
        """Utility — returns OBS version string for diagnostics."""
        if self._req_client is None:
            return "not connected"
        try:
            return self._req_client.get_version().obs_version
        except Exception:
            return "unknown"

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "OBSRecorder":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        if self._recording:
            try:
                self.stop_recording()
            except Exception:
                pass
        self.disconnect()


# ── Standalone CLI ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone OBS recording controller. Mirrors windows_input_recorder.py."
    )
    parser.add_argument(
        "--session-dir", required=True,
        help="Session directory, e.g. recordings/session_001"
    )
    parser.add_argument(
        "--session-name", default=None,
        help="Session name for output file. Default: basename of --session-dir"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"OBS WebSocket host. Default: {DEFAULT_HOST}"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"OBS WebSocket port. Default: {DEFAULT_PORT}"
    )
    parser.add_argument(
        "--password", default=DEFAULT_PASSWORD,
        help="OBS WebSocket password. Default: from OBS_PASSWORD env var"
    )
    parser.add_argument(
        "--stop-after", type=float, default=None,
        help="Automatically stop after N seconds (for testing). Default: wait for Ctrl+C"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    session_dir_input = Path(args.session_dir)
    session_dir = session_dir_input if session_dir_input.is_absolute() else (repo_root / session_dir_input)
    session_name = args.session_name or session_dir.name

    recorder = OBSRecorder(
        session_dir  = session_dir,
        session_name = session_name,
        host         = args.host,
        port         = args.port,
        password     = args.password,
    )

    print(f"Connecting to OBS at {args.host}:{args.port} …")
    recorder.connect()
    print(f"  OBS version: {recorder.get_obs_version()}")

    print("Starting recording …")
    started_utc = recorder.start_recording()
    print(f"  Recording started at {started_utc}")
    print(f"  Session: {session_name}  →  {session_dir}")

    try:
        if args.stop_after:
            print(f"  Auto-stopping after {args.stop_after}s …")
            time.sleep(args.stop_after)
        else:
            print("  Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        output = recorder.stop_recording()
        recorder.disconnect()
        print(f"Done. Output: {output}")


if __name__ == "__main__":
    main()
