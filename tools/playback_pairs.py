"""
playback_pairs.py
─────────────────
Replays a session_NNN_pairs.jsonl file back into a game window.

  1. Run this script
  2. Alt-tab to your game within 3 seconds (countdown in terminal)
  3. Sit back — keyboard + mouse actions replay at original timing

Controls (from terminal, not the game):
  Ctrl+C  → stop playback immediately

Usage:
    python playback_pairs.py session_001_pairs.jsonl
    python playback_pairs.py session_001_pairs.jsonl --start-frame 50
    python playback_pairs.py session_001_pairs.jsonl --start-frame 50 --end-frame 100
    python playback_pairs.py session_001_pairs.jsonl --speed 0.5   # half speed
    python playback_pairs.py session_001_pairs.jsonl --skip-idle   # skip idle frames

Action string format (Lumine paper):
    DX DY DZ ; chunk1 ; chunk2 ; chunk3 ; chunk4 ; chunk5 ; chunk6
    Each chunk = comma-separated keys held during that 33ms sub-window.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import time
from pathlib import Path

if platform.system() != "Windows":
    raise SystemExit("Playback is Windows-only (uses SendInput).")

from ctypes import wintypes

# ── Win32 SendInput setup ──────────────────────────────────────────────────

user32   = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
winmm    = ctypes.windll.winmm   # for timeBeginPeriod / timeEndPeriod

INPUT_KEYBOARD = 1
INPUT_MOUSE    = 0

KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_SCANCODE    = 0x0008
MOUSEEVENTF_MOVE      = 0x0001
MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP   = 0x0010
MOUSEEVENTF_MIDDLEDOWN= 0x0020
MOUSEEVENTF_MIDDLEUP  = 0x0040
MOUSEEVENTF_XDOWN     = 0x0080
MOUSEEVENTF_XUP       = 0x0100

XBUTTON1 = 0x0001
XBUTTON2 = 0x0002

# Virtual key codes for named keys
VK_MAP: dict[str, int] = {
    "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45,
    "f": 0x46, "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4A,
    "k": 0x4B, "l": 0x4C, "m": 0x4D, "n": 0x4E, "o": 0x4F,
    "p": 0x50, "q": 0x51, "r": 0x52, "s": 0x53, "t": 0x54,
    "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58, "y": 0x59,
    "z": 0x5A,
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    "space":      0x20,
    "lshift":     0xA0, "rshift": 0xA1, "shift": 0xA0,
    "lctrl":      0xA2, "rctrl":  0xA3, "ctrl":  0xA2,
    "lalt":       0xA4, "ralt":   0xA5, "alt":   0xA4,
    "enter":      0x0D,
    "esc":        0x1B,
    "tab":        0x09,
    "backspace":  0x08,
    "delete":     0x2E,
    "insert":     0x2D,
    "home":       0x24,
    "end":        0x23,
    "page_up":    0x21,
    "page_down":  0x22,
    "up":         0x26,
    "down":       0x28,
    "left":       0x25,
    "right":      0x27,
    "f1":  0x70, "f2":  0x71, "f3":  0x72, "f4":  0x73,
    "f5":  0x74, "f6":  0x75, "f7":  0x76, "f8":  0x77,
    "f9":  0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "num_lock":   0x90,
    "scroll_lock":0x91,
    "caps_lock":  0x14,
    "lwin":       0x5B, "rwin": 0x5C,
    "numpad_0":   0x60, "numpad_1": 0x61, "numpad_2": 0x62,
    "numpad_3":   0x63, "numpad_4": 0x64, "numpad_5": 0x65,
    "numpad_6":   0x66, "numpad_7": 0x67, "numpad_8": 0x68,
    "numpad_9":   0x69,
}

# Mouse button keys — handled via MOUSEEVENTF_* not VK
# Names match recorder: lbutton/rbutton/mbutton/xbutton1/xbutton2
MOUSE_BUTTON_KEYS = {"lbutton", "rbutton", "mbutton", "xbutton1", "xbutton2"}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("_input", _INPUTunion)]


def _send_inputs(inputs: list[INPUT]) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def _key_input(vk: int, key_up: bool) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp._input.ki.wVk = vk
    inp._input.ki.dwFlags = KEYEVENTF_KEYUP if key_up else 0
    return inp


def _mouse_move_input(dx: int, dy: int) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp._input.mi.dx = dx
    inp._input.mi.dy = dy
    inp._input.mi.dwFlags = MOUSEEVENTF_MOVE
    return inp


def _mouse_button_input(button: str, down: bool) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    flags = {
        ("lbutton", True):  MOUSEEVENTF_LEFTDOWN,
        ("lbutton", False): MOUSEEVENTF_LEFTUP,
        ("rbutton", True):  MOUSEEVENTF_RIGHTDOWN,
        ("rbutton", False): MOUSEEVENTF_RIGHTUP,
        ("mbutton", True):  MOUSEEVENTF_MIDDLEDOWN,
        ("mbutton", False): MOUSEEVENTF_MIDDLEUP,
        ("xbutton1", True):  MOUSEEVENTF_XDOWN,
        ("xbutton1", False): MOUSEEVENTF_XUP,
        ("xbutton2", True):  MOUSEEVENTF_XDOWN,
        ("xbutton2", False): MOUSEEVENTF_XUP,
    }
    inp._input.mi.dwFlags = flags.get((button, down), 0)
    if button == "xbutton1":
        inp._input.mi.mouseData = XBUTTON1
    elif button == "xbutton2":
        inp._input.mi.mouseData = XBUTTON2
    return inp


# ── Action string parser ───────────────────────────────────────────────────

def parse_action(action_str: str) -> tuple[int, int, list[set[str]]]:
    """
    Parse 'DX DY DZ ; c1 ; c2 ; c3 ; c4 ; c5 ; c6'
    Returns (dx, dy, [chunk_set, ...])
    """
    parts = [p.strip() for p in action_str.split(";")]
    mouse_part = parts[0].split()
    dx = int(mouse_part[0]) if len(mouse_part) > 0 else 0
    dy = int(mouse_part[1]) if len(mouse_part) > 1 else 0

    chunks: list[set[str]] = []
    for chunk_str in parts[1:]:
        if chunk_str:
            keys = {k.strip() for k in chunk_str.split(",") if k.strip()}
        else:
            keys = set()
        chunks.append(keys)

    return dx, dy, chunks


# ── Core replay logic ──────────────────────────────────────────────────────

CHUNK_S = 200 / 6 / 1000   # ~0.0333s per chunk

# How many SendInput mouse calls to spread across one 33ms chunk.
# 2 sub-steps per chunk = one send every ~16ms, close to real hardware polling.
_MOUSE_SUBSTEPS = 2


def _spread_delta(total: int, n: int) -> list[int]:
    """
    Distribute total mouse delta across n steps as evenly as possible,
    preserving the exact integer sum (no rounding loss).
    e.g. _spread_delta(10, 3) -> [4, 3, 3]
         _spread_delta(-7, 3) -> [-3, -2, -2]
    """
    if n <= 0 or total == 0:
        return [0] * max(n, 0)
    base, rem = divmod(abs(total), n)
    sign = 1 if total > 0 else -1
    return [sign * (base + (1 if i < rem else 0)) for i in range(n)]


def replay_frame(
    dx: int,
    dy: int,
    chunks: list[set[str]],
    prev_held: set[str],
    warned_keys: set[str],
) -> set[str]:
    """
    Execute one 200ms frame:
      - Spread mouse delta evenly across all 6 chunks x _MOUSE_SUBSTEPS,
        injected with MOUSEEVENTF_MOVE so games receive it via Raw Input.
      - For each 33ms chunk, press/release keys that changed vs previous chunk.
    Returns the held key set after the last chunk (for next frame's seeding).

    Why spread instead of one shot:
      Games read Raw Input events accumulated each frame (~16ms at 60fps).
      A single huge delta arrives as one raw event — some engines clamp or
      ignore it. Spreading across 12 sub-steps across 200ms matches how
      a real mouse produces events at ~125-1000Hz.
    """
    n_chunks = len(chunks)
    total_steps = n_chunks * _MOUSE_SUBSTEPS

    # Pre-compute per-step mouse deltas that sum exactly to (dx, dy)
    dx_steps = _spread_delta(dx, total_steps)
    dy_steps = _spread_delta(dy, total_steps)

    held = set(prev_held)

    for c, chunk_keys in enumerate(chunks):
        chunk_start = time.perf_counter()

        # ── Key state changes at chunk boundary ───────────────────────────
        newly_down = chunk_keys - held
        newly_up   = held - chunk_keys

        key_inputs: list[INPUT] = []
        for key in newly_up:
            if key in MOUSE_BUTTON_KEYS:
                key_inputs.append(_mouse_button_input(key, False))
            elif key in VK_MAP:
                key_inputs.append(_key_input(VK_MAP[key], key_up=True))
            elif key not in warned_keys:
                print(f"  [warn] unknown key '{key}' — skipped")
                warned_keys.add(key)
        for key in newly_down:
            if key in MOUSE_BUTTON_KEYS:
                key_inputs.append(_mouse_button_input(key, True))
            elif key in VK_MAP:
                key_inputs.append(_key_input(VK_MAP[key], key_up=False))
            elif key not in warned_keys:
                print(f"  [warn] unknown key '{key}' — skipped")
                warned_keys.add(key)
        if key_inputs:
            _send_inputs(key_inputs)

        held = set(chunk_keys)

        # ── Mouse sub-steps spread across the chunk ───────────────────────
        sub_duration = CHUNK_S / _MOUSE_SUBSTEPS
        for s in range(_MOUSE_SUBSTEPS):
            sub_start = time.perf_counter()
            step_idx  = c * _MOUSE_SUBSTEPS + s
            sdx = dx_steps[step_idx]
            sdy = dy_steps[step_idx]
            if sdx != 0 or sdy != 0:
                _send_inputs([_mouse_move_input(sdx, sdy)])
            elapsed = time.perf_counter() - sub_start
            remaining = sub_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

    return held


def release_all(held: set[str]) -> None:
    """Release all currently held keys/buttons — call on stop."""
    inputs: list[INPUT] = []
    for key in held:
        if key in MOUSE_BUTTON_KEYS:
            inputs.append(_mouse_button_input(key, False))
        elif key in VK_MAP:
            inputs.append(_key_input(VK_MAP[key], key_up=True))
    if inputs:
        _send_inputs(inputs)


# ── Main playback loop ─────────────────────────────────────────────────────

def countdown(seconds: int) -> None:
    print(f"\nAlt-tab to your game now!")
    for i in range(seconds, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  Starting playback!     ")


def load_pairs(path: Path, start_frame: int, end_frame: int | None, skip_idle: bool) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["frame_index"] < start_frame:
                continue
            if end_frame is not None and r["frame_index"] > end_frame:
                break
            if skip_idle and r.get("is_idle", False):
                continue
            pairs.append(r)
    return pairs


def playback(
    pairs_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    speed: float = 1.0,
    skip_idle: bool = False,
    countdown_s: int = 3,
) -> None:
    pairs = load_pairs(pairs_path, start_frame, end_frame, skip_idle)
    if not pairs:
        print("No frames to replay.")
        return

    total = len(pairs)
    print(f"Loaded {total} frames from {pairs_path.name}")
    print(f"  Start frame : {pairs[0]['frame_index']}  (t={pairs[0]['t_start_ms']:.0f}ms)")
    print(f"  End frame   : {pairs[-1]['frame_index']} (t={pairs[-1]['t_start_ms']:.0f}ms)")
    print(f"  Speed       : {speed}x")
    print(f"  Skip idle   : {skip_idle}")

    countdown(countdown_s)

    # Set Windows timer resolution to 1ms for accurate sleep() calls.
    # Default resolution is ~15ms which causes drift across the 12 sub-steps per frame.
    # Always restored in finally.
    winmm.timeBeginPeriod(1)

    held: set[str] = set()
    warned_keys: set[str] = set()
    last_frame_idx: int | str | None = None

    try:
        for i, pair in enumerate(pairs):
            frame_start = time.perf_counter()
            last_frame_idx = pair["frame_index"]

            dx, dy, chunks = parse_action(pair["action"])
            held = replay_frame(dx, dy, chunks, held, warned_keys)

            # Print progress every 50 frames
            if i % 50 == 0:
                pct = 100 * i / total
                print(f"  Frame {pair['frame_index']:>5}  ({pct:.0f}%)  held={sorted(held)}", end="\r")

            # Wait until the next included frame's original start time.
            # This preserves real timing even when idle frames were skipped.
            elapsed = time.perf_counter() - frame_start
            if i + 1 < total:
                current_t = float(pair["t_start_ms"])
                next_t = float(pairs[i + 1]["t_start_ms"])
                target_sleep = max((next_t - current_t) / 1000.0 / speed, 0.0)
                remaining = target_sleep - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\nPlayback interrupted.")
    finally:
        winmm.timeEndPeriod(1)
        release_all(held)
        stopped_at = last_frame_idx if last_frame_idx is not None else "N/A"
        print(f"\nAll keys released. Stopped at frame {stopped_at}.")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a session_NNN_pairs.jsonl into your game."
    )
    parser.add_argument("input", help="Path to *_pairs.jsonl")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame index to start from. Default: 0")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Frame index to stop at (inclusive). Default: end of file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier. 0.5=half speed, 2.0=double. Default: 1.0")
    parser.add_argument("--skip-idle", action="store_true",
                        help="Skip frames where is_idle=true")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Seconds to wait before starting (alt-tab time). Default: 3")
    args = parser.parse_args()

    playback(
        pairs_path  = Path(args.input),
        start_frame = args.start_frame,
        end_frame   = args.end_frame,
        speed       = args.speed,
        skip_idle   = args.skip_idle,
        countdown_s = args.countdown,
    )


if __name__ == "__main__":
    main()
