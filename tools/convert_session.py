"""
convert_session.py
──────────────────
Converts a windows_input_recorder session JSONL into Lumine-style
(frame_index, action_string) pairs, exactly as described in the
Lumine pipeline reference document.

Since this is input-only data (no video), "frames" are synthetic 5Hz
time slots. Each output row maps a 200ms window to its action string.

Action string format (from Lumine paper):
    DX DY DZ ; chunk1 ; chunk2 ; chunk3 ; chunk4 ; chunk5 ; chunk6

Where:
  - DX, DY = summed relative mouse displacement over the full 200ms window
  - DZ     = scroll delta (always 0 here — logger doesn't capture scroll)
  - chunk1..6 = keyboard state at each 33ms sub-window (sorted, comma-joined)

Keys filtered:  f9, f10 (orchestrator hotkeys — never real game input)

Usage:
    python convert_session.py recordings/session_001/session_001.jsonl
    python convert_session.py recordings/session_001/session_001.jsonl --output pairs.jsonl
    python convert_session.py recordings/session_001/session_001.jsonl --fps 5 --chunks 6
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────

WINDOW_MS   = 200          # one frame = 200ms  (5 Hz)
CHUNKS      = 6            # sub-windows per frame = 6 × 33ms
CHUNK_MS    = WINDOW_MS / CHUNKS   # ~33.3ms

# Keys that belong to the orchestrator, not the game
IGNORE_KEYS = {"f9", "f10"}

# ── Load events ──────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict]:
    events = []
    skipped = 0
    null = chr(0)  # null byte — Windows write artifact
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip().strip(null)
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                skipped += 1
                print(f"  [warn] skipping bad line {lineno}: {exc}")
    if skipped:
        print(f"  [warn] {skipped} corrupted line(s) skipped")
    return events


# ── Find session boundaries ───────────────────────────────────────────────

def find_time_range(events: list[dict]) -> tuple[float, float]:
    """
    Returns (start_ms, end_ms) of the usable session window.
    start_ms: elapsed_ms of the session_start event (t=0 reference)
    end_ms:   elapsed_ms of the session_end event
    """
    start_ms = 0.0
    end_ms   = 0.0
    for e in events:
        if e["event_type"] == "session_start":
            start_ms = e["elapsed_ms"]
        if e["event_type"] == "session_end":
            end_ms = e["elapsed_ms"]
    return start_ms, end_ms


# ── Build a sorted flat event list ───────────────────────────────────────

def extract_action_events(events: list[dict], start_ms: float, end_ms: float) -> list[dict]:
    """
    Pull keyboard, mouse_button, and mouse_relative events; re-zero timestamps
    to session start; drop orchestrator keys and out-of-range events.

    Mouse button state comes directly from native mouse_button events logged by
    the recorder (field: "button"). We also track a running held_keys and
    held_buttons set so every event in the useful list carries accurate
    "held_keys" and "held_buttons" snapshots — used by build_action_string
    to seed chunk state across window boundaries.
    """
    duration_ms = end_ms - start_ms
    useful: list[dict] = []

    # Running state — kept in sync with every event we process
    held_keys: set[str] = set()
    held_buttons: set[str] = set()

    for e in events:
        etype = e["event_type"]
        ms = e["elapsed_ms"] - start_ms

        # Update running state from every event that carries snapshots
        if "held_keys" in e:
            held_keys = set(e["held_keys"]) - IGNORE_KEYS
        if "held_buttons" in e:
            held_buttons = set(e["held_buttons"])

        if ms < 0 or ms > duration_ms:
            continue

        if etype == "keyboard":
            key = e.get("key", "")
            if key in IGNORE_KEYS:
                continue
            useful.append({
                **e,
                "t_ms": ms,
                "held_keys": sorted(held_keys),
                "held_buttons": sorted(held_buttons),
            })

        elif etype == "mouse_button":
            # Native event — use "button" field, normalise to "key" for uniform access
            btn = e.get("button", "")
            useful.append({
                "event_type": "mouse_button",
                "action": e["action"],
                "key": btn,           # unified field name used by build_action_string
                "t_ms": ms,
                "held_keys": sorted(held_keys),
                "held_buttons": sorted(held_buttons),
            })

        elif etype == "mouse_relative":
            useful.append({**e, "t_ms": ms})

    return sorted(useful, key=lambda x: x["t_ms"])


# ── Action-string builder ─────────────────────────────────────────────────

def build_action_string(
    window_events: list[dict],
    window_start_ms: float,
    pre_window_events: list[dict],
) -> str:
    """
    Given all events inside one 200ms window (and all action events before it
    for seeding held state), produce the action string:
      DX DY DZ ; chunk1 ; chunk2 ; chunk3 ; chunk4 ; chunk5 ; chunk6

    Chunks include both keyboard keys and mouse buttons (lbutton, rbutton, …).
    """
    # ── Mouse: sum all relative deltas in the window ─────────────────────
    dx_total = 0
    dy_total = 0
    for e in window_events:
        if e["event_type"] == "mouse_relative":
            dx_total += e.get("dx", 0)
            dy_total += e.get("dy", 0)

    # ── Seed held state from the last pre-window event that carries held state ──
    # Every event in the useful list now carries accurate held_keys + held_buttons
    # snapshots (set by extract_action_events). We just take the last one before
    # this window — keyboard OR mouse_button, whichever came most recently.
    seed_held: set[str] = set()
    if pre_window_events:
        for e in reversed(pre_window_events):
            if "held_keys" in e or "held_buttons" in e:
                seed_held = set(e.get("held_keys", []))
                seed_held |= set(e.get("held_buttons", []))
                break
    seed_held -= IGNORE_KEYS

    # ── Build change timeline from events inside the window ───────────────
    # Both keyboard and mouse_button events contribute.
    changes: list[tuple[float, str, str]] = []  # (t_rel_ms, key, "down"/"up")
    for e in window_events:
        if e["event_type"] in ("keyboard", "mouse_button"):
            t_rel = e["t_ms"] - window_start_ms
            changes.append((t_rel, e["key"], e["action"]))

    # ── For each chunk, replay changes up to chunk_start ─────────────────
    chunks: list[str] = []
    for c in range(CHUNKS):
        chunk_start = c * CHUNK_MS
        held = set(seed_held)
        for t_rel, key, action in changes:
            if t_rel < chunk_start:
                if action == "down":
                    held.add(key)
                elif action == "up":
                    held.discard(key)
        held -= IGNORE_KEYS
        chunks.append(",".join(sorted(held)) if held else "")

    return f"{dx_total} {dy_total} 0 ; {' ; '.join(chunks)}"


# ── Main conversion loop ──────────────────────────────────────────────────

def convert(
    input_path: Path,
    output_path: Path,
    fps: int = 5,
) -> None:
    window_ms = 1000.0 / fps

    print(f"Loading {input_path} …")
    events = load_events(input_path)
    print(f"  {len(events)} total events")

    start_ms, end_ms = find_time_range(events)
    duration_ms = end_ms - start_ms
    print(f"  Session duration: {duration_ms/1000:.1f}s  ({duration_ms:.0f}ms)")

    action_events = extract_action_events(events, start_ms, end_ms)
    print(f"  Usable action events: {len(action_events)}")

    total_windows = int(duration_ms / window_ms)
    print(f"  Windows at {fps}Hz: {total_windows}")

    written = 0
    skipped_empty = 0

    with open(output_path, "w") as out:
        for win_idx in range(total_windows):
            win_start = win_idx * window_ms
            win_end   = win_start + window_ms

            # Events that fall inside this window
            win_events = [
                e for e in action_events
                if win_start <= e["t_ms"] < win_end
            ]

            # All action events strictly before this window — used to seed held state
            pre_events = [
                e for e in action_events
                if e["t_ms"] < win_start
            ]

            action_str = build_action_string(win_events, win_start, pre_events)

            # A truly idle window: no mouse movement AND all 6 chunks empty.
            # The idle string is exactly "0 0 0 ;  ;  ;  ;  ;  ; "
            # (six semicolon-separated empty strings, spaces around each ;)
            IDLE_STR = "0 0 0 ; " + " ; ".join([""] * CHUNKS)
            is_idle = (action_str == IDLE_STR)

            record = {
                "frame_index": win_idx,
                "t_start_ms":  round(win_start, 1),
                "action":      action_str,
                "is_idle":     is_idle,
            }
            out.write(json.dumps(record) + "\n")
            written += 1
            if is_idle:
                skipped_empty += 1

    active = written - skipped_empty
    print(f"\nDone → {output_path}")
    print(f"  Total pairs  : {written}")
    print(f"  Active frames: {active}  ({100*active/written:.1f}%)")
    print(f"  Idle frames  : {skipped_empty}  ({100*skipped_empty/written:.1f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert windows_input_recorder JSONL → Lumine-style pairs.jsonl"
    )
    parser.add_argument("input", help="Path to recordings/session_NNN/session_NNN.jsonl")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: <input_stem>_pairs.jsonl)",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Sampling rate in Hz. Default: 5 (= 200ms windows)",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_pairs.jsonl"
    )
    convert(input_path, output_path, fps=args.fps)


if __name__ == "__main__":
    main()
