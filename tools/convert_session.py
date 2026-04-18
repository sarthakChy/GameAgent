"""
convert_session.py
──────────────────
Converts a session folder (session_NNN.jsonl + session_NNN.mkv + meta.json)
into Lumine-style (frame_path, action_string) pairs ready for model training.

Two modes:
  1. Full mode (default) — requires .mkv + ffmpeg:
       Extracts frames from video at 5Hz, pairs with action strings.
       Output: frames/ folder + pairs.jsonl with frame_path + action.

  2. Input-only mode (--no-video):
       No video needed. Synthetic frame indices only.
       Output: pairs.jsonl with frame_index + action (no images).

Action string format (Lumine paper):
    DX DY DZ ; chunk1 ; chunk2 ; chunk3 ; chunk4 ; chunk5 ; chunk6

Requires ffmpeg on PATH for frame extraction.

Usage:
    # Full pipeline — point at the session folder:
    python convert_session.py recordings/session_001/

    # Input only (no video):
    python convert_session.py recordings/session_001/ --no-video

    # Custom fps:
    python convert_session.py recordings/session_001/ --fps 5
"""

from __future__ import annotations
import argparse
import contextlib
from datetime import datetime
import json
import subprocess
import shutil
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

WINDOW_MS = 200          # one frame = 200ms  (5 Hz)
CHUNKS    = 6            # sub-windows per frame = 6 × 33ms
CHUNK_MS  = WINDOW_MS / CHUNKS

IGNORE_KEYS = {"f9", "f10"}

# ── Session folder resolver ───────────────────────────────────────────────────

def resolve_session(input_arg: str) -> tuple[Path, Path, Path | None, dict | None]:
    """
    Given a session folder or .jsonl path, return:
        (jsonl_path, session_dir, mkv_path_or_None, meta_or_None)
    """
    p = Path(input_arg)

    # If it's a directory, derive file paths from folder name
    if p.is_dir():
        session_dir  = p
        session_name = p.name
        jsonl_path   = p / f"{session_name}.jsonl"
    else:
        # It's a .jsonl file — derive session dir from parent
        jsonl_path   = p
        session_dir  = p.parent
        session_name = session_dir.name

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input log not found: {jsonl_path}")

    mkv_path  = session_dir / f"{session_name}.mkv"
    meta_path = session_dir / f"{session_name}_meta.json"

    mkv  = mkv_path  if mkv_path.exists()  else None
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    return jsonl_path, session_dir, mkv, meta


# ── Load events ───────────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict]:
    events = []
    skipped = 0
    null = chr(0)
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


# ── Session boundaries ────────────────────────────────────────────────────────

def find_time_range(events: list[dict]) -> tuple[float, float]:
    start_ms = end_ms = 0.0
    for e in events:
        if e["event_type"] == "session_start":
            start_ms = e["elapsed_ms"]
        if e["event_type"] == "session_end":
            end_ms = e["elapsed_ms"]
    if end_ms == 0.0 and events:
        end_ms = events[-1]["elapsed_ms"]
        print("  [warn] no session_end — using last event timestamp")
    return start_ms, end_ms


# ── Action events extraction ──────────────────────────────────────────────────

def extract_action_events(events: list[dict], start_ms: float, end_ms: float) -> list[dict]:
    duration_ms = end_ms - start_ms
    useful: list[dict] = []
    held_keys: set[str] = set()
    held_buttons: set[str] = set()

    for e in events:
        etype = e["event_type"]
        ms = e["elapsed_ms"] - start_ms
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
            useful.append({**e, "t_ms": ms,
                           "held_keys": sorted(held_keys),
                           "held_buttons": sorted(held_buttons)})
        elif etype == "mouse_button":
            btn = e.get("button", "")
            useful.append({"event_type": "mouse_button", "action": e["action"],
                           "key": btn, "t_ms": ms,
                           "held_keys": sorted(held_keys),
                           "held_buttons": sorted(held_buttons)})
        elif etype == "mouse_relative":
            useful.append({**e, "t_ms": ms})

    return sorted(useful, key=lambda x: x["t_ms"])


# ── Action string builder ─────────────────────────────────────────────────────

def build_action_string(window_events: list[dict], window_start_ms: float,
                        seed_held: set[str]) -> str:
    dx_total = dy_total = 0
    for e in window_events:
        if e["event_type"] == "mouse_relative":
            dx_total += e.get("dx", 0)
            dy_total += e.get("dy", 0)

    seed_held = set(seed_held) - IGNORE_KEYS
    changes = [(e["t_ms"] - window_start_ms, e["key"], e["action"])
               for e in window_events if e["event_type"] in ("keyboard", "mouse_button")]

    chunks: list[str] = []
    for c in range(CHUNKS):
        chunk_end = (c + 1) * CHUNK_MS
        held = set(seed_held)
        for t_rel, key, action in changes:
            if t_rel < chunk_end:
                if action == "down": held.add(key)
                elif action == "up": held.discard(key)
        held -= IGNORE_KEYS
        chunks.append(",".join(sorted(held)) if held else "")

    return f"{dx_total} {dy_total} 0 ; {' ; '.join(chunks)}"


# ── Frame extraction ──────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def count_video_frames(mkv_path: Path) -> int:
    """
    Count actual encoded video frames using ffprobe -count_packets.
    Works even when MKV has no duration header (common with OBS .mkv output).
    """
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-count_packets", "-show_streams", str(mkv_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return 0
    try:
        data = json.loads(result.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                return int(s.get("nb_read_packets", 0))
    except (json.JSONDecodeError, ValueError):
        pass
    return 0


def get_video_fps(mkv_path: Path) -> float:
    """Get actual video fps from stream header."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(mkv_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return 30.0
    try:
        data = json.loads(result.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                r = s.get("r_frame_rate", "30/1")
                num, den = r.split("/")
                return float(num) / float(den)
    except (json.JSONDecodeError, ValueError, ZeroDivisionError):
        pass
    return 30.0


def extract_frames(
    mkv_path: Path,
    frames_dir: Path,
    fps: int,
    obs_input_offset_ms: float,
) -> int:
    """
    Extract frames from mkv at given fps, accounting for obs_input_offset_ms.

    Counts actual encoded frames first (works without MKV duration header).
    Derives n_5hz from real frame count, not container duration.
    Returns number of frames extracted.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    source_fps   = get_video_fps(mkv_path)
    total_source = count_video_frames(mkv_path)
    if total_source == 0:
        print("  [warn] Could not count video frames")
        n_5hz = None
    else:
        video_duration_s = total_source / source_fps
        n_5hz = int(video_duration_s * fps)
        print(f"  Video: {total_source} frames @ {source_fps:.0f}fps = {video_duration_s:.2f}s")
        print(f"  Extracting {n_5hz} frames @ {fps}Hz ...")

    offset_s = max(0.0, obs_input_offset_ms / 1000.0)
    print(f"  Offset: {obs_input_offset_ms:+.1f}ms "
          f"({'aligned' if abs(obs_input_offset_ms) < 5 else 'skip start'})")

    cmd = [
        "ffmpeg",
        "-ss", str(offset_s),
        "-i", str(mkv_path),
        "-vf", f"fps={fps}",
    ]
    if n_5hz is not None:
        cmd += ["-vframes", str(n_5hz)]
    cmd += [
        "-q:v", "2",
        "-f", "image2",
        str(frames_dir / "%06d.jpg"),
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] ffmpeg failed: {result.stderr[-300:]}")
        return 0

    extracted = sorted(frames_dir.glob("*.jpg"))
    print(f"  Extracted {len(extracted)} frames -> {frames_dir}")
    return len(extracted)

def convert(
    input_arg: str,
    fps: int = 5,
    no_video: bool = False,
    output_path: Path | None = None,
) -> None:
    window_ms = 1000.0 / fps

    # ── Resolve paths ──────────────────────────────────────────────────────
    jsonl_path, session_dir, mkv_path, meta = resolve_session(input_arg)
    session_name = session_dir.name

    use_video = (not no_video) and (mkv_path is not None)
    if not no_video and mkv_path is None:
        print("  [warn] No .mkv found — running in input-only mode.")
        use_video = False
    if use_video and not check_ffmpeg():
        print("  [warn] ffmpeg not found on PATH — running in input-only mode.")
        use_video = False

    obs_input_offset_ms = 0.0
    if meta and meta.get("obs_input_offset_ms") is not None:
        obs_input_offset_ms = float(meta["obs_input_offset_ms"])

    out_path = output_path or (session_dir / f"{session_name}_pairs.jsonl")
    frames_dir = session_dir / "frames"

    print(f"\nSession  : {session_name}")
    print(f"Input log: {jsonl_path.name}")
    print(f"Video    : {mkv_path.name if mkv_path else 'none'}")
    print(f"Mode     : {'full (video + input)' if use_video else 'input only'}")

    # ── Process input log ──────────────────────────────────────────────────
    print(f"\nLoading {jsonl_path.name} …")
    events = load_events(jsonl_path)
    print(f"  {len(events)} total events")

    start_ms, end_ms = find_time_range(events)
    duration_ms = end_ms - start_ms
    print(f"  Duration: {duration_ms/1000:.1f}s")

    action_events = extract_action_events(events, start_ms, end_ms)
    print(f"  Usable action events: {len(action_events)}")

    total_windows = int(duration_ms / window_ms)
    print(f"  Windows at {fps}Hz: {total_windows}")

    # ── Extract video frames ───────────────────────────────────────────────
    n_extracted = 0
    if use_video:
        n_extracted = extract_frames(
            mkv_path, frames_dir, fps, obs_input_offset_ms
        )
        # Video is ground truth — use actual extracted frame count
        # (input log may be longer if OBS stopped slightly before input log)
        if n_extracted > 0:
            total_windows = min(total_windows, n_extracted)

    # ── Build pairs ────────────────────────────────────────────────────────
    IDLE_STR   = "0 0 0 ; " + " ; ".join([""] * CHUNKS)
    event_idx  = 0
    seed_held: set[str] = set()
    written = active = idle = 0

    with open(out_path, "w") as out:
        for win_idx in range(total_windows):
            win_start = win_idx * window_ms
            win_end   = win_start + window_ms
            current_held = set(seed_held)

            # Advance pointer past pre-window events, updating seed
            while event_idx < len(action_events) and action_events[event_idx]["t_ms"] < win_start:
                prev = action_events[event_idx]
                if "held_keys" in prev or "held_buttons" in prev:
                    current_held = (set(prev.get("held_keys", [])) |
                                    set(prev.get("held_buttons", [])))
                event_idx += 1

            # Slice window events
            win_start_idx = event_idx
            while event_idx < len(action_events) and action_events[event_idx]["t_ms"] < win_end:
                event_idx += 1
            win_events = action_events[win_start_idx:event_idx]

            action_str = build_action_string(win_events, win_start, current_held)

            # Carry seed forward from in-window events
            for e in win_events:
                if "held_keys" in e or "held_buttons" in e:
                    current_held = (set(e.get("held_keys", [])) |
                                    set(e.get("held_buttons", [])))
            seed_held = current_held

            is_idle = (action_str == IDLE_STR)

            # Frame path — ffmpeg outputs 1-indexed: 000001.jpg for frame 0
            record: dict = {
                "frame_index": win_idx,
                "t_start_ms":  round(win_start, 1),
                "action":      action_str,
                "is_idle":     is_idle,
            }
            if use_video and n_extracted > 0:
                frame_file = frames_dir / f"{win_idx + 1:06d}.jpg"
                record["frame_path"] = str(frame_file) if frame_file.exists() else None

            out.write(json.dumps(record) + "\n")
            written += 1
            if is_idle: idle += 1
            else: active += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\nDone → {out_path}")
    print(f"  Total pairs  : {written}")
    print(f"  Active frames: {active}  ({100*active/max(written,1):.1f}%)")
    print(f"  Idle frames  : {idle}  ({100*idle/max(written,1):.1f}%)")
    if use_video:
        missing = sum(1 for i in range(written)
                      if not (frames_dir / f"{i+1:06d}.jpg").exists())
        if missing:
            print(f"  [warn] {missing} frame files missing — check ffmpeg output")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert session folder → Lumine-style (frame, action) pairs."
    )
    parser.add_argument(
        "input",
        help="Session folder (e.g. recordings/session_001/) or path to session .jsonl"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output pairs.jsonl path. Default: <session_dir>/<session_name>_pairs.jsonl"
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Sampling rate in Hz. Default: 5 (= 200ms windows)"
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip frame extraction — output action strings only (no frame_path field)"
    )
    args = parser.parse_args()

    # Mirror console output into a run log in the same session folder.
    # We resolve the session once here to determine the target .txt path.
    _, session_dir, _, _ = resolve_session(args.input)
    session_name = session_dir.name
    txt_log_path = session_dir / f"{session_name}_convert_log.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> int:
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self) -> None:
            for s in self.streams:
                s.flush()

    with open(txt_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Run at {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.flush()

        tee = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee_err):
            convert(
                input_arg   = args.input,
                fps         = args.fps,
                no_video    = args.no_video,
                output_path = Path(args.output) if args.output else None,
            )

    print(f"Saved run log -> {txt_log_path}")


if __name__ == "__main__":
    main()