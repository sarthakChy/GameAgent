# Game Agent Recording Pipeline

A complete end-to-end pipeline for recording and converting game sessions into (frame, action) pairs for training machine learning models.

## Overview

This pipeline captures game sessions with synchronized video and input events, then converts them into structured training data with:
- Video frames extracted at configurable frame rates (default 5 Hz)
- Action strings encoding mouse movements and keyboard/button states
- Metadata for idle vs. active frames

## Project Structure

```
GAMEAGENT/
├── tools/                           # Core conversion and recording tools
│   ├── convert_session.py          # Main converter (session → training pairs)
│   ├── input_recorder_orchestrator.py  # Orchestrates recording sessions
│   ├── windows_input_recorder.py   # Captures Windows mouse/keyboard events
│   ├── obs_recorder.py             # OBS video recording wrapper
│   └── playback_pairs.py           # Replay training pairs from session data
├── recordings/                      # Session data storage
│   ├── session_001/
│   │   ├── session_001.jsonl       # Input events log (JSONL)
│   │   ├── session_001.mkv         # Video recording (Matroska)
│   │   ├── session_001_meta.json   # Session metadata
│   │   ├── session_001_pairs.jsonl # Training pairs (output)
│   │   ├── session_001_convert_log.txt # Conversion run log
│   │   └── frames/                 # Extracted frames (JPEG)
│   └── ...
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Prerequisites

### System Requirements
- **ffmpeg**: Required for video frame extraction
  - Windows: Install via [ffmpeg.org](https://ffmpeg.org) or package manager
  - macOS: `brew install ffmpeg`
  - Linux: `apt-get install ffmpeg` or `yum install ffmpeg`
- **OBS Studio**: For recording video (if using obs_recorder.py)
- **Python 3.8+**

### Software Dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy opencv-python jsonschema
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:
```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify ffmpeg

```bash
ffmpeg -version
ffprobe -version
```

## Usage

### Recording a New Session

Use the input/video recorder orchestrator:

```bash
python tools/input_recorder_orchestrator.py
```

1. Switch to game window and press f9 to start recording the video and inputs. Press f10 to stop recording. 
2. There is no need to switch between the game window and terminal, the hotkey for starting and stopping the recordings will work over the game. You can hear audio ques as well
3. It generates a session folder with `session_NNN.jsonl`, `session_NNN.mkv`, and `session_NNN_meta.json`

### Converting a Session to Training Data

Convert a recorded session into (frame, action) pairs:

```bash
python tools/convert_session.py recordings/session_001/
```

#### Options

- `--fps N`: Sampling rate in Hz (default: 5 = 200ms windows)
  ```bash
  python tools/convert_session.py recordings/session_001/ --fps 10
  ```

- `--no-video`: Skip frame extraction (input-only mode, no frame files)
  ```bash
  python tools/convert_session.py recordings/session_001/ --no-video
  ```

- `--output PATH`: Custom output pairs file
  ```bash
  python tools/convert_session.py recordings/session_001/ --output my_pairs.jsonl
  ```

#### Output Files

After conversion, the session folder contains:

| File | Description |
|------|-------------|
| `session_NNN_pairs.jsonl` | Training pairs (one per line, JSON) |
| `session_NNN_convert_log.txt` | Conversion run log (timestamp, stats, warnings) |
| `frames/` | Extracted JPEG frames (numbered 000001.jpg, etc.) |

### Replaying Session Data

Visualize or inspect recorded session pairs:

```bash
python tools/playback_pairs.py recordings/session_001/session_001_pairs.jsonl
```

#### Playback Options

- `--start-frame N`: Start playback from a specific frame index
- `--end-frame N`: Stop playback at a specific frame index (inclusive)
- `--speed X`: Playback speed multiplier (`0.5` = half speed, `2.0` = double speed)
- `--skip-idle`: Skip frames where `is_idle=true`
- `--countdown N`: Seconds to wait before playback starts (time to alt-tab into game)
- `--keyboard-mode MODE`: Keyboard injection mode (`scancode`, `vk`, `hybrid`)

Keyboard mode behavior:
- `scancode`: Sends hardware-like scan-code events (recommended for most games)
- `vk`: Sends virtual-key events (better for normal desktop apps and some menus)
- `hybrid`: Sends both scan-code and virtual-key events

Game-focused examples:

```bash
# Recommended first try for games (DirectInput/Raw Input friendly)
python tools/playback_pairs.py recordings/session_001/session_001_pairs.jsonl --keyboard-mode scancode --countdown 5

# Fallback when key input is inconsistent
python tools/playback_pairs.py recordings/session_001/session_001_pairs.jsonl --keyboard-mode hybrid --countdown 5
```

## Data Formats

### Input Events Log (`.jsonl`)

One JSON object per line, recording input events:

```json
{"event_type": "keyboard", "key": "w", "action": "down", "elapsed_ms": 1234.5, "held_keys": ["w"], "held_buttons": []}
{"event_type": "mouse_relative", "dx": 10, "dy": -5, "elapsed_ms": 1235.0}
{"event_type": "mouse_button", "button": "left", "action": "down", "elapsed_ms": 1235.5}
{"event_type": "session_start", "elapsed_ms": 0.0}
{"event_type": "session_end", "elapsed_ms": 20200.0}
```

### Training Pairs (`.jsonl`)

One pair per line, each with frame and action:

```json
{"frame_index": 0, "t_start_ms": 0.0, "action": "0 0 0 ; w ; w ; w ; w ; w ; w", "is_idle": false, "frame_path": "/path/to/frames/000001.jpg"}
{"frame_index": 1, "t_start_ms": 200.0, "action": "5 -3 0 ; w ; w ; w ; ; ; ", "is_idle": false, "frame_path": "/path/to/frames/000002.jpg"}
{"frame_index": 2, "t_start_ms": 400.0, "action": "0 0 0 ; ; ; ; ; ; ", "is_idle": true, "frame_path": "/path/to/frames/000003.jpg"}
```

**Action String Format**:
```
DX DY DZ ; chunk1 ; chunk2 ; chunk3 ; chunk4 ; chunk5 ; chunk6
```

- `DX, DY, DZ`: Mouse delta movement over the frame window
- 6 chunks: Each chunk represents a 33ms sub-window; contains comma-separated key/button names held during that chunk
- Empty chunk `""` means no keys/buttons held

### Session Metadata (`.json`)

```json
{
  "session_name": "session_001",
  "obs_input_offset_ms": -0.7,
  "recorded_at": "2026-04-18T12:34:56",
  "notes": "Test gameplay session"
}
```
## Common Workflows

### Record and Convert in One Session

```bash
# 1. Record new session
python tools/input_recorder_orchestrator.py

# 2. Convert the latest session (e.g., session_005)
python tools/convert_session.py recordings/session_005/

# 3. Check conversion log
type recordings\session_005\session_005_convert_log.txt  # Windows
cat recordings/session_005/session_005_convert_log.txt   # macOS/Linux
```

### Batch Convert Multiple Sessions

```bash
# Windows PowerShell
Get-ChildItem recordings -Directory | ForEach-Object {
    python tools/convert_session.py $_.FullName
}

# macOS/Linux bash
for dir in recordings/session_*; do
    python tools/convert_session.py "$dir"
done
```

### Extract Frames at Different Rates

```bash
# 10 Hz (100ms windows)
python tools/convert_session.py recordings/session_001/ --fps 10

# 2 Hz (500ms windows) for slower games
python tools/convert_session.py recordings/session_001/ --fps 2
```

### Skip Video, Use Input Only

```bash
# Useful for quick iteration without re-extracting large frame sets
python tools/convert_session.py recordings/session_001/ --no-video
```

## Troubleshooting

### ffmpeg Error: "Unable to choose an output format"
- Ensure ffmpeg is installed and on your PATH
- Verify `ffmpeg -version` runs from terminal
- Check that frames directory has write permissions

### Missing Frames in Output
- Check `session_NNN_convert_log.txt` for warnings about missing frame files
- Ensure video file (`.mkv`) is valid: `ffprobe recordings/session_NNN/session_NNN.mkv`
- Try re-running conversion

### Timestamp Misalignment (large offset_ms)
- OBS input offset is recorded in `_meta.json`
- Large offsets may indicate video/input sync issues during recording
- Check the offset in the conversion log: typically ±10ms is acceptable

### Input Events Missing
- Verify `_jsonl` file is not corrupted: `python -c "import jsonlines; print(len(list(jsonlines.open(...)))"`
- Check for null bytes or encoding issues in the log file

### Playback Mouse Works But Keyboard Does Not
- Use scan-code mode first: `python tools/playback_pairs.py recordings/session_001/session_001_pairs.jsonl --keyboard-mode scancode --countdown 5`
- If still failing, try hybrid mode: `--keyboard-mode hybrid`
- Make sure the game and playback script run at the same privilege level (both normal user or both administrator)
- Disable overlays/hotkeys that may capture keyboard input (Ubisoft, Steam, Discord, Xbox Game Bar, GPU overlays)

## Advanced

### Configuring OBS Recording

- Video codec (default: H.264)
- Bitrate (default: 10000 kbps)
- Resolution (default: 1920×1080)
- Audio settings
- Video recording 60fps
- You can change these according the quality of video/screenshot you need

### Custom Frame Extraction

For advanced frame processing, use ffmpeg directly:

```bash
ffmpeg -i session_001.mkv -vf fps=5 frames/%06d.jpg
```

## References

- **OBS Studio**: https://obsproject.com/
- **ffmpeg**: https://ffmpeg.org/
- **Matroska (MKV)**: https://www.matroska.org/




