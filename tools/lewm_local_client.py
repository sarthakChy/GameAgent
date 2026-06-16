# lewm_local_client.py  (RUN ON YOUR LOCAL WINDOWS PC)
# Captures screen with dxcam, sends frames to LeWM cloud server,
# receives action string, executes via playback_pairs.
#
# Usage:
#   1. Set CLOUD_URL below to your Lightning AI studio URL + /predict
#   2. python lewm_local_client.py
#
import io
import queue
import time
import threading

import dxcam
import requests
from PIL import Image

from playback_pairs import parse_action, replay_frame, release_all

# ── CONFIGURE THIS ──────────────────────────────────────────────────────────
# Paste your Lightning AI port-forward URL (ends with /predict)
CLOUD_URL = "https://8000-<YOUR-STUDIO-ID>.cloudspaces.litng.ai/predict"
HZ = 5                   # control frequency — match what the server expects
TICK_RATE = 1.0 / HZ
FRAME_SIZE = (640, 360)  # resize before upload (saves bandwidth; server rescales to 224)
JPEG_QUALITY = 80        # JPEG compression quality for upload
KEYBOARD_MODE = "scancode"
# ────────────────────────────────────────────────────────────────────────────

action_queue: queue.Queue[str] = queue.Queue(maxsize=2)


def action_worker() -> None:
    held_keys: set[str] = set()
    warned_keys: set[str] = set()
    print("[executor] Online. Awaiting network commands...")
    try:
        while True:
            action_str = action_queue.get()
            if action_str == "STOP":
                break
            dx, dy, dz, chunks = parse_action(action_str)
            held_keys = replay_frame(dx, dy, dz, 0, chunks,
                                     held_keys, warned_keys, KEYBOARD_MODE)
    finally:
        release_all(held_keys, KEYBOARD_MODE)
        print("[executor] Shut down, all keys released.")


def network_inference_loop() -> None:
    # Health check first
    try:
        health_url = CLOUD_URL.replace("/predict", "/health")
        r = requests.get(health_url, timeout=5)
        h = r.json()
        print(f"[health] {h}")
        print(f"[health] {h.get('candidates', '?')} candidates loaded on server")
    except Exception as e:
        print(f"[health] ⚠ could not reach server: {e}")
        print("[health]  → make sure CLOUD_URL is correct and the server is running")

    print("Initialising DXCam...")
    camera = dxcam.create()
    camera.start(target_fps=HZ)

    executor = threading.Thread(target=action_worker, daemon=True)
    executor.start()

    print(f"\n{'='*55}")
    print(f"  LeWM Local Client LIVE — {HZ}Hz → {CLOUD_URL}")
    print(f"  Click into your game window to start.")
    print(f"  Press Ctrl+C in this terminal to stop.")
    print(f"{'='*55}\n")

    step = 0
    try:
        while True:
            loop_start = time.perf_counter()

            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 1. Compress and upload frame
            img = Image.fromarray(frame).convert("RGB").resize(
                FRAME_SIZE, Image.Resampling.BILINEAR
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            image_bytes = buf.getvalue()

            # 2. Send to cloud, get action
            try:
                resp = requests.post(
                    CLOUD_URL,
                    files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    action_str = data["action"]
                    cand_idx = data.get("candidate_idx", "?")
                    step += 1
                    if step % 10 == 0:
                        print(f"[step={step:5d}] cand={cand_idx}  action={action_str[:40]}")
                    if not action_queue.full():
                        action_queue.put(action_str)
                else:
                    print(f"[predict] server error {resp.status_code}: {resp.text[:80]}")
            except requests.exceptions.RequestException as e:
                print(f"[predict] network error: {e}")

            # 3. Rate-limit
            elapsed = time.perf_counter() - loop_start
            remaining = TICK_RATE - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[client] stopping...")
    finally:
        camera.stop()
        action_queue.put("STOP")
        executor.join(timeout=3.0)


if __name__ == "__main__":
    network_inference_loop()
