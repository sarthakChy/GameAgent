# local_client.py (RUN ON YOUR LOCAL WINDOWS PC)
import time
import io
import queue
import requests
import threading
import dxcam
from PIL import Image

# Import your Windows execution logic
from playback_pairs import parse_action, replay_frame, release_all

# PASTE YOUR LIGHTNING AI URL HERE (ensure no trailing slash)
CLOUD_URL = "https://8000-01kpxnjtgahc3s8za4dm0qfa7j.cloudspaces.litng.ai/predict" 
HZ = 5
TICK_RATE = 1.0 / HZ

action_queue = queue.Queue(maxsize=2)

def action_worker():
    held_keys = set()
    warned_keys = set()
    keyboard_mode = "scancode"
    
    print("[Hands] Online. Awaiting network commands...")
    try:
        while True:
            action_str = action_queue.get()
            if action_str == "STOP":
                break
                
            dx, dy, dz, chunks = parse_action(action_str)
            # Replay frame blocks for exactly 200ms
            held_keys = replay_frame(dx, dy, dz, 0, chunks, held_keys, warned_keys, keyboard_mode)
            
    finally:
        release_all(held_keys, keyboard_mode)
        print("[Hands] Shutting down, all keys released.")

def network_inference_loop():
    print("Initializing DXCam...")
    camera = dxcam.create()
    camera.start(target_fps=HZ)
    
    executor = threading.Thread(target=action_worker, daemon=True)
    executor.start()
    
    print("\nAGENT IS LIVE! Click into your game.")
    print("Press Ctrl+C to kill the agent.\n")
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 1. Compress frame to JPEG instantly to save upload bandwidth
            img = Image.fromarray(frame).convert("RGB")
            img = img.resize((1280, 720), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            image_bytes = buffer.getvalue()
            
            # 2. Fire over the network to the Cloud Brain
            try:
                files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
                response = requests.post(CLOUD_URL, files=files, timeout=3.0)
                
                if response.status_code == 200:
                    action_str = response.json()["action"]
                    print(f"[predict] {action_str}")
                    if not action_queue.full():
                        action_queue.put(action_str)
                else:
                    print(f"Server Error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Network Lag / Timeout: {e}")
            
            # Maintain strict 5Hz tick rate
            elapsed = time.perf_counter() - loop_start
            if elapsed < TICK_RATE:
                time.sleep(TICK_RATE - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping Agent...")
    finally:
        camera.stop()
        action_queue.put("STOP")
        executor.join()

if __name__ == "__main__":
    network_inference_loop()