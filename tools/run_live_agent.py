from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import dxcam
import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor


# Ensure project root imports work whether run from repo root or tools/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.action_model import MiniTransformerActionDecoder
from data_processing.vjepa2_dataset import ActionTokenizer

try:
    from tools.playback_pairs import parse_action, release_all, replay_frame
except ImportError:
    from playback_pairs import parse_action, release_all, replay_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live GAMEAGENT inference: V-JEPA2 vision encoder + action decoder + input playback.",
    )
    parser.add_argument(
        "--vocab-path",
        default="data_processing/outputs/action_vocab.json",
        help="Path to tokenizer vocabulary JSON.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="data_processing/outputs/best.pt",
        help="Path to action decoder checkpoint file.",
    )
    parser.add_argument(
        "--vision-model",
        default="facebook/vjepa2-vitg-fpc64-384",
        help="Hugging Face model id for V-JEPA2.",
    )
    parser.add_argument("--hz", type=float, default=5.0, help="Agent control frequency.")
    parser.add_argument(
        "--history-frames",
        type=int,
        default=4,
        help="Number of most recent frames in temporal context window.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated action tokens per step.",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling. 0 disables top-k.")
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="Bounded queue length between inference and action executor threads.",
    )
    parser.add_argument(
        "--keyboard-mode",
        choices=["scancode", "virtual_key"],
        default="scancode",
        help="Keyboard injection mode used by playback_pairs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Execution device for V-JEPA and decoder.",
    )
    return parser.parse_args()


def pool_embedding(tokens: torch.Tensor) -> torch.Tensor:
    # V-JEPA output rank may vary with config; average all non-batch axes.
    if tokens.ndim < 2:
        raise RuntimeError(f"Unexpected vision token shape: {tuple(tokens.shape)}")
    if tokens.ndim == 2:
        return tokens
    reduce_dims = tuple(range(1, tokens.ndim - 1))
    if not reduce_dims:
        return tokens
    return tokens.mean(dim=reduce_dims)


def get_vjepa_vision_tokens(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    pixel_values = inputs.get("pixel_values_videos")
    if pixel_values is None:
        pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        raise KeyError("V-JEPA inputs missing pixel_values_videos/pixel_values")

    if hasattr(model, "get_vision_features"):
        return model.get_vision_features(pixel_values)

    outputs = model(pixel_values=pixel_values)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise RuntimeError("V-JEPA model did not return vision features")


def decode_tokens_to_action_string(tokens: list[str]) -> str:
    dx, dy, dz = "0", "0", "0"
    groups: list[list[str]] = [[] for _ in range(6)]
    active_group_idx: int | None = None

    for token in tokens:
        if token in {"<action_start>", "<action_end>", "<pad>"}:
            continue

        motion_value = ActionTokenizer.motion_token_to_value(token)
        if motion_value is not None:
            if token.startswith("dx_"):
                dx = str(motion_value)
            elif token.startswith("dy_"):
                dy = str(motion_value)
            elif token.startswith("dz_"):
                dz = str(motion_value)
            continue

        if token.startswith("<group_") and token.endswith(">"):
            inner = token[len("<group_") : -1]
            if inner.isdigit():
                idx = int(inner) - 1
                if 0 <= idx < 6:
                    active_group_idx = idx
            continue

        if token == "<empty_group>":
            continue

        if token.startswith("key_") and active_group_idx is not None:
            key_name = token.split("_", 1)[1]
            groups[active_group_idx].append(key_name)

    group_strs = [",".join(g) for g in groups]
    return f"{dx} {dy} {dz} ; " + " ; ".join(group_strs)


def action_worker(
    action_queue: queue.Queue[str],
    keyboard_mode: str,
) -> None:
    held_keys: set[str] = set()
    warned_keys: set[str] = set()

    print("[Hands] Online and waiting for commands...")
    try:
        while True:
            action_str = action_queue.get()
            if action_str == "STOP":
                break

            dx, dy, dz, chunks = parse_action(action_str)
            held_keys = replay_frame(
                dx,
                dy,
                dz,
                0,
                chunks,
                held_keys,
                warned_keys,
                keyboard_mode,
            )
    finally:
        release_all(held_keys, keyboard_mode)
        print("[Hands] Shutting down, all keys released.")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tick_rate = 1.0 / args.hz

    print(f"Using device: {device}")
    print("Loading Visual Cortex (V-JEPA2)...")
    processor = AutoVideoProcessor.from_pretrained(args.vision_model)

    vjepa_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vjepa = AutoModel.from_pretrained(args.vision_model, torch_dtype=vjepa_dtype).eval().to(device)

    print("Loading Motor Cortex and tokenizer...")
    tokenizer = ActionTokenizer.load(args.vocab_path)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    vision_dim = int(ckpt_args.get("vision_dim", 1408))
    model = MiniTransformerActionDecoder(
        vocab_size=len(tokenizer.token_to_id),
        vision_dim=vision_dim,
        d_model=int(ckpt_args.get("d_model", 256)),
        nhead=int(ckpt_args.get("nhead", 8)),
        num_layers=int(ckpt_args.get("num_layers", 4)),
        dim_feedforward=int(ckpt_args.get("dim_feedforward", 1024)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        max_seq_len=int(ckpt_args.get("max_seq_len", 128)),
        pad_id=tokenizer.pad_id,
        temporal_hidden_dim=int(ckpt_args.get("temporal_hidden_dim", 0)),
        temporal_num_layers=int(ckpt_args.get("temporal_num_layers", 1)),
        temporal_dropout=float(ckpt_args.get("temporal_dropout", 0.0)),
        inverse_dynamics_classes=int(ckpt_args.get("inverse_dynamics_classes", 0)),
        inverse_dynamics_hidden_dim=int(ckpt_args.get("inverse_dynamics_hidden_dim", 256)),
    ).eval().to(device)
    model.load_state_dict(ckpt["model_state"])

    start_id = tokenizer.token_to_id[ActionTokenizer.ACTION_START]
    end_id = tokenizer.token_to_id[ActionTokenizer.ACTION_END]
    top_k = args.top_k if args.top_k > 0 else None
    temporal_state: torch.Tensor | None = None

    action_queue: queue.Queue[str] = queue.Queue(maxsize=max(1, args.queue_size))
    executor = threading.Thread(
        target=action_worker,
        kwargs={"action_queue": action_queue, "keyboard_mode": args.keyboard_mode},
        daemon=True,
    )
    executor.start()

    print("Initializing DXCam...")
    camera = dxcam.create()
    camera.start(target_fps=max(1, int(args.hz)))

    frame_buffer = deque(maxlen=max(1, args.history_frames))

    print("\nAGENT IS LIVE. Click into your game window.")
    print("Press Ctrl+C in this terminal to stop.\n")

    try:
        while True:
            loop_start = time.perf_counter()

            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            img = Image.fromarray(frame).convert("RGB")
            frame_buffer.append(img)
            if len(frame_buffer) < frame_buffer.maxlen:
                continue

            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

            with torch.inference_mode(), autocast_ctx:
                clip = list(frame_buffer)
                inputs = processor(videos=[clip], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                vision_tokens = get_vjepa_vision_tokens(vjepa, inputs)
                embedding = pool_embedding(vision_tokens)
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)

                out_ids, temporal_state = model.generate(
                    embedding,
                    start_id=start_id,
                    end_id=end_id,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=top_k,
                    hidden_state=temporal_state,
                    return_temporal_state=True,
                )

                if temporal_state is not None:
                    temporal_state = temporal_state.detach()

                tokens = tokenizer.decode(out_ids[0])

            action_str = decode_tokens_to_action_string(tokens)

            if not action_queue.full():
                action_queue.put(action_str)

            elapsed = time.perf_counter() - loop_start
            remaining = tick_rate - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        camera.stop()
        action_queue.put("STOP")
        executor.join()


if __name__ == "__main__":
    main()
