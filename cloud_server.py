from __future__ import annotations

import io
import sys
from collections import deque
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.action_model import MiniTransformerActionDecoder
from data_processing.vjepa2_dataset import ActionTokenizer


app = FastAPI()
SERVER_FRAME_SIZE = (1280, 720)


def _resolve_best_checkpoint(default_path: str) -> Path:
    preferred = Path(default_path)
    if preferred.exists():
        return preferred

    run_dirs = sorted(
        Path(PROJECT_ROOT, "data_processing", "outputs", "action_decoder_runs").glob("*/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0]

    raise FileNotFoundError(f"Could not find checkpoint at {default_path} or any action_decoder_runs/*/best.pt")


def pool_embedding(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 2:
        return tokens
    if tokens.ndim == 3:
        return tokens.mean(dim=1)
    if tokens.ndim == 4:
        return tokens.mean(dim=(1, 2))
    raise RuntimeError(f"Unexpected vision token shape: {tuple(tokens.shape)}")


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
            groups[active_group_idx].append(token.split("_", 1)[1])

    return f"{dx} {dy} {dz} ; " + " ; ".join(",".join(group) for group in groups)


def normalize_frame(img: Image.Image) -> Image.Image:
    # Enforce a stable shape before buffering so video batching cannot mix sizes.
    return img.convert("RGB").resize(SERVER_FRAME_SIZE, Image.Resampling.BILINEAR)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

VOCAB_PATH = PROJECT_ROOT / "data_processing" / "outputs" / "action_vocab.json"
BEST_PT_PATH = _resolve_best_checkpoint(
    str(PROJECT_ROOT / "data_processing" / "outputs" / "action_decoder_runs" / "20260423_160657" / "best.pt")
)

print("Loading Visual Cortex (V-JEPA)...")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-384")
vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384", torch_dtype=TORCH_DTYPE).eval().to(DEVICE)

print("Loading Motor Cortex...")
tokenizer = ActionTokenizer.load(VOCAB_PATH)
ckpt = torch.load(BEST_PT_PATH, map_location=DEVICE)
args = ckpt["args"]

model = MiniTransformerActionDecoder(
    vocab_size=len(tokenizer.token_to_id),
    vision_dim=int(args.get("vision_dim", 1408)),
    d_model=int(args["d_model"]),
    nhead=int(args["nhead"]),
    num_layers=int(args["num_layers"]),
    dim_feedforward=int(args["dim_feedforward"]),
    max_seq_len=int(args["max_seq_len"]),
    pad_id=tokenizer.pad_id,
    temporal_hidden_dim=int(args.get("temporal_hidden_dim", 0)),
    temporal_num_layers=int(args.get("temporal_num_layers", 1)),
    temporal_dropout=float(args.get("temporal_dropout", 0.0)),
    inverse_dynamics_classes=int(args.get("inverse_dynamics_classes", 0)),
    inverse_dynamics_hidden_dim=int(args.get("inverse_dynamics_hidden_dim", 256)),
).eval().to(DEVICE)
model.load_state_dict(ckpt["model_state"])

client_frame_buffers: dict[str, deque[Image.Image]] = {}
client_temporal_states: dict[str, torch.Tensor | None] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.post("/reset")
def reset_client_state(request: Request) -> dict[str, str]:
    client_id = request.client.host if request.client else "unknown"
    client_frame_buffers.pop(client_id, None)
    client_temporal_states.pop(client_id, None)
    return {"status": "reset", "client_id": client_id}


@app.post("/predict")
async def predict_action(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    image_bytes = await file.read()
    img = normalize_frame(Image.open(io.BytesIO(image_bytes)))

    client_id = request.client.host if request.client else "unknown"
    frame_buffer = client_frame_buffers.setdefault(client_id, deque(maxlen=4))
    frame_buffer.append(img)

    if len(frame_buffer) < frame_buffer.maxlen:
        return {"action": "0 0 0 ; ; ; ; ; ; "}

    temporal_state = client_temporal_states.get(client_id)
    if temporal_state is not None:
        temporal_state = temporal_state.to(DEVICE)

    with torch.inference_mode():
        if DEVICE == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

        with autocast_ctx:
            try:
                inputs = processor(videos=[list(frame_buffer)], return_tensors="pt")
            except ValueError as exc:
                # Self-heal if a malformed frame batch slips in during transitions.
                frame_buffer.clear()
                client_temporal_states.pop(client_id, None)
                if "same shape" in str(exc):
                    return {"action": "0 0 0 ; ; ; ; ; ; "}
                raise
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            vision_tokens = get_vjepa_vision_tokens(vjepa, inputs)
            embedding = pool_embedding(vision_tokens)
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)

            start_id = tokenizer.token_to_id[ActionTokenizer.ACTION_START]
            end_id = tokenizer.token_to_id[ActionTokenizer.ACTION_END]
            out_ids, new_temporal_state = model.generate(
                embedding,
                start_id=start_id,
                end_id=end_id,
                max_new_tokens=64,
                temperature=0.1,
                hidden_state=temporal_state,
                return_temporal_state=True,
            )
            client_temporal_states[client_id] = None if new_temporal_state is None else new_temporal_state.detach()
            tokens = tokenizer.decode(out_ids[0])

    return {"action": decode_tokens_to_action_string(tokens)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
