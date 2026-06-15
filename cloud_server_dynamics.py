from __future__ import annotations

import io
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.dynamics_inference import LatentDynamicsRollout
from data_processing.latent_dynamics_model import LatentDynamicsModel
from data_processing.vjepa2_dataset import (
    ActionTokenizer,
    ShardedEmbeddingActionDataset,
    _load_json,
    _parse_shards,
)


app = FastAPI()
SERVER_FRAME_SIZE = (1280, 720)
IDLE_ACTION = "0 0 0 ; ; ; ; ; ;"


def _resolve_best_checkpoint(default_path: str, search_root: Path) -> Path:
    preferred = Path(default_path)
    if preferred.exists():
        return preferred

    candidates: list[Path] = []
    if search_root.exists():
        for pattern in ("*/best_model.pt", "*/best.pt"):
            candidates.extend(search_root.glob(pattern))

    if candidates:
        return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"Could not find checkpoint at {default_path} or any dynamics_runs/*/best_model.pt"
    )


def _infer_model_config_from_state_dict(model_state: dict[str, torch.Tensor], dropout: float = 0.1) -> dict:
    latent_dim = int(model_state["mu_head.weight"].shape[0])
    action_vocab_size = int(model_state["action_encoder.token_embedding.weight"].shape[0])
    action_embedding_dim = int(model_state["action_encoder.token_embedding.weight"].shape[1])
    hidden_dim = int(model_state["state_proj.weight"].shape[0])
    num_layers = sum(
        1
        for key, value in model_state.items()
        if key.startswith("mlp.") and key.endswith(".weight") and value.ndim == 2
    )

    return {
        "latent_dim": latent_dim,
        "action_vocab_size": action_vocab_size,
        "action_embedding_dim": action_embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": float(dropout),
    }


def _load_dynamics_model(checkpoint_path: Path, device: str) -> LatentDynamicsModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ckpt.get("model_config")
    if model_config is None:
        model_config = _infer_model_config_from_state_dict(
            ckpt["model_state"],
            dropout=float(ckpt.get("args", {}).get("dropout", 0.1)),
        )

    dynamics = LatentDynamicsModel(
        latent_dim=int(model_config["latent_dim"]),
        action_vocab_size=int(model_config["action_vocab_size"]),
        action_embedding_dim=int(model_config["action_embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    dynamics.load_state_dict(ckpt["model_state"])
    dynamics.eval()
    return dynamics


def _load_action_candidates_balanced(
    train_index_path: Path,
    limit_per_game: int = 100,
) -> list[str]:
    """
    Build a per-game balanced candidate list from training data,
    filtering out unsafe keys (esc, tab).
    """
    index_data = _load_json(train_index_path)
    shards = _parse_shards(index_data, train_index_path)

    game_counts = defaultdict(Counter)

    print("Scanning training embeddings for action candidates (balanced per game)...")
    for shard in shards:
        payload = torch.load(shard.path, map_location="cpu")
        actions = [str(a) for a in payload.get("action_text", [])]
        games = [str(g) for g in payload.get("game", ["unknown"] * len(actions))]

        for action, game in zip(actions, games):
            parts = action.split(";")
            key_chunks = [chunk.strip() for chunk in parts[1:]]
            all_keys = []
            for chunk in key_chunks:
                if chunk:
                    all_keys.extend(k.strip() for k in chunk.split(","))
            if any(k in {"esc", "tab"} for k in all_keys):
                continue

            game_counts[game][action] += 1

    candidates_all: list[str] = []
    for game, counts in game_counts.items():
        top_actions = [action for action, _ in counts.most_common(limit_per_game)]
        candidates_all.extend(top_actions)
        print(f"  {game}: added {len(top_actions)} candidates")

    seen = set()
    candidates: list[str] = []
    for action in candidates_all:
        if action in seen:
            continue
        seen.add(action)
        candidates.append(action)

    if not candidates:
        print("[warn] No valid candidates found after filtering - using fallback idle action.")
        return [IDLE_ACTION]

    print(f"Total unique balanced candidates: {len(candidates)}")
    return candidates


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


def _load_goal_embedding_from_session(
    train_ds: ShardedEmbeddingActionDataset,
    *,
    session_name: str = "session_007",
    offset: int = 50,
) -> tuple[int, dict]:
    print(f"Scanning for a guaranteed {session_name} goal embedding...")

    goal_sample_idx = 0
    found_session = False

    for i in range(len(train_ds)):
        episode_id = str(train_ds[i].get("episode_id", ""))
        if session_name in episode_id:
            goal_sample_idx = min(i + offset, len(train_ds) - 1)
            found_session = True
            break

    if not found_session:
        fallback_episode = str(train_ds[0].get("episode_id", ""))
        print(f"[warn] {session_name} not found in training data - falling back to first sample.")
        if session_name not in fallback_episode:
            print(f"[warn] fallback sample episode is {fallback_episode!r}.")
        goal_sample_idx = 0

    goal_sample = train_ds[goal_sample_idx]
    return goal_sample_idx, goal_sample


def normalize_frame(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize(SERVER_FRAME_SIZE, Image.Resampling.BILINEAR)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

VOCAB_PATH = PROJECT_ROOT / "data_processing" / "outputs" / "action_vocab.json"
TRAIN_INDEX_PATH = PROJECT_ROOT / "data_processing" / "outputs" / "train_embeddings.index.json"
DYNAMICS_CKPT_PATH = _resolve_best_checkpoint(
    str(PROJECT_ROOT / "data_processing" / "outputs" / "dynamics_runs" / "latest" / "best_model.pt"),
    PROJECT_ROOT / "data_processing" / "outputs" / "dynamics_runs",
)

print("Loading Visual Cortex (V-JEPA)...")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-384")
vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384", torch_dtype=TORCH_DTYPE).eval().to(DEVICE)

print("Loading Motor Cortex (latent dynamics planner)...")
tokenizer = ActionTokenizer.load(VOCAB_PATH)
dynamics = _load_dynamics_model(DYNAMICS_CKPT_PATH, DEVICE)
rollout = LatentDynamicsRollout(dynamics, tokenizer, torch.device(DEVICE))

print("Loading action candidates (balanced per game)...")
ACTION_CANDIDATES = _load_action_candidates_balanced(TRAIN_INDEX_PATH, limit_per_game=100)
print(f"Planner ready with {len(ACTION_CANDIDATES)} balanced action candidates.")

print("Loading goal embedding from training set...")
train_ds = ShardedEmbeddingActionDataset(
    index_path=TRAIN_INDEX_PATH,
    tokenizer=tokenizer,
    max_action_tokens=128,
    pad_to_max_action_tokens=False,
    shard_cache_size=2,
)
if len(train_ds) == 0:
    raise ValueError(f"Training embedding dataset is empty: {TRAIN_INDEX_PATH}")

goal_sample_idx, goal_sample = _load_goal_embedding_from_session(train_ds)
goal_embedding = goal_sample["embedding"].to(DEVICE)
print(
    "Goal locked! Using: "
    f"{goal_sample_idx} (episode={goal_sample['episode_id']}, frame={goal_sample['frame_index']})."
)

client_frame_buffers: dict[str, deque[Image.Image]] = {}


@app.get("/")
def home():
    return {"server working"}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict_action(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    image_bytes = await file.read()
    img = normalize_frame(Image.open(io.BytesIO(image_bytes)))

    client_id = request.client.host if request.client else "unknown"
    frame_buffer = client_frame_buffers.setdefault(client_id, deque(maxlen=4))
    frame_buffer.append(img)

    if len(frame_buffer) < frame_buffer.maxlen:
        return {"action": IDLE_ACTION}

    with torch.inference_mode():
        if DEVICE == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

        with autocast_ctx:
            try:
                inputs = processor(videos=[list(frame_buffer)], return_tensors="pt")
            except ValueError as exc:
                frame_buffer.clear()
                if "same shape" in str(exc):
                    return {"action": IDLE_ACTION}
                raise

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            vision_tokens = get_vjepa_vision_tokens(vjepa, inputs)
            embedding = pool_embedding(vision_tokens)
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            result = rollout.plan_trajectory(
                z_init=embedding,
                z_target=goal_embedding,
                num_actions=1,
                action_candidates=ACTION_CANDIDATES,
                num_samples=30,
                max_action_tokens=128,
            )

            best_actions = result.get("best_actions") or []
            action_str = best_actions[0] if best_actions else IDLE_ACTION

    return {"action": action_str}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)






#  Minor things to keep in mind
# Layer counting in _infer_model_config_from_state_dict
# The function counts Linear layers in the mlp by looking for ".weight" keys with ndim == 2. 
# This is correct because the MLP is built as a sequence of Linear layers. 
# However, if you ever change the model architecture (e.g., add normalization layers that also have 2‑D weights), 
# the count could be thrown off.
# For now it’s fine, and your saved checkpoints already contain model_config, so the fallback is unlikely to be used.

# Goal embedding is static
# Currently, goal_embedding is loaded once at server startup and never changes. 
# For a first test, that’s ideal. 
# Later, you can add an endpoint like /set_goal that accepts a new image or a set of goal coordinates, and updates the server’s internal goal embedding on the fly.

# Candidate‑loading efficiency
# _load_action_candidates opens the entire training index and counts all actions. 
# If your training set grows very large, you might want to cache the list of top actions alongside the checkpoint so you don’t have to re‑scan every time the server starts. 
# For now, it takes only a second or two, so it’s okay.

# Planning horizon and sample count
# num_actions=1 and num_samples=30 are sensible for real‑time use. 
# If you find the agent moves too erratically or too slowly, you can increase the number of samples (e.g., 50) or add a cost term for action smoothness in the future.