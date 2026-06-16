"""
LeWM Cloud Server — MPC inference server (runs on Lightning AI).

Replaces the V-JEPA2 cloud_server.py. Instead of generating action tokens
autoregressively, this server does one-step Model Predictive Control:
  - Maintains a rolling embedding history per client
  - Scores all candidate actions via the world-model predictor
  - Returns the best action string to the local Windows client

Endpoints:
  GET  /health                → {"status": "ok", "device": "cuda", "candidates": N}
  POST /reset                 → clears per-client state
  POST /predict  (file=frame) → {"action": "dx dy dz ; g1 ; g2 ; g3 ; g4 ; g5 ; g6"}

Setup on Lightning AI:
  pip install fastapi uvicorn python-multipart pillow
  # le-wm and GameAgent repos must already be cloned (by setup_lightning.sh)
  python lewm_cloud_server.py \
      --checkpoint /teamspace/studios/this_studio/stable-wm/checkpoints/lewm/weights_epoch_100.pt \
      --candidates le-wm/candidates/gameagent_actions.txt \
      --lewm-dir   le-wm \
      --vocab-path GameAgent/data_processing/outputs/action_vocab.json
"""
from __future__ import annotations

import argparse
import io
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from PIL import Image

# ── parse args before anything else ─────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to weights_epoch_N.pt")
    p.add_argument("--vocab-path",
                   default="GameAgent/data_processing/outputs/action_vocab.json")
    p.add_argument("--lewm-dir", default="le-wm")
    p.add_argument("--candidates",
                   default="le-wm/candidates/gameagent_actions.txt")
    p.add_argument("--policy", choices=["novelty", "goal"], default="novelty")
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--max-action-tokens", type=int, default=128)
    p.add_argument("--mpc-batch", type=int, default=64,
                   help="Candidate batch size for MPC scoring (tune for GPU VRAM).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


ARGS = _parse()

# ── sys.path setup ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
for d in [str(PROJECT_ROOT),
          str((PROJECT_ROOT / ARGS.lewm_dir).resolve()),
          str((PROJECT_ROOT / "GameAgent").resolve())]:
    if d not in sys.path:
        sys.path.insert(0, d)

# these imports require the paths above
from data_processing.vjepa2_dataset import ActionTokenizer  # noqa: E402

# ── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[server] device={DEVICE}")


# ────────────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────────────

def _load_model() -> torch.nn.Module:
    import stable_pretraining as spt  # type: ignore
    from jepa import JEPA            # type: ignore
    from module import ARPredictor, MLP  # type: ignore
    from discrete_action_encoder import DiscreteActionEncoder  # type: ignore

    embed_dim = ARGS.embed_dim
    vocab_size = len(ActionTokenizer.load(ARGS.vocab_path).token_to_id)

    encoder = spt.backbone.utils.vit_hf(
        size="tiny", patch_size=14, image_size=224,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=ARGS.history_size,
        input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
        depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.0, emb_dropout=0.0,
    )
    action_encoder = DiscreteActionEncoder(
        vocab_size=vocab_size, embed_dim=embed_dim,
        max_seq_len=128, num_layers=2, nhead=4, mlp_dim=512,
        pad_id=0, dropout=0.0,
    )

    def _mlp():
        return MLP(input_dim=embed_dim, output_dim=embed_dim,
                   hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(encoder=encoder, predictor=predictor,
                 action_encoder=action_encoder,
                 projector=_mlp(), pred_proj=_mlp())

    print(f"[server] loading weights: {ARGS.checkpoint}")
    state = torch.load(ARGS.checkpoint, map_location="cpu", weights_only=True)
    raw = state.get("state_dict", state)
    raw = {k.removeprefix("model."): v for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing:
        print(f"[server] ⚠ missing keys ({len(missing)}): {missing[:3]}")
    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[server] ✓ model loaded ({n_params:.1f}M params)")
    return model


def _load_candidates() -> tuple[list[str], torch.Tensor]:
    """Load candidate action strings and pre-tokenize them."""
    path = Path(ARGS.candidates)
    if path.exists():
        texts = [l.strip() for l in path.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    else:
        print(f"[server] ⚠ candidates file not found: {path}, using built-in set")
        texts = [
            "0 0 0 ;  ;  ;  ;  ;  ;",
            "5 0 0 ;  ;  ;  ;  ;  ;",
            "-5 0 0 ;  ;  ;  ;  ;  ;",
            "0 5 0 ;  ;  ;  ;  ;  ;",
            "0 -5 0 ;  ;  ;  ;  ;  ;",
            "0 0 0 ; w ;  ;  ;  ;  ;",
            "0 0 0 ; s ;  ;  ;  ;  ;",
            "0 0 0 ; a ;  ;  ;  ;  ;",
            "0 0 0 ; d ;  ;  ;  ;  ;",
        ]
    tokenizer_ = ActionTokenizer.load(ARGS.vocab_path)
    ids = [tokenizer_.encode(t, max_length=ARGS.max_action_tokens,
                             pad_to_max_length=True) for t in texts]
    tokens_tensor = torch.stack(ids, dim=0).to(DEVICE)  # [N, L]
    print(f"[server] ✓ {len(texts)} candidates loaded")
    return texts, tokens_tensor


# ── globals ──────────────────────────────────────────────────────────────────
print("[server] initialising model…")
MODEL = _load_model()
TOKENIZER = ActionTokenizer.load(ARGS.vocab_path)
CANDIDATE_TEXTS, CANDIDATE_TOKENS = _load_candidates()
N_CAND = len(CANDIDATE_TEXTS)

# Per-client state: rolling embedding buffers
_emb_buffers: dict[str, deque[torch.Tensor]] = {}   # [1,D] each
_act_buffers: dict[str, deque[torch.Tensor]] = {}   # [1,D] each
_zero_emb = torch.zeros(1, ARGS.embed_dim, device=DEVICE)


# ────────────────────────────────────────────────────────────────────────────
# Image helpers
# ────────────────────────────────────────────────────────────────────────────

def _preprocess(img: Image.Image) -> torch.Tensor:
    """PIL → [1, 1, 3, H, W] float32."""
    img = img.convert("RGB").resize((ARGS.img_size, ARGS.img_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)   # [3, H, W]
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, 3, H, W]


# ────────────────────────────────────────────────────────────────────────────
# MPC scoring
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _mpc_pick(
    ctx_emb: torch.Tensor,      # [1, T, D]
    ctx_act_emb: torch.Tensor,  # [1, T, D]
) -> int:
    """Return index of best candidate under the novelty policy."""
    best_idx = 0
    best_score = float("-inf")

    anchor = ctx_emb.mean(dim=1)  # [1, D]

    for start in range(0, N_CAND, ARGS.mpc_batch):
        cand = CANDIDATE_TOKENS[start:start + ARGS.mpc_batch]  # [B, L]
        B = cand.shape[0]

        ctx_exp = ctx_emb.expand(B, -1, -1)         # [B, T, D]
        ctx_act_exp = ctx_act_emb.expand(B, -1, -1) # [B, T, D]

        cand_act_emb = MODEL.action_encoder(cand.unsqueeze(1))  # [B, 1, D]
        # Replace last action slot with candidate
        all_act = torch.cat([ctx_act_exp[:, :-1, :], cand_act_emb], dim=1)  # [B, T, D]

        pred = MODEL.predict(ctx_exp, all_act)  # [B, T, D]
        pred_last = pred[:, -1, :]              # [B, D]

        # Novelty: maximise change from anchor
        anchor_exp = anchor.expand(B, -1)
        scores = (pred_last - anchor_exp).pow(2).sum(dim=-1)  # [B]

        batch_max_score, batch_max_rel = scores.max(dim=0)
        if batch_max_score.item() > best_score:
            best_score = batch_max_score.item()
            best_idx = start + int(batch_max_rel.item())

    return best_idx


# ────────────────────────────────────────────────────────────────────────────
# Action string decoder
# ────────────────────────────────────────────────────────────────────────────

def _action_str(cand_idx: int) -> str:
    """Decode candidate token ids → action string for replay_frame."""
    ids = CANDIDATE_TOKENS[cand_idx]
    tokens = TOKENIZER.decode(ids.tolist())
    dx = dy = dz = "0"
    groups: list[list[str]] = [[] for _ in range(6)]
    active: int | None = None

    for tok in tokens:
        if tok in ("<pad>", "<action_start>", "<action_end>"):
            continue
        if hasattr(ActionTokenizer, "motion_token_to_value"):
            val = ActionTokenizer.motion_token_to_value(tok)
        else:
            val = None
        if val is not None:
            if tok.startswith("dx_"):
                dx = str(val)
            elif tok.startswith("dy_"):
                dy = str(val)
            elif tok.startswith("dz_"):
                dz = str(val)
            continue
        if tok.startswith("<group_") and tok.endswith(">"):
            inner = tok[len("<group_"):-1]
            if inner.isdigit():
                active = int(inner) - 1
            continue
        if tok == "<empty_group>":
            continue
        if tok.startswith("key_") and active is not None:
            groups[active].append(tok.split("_", 1)[1])

    return f"{dx} {dy} {dz} ; " + " ; ".join(",".join(g) for g in groups)


# ────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="LeWM MPC Server")


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _get_buffers(cid: str) -> tuple[deque, deque]:
    if cid not in _emb_buffers:
        _emb_buffers[cid] = deque(
            [_zero_emb.clone() for _ in range(ARGS.history_size)],
            maxlen=ARGS.history_size,
        )
        _act_buffers[cid] = deque(
            [_zero_emb.clone() for _ in range(ARGS.history_size)],
            maxlen=ARGS.history_size,
        )
    return _emb_buffers[cid], _act_buffers[cid]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "candidates": N_CAND,
        "policy": ARGS.policy,
        "history_size": ARGS.history_size,
    }


@app.post("/reset")
def reset(request: Request) -> dict:
    cid = _client_id(request)
    _emb_buffers.pop(cid, None)
    _act_buffers.pop(cid, None)
    return {"status": "reset", "client_id": cid}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict:
    cid = _client_id(request)
    emb_buf, act_buf = _get_buffers(cid)

    # 1. Decode frame
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    frame_t = _preprocess(img)  # [1, 1, 3, H, W]

    with torch.inference_mode():
        # 2. Encode frame
        out = MODEL.encode({"pixels": frame_t})
        cur_emb = out["emb"][:, 0, :]  # [1, D]
        emb_buf.append(cur_emb)

        # 3. Stack context
        ctx_emb = torch.stack(list(emb_buf), dim=1)      # [1, T, D]
        ctx_act_emb = torch.stack(list(act_buf), dim=1)  # [1, T, D]

        # 4. MPC: pick best candidate
        sel_idx = _mpc_pick(ctx_emb, ctx_act_emb)

        # 5. Store selected action embedding for next step context
        sel_tok = CANDIDATE_TOKENS[sel_idx:sel_idx + 1].unsqueeze(1)  # [1, 1, L]
        sel_act_emb = MODEL.action_encoder(sel_tok)[:, 0, :]          # [1, D]
        act_buf.append(sel_act_emb)

    action_str = _action_str(sel_idx)
    return {"action": action_str, "candidate_idx": sel_idx}


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
