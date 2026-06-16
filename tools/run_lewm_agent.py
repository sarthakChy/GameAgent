"""LeWM live game agent — MPC-based controller using the trained world model.

Replaces the V-JEPA2 + action-decoder pipeline with the LeWM world model.
At each tick:
  1. Capture the current screen frame.
  2. Encode it with the ViT-tiny encoder.
  3. Roll a buffer of the last `history_size` frame embeddings.
  4. For every candidate action, predict the next embedding.
  5. Choose the action via the selected policy (novelty / goal-image).
  6. Execute it via the existing playback_pairs infrastructure.

Usage
-----
python tools/run_lewm_agent.py \\
    --checkpoint weights_epoch_100.pt \\
    --vocab-path  data_processing/outputs/action_vocab.json \\
    --lewm-dir    ../le-wm \\
    --candidates  ../le-wm/candidates/gameagent_actions.txt \\
    --policy      novelty \\
    --hz          5

Screen capture backends (in priority order):
  Windows: dxcam  (fastest, DirectX-level capture)
  Linux  : mss    (pip install mss)
  Fallback: PIL.ImageGrab (slow but universal)
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.vjepa2_dataset import ActionTokenizer

try:
    from tools.playback_pairs import parse_action, release_all, replay_frame
except ImportError:
    from playback_pairs import parse_action, release_all, replay_frame


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LeWM world-model live agent (MPC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to weights_epoch_N.pt saved by training.")
    p.add_argument("--vocab-path",
                   default="data_processing/outputs/action_vocab.json",
                   help="Path to action_vocab.json.")
    p.add_argument("--lewm-dir",
                   default="../le-wm",
                   help="Path to the le-wm repo (contains jepa.py, module.py, discrete_action_encoder.py).")
    p.add_argument("--candidates",
                   default="../le-wm/candidates/gameagent_actions.txt",
                   help="File with one candidate action string per line.")
    p.add_argument("--policy",
                   choices=["novelty", "goal"],
                   default="novelty",
                   help="novelty: pick action that changes world state most. "
                        "goal: pick action closest to --goal-image embedding.")
    p.add_argument("--goal-image", default=None,
                   help="(goal policy only) Path to a goal screenshot PNG/JPG.")
    p.add_argument("--history-size", type=int, default=3,
                   help="Context frames fed to the predictor (must match training).")
    p.add_argument("--max-action-tokens", type=int, default=128,
                   help="Must match --max-action-tokens used during HDF5 conversion.")
    p.add_argument("--hz", type=float, default=5.0,
                   help="Agent control frequency in Hz.")
    p.add_argument("--keyboard-mode",
                   choices=["scancode", "virtual_key"], default="scancode",
                   help="Keyboard injection mode.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embed-dim", type=int, default=192,
                   help="Must match training embed_dim.")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Model loading — reconstruct JEPA from hardcoded architecture + .pt weights
# ────────────────────────────────────────────────────────────────────────────

def load_lewm(checkpoint_path: str, lewm_dir: str, vocab_size: int,
              embed_dim: int, history_size: int,
              device: torch.device) -> torch.nn.Module:
    """Reconstruct JEPA from architecture config and load state dict from .pt."""
    lewm_path = Path(lewm_dir).resolve()
    if str(lewm_path) not in sys.path:
        sys.path.insert(0, str(lewm_path))

    # These imports require le-wm to be on sys.path
    import stable_pretraining as spt  # type: ignore
    from jepa import JEPA  # type: ignore
    from module import ARPredictor, MLP  # type: ignore
    from discrete_action_encoder import DiscreteActionEncoder  # type: ignore

    print(f"[model] building ViT-tiny (embed_dim={embed_dim}, img_size=224, patch=14)")
    encoder = spt.backbone.utils.vit_hf(
        size="tiny",
        patch_size=14,
        image_size=224,
        pretrained=False,
        use_mask_token=False,
    )

    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.0,      # no dropout at inference
        emb_dropout=0.0,
    )

    action_encoder = DiscreteActionEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=128,
        num_layers=2,
        nhead=4,
        mlp_dim=512,
        pad_id=0,
        dropout=0.0,      # no dropout at inference
    )

    def _mlp():
        return MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=_mlp(),
        pred_proj=_mlp(),
    )

    print(f"[model] loading weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # The SaveCkptCallback saves the bare JEPA state_dict in the .pt file
    # (not wrapped in a lightning checkpoint dict).
    if "state_dict" in state:
        # Unwrap lightning checkpoint key prefix 'model.'
        raw = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
    else:
        raw = state
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing:
        print(f"[model] ⚠ missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[model] ⚠ unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device).eval()
    print(f"[model] ✓ loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model


# ────────────────────────────────────────────────────────────────────────────
# Image preprocessing — matches training pipeline
# ────────────────────────────────────────────────────────────────────────────

def preprocess_frame(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Convert PIL frame → [1, 1, 3, H, W] float32 in [0,1]."""
    img = img.convert("RGB").resize((img_size, img_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return t.unsqueeze(0).unsqueeze(0)           # [1, 1, 3, H, W]


# ────────────────────────────────────────────────────────────────────────────
# Screen capture — platform-aware
# ────────────────────────────────────────────────────────────────────────────

def make_capturer():
    """Return a callable() → PIL.Image that captures the screen."""
    # Try dxcam (Windows, fastest)
    try:
        import dxcam  # type: ignore
        cam = dxcam.create()
        cam.start()

        def _capture_dxcam() -> Image.Image | None:
            frame = cam.get_latest_frame()
            if frame is None:
                return None
            return Image.fromarray(frame).convert("RGB")

        print("[capture] using dxcam (DirectX)")
        return _capture_dxcam, lambda: cam.stop()

    except (ImportError, Exception):
        pass

    # Try mss (Linux / Windows fallback)
    try:
        import mss  # type: ignore
        sct = mss.mss()
        monitor = sct.monitors[1]  # primary monitor

        def _capture_mss() -> Image.Image:
            raw = sct.grab(monitor)
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        print("[capture] using mss (cross-platform)")
        return _capture_mss, lambda: None

    except ImportError:
        pass

    # PIL fallback
    print("[capture] using PIL.ImageGrab (slow fallback)")
    from PIL import ImageGrab  # type: ignore

    def _capture_pil() -> Image.Image:
        return ImageGrab.grab().convert("RGB")

    return _capture_pil, lambda: None


# ────────────────────────────────────────────────────────────────────────────
# Action encoding
# ────────────────────────────────────────────────────────────────────────────

def encode_candidates(tokenizer: ActionTokenizer,
                      candidate_texts: list[str],
                      max_tokens: int,
                      device: torch.device) -> torch.Tensor:
    """Encode candidate action strings → [N, L] int64 tensor."""
    ids = [tokenizer.encode(txt, max_length=max_tokens, pad_to_max_length=True)
           for txt in candidate_texts]
    return torch.stack(ids, dim=0).to(device)  # [N, L]


def action_string_from_token_ids(tokenizer: ActionTokenizer,
                                  ids: torch.Tensor) -> str:
    """Decode [L] token IDs → action string compatible with replay_frame."""
    tokens = tokenizer.decode(ids.tolist())
    dx = dy = dz = "0"
    groups: list[list[str]] = [[] for _ in range(6)]
    active_group: int | None = None

    for tok in tokens:
        if tok in ("<pad>", "<action_start>", "<action_end>"):
            continue
        val = ActionTokenizer.motion_token_to_value(tok) if hasattr(ActionTokenizer, "motion_token_to_value") else None
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
                active_group = int(inner) - 1
            continue
        if tok == "<empty_group>":
            continue
        if tok.startswith("key_") and active_group is not None:
            groups[active_group].append(tok.split("_", 1)[1])

    group_strs = [",".join(g) for g in groups]
    return f"{dx} {dy} {dz} ; " + " ; ".join(group_strs)


# ────────────────────────────────────────────────────────────────────────────
# Action executor thread
# ────────────────────────────────────────────────────────────────────────────

def action_worker(action_queue: queue.Queue[str], keyboard_mode: str) -> None:
    held_keys: set[str] = set()
    warned_keys: set[str] = set()
    print("[executor] online, waiting for commands...")
    try:
        while True:
            action_str = action_queue.get()
            if action_str == "STOP":
                break
            dx, dy, dz, chunks = parse_action(action_str)
            held_keys = replay_frame(dx, dy, dz, 0, chunks, held_keys, warned_keys, keyboard_mode)
    finally:
        release_all(held_keys, keyboard_mode)
        print("[executor] shut down, all keys released.")


# ────────────────────────────────────────────────────────────────────────────
# Encode a single frame through the ViT encoder
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frame(model: torch.nn.Module,
                 frame: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
    """
    frame: [1, 1, 3, H, W] float32
    Returns: [1, D] embedding
    """
    frame = frame.to(device)
    batch = {"pixels": frame}
    out = model.encode(batch)
    emb = out["emb"]          # [1, 1, D]
    return emb[:, 0, :]       # [1, D]


# ────────────────────────────────────────────────────────────────────────────
# MPC: score candidates and pick best
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def pick_action_mpc(
    model: torch.nn.Module,
    ctx_emb: torch.Tensor,       # [1, T, D]  context embeddings
    ctx_act_emb: torch.Tensor,   # [1, T, D]  context action embeddings
    candidate_tokens: torch.Tensor,  # [N, L]
    policy: str,
    goal_emb: torch.Tensor | None,   # [1, D] or None
    history_size: int,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[int, torch.Tensor]:
    """
    Returns (selected_idx, all_scores).
    novelty:  score = ||pred - mean(ctx_emb)|| (larger = more novel)
    goal:     score = -||pred - goal_emb||     (larger = closer to goal)
    """
    n_cand = candidate_tokens.shape[0]
    T = ctx_emb.shape[1]
    all_scores = []

    if policy == "novelty":
        anchor = ctx_emb.mean(dim=1)  # [1, D]

    for start in range(0, n_cand, batch_size):
        cand = candidate_tokens[start:start + batch_size]  # [B, L]
        B = cand.shape[0]

        ctx_exp = ctx_emb.expand(B, -1, -1)                # [B, T, D]
        ctx_act_exp = ctx_act_emb.expand(B, -1, -1)        # [B, T, D]

        # Encode each candidate action → [B, 1, D]
        cand_act_emb = model.action_encoder(cand.unsqueeze(1))   # [B, 1, D]

        # Replace the LAST action slot with the candidate so the predictor
        # sees: [past_T-1_actions, candidate] → predicts next state
        all_act = torch.cat([ctx_act_exp[:, :-1, :], cand_act_emb], dim=1)  # [B, T, D]

        pred = model.predict(ctx_exp, all_act)  # [B, T, D]
        pred_last = pred[:, -1, :]              # [B, D]  — predicted next embedding

        if policy == "novelty":
            anchor_exp = anchor.expand(B, -1)
            scores = (pred_last - anchor_exp).pow(2).sum(dim=-1)  # larger = more change
        else:  # goal
            goal_exp = goal_emb.expand(B, -1)
            scores = -(pred_last - goal_exp).pow(2).sum(dim=-1)   # larger = closer

        all_scores.append(scores.cpu())

    all_scores_t = torch.cat(all_scores, dim=0)  # [N]
    selected = int(all_scores_t.argmax().item())
    return selected, all_scores_t


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tick_rate = 1.0 / args.hz

    # --- Load tokenizer ---
    tokenizer = ActionTokenizer.load(args.vocab_path)
    vocab_size = len(tokenizer.token_to_id)
    print(f"[tokenizer] vocab_size={vocab_size}")

    # --- Load candidates ---
    cand_path = Path(args.candidates)
    if cand_path.exists():
        cand_texts = [l.strip() for l in cand_path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    else:
        print(f"[candidates] {cand_path} not found, using built-in set")
        cand_texts = [
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
    print(f"[candidates] {len(cand_texts)} actions loaded")

    candidate_tokens = encode_candidates(
        tokenizer, cand_texts, args.max_action_tokens, device
    )

    # --- Load world model ---
    model = load_lewm(
        checkpoint_path=args.checkpoint,
        lewm_dir=args.lewm_dir,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        device=device,
    )

    # --- Load goal image if needed ---
    goal_emb: torch.Tensor | None = None
    if args.policy == "goal":
        if args.goal_image is None:
            print("[goal] --policy goal requires --goal-image. Falling back to novelty.")
            args.policy = "novelty"
        else:
            goal_img = Image.open(args.goal_image).convert("RGB")
            goal_t = preprocess_frame(goal_img, args.img_size).to(device)
            goal_emb = encode_frame(model, goal_t, device)  # [1, D]
            print(f"[goal] goal embedding encoded from {args.goal_image}")

    # --- Screen capturer ---
    capture, capturer_stop = make_capturer()

    # --- Action executor thread ---
    action_queue: queue.Queue[str] = queue.Queue(maxsize=2)
    executor = threading.Thread(
        target=action_worker,
        kwargs={"action_queue": action_queue, "keyboard_mode": args.keyboard_mode},
        daemon=True,
    )
    executor.start()

    # --- Rolling buffers ---
    emb_buffer: deque[torch.Tensor] = deque(maxlen=args.history_size)   # [1, D] each
    act_emb_buffer: deque[torch.Tensor] = deque(maxlen=args.history_size)  # [1, D] each

    # Pre-fill buffers with zero embeddings so we can start immediately
    zero_emb = torch.zeros(1, args.embed_dim, device=device)
    for _ in range(args.history_size):
        emb_buffer.append(zero_emb)
        act_emb_buffer.append(zero_emb)

    print(f"\n{'='*55}")
    print(f"  LeWM Agent LIVE — policy={args.policy}  hz={args.hz}")
    print(f"  {len(cand_texts)} candidates  |  history={args.history_size}")
    print(f"  Click into your game window to start.")
    print(f"  Press Ctrl+C in this terminal to stop.")
    print(f"{'='*55}\n")

    step = 0
    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Capture frame
            frame_img = capture()
            if frame_img is None:
                time.sleep(0.001)
                continue

            with torch.inference_mode():
                # 2. Encode current frame
                frame_t = preprocess_frame(frame_img, args.img_size).to(device)
                batch = {"pixels": frame_t}
                out = model.encode(batch)
                cur_emb = out["emb"][:, 0, :]       # [1, D]
                emb_buffer.append(cur_emb)

                # 3. Build context tensors from buffers
                ctx_emb = torch.stack(list(emb_buffer), dim=1)        # [1, T, D]
                ctx_act_emb = torch.stack(list(act_emb_buffer), dim=1) # [1, T, D]

                # 4. Score candidates and pick best action
                sel_idx, scores = pick_action_mpc(
                    model=model,
                    ctx_emb=ctx_emb,
                    ctx_act_emb=ctx_act_emb,
                    candidate_tokens=candidate_tokens,
                    policy=args.policy,
                    goal_emb=goal_emb,
                    history_size=args.history_size,
                    device=device,
                )

                # 5. Encode the selected action for the next step's context
                sel_tok = candidate_tokens[sel_idx:sel_idx+1].unsqueeze(1)  # [1, 1, L]
                sel_act_emb = model.action_encoder(sel_tok)[:, 0, :]        # [1, D]
                act_emb_buffer.append(sel_act_emb)

            # 6. Decode → action string
            action_str = action_string_from_token_ids(
                tokenizer, candidate_tokens[sel_idx]
            )

            # 7. Send to executor
            if not action_queue.full():
                action_queue.put(action_str)

            step += 1
            if step % 20 == 0:
                score_val = float(scores[sel_idx].item())
                print(f"  step={step:5d}  action=[{sel_idx:2d}] {cand_texts[sel_idx][:30]:<30s}"
                      f"  score={score_val:.4f}")

            # Rate-limit to target Hz
            elapsed = time.perf_counter() - loop_start
            remaining = tick_rate - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[agent] stopping...")
    finally:
        capturer_stop()
        action_queue.put("STOP")
        executor.join(timeout=3.0)
        print("[agent] shut down cleanly.")


if __name__ == "__main__":
    main()
