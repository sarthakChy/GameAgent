#!/usr/bin/env bash
# =============================================================================
# Lightning AI — LeWM Inference Server Setup & Launch
# Run this in a FRESH Lightning AI studio terminal to start the MPC server.
#
# What it does:
#   0. Logs in to HuggingFace (needed to pull checkpoint from HF)
#   1. Clones GameAgent (LeWM branch) and le-wm (gameagent branch)
#   2. Installs inference-only pip deps (no training stack)
#   3. Downloads the trained checkpoint from HuggingFace
#   4. Starts lewm_cloud_server.py on port 8000
#
# ─── SECRETS (set in Lightning AI UI: Studio → menu → "Secrets") ─────────────
#   HF_TOKEN    → HuggingFace token (hf.co → Settings → Access Tokens)
#                 Needed if your checkpoint repo is private.
# ─────────────────────────────────────────────────────────────────────────────
#
# ─── ONE-TIME CHECKPOINT UPLOAD (do this from your local machine) ─────────────
#   pip install huggingface_hub
#   huggingface-cli upload choudharysarthak-6/gameagent-lewm \
#       /path/to/weights_epoch_100.pt weights_epoch_100.pt \
#       --repo-type model
#
#   OR upload all epochs at once:
#   for f in weights_epoch_9*.pt weights_epoch_100.pt; do
#     huggingface-cli upload choudharysarthak-6/gameagent-lewm "$f" "$f" --repo-type model
#   done
# ─────────────────────────────────────────────────────────────────────────────
#
# Optional overrides (set before running):
#   export HF_CHECKPOINT_REPO=choudharysarthak-6/gameagent-lewm
#   export CHECKPOINT_FILE=weights_epoch_100.pt
#   export SERVER_PORT=8000
# =============================================================================

set -eo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
GAMEAGENT_REPO="https://github.com/sarthakChy/GameAgent.git"
GAMEAGENT_BRANCH="LeWM"
LEWM_REPO="https://github.com/sarthakChy/le-wm.git"
LEWM_BRANCH="gameagent"

HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-choudharysarthak-6/gameagent-lewm}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-weights_epoch_100.pt}"
CHECKPOINT_DIR="$HOME/checkpoints/lewm"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_FILE"

VOCAB_PATH="GameAgent/data_processing/outputs/action_vocab.json"
CANDIDATES_PATH="le-wm/candidates/gameagent_actions.txt"
LEWM_DIR="le-wm"

SERVER_PORT="${SERVER_PORT:-8000}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
die()  { echo -e "  ${RED}✗${NC} $1"; exit 1; }

# ── [0/4] Auth ──────────────────────────────────────────────────────────────
echo ""
echo "=== [0/4] Checking auth ==="

if [ -n "${HF_TOKEN:-}" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>/dev/null
    ok "HuggingFace logged in"
else
    warn "HF_TOKEN not set — checkpoint download may fail if repo is private"
    warn "Add HF_TOKEN in Lightning AI: Studio → menu → Secrets"
fi

# ── [1/4] Clone repos ───────────────────────────────────────────────────────
echo ""
echo "=== [1/4] Cloning repos ==="

if [ -d "GameAgent" ]; then
    warn "GameAgent/ already exists — pulling latest"
    git -C GameAgent pull --ff-only 2>/dev/null && ok "GameAgent updated" || warn "GameAgent pull skipped (local changes?)"
else
    git clone --branch "$GAMEAGENT_BRANCH" --depth 1 "$GAMEAGENT_REPO" GameAgent
    ok "GameAgent cloned (branch: $GAMEAGENT_BRANCH)"
fi

if [ -d "le-wm" ]; then
    warn "le-wm/ already exists — pulling latest"
    git -C le-wm pull --ff-only 2>/dev/null && ok "le-wm updated" || warn "le-wm pull skipped (local changes?)"
else
    git clone --branch "$LEWM_BRANCH" --depth 1 "$LEWM_REPO" le-wm
    ok "le-wm cloned (branch: $LEWM_BRANCH)"
fi

# ── [2/4] Install inference deps ───────────────────────────────────────────
echo ""
echo "=== [2/4] Installing deps ==="

# le-wm repo deps (stable-pretraining, stable-worldmodel)
if [ -f "le-wm/requirements.txt" ]; then
    pip install -q -r le-wm/requirements.txt
fi

# GameAgent inference deps
pip install -q \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pillow \
    numpy \
    einops \
    huggingface_hub

ok "deps installed"

# ── [3/4] Download checkpoint ───────────────────────────────────────────────
echo ""
echo "=== [3/4] Downloading checkpoint ==="

mkdir -p "$CHECKPOINT_DIR"

if [ -f "$CHECKPOINT_PATH" ]; then
    ok "Checkpoint already present: $CHECKPOINT_PATH"
else
    echo "  Downloading $CHECKPOINT_FILE from HF repo: $HF_CHECKPOINT_REPO"

    python - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil, os

local = hf_hub_download(
    repo_id="$HF_CHECKPOINT_REPO",
    filename="$CHECKPOINT_FILE",
    repo_type="model",
    local_dir="$CHECKPOINT_DIR",
)
print(f"  Downloaded to: {local}")
PYEOF

    if [ -f "$CHECKPOINT_PATH" ]; then
        ok "Checkpoint downloaded: $CHECKPOINT_PATH"
    else
        die "Checkpoint download failed. Check HF_TOKEN and repo name: $HF_CHECKPOINT_REPO"
    fi
fi

# Verify the checkpoint is readable
python -c "
import torch
ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu', weights_only=True)
keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ['<raw tensor>']
print(f'  Checkpoint keys: {keys[:5]}')
print(f'  Checkpoint size: {sum(p.numel() for p in [v for v in ckpt.values() if hasattr(v, \"numel\")] )/ 1e6:.1f}M params')
" && ok "Checkpoint verified" || warn "Could not verify checkpoint — server will try anyway"

# ── [4/4] Start server ──────────────────────────────────────────────────────
echo ""
echo "=== [4/4] Starting LeWM inference server ==="
echo ""
echo "  Checkpoint : $CHECKPOINT_PATH"
echo "  Vocab      : $VOCAB_PATH"
echo "  Candidates : $CANDIDATES_PATH"
echo "  Port       : $SERVER_PORT"
echo ""
echo "  ─── Get your public URL from Lightning AI UI ───"
echo "  Teamspace → your studio → 'Open Port' → enter $SERVER_PORT"
echo "  Copy the URL and paste it into tools/lewm_local_client.py as CLOUD_URL"
echo "  ────────────────────────────────────────────────"
echo ""

# Abort if vocab doesn't exist
[ -f "$VOCAB_PATH" ] || die "Vocab not found: $VOCAB_PATH — is GameAgent cloned correctly?"
[ -f "$CANDIDATES_PATH" ] || warn "Candidates file not found: $CANDIDATES_PATH — server will use built-in fallback"

python GameAgent/lewm_cloud_server.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --vocab-path  "$VOCAB_PATH" \
    --lewm-dir    "$LEWM_DIR" \
    --candidates  "$CANDIDATES_PATH" \
    --port        "$SERVER_PORT"
