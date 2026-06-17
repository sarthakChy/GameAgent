#!/usr/bin/env bash
# =============================================================================
# Lightning AI — LeWM Inference Server Setup & Launch
# Run this in a FRESH Lightning AI studio terminal to start the MPC server.
#
# Before running, manually upload these files to the studio:
#   weights_epoch_100.pt  → drag into the file browser, or use the
#                           Lightning AI "Upload" button in the file panel
#   gameagent.h5          → same (optional — only needed for dataset candidates)
#
# What this script does:
#   1. Clones GameAgent (LeWM branch) and le-wm (gameagent branch)
#   2. Installs inference-only pip deps (fast — no training stack)
#   3. Finds your checkpoint, fails clearly if it is missing
#   4. Optionally regenerates candidates from the real dataset (if .h5 uploaded)
#   5. Starts lewm_cloud_server.py on port 8000
#
# ─── Override defaults with env vars before running ──────────────────────────
#   export CHECKPOINT_PATH=/teamspace/studios/this_studio/weights_epoch_100.pt
#   export HDF5_PATH=/teamspace/studios/this_studio/gameagent.h5
#   export SERVER_PORT=8000
#   bash run_server_lightning.sh
# =============================================================================

set -eo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
GAMEAGENT_REPO="https://github.com/sarthakChy/GameAgent.git"
GAMEAGENT_BRANCH="LeWM"
LEWM_REPO="https://github.com/sarthakChy/le-wm.git"
LEWM_BRANCH="gameagent"

# Path to the .pt file you uploaded manually
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/teamspace/studios/this_studio/weights_epoch_100.pt}"

# Path to the .h5 file you uploaded manually (optional)
HDF5_PATH="${HDF5_PATH:-/teamspace/studios/this_studio/gameagent.h5}"

VOCAB_PATH="GameAgent/data_processing/outputs/action_vocab.json"
CANDIDATES_PATH="le-wm/candidates/gameagent_actions.txt"
LEWM_DIR="le-wm"
SERVER_PORT="${SERVER_PORT:-8000}"

# auto = extract from HDF5 if present, else use 291 cloned candidates
# yes  = always extract (fails if HDF5 missing)
# no   = always use cloned candidates
GEN_CANDIDATES="${GEN_CANDIDATES:-auto}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}ok${NC}  $1"; }
warn() { echo -e "  ${YELLOW}!!${NC}  $1"; }
die()  { echo -e "  ${RED}ERR${NC} $1"; exit 1; }

# ── [1/4] Clone repos ────────────────────────────────────────────────────────
echo ""
echo "=== [1/4] Cloning repos ==="

if [ -d "GameAgent" ]; then
    warn "GameAgent/ already exists — pulling latest"
    git -C GameAgent pull --ff-only 2>/dev/null && ok "GameAgent updated" || warn "pull skipped (local changes?)"
else
    git clone --branch "$GAMEAGENT_BRANCH" --depth 1 "$GAMEAGENT_REPO" GameAgent
    ok "GameAgent cloned (branch: $GAMEAGENT_BRANCH)"
fi

if [ -d "le-wm" ]; then
    warn "le-wm/ already exists — pulling latest"
    git -C le-wm pull --ff-only 2>/dev/null && ok "le-wm updated" || warn "pull skipped (local changes?)"
else
    git clone --branch "$LEWM_BRANCH" --depth 1 "$LEWM_REPO" le-wm
    ok "le-wm cloned (branch: $LEWM_BRANCH)"
fi

# ── [2/4] Install inference deps ─────────────────────────────────────────────
echo ""
echo "=== [2/4] Installing deps ==="

if [ -f "le-wm/requirements.txt" ]; then
    pip install -q -r le-wm/requirements.txt
fi

pip install -q \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    numpy \
    einops

ok "deps installed"

# ── [3/4] Locate checkpoint ──────────────────────────────────────────────────
echo ""
echo "=== [3/4] Locating checkpoint ==="

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    die "Checkpoint not found at: $CHECKPOINT_PATH
    Upload weights_epoch_100.pt to the studio, then:
      export CHECKPOINT_PATH=/path/to/weights_epoch_100.pt
      bash run_server_lightning.sh"
fi

python -c "
import torch, sys
ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu', weights_only=True)
n = sum(v.numel() for v in ckpt.values() if hasattr(v, 'numel'))
print(f'  params: {n/1e6:.1f}M   top keys: {list(ckpt.keys())[:3]}')
" && ok "Checkpoint OK: $CHECKPOINT_PATH" || warn "Could not read checkpoint — will try anyway"

# ── [3.5] Candidates ─────────────────────────────────────────────────────────
echo ""
echo "=== [3.5] Candidates ==="

_should_gen=false
if   [ "$GEN_CANDIDATES" = "yes" ]; then
    _should_gen=true
elif [ "$GEN_CANDIDATES" = "auto" ] && [ -f "$HDF5_PATH" ]; then
    _should_gen=true
fi

if [ "$_should_gen" = true ]; then
    [ -f "$HDF5_PATH" ] || die "HDF5 not found: $HDF5_PATH
    Upload gameagent.h5 or run with GEN_CANDIDATES=no to skip extraction."

    echo "  gameagent.h5 found — extracting top-300 real candidates..."
    python GameAgent/scripts/generate_candidates.py \
        --mode  dataset \
        --hdf5  "$HDF5_PATH" \
        --vocab "$VOCAB_PATH" \
        --top-n 300 \
        --out   "$CANDIDATES_PATH"
    ok "Dataset candidates written: $CANDIDATES_PATH"
else
    cnt=$(grep -c '^[^#]' "$CANDIDATES_PATH" 2>/dev/null || echo "?")
    ok "Using cloned candidates ($cnt actions)"
    if [ ! -f "$HDF5_PATH" ]; then
        warn "gameagent.h5 not found — upload it to get dataset-extracted candidates"
        warn "or set: export GEN_CANDIDATES=no  to silence this warning"
    fi
fi

# ── [4/4] Start server ───────────────────────────────────────────────────────
echo ""
echo "=== [4/4] Starting LeWM inference server ==="
echo ""
printf "  Checkpoint : %s\n" "$CHECKPOINT_PATH"
printf "  Vocab      : %s\n" "$VOCAB_PATH"
printf "  Candidates : %s\n" "$CANDIDATES_PATH"
printf "  Port       : %s\n" "$SERVER_PORT"
echo ""
echo "  ── Get your public URL ────────────────────────────────────────────"
echo "  Lightning AI UI -> your studio -> 'Open Port' -> $SERVER_PORT"
echo "  Paste that URL into tools/lewm_local_client.py  (CLOUD_URL line)"
echo "  ───────────────────────────────────────────────────────────────────"
echo ""

[ -f "$VOCAB_PATH" ]      || die "Vocab not found: $VOCAB_PATH — is GameAgent cloned correctly?"
[ -f "$CANDIDATES_PATH" ] || warn "Candidates file missing — server will use a built-in fallback"

python GameAgent/lewm_cloud_server.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --vocab-path  "$VOCAB_PATH" \
    --lewm-dir    "$LEWM_DIR" \
    --candidates  "$CANDIDATES_PATH" \
    --port        "$SERVER_PORT"
