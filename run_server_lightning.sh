#!/usr/bin/env bash
# =============================================================================
# Lightning AI — LeWM Inference Server Setup & Launch (Valheim GameAgent)
# Run this in a FRESH Lightning AI studio terminal to start the MPC server.
#
# What this script does:
#   0. Logs in to HuggingFace using HF_TOKEN secret
#   1. Clones GameAgent (LeWM branch) and le-wm (gameagent branch)
#   2. Installs inference-only pip deps (no training stack)
#   3. Downloads weights + dataset from HuggingFace
#   4. Optionally extracts dataset-based candidates from the HDF5
#   5. Starts lewm_cloud_server.py on port 8000
#
# ─── ONE-TIME UPLOAD (run from your local machine) ───────────────────────────
#   pip install huggingface_hub
#
#   # Create both repos:
#   huggingface-cli repo create valheim-gameagent-lewm --type model
#   huggingface-cli repo create valheim-gameagent-data --type dataset
#
#   # Upload checkpoint to the MODEL repo:
#   huggingface-cli upload sarthak2314/valheim-gameagent-lewm \
#       /path/to/weights_epoch_100.pt weights_epoch_100.pt \
#       --repo-type model
#
#   # Upload HDF5 to the DATASET repo (8 GB — HF handles LFS automatically):
#   huggingface-cli upload sarthak2314/valheim-gameagent-data \
#       /path/to/gameagent.h5 gameagent.h5 \
#       --repo-type dataset
#
# ─── SECRETS (Lightning AI UI: Studio -> menu -> "Secrets") ──────────────────
#   HF_TOKEN  ->  your HuggingFace token  (hf.co -> Settings -> Access Tokens)
#
# ─── Optional env overrides ──────────────────────────────────────────────────
#   export HF_MODEL_REPO=sarthak2314/valheim-gameagent-lewm
#   export HF_DATA_REPO=sarthak2314/valheim-gameagent-data
#   export CHECKPOINT_FILE=weights_epoch_100.pt
#   export HDF5_FILE=gameagent.h5
#   export SERVER_PORT=8000
#   export GEN_CANDIDATES=auto   # auto | yes | no
#   bash run_server_lightning.sh
# =============================================================================

set -eo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
GAMEAGENT_REPO="https://github.com/sarthakChy/GameAgent.git"
GAMEAGENT_BRANCH="LeWM"
LEWM_REPO="https://github.com/sarthakChy/le-wm.git"
LEWM_BRANCH="gameagent"

HF_MODEL_REPO="${HF_MODEL_REPO:-sarthak2314/valheim-gameagent-lewm}"   # .pt lives here
HF_DATA_REPO="${HF_DATA_REPO:-sarthak2314/valheim-gameagent-data}"    # .h5 lives here
CHECKPOINT_FILE="${CHECKPOINT_FILE:-weights_epoch_100.pt}"
HDF5_FILE="${HDF5_FILE:-gameagent.h5}"

DOWNLOAD_DIR="/teamspace/studios/this_studio/valheim-lewm"
CHECKPOINT_PATH="$DOWNLOAD_DIR/$CHECKPOINT_FILE"
HDF5_PATH="$DOWNLOAD_DIR/$HDF5_FILE"

VOCAB_PATH="GameAgent/data_processing/outputs/action_vocab.json"
CANDIDATES_PATH="le-wm/candidates/gameagent_actions.txt"
LEWM_DIR="le-wm"
SERVER_PORT="${SERVER_PORT:-8000}"

# auto = extract from HDF5 after download; yes = force; no = use cloned 291
GEN_CANDIDATES="${GEN_CANDIDATES:-auto}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}ok${NC}  $1"; }
warn() { echo -e "  ${YELLOW}!!${NC}  $1"; }
die()  { echo -e "  ${RED}ERR${NC} $1"; exit 1; }

# ── [0/5] HuggingFace auth ───────────────────────────────────────────────────
echo ""
echo "=== [0/5] HuggingFace auth ==="

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not set.
  Add it in Lightning AI: Studio -> menu -> Secrets -> + Add Secret
  Key: HF_TOKEN   Value: your token from hf.co/settings/tokens"

python -c "
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=False)
print('  logged in')
"
ok "HuggingFace authenticated"

# ── [1/5] Clone repos ────────────────────────────────────────────────────────
echo ""
echo "=== [1/5] Cloning repos ==="

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

# ── [2/5] Install inference deps ─────────────────────────────────────────────
echo ""
echo "=== [2/5] Installing deps ==="

if [ -f "le-wm/requirements.txt" ]; then
    pip install -q -r le-wm/requirements.txt
fi

pip install -q \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    numpy \
    einops \
    huggingface_hub \
    hf_transfer         # enables fast multi-part download for large files

ok "deps installed"

# ── [3/5] Download files from HuggingFace ────────────────────────────────────
echo ""
echo "=== [3/5] Downloading from HuggingFace ==="
echo "  model repo : $HF_MODEL_REPO"
echo "  data  repo : $HF_DATA_REPO"

mkdir -p "$DOWNLOAD_DIR"

# Enable fast multi-part downloads (hf_transfer makes the 8 GB HDF5 much faster)
export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<PYEOF
import os
from huggingface_hub import hf_hub_download

dst = "$DOWNLOAD_DIR"

# ── checkpoint from MODEL repo ──
fname = "$CHECKPOINT_FILE"
target = os.path.join(dst, fname)
if os.path.exists(target):
    print(f"  already present: {fname} ({os.path.getsize(target)/1e9:.2f} GB)")
else:
    print(f"  downloading {fname} from model repo ...")
    local = hf_hub_download(
        repo_id="$HF_MODEL_REPO",
        filename=fname,
        repo_type="model",
        local_dir=dst,
        local_dir_use_symlinks=False,
    )
    print(f"  done: {local} ({os.path.getsize(local)/1e9:.2f} GB)")

# ── HDF5 from DATASET repo ──
fname = "$HDF5_FILE"
target = os.path.join(dst, fname)
if os.path.exists(target):
    print(f"  already present: {fname} ({os.path.getsize(target)/1e9:.2f} GB)")
else:
    print(f"  downloading {fname} from dataset repo (8 GB — grab a coffee) ...")
    local = hf_hub_download(
        repo_id="$HF_DATA_REPO",
        filename=fname,
        repo_type="dataset",
        local_dir=dst,
        local_dir_use_symlinks=False,
    )
    print(f"  done: {local} ({os.path.getsize(local)/1e9:.2f} GB)")
PYEOF

[ -f "$CHECKPOINT_PATH" ] || die "Checkpoint download failed: $CHECKPOINT_PATH"
ok "Checkpoint: $CHECKPOINT_PATH"

if [ -f "$HDF5_PATH" ]; then
    hdf5_size=$(du -sh "$HDF5_PATH" | cut -f1)
    ok "HDF5 dataset: $HDF5_PATH ($hdf5_size)"
else
    warn "HDF5 not found in repo — using cloned candidates instead"
    warn "Upload gameagent.h5 to $HF_REPO to enable dataset extraction"
fi

# Verify checkpoint
python -c "
import torch
ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu', weights_only=True)
n = sum(v.numel() for v in ckpt.values() if hasattr(v, 'numel'))
print(f'  params: {n/1e6:.1f}M   top keys: {list(ckpt.keys())[:3]}')
" && ok "Checkpoint verified" || warn "Could not read checkpoint — server will try anyway"

# ── [4/5] Candidates ─────────────────────────────────────────────────────────
echo ""
echo "=== [4/5] Candidates ==="

_should_gen=false
if   [ "$GEN_CANDIDATES" = "yes" ]; then
    _should_gen=true
elif [ "$GEN_CANDIDATES" = "auto" ] && [ -f "$HDF5_PATH" ]; then
    _should_gen=true
fi

if [ "$_should_gen" = true ]; then
    [ -f "$HDF5_PATH" ] || die "GEN_CANDIDATES=yes but HDF5 not found: $HDF5_PATH"
    echo "  Extracting top-300 candidates from real Valheim dataset..."
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
    [ ! -f "$HDF5_PATH" ] && warn "Upload gameagent.h5 to HF to enable dataset-extracted candidates"
fi

# ── [5/5] Start server ───────────────────────────────────────────────────────
echo ""
echo "=== [5/5] Starting Valheim LeWM inference server ==="
echo ""
printf "  Model repo : %s\n" "$HF_MODEL_REPO"
printf "  Data  repo : %s\n" "$HF_DATA_REPO"
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
[ -f "$CANDIDATES_PATH" ] || warn "Candidates file missing — server will use built-in fallback"

python GameAgent/lewm_cloud_server.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --vocab-path  "$VOCAB_PATH" \
    --lewm-dir    "$LEWM_DIR" \
    --candidates  "$CANDIDATES_PATH" \
    --port        "$SERVER_PORT"
