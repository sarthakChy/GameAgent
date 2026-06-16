#!/usr/bin/env bash
# =============================================================================
# Lightning AI — GameAgent + LeWM full setup script
# Run this once in a fresh Lightning AI studio terminal.
#
# What it does:
#   0. Logs in to HuggingFace and W&B using secrets from environment
#   1. Clones GameAgent (LeWM branch) and le-wm (gameagent branch)
#   2. Installs all pip deps
#   3. Converts HF dataset → gameagent.h5
#   4. Prints vocab size
#   5. Prints the ready-to-run training command
#
# ─── WHERE TO SET SECRETS (Lightning AI) ────────────────────────────────────
#  In the Lightning AI UI:
#    Studio → top-right menu → "Secrets" → "+ Add Secret"
#
#  Add these two secrets:
#    HF_TOKEN      → your HuggingFace token  (Settings → Access Tokens on hf.co)
#    WANDB_API_KEY → your W&B API key        (wandb.ai/authorize)
#
#  Lightning AI injects secrets as env vars into every studio terminal
#  automatically — you never paste tokens into the terminal or this script.
# ─────────────────────────────────────────────────────────────────────────────
#
# Optional overrides (set before running):
#   export STABLEWM_HOME=/teamspace/studios/this_studio/storage
# =============================================================================

set -eo pipefail   # exit on error; -u removed so optional vars don't abort

# ---------------------------------------------------------------------------
# Config — edit these if needed
# ---------------------------------------------------------------------------
STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable-wm}"
GAMEAGENT_REPO="https://github.com/sarthakChy/GameAgent.git"
GAMEAGENT_BRANCH="LeWM"    # contains converter + vjepa2_dataset.py + plan
LEWM_REPO="https://github.com/sarthakChy/le-wm.git"
LEWM_BRANCH="gameagent"    # contains DiscreteActionEncoder + gameagent configs
HF_DATASET="sarthak2314/gameagent-canonical"
HF_SPLIT="train"
IMAGE_SIZE=224
MAX_ACTION_TOKENS=128

# ---------------------------------------------------------------------------
# 0. Auth — HuggingFace + W&B
# ---------------------------------------------------------------------------
echo ""
echo "=== [0/5] Checking auth ==="

# HuggingFace
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF_TOKEN found — logging in to HuggingFace"
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
        python -c "from huggingface_hub import login; login('$HF_TOKEN')"
    echo "  ✓ HuggingFace logged in"
else
    echo "  ⚠ HF_TOKEN not set."
    echo "    If your dataset is private, add it under: Studio → Secrets → HF_TOKEN"
    echo "    Continuing (will fail at conversion step if dataset is private)."
fi

# W&B
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "  WANDB_API_KEY found — logging in to W&B"
    python -c "import wandb; wandb.login(key='$WANDB_API_KEY', relogin=True)" 2>/dev/null
    echo "  ✓ W&B logged in"
else
    echo "  ⚠ WANDB_API_KEY not set."
    echo "    Add it under: Studio → Secrets → WANDB_API_KEY"
    echo "    Training will use wandb.enabled=false until you add it."
fi

# ---------------------------------------------------------------------------
# 1. Clone repos
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/5] Cloning repos ==="

if [ ! -d "GameAgent" ]; then
    git clone -b "$GAMEAGENT_BRANCH" "$GAMEAGENT_REPO" GameAgent
else
    echo "  GameAgent/ already exists, skipping clone"
    echo "  (should be on branch: $GAMEAGENT_BRANCH)"
fi

if [ ! -d "le-wm" ]; then
    git clone -b "$LEWM_BRANCH" "$LEWM_REPO" le-wm
else
    echo "  le-wm/ already exists, skipping clone"
    echo "  (should be on branch: $LEWM_BRANCH)"
fi

# ---------------------------------------------------------------------------
# 2. Install Python deps
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/5] Installing dependencies ==="

pip install -q "stable-worldmodel[train,format]" \
               datasets \
               Pillow \
               huggingface_hub \
               python-dotenv \
               h5py \
               hdf5plugin

echo "  ✓ deps installed"

# ---------------------------------------------------------------------------
# 3. Set up STABLEWM_HOME
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/5] Setting STABLEWM_HOME ==="

export STABLEWM_HOME="$STABLEWM_HOME"
mkdir -p "$STABLEWM_HOME/datasets"
echo "  STABLEWM_HOME=$STABLEWM_HOME"

VOCAB_PATH="GameAgent/data_processing/outputs/action_vocab.json"
OUTPUT_H5="$STABLEWM_HOME/datasets/gameagent.h5"

# ---------------------------------------------------------------------------
# 4. Convert HF dataset → HDF5
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/5] Converting HF dataset to HDF5 ==="

if [ -f "$OUTPUT_H5" ]; then
    echo "  $OUTPUT_H5 already exists — skipping conversion"
    echo "  Delete it and re-run if you want to reconvert."
else
    BUILD_VOCAB_FLAG=""
    if [ ! -f "$VOCAB_PATH" ]; then
        echo "  action_vocab.json not found — will build it from dataset rows"
        BUILD_VOCAB_FLAG="--build-vocab"
    fi

    HF_TOKEN_ARG=""
    if [ -n "${HF_TOKEN:-}" ]; then
        HF_TOKEN_ARG="--hf-token $HF_TOKEN"
    fi

    python GameAgent/data_processing/convert_hf_to_lewm_h5.py \
        --repo "$HF_DATASET" \
        --split "$HF_SPLIT" \
        --vocab-path "$VOCAB_PATH" \
        --output "$OUTPUT_H5" \
        --image-size "$IMAGE_SIZE" \
        --max-action-tokens "$MAX_ACTION_TOKENS" \
        --mode overwrite \
        $BUILD_VOCAB_FLAG \
        $HF_TOKEN_ARG
fi

# ---------------------------------------------------------------------------
# 5. Print vocab size + training command
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/5] Ready to train ==="

VOCAB_SIZE=$(python -c "
import json, sys
try:
    d = json.load(open('$VOCAB_PATH'))
    print(len(d['token_to_id']))
except Exception as e:
    print('ERROR: ' + str(e), file=sys.stderr)
    sys.exit(1)
")

# Decide wandb flag
WANDB_FLAG="wandb.enabled=false"
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_FLAG="wandb.enabled=true wandb.config.entity=<your_entity> wandb.config.project=gameagent-lewm"
fi

echo ""
echo "  vocab_size    = $VOCAB_SIZE"
echo "  dataset       = $OUTPUT_H5"
echo "  STABLEWM_HOME = $STABLEWM_HOME"
echo ""
echo "=== Run this to start training: ==="
echo ""
echo "  cd le-wm && \\"
echo "  STABLEWM_HOME=$STABLEWM_HOME \\"
echo "  python train.py \\"
echo "    data=gameagent \\"
echo "    model=lewm_gameagent \\"
echo "    model.action_encoder.vocab_size=$VOCAB_SIZE \\"
echo "    $WANDB_FLAG"
echo ""
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "  (Replace <your_entity> with your W&B username or team name)"
    echo ""
fi
echo "✓ All done."
