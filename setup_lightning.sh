#!/usr/bin/env bash
# =============================================================================
# Lightning AI — GameAgent + LeWM full setup script
# Run this once in a fresh Lightning AI studio terminal.
#
# What it does:
#   1. Clones GameAgent and le-wm (gameagent branch)
#   2. Installs all pip deps
#   3. Converts HF dataset → gameagent.h5
#   4. Prints the vocab size
#   5. Prints the training command to run
#
# Usage:
#   bash setup_lightning.sh
#
# Env vars you can override:
#   STABLEWM_HOME   storage root  (default: ~/stable-wm)
#   HF_TOKEN        HuggingFace token (needed if dataset is private)
#   WANDB_API_KEY   WandB key (needed if wandb.enabled=true)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these if needed
# ---------------------------------------------------------------------------
STABLEWM_HOME="${STABLEWM_HOME:-$HOME/stable-wm}"
GAMEAGENT_REPO="https://github.com/sarthakChy/GameAgent.git"
GAMEAGENT_BRANCH="LeWM"   # contains converter + vjepa2_dataset.py + plan
LEWM_REPO="https://github.com/sarthakChy/le-wm.git"
LEWM_BRANCH="gameagent"   # contains DiscreteActionEncoder + gameagent configs
HF_DATASET="sarthak2314/gameagent-canonical"
HF_SPLIT="train"
IMAGE_SIZE=224
MAX_ACTION_TOKENS=128

# ---------------------------------------------------------------------------
# 1. Clone repos
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/5] Cloning repos ==="

if [ ! -d "GameAgent" ]; then
    git clone -b "$GAMEAGENT_BRANCH" "$GAMEAGENT_REPO" GameAgent
else
    echo "  GameAgent/ already exists, skipping clone"
    echo "  (branch should be $GAMEAGENT_BRANCH — check with: cd GameAgent && git branch)"
fi

if [ ! -d "le-wm" ]; then
    git clone -b "$LEWM_BRANCH" "$LEWM_REPO" le-wm
else
    echo "  le-wm/ already exists, skipping clone"
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

echo "  deps installed"

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
    echo "  $OUTPUT_H5 already exists, skipping conversion"
    echo "  Delete it and re-run if you want to reconvert."
else
    BUILD_VOCAB_FLAG=""
    if [ ! -f "$VOCAB_PATH" ]; then
        echo "  action_vocab.json not found — will build it from dataset"
        BUILD_VOCAB_FLAG="--build-vocab"
    fi

    python GameAgent/data_processing/convert_hf_to_lewm_h5.py \
        --repo "$HF_DATASET" \
        --split "$HF_SPLIT" \
        --vocab-path "$VOCAB_PATH" \
        --output "$OUTPUT_H5" \
        --image-size "$IMAGE_SIZE" \
        --max-action-tokens "$MAX_ACTION_TOKENS" \
        --mode overwrite \
        $BUILD_VOCAB_FLAG
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

echo ""
echo "  vocab_size = $VOCAB_SIZE"
echo "  dataset    = $OUTPUT_H5"
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
echo "    wandb.enabled=false"
echo ""
echo "  (Set wandb.enabled=true and add wandb.config.entity/project for logging)"
echo ""
echo "Done. All set up."
