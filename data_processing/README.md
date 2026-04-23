# Canonical HF Upload

This folder contains the canonical dataset uploader:
- hf_converter.py
- vjepa2_extractor.py
- vjepa2_dataset.py
- action_model.py
- train_action_decoder.py

It builds a canonical Hugging Face DatasetDict from converted GAMEAGENT recordings and pushes it to a dataset repo.

## What it uploads

Each row is one frame-action pair with these columns:
- image (HF Image feature)
- action_text
- game
- episode_id
- frame_index
- t_start_ms
- is_idle
- horizontal_scroll_steps
- source_session_relpath

The `game` value is read from each session metadata file:
- `recordings/session_NNN/session_NNN_meta.json`
- key: `game_name` (fallback key: `game`)

If a session meta file is missing or malformed, the uploader uses `unknown` for that session.

Default split strategy is temporal per session:
- Every session contributes to train.
- Earlier frames go to train, later frames go to validation/test.

You can also use full session-level splitting via `--split-strategy session-level`.
For pure pretraining, use `--split-strategy all-train`.

## Requirements

From repo root:

```powershell
pip install -r .\requirements.txt
```

Set token in .env at repo root (optional if already logged in):

```env
HF_TOKEN=your_hf_token
```

## Usage

Run from repo root.

### 1) Dry run (no upload)

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --dry-run
```

### 2) Push canonical dataset (private)

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical
```

### 3) Push as public dataset

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --public
```

### 4) Use specific sessions only

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --sessions session_001 session_004
```

### 5) Custom split ratios and seed

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --split-ratios 0.8 0.1 0.1 --seed 42
```

### 6) Force session-level split strategy

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --split-strategy session-level
```

### 7) Pretraining mode (all rows in train)

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --split-strategy all-train
```

Dry-run for pretraining split:

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --split-strategy all-train --dry-run
```

### 8) Incremental upload (add new sessions like session_007)

Use this when you already uploaded earlier sessions and only want to add new ones.

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --incremental
```

If you want to upload only one new session explicitly:

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo yourname/gameagent-canonical --incremental --sessions session_007
```

```powershell
python .\data_processing\hf_converter.py --recordings-dir .\recordings --repo sarthak2314/gameagent-canonical --split-strategy all-train --incremental
```


Notes for incremental mode:
- It checks existing `episode_id` values on the Hub and skips sessions already present.
- It merges existing Hub data with new rows before pushing, so old data is preserved.
- On the Hub this still creates a new dataset revision (normal behavior), but you do not need to reprocess old sessions locally.

## V-JEPA2 Frozen Embedding Extraction

Use `vjepa2_extractor.py` to generate phase-1 training data:
- input: canonical HF dataset rows
- model: V-JEPA2 from Hugging Face
- output: `.pt` file with pooled embeddings + action labels

The extractor applies:
- 4-frame sliding window by default (`[t-3, t-2, t-1, t]`)
- optional cutscene filter via idle streak dropping (default: drop `is_idle=True` only when streak > 25)
- frame-to-action mapping on the final frame `t`

### Quick start

From repo root:

```powershell
python .\data_processing\vjepa2_extractor.py --dataset-repo sarthak2314/gameagent-canonical --split train --output .\data_processing\outputs\vjepa2_embeddings.pt
```

This fetches both dataset and model from Hugging Face by default.

### Useful options

```powershell
# Limit to selected sessions
python .\data_processing\vjepa2_extractor.py --sessions session_009 session_010

# Quick smoke test
python .\data_processing\vjepa2_extractor.py --max-samples 200

# Change model or device
python .\data_processing\vjepa2_extractor.py --model-id facebook/vjepa2-vitg-fpc64-384 --device cuda

# Large runs: write shards to disk every 50k samples (prevents RAM OOM)
python .\data_processing\vjepa2_extractor.py --chunk-size 50000 --output .\data_processing\outputs\vjepa2_embeddings.pt
```

### Output modes

- Single-file mode (default): writes one `.pt` file to `--output`.
- Chunked mode (`--chunk-size > 0`): writes shard files plus an index JSON.

Example chunked outputs for `--output data_processing/outputs/vjepa2_embeddings.pt`:
- `data_processing/outputs/vjepa2_embeddings.part-00000.pt`
- `data_processing/outputs/vjepa2_embeddings.part-00001.pt`
- `data_processing/outputs/vjepa2_embeddings.index.json`

Use chunked mode when scaling to many hours of data.

### Run on cloud notebooks (Google Colab / Google Cloud / Kaggle)

The extractor can run fully from cloud notebooks and will fetch dataset + model from Hugging Face.

#### Option A: Google Colab (recommended)

1. Create a new notebook and enable GPU:
	 - Runtime -> Change runtime type -> GPU
2. Clone this repo and install dependencies:

```bash
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -r requirements.txt
```

3. Set your HF token as an environment variable:

```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

4. Run extraction:

```bash
!python data_processing/vjepa2_extractor.py \
	--dataset-repo sarthak2314/gameagent-canonical \
	--split train \
	--chunk-size 50000 \
	--output /content/vjepa2_embeddings.pt
```

5. Download outputs:

```python
from google.colab import files
files.download("/content/vjepa2_embeddings.index.json")
```

If you ran single-file mode (no `--chunk-size`), download the `.pt` file instead.

#### Option B: Google Cloud notebook VM (Vertex AI Workbench or Compute Engine Jupyter)

Use the same Linux commands as Colab, but write output to your persistent disk path, for example:

```bash
python data_processing/vjepa2_extractor.py \
	--dataset-repo sarthak2314/gameagent-canonical \
	--split train \
	--chunk-size 50000 \
	--output /home/jupyter/vjepa2_embeddings.pt
```

#### Option C: Kaggle notebook

1. Enable Internet and GPU in notebook settings.
2. Install dependencies and set token:

```python
!pip install -r /kaggle/working/GAMEAGENT/requirements.txt
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

3. Run extraction:

```bash
!python /kaggle/working/GAMEAGENT/data_processing/vjepa2_extractor.py \
	--dataset-repo sarthak2314/gameagent-canonical \
	--split train \
	--chunk-size 50000 \
	--output /kaggle/working/vjepa2_embeddings.pt
```

4. Save output as a Kaggle Dataset or download it from notebook output files.

#### Cloud tips

- Start with a smoke test first:

```bash
python data_processing/vjepa2_extractor.py --max-samples 200
```

- If CUDA memory is tight, keep the same script and reduce `--max-samples` for debugging before full runs.
- If host RAM is limited, always enable `--chunk-size` so samples are flushed to disk continuously.
- Keep `HF_TOKEN` in notebook secrets where possible instead of hardcoding it in cells.

## Phase 2: PyTorch Dataset + Tokenizer Bridge

Use `vjepa2_dataset.py` to bridge embedding shards and model training.

It provides:
- `ShardedEmbeddingActionDataset`: lazy row access from shard index JSON
- `ActionTokenizer`: action string -> integer ids
- `make_collate_fn`: dynamic batch padding for variable-length action sequences

### Recommended workflow

1. Build/load tokenizer vocabulary from your shard index.
2. Build dataset from `*.index.json`.
3. Use DataLoader with `make_collate_fn(tokenizer.pad_id)`.

### Minimal training-side example

```python
from torch.utils.data import DataLoader
from data_processing.vjepa2_dataset import (
	ActionTokenizer,
	ShardedEmbeddingActionDataset,
	make_collate_fn,
)

index_path = "data_processing/outputs/vjepa2_embeddings.index.json"
vocab_path = "data_processing/outputs/action_vocab.json"

# Build once, then save.
tokenizer = ActionTokenizer.build_from_index(index_path, min_freq=1)
tokenizer.save(vocab_path)

# Later you can reload instead:
# tokenizer = ActionTokenizer.load(vocab_path)

dataset = ShardedEmbeddingActionDataset(
	index_path=index_path,
	tokenizer=tokenizer,
	max_action_tokens=None,
	pad_to_max_action_tokens=False,
	shard_cache_size=2,
)

loader = DataLoader(
	dataset,
	batch_size=64,
	shuffle=True,
	num_workers=0,
	collate_fn=make_collate_fn(tokenizer.pad_id),
)

batch = next(iter(loader))
print(batch["embedding"].shape)   # [B, D]
print(batch["action_ids"].shape)  # [B, T_max]
print(batch["action_length"].shape)
```

### Tokenization scheme used

Action strings are canonicalized to tokens like:
- `<action_start>`
- `dx_<n>`, `dy_<n>`, `dz_<n>`
- `<group_1>` ... `<group_6>`
- `key_w`, `key_space`, ...
- `<empty_group>` (for empty key groups)
- `<action_end>`

This keeps sequence structure explicit and easy to debug.

## Phase 3: Action Decoder (The Brain)

Use `action_model.py` to map V-JEPA embeddings to action token sequences.

Implemented model:
- `MiniTransformerActionDecoder`: compact autoregressive decoder conditioned on one visual context token.

Why this default:
- Works naturally with the tokenizer/vocabulary you already built.
- Handles variable-length action sequences.
- Preserves ordering across action groups and special tokens.

### Minimal training-side example

```python
import torch
from data_processing.action_model import MiniTransformerActionDecoder, make_teacher_forcing_batch

# From tokenizer/dataset setup
vocab_size = len(tokenizer.token_to_id)
pad_id = tokenizer.pad_id
start_id = tokenizer.token_to_id["<action_start>"]

model = MiniTransformerActionDecoder(
	vocab_size=vocab_size,
	vision_dim=batch["embedding"].shape[-1],
	d_model=256,
	nhead=8,
	num_layers=4,
	max_seq_len=128,
	pad_id=pad_id,
).to("cuda" if torch.cuda.is_available() else "cpu")

vision = batch["embedding"].to(model.lm_head.weight.device)          # [B, D]
target_ids = batch["action_ids"].to(model.lm_head.weight.device)     # [B, T]

input_ids, labels = make_teacher_forcing_batch(
	target_ids,
	start_id=start_id,
	pad_id=pad_id,
)

loss = model.compute_loss(
	vision_embedding=vision,
	input_ids=input_ids,
	target_ids=labels,
	ignore_index=pad_id,
)
loss.backward()
```

### Inference example

```python
generated = model.generate(
	vision_embedding=vision[:1],
	start_id=start_id,
	end_id=tokenizer.token_to_id["<action_end>"],
	max_new_tokens=64,
)

tokens = tokenizer.decode(generated[0])
print(tokens)
```

This module is intentionally small so it can train on solo-developer hardware and cloud notebooks.

## End-to-End Training Script

Use `train_action_decoder.py` to run full Phase 3 training:
- loads/builds vocabulary
- loads sharded dataset lazily
- trains MiniTransformerActionDecoder with teacher forcing
- runs validation each epoch
- saves checkpoints + JSONL metrics

### Local run

```powershell
python .\data_processing\train_action_decoder.py `
	--index-path .\data_processing\outputs\vjepa2_embeddings.index.json `
	--vocab-path .\data_processing\outputs\action_vocab.json `
	--epochs 10 `
	--batch-size 64 `
	--learning-rate 3e-4 `
	--device cuda `
	--use-amp
```

### With explicit validation index

```powershell
python .\data_processing\train_action_decoder.py `
	--index-path .\data_processing\outputs\train.index.json `
	--val-index-path .\data_processing\outputs\val.index.json `
	--vocab-path .\data_processing\outputs\action_vocab.json `
	--device cuda `
	--use-amp
```

### Cloud notebook run (Colab/Kaggle)

```bash
python data_processing/train_action_decoder.py \
	--index-path /content/vjepa2_embeddings.index.json \
	--vocab-path /content/action_vocab.json \
	--epochs 10 \
	--batch-size 64 \
	--device cuda \
	--use-amp
```

### Outputs

Each run creates a timestamped directory under:
- `data_processing/outputs/action_decoder_runs/`

Inside each run directory:
- `best.pt` (best validation checkpoint)
- `epoch_XXX.pt` (periodic checkpoints)
- `metrics.jsonl` (epoch-wise train/val loss + learning rate)

## Dataset card behavior

When you push, the script uploads a generated dataset card to the HF dataset repo as README.md.
It includes:
- schema version
- split strategy and requested ratios
- seed
- per-session allocation summary
- feature list
- action format

## Notes

- The script expects converted session files like session_NNN_pairs.jsonl and frames/ to already exist.
- If frame_path is missing in a pair row or the frame file does not exist on disk, that row is skipped with warnings.
- Temporal strategy is usually better when each session has unique game states and you still want test data.
- In incremental mode, older hub rows without a `game` column are normalized to `unknown` before merge.
