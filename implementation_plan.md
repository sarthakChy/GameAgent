# GameAgent + LeWM Integration Implementation Plan

## Summary

This plan adapts LeWM to GameAgent's frame-action data without changing
`stable-worldmodel` internals. The integration spans three local repos:

- `GameAgent`: owns the canonical Hugging Face dataset, action tokenizer, and
  recording/action-string format.
- `le-wm`: owns the LeWM training script, Hydra configs, JEPA model wiring, and
  planning/evaluation entrypoints.
- `stable-worldmodel`: provides the dataset loaders, HDF5 format, sequence
  slicing, checkpoint utilities, and solvers used by LeWM.

The rough implementation needs four important corrections from repo inspection:

- HDF5 files loaded by `stable_worldmodel.data.HDF5Dataset` must contain
  `ep_len` and `ep_offset`; `episode_idx` and `step_idx` alone are not enough
  for training sequence slicing.
- GameAgent actions are discrete token IDs, so LeWM must skip action
  normalization.
- `le-wm/train.py` currently assigns `cfg.model.action_encoder.input_dim` from
  `dataset.get_dim("action")`; this must only happen for the existing
  continuous `module.Embedder`, not for the new discrete encoder.
- `lejepa_forward` currently applies `torch.nan_to_num` to actions as if they
  are float tensors; integer token actions must pass through unchanged.

## Current Repo Facts

`GameAgent/data_processing/vjepa2_dataset.py` already contains the canonical
`ActionTokenizer`. It tokenizes action strings into:

- special tokens: `<pad>`, `<unk>`, `<action_start>`, `<action_end>`,
  `<empty_group>`
- mouse motion bucket tokens: `dx_bin_*`, `dy_bin_*`, `dz_bin_*`
- group markers: `<group_1>` through `<group_6>`
- key tokens: `key_<name>`

`le-wm/jepa.py` calls `self.action_encoder(info["action"])` inside
`JEPA.encode()`, so the discrete encoder only needs to match the existing action
encoder interface: input `[B, T, action_payload]`, output `[B, T, embed_dim]`.

`stable-worldmodel/stable_worldmodel/data/formats/hdf5.py` loads HDF5 columns
from a flat row layout and requires:

- `ep_len`: per-episode lengths
- `ep_offset`: per-episode starting offsets
- one dataset per data column, such as `pixels`, `action`, `episode_idx`,
  `step_idx`

`stable-worldmodel` skips `frameskip` on `action` columns and then reshapes
loaded actions in `Dataset.__getitem__()` to `[num_steps, -1]`. Therefore each
per-step GameAgent action should be stored as a fixed-size token vector
`[max_action_tokens]`, so the sampled sequence becomes `[num_steps,
max_action_tokens]`.

## Implementation Changes

### 1. Convert GameAgent HF Dataset To HDF5

Add a conversion script in `GameAgent`, for example:

`data_processing/convert_hf_to_lewm_h5.py`

Responsibilities:

- Load `sarthak2314/gameagent-canonical` using `datasets.load_dataset`.
- Support `--split train`, `--split validation`, and optional output paths.
- Load or build the tokenizer from
  `data_processing/outputs/action_vocab.json`.
- Sort rows by `(episode_id, frame_index)`.
- Resize every image to `224x224`.
- Convert images to `np.uint8` in HWC layout: `[H, W, C]`.
- Encode `action_text` using `ActionTokenizer.encode(..., max_length=128,
  pad_to_max_length=True)`.
- Store token IDs as `np.int64` with shape `[N, 128]`.
- Write the HDF5 file to `$STABLEWM_HOME/datasets/gameagent.h5` by default.

Required HDF5 datasets:

```text
pixels       uint8   [N, 224, 224, 3]
action       int64   [N, 128]
episode_idx  int32   [N]
step_idx     int32   [N]
ep_len       int32   [num_episodes]
ep_offset    int64   [num_episodes]
```

Implementation details:

- `episode_idx` should be a contiguous integer ID assigned after sorting
  episodes.
- `step_idx` should start at `0` inside every episode and increment by one.
- `ep_len` should contain the number of rows in each episode.
- `ep_offset` should contain the cumulative flat-row start for each episode.
- Keep `episode_id` string metadata out of the first HDF5 version unless needed;
  the LeWM training path does not require it.

Validation command:

```bash
python data_processing/convert_hf_to_lewm_h5.py \
  --repo sarthak2314/gameagent-canonical \
  --split train \
  --vocab-path data_processing/outputs/action_vocab.json \
  --output "$STABLEWM_HOME/datasets/gameagent.h5" \
  --image-size 224 \
  --max-action-tokens 128
```

### 2. Add A Discrete Action Encoder In LeWM

Add `le-wm/discrete_action_encoder.py`.

Required interface:

```python
class DiscreteActionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        mlp_dim: int = 512,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, L] integer token ids
        # return: [B, T, embed_dim]
        ...
```

Forward-pass behavior:

- Require input rank `3`.
- Convert `x` to `long`.
- Flatten `[B, T, L]` to `[B*T, L]`.
- Build padding mask with `x.eq(pad_id)`.
- Embed tokens and positions.
- Run a small `nn.TransformerEncoder` with `src_key_padding_mask`.
- Mean-pool non-padding positions.
- Clamp the pooling denominator to at least `1`.
- Project or normalize the pooled vector and reshape to `[B, T, embed_dim]`.

The encoder must not import GameAgent's tokenizer at runtime. The tokenizer is
only needed during dataset conversion and candidate-action preparation.

### 3. Add LeWM Training Config

Add `le-wm/config/train/data/gameagent.yaml`:

```yaml
dataset:
  num_steps: ${eval:'${num_preds} + ${history_size}'}
  frameskip: 1
  name: gameagent.h5
  keys_to_load:
    - pixels
    - action
  keys_to_cache:
    - action
```

Update `le-wm/config/train/model/lewm.yaml` action encoder block:

```yaml
action_encoder:
  _target_: discrete_action_encoder.DiscreteActionEncoder
  vocab_size: 120
  embed_dim: ${embed_dim}
  max_seq_len: 128
  num_layers: 2
  nhead: 4
  mlp_dim: 512
  pad_id: 0
  dropout: 0.1
```

Set `vocab_size` to the exact value from
`data_processing/outputs/action_vocab.json` before training. If the value is not
known when editing the config, use a Hydra override at launch:

```bash
python train.py data=gameagent model.action_encoder.vocab_size=<VOCAB_SIZE>
```

### 4. Update LeWM Training Script For Discrete Actions

Modify `le-wm/train.py`.

In `lejepa_forward`, replace unconditional action cleanup:

```python
batch["action"] = torch.nan_to_num(batch["action"], 0.0)
```

with:

```python
if batch["action"].is_floating_point():
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
```

In the transform setup, skip action normalization:

```python
for col in cfg.data.dataset.keys_to_load:
    if col.startswith("pixels") or col == "action":
        continue
    normalizer = get_column_normalizer(dataset, col, col)
    transforms.append(normalizer)
```

Guard the continuous-action input-dim override so it only applies to the
existing `module.Embedder`:

```python
if cfg.model.action_encoder.get("_target_") == "module.Embedder":
    cfg.model.action_encoder.input_dim = (
        cfg.data.dataset.frameskip * dataset.get_dim("action")
    )
```

Do not set `input_dim` for `DiscreteActionEncoder`.

### 5. Offline Evaluation Before Live-Agent Integration

Add an offline planning script in `le-wm`, for example:

`eval_gameagent_offline.py`

Inputs:

- trained LeWM checkpoint
- `gameagent.h5`
- tokenizer vocab JSON
- candidate action JSON or text file
- `history_size`, default `3`
- `goal_offset`, default `25`

Candidate actions:

- Build a fixed list from frequent normalized training actions.
- Include a no-op action.
- Start with exhaustive one-step greedy enumeration over the candidate list.
- Keep Categorical CEM out of v1 evaluation because its current implementation
  emits category indices/one-hot vectors, not token-id action sequences.

Evaluation flow:

- Load validation episodes from HDF5.
- Sample start positions where `start + history_size + goal_offset` is inside
  the same episode.
- Load context frames `[t, t+history_size)`.
- Load goal frame `t + goal_offset`.
- Encode candidate actions to `[num_candidates, 128]`.
- Repeat context across candidates.
- Use `model.rollout()` or direct `encode()` + `predict()` calls to predict the
  next embedding under each candidate.
- Encode the goal image with `model.encode()`.
- Score candidates by MSE distance to goal embedding.
- Compare selected action distance improvement against:
  - no-op baseline
  - random candidate baseline
  - behavior-cloning action from the dataset, if available

Primary metrics:

- mean selected predicted distance
- mean random predicted distance
- selected-vs-random win rate
- selected-vs-no-op win rate
- percentage of samples where predicted distance to goal decreases

Only after this offline evaluation is better than random/no-op should the world
model be connected to the live agent.

## Test Plan

### Dataset Conversion Smoke Test

After creating `gameagent.h5`, run:

```python
import stable_worldmodel as swm

ds = swm.data.load_dataset(
    "gameagent.h5",
    num_steps=4,
    frameskip=1,
    keys_to_load=["pixels", "action"],
    keys_to_cache=["action"],
)

sample = ds[0]
assert sample["pixels"].shape[0] == 4
assert sample["action"].shape == (4, 128)
assert str(sample["action"].dtype) in {"torch.int64", "torch.long"}
```

### LeWM Encoder Smoke Test

Run one forward pass with a synthetic token batch:

```python
import torch
from discrete_action_encoder import DiscreteActionEncoder

enc = DiscreteActionEncoder(vocab_size=120, embed_dim=192)
x = torch.zeros(2, 4, 128, dtype=torch.long)
y = enc(x)
assert y.shape == (2, 4, 192)
assert y.dtype.is_floating_point
```

### Training Smoke Test

Run a short LeWM training job:

```bash
python train.py \
  data=gameagent \
  trainer.max_epochs=1 \
  loader.batch_size=4 \
  loader.num_workers=0 \
  model.action_encoder.vocab_size=<VOCAB_SIZE>
```

Expected result:

- dataset loads
- no action normalizer is fitted
- `DiscreteActionEncoder` receives integer token tensors
- one train step completes
- checkpoint writing reaches `$STABLEWM_HOME/checkpoints`

### Offline Planning Smoke Test

Run the offline evaluator on a small number of validation samples:

```bash
python eval_gameagent_offline.py \
  --checkpoint <RUN_NAME_OR_PATH> \
  --dataset "$STABLEWM_HOME/datasets/gameagent.h5" \
  --vocab-path /home/sarthak/Desktop/GameAgent/data_processing/outputs/action_vocab.json \
  --candidate-actions candidates/gameagent_actions.txt \
  --num-samples 32 \
  --history-size 3 \
  --goal-offset 25
```

Expected result:

- script reports selected, random, and no-op distances
- selected candidate beats random/no-op above chance before live integration

## Rollout Order

1. Implement and validate the HDF5 conversion script in `GameAgent`.
2. Generate `gameagent.h5` under `$STABLEWM_HOME/datasets/`.
3. Add `DiscreteActionEncoder` and LeWM config updates.
4. Patch `le-wm/train.py` for discrete action handling.
5. Run dataset and encoder smoke tests.
6. Run a one-epoch training smoke test.
7. Train a real checkpoint on GPU.
8. Run offline planning evaluation.
9. Only after offline metrics beat baselines, connect the planner to the live
   GameAgent loop.

## Assumptions And Defaults

- Use `implementation_plan.md` as the handoff filename.
- Keep the plan document in the `GameAgent` repo root.
- Keep `stable-worldmodel` unchanged for the first implementation.
- Use `ActionTokenizer` from `GameAgent` as the only canonical tokenizer.
- Use image size `224`.
- Use `max_action_tokens=128`.
- Use `pad_id=0`.
- Use `history_size=3` and `num_preds=1`, so training samples contain
  `num_steps=4`.
- Use enumerative offline planning first. Treat `CategoricalCEMSolver` as a
  later optimization after token-sequence action planning is validated.
