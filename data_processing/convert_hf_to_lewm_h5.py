"""Convert sarthak2314/gameagent-canonical (HuggingFace) to LeWM HDF5 format.

Output HDF5 schema
------------------
pixels      uint8   [N, 224, 224, 3]   -- HWC layout (permuted to CHW by HDF5Dataset loader)
action      int64   [N, max_tokens]    -- token IDs from ActionTokenizer, pad_id=0
episode_idx int32   [N]               -- contiguous integer episode id (0-based)
step_idx    int32   [N]               -- per-episode step counter (0-based)
ep_len      int32   [num_episodes]    -- number of rows in each episode
ep_offset   int64   [num_episodes]    -- cumulative flat-row start for each episode

Notes
-----
- stable_worldmodel HDF5Dataset appends '.h5' to the name automatically, so the
  output file should be named 'gameagent.h5' on disk, but the config uses
  name: gameagent (no extension).
- Actions are stored as fixed-size int64 vectors of length --max-action-tokens
  (default 128). stable_worldmodel skips frameskip on the 'action' column and
  reshapes to [num_steps, -1] in __getitem__.
- Images are decoded eagerly inside this script to avoid PIL lazy-load issues
  under DataLoader multiprocessing in training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert gameagent HF dataset to LeWM HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo",
        default="sarthak2314/gameagent-canonical",
        help="HuggingFace dataset repo id.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to convert. Use 'all' to convert all splits into one file.",
    )
    p.add_argument(
        "--vocab-path",
        required=True,
        help="Path to action_vocab.json (produced by ActionTokenizer.save()).",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output .h5 file path. Defaults to $STABLEWM_HOME/datasets/gameagent.h5 "
            "or ./gameagent.h5 if STABLEWM_HOME is not set."
        ),
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size (square).",
    )
    p.add_argument(
        "--max-action-tokens",
        type=int,
        default=128,
        help="Pad/truncate every action to this many token IDs.",
    )
    p.add_argument(
        "--build-vocab",
        action="store_true",
        help=(
            "If set, build a new vocab from the dataset rows and save it to "
            "--vocab-path before converting. Requires the dataset to have "
            "action_text column."
        ),
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN env var.",
    )
    p.add_argument(
        "--mode",
        choices=["overwrite", "append", "error"],
        default="overwrite",
        help="HDF5Writer write mode.",
    )
    p.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes for HF dataset image decoding.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

def _get_token() -> str | None:
    import os
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _resolve_output(args: argparse.Namespace) -> Path:
    import os
    if args.output:
        return Path(args.output)
    base = os.environ.get("STABLEWM_HOME")
    if base:
        out = Path(base) / "datasets" / "gameagent.h5"
    else:
        out = Path("gameagent.h5")
    return out


def _check_vocab_path(vocab_path: Path, build: bool) -> None:
    if build:
        return  # will be created
    if not vocab_path.exists():
        print(
            f"[error] vocab file not found: {vocab_path}\n"
            "  Run with --build-vocab to create it from the dataset, or provide\n"
            "  an existing action_vocab.json via --vocab-path.",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_vocab(rows: list[dict], vocab_path: Path) -> None:
    """Build an ActionTokenizer vocab from a flat list of rows and save it."""
    # Import here so the module can be used without GameAgent on PYTHONPATH
    # by passing an existing vocab.
    sys.path.insert(0, str(Path(__file__).parent))
    from vjepa2_dataset import ActionTokenizer  # type: ignore

    from collections import Counter

    counter: Counter[str] = Counter()
    for row in rows:
        text = str(row.get("action_text", ""))
        counter.update(ActionTokenizer._tokenize_action(text))

    special_tokens = [
        ActionTokenizer.PAD,
        ActionTokenizer.UNK,
        ActionTokenizer.ACTION_START,
        ActionTokenizer.ACTION_END,
        ActionTokenizer.EMPTY_GROUP,
        "<group_1>", "<group_2>", "<group_3>",
        "<group_4>", "<group_5>", "<group_6>",
    ]
    motion_tokens = [
        f"{axis}_bin_{bucket}"
        for axis in ("dx", "dy", "dz")
        for bucket in range(ActionTokenizer.MOTION_BUCKET_MIN, ActionTokenizer.MOTION_BUCKET_MAX + 1)
    ]

    seen: set[str] = set()
    vocab: list[str] = []
    for t in special_tokens + motion_tokens:
        if t not in seen:
            vocab.append(t)
            seen.add(t)
    for t, _ in sorted(counter.items()):
        if t not in seen:
            vocab.append(t)
            seen.add(t)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps({"token_to_id": token_to_id}, indent=2), encoding="utf-8")
    print(f"[vocab] built vocab with {len(token_to_id)} tokens → {vocab_path}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _decode_image_eager(img_field, target_size: int) -> np.ndarray:
    """Decode a HuggingFace Image field to uint8 HWC numpy array.

    HF Image columns can be PIL Images, dicts with 'path'/'bytes', or raw
    bytes. We decode eagerly here to avoid PIL lazy-load under multiprocessing.
    """
    from PIL import Image

    if isinstance(img_field, dict):
        # HF datasets returns {"bytes": ..., "path": ...} for Image features
        raw_bytes = img_field.get("bytes")
        raw_path = img_field.get("path")
        if raw_bytes:
            import io
            img = Image.open(io.BytesIO(raw_bytes))
        elif raw_path:
            img = Image.open(raw_path)
        else:
            raise ValueError(f"Cannot decode image field: {img_field!r}")
    elif hasattr(img_field, "convert"):
        # Already a PIL Image
        img = img_field
    else:
        raise TypeError(f"Unexpected image field type: {type(img_field)}")

    img = img.convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)  # HWC [H, W, 3]
    return arr


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _load_rows(args: argparse.Namespace, token: str | None) -> list[dict]:
    """Load all rows from the HF dataset as plain Python dicts (no PIL objects)."""
    from datasets import load_dataset  # type: ignore

    print(f"[load] loading '{args.repo}' split='{args.split}' …")

    if args.split == "all":
        ds_dict = load_dataset(args.repo, token=token)
        from datasets import concatenate_datasets
        ds = concatenate_datasets(list(ds_dict.values()))
    else:
        ds = load_dataset(args.repo, split=args.split, token=token)

    print(f"[load] {len(ds)} rows loaded")
    return ds


def _convert(
    rows,
    tokenizer,
    output_path: Path,
    image_size: int,
    max_tokens: int,
    mode: str,
) -> None:
    """Sort rows, group into episodes, write HDF5."""
    import stable_worldmodel as swm  # type: ignore

    print("[convert] grouping and sorting rows by episode …")

    # Group by episode_id
    episodes: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx in range(len(rows)):
        row = rows[idx]
        ep_id = str(row["episode_id"])
        frame_idx = int(row["frame_index"])
        episodes[ep_id].append((frame_idx, idx))

    # Sort episodes by name (deterministic), then rows by frame_index within each
    sorted_ep_ids = sorted(episodes.keys())
    print(f"[convert] {len(sorted_ep_ids)} episodes found")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = swm.data.formats.hdf5.HDF5Writer(output_path, mode=mode)
    with writer:
        for ep_int, ep_id in enumerate(sorted_ep_ids):
            frame_row_pairs = sorted(episodes[ep_id], key=lambda x: x[0])

            ep_pixels = []
            ep_action = []
            ep_episode_idx = []
            ep_step_idx = []

            for step_int, (_, row_idx) in enumerate(frame_row_pairs):
                row = rows[row_idx]

                # --- image (eager decode) ---
                pixel = _decode_image_eager(row["image"], image_size)  # [H, W, 3] uint8
                ep_pixels.append(pixel)

                # --- action tokens ---
                action_text = str(row.get("action_text", ""))
                token_ids = tokenizer.encode(
                    action_text,
                    max_length=max_tokens,
                    pad_to_max_length=True,
                )
                ep_action.append(token_ids.numpy().astype(np.int64))  # [max_tokens]

                ep_episode_idx.append(np.int32(ep_int))
                ep_step_idx.append(np.int32(step_int))

            ep_data = {
                "pixels": ep_pixels,          # list of [H, W, 3] uint8
                "action": ep_action,           # list of [max_tokens] int64
                "episode_idx": ep_episode_idx, # list of int32 scalars
                "step_idx": ep_step_idx,       # list of int32 scalars
            }
            writer.write_episode(ep_data)

            if (ep_int + 1) % 50 == 0 or ep_int == len(sorted_ep_ids) - 1:
                total_rows = sum(len(v) for v in episodes.values())
                done = sum(len(episodes[eid]) for eid in sorted_ep_ids[:ep_int + 1])
                print(f"  [{ep_int + 1}/{len(sorted_ep_ids)}] {done}/{total_rows} rows written")

    print(f"[done] HDF5 written to: {output_path}")
    _print_summary(output_path)


def _print_summary(path: Path) -> None:
    """Print a quick sanity summary of the written file."""
    try:
        import h5py  # type: ignore
        with h5py.File(path, "r") as f:
            n_eps = len(f["ep_len"])
            n_rows = int(f["ep_len"][:].sum())
            pixels_shape = f["pixels"].shape
            action_shape = f["action"].shape
            action_dtype = f["action"].dtype
        print(
            f"[summary] episodes={n_eps}  rows={n_rows}\n"
            f"          pixels {pixels_shape} {np.dtype('uint8')}\n"
            f"          action {action_shape} {action_dtype}"
        )
    except Exception as e:
        print(f"[summary] could not read back file: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    vocab_path = Path(args.vocab_path)
    output_path = _resolve_output(args)

    # Validate vocab path (or signal we'll build it)
    _check_vocab_path(vocab_path, args.build_vocab)

    token = args.hf_token or _get_token()

    # Load rows from HF
    rows = _load_rows(args, token)

    # Optionally build vocab from data
    if args.build_vocab:
        print("[vocab] building vocab from dataset rows …")
        all_rows = [rows[i] for i in range(len(rows))]
        _build_vocab(all_rows, vocab_path)

    # Load tokenizer
    sys.path.insert(0, str(Path(__file__).parent))
    from vjepa2_dataset import ActionTokenizer  # type: ignore

    tokenizer = ActionTokenizer.load(vocab_path)
    vocab_size = len(tokenizer.token_to_id)
    print(f"[tokenizer] vocab_size={vocab_size}  pad_id={tokenizer.pad_id}")

    # Convert
    _convert(
        rows=rows,
        tokenizer=tokenizer,
        output_path=output_path,
        image_size=args.image_size,
        max_tokens=args.max_action_tokens,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
