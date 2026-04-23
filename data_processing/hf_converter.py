from __future__ import annotations

import argparse
import io
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, login


SCHEMA_VERSION = "canonical-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical session-split dataset and upload to Hugging Face."
    )
    parser.add_argument(
        "--recordings-dir",
        required=True,
        help="Path to recordings directory containing session_* folders.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="HF dataset repo id, for example 'username/gameagent-canonical'.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional allowlist of session folder names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print stats without pushing.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Push as public dataset. Default is private.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN or HUGGINGFACE_TOKEN from environment.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Append new sessions to an existing HF dataset repo by skipping already uploaded "
            "episode_id values and merging old+new before push."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic session split.",
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=[0.8, 0.1, 0.1],
        help="Ratios for train/validation/test splits. Must sum to 1.0.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["temporal-per-session", "session-level", "all-train"],
        default="temporal-per-session",
        help=(
            "Split strategy. 'temporal-per-session' puts early frames in train and later "
            "frames in validation/test within each session. 'session-level' assigns whole sessions. "
            "'all-train' puts all rows in train for pretraining."
        ),
    )
    return parser.parse_args()


def find_sessions(recordings_dir: Path, selected: list[str] | None) -> list[Path]:
    all_sessions = sorted(
        p for p in recordings_dir.iterdir() if p.is_dir() and p.name.startswith("session_")
    )
    if selected:
        selected_set = set(selected)
        all_sessions = [p for p in all_sessions if p.name in selected_set]

    valid = []
    for session_dir in all_sessions:
        pairs_path = session_dir / f"{session_dir.name}_pairs.jsonl"
        if not pairs_path.exists():
            print(f"  [skip] {session_dir.name}: missing {pairs_path.name}")
            continue
        valid.append(session_dir)
    return valid


def load_session_rows(session_dir: Path) -> list[dict]:
    pairs_path = session_dir / f"{session_dir.name}_pairs.jsonl"
    frames_dir = session_dir / "frames"

    rows: list[dict] = []
    skipped_no_frame = 0
    skipped_missing_file = 0
    skipped_bad_json = 0

    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad_json += 1
                continue

            frame_path_raw = record.get("frame_path")
            if not frame_path_raw:
                skipped_no_frame += 1
                continue

            local_frame = frames_dir / Path(frame_path_raw).name
            if not local_frame.exists():
                skipped_missing_file += 1
                continue

            rows.append(
                {
                    "image": {"path": str(local_frame)},
                    "action_text": str(record.get("action", "")),
                    "episode_id": session_dir.name,
                    "frame_index": int(record.get("frame_index", -1)),
                    "t_start_ms": float(record.get("t_start_ms", 0.0)),
                    "is_idle": bool(record.get("is_idle", False)),
                    "horizontal_scroll_steps": int(record.get("horizontal_scroll_steps", 0)),
                    "source_session_relpath": str(Path("recordings") / session_dir.name),
                }
            )

    if skipped_bad_json:
        print(f"    [warn] {session_dir.name}: skipped {skipped_bad_json} malformed json line(s)")
    if skipped_no_frame:
        print(f"    [warn] {session_dir.name}: skipped {skipped_no_frame} row(s) with no frame_path")
    if skipped_missing_file:
        print(f"    [warn] {session_dir.name}: skipped {skipped_missing_file} row(s) with missing frame file")

    return rows


def get_session_game_name(session_dir: Path) -> str:
    meta_path = session_dir / f"{session_dir.name}_meta.json"
    if not meta_path.exists():
        print(f"    [warn] {session_dir.name}: missing meta file, using game='unknown'")
        return "unknown"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"    [warn] {session_dir.name}: malformed meta json, using game='unknown'")
        return "unknown"
    return str(meta.get("game_name") or meta.get("game") or "unknown")


def add_game_field(rows: list[dict], game_name: str) -> list[dict]:
    return [{**row, "game": game_name} for row in rows]


def assign_session_splits(
    sessions: list[str], ratios: list[float], seed: int
) -> dict[str, str]:
    train_r, val_r, test_r = ratios
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    names = sessions[:]
    random.Random(seed).shuffle(names)

    total = len(names)
    if total == 0:
        return {}
    if total == 1:
        return {names[0]: "train"}
    if total == 2:
        return {
            names[0]: "train",
            names[1]: "validation",
        }

    train_n = int(total * train_r)
    val_n = int(total * val_r)
    test_n = total - train_n - val_n

    train_n = max(1, train_n)
    val_n = max(1, val_n)
    test_n = max(1, test_n)

    while train_n + val_n + test_n > total:
        if train_n >= val_n and train_n >= test_n and train_n > 1:
            train_n -= 1
        elif val_n >= test_n and val_n > 1:
            val_n -= 1
        elif test_n > 1:
            test_n -= 1

    while train_n + val_n + test_n < total:
        train_n += 1

    split_map: dict[str, str] = {}
    for i, name in enumerate(names):
        if i < train_n:
            split_map[name] = "train"
        elif i < train_n + val_n:
            split_map[name] = "validation"
        else:
            split_map[name] = "test"
    return split_map


def split_rows_temporal(rows: list[dict], ratios: list[float]) -> dict[str, list[dict]]:
    train_r, val_r, test_r = ratios
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    ordered = sorted(rows, key=lambda r: (r["frame_index"], r["t_start_ms"]))
    n = len(ordered)
    if n == 0:
        return {"train": [], "validation": [], "test": []}
    if n == 1:
        return {"train": ordered, "validation": [], "test": []}
    if n == 2:
        return {
            "train": [ordered[0]],
            "validation": [ordered[1]],
            "test": [],
        }

    train_n = max(1, int(n * train_r))
    val_n = max(1, int(n * val_r))
    test_n = max(1, n - train_n - val_n)

    while train_n + val_n + test_n > n:
        if train_n >= val_n and train_n >= test_n and train_n > 1:
            train_n -= 1
        elif val_n >= test_n and val_n > 1:
            val_n -= 1
        elif test_n > 1:
            test_n -= 1

    while train_n + val_n + test_n < n:
        train_n += 1

    return {
        "train": ordered[:train_n],
        "validation": ordered[train_n:train_n + val_n],
        "test": ordered[train_n + val_n:train_n + val_n + test_n],
    }


def split_rows_all_train(rows: list[dict]) -> dict[str, list[dict]]:
    ordered = sorted(rows, key=lambda r: (r["frame_index"], r["t_start_ms"]))
    return {
        "train": ordered,
        "validation": [],
        "test": [],
    }


def build_datasetdict(rows_by_split: dict[str, list[dict]]) -> DatasetDict:
    features = Features(
        {
            "image": Image(),
            "action_text": Value("string"),
            "game": Value("string"),
            "episode_id": Value("string"),
            "frame_index": Value("int32"),
            "t_start_ms": Value("float32"),
            "is_idle": Value("bool"),
            "horizontal_scroll_steps": Value("int16"),
            "source_session_relpath": Value("string"),
        }
    )

    result = DatasetDict()
    for split in ("train", "validation", "test"):
        rows = rows_by_split.get(split, [])
        payload = {
            "image": [r["image"] for r in rows],
            "action_text": [r["action_text"] for r in rows],
            "game": [r["game"] for r in rows],
            "episode_id": [r["episode_id"] for r in rows],
            "frame_index": [r["frame_index"] for r in rows],
            "t_start_ms": [r["t_start_ms"] for r in rows],
            "is_idle": [r["is_idle"] for r in rows],
            "horizontal_scroll_steps": [r["horizontal_scroll_steps"] for r in rows],
            "source_session_relpath": [r["source_session_relpath"] for r in rows],
        }
        result[split] = Dataset.from_dict(payload, features=features)

    return result


def get_token(cli_token: str | None) -> str | None:
    load_dotenv()
    return cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def load_existing_datasetdict(repo_id: str, token: str | None) -> DatasetDict | None:
    try:
        existing = load_dataset(repo_id, token=token)
    except Exception:
        return None
    if isinstance(existing, DatasetDict):
        return existing
    return None


def get_existing_episode_ids(existing: DatasetDict | None) -> set[str]:
    if existing is None:
        return set()
    episode_ids: set[str] = set()
    for split in existing.keys():
        ds = existing[split]
        if "episode_id" not in ds.column_names:
            continue
        for eid in ds["episode_id"]:
            episode_ids.add(str(eid))
    return episode_ids


def merge_existing_and_new(existing: DatasetDict | None, new_data: DatasetDict) -> DatasetDict:
    if existing is None:
        return new_data

    merged = DatasetDict()
    for split in ("train", "validation", "test"):
        existing_ds = existing[split] if split in existing else None
        new_ds = new_data[split] if split in new_data else None

        if existing_ds is not None and new_ds is not None:
            if len(new_ds) == 0:
                merged[split] = existing_ds
            elif len(existing_ds) == 0:
                merged[split] = new_ds
            else:
                merged[split] = concatenate_datasets([existing_ds, new_ds])
        elif existing_ds is not None:
            merged[split] = existing_ds
        elif new_ds is not None:
            merged[split] = new_ds

    return merged


def normalize_game_column(ds_dict: DatasetDict, default_game: str = "unknown") -> DatasetDict:
    normalized = DatasetDict()
    for split_name, split_ds in ds_dict.items():
        if "game" in split_ds.column_names:
            normalized[split_name] = split_ds
            continue
        normalized[split_name] = split_ds.add_column("game", [default_game] * len(split_ds))
    return normalized


def drop_empty_splits(ds_dict: DatasetDict) -> DatasetDict:
    filtered = DatasetDict()
    for split_name, split_ds in ds_dict.items():
        if len(split_ds) > 0:
            filtered[split_name] = split_ds
    return filtered


def build_dataset_card(
    split_strategy: str,
    allocation_text: str,
    ratios: list[float],
    seed: int,
) -> str:
    return f"""---
pretty_name: GAMEAGENT Canonical Dataset
license: mit
task_categories:
- image-to-text
tags:
- game-agent
- imitation-learning
- behavioral-cloning
---

# GAMEAGENT Canonical Dataset

Schema version: {SCHEMA_VERSION}

This dataset contains canonical frame-action pairs from GAMEAGENT recordings.

## Splits

- Split strategy: {split_strategy}
- Ratios (requested): train={ratios[0]}, validation={ratios[1]}, test={ratios[2]}
- Seed: {seed}

Session allocation:
{allocation_text}

## Features

- image: datasets.Image
- action_text: string
- game: string
- episode_id: string
- frame_index: int32
- t_start_ms: float32
- is_idle: bool
- horizontal_scroll_steps: int16
- source_session_relpath: string

## Action format

DX DY DZ ; chunk1 ; chunk2 ; chunk3 ; chunk4 ; chunk5 ; chunk6

- DX, DY: mouse motion over a 200ms window
- DZ: vertical scroll steps
- horizontal_scroll_steps: horizontal wheel steps in a separate field
- chunk1..chunk6: comma-separated held keys/buttons for ~33ms sub-windows
"""


def upload_dataset_card(repo_id: str, card_text: str, token: str | None) -> None:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=io.BytesIO(card_text.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def main() -> None:
    args = parse_args()

    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.exists():
        raise SystemExit(f"Recordings directory not found: {recordings_dir}")

    print(f"Scanning recordings in: {recordings_dir}")
    session_dirs = find_sessions(recordings_dir, args.sessions)
    if not session_dirs:
        raise SystemExit("No valid session directories found.")

    token = get_token(args.hf_token)
    if token:
        login(token=token)

    existing_ds: DatasetDict | None = None
    existing_episode_ids: set[str] = set()
    if args.incremental:
        if not args.repo:
            raise SystemExit("--repo is required when --incremental is used.")
        existing_ds = load_existing_datasetdict(args.repo, token)
        if existing_ds is not None:
            existing_ds = normalize_game_column(existing_ds)
        existing_episode_ids = get_existing_episode_ids(existing_ds)
        if existing_episode_ids:
            print(f"Incremental mode: found {len(existing_episode_ids)} existing episode_id value(s) on Hub.")
        else:
            print("Incremental mode: no existing dataset found (or repo is empty).")

    print(f"Found {len(session_dirs)} session(s).")
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    per_session_counts: dict[str, int] = {}
    allocation_lines: list[str] = []

    if args.split_strategy == "session-level":
        session_names = [s.name for s in session_dirs]
        split_map = assign_session_splits(session_names, args.split_ratios, args.seed)

        for session_dir in session_dirs:
            if session_dir.name in existing_episode_ids:
                print(f"  {session_dir.name}: already on hub, skipping")
                continue
            game_name = get_session_game_name(session_dir)
            rows = add_game_field(load_session_rows(session_dir), game_name)
            split = split_map[session_dir.name]
            rows_by_split[split].extend(rows)
            per_session_counts[session_dir.name] = len(rows)
            allocation_lines.append(f"- {session_dir.name}: {split}")
            print(f"  {session_dir.name}: {len(rows)} rows -> {split} [{game_name}]")
    elif args.split_strategy == "temporal-per-session":
        for session_dir in session_dirs:
            if session_dir.name in existing_episode_ids:
                print(f"  {session_dir.name}: already on hub, skipping")
                continue
            game_name = get_session_game_name(session_dir)
            rows = add_game_field(load_session_rows(session_dir), game_name)
            split_rows = split_rows_temporal(rows, args.split_ratios)
            for split in ("train", "validation", "test"):
                rows_by_split[split].extend(split_rows[split])
            per_session_counts[session_dir.name] = len(rows)
            allocation_lines.append(
                f"- {session_dir.name}: train={len(split_rows['train'])}, "
                f"validation={len(split_rows['validation'])}, test={len(split_rows['test'])}"
            )
            print(
                f"  {session_dir.name}: total={len(rows)} "
                f"(train={len(split_rows['train'])}, validation={len(split_rows['validation'])}, test={len(split_rows['test'])}) [{game_name}]"
            )
    else:
        for session_dir in session_dirs:
            if session_dir.name in existing_episode_ids:
                print(f"  {session_dir.name}: already on hub, skipping")
                continue
            game_name = get_session_game_name(session_dir)
            rows = add_game_field(load_session_rows(session_dir), game_name)
            split_rows = split_rows_all_train(rows)
            rows_by_split["train"].extend(split_rows["train"])
            per_session_counts[session_dir.name] = len(rows)
            allocation_lines.append(f"- {session_dir.name}: train={len(split_rows['train'])}, validation=0, test=0")
            print(f"  {session_dir.name}: total={len(rows)} (train={len(split_rows['train'])}, validation=0, test=0) [{game_name}]")

    total_rows = sum(len(v) for v in rows_by_split.values())
    if total_rows == 0:
        raise SystemExit("No rows available after filtering and frame checks.")

    print("\nSplit summary:")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {len(rows_by_split[split])} rows")

    print("\nSession summary:")
    for name, cnt in sorted(per_session_counts.items()):
        print(f"  {name}: {cnt} rows")

    print("\nSession allocation:")
    if args.split_strategy == "session-level":
        split_session_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        for line in allocation_lines:
            if line.endswith(": train"):
                split_session_counts["train"] += 1
            elif line.endswith(": validation"):
                split_session_counts["validation"] += 1
            elif line.endswith(": test"):
                split_session_counts["test"] += 1
        for split in ("train", "validation", "test"):
            print(f"  {split}: {split_session_counts.get(split, 0)} session(s)")
    elif args.split_strategy == "temporal-per-session":
        print("  Per-session temporal split (every session contributes to train).")
    else:
        print("  All-train split (all sessions and rows are in train).")

    new_ds_dict = build_datasetdict(rows_by_split)
    ds_dict = merge_existing_and_new(existing_ds, new_ds_dict) if args.incremental else new_ds_dict
    print("\nBuilt canonical DatasetDict:")
    print(ds_dict)

    if args.dry_run:
        print("\n[dry-run] Skipping push to Hugging Face.")
        return

    if not args.repo:
        raise SystemExit("--repo is required unless --dry-run is used.")

    ds_for_push = drop_empty_splits(ds_dict)
    if len(ds_for_push) == 0:
        raise SystemExit("No non-empty split to push. Nothing to upload.")

    print(f"\nPushing DatasetDict to: {args.repo}")
    if len(ds_for_push) != len(ds_dict):
        kept = ", ".join(ds_for_push.keys())
        print(f"  Dropping empty split(s) for upload. Keeping: {kept}")

    ds_for_push.push_to_hub(
        repo_id=args.repo,
        private=not args.public,
        token=token,
        max_shard_size="500MB",
    )
    card_text = build_dataset_card(
        split_strategy=args.split_strategy,
        allocation_text="\n".join(allocation_lines) if allocation_lines else "- none",
        ratios=args.split_ratios,
        seed=args.seed,
    )
    upload_dataset_card(args.repo, card_text, token)
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

