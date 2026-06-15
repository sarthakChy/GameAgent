from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from .vjepa2_dataset import (
    ShardedEmbeddingActionDataset,
    ActionTokenizer,
    _load_json,
    _parse_shards,
)


class ConsecutivePairDataset(Dataset):
    """
    Builds consecutive (z_t, action, z_{t+1}) pairs from a sharded embedding
    dataset, respecting episode boundaries (no cross‑episode pairs).

    Args:
        base_dataset: The underlying ShardedEmbeddingActionDataset.
    """

    def __init__(self, base_dataset: ShardedEmbeddingActionDataset) -> None:
        self.base_dataset = base_dataset
        self.tokenizer = base_dataset.tokenizer  # for action tokenisation
        self.max_action_tokens = base_dataset.max_action_tokens
        self.pad_to_max_action_tokens = base_dataset.pad_to_max_action_tokens
        self.pair_index_path = self._default_pair_index_path(self.base_dataset.index_path)

        self.pairs: list[tuple[int, int]] = []
        if not self._load_cached_pairs():
            self._build_pairs()
            self._save_cached_pairs()

    @staticmethod
    def _default_pair_index_path(index_path: Path) -> Path:
        name = index_path.name
        for suffix in (".index.json", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return index_path.with_name(f"{name}.pair_index.json")

    def _source_signature(self) -> dict:
        index_data = _load_json(self.base_dataset.index_path)
        shards = _parse_shards(index_data, self.base_dataset.index_path)

        shard_signature = []
        for shard in shards:
            stat = shard.path.stat()
            shard_signature.append(
                {
                    "path": str(shard.path),
                    "num_samples": shard.num_samples,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                }
            )

        return {
            "index_path": str(self.base_dataset.index_path),
            "index_mtime_ns": self.base_dataset.index_path.stat().st_mtime_ns,
            "index_size": self.base_dataset.index_path.stat().st_size,
            "shards": shard_signature,
        }

    def _load_cached_pairs(self) -> bool:
        if not self.pair_index_path.exists():
            return False

        try:
            cache = json.loads(self.pair_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if cache.get("cache_version") != 1:
            return False

        if cache.get("source_signature") != self._source_signature():
            return False

        raw_pairs = cache.get("pairs", [])
        self.pairs = [(int(pair[0]), int(pair[1])) for pair in raw_pairs]
        return len(self.pairs) > 0

    def _save_cached_pairs(self) -> None:
        cache = {
            "cache_version": 1,
            "source_signature": self._source_signature(),
            "pairs": [[int(idx_t), int(idx_t1)] for idx_t, idx_t1 in self.pairs],
        }

        try:
            self.pair_index_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError as exc:
            warnings.warn(f"Could not write pair index cache {self.pair_index_path}: {exc}")

    def _build_pairs(self) -> None:
        """Scans episode groups and creates (idx_t, idx_t+1) tuples."""
        # Re‑open the shard index to access raw metadata quickly.
        index_data = _load_json(self.base_dataset.index_path)
        shards = _parse_shards(index_data, self.base_dataset.index_path)

        # Collect all episode data sorted by episode_id and frame_index.
        episode_frames: dict[str, list[tuple[int, int]]] = {}
        global_idx = 0
        for shard in shards:
            payload = torch.load(shard.path, map_location="cpu")
            num_in_shard = int(payload["meta"]["num_samples"])
            episode_ids = [str(e) for e in payload["episode_id"]]
            frame_indices = [int(f) for f in payload["frame_index"]]

            for local_idx in range(num_in_shard):
                ep = episode_ids[local_idx]
                frm = frame_indices[local_idx]
                episode_frames.setdefault(ep, []).append((global_idx, frm))
                global_idx += 1

        # Within each episode, sort by frame_index and create consecutive pairs.
        for ep, items in episode_frames.items():
            sorted_items = sorted(items, key=lambda x: x[1])
            for i in range(len(sorted_items) - 1):
                idx_t = sorted_items[i][0]
                idx_t1 = sorted_items[i + 1][0]
                self.pairs.append((idx_t, idx_t1))

        if not self.pairs:
            raise ValueError(
                "No consecutive pairs could be built. "
                "Check episode consistency and that each episode has at least 2 frames."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        idx_t, idx_t1 = self.pairs[index]

        # Fetch current and next samples from the base dataset.
        sample_t = self.base_dataset[idx_t]
        sample_t1 = self.base_dataset[idx_t1]

        # Tokenise the action (from the current frame).
        action_ids = self.tokenizer.encode(
            sample_t["action_text"],
            max_length=self.max_action_tokens,
            pad_to_max_length=self.pad_to_max_action_tokens,
        )

        return {
            "embedding": sample_t["embedding"],           # z_t
            "action_ids": action_ids,
            "action_length": action_ids.numel(),
            "next_embedding": sample_t1["embedding"],     # z_{t+1}
            "episode_id": sample_t["episode_id"],
            "frame_index": sample_t["frame_index"],
        }


def make_consecutive_pair_collate(pad_id: int) -> Callable[[list[dict]], dict]:
    """Creates a collate function for ConsecutivePairDataset with dynamic action padding."""

    def collate(batch: list[dict]) -> dict:
        embeddings = torch.stack([b["embedding"] for b in batch], dim=0)
        next_embeddings = torch.stack([b["next_embedding"] for b in batch], dim=0)

        # Pad action_ids
        action_lens = torch.tensor([b["action_ids"].numel() for b in batch], dtype=torch.long)
        max_len = int(action_lens.max().item()) if len(batch) else 0
        action_ids = torch.full((len(batch), max_len), fill_value=pad_id, dtype=torch.long)

        for i, item in enumerate(batch):
            n = item["action_ids"].numel()
            action_ids[i, :n] = item["action_ids"]

        return {
            "embedding": embeddings,  # z_t
            "action_ids": action_ids,
            "action_length": action_lens,
            "next_embedding": next_embeddings,  # z_{t+1}
            "episode_id": [b["episode_id"] for b in batch],
            "frame_index": torch.tensor([b["frame_index"] for b in batch], dtype=torch.long),
        }

    return collate
