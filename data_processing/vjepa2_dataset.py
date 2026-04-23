from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    num_samples: int


class ActionTokenizer:
    """Tokenizes GAMEAGENT action strings into integer ids.

    Expected raw action format:
    "DX DY DZ ; group1 ; group2 ; group3 ; group4 ; group5 ; group6"

    Output token sequence is canonicalized to:
    <action_start>, dx_<n>, dy_<n>, dz_<n>, <group_1>, ... , <group_6>, <action_end>

    Each key inside groups is emitted as key_<name>. Empty groups emit <empty_group>.
    """

    PAD = "<pad>"
    UNK = "<unk>"
    ACTION_START = "<action_start>"
    ACTION_END = "<action_end>"
    EMPTY_GROUP = "<empty_group>"

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

        required = [self.PAD, self.UNK, self.ACTION_START, self.ACTION_END, self.EMPTY_GROUP]
        for token in required:
            if token not in self.token_to_id:
                raise ValueError(f"Missing required token in vocabulary: {token}")

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @classmethod
    def build_from_index(
        cls,
        index_path: str | Path,
        *,
        min_freq: int = 1,
    ) -> "ActionTokenizer":
        index_data = _load_json(index_path)
        shards = _parse_shards(index_data, index_path)

        counter: Counter[str] = Counter()
        for shard in shards:
            payload = torch.load(shard.path, map_location="cpu")
            action_texts = payload.get("action_text", [])
            for action_text in action_texts:
                counter.update(cls._tokenize_action(str(action_text)))

        special_tokens = [
            cls.PAD,
            cls.UNK,
            cls.ACTION_START,
            cls.ACTION_END,
            cls.EMPTY_GROUP,
            "<group_1>",
            "<group_2>",
            "<group_3>",
            "<group_4>",
            "<group_5>",
            "<group_6>",
        ]

        vocab_tokens: list[str] = []
        seen = set()
        for token in special_tokens:
            if token not in seen:
                vocab_tokens.append(token)
                seen.add(token)

        for token, freq in sorted(counter.items()):
            if freq < min_freq:
                continue
            if token in seen:
                continue
            vocab_tokens.append(token)
            seen.add(token)

        token_to_id = {token: i for i, token in enumerate(vocab_tokens)}
        return cls(token_to_id)

    @classmethod
    def load(cls, vocab_path: str | Path) -> "ActionTokenizer":
        data = _load_json(vocab_path)
        token_to_id = data["token_to_id"]
        token_to_id = {str(k): int(v) for k, v in token_to_id.items()}
        return cls(token_to_id)

    def save(self, vocab_path: str | Path) -> None:
        path = Path(vocab_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"token_to_id": self.token_to_id}, indent=2), encoding="utf-8")

    @classmethod
    def _tokenize_action(cls, action_text: str) -> list[str]:
        segments = [seg.strip() for seg in action_text.split(";")]

        if not segments:
            return [cls.ACTION_START, cls.EMPTY_GROUP, cls.ACTION_END]

        mouse_tokens: list[str] = ["dx_0", "dy_0", "dz_0"]
        mouse_parts = [p for p in segments[0].split() if p]
        if len(mouse_parts) >= 1:
            mouse_tokens[0] = f"dx_{mouse_parts[0]}"
        if len(mouse_parts) >= 2:
            mouse_tokens[1] = f"dy_{mouse_parts[1]}"
        if len(mouse_parts) >= 3:
            mouse_tokens[2] = f"dz_{mouse_parts[2]}"

        groups = segments[1:7]
        if len(groups) < 6:
            groups += [""] * (6 - len(groups))

        tokens: list[str] = [cls.ACTION_START, *mouse_tokens]
        for i, group in enumerate(groups, start=1):
            tokens.append(f"<group_{i}>")
            keys = [k.strip().lower() for k in group.split(",") if k.strip()]
            if not keys:
                tokens.append(cls.EMPTY_GROUP)
                continue
            tokens.extend([f"key_{key}" for key in keys])

        tokens.append(cls.ACTION_END)
        return tokens

    def encode(
        self,
        action_text: str,
        *,
        max_length: int | None = None,
        pad_to_max_length: bool = False,
    ) -> torch.Tensor:
        tokens = self._tokenize_action(action_text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]

        if max_length is not None:
            ids = ids[:max_length]
            if pad_to_max_length and len(ids) < max_length:
                ids += [self.pad_id] * (max_length - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor | list[int]) -> list[str]:
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return [self.id_to_token.get(int(i), self.UNK) for i in ids]


class ShardedEmbeddingActionDataset(Dataset):
    """Reads chunked V-JEPA2 embedding shards lazily via index JSON.

    Each item returns:
    - embedding: FloatTensor[D]
    - action_ids: LongTensor[T]
    - action_length: int
    - metadata fields: episode_id, frame_index, game, is_idle, etc.
    """

    def __init__(
        self,
        *,
        index_path: str | Path,
        tokenizer: ActionTokenizer,
        max_action_tokens: int | None = None,
        pad_to_max_action_tokens: bool = False,
        shard_cache_size: int = 2,
    ) -> None:
        self.index_path = Path(index_path)
        self.tokenizer = tokenizer
        self.max_action_tokens = max_action_tokens
        self.pad_to_max_action_tokens = pad_to_max_action_tokens
        self.shard_cache_size = max(1, shard_cache_size)

        index_data = _load_json(self.index_path)
        self.shards = _parse_shards(index_data, self.index_path)
        if not self.shards:
            raise ValueError("No shards found in index file.")

        self._offsets: list[int] = []
        running = 0
        for shard in self.shards:
            self._offsets.append(running)
            running += shard.num_samples
        self._total = running

        self._cache: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return self._total

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._total:
            raise IndexError(f"Index out of range: {index}")

        shard_idx = bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[shard_idx]
        return shard_idx, local_idx

    def _get_shard_payload(self, shard_idx: int) -> dict:
        if shard_idx in self._cache:
            payload = self._cache.pop(shard_idx)
            self._cache[shard_idx] = payload
            return payload

        payload = torch.load(self.shards[shard_idx].path, map_location="cpu")
        self._cache[shard_idx] = payload

        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)

        return payload

    def __getitem__(self, index: int) -> dict:
        shard_idx, local_idx = self._locate(index)
        payload = self._get_shard_payload(shard_idx)

        embedding = payload["embeddings"][local_idx].to(torch.float32)
        action_text = str(payload["action_text"][local_idx])
        action_ids = self.tokenizer.encode(
            action_text,
            max_length=self.max_action_tokens,
            pad_to_max_length=self.pad_to_max_action_tokens,
        )

        return {
            "embedding": embedding,
            "action_ids": action_ids,
            "action_length": int(action_ids.numel()),
            "action_text": action_text,
            "episode_id": str(payload["episode_id"][local_idx]),
            "frame_index": int(payload["frame_index"][local_idx]),
            "t_start_ms": float(payload["t_start_ms"][local_idx]),
            "is_idle": bool(payload["is_idle"][local_idx]),
            "horizontal_scroll_steps": int(payload["horizontal_scroll_steps"][local_idx]),
            "source_session_relpath": str(payload["source_session_relpath"][local_idx]),
            "game": str(payload["game"][local_idx]),
        }


def make_collate_fn(pad_id: int) -> Callable[[list[dict]], dict]:
    """Creates a collate function with dynamic action padding per batch."""

    def collate(batch: list[dict]) -> dict:
        embeddings = torch.stack([b["embedding"] for b in batch], dim=0)

        action_lens = torch.tensor([b["action_ids"].numel() for b in batch], dtype=torch.long)
        max_len = int(action_lens.max().item()) if len(batch) else 0
        action_ids = torch.full((len(batch), max_len), fill_value=pad_id, dtype=torch.long)

        for i, item in enumerate(batch):
            n = item["action_ids"].numel()
            action_ids[i, :n] = item["action_ids"]

        return {
            "embedding": embeddings,
            "action_ids": action_ids,
            "action_length": action_lens,
            "episode_id": [b["episode_id"] for b in batch],
            "frame_index": torch.tensor([b["frame_index"] for b in batch], dtype=torch.long),
            "t_start_ms": torch.tensor([b["t_start_ms"] for b in batch], dtype=torch.float32),
            "is_idle": torch.tensor([b["is_idle"] for b in batch], dtype=torch.bool),
            "game": [b["game"] for b in batch],
        }

    return collate


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_shards(index_data: dict, index_path: str | Path) -> list[ShardInfo]:
    base_dir = Path(index_path).parent
    raw_shards = index_data.get("shards", [])

    shards: list[ShardInfo] = []
    for shard in raw_shards:
        raw_path = Path(str(shard["path"]))
        shard_path = raw_path if raw_path.is_absolute() else (base_dir / raw_path)
        shards.append(ShardInfo(path=shard_path, num_samples=int(shard["num_samples"])))

    return shards
