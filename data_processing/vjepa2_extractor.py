from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor


@dataclass
class Sample:
    embedding: torch.Tensor
    action_text: str
    game: str
    episode_id: str
    frame_index: int
    t_start_ms: float
    is_idle: bool
    horizontal_scroll_steps: int
    source_session_relpath: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen V-JEPA2 embeddings from GAMEAGENT HF dataset using "
            "4-frame windows [t-3, t-2, t-1, t]."
        )
    )
    parser.add_argument(
        "--dataset-repo",
        default="sarthak2314/gameagent-canonical",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to read.",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/vjepa2-vitg-fpc64-384",
        help="Hugging Face model id for V-JEPA2.",
    )
    parser.add_argument(
        "--output",
        default="data_processing/outputs/vjepa2_embeddings.pt",
        help="Output .pt path.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Temporal window size in frames (default: 4 for 800ms at 5Hz).",
    )
    parser.add_argument(
        "--idle-streak-max",
        type=int,
        default=25,
        help="Keep idle streaks up to this many frames; drop longer idle runs.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional allowlist of episode_id values.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap for extracted samples (for quick smoke tests).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "If > 0, save samples in shard files every N rows to avoid RAM OOM. "
            "Example: --chunk-size 50000"
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token; defaults to HF_TOKEN/HUGGINGFACE_TOKEN from env.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for inference.",
    )
    return parser.parse_args()


def get_token(cli_token: str | None) -> str | None:
    load_dotenv()
    return cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def to_sorted_session_rows(dataset, selected_sessions: set[str] | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in dataset:
        episode_id = str(row["episode_id"])
        if selected_sessions and episode_id not in selected_sessions:
            continue
        grouped.setdefault(episode_id, []).append(row)

    for episode_id in grouped:
        grouped[episode_id].sort(key=lambda r: int(r["frame_index"]))

    return grouped


def extract_vision_tokens(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if "pixel_values_videos" in inputs:
        pixel_values = inputs["pixel_values_videos"]
    elif "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"]
    else:
        available = ", ".join(sorted(inputs.keys()))
        raise RuntimeError(f"Unsupported processor output keys: {available}")

    if hasattr(model, "get_vision_features"):
        return model.get_vision_features(pixel_values)

    outputs = model(pixel_values=pixel_values)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, tuple) and len(outputs) > 0:
        return outputs[0]
    raise RuntimeError("Could not extract vision features from model outputs.")


def pool_embedding(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 3:
        # [B, N, D] -> [B, D]
        return tokens.mean(dim=1)
    if tokens.ndim == 4:
        # [B, T, N, D] -> [B, D]
        return tokens.mean(dim=(1, 2))
    raise RuntimeError(f"Unexpected vision token shape: {tuple(tokens.shape)}")


def build_payload(samples: list[Sample], meta: dict) -> dict:
    embedding_dim = int(samples[0].embedding.shape[-1])
    embeddings = torch.stack([s.embedding for s in samples], dim=0)

    return {
        "meta": {
            **meta,
            "num_samples": len(samples),
            "embedding_dim": embedding_dim,
        },
        "embeddings": embeddings,
        "action_text": [s.action_text for s in samples],
        "game": [s.game for s in samples],
        "episode_id": [s.episode_id for s in samples],
        "frame_index": torch.tensor([s.frame_index for s in samples], dtype=torch.int32),
        "t_start_ms": torch.tensor([s.t_start_ms for s in samples], dtype=torch.float32),
        "is_idle": torch.tensor([s.is_idle for s in samples], dtype=torch.bool),
        "horizontal_scroll_steps": torch.tensor(
            [s.horizontal_scroll_steps for s in samples], dtype=torch.int16
        ),
        "source_session_relpath": [s.source_session_relpath for s in samples],
    }


def flush_chunk(
    *,
    samples: list[Sample],
    out_path: Path,
    shard_idx: int,
    base_meta: dict,
) -> tuple[Path, int, int]:
    shard_path = out_path.parent / f"{out_path.stem}.part-{shard_idx:05d}.pt"
    payload = build_payload(
        samples,
        {
            **base_meta,
            "storage": "chunked",
            "shard_index": shard_idx,
        },
    )
    torch.save(payload, shard_path)
    return shard_path, int(payload["meta"]["num_samples"]), int(payload["meta"]["embedding_dim"])


def main() -> None:
    args = parse_args()

    token = get_token(args.hf_token)
    if token:
        login(token=token)

    print(f"Loading dataset split from HF: {args.dataset_repo} [{args.split}]")
    dataset = load_dataset(args.dataset_repo, split=args.split, token=token)

    selected_sessions = set(args.sessions) if args.sessions else None
    sessions = to_sorted_session_rows(dataset, selected_sessions)
    if not sessions:
        raise SystemExit("No matching rows found for selected filters.")

    print(f"Loading V-JEPA2 model from HF: {args.model_id}")
    processor = AutoVideoProcessor.from_pretrained(args.model_id, token=token)
    model = AutoModel.from_pretrained(args.model_id, token=token)
    model.eval().to(args.device)

    samples: list[Sample] = []
    dropped_for_idle = 0
    skipped_for_short_context = 0
    total_samples_written = 0
    chunked_shards: list[dict] = []

    chunk_size = max(0, int(args.chunk_size))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_meta = {
        "dataset_repo": args.dataset_repo,
        "split": args.split,
        "model_id": args.model_id,
        "window_size": args.window_size,
        "idle_streak_max": args.idle_streak_max,
    }

    session_items = sorted(sessions.items(), key=lambda kv: kv[0])
    for episode_id, rows in session_items:
        idle_streak = 0
        print(f"Processing {episode_id}: {len(rows)} frames")

        for idx in tqdm(range(len(rows)), desc=f"{episode_id}", leave=False):
            row = rows[idx]
            is_idle = bool(row.get("is_idle", False))
            idle_streak = idle_streak + 1 if is_idle else 0

            if idx < args.window_size - 1:
                skipped_for_short_context += 1
                continue

            # Cutscene/menu-like idle streaks are dropped once the streak is too long.
            if is_idle and idle_streak > args.idle_streak_max:
                dropped_for_idle += 1
                continue

            clip_rows = rows[idx - args.window_size + 1: idx + 1]
            clip_frames = [clip_row["image"] for clip_row in clip_rows]

            with torch.inference_mode():
                inputs = processor(videos=[clip_frames], return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                tokens = extract_vision_tokens(model, inputs)
                embedding = pool_embedding(tokens)[0].detach().cpu().to(torch.float32)

            samples.append(
                Sample(
                    embedding=embedding,
                    action_text=str(row["action_text"]),
                    game=str(row.get("game", "unknown")),
                    episode_id=episode_id,
                    frame_index=int(row["frame_index"]),
                    t_start_ms=float(row["t_start_ms"]),
                    is_idle=is_idle,
                    horizontal_scroll_steps=int(row.get("horizontal_scroll_steps", 0)),
                    source_session_relpath=str(row.get("source_session_relpath", "")),
                )
            )

            if args.max_samples is not None and len(samples) >= args.max_samples:
                break

            if chunk_size > 0 and len(samples) >= chunk_size:
                shard_path, shard_count, shard_embedding_dim = flush_chunk(
                    samples=samples,
                    out_path=out_path,
                    shard_idx=len(chunked_shards),
                    base_meta={
                        **base_meta,
                        "dropped_for_idle": dropped_for_idle,
                        "skipped_for_short_context": skipped_for_short_context,
                    },
                )
                chunked_shards.append(
                    {
                        "path": str(shard_path),
                        "num_samples": shard_count,
                        "embedding_dim": shard_embedding_dim,
                    }
                )
                total_samples_written += shard_count
                samples = []

        if args.max_samples is not None and len(samples) >= args.max_samples:
            break

    if not samples and not chunked_shards:
        raise SystemExit("No samples were extracted. Check filters and input data.")

    if chunk_size > 0:
        if samples:
            shard_path, shard_count, shard_embedding_dim = flush_chunk(
                samples=samples,
                out_path=out_path,
                shard_idx=len(chunked_shards),
                base_meta={
                    **base_meta,
                    "dropped_for_idle": dropped_for_idle,
                    "skipped_for_short_context": skipped_for_short_context,
                },
            )
            chunked_shards.append(
                {
                    "path": str(shard_path),
                    "num_samples": shard_count,
                    "embedding_dim": shard_embedding_dim,
                }
            )
            total_samples_written += shard_count

        index_path = out_path.parent / f"{out_path.stem}.index.json"
        embedding_dim = int(chunked_shards[0]["embedding_dim"])
        index_payload = {
            "meta": {
                **base_meta,
                "storage": "chunked",
                "chunk_size": chunk_size,
                "num_samples": total_samples_written,
                "embedding_dim": embedding_dim,
                "num_shards": len(chunked_shards),
                "dropped_for_idle": dropped_for_idle,
                "skipped_for_short_context": skipped_for_short_context,
            },
            "shards": chunked_shards,
        }
        index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

        print("\nExtraction complete.")
        print(f"  mode: chunked")
        print(f"  index: {index_path}")
        print(f"  shards: {len(chunked_shards)}")
        print(f"  samples: {total_samples_written}")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  dropped_for_idle: {dropped_for_idle}")
        print(f"  skipped_for_short_context: {skipped_for_short_context}")
        return

    payload = build_payload(
        samples,
        {
            **base_meta,
            "storage": "single-file",
            "dropped_for_idle": dropped_for_idle,
            "skipped_for_short_context": skipped_for_short_context,
        },
    )
    torch.save(payload, out_path)

    print("\nExtraction complete.")
    print(f"  mode: single-file")
    print(f"  output: {out_path}")
    print(f"  samples: {payload['meta']['num_samples']}")
    print(f"  embedding_dim: {payload['meta']['embedding_dim']}")
    print(f"  dropped_for_idle: {dropped_for_idle}")
    print(f"  skipped_for_short_context: {skipped_for_short_context}")


if __name__ == "__main__":
    main()
