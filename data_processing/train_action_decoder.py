from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from data_processing.action_model import MiniTransformerActionDecoder, make_teacher_forcing_batch
from data_processing.vjepa2_dataset import (
    ActionTokenizer,
    ShardedEmbeddingActionDataset,
    make_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mini transformer action decoder on V-JEPA embeddings.")

    parser.add_argument("--index-path", required=True, help="Path to chunked index JSON file.")
    parser.add_argument(
        "--val-index-path",
        default=None,
        help="Optional separate validation index JSON. If omitted, --val-ratio split is used.",
    )
    parser.add_argument(
        "--vocab-path",
        default="data_processing/outputs/action_vocab.json",
        help="Path to tokenizer vocab JSON (loaded if exists, otherwise created).",
    )
    parser.add_argument(
        "--build-vocab-min-freq",
        type=int,
        default=1,
        help="Minimum token frequency when building vocab from index.",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision training.")

    parser.add_argument(
        "--output-dir",
        default="data_processing/outputs/action_decoder_runs",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_datasets(args: argparse.Namespace, tokenizer: ActionTokenizer) -> tuple[Subset | ShardedEmbeddingActionDataset, Subset | ShardedEmbeddingActionDataset]:
    if args.val_index_path:
        train_ds = ShardedEmbeddingActionDataset(
            index_path=args.index_path,
            tokenizer=tokenizer,
            max_action_tokens=args.max_seq_len,
            pad_to_max_action_tokens=False,
            shard_cache_size=2,
        )
        val_ds = ShardedEmbeddingActionDataset(
            index_path=args.val_index_path,
            tokenizer=tokenizer,
            max_action_tokens=args.max_seq_len,
            pad_to_max_action_tokens=False,
            shard_cache_size=2,
        )
        return train_ds, val_ds

    full_ds = ShardedEmbeddingActionDataset(
        index_path=args.index_path,
        tokenizer=tokenizer,
        max_action_tokens=args.max_seq_len,
        pad_to_max_action_tokens=False,
        shard_cache_size=2,
    )

    total = len(full_ds)
    val_len = max(1, int(total * args.val_ratio))
    train_len = total - val_len
    if train_len < 1:
        raise ValueError("Dataset too small for chosen val split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, lengths=[train_len, val_len], generator=generator)
    return train_ds, val_ds


def train_one_epoch(
    *,
    model: MiniTransformerActionDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    start_id: int,
    pad_id: int,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    total_steps = 0

    for batch in loader:
        vision = batch["embedding"].to(device)
        target_ids = batch["action_ids"].to(device)

        input_ids, labels = make_teacher_forcing_batch(
            target_ids,
            start_id=start_id,
            pad_id=pad_id,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = model.compute_loss(
                vision_embedding=vision,
                input_ids=input_ids,
                target_ids=labels,
                ignore_index=pad_id,
            )

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Step LR only when optimizer update was not skipped by AMP overflow.
        if scaler.get_scale() >= scale_before:
            scheduler.step()

        running_loss += float(loss.item())
        total_steps += 1

    return running_loss / max(1, total_steps)


@torch.no_grad()
def eval_one_epoch(
    *,
    model: MiniTransformerActionDecoder,
    loader: DataLoader,
    device: torch.device,
    start_id: int,
    pad_id: int,
    use_amp: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    total_steps = 0

    for batch in loader:
        vision = batch["embedding"].to(device)
        target_ids = batch["action_ids"].to(device)
        input_ids, labels = make_teacher_forcing_batch(
            target_ids,
            start_id=start_id,
            pad_id=pad_id,
        )

        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = model.compute_loss(
                vision_embedding=vision,
                input_ids=input_ids,
                target_ids=labels,
                ignore_index=pad_id,
            )

        running_loss += float(loss.item())
        total_steps += 1

    return running_loss / max(1, total_steps)


def save_checkpoint(
    *,
    path: Path,
    model: MiniTransformerActionDecoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        tokenizer = ActionTokenizer.load(vocab_path)
        print(f"Loaded existing vocab: {vocab_path}")
    else:
        tokenizer = ActionTokenizer.build_from_index(args.index_path, min_freq=args.build_vocab_min_freq)
        tokenizer.save(vocab_path)
        print(f"Built vocab and saved: {vocab_path}")

    train_ds, val_ds = make_datasets(args, tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
    )

    sample_batch = next(iter(train_loader))
    vision_dim = int(sample_batch["embedding"].shape[-1])

    device = torch.device(args.device)
    model = MiniTransformerActionDecoder(
        vocab_size=len(tokenizer.token_to_id),
        vision_dim=vision_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    use_amp = args.use_amp and (args.device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_id = tokenizer.token_to_id[ActionTokenizer.ACTION_START]
    log_path = run_dir / "metrics.jsonl"

    best_val = math.inf

    print("Training config:")
    print(f"  run_dir: {run_dir}")
    print(f"  train_samples: {len(train_ds)}")
    print(f"  val_samples: {len(val_ds)}")
    print(f"  vocab_size: {len(tokenizer.token_to_id)}")
    print(f"  vision_dim: {vision_dim}")
    print(f"  device: {device}")
    print(f"  amp: {use_amp}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            start_id=start_id,
            pad_id=tokenizer.pad_id,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
        )

        val_loss = eval_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            start_id=start_id,
            pad_id=tokenizer.pad_id,
            use_amp=use_amp,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr_now:.6e}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                path=run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
            )

        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_checkpoint(
                path=run_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
            )

    print("Training complete.")
    print(f"  best_val_loss: {best_val:.4f}")
    print(f"  run_dir: {run_dir}")
    print(f"  best_checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
