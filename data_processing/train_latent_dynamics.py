"""Train latent dynamics model using consecutive frame pairs."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from data_processing.consecutive_pair_dataset import (
    ConsecutivePairDataset,
    make_consecutive_pair_collate,
)
from data_processing.latent_dynamics_model import LatentDynamicsModel
from data_processing.probabilistic_losses import vjepa_prob_loss, bjepa_loss, PriorExpert
from data_processing.vjepa2_dataset import (
    ActionTokenizer,
    ShardedEmbeddingActionDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train latent dynamics model on consecutive frame pairs."
    )

    parser.add_argument("--index-path", required=True, help="Path to embedding index JSON file.")
    parser.add_argument(
        "--val-index-path",
        default=None,
        help="Optional separate validation index JSON.",
    )
    parser.add_argument(
        "--vocab-path",
        default="data_processing/outputs/action_vocab.json",
        help="Path to action tokenizer vocab JSON.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Optional expected embedding dimension; if set, must match the data.",
    )
    parser.add_argument("--action-embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=128)

    # Loss hyperparameters
    parser.add_argument("--beta", type=float, default=0.01, help="KL weight in VJEPA loss.")
    parser.add_argument("--use-bjepa", action="store_true", help="Use BJEPA loss with learnable prior.")
    parser.add_argument("--prior-weight", type=float, default=0.1, help="Weight for BJEPA prior KL.")

    # Device and precision
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision.")

    # Output
    parser.add_argument(
        "--output-dir",
        default="data_processing/outputs/dynamics_runs",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_datasets(
    args: argparse.Namespace, tokenizer: ActionTokenizer
) -> tuple[ConsecutivePairDataset | Subset, ConsecutivePairDataset | Subset]:
    """Create train and validation ConsecutivePairDatasets."""
    base_ds = ShardedEmbeddingActionDataset(
        index_path=args.index_path,
        tokenizer=tokenizer,
        max_action_tokens=args.max_seq_len,
        pad_to_max_action_tokens=False,
        shard_cache_size=2,
    )

    pair_ds = ConsecutivePairDataset(base_ds)
    if len(pair_ds) < 1:
        raise ValueError("No consecutive pairs could be built from the training index.")

    if args.val_index_path:
        base_val_ds = ShardedEmbeddingActionDataset(
            index_path=args.val_index_path,
            tokenizer=tokenizer,
            max_action_tokens=args.max_seq_len,
            pad_to_max_action_tokens=False,
            shard_cache_size=2,
        )
        val_pair_ds = ConsecutivePairDataset(base_val_ds)
        if len(val_pair_ds) < 1:
            raise ValueError("No consecutive pairs could be built from the validation index.")
        return pair_ds, val_pair_ds

    total = len(pair_ds)
    val_len = max(1, int(total * args.val_ratio))
    train_len = total - val_len
    if train_len < 1:
        raise ValueError("Dataset too small for chosen val split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(pair_ds, lengths=[train_len, val_len], generator=generator)
    return train_ds, val_ds


def train_one_epoch(
    *,
    model: LatentDynamicsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    pad_id: int,
    use_amp: bool,
    grad_clip: float,
    beta: float,
    prior_expert: PriorExpert | None = None,
    prior_weight: float = 0.1,
    use_bjepa: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    if prior_expert is not None:
        prior_expert.train()

    running_loss = 0.0
    running_nll = 0.0
    running_kl = 0.0
    total_steps = 0

    for batch in loader:
        z_t = batch["embedding"].to(device)
        action_ids = batch["action_ids"].to(device)
        action_length = batch["action_length"].to(device)
        z_next = batch["next_embedding"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            # Forward pass
            mu_pred, logvar_pred = model(z_t, action_ids, action_length)

            # Target distribution: Dirac delta (actual next embedding) with tiny variance
            mu_target = z_next
            logvar_target = torch.full_like(z_next, -6.0)  # small variance

            # Sample from target for loss
            std_target = torch.exp(0.5 * logvar_target)
            z_sample = mu_target + torch.randn_like(std_target) * std_target

            if use_bjepa and prior_expert is not None:
                # BJEPA: include learnable prior
                mu_prior, logvar_prior = prior_expert(z_t.shape[0], device)
                loss = bjepa_loss(
                    z_sample,
                    (mu_pred, logvar_pred),
                    (mu_target, logvar_target),
                    (mu_prior, logvar_prior),
                    beta=beta,
                    prior_weight=prior_weight,
                )
            else:
                # Simple VJEPA
                loss = vjepa_prob_loss(
                    z_sample,
                    (mu_pred, logvar_pred),
                    (mu_target, logvar_target),
                    beta=beta,
                )

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        if scaler.get_scale() >= scale_before:
            scheduler.step()

        running_loss += float(loss.item())
        total_steps += 1

    return {
        "loss": running_loss / max(1, total_steps),
    }


@torch.no_grad()
def eval_one_epoch(
    *,
    model: LatentDynamicsModel,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    use_amp: bool,
    beta: float,
    prior_expert: PriorExpert | None = None,
    prior_weight: float = 0.1,
    use_bjepa: bool = False,
) -> dict:
    """Evaluate for one epoch."""
    model.eval()
    if prior_expert is not None:
        prior_expert.eval()

    running_loss = 0.0
    total_steps = 0

    for batch in loader:
        z_t = batch["embedding"].to(device)
        action_ids = batch["action_ids"].to(device)
        action_length = batch["action_length"].to(device)
        z_next = batch["next_embedding"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            mu_pred, logvar_pred = model(z_t, action_ids, action_length)

            mu_target = z_next
            logvar_target = torch.full_like(z_next, -6.0)

            std_target = torch.exp(0.5 * logvar_target)
            z_sample = mu_target + torch.randn_like(std_target) * std_target

            if use_bjepa and prior_expert is not None:
                mu_prior, logvar_prior = prior_expert(z_t.shape[0], device)
                loss = bjepa_loss(
                    z_sample,
                    (mu_pred, logvar_pred),
                    (mu_target, logvar_target),
                    (mu_prior, logvar_prior),
                    beta=beta,
                    prior_weight=prior_weight,
                )
            else:
                loss = vjepa_prob_loss(
                    z_sample,
                    (mu_pred, logvar_pred),
                    (mu_target, logvar_target),
                    beta=beta,
                )

        running_loss += float(loss.item())
        total_steps += 1

    return {
        "loss": running_loss / max(1, total_steps),
    }


def save_checkpoint(
    *,
    path: Path,
    model: LatentDynamicsModel,
    model_config: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
    prior_expert: PriorExpert | None = None,
) -> None:
    """Save checkpoint."""
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": model_config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args),
    }
    if prior_expert is not None:
        ckpt["prior_expert_state"] = prior_expert.state_dict()

    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found: {vocab_path}")
    tokenizer = ActionTokenizer.load(vocab_path)
    print(f"Loaded tokenizer from {vocab_path} (vocab size: {len(tokenizer.token_to_id)})")

    # Create datasets
    print("Building consecutive pair dataset...")
    train_ds, val_ds = make_datasets(args, tokenizer)
    print(f"Train pairs: {len(train_ds)}, Val pairs: {len(val_ds)}")

    # Create dataloaders
    collate_fn = make_consecutive_pair_collate(tokenizer.pad_id)
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

    # Infer dimensions
    sample_batch = next(iter(train_loader))
    latent_dim = int(sample_batch["embedding"].shape[-1])
    if args.latent_dim is not None and args.latent_dim != latent_dim:
        raise ValueError(
            f"--latent-dim={args.latent_dim} does not match the embedding dimension "
            f"found in data ({latent_dim})."
        )
    vocab_size = len(tokenizer.token_to_id)

    print(f"Latent dim: {latent_dim}, Vocab size: {vocab_size}")

    # Create model
    device = torch.device(args.device)
    model = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_vocab_size=vocab_size,
        action_embedding_dim=args.action_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(
        f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    model_config = {
        "latent_dim": latent_dim,
        "action_vocab_size": vocab_size,
        "action_embedding_dim": args.action_embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
    }

    # Optional prior expert for BJEPA
    prior_expert = None
    if args.use_bjepa:
        prior_expert = PriorExpert(
            latent_dim=latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        print(
            f"Prior expert params: {sum(p.numel() for p in prior_expert.parameters() if p.requires_grad):,}"
        )

    # Optimizer and scheduler
    params = list(model.parameters())
    if prior_expert is not None:
        params.extend(prior_expert.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            pad_id=tokenizer.pad_id,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            beta=args.beta,
            prior_expert=prior_expert,
            prior_weight=args.prior_weight,
            use_bjepa=args.use_bjepa,
        )

        val_metrics = eval_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            pad_id=tokenizer.pad_id,
            use_amp=args.use_amp,
            beta=args.beta,
            prior_expert=prior_expert,
            prior_weight=args.prior_weight,
            use_bjepa=args.use_bjepa,
        )

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f}"
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
                prior_expert=prior_expert,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = run_dir / "best_model.pt"
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
                prior_expert=prior_expert,
            )
            print(f"New best model! Saved: {best_ckpt_path}")

    print(f"\nTraining completed! Run dir: {run_dir}")


if __name__ == "__main__":
    main()
