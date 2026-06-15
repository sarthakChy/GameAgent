"""Evaluation script for latent dynamics model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.consecutive_pair_dataset import (
    ConsecutivePairDataset,
    make_consecutive_pair_collate,
)
from data_processing.latent_dynamics_model import LatentDynamicsModel
from data_processing.vjepa2_dataset import (
    ActionTokenizer,
    ShardedEmbeddingActionDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate latent dynamics model on test set."
    )

    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--index-path", required=True, help="Path to test embedding index JSON.")
    parser.add_argument(
        "--vocab-path",
        default="data_processing/outputs/action_vocab.json",
        help="Path to action tokenizer vocab.",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--output-dir",
        default="data_processing/outputs/eval_results",
        help="Directory for evaluation results.",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[LatentDynamicsModel, dict]:
    """Load model and training args from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    args_dict = ckpt["args"]
    model_config = ckpt.get("model_config")
    if model_config is None:
        model_state = ckpt["model_state"]
        model_config = {
            "latent_dim": int(model_state["mu_head.weight"].shape[0]),
            "action_vocab_size": int(model_state["action_encoder.token_embedding.weight"].shape[0]),
            "action_embedding_dim": int(model_state["action_encoder.token_embedding.weight"].shape[1]),
            "hidden_dim": int(model_state["state_proj.weight"].shape[0]),
            "num_layers": sum(
                1
                for key, value in model_state.items()
                if key.startswith("mlp.") and key.endswith(".weight") and value.ndim == 2
            ),
            "dropout": float(args_dict.get("dropout", 0.1)),
        }

    model = LatentDynamicsModel(
        latent_dim=int(model_config["latent_dim"]),
        action_vocab_size=int(model_config["action_vocab_size"]),
        action_embedding_dim=int(model_config["action_embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config.get("dropout", args_dict.get("dropout", 0.1))),
    )

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, args_dict


def evaluate(
    *,
    model: LatentDynamicsModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model and compute metrics."""
    metrics = {
        "mse_list": [],
        "mae_list": [],
        "cosine_sim_list": [],
        "pred_norm_list": [],
        "target_norm_list": [],
    }

    with torch.no_grad():
        for batch in loader:
            z_t = batch["embedding"].to(device)
            action_ids = batch["action_ids"].to(device)
            action_length = batch["action_length"].to(device)
            z_next = batch["next_embedding"].to(device)

            # Forward pass
            mu_pred, logvar_pred = model(z_t, action_ids, action_length)

            # Compute metrics
            mse = F.mse_loss(mu_pred, z_next, reduction="none").mean(dim=1)  # [B]
            mae = F.l1_loss(mu_pred, z_next, reduction="none").mean(dim=1)  # [B]

            # Cosine similarity
            cosine_sim = F.cosine_similarity(mu_pred, z_next, dim=1)  # [B]

            # Norms
            pred_norm = torch.norm(mu_pred, dim=1)  # [B]
            target_norm = torch.norm(z_next, dim=1)  # [B]

            metrics["mse_list"].append(mse.cpu())
            metrics["mae_list"].append(mae.cpu())
            metrics["cosine_sim_list"].append(cosine_sim.cpu())
            metrics["pred_norm_list"].append(pred_norm.cpu())
            metrics["target_norm_list"].append(target_norm.cpu())

    # Aggregate
    mse_all = torch.cat(metrics["mse_list"])
    mae_all = torch.cat(metrics["mae_list"])
    cosine_sim_all = torch.cat(metrics["cosine_sim_list"])
    pred_norm_all = torch.cat(metrics["pred_norm_list"])
    target_norm_all = torch.cat(metrics["target_norm_list"])

    results = {
        "num_samples": len(mse_all),
        "mse": {
            "mean": float(mse_all.mean()),
            "std": float(mse_all.std()),
            "min": float(mse_all.min()),
            "max": float(mse_all.max()),
        },
        "mae": {
            "mean": float(mae_all.mean()),
            "std": float(mae_all.std()),
            "min": float(mae_all.min()),
            "max": float(mae_all.max()),
        },
        "cosine_similarity": {
            "mean": float(cosine_sim_all.mean()),
            "std": float(cosine_sim_all.std()),
            "min": float(cosine_sim_all.min()),
            "max": float(cosine_sim_all.max()),
        },
        "pred_norm": {
            "mean": float(pred_norm_all.mean()),
            "std": float(pred_norm_all.std()),
        },
        "target_norm": {
            "mean": float(target_norm_all.mean()),
            "std": float(target_norm_all.std()),
        },
    }

    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, train_args = load_checkpoint(checkpoint_path, device)

    # Load tokenizer
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    tokenizer = ActionTokenizer.load(vocab_path)

    # Create test dataset
    print("Loading test dataset...")
    base_ds = ShardedEmbeddingActionDataset(
        index_path=args.index_path,
        tokenizer=tokenizer,
        max_action_tokens=train_args.get("max_seq_len", 128),
        pad_to_max_action_tokens=False,
        shard_cache_size=2,
    )
    pair_ds = ConsecutivePairDataset(base_ds)
    print(f"Test pairs: {len(pair_ds)}")

    # Create dataloader
    collate_fn = make_consecutive_pair_collate(tokenizer.pad_id)
    test_loader = DataLoader(
        pair_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
    )

    # Evaluate
    print("Evaluating model...")
    results = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Num samples: {results['num_samples']}")
    print("\nMSE (lower is better):")
    print(f"  Mean: {results['mse']['mean']:.6f} ± {results['mse']['std']:.6f}")
    print(f"  Range: [{results['mse']['min']:.6f}, {results['mse']['max']:.6f}]")
    print("\nMAE (lower is better):")
    print(f"  Mean: {results['mae']['mean']:.6f} ± {results['mae']['std']:.6f}")
    print(f"  Range: [{results['mae']['min']:.6f}, {results['mae']['max']:.6f}]")
    print("\nCosine Similarity (higher is better, 1.0 = perfect):")
    print(f"  Mean: {results['cosine_similarity']['mean']:.6f} ± {results['cosine_similarity']['std']:.6f}")
    print(f"  Range: [{results['cosine_similarity']['min']:.6f}, {results['cosine_similarity']['max']:.6f}]")
    print("\nEmbedding norms:")
    print(f"  Pred:   {results['pred_norm']['mean']:.6f} ± {results['pred_norm']['std']:.6f}")
    print(f"  Target: {results['target_norm']['mean']:.6f} ± {results['target_norm']['std']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
