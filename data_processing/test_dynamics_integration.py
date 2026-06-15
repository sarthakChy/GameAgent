"""Quick integration test for latent dynamics pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.latent_dynamics_model import LatentDynamicsModel, ActionEncoder
from data_processing.probabilistic_losses import vjepa_prob_loss, bjepa_loss, PriorExpert, gaussian_product
from data_processing.dynamics_inference import LatentDynamicsRollout
from data_processing.vjepa2_dataset import ActionTokenizer


def test_action_encoder():
    """Test ActionEncoder module."""
    print("Testing ActionEncoder...")
    encoder = ActionEncoder(vocab_size=1000, embedding_dim=128)

    # Batch of padded action sequences
    action_ids = torch.randint(0, 1000, (32, 50))
    action_length = torch.randint(10, 50, (32,))

    output = encoder(action_ids, action_length)
    assert output.shape == (32, 128), f"Expected (32, 128), got {output.shape}"
    print("✓ ActionEncoder works")


def test_latent_dynamics_model():
    """Test LatentDynamicsModel forward pass."""
    print("Testing LatentDynamicsModel...")
    model = LatentDynamicsModel(
        latent_dim=256,
        action_vocab_size=1000,
        action_embedding_dim=128,
        hidden_dim=512,
        num_layers=2,
    )

    # Batch of data
    z_t = torch.randn(32, 256)
    action_ids = torch.randint(0, 1000, (32, 50))
    action_length = torch.randint(10, 50, (32,))

    mu, logvar = model(z_t, action_ids, action_length)
    assert mu.shape == (32, 256), f"Expected (32, 256), got {mu.shape}"
    assert logvar.shape == (32, 256), f"Expected (32, 256), got {logvar.shape}"

    # Test sampling
    z_sample = model.sample(z_t, action_ids, action_length)
    assert z_sample.shape == (32, 256), f"Expected (32, 256), got {z_sample.shape}"
    print("✓ LatentDynamicsModel works")


def test_vjepa_loss():
    """Test VJEPA loss function."""
    print("Testing VJEPA loss...")
    z_sample = torch.randn(32, 256)
    pred_dist = (torch.randn(32, 256), torch.randn(32, 256))
    target_dist = (z_sample, torch.full((32, 256), -6.0))

    loss = vjepa_prob_loss(z_sample, pred_dist, target_dist, beta=0.01)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"✓ VJEPA loss works (value: {loss.item():.4f})")


def test_bjepa_loss():
    """Test BJEPA loss function."""
    print("Testing BJEPA loss...")
    z_sample = torch.randn(32, 256)
    pred_dist = (torch.randn(32, 256), torch.randn(32, 256))
    target_dist = (z_sample, torch.full((32, 256), -6.0))
    prior_dist = (torch.randn(32, 256), torch.randn(32, 256))

    loss = bjepa_loss(z_sample, pred_dist, target_dist, prior_dist, beta=0.01, prior_weight=0.1)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"✓ BJEPA loss works (value: {loss.item():.4f})")


def test_gaussian_product():
    """Test Gaussian product fusion."""
    print("Testing Gaussian product...")
    dist1 = (torch.randn(32, 256), torch.randn(32, 256))
    dist2 = (torch.randn(32, 256), torch.randn(32, 256))

    mu_prod, logvar_prod = gaussian_product(dist1, dist2)
    assert mu_prod.shape == (32, 256)
    assert logvar_prod.shape == (32, 256)
    print("✓ Gaussian product works")


def test_prior_expert():
    """Test PriorExpert module."""
    print("Testing PriorExpert...")
    prior = PriorExpert(latent_dim=256, hidden_dim=512)

    mu, logvar = prior(batch_size=32, device=torch.device("cpu"))
    assert mu.shape == (32, 256)
    assert logvar.shape == (32, 256)
    print("✓ PriorExpert works")


def test_rollout():
    """Test inference rollout."""
    print("Testing inference rollout...")

    # Create a dummy tokenizer
    token_to_id = {
        "<pad>": 0,
        "<unk>": 1,
        "<action_start>": 2,
        "<action_end>": 3,
        "<empty_group>": 4,
        "<group_1>": 5,
        "<group_2>": 6,
        "<group_3>": 7,
        "<group_4>": 8,
        "<group_5>": 9,
        "<group_6>": 10,
    }
    # Add motion tokens
    for axis in ("dx", "dy", "dz"):
        for bucket in range(-10, 11):
            token_to_id[f"{axis}_bin_{bucket}"] = len(token_to_id)

    tokenizer = ActionTokenizer(token_to_id)

    # Create model
    model = LatentDynamicsModel(
        latent_dim=256,
        action_vocab_size=len(token_to_id),
    )
    model.eval()

    # Create rollout
    rollout = LatentDynamicsRollout(model, tokenizer, torch.device("cpu"))

    z_init = torch.randn(256)
    actions = ["10 5 0 ; ; ; ; ; ;", "0 0 0 ; ; ; ; ; ;"]

    result = rollout.rollout(z_init, actions, deterministic=False)
    assert result["z_trajectory"].shape[0] == 3, "Should have T+1 steps"
    assert result["z_trajectory"].shape[1] == 256
    print("✓ Inference rollout works")


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")

    model = LatentDynamicsModel(latent_dim=128, action_vocab_size=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy batch
    z_t = torch.randn(8, 128)
    action_ids = torch.randint(0, 100, (8, 20))
    action_length = torch.randint(5, 20, (8,))
    z_next = torch.randn(8, 128)

    # Forward pass
    mu_pred, logvar_pred = model(z_t, action_ids, action_length)

    # Compute loss
    mu_target = z_next
    logvar_target = torch.full_like(z_next, -6.0)
    std_target = torch.exp(0.5 * logvar_target)
    z_sample = mu_target + torch.randn_like(std_target) * std_target

    loss = vjepa_prob_loss(z_sample, (mu_pred, logvar_pred), (mu_target, logvar_target), beta=0.01)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"✓ Training step works (loss: {loss.item():.4f})")


def main() -> None:
    print("\n" + "=" * 60)
    print("LATENT DYNAMICS PIPELINE - INTEGRATION TEST")
    print("=" * 60 + "\n")

    test_action_encoder()
    test_latent_dynamics_model()
    test_vjepa_loss()
    test_bjepa_loss()
    test_gaussian_product()
    test_prior_expert()
    test_rollout()
    test_training_step()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe latent dynamics pipeline is ready for training.")
    print("Next steps:")
    print("1. Verify your embedding index and action vocab exist")
    print("2. Run: python -m data_processing.train_latent_dynamics \\")
    print("      --index-path <path> --vocab-path <path>")


if __name__ == "__main__":
    main()
