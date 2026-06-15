"""Loss functions for probabilistic latent dynamics models (VJEPA/BJEPA)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def vjepa_prob_loss(
    z_sample: torch.Tensor,
    pred_dist: tuple[torch.Tensor, torch.Tensor],
    target_dist: tuple[torch.Tensor, torch.Tensor],
    beta: float = 0.01,
) -> torch.Tensor:
    """
    VJEPA probabilistic loss: negative log-likelihood + KL divergence.

    Equation 11 from the VJEPA paper.

    Args:
        z_sample: [B, D] sample from the target distribution (typically just the target z_{t+1})
        pred_dist: tuple of (mu_pred, logvar_pred) from the dynamics model
        target_dist: tuple of (mu_target, logvar_target) representing the target distribution
            (typically a Dirac delta with small variance: mu=z_t+1, logvar≈-6)
        beta: weight for KL divergence term (Lagrange multiplier)

    Returns:
        scalar loss
    """
    mu_pred, logvar_pred = pred_dist
    mu_target, logvar_target = target_dist

    pred_var = torch.exp(logvar_pred).clamp(min=1e-8)
    target_var = torch.exp(logvar_target).clamp(min=1e-8)

    # Negative log-likelihood of the provided target sample under the predictor.
    nll = 0.5 * (
        math.log(2.0 * math.pi)
        + logvar_pred
        + (z_sample - mu_pred) ** 2 / pred_var
    )
    nll = nll.sum(dim=-1).mean()

    # KL divergence: KL(target || pred)
    kl_div = 0.5 * (
        logvar_pred - logvar_target
        + (target_var + (mu_target - mu_pred) ** 2) / pred_var
        - 1.0
    )
    kl_div = kl_div.sum(dim=-1).mean()

    loss = nll + beta * kl_div
    return loss


def gaussian_product(
    dist1: tuple[torch.Tensor, torch.Tensor],
    dist2: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the product of two Gaussians (hard fusion).

    Product of N(mu1, sigma1^2) and N(mu2, sigma2^2) is proportional to
    N(mu_prod, sigma_prod^2) where the precision matrices are summed.

    Args:
        dist1: tuple of (mu1, logvar1)
        dist2: tuple of (mu2, logvar2)

    Returns:
        (mu_prod, logvar_prod)
    """
    mu1, logvar1 = dist1
    mu2, logvar2 = dist2

    # Precision (inverse covariance)
    prec1 = torch.exp(-logvar1)  # 1/sigma^2
    prec2 = torch.exp(-logvar2)

    # Product of precisions and weighted means
    prec_prod = prec1 + prec2
    mu_prod = (mu1 * prec1 + mu2 * prec2) / (prec_prod + 1e-8)

    # Covariance and logvar of product
    var_prod = 1.0 / (prec_prod + 1e-8)
    logvar_prod = torch.log(var_prod.clamp(min=1e-8))

    return mu_prod, logvar_prod


def product_of_experts(
    dist1: tuple[torch.Tensor, torch.Tensor],
    dist2: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias for the analytic Gaussian fusion used by BJEPA inference."""
    return gaussian_product(dist1, dist2)


def bjepa_loss(
    z_sample: torch.Tensor,
    pred_dist: tuple[torch.Tensor, torch.Tensor],
    target_dist: tuple[torch.Tensor, torch.Tensor],
    prior_dist: tuple[torch.Tensor, torch.Tensor],
    beta: float = 0.01,
    prior_weight: float = 0.1,
) -> torch.Tensor:
    """
    BJEPA loss: combines dynamics loss with a learnable prior via soft fusion KL.

    Equation 33 from the BJEPA paper.

    Args:
        z_sample: [B, D] sample from target (or could sample from fused dist)
        pred_dist: tuple of (mu_pred, logvar_pred) from dynamics model
        target_dist: tuple of (mu_target, logvar_target) representing ground truth
        prior_dist: tuple of (mu_prior, logvar_prior) from a learnable prior expert
        beta: weight for KL divergence in main loss
        prior_weight: weight for KL divergence between fused and prior

    Returns:
        scalar loss
    """
    # Main dynamics loss (VJEPA)
    loss_vjepa = vjepa_prob_loss(z_sample, pred_dist, target_dist, beta=beta)

    # Soft fusion: fuse dynamics prediction with prior
    fused_dist = product_of_experts(pred_dist, prior_dist)

    # KL regularizer: encourage fused distribution to stay close to prior
    # (prevents prior from being ignored by dynamics alone)
    mu_fused, logvar_fused = fused_dist
    mu_prior, logvar_prior = prior_dist

    prior_var = torch.exp(logvar_prior).clamp(min=1e-8)
    fused_var = torch.exp(logvar_fused).clamp(min=1e-8)

    kl_to_prior = 0.5 * (
        logvar_prior - logvar_fused
        + (fused_var + (mu_fused - mu_prior) ** 2) / prior_var
        - 1.0
    )
    kl_to_prior = kl_to_prior.sum(dim=-1).mean()

    loss = loss_vjepa + prior_weight * kl_to_prior
    return loss


class PriorExpert(torch.nn.Module):
    """Learnable static (or goal-conditioned) prior expert for BJEPA."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        hidden_dim = hidden_dim or (latent_dim * 2)

        # Simple MLP to output (mu, logvar)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),  # 1 input: task_id or constant
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )

        self.mu_head = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = torch.nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.mu_head, self.logvar_head]:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_size: number of samples
            device: torch device

        Returns:
            (mu, logvar) of shape [batch_size, latent_dim]
        """
        # Input: learnable parameter (or task indicator)
        task_indicator = torch.ones((batch_size, 1), device=device)

        hidden = self.net(task_indicator)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)

        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar
