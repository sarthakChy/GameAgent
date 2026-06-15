"""Latent dynamics model for probabilistic action-conditioned state prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """Encodes variable-length action token sequences into a fixed-size embedding."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        hidden_dim = hidden_dim or embedding_dim

        # Token embedding table
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Small transformer or RNN to aggregate action sequence
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=max(1, min(8, embedding_dim // 64)),  # Auto-scale nhead
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Output projection to fixed size
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        action_ids: torch.Tensor,
        action_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            action_ids: [B, T] long tensor of action token ids (padded)
            action_length: [B] tensor of actual lengths (for masking), or None to use all tokens

        Returns:
            embeddings: [B, D] action embeddings
        """
        bsz, seq_len = action_ids.shape

        # Embed tokens
        x = self.token_embedding(action_ids)  # [B, T, D]

        # Create key padding mask if lengths provided
        key_padding_mask = None
        if action_length is not None:
            key_padding_mask = torch.arange(
                seq_len, device=action_ids.device
            ).unsqueeze(0) >= action_length.unsqueeze(1)

        # Encode with transformer
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

        # Pool: mean over sequence (respecting masking if provided)
        if action_length is not None:
            mask = ~key_padding_mask  # [B, T]
            masked = encoded * mask.unsqueeze(-1).float()  # [B, T, D]
            pooled = masked.sum(dim=1) / action_length.unsqueeze(-1).clamp(min=1).float()  # [B, D]
        else:
            pooled = encoded.mean(dim=1)  # [B, D]

        # Output projection
        output = self.output_proj(pooled)  # [B, D]
        return output


class LatentDynamicsModel(nn.Module):
    """
    Probabilistic latent dynamics model.

    Maps (z_t, action) → (mu, logvar) for z_{t+1}.
    Outputs a Gaussian distribution over the next latent state.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_vocab_size: int,
        action_embedding_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        action_embedding_dim = action_embedding_dim or latent_dim
        hidden_dim = hidden_dim or (latent_dim * 2)

        # Action encoder
        self.action_encoder = ActionEncoder(
            vocab_size=action_vocab_size,
            embedding_dim=action_embedding_dim,
            hidden_dim=action_embedding_dim * 2,
        )

        # Projection layers to combine z_t and action
        self.state_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_proj = nn.Linear(action_embedding_dim, hidden_dim)

        # MLP layers to predict (mu, logvar)
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))

        self.mlp = nn.Sequential(*layers)

        # Output heads
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.state_proj, self.action_proj, self.mu_head, self.logvar_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        action_ids: torch.Tensor,
        action_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: [B, D_latent] current latent state embeddings
            action_ids: [B, T] action token ids (padded)
            action_length: [B] actual action lengths for masking, or None

        Returns:
            mu: [B, D_latent] predicted mean
            logvar: [B, D_latent] predicted log-variance (clamped for stability)
        """
        # Encode state and action
        state_feat = self.state_proj(z_t)  # [B, H]
        action_feat = self.action_encoder(action_ids, action_length)  # [B, D_act]
        action_feat = self.action_proj(action_feat)  # [B, H]

        # Combine features
        combined = state_feat + action_feat  # [B, H]

        # MLP transformation
        hidden = self.mlp(combined)  # [B, H]

        # Output heads
        mu = self.mu_head(hidden)  # [B, D_latent]
        logvar = self.logvar_head(hidden)  # [B, D_latent]

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar

    def sample(
        self,
        z_t: torch.Tensor,
        action_ids: torch.Tensor,
        action_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the predicted distribution."""
        mu, logvar = self.forward(z_t, action_ids, action_length)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_next_sample = mu + eps * std

        return z_next_sample
