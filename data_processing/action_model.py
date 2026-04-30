from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatefulActionDecoder(nn.Module):
    """A compact autoregressive decoder conditioned on V-JEPA embeddings.

    Training usage:
    - input_ids: teacher-forcing tokens (usually target shifted right)
    - target_ids: next-token labels aligned with input_ids positions

    Inference usage:
    - call generate(...) with start/end token ids
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        vision_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_id: int = 0,
        temporal_hidden_dim: int = 0,
        temporal_num_layers: int = 1,
        temporal_dropout: float = 0.0,
        inverse_dynamics_classes: int = 0,
        inverse_dynamics_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.vocab_size = vocab_size
        self.vision_dim = vision_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.temporal_num_layers = int(temporal_num_layers)
        self.temporal_dropout = float(temporal_dropout)
        self.inverse_dynamics_classes = int(inverse_dynamics_classes)
        self.inverse_dynamics_hidden_dim = int(inverse_dynamics_hidden_dim)

        if self.temporal_hidden_dim > 0:
            gru_dropout = self.temporal_dropout if self.temporal_num_layers > 1 else 0.0
            self.temporal_encoder = nn.GRU(
                input_size=vision_dim,
                hidden_size=self.temporal_hidden_dim,
                num_layers=self.temporal_num_layers,
                batch_first=True,
                dropout=gru_dropout,
            )
            self.vision_proj = nn.Linear(self.temporal_hidden_dim, d_model)
        else:
            self.temporal_encoder = None
            self.vision_proj = nn.Linear(vision_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len + 1, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        if self.inverse_dynamics_classes > 0:
            inverse_input_dim = vision_dim * 4
            self.inverse_dynamics_proj = nn.Sequential(
                nn.Linear(inverse_input_dim, self.inverse_dynamics_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.inverse_dynamics_hidden_dim),
                nn.Dropout(dropout),
            )
            self.inverse_dynamics_head = nn.Linear(self.inverse_dynamics_hidden_dim, self.inverse_dynamics_classes)
        else:
            self.inverse_dynamics_proj = None
            self.inverse_dynamics_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.vision_proj.weight)
        nn.init.zeros_(self.vision_proj.bias)
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)
        if self.inverse_dynamics_head is not None:
            for module in self.inverse_dynamics_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(self.inverse_dynamics_head.weight)
            nn.init.zeros_(self.inverse_dynamics_head.bias)

    def init_temporal_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.temporal_encoder is None:
            return None
        return torch.zeros(self.temporal_num_layers, batch_size, self.temporal_hidden_dim, device=device)

    def _encode_context(
        self,
        vision_embedding: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if vision_embedding.ndim == 2:
            vision_sequence = vision_embedding.unsqueeze(1)
        elif vision_embedding.ndim == 3:
            vision_sequence = vision_embedding
        else:
            raise ValueError("vision_embedding must have shape [B, Dv] or [B, S, Dv]")

        if vision_sequence.size(-1) != self.vision_dim:
            raise ValueError(f"Expected vision embedding dim {self.vision_dim}, got {vision_sequence.size(-1)}")

        if self.temporal_encoder is None:
            pooled = vision_sequence[:, -1, :] if vision_sequence.size(1) > 0 else vision_sequence.squeeze(1)
            return self.vision_proj(pooled), hidden_state

        temporal_features, new_hidden = self.temporal_encoder(vision_sequence, hidden_state)
        return self.vision_proj(temporal_features[:, -1, :]), new_hidden

    def _decode_from_context(self, context: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")

        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"input_ids length {seq_len} exceeds max_seq_len={self.max_seq_len}; "
                "increase max_seq_len or truncate inputs"
            )

        device = input_ids.device

        ctx = context.unsqueeze(1)  # [B, 1, d_model]
        tok = self.token_embedding(input_ids)  # [B, T, d_model]
        x = torch.cat([ctx, tok], dim=1)  # [B, T+1, d_model]

        positions = torch.arange(seq_len + 1, device=device).unsqueeze(0).expand(bsz, -1)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._causal_mask(seq_len + 1, device=device)

        pad_mask = input_ids.eq(self.pad_id)
        key_padding_mask = torch.cat(
            [torch.zeros((bsz, 1), device=device, dtype=torch.bool), pad_mask],
            dim=1,
        )

        h = self.decoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.lm_head(h[:, 1:, :])

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        # True means masked in PyTorch Transformer attention masks.
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        vision_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
        return_hidden_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Compute token logits for each position in input_ids.

        Args:
            vision_embedding: [B, Dv]
            input_ids: [B, T]

        Returns:
            logits: [B, T, V]
        """
        context, new_hidden = self._encode_context(vision_embedding, hidden_state=hidden_state)
        logits = self._decode_from_context(context, input_ids)
        if return_hidden_state:
            return logits, new_hidden
        return logits

    def compute_loss(
        self,
        vision_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        ignore_index: int = -100,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy loss for autoregressive training.

        Typical setup:
        - input_ids = target_ids shifted right with BOS/start token.
        - target_ids contains true next-token ids.
        """
        logits = self.forward(vision_embedding, input_ids)
        if target_ids.shape != input_ids.shape:
            raise ValueError("target_ids must have same shape as input_ids")

        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view_as(target_ids)

        valid_mask = target_ids.ne(ignore_index)
        if sample_weights is not None:
            if sample_weights.ndim != 1 or sample_weights.size(0) != target_ids.size(0):
                raise ValueError("sample_weights must have shape [B]")
            weight_mask = sample_weights.to(per_token_loss.device).unsqueeze(1)
            weighted_loss = per_token_loss * valid_mask.to(per_token_loss.dtype) * weight_mask
            denom = (valid_mask.to(per_token_loss.dtype) * weight_mask).sum().clamp_min(1.0)
            return weighted_loss.sum() / denom

        denom = valid_mask.sum().clamp_min(1).to(per_token_loss.dtype)
        return (per_token_loss * valid_mask.to(per_token_loss.dtype)).sum() / denom

    def inverse_dynamics_logits(self, current_embedding: torch.Tensor, next_embedding: torch.Tensor) -> torch.Tensor:
        if self.inverse_dynamics_head is None or self.inverse_dynamics_proj is None:
            raise RuntimeError("inverse dynamics head is disabled for this decoder")
        if current_embedding.ndim != 2 or next_embedding.ndim != 2:
            raise ValueError("inverse_dynamics_logits expects [B, D] tensors")
        if current_embedding.shape != next_embedding.shape:
            raise ValueError("current_embedding and next_embedding must have the same shape")

        pair_features = torch.cat(
            [
                current_embedding,
                next_embedding,
                next_embedding - current_embedding,
                current_embedding * next_embedding,
            ],
            dim=-1,
        )
        hidden = self.inverse_dynamics_proj(pair_features)
        return self.inverse_dynamics_head(hidden)

    def compute_inverse_dynamics_loss(
        self,
        current_embedding: torch.Tensor,
        next_embedding: torch.Tensor,
        target_action_ids: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        logits = self.inverse_dynamics_logits(current_embedding, next_embedding)
        if target_action_ids.ndim != 1:
            raise ValueError("target_action_ids must have shape [B]")
        return F.cross_entropy(logits, target_action_ids, ignore_index=ignore_index)

    @torch.no_grad()
    def generate(
        self,
        vision_embedding: torch.Tensor,
        *,
        start_id: int,
        end_id: int,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        hidden_state: torch.Tensor | None = None,
        return_temporal_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Autoregressively sample token ids.

        Args:
            vision_embedding: [B, Dv]
            start_id: sequence start token id
            end_id: stop token id

        Returns:
            generated token ids including start token: [B, <= max_new_tokens+1]
        """
        bsz = vision_embedding.size(0)
        device = vision_embedding.device
        context, new_hidden = self._encode_context(vision_embedding, hidden_state=hidden_state)
        generated = torch.full((bsz, 1), start_id, dtype=torch.long, device=device)
        finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if generated.size(1) > self.max_seq_len:
                break

            logits = self._decode_from_context(context, generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, k=k, dim=-1)
                cutoff = values[:, -1].unsqueeze(1)
                next_logits = torch.where(
                    next_logits < cutoff,
                    torch.full_like(next_logits, -float("inf")),
                    next_logits,
                )

            probs = torch.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]

            next_ids = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_ids, end_id),
                next_ids,
            )

            generated = torch.cat([generated, next_ids], dim=1)
            finished = finished | next_ids.squeeze(1).eq(end_id)
            if torch.all(finished):
                break

        if return_temporal_state:
            return generated, new_hidden
        return generated


MiniTransformerActionDecoder = StatefulActionDecoder


def make_teacher_forcing_batch(
    target_ids: torch.Tensor,
    *,
    start_id: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create autoregressive (input_ids, labels) from target token ids.

    target_ids should already include action content and can include pad tokens.
    Returns tensors shaped [B, T] suitable for compute_loss(...).
    """
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [B, T]")

    bsz, seq_len = target_ids.shape
    if seq_len < 1:
        raise ValueError("target_ids must have T >= 1")

    input_ids = torch.full((bsz, seq_len), fill_value=pad_id, dtype=torch.long, device=target_ids.device)
    input_ids[:, 0] = start_id
    if seq_len > 1:
        input_ids[:, 1:] = target_ids[:, :-1]

    labels = target_ids.clone()
    return input_ids, labels
