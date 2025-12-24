"""
Loss functions for CRAFT: Contrastive Representation Aware Fine-Tuning.

This module provides:
- Configurable pooling strategies for sequence representations
- InfoNCE contrastive loss with learnable projection heads
- Loss combination utilities for multi-objective training

References:
- Oord et al. "Representation Learning with Contrastive Predictive Coding"
  (InfoNCE/CPC), 2018
- Chen et al. "A Simple Framework for Contrastive Learning of Visual
  Representations" (SimCLR), 2020 - projection head design
- Gao et al. "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
  2021 - temperature and pooling insights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolingStrategy = Literal["last_token", "mean", "cls", "weighted_mean"]


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: PoolingStrategy,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pool sequence hidden states according to strategy.

    Args:
        hidden_states: Hidden states [batch, seq, hidden].
        attention_mask: Attention mask [batch, seq].
        strategy: Pooling strategy to use.
        weights: Optional per-position weights for weighted_mean [seq].

    Returns:
        Pooled representations [batch, hidden].
    """
    if strategy == "last_token":
        # Get the last non-padded token for each sequence
        last_token_indices = attention_mask.sum(1) - 1
        last_token_indices = last_token_indices.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_token_indices]

    if strategy == "mean":
        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask_expanded).sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden_states.dtype)
        return summed / lengths

    if strategy == "weighted_mean":
        # Weighted mean pooling (useful for position-aware pooling)
        if weights is None:
            # Default: linear decay from end (more weight on recent tokens)
            seq_len = hidden_states.size(1)
            weights = torch.linspace(0.5, 1.0, seq_len, device=hidden_states.device)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, seq, 1]
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        weighted = hidden_states * mask_expanded * weights
        summed = weighted.sum(dim=1)
        total_weight = (mask_expanded * weights).sum(dim=1).clamp(min=1e-9)
        return summed / total_weight

    if strategy == "cls":
        return hidden_states[:, 0]

    raise ValueError(f"Unsupported pooling strategy: {strategy}")


# Keep old name for backward compatibility
_pool_hidden_states = pool_hidden_states


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Following SimCLR (Chen et al., 2020), we use a 2-layer MLP with:
    - Hidden layer with same dimension as input
    - GELU activation (smoother than ReLU, no saturation like Tanh)
    - Output projection to lower dimension for efficiency
    - L2 normalization on output

    The projection head learns a representation space better suited for
    contrastive learning than the raw hidden states.

    References:
    - Chen et al. "A Simple Framework for Contrastive Learning of Visual
      Representations" (SimCLR), 2020
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize."""
        return F.normalize(self.net(x), p=2, dim=-1)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with configurable pooling and projection.

    This implementation supports:
    - Multiple pooling strategies (last_token, mean, cls, weighted_mean)
    - Learnable projection head (2-layer MLP following SimCLR)
    - Configurable temperature for logit scaling
    - Optional learnable temperature (like CLIP)

    The loss encourages aligned pairs to have high similarity while
    pushing apart non-aligned pairs (in-batch negatives).

    References:
    - Oord et al. "Representation Learning with Contrastive Predictive
      Coding" (CPC/InfoNCE), 2018
    - Radford et al. "Learning Transferable Visual Models From Natural
      Language Supervision" (CLIP), 2021 - learnable temperature
    """

    def __init__(
        self,
        temperature: float = 0.05,
        reduction: str = "mean",
        hidden_size: Optional[int] = None,
        projection_dim: int = 256,
        pooling: PoolingStrategy = "last_token",
        learnable_temperature: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.pooling = pooling
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.projector: Optional[ProjectionHead] = None

        # Temperature can be fixed or learnable
        if learnable_temperature:
            # Initialize log(temperature) for numerical stability
            # CLIP uses initial value of log(1/0.07) ≈ 2.66
            self.log_temperature = nn.Parameter(torch.tensor(2.66))
        else:
            self.register_buffer("log_temperature", torch.tensor(float(temperature)).log())

        if hidden_size is not None:
            self._init_projector(hidden_size)

    @property
    def temperature(self) -> torch.Tensor:
        """Return current temperature value."""
        return self.log_temperature.exp()

    def _init_projector(
        self,
        hidden_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the projection head."""
        self.projector = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=self.projection_dim,
            dropout=self.dropout,
        )
        if device is not None:
            self.projector.to(device)

    def _project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings through the projection head."""
        if self.projector is None:
            raise RuntimeError(
                "Projector not initialized. Either pass hidden_size to __init__ "
                "or call forward with hidden states first."
            )
        return self.projector(embeddings)

    def forward(
        self,
        hidden_states_anchor: torch.Tensor,
        hidden_states_positive: torch.Tensor,
        mask_anchor: torch.Tensor,
        mask_positive: torch.Tensor,
        *,
        return_details: bool = False,
        additional_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute InfoNCE loss between anchor and positive hidden states.

        Args:
            hidden_states_anchor: Anchor hidden states [batch, seq, hidden].
            hidden_states_positive: Positive hidden states [batch, seq, hidden].
            mask_anchor: Anchor attention mask [batch, seq].
            mask_positive: Positive attention mask [batch, seq].
            return_details: If True, return additional info (embeddings, logits).
            additional_negatives: Optional additional negative embeddings [num_neg, proj_dim].

        Returns:
            Loss tensor, and optionally a dict with embeddings and logits.
        """
        # Lazy projector initialization
        if self.projector is None:
            self._init_projector(
                hidden_states_anchor.size(-1),
                hidden_states_anchor.device,
            )

        # Pool hidden states to sequence representations
        emb_anchor = pool_hidden_states(hidden_states_anchor, mask_anchor, self.pooling)
        emb_positive = pool_hidden_states(hidden_states_positive, mask_positive, self.pooling)

        # Project to contrastive space (includes normalization)
        proj_anchor = self._project(emb_anchor)
        proj_positive = self._project(emb_positive)

        # Compute similarity matrix
        temperature = self.temperature.clamp(min=0.01, max=100.0)
        logits = torch.matmul(proj_anchor, proj_positive.T) / temperature

        # Add additional negatives if provided
        if additional_negatives is not None and additional_negatives.size(0) > 0:
            neg_logits = torch.matmul(proj_anchor, additional_negatives.T) / temperature
            logits = torch.cat([logits, neg_logits], dim=1)

        # Labels: each anchor matches with its corresponding positive (diagonal)
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        if not return_details:
            return loss

        details = {
            "anchor_embeddings": proj_anchor.detach(),
            "positive_embeddings": proj_positive.detach(),
            "logits": logits.detach(),
            "temperature": temperature.detach(),
        }
        return loss, details

    def forward_from_pooled(
        self,
        anchor_pooled: torch.Tensor,
        positive_pooled: torch.Tensor,
        *,
        additional_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss from already-pooled representations.

        This is more efficient when pooling is done externally (e.g., in
        the trainer for dual-pooling self-align).

        Args:
            anchor_pooled: Pooled anchor representations [batch, hidden].
            positive_pooled: Pooled positive representations [batch, hidden].
            additional_negatives: Optional additional negatives [num_neg, proj_dim].

        Returns:
            Loss tensor.
        """
        if self.projector is None:
            self._init_projector(anchor_pooled.size(-1), anchor_pooled.device)

        proj_anchor = self._project(anchor_pooled)
        proj_positive = self._project(positive_pooled)

        temperature = self.temperature.clamp(min=0.01, max=100.0)
        logits = torch.matmul(proj_anchor, proj_positive.T) / temperature

        if additional_negatives is not None and additional_negatives.size(0) > 0:
            neg_logits = torch.matmul(proj_anchor, additional_negatives.T) / temperature
            logits = torch.cat([logits, neg_logits], dim=1)

        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)

        return F.cross_entropy(logits, labels, reduction=self.reduction)


@dataclass
class CRAFTLossOutputs:
    """Container for the individual components of the CRAFT loss."""

    total_loss: torch.Tensor
    sft_loss: torch.Tensor
    contrastive_loss: torch.Tensor


def combine_craft_losses(
    *,
    sft_loss: torch.Tensor,
    contrastive_loss: torch.Tensor,
    alpha: float,
) -> CRAFTLossOutputs:
    """Return weighted CRAFT loss components."""

    alpha = float(min(max(alpha, 0.0), 1.0))
    total = alpha * sft_loss + (1.0 - alpha) * contrastive_loss
    return CRAFTLossOutputs(total_loss=total, sft_loss=sft_loss, contrastive_loss=contrastive_loss)
