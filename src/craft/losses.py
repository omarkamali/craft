from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolingStrategy = Literal["last_token", "mean", "cls"]


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: PoolingStrategy,
) -> torch.Tensor:
    """Pool sequence hidden states according to ``strategy``."""

    if strategy == "last_token":
        last_token_indices = attention_mask.sum(1) - 1
        last_token_indices = last_token_indices.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_token_indices]

    if strategy == "mean":
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        summed = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
        return summed / lengths

    if strategy == "cls":
        return hidden_states[:, 0]

    raise ValueError(f"Unsupported pooling strategy: {strategy}")


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with configurable pooling and projection."""

    def __init__(
        self,
        temperature: float = 0.05,
        reduction: str = "mean",
        hidden_size: Optional[int] = None,
        pooling: PoolingStrategy = "last_token",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.pooling = pooling
        self.projector: Optional[nn.Module] = None

        if hidden_size is not None:
            self._init_projector(hidden_size)

    def _init_projector(self, hidden_size: int, device: Optional[torch.device] = None) -> None:
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        if device is not None:
            self.projector.to(device)

    def _project(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            raise RuntimeError("Projector not initialised; call forward with hidden states first")
        return F.normalize(self.projector(embeddings), p=2, dim=1)

    def forward(
        self,
        hidden_states_anchor: torch.Tensor,
        hidden_states_positive: torch.Tensor,
        mask_anchor: torch.Tensor,
        mask_positive: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.projector is None:
            self._init_projector(hidden_states_anchor.size(-1), hidden_states_anchor.device)

        emb_anchor = _pool_hidden_states(hidden_states_anchor, mask_anchor, self.pooling)
        emb_positive = _pool_hidden_states(hidden_states_positive, mask_positive, self.pooling)

        proj_anchor = self._project(emb_anchor)
        proj_positive = self._project(emb_positive)

        logits = torch.matmul(proj_anchor, proj_positive.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        if not return_details:
            return loss

        details = {
            "anchor_embeddings": proj_anchor.detach(),
            "positive_embeddings": proj_positive.detach(),
        }
        return loss, details


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
