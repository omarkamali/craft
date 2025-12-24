from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_contrastive_accuracy(anchor: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Top-1 accuracy where each anchor's positive should rank highest."""

    sim = anchor @ positives.T
    predicted = sim.argmax(dim=1)
    targets = torch.arange(anchor.size(0), device=anchor.device)
    return predicted.eq(targets).float().mean()


def compute_representation_consistency(
    current: torch.Tensor,
    reference: Optional[torch.Tensor],
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity between current and reference embeddings."""

    if reference is None:
        return torch.tensor(float("nan"), device=current.device)

    # Ensure reference is on the same device as current
    reference = reference.to(current.device)
    
    current_norm = F.normalize(current, dim=-1, eps=epsilon)
    reference_norm = F.normalize(reference, dim=-1, eps=epsilon)
    return (current_norm * reference_norm).sum(dim=-1).mean()


def update_representation_reference(
    prev_reference: Optional[torch.Tensor],
    current: torch.Tensor,
    *,
    momentum: float = 0.9,
) -> torch.Tensor:
    """Exponential moving average of embeddings."""
    
    cur = current.detach().mean(dim=0)  # [D] - mean pool to single vector
    if prev_reference is None:
        return cur
    return momentum * prev_reference + (1 - momentum) * cur
