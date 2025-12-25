"""
Accumulation-aware loss scaling for multi-objective training.

This module implements proper gradient accumulation scaling for combining
SFT and contrastive objectives. The key insight from large-scale multi-task
training (T5, PaLM) is that gradient accumulation IS the combination mechanism,
not an afterthought.

References:
- Raffel et al. "Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer" (T5), 2020
- Chowdhery et al. "PaLM: Scaling Language Modeling with Pathways", 2022
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AccumulationScales:
    """Precomputed loss scales for consistent gradient contribution."""

    sft_scale: float
    contrastive_scale: float
    n_sft: int
    n_contrastive: int
    effective_alpha: float


class CRAFTGradientAccumulator:
    """
    Handles proper loss scaling across heterogeneous gradient accumulation.

    Problem: When alternating between SFT and contrastive batches, naive scaling
    (alpha on SFT, 1-alpha on contrastive) results in incorrect effective weights
    due to batch frequency interaction.

    Solution: Scale losses so that the accumulated gradient over the full window
    matches the desired alpha:(1-alpha) ratio, regardless of how batches are
    distributed.

    Example:
        With alpha=0.6, n_sft=5, n_contrastive=3 per window:
        - Naive: SFT contributes 5*0.6=3.0, contrastive 3*0.4=1.2 → ratio 71:29
        - Correct: Scale SFT by 0.96, contrastive by 1.07 → ratio 60:40

    Attributes:
        alpha: Desired weight on SFT loss in final gradient.
        gradient_accumulation_steps: Total steps per optimizer update.
        n_sft: Number of SFT batches per accumulation window.
        n_contrastive: Number of contrastive batches per accumulation window.
    """

    def __init__(
        self,
        alpha: float,
        gradient_accumulation_steps: int,
        n_sft: int,
        n_contrastive: int,
    ) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.n_sft = max(0, n_sft)
        self.n_contrastive = max(0, n_contrastive)

        self._scales = self._compute_scales()

    def _compute_scales(self) -> AccumulationScales:
        """
        Compute loss scales ensuring correct gradient ratio after accumulation.

        The math:
        - We want final gradient G = alpha * G_sft + (1-alpha) * G_con
        - With n_sft SFT steps and n_con contrastive steps, accumulated gradient is:
          G_accum = sum(scale_sft * g_sft_i) + sum(scale_con * g_con_j)
                  = n_sft * scale_sft * avg_g_sft + n_con * scale_con * avg_g_con
        - For this to equal alpha * G_sft + (1-alpha) * G_con:
          scale_sft = alpha / n_sft
          scale_con = (1-alpha) / n_con

        IMPORTANT: Modern HF Trainer (4.43+) with PEFT models does NOT divide
        the loss by gradient_accumulation_steps when num_items_in_batch is computed.
        This is because PEFT models have **kwargs in forward, setting
        model_accepts_loss_kwargs=True. We must account for this by NOT
        multiplying by total_steps in our scaling.

        Note: When either n_sft or n_contrastive is 0, we handle gracefully.
        """
        total = self.n_sft + self.n_contrastive

        if total == 0:
            # No batches at all - shouldn't happen, but handle gracefully
            return AccumulationScales(
                sft_scale=1.0,
                contrastive_scale=1.0,
                n_sft=0,
                n_contrastive=0,
                effective_alpha=self.alpha,
            )

        if self.n_sft == 0:
            # Contrastive only - scale to get (1-alpha) contribution
            return AccumulationScales(
                sft_scale=0.0,
                contrastive_scale=(1.0 - self.alpha) / self.n_contrastive,
                n_sft=0,
                n_contrastive=self.n_contrastive,
                effective_alpha=0.0,
            )

        if self.n_contrastive == 0:
            # SFT only - scale to get alpha contribution
            return AccumulationScales(
                sft_scale=self.alpha / self.n_sft,
                contrastive_scale=0.0,
                n_sft=self.n_sft,
                n_contrastive=0,
                effective_alpha=1.0,
            )

        # Both present - compute proper scales
        # Each SFT batch contributes: loss * (alpha / n_sft)
        # Over n_sft batches: n_sft * loss * (alpha / n_sft) = alpha * loss ✓
        # Each contrastive batch contributes: loss * ((1-alpha) / n_con)
        # Over n_con batches: n_con * loss * ((1-alpha) / n_con) = (1-alpha) * loss ✓
        sft_scale = self.alpha / self.n_sft
        con_scale = (1.0 - self.alpha) / self.n_contrastive

        return AccumulationScales(
            sft_scale=sft_scale,
            contrastive_scale=con_scale,
            n_sft=self.n_sft,
            n_contrastive=self.n_contrastive,
            effective_alpha=self.alpha,
        )

    @property
    def scales(self) -> AccumulationScales:
        """Return precomputed scales."""
        return self._scales

    def scale_sft_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale SFT loss for proper gradient contribution."""
        return loss * self._scales.sft_scale

    def scale_contrastive_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale contrastive loss for proper gradient contribution."""
        return loss * self._scales.contrastive_scale

    def update_batch_counts(self, n_sft: int, n_contrastive: int) -> None:
        """
        Update batch counts and recompute scales.

        Useful when batch distribution changes dynamically (e.g., auto_beta mode).
        """
        self.n_sft = max(0, n_sft)
        self.n_contrastive = max(0, n_contrastive)
        self._scales = self._compute_scales()

    def __repr__(self) -> str:
        return (
            f"CRAFTGradientAccumulator(alpha={self.alpha}, "
            f"sft_scale={self._scales.sft_scale:.3f}, "
            f"con_scale={self._scales.contrastive_scale:.3f}, "
            f"n_sft={self.n_sft}, n_con={self.n_contrastive})"
        )


def compute_batch_distribution(
    beta: float,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """
    Compute SFT/contrastive batch counts from beta ratio.

    Args:
        beta: Fraction of steps allocated to SFT (0 to 1).
        gradient_accumulation_steps: Total steps per window.

    Returns:
        Tuple of (n_sft, n_contrastive).
    """
    beta = float(max(0.0, min(1.0, beta)))
    total = max(1, gradient_accumulation_steps)

    n_sft = int(round(beta * total))
    n_sft = max(0, min(total, n_sft))
    n_contrastive = total - n_sft

    # Ensure at least one of each if both are desired
    if beta > 0 and n_sft == 0:
        n_sft = 1
        n_contrastive = max(0, total - 1)
    if beta < 1 and n_contrastive == 0:
        n_contrastive = 1
        n_sft = max(0, total - 1)

    return n_sft, n_contrastive
