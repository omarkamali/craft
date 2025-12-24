"""Tests for the CRAFT gradient accumulator."""

import pytest
import torch

from craft.accumulator import (
    CRAFTGradientAccumulator,
    AccumulationScales,
    compute_batch_distribution,
)


class TestComputeBatchDistribution:
    """Tests for compute_batch_distribution function."""

    def test_basic_distribution(self):
        """Test basic batch distribution computation."""
        n_sft, n_con = compute_batch_distribution(beta=0.5, gradient_accumulation_steps=4)
        assert n_sft == 2
        assert n_con == 2

    def test_all_sft(self):
        """Test beta=1.0 gives all SFT batches."""
        n_sft, n_con = compute_batch_distribution(beta=1.0, gradient_accumulation_steps=4)
        assert n_sft == 4
        assert n_con == 0

    def test_all_contrastive(self):
        """Test beta=0.0 gives all contrastive batches."""
        n_sft, n_con = compute_batch_distribution(beta=0.0, gradient_accumulation_steps=4)
        assert n_sft == 0
        assert n_con == 4

    def test_ensures_at_least_one_if_beta_nonzero(self):
        """Test that at least one SFT batch is included if beta > 0."""
        n_sft, n_con = compute_batch_distribution(beta=0.1, gradient_accumulation_steps=4)
        assert n_sft >= 1

    def test_ensures_at_least_one_contrastive_if_beta_less_than_one(self):
        """Test that at least one contrastive batch is included if beta < 1."""
        n_sft, n_con = compute_batch_distribution(beta=0.9, gradient_accumulation_steps=4)
        assert n_con >= 1

    def test_rounding(self):
        """Test rounding behavior for fractional distributions."""
        # 0.6 * 5 = 3
        n_sft, n_con = compute_batch_distribution(beta=0.6, gradient_accumulation_steps=5)
        assert n_sft == 3
        assert n_con == 2


class TestCRAFTGradientAccumulator:
    """Tests for CRAFTGradientAccumulator class."""

    def test_initialization(self):
        """Test accumulator initialization."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        assert acc.alpha == 0.6
        assert acc.n_sft == 2
        assert acc.n_contrastive == 2

    def test_scales_computation(self):
        """Test that scales are computed correctly for equal split."""
        # With alpha=0.6, n_sft=2, n_con=2, total=4
        # sft_scale = 0.6 * 4 / 2 = 1.2
        # con_scale = 0.4 * 4 / 2 = 0.8
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        assert abs(acc.scales.sft_scale - 1.2) < 1e-6
        assert abs(acc.scales.contrastive_scale - 0.8) < 1e-6

    def test_scales_unequal_split(self):
        """Test scales with unequal batch distribution."""
        # With alpha=0.6, n_sft=5, n_con=3, total=8
        # sft_scale = 0.6 * 8 / 5 = 0.96
        # con_scale = 0.4 * 8 / 3 = 1.0667
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=8,
            n_sft=5,
            n_contrastive=3,
        )
        assert abs(acc.scales.sft_scale - 0.96) < 1e-6
        assert abs(acc.scales.contrastive_scale - (0.4 * 8 / 3)) < 1e-6

    def test_sft_only(self):
        """Test scales when only SFT batches are present."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=4,
            n_contrastive=0,
        )
        assert acc.scales.sft_scale == 1.0
        assert acc.scales.contrastive_scale == 0.0

    def test_contrastive_only(self):
        """Test scales when only contrastive batches are present."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=0,
            n_contrastive=4,
        )
        assert acc.scales.sft_scale == 0.0
        assert acc.scales.contrastive_scale == 1.0

    def test_scale_sft_loss(self):
        """Test scaling SFT loss."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        loss = torch.tensor(1.0)
        scaled = acc.scale_sft_loss(loss)
        assert abs(scaled.item() - 1.2) < 1e-6

    def test_scale_contrastive_loss(self):
        """Test scaling contrastive loss."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        loss = torch.tensor(1.0)
        scaled = acc.scale_contrastive_loss(loss)
        assert abs(scaled.item() - 0.8) < 1e-6

    def test_update_batch_counts(self):
        """Test updating batch counts dynamically."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        initial_sft_scale = acc.scales.sft_scale

        acc.update_batch_counts(n_sft=3, n_contrastive=1)

        assert acc.n_sft == 3
        assert acc.n_contrastive == 1
        assert acc.scales.sft_scale != initial_sft_scale

    def test_repr(self):
        """Test string representation."""
        acc = CRAFTGradientAccumulator(
            alpha=0.6,
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        repr_str = repr(acc)
        assert "CRAFTGradientAccumulator" in repr_str
        assert "alpha=0.6" in repr_str

    def test_effective_gradient_ratio(self):
        """
        Verify that the scaling produces the correct effective gradient ratio.

        If we accumulate n_sft SFT gradients scaled by sft_scale and n_con
        contrastive gradients scaled by con_scale, the ratio of total
        contributions should be alpha : (1-alpha).
        """
        alpha = 0.6
        n_sft, n_con = 5, 3

        acc = CRAFTGradientAccumulator(
            alpha=alpha,
            gradient_accumulation_steps=8,
            n_sft=n_sft,
            n_contrastive=n_con,
        )

        # Total SFT contribution = n_sft * sft_scale
        sft_total = n_sft * acc.scales.sft_scale
        # Total contrastive contribution = n_con * con_scale
        con_total = n_con * acc.scales.contrastive_scale

        # Ratio should be alpha : (1-alpha)
        total = sft_total + con_total
        actual_sft_ratio = sft_total / total
        actual_con_ratio = con_total / total

        assert abs(actual_sft_ratio - alpha) < 1e-6
        assert abs(actual_con_ratio - (1 - alpha)) < 1e-6

    def test_clamping(self):
        """Test that alpha is clamped to [0, 1]."""
        acc = CRAFTGradientAccumulator(
            alpha=1.5,  # Should be clamped to 1.0
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        assert acc.alpha == 1.0

        acc2 = CRAFTGradientAccumulator(
            alpha=-0.5,  # Should be clamped to 0.0
            gradient_accumulation_steps=4,
            n_sft=2,
            n_contrastive=2,
        )
        assert acc2.alpha == 0.0
