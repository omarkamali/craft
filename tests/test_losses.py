import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from craft.losses import (
    InfoNCELoss,
    ProjectionHead,
    pool_hidden_states,
    _pool_hidden_states,
    combine_craft_losses,
)


def make_batch(batch_size: int, seq_len: int, hidden: int):
    hidden_anchor = torch.zeros(batch_size, seq_len, hidden)
    hidden_pos = torch.zeros_like(hidden_anchor)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    for i in range(batch_size):
        idx = seq_len - 1
        hidden_anchor[i, idx, i % hidden] = 1.0
        hidden_pos[i, idx, i % hidden] = 1.0

    return hidden_anchor, hidden_pos, mask.clone(), mask.clone()


class TestPoolHiddenStates:
    """Tests for pool_hidden_states function."""

    def test_last_token_pooling(self):
        hidden_anchor, _, mask_anchor, _ = make_batch(2, 4, 6)
        pooled = pool_hidden_states(hidden_anchor, mask_anchor, "last_token")
        assert pooled.shape == (2, 6)
        assert torch.allclose(pooled[0], hidden_anchor[0, -1])

    def test_mean_pooling(self):
        hidden = torch.ones(2, 4, 6)
        mask = torch.ones(2, 4, dtype=torch.long)
        pooled = pool_hidden_states(hidden, mask, "mean")
        assert pooled.shape == (2, 6)
        assert torch.allclose(pooled, torch.ones(2, 6))

    def test_mean_pooling_with_padding(self):
        hidden = torch.ones(2, 4, 6)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        pooled = pool_hidden_states(hidden, mask, "mean")
        assert pooled.shape == (2, 6)
        # All non-padded values are 1, so mean should be 1
        assert torch.allclose(pooled, torch.ones(2, 6))

    def test_cls_pooling(self):
        hidden = torch.randn(2, 4, 6)
        mask = torch.ones(2, 4, dtype=torch.long)
        pooled = pool_hidden_states(hidden, mask, "cls")
        assert pooled.shape == (2, 6)
        assert torch.allclose(pooled, hidden[:, 0])

    def test_weighted_mean_pooling(self):
        hidden = torch.ones(2, 4, 6)
        mask = torch.ones(2, 4, dtype=torch.long)
        pooled = pool_hidden_states(hidden, mask, "weighted_mean")
        assert pooled.shape == (2, 6)

    def test_invalid_strategy_raises(self):
        hidden = torch.randn(2, 4, 6)
        mask = torch.ones(2, 4, dtype=torch.long)
        with pytest.raises(ValueError):
            pool_hidden_states(hidden, mask, "invalid")

    def test_backward_compatibility(self):
        """Test that _pool_hidden_states is still available."""
        hidden, _, mask, _ = make_batch(2, 4, 6)
        pooled = _pool_hidden_states(hidden, mask, "last_token")
        assert pooled.shape == (2, 6)


class TestProjectionHead:
    """Tests for ProjectionHead class."""

    def test_initialization(self):
        proj = ProjectionHead(input_dim=64, output_dim=32)
        assert proj.net is not None

    def test_forward_shape(self):
        proj = ProjectionHead(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 32)

    def test_output_is_normalized(self):
        proj = ProjectionHead(input_dim=64, output_dim=32)
        x = torch.randn(4, 64)
        out = proj(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_custom_hidden_dim(self):
        proj = ProjectionHead(input_dim=64, hidden_dim=128, output_dim=32)
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 32)

    def test_with_dropout(self):
        proj = ProjectionHead(input_dim=64, output_dim=32, dropout=0.1)
        proj.train()
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 32)


class TestInfoNCELoss:
    """Tests for InfoNCELoss class."""

    def test_forward_basic(self):
        bsz, seq, dim = 4, 3, 8
        h_a, h_p, m_a, m_p = make_batch(bsz, seq, dim)

        loss_fn = InfoNCELoss(temperature=0.1, hidden_size=dim)
        loss = loss_fn(h_a, h_p, m_a, m_p)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_alignment_vs_mismatch(self):
        bsz, seq, dim = 6, 3, 10
        h_a, h_p, m_a, m_p = make_batch(bsz, seq, dim)

        loss_fn = InfoNCELoss(temperature=0.1, hidden_size=dim)

        aligned = loss_fn(h_a, h_p, m_a, m_p).item()
        perm = torch.randperm(bsz)
        mismatched = loss_fn(h_a, h_p[perm], m_a, m_p[perm]).item()

        assert aligned < mismatched

    def test_return_details(self):
        h_a, h_p, m_a, m_p = make_batch(3, 4, 64)
        loss_fn = InfoNCELoss(temperature=0.2, hidden_size=64, projection_dim=32)

        loss, details = loss_fn(h_a, h_p, m_a, m_p, return_details=True)

        assert torch.isfinite(loss)
        assert "anchor_embeddings" in details
        assert "positive_embeddings" in details
        assert "logits" in details
        assert "temperature" in details
        assert details["anchor_embeddings"].shape == (3, 32)

    def test_learnable_temperature(self):
        loss_fn = InfoNCELoss(
            temperature=0.05,
            hidden_size=64,
            learnable_temperature=True,
        )

        # Check that log_temperature is a parameter
        assert isinstance(loss_fn.log_temperature, nn.Parameter)

        # Temperature should be exp(log_temperature)
        temp = loss_fn.temperature
        assert torch.isfinite(temp)

    def test_fixed_temperature(self):
        loss_fn = InfoNCELoss(
            temperature=0.05,
            hidden_size=64,
            learnable_temperature=False,
        )

        # Check that log_temperature is a buffer, not parameter
        assert "log_temperature" in dict(loss_fn.named_buffers())
        # Verify it's not a parameter by checking the parameters list
        param_names = [name for name, _ in loss_fn.named_parameters()]
        assert "log_temperature" not in param_names

    def test_forward_from_pooled(self):
        anchor_pooled = torch.randn(4, 64)
        positive_pooled = torch.randn(4, 64)

        loss_fn = InfoNCELoss(temperature=0.1, hidden_size=64)
        loss = loss_fn.forward_from_pooled(anchor_pooled, positive_pooled)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_with_additional_negatives(self):
        h_a, h_p, m_a, m_p = make_batch(4, 5, 64)
        additional_neg = F.normalize(torch.randn(10, 256), dim=-1)  # projection_dim

        loss_fn = InfoNCELoss(temperature=0.1, hidden_size=64, projection_dim=256)
        loss = loss_fn(h_a, h_p, m_a, m_p, additional_negatives=additional_neg)

        assert torch.isfinite(loss)

    def test_projection_dim_customizable(self):
        loss_fn = InfoNCELoss(
            temperature=0.1,
            hidden_size=64,
            projection_dim=128,
        )

        h_a, h_p, m_a, m_p = make_batch(4, 5, 64)
        loss, details = loss_fn(h_a, h_p, m_a, m_p, return_details=True)

        assert details["anchor_embeddings"].shape[-1] == 128

    def test_different_pooling_strategies(self):
        h_a, h_p, m_a, m_p = make_batch(4, 5, 64)

        for pooling in ["last_token", "mean", "cls"]:
            loss_fn = InfoNCELoss(
                temperature=0.1,
                hidden_size=64,
                pooling=pooling,
            )
            loss = loss_fn(h_a, h_p, m_a, m_p)
            assert torch.isfinite(loss)


class TestCombineCraftLosses:
    """Tests for combine_craft_losses function."""

    def test_balances_alpha(self):
        sft = torch.tensor(2.0)
        contrastive = torch.tensor(1.0)

        result = combine_craft_losses(sft_loss=sft, contrastive_loss=contrastive, alpha=0.25)
        assert torch.isclose(result.total_loss, torch.tensor(1.25))
        assert result.sft_loss is sft
        assert result.contrastive_loss is contrastive

    def test_alpha_one_is_sft_only(self):
        sft = torch.tensor(2.0)
        contrastive = torch.tensor(1.0)

        result = combine_craft_losses(sft_loss=sft, contrastive_loss=contrastive, alpha=1.0)
        assert torch.isclose(result.total_loss, sft)

    def test_alpha_zero_is_contrastive_only(self):
        sft = torch.tensor(2.0)
        contrastive = torch.tensor(1.0)

        result = combine_craft_losses(sft_loss=sft, contrastive_loss=contrastive, alpha=0.0)
        assert torch.isclose(result.total_loss, contrastive)

    def test_alpha_clamping(self):
        sft = torch.tensor(2.0)
        contrastive = torch.tensor(1.0)

        # Alpha > 1 should be clamped
        result = combine_craft_losses(sft_loss=sft, contrastive_loss=contrastive, alpha=1.5)
        assert torch.isclose(result.total_loss, sft)

        # Alpha < 0 should be clamped
        result = combine_craft_losses(sft_loss=sft, contrastive_loss=contrastive, alpha=-0.5)
        assert torch.isclose(result.total_loss, contrastive)
