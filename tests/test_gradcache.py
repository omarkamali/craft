"""Tests for the CRAFT GradCache module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from craft.gradcache import (
    GradCacheConfig,
    GradCacheContrastiveLoss,
    CachedEmbeddingBank,
    compute_infonce_with_negatives,
)


class SimpleBackbone(nn.Module):
    """Simple backbone for testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.linear(x)
        return type("Output", (), {"last_hidden_state": x})()


class TestGradCacheConfig:
    """Tests for GradCacheConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GradCacheConfig()
        assert config.chunk_size == 4
        assert config.temperature == 0.05
        assert config.use_amp is True
        assert config.sync_gradients is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GradCacheConfig(
            chunk_size=8,
            temperature=0.1,
            use_amp=False,
        )
        assert config.chunk_size == 8
        assert config.temperature == 0.1
        assert config.use_amp is False


class TestCachedEmbeddingBank:
    """Tests for CachedEmbeddingBank."""

    def test_initialization(self):
        """Test bank initialization."""
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=100)
        assert bank.embedding_dim == 64
        assert bank.bank_size == 100
        assert bank.size == 0
        assert not bank.is_full

    def test_enqueue(self):
        """Test adding embeddings to the bank."""
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=100)

        embeddings = torch.randn(10, 64)
        bank.enqueue(embeddings)

        assert bank.size == 10

    def test_enqueue_wraps_around(self):
        """Test that bank wraps around when full."""
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=20)

        # Add 15 embeddings
        bank.enqueue(torch.randn(15, 64))
        assert bank.size == 15

        # Add 10 more - should wrap around
        bank.enqueue(torch.randn(10, 64))
        assert bank.is_full
        assert bank.size == 20  # Full bank size

    def test_get_negatives(self):
        """Test retrieving negatives."""
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=100)

        embeddings = torch.randn(10, 64)
        bank.enqueue(embeddings)

        negatives = bank.get_negatives()
        assert negatives.shape == (10, 64)

    def test_get_negatives_when_full(self):
        """Test retrieving negatives when bank is full."""
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=20)

        # Fill the bank
        bank.enqueue(torch.randn(25, 64))

        negatives = bank.get_negatives()
        assert negatives.shape == (20, 64)  # Full bank

    def test_clear(self):
        """Test clearing the bank."""
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=100)

        bank.enqueue(torch.randn(10, 64))
        assert bank.size == 10

        bank.clear()
        assert bank.size == 0
        assert not bank.is_full

    def test_device_handling(self):
        """Test that bank respects device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")
        bank = CachedEmbeddingBank(embedding_dim=64, bank_size=100, device=device)

        embeddings = torch.randn(10, 64)  # On CPU
        bank.enqueue(embeddings)

        negatives = bank.get_negatives()
        assert negatives.device == device


class TestComputeInfoNCEWithNegatives:
    """Tests for compute_infonce_with_negatives function."""

    def test_basic_infonce(self):
        """Test basic InfoNCE computation."""
        anchor = F.normalize(torch.randn(4, 64), dim=-1)
        positive = F.normalize(torch.randn(4, 64), dim=-1)

        loss = compute_infonce_with_negatives(anchor, positive)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_with_additional_negatives(self):
        """Test InfoNCE with additional negatives."""
        anchor = F.normalize(torch.randn(4, 64), dim=-1)
        positive = F.normalize(torch.randn(4, 64), dim=-1)
        negatives = F.normalize(torch.randn(10, 64), dim=-1)

        loss = compute_infonce_with_negatives(anchor, positive, negatives)

        assert loss.ndim == 0
        assert loss.item() > 0

    def test_perfect_alignment_low_loss(self):
        """Test that perfect alignment gives lower loss."""
        anchor = F.normalize(torch.randn(4, 64), dim=-1)

        # Perfect alignment: positive = anchor
        loss_aligned = compute_infonce_with_negatives(anchor, anchor.clone())

        # Random positive
        loss_random = compute_infonce_with_negatives(
            anchor, F.normalize(torch.randn(4, 64), dim=-1)
        )

        assert loss_aligned < loss_random

    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        anchor = F.normalize(torch.randn(4, 64), dim=-1)
        positive = F.normalize(torch.randn(4, 64), dim=-1)

        loss_low_temp = compute_infonce_with_negatives(
            anchor, positive, temperature=0.01
        )
        loss_high_temp = compute_infonce_with_negatives(
            anchor, positive, temperature=1.0
        )

        # Lower temperature typically gives higher loss for non-perfect alignment
        assert loss_low_temp != loss_high_temp


class TestGradCacheContrastiveLoss:
    """Tests for GradCacheContrastiveLoss."""

    def _make_pooling_fn(self, strategy="mean"):
        """Create a simple pooling function."""
        def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            mask_expanded = mask.unsqueeze(-1).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return summed / lengths
        return mean_pool

    def test_forward_pass(self):
        """Test forward pass computes loss."""
        backbone = SimpleBackbone(hidden_size=64)
        projector = nn.Linear(64, 32)
        pooling_fn = self._make_pooling_fn()

        gradcache = GradCacheContrastiveLoss(
            backbone=backbone,
            projector=projector,
            pooling_fn=pooling_fn,
            config=GradCacheConfig(chunk_size=2),
        )

        anchor_ids = torch.randint(0, 100, (4, 10))
        anchor_mask = torch.ones(4, 10, dtype=torch.long)
        positive_ids = torch.randint(0, 100, (4, 10))
        positive_mask = torch.ones(4, 10, dtype=torch.long)

        loss = gradcache(anchor_ids, anchor_mask, positive_ids, positive_mask)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_return_embeddings(self):
        """Test that embeddings can be returned."""
        backbone = SimpleBackbone(hidden_size=64)
        projector = nn.Linear(64, 32)
        pooling_fn = self._make_pooling_fn()

        gradcache = GradCacheContrastiveLoss(
            backbone=backbone,
            projector=projector,
            pooling_fn=pooling_fn,
        )

        anchor_ids = torch.randint(0, 100, (4, 10))
        anchor_mask = torch.ones(4, 10, dtype=torch.long)
        positive_ids = torch.randint(0, 100, (4, 10))
        positive_mask = torch.ones(4, 10, dtype=torch.long)

        loss, anchor_emb, positive_emb = gradcache(
            anchor_ids, anchor_mask, positive_ids, positive_mask,
            return_embeddings=True,
        )

        assert loss.ndim == 0
        assert anchor_emb.shape == (4, 32)
        assert positive_emb.shape == (4, 32)

    def test_embeddings_are_normalized(self):
        """Test that returned embeddings are L2-normalized."""
        backbone = SimpleBackbone(hidden_size=64)
        projector = nn.Linear(64, 32)
        pooling_fn = self._make_pooling_fn()

        gradcache = GradCacheContrastiveLoss(
            backbone=backbone,
            projector=projector,
            pooling_fn=pooling_fn,
        )

        anchor_ids = torch.randint(0, 100, (4, 10))
        anchor_mask = torch.ones(4, 10, dtype=torch.long)
        positive_ids = torch.randint(0, 100, (4, 10))
        positive_mask = torch.ones(4, 10, dtype=torch.long)

        _, anchor_emb, _ = gradcache(
            anchor_ids, anchor_mask, positive_ids, positive_mask,
            return_embeddings=True,
        )

        # Check L2 norms are approximately 1
        norms = anchor_emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_eval_mode(self):
        """Test that GradCache works in eval mode."""
        backbone = SimpleBackbone(hidden_size=64)
        projector = nn.Linear(64, 32)
        pooling_fn = self._make_pooling_fn()

        gradcache = GradCacheContrastiveLoss(
            backbone=backbone,
            projector=projector,
            pooling_fn=pooling_fn,
        )
        gradcache.eval()

        anchor_ids = torch.randint(0, 100, (4, 10))
        anchor_mask = torch.ones(4, 10, dtype=torch.long)
        positive_ids = torch.randint(0, 100, (4, 10))
        positive_mask = torch.ones(4, 10, dtype=torch.long)

        with torch.no_grad():
            loss = gradcache(anchor_ids, anchor_mask, positive_ids, positive_mask)

        assert torch.isfinite(loss)
