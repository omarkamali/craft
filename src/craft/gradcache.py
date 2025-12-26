"""
GradCache: Memory-efficient contrastive learning via gradient caching.

This module implements the GradCache technique for computing contrastive losses
with large batch sizes under memory constraints. The key insight is to decouple
representation computation from gradient computation.

Phase 1: Cache representations (no gradient on backbone)
- Forward all examples through backbone with torch.no_grad()
- Cache the pooled embeddings

Phase 2: Compute loss and gradients (chunked)
- Compute similarity matrix from cached embeddings
- Compute InfoNCE loss
- Backprop through projector
- Recompute backbone activations in chunks for gradient computation

This allows effective batch sizes of 1000+ even on a single GPU, which is
critical for contrastive learning quality.

References:
- Gao et al. "Scaling Deep Contrastive Learning Batch Size under Memory
  Limited Setup" (GradCache), EMNLP 2021
- Xiong et al. "Approximate Nearest Neighbor Negative Contrastive Learning
  for Dense Text Retrieval" (ANCE), ICLR 2021
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


@dataclass
class GradCacheConfig:
    """Configuration for GradCache contrastive computation."""

    chunk_size: int = 4
    """Number of examples to process per backward chunk."""

    temperature: float = 0.05
    """Temperature for InfoNCE logits scaling."""

    use_amp: bool = True
    """Whether to use automatic mixed precision."""

    sync_gradients: bool = True
    """Whether to sync gradients in distributed training."""


class GradCacheContrastiveLoss(nn.Module):
    """
    Memory-efficient contrastive loss using gradient caching.

    This implementation allows computing InfoNCE loss with arbitrarily large
    batch sizes by:
    1. Caching representations without gradients
    2. Computing the full similarity matrix
    3. Backpropagating through backbone in chunks

    Memory complexity: O(B×D + chunk×S×H) instead of O(B×S×H×L)
    where B=batch, D=projection_dim, S=seq_len, H=hidden, L=layers

    Example:
        gradcache = GradCacheContrastiveLoss(
            backbone=model.model,
            projector=projector,
            pooling_fn=pool_last_token,
            config=GradCacheConfig(chunk_size=8),
        )

        loss = gradcache(
            anchor_ids, anchor_mask,
            positive_ids, positive_mask,
        )
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        pooling_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: Optional[GradCacheConfig] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.pooling_fn = pooling_fn
        self.config = config or GradCacheConfig()

    def forward(
        self,
        anchor_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute InfoNCE loss with gradient caching.

        Args:
            anchor_ids: Anchor input IDs [batch, seq].
            anchor_mask: Anchor attention mask [batch, seq].
            positive_ids: Positive input IDs [batch, seq].
            positive_mask: Positive attention mask [batch, seq].
            return_embeddings: If True, also return anchor and positive embeddings.

        Returns:
            InfoNCE loss scalar, and optionally (anchor_emb, positive_emb).
        """
        batch_size = anchor_ids.size(0)
        device = anchor_ids.device

        # Phase 1: Cache representations (no gradients on backbone)
        with torch.no_grad():
            anchor_hidden = self._forward_backbone(anchor_ids, anchor_mask)
            positive_hidden = self._forward_backbone(positive_ids, positive_mask)

            anchor_pooled = self.pooling_fn(anchor_hidden, anchor_mask)
            positive_pooled = self.pooling_fn(positive_hidden, positive_mask)

        # Detach and enable gradients for projector
        anchor_pooled = anchor_pooled.detach().requires_grad_(True)
        positive_pooled = positive_pooled.detach().requires_grad_(True)

        # Project and normalize
        anchor_emb = F.normalize(self.projector(anchor_pooled), dim=-1)
        positive_emb = F.normalize(self.projector(positive_pooled), dim=-1)

        # Compute InfoNCE loss
        logits = torch.matmul(anchor_emb, positive_emb.T) / self.config.temperature
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)

        # Phase 2: Backprop through backbone in chunks
        if self.training and anchor_pooled.grad_fn is not None:
            # Get gradients on pooled representations
            loss.backward(retain_graph=True)

            if anchor_pooled.grad is not None and positive_pooled.grad is not None:
                self._chunked_backward(
                    anchor_ids,
                    anchor_mask,
                    anchor_pooled.grad,
                    positive_ids,
                    positive_mask,
                    positive_pooled.grad,
                )

        if return_embeddings:
            return loss, anchor_emb.detach(), positive_emb.detach()
        return loss

    def _forward_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through backbone, returning hidden states."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,  # We use hook or direct output
        )

        # Handle different output types
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        # For backbone-only models that return tensor directly
        if isinstance(outputs, torch.Tensor):
            return outputs
        # Last resort
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    @torch.enable_grad()
    def _chunked_backward(
        self,
        anchor_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        anchor_grad: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        positive_grad: torch.Tensor,
    ) -> None:
        """
        Backpropagate gradients through backbone in chunks.

        This recomputes activations for small chunks at a time, limiting
        peak memory usage during backward pass.
        
        Note: We temporarily disable gradient checkpointing during chunked
        backward since GradCache already handles memory efficiency through
        chunking. This avoids the "None of the inputs have requires_grad=True"
        warning from checkpointing with integer input_ids.
        """
        chunk_size = self.config.chunk_size
        batch_size = anchor_ids.size(0)

        # Temporarily disable gradient checkpointing if enabled
        # GradCache chunking already provides memory efficiency
        gc_enabled = getattr(self.backbone, 'gradient_checkpointing', False)
        if gc_enabled:
            self.backbone.gradient_checkpointing_disable()

        try:
            # Process anchor chunks
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)

                chunk_ids = anchor_ids[start:end]
                chunk_mask = anchor_mask[start:end]
                chunk_grad = anchor_grad[start:end]

                # Recompute with gradients
                chunk_hidden = self._forward_backbone(chunk_ids, chunk_mask)
                chunk_pooled = self.pooling_fn(chunk_hidden, chunk_mask)

                # Backward through this chunk
                chunk_pooled.backward(chunk_grad)

            # Process positive chunks
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)

                chunk_ids = positive_ids[start:end]
                chunk_mask = positive_mask[start:end]
                chunk_grad = positive_grad[start:end]

                chunk_hidden = self._forward_backbone(chunk_ids, chunk_mask)
                chunk_pooled = self.pooling_fn(chunk_hidden, chunk_mask)

                chunk_pooled.backward(chunk_grad)
        finally:
            # Re-enable gradient checkpointing if it was enabled
            if gc_enabled:
                self.backbone.gradient_checkpointing_enable()


class CachedEmbeddingBank:
    """
    Memory bank for storing embeddings across batches.

    This implements a queue-based embedding bank for extending the effective
    number of negatives beyond the current batch, similar to MoCo.

    References:
    - He et al. "Momentum Contrast for Unsupervised Visual Representation
      Learning" (MoCo), CVPR 2020
    """

    def __init__(
        self,
        embedding_dim: int,
        bank_size: int = 65536,
        device: Optional[torch.device] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.bank_size = bank_size
        self.device = device or torch.device("cpu")

        # Initialize bank
        self.bank = torch.zeros(bank_size, embedding_dim, device=self.device)
        self.ptr = 0
        self.is_full = False

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor) -> None:
        """Add embeddings to the bank, replacing oldest if full."""
        embeddings = embeddings.detach().to(self.device)
        batch_size = embeddings.size(0)

        if batch_size > self.bank_size:
            # Only keep the last bank_size embeddings
            embeddings = embeddings[-self.bank_size:]
            batch_size = self.bank_size

        # Handle wraparound
        end_ptr = self.ptr + batch_size
        if end_ptr <= self.bank_size:
            self.bank[self.ptr:end_ptr] = embeddings
        else:
            # Split across end and beginning
            first_part = self.bank_size - self.ptr
            self.bank[self.ptr:] = embeddings[:first_part]
            self.bank[:batch_size - first_part] = embeddings[first_part:]
            self.is_full = True

        self.ptr = end_ptr % self.bank_size
        if end_ptr >= self.bank_size:
            self.is_full = True

    def get_negatives(self) -> torch.Tensor:
        """Return all stored embeddings as potential negatives."""
        if self.is_full:
            return self.bank
        return self.bank[:self.ptr]

    @property
    def size(self) -> int:
        """Return current number of stored embeddings."""
        return self.bank_size if self.is_full else self.ptr

    def clear(self) -> None:
        """Clear the embedding bank."""
        self.bank.zero_()
        self.ptr = 0
        self.is_full = False


def compute_infonce_with_negatives(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Compute InfoNCE loss with optional additional negatives.

    Args:
        anchor: Anchor embeddings [batch, dim], normalized.
        positive: Positive embeddings [batch, dim], normalized.
        negatives: Optional additional negatives [num_neg, dim], normalized.
        temperature: Temperature scaling.

    Returns:
        InfoNCE loss scalar.
    """
    batch_size = anchor.size(0)
    device = anchor.device

    # Similarity with positives (diagonal should be high)
    pos_sim = torch.matmul(anchor, positive.T) / temperature  # [B, B]

    if negatives is not None and negatives.size(0) > 0:
        # Similarity with additional negatives
        neg_sim = torch.matmul(anchor, negatives.T) / temperature  # [B, N]
        # Concatenate: [positives (B), negatives (N)]
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [B, B+N]
    else:
        all_sim = pos_sim

    # Labels are the diagonal indices (each anchor's positive is at its own index)
    labels = torch.arange(batch_size, device=device, dtype=torch.long)

    return F.cross_entropy(all_sim, labels)
