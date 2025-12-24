"""
Configuration for CRAFT: Contrastive Representation Aware Fine-Tuning.

This module provides configuration mixins and dataclasses for CRAFT training,
including loss balancing, projection head settings, and memory optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:  # Optional TRL dependencies
    from trl import (
        SFTConfig,
        ORPOConfig,
        PPOConfig,
        DPOConfig,
    )
    try:
        from trl import GRPOConfig  # type: ignore
    except ImportError:  # pragma: no cover - missing in older TRL versions
        GRPOConfig = PPOConfig  # type: ignore
except ImportError:  # pragma: no cover - allow import without TRL
    SFTConfig = object  # type: ignore
    ORPOConfig = object  # type: ignore
    PPOConfig = object  # type: ignore
    DPOConfig = object  # type: ignore
    GRPOConfig = object  # type: ignore


_DEFAULT_KEYS: Dict[str, str] = {
    "anchor_input_ids": "input_ids",
    "anchor_attention_mask": "attention_mask",
    "anchor_labels": "labels",
    "positive_input_ids": "input_ids_tgt",
    "positive_attention_mask": "attention_mask_tgt",
}


@dataclass
class CRAFTConfigMixin:
    """
    Mixin adding CRAFT-specific configuration to trainer configs.

    This mixin provides configuration for:
    - Loss balancing (alpha for SFT vs contrastive weight)
    - Projection head architecture
    - Memory optimization (GradCache, hooks)
    - Batch mixing strategy
    - Negative sampling

    All parameters have sensible defaults that work well for most use cases.
    """

    # -------------------------------------------------------------------------
    # Loss Balancing
    # -------------------------------------------------------------------------
    craft_alpha: float = field(
        default=0.6,
        metadata={
            "help": "Weight on SFT loss; (1-alpha) applies to InfoNCE. "
            "With accumulation-aware scaling, this is the true gradient ratio."
        },
    )
    craft_temperature: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for InfoNCE logits. Lower = sharper distribution. "
            "Typical range: 0.01-0.1. See SimCSE (Gao et al., 2021)."
        },
    )
    craft_learnable_temperature: bool = field(
        default=False,
        metadata={
            "help": "If True, temperature is a learnable parameter (like CLIP). "
            "Useful for finding optimal temperature during training."
        },
    )

    # -------------------------------------------------------------------------
    # Projection Head
    # -------------------------------------------------------------------------
    craft_projection_dim: int = field(
        default=256,
        metadata={
            "help": "Output dimension of the projection head. Lower = more memory "
            "efficient. Typical range: 128-512. See SimCLR (Chen et al., 2020)."
        },
    )
    craft_projection_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout rate in projection head. Usually 0 for contrastive."
        },
    )
    craft_pooling: str = field(
        default="last_token",
        metadata={
            "help": "Pooling strategy: last_token|mean|cls|weighted_mean. "
            "last_token works best for causal LMs, mean for bidirectional."
        },
    )

    # -------------------------------------------------------------------------
    # Memory Optimization
    # -------------------------------------------------------------------------
    craft_use_gradcache: bool = field(
        default=False,
        metadata={
            "help": "Use GradCache for memory-efficient contrastive learning. "
            "Enables larger effective batch sizes. See Gao et al., 2021."
        },
    )
    craft_gradcache_chunk_size: int = field(
        default=4,
        metadata={
            "help": "Chunk size for GradCache backward pass. Smaller = less memory "
            "but slower. Only used if craft_use_gradcache=True."
        },
    )
    craft_use_hidden_state_hook: bool = field(
        default=True,
        metadata={
            "help": "Use hook-based hidden state capture instead of output_hidden_states. "
            "Saves memory by only capturing the last layer."
        },
    )

    # -------------------------------------------------------------------------
    # Batch Mixing
    # -------------------------------------------------------------------------
    craft_beta: float = field(
        default=0.6,
        metadata={
            "help": "Fraction of gradient accumulation steps allocated to SFT batches. "
            "With accumulation-aware scaling, this controls batch distribution, "
            "not the final gradient ratio (which is controlled by craft_alpha)."
        },
    )
    craft_beta_mode: str = field(
        default="fixed",
        metadata={
            "help": "How to interpret craft_beta: fixed|auto. "
            "auto adjusts based on dataset lengths."
        },
    )
    craft_length_strategy: str = field(
        default="oversample",
        metadata={
            "help": "Handle dataset length mismatch: oversample|cap|auto_beta|error. "
            "oversample loops shorter dataset, cap stops at shorter, "
            "auto_beta adjusts beta, error raises ValueError."
        },
    )

    # -------------------------------------------------------------------------
    # Negative Sampling
    # -------------------------------------------------------------------------
    craft_negative_strategy: str = field(
        default="in_batch",
        metadata={
            "help": "Negative sampling: in_batch|queue|none. "
            "in_batch uses other batch items as negatives. "
            "queue maintains a memory bank (MoCo-style)."
        },
    )
    craft_negative_queue_size: int = field(
        default=65536,
        metadata={
            "help": "Size of negative queue when using queue strategy. "
            "Larger = more negatives but more memory. See MoCo."
        },
    )

    # -------------------------------------------------------------------------
    # Data Keys
    # -------------------------------------------------------------------------
    craft_contrastive_keys: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_KEYS),
        metadata={"help": "Mapping from canonical CRAFT keys to dataset columns."},
    )
    craft_assistant_mask_strategy: str = field(
        default="auto",
        metadata={
            "help": "How to derive assistant masks: auto|provided|none. "
            "auto uses labels != -100, provided expects explicit mask."
        },
    )
    craft_assistant_mask_key: str | None = field(
        default="assistant_mask",
        metadata={
            "help": "Dataset column providing assistant-token mask for self-align."
        },
    )
    craft_contrastive_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Override batch size for contrastive dataloader. "
            "Defaults to SFT batch size if None."
        },
    )

    # -------------------------------------------------------------------------
    # Metrics & Debugging
    # -------------------------------------------------------------------------
    craft_report_metrics: List[str] = field(
        default_factory=lambda: [
            "contrastive_accuracy",
            "representation_consistency",
        ],
        metadata={
            "help": "Metrics to log: contrastive_accuracy, representation_consistency, "
            "temperature (if learnable), gradient_norm."
        },
    )
    craft_debug: bool = field(
        default=False,
        metadata={
            "help": "Enable debug logging (memory usage, shapes, etc.). "
            "Disable in production for performance."
        },
    )


@dataclass
class CRAFTSFTConfig(CRAFTConfigMixin, SFTConfig):  # type: ignore[misc]
    """Config for the CRAFT-augmented SFT trainer."""


@dataclass
class CRAFTORPOConfig(CRAFTConfigMixin, ORPOConfig):  # type: ignore[misc]
    """Config for CRAFT ORPO trainer."""


@dataclass
class CRAFTGRPOConfig(CRAFTConfigMixin, GRPOConfig):  # type: ignore[misc]
    """Config for CRAFT GRPO trainer."""


@dataclass
class CRAFTPPOConfig(CRAFTConfigMixin, PPOConfig):  # type: ignore[misc]
    """Config for CRAFT PPO trainer."""


@dataclass
class CRAFTDPOConfig(CRAFTConfigMixin, DPOConfig):  # type: ignore[misc]
    """Config for CRAFT DPO trainer."""
