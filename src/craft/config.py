from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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
    """Mixin adding CRAFT-specific knobs to trainer configs."""

    craft_alpha: float = field(
        default=0.6,
        metadata={"help": "Weight on SFT loss; (1-alpha) applies to InfoNCE"},
    )
    craft_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for InfoNCE logits"},
    )
    craft_pooling: str = field(
        default="last_token",
        metadata={"help": "Pooling strategy before InfoNCE (last_token|mean|cls)"},
    )
    craft_contrastive_keys: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_KEYS),
        metadata={"help": "Mapping from canonical CRAFT keys to dataset columns"},
    )
    craft_assistant_mask_strategy: str = field(
        default="auto",
        metadata={"help": "How to derive assistant masks (auto|provided|none)"},
    )
    craft_negative_strategy: str = field(
        default="in_batch",
        metadata={"help": "Negative sampling strategy (in_batch|queue|custom)"},
    )
    craft_report_metrics: List[str] = field(
        default_factory=lambda: [
            "contrastive_accuracy",
            "representation_consistency",
        ],
        metadata={"help": "List of craft metrics to log"},
    )
    craft_beta: float = field(
        default=0.6,
        metadata={
            "help": "Fraction of gradient accumulation steps allocated to SFT batches",
        },
    )
    craft_beta_mode: str = field(
        default="fixed",
        metadata={
            "help": "How to interpret craft_beta (fixed|auto)",
        },
    )
    craft_contrastive_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Override batch size for contrastive dataloader (defaults to SFT batch size)",
        },
    )
    craft_length_strategy: str = field(
        default="oversample",
        metadata={
            "help": "Handle dataset length mismatch: oversample|cap|auto_beta|error",
        },
    )
    craft_assistant_mask_key: str | None = field(
        default="assistant_mask",
        metadata={
            "help": "Dataset column providing assistant-token mask for self-align positives.",
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
