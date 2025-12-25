"""
Configuration for CRAFT: Contrastive Representation Aware Fine-Tuning.

This module provides configuration mixins and dataclasses for CRAFT training,
including loss balancing, projection head settings, and memory optimization.

Presets are available for common use cases:
- "minimal": Just add contrastive with minimal changes
- "balanced": Good defaults for most cases (recommended)
- "memory_efficient": For limited GPU memory
- "large_batch": For 1000+ effective batch size
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

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


# =============================================================================
# Presets: Pre-configured settings for common use cases
# =============================================================================

CRAFT_PRESETS: Dict[str, Dict[str, Any]] = {
    "minimal": {
        # Just add contrastive, minimal changes from base training
        # Use when: You want to try CRAFT with minimal risk
        "craft_alpha": 0.8,  # Mostly SFT
        "craft_gradient_balancing": "none",
        "craft_learnable_temperature": False,
        "craft_use_gradcache": False,
        "craft_beta_mode": "fixed",
        "craft_length_strategy": "oversample",
    },
    "balanced": {
        # Good defaults for most cases (RECOMMENDED)
        # Use when: Starting a new project, unsure what settings to use
        "craft_alpha": 0.6,
        "craft_gradient_balancing": "loss_scale",
        "craft_loss_scale_momentum": 0.99,
        "craft_learnable_temperature": True,
        "craft_beta_mode": "auto",
        "craft_length_strategy": "auto_beta",
        "craft_use_hidden_state_hook": True,
    },
    "memory_efficient": {
        # For limited GPU memory
        # Use when: OOM errors, single GPU, large models
        "craft_alpha": 0.6,
        "craft_use_gradcache": True,
        "craft_gradcache_chunk_size": 4,
        "craft_use_hidden_state_hook": True,
        "craft_projection_dim": 128,  # Smaller projection
        "craft_gradient_balancing": "loss_scale",
        "craft_learnable_temperature": True,
        "craft_beta_mode": "auto",
        "craft_length_strategy": "auto_beta",
    },
    "large_batch": {
        # For 1000+ effective batch size contrastive learning
        # Use when: You have multiple GPUs, want maximum contrastive signal
        "craft_alpha": 0.5,  # Equal weight
        "craft_use_gradcache": True,
        "craft_gradcache_chunk_size": 8,
        "craft_negative_strategy": "queue",
        "craft_negative_queue_size": 65536,
        "craft_gradient_balancing": "loss_scale",
        "craft_learnable_temperature": True,
        "craft_use_hidden_state_hook": True,
        "craft_beta_mode": "auto",
        "craft_length_strategy": "auto_beta",
    },
    "aggressive": {
        # Strong contrastive signal, use with caution
        # Use when: Representation quality is critical, willing to trade SFT performance
        "craft_alpha": 0.4,  # More contrastive
        "craft_gradient_balancing": "gradnorm",
        "craft_gradnorm_alpha": 1.5,
        "craft_learnable_temperature": True,
        "craft_use_gradcache": True,
        "craft_negative_strategy": "queue",
        "craft_negative_queue_size": 32768,
        "craft_beta_mode": "auto",
        "craft_length_strategy": "auto_beta",
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.
    
    Available presets:
    - "minimal": Just add contrastive with minimal changes
    - "balanced": Good defaults for most cases (recommended)
    - "memory_efficient": For limited GPU memory
    - "large_batch": For 1000+ effective batch size
    - "aggressive": Strong contrastive signal
    
    Args:
        name: Preset name
        
    Returns:
        Dict of configuration values
        
    Raises:
        ValueError: If preset name is unknown
    """
    if name not in CRAFT_PRESETS:
        available = ", ".join(sorted(CRAFT_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return dict(CRAFT_PRESETS[name])


def auto_configure(
    model: Optional["nn.Module"] = None,
    sft_dataset: Optional[Any] = None,
    contrastive_dataset: Optional[Any] = None,
    available_memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Auto-detect optimal CRAFT configuration based on model and data.
    
    This function analyzes the provided model and datasets to determine
    sensible defaults. It's a good starting point that can be overridden.
    
    Args:
        model: The model to be trained (used to detect hidden size)
        sft_dataset: SFT dataset (used to estimate size)
        contrastive_dataset: Contrastive dataset (None for self-align)
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
        
    Returns:
        Dict of recommended configuration values
    """
    config: Dict[str, Any] = {}
    
    # Start with balanced preset as base
    config.update(CRAFT_PRESETS["balanced"])
    
    # Detect hidden size and scale projection dim
    if model is not None:
        hidden_size = _get_hidden_size(model)
        if hidden_size is not None:
            # Scale projection dim: typically hidden_size // 4, capped at 256
            config["craft_projection_dim"] = min(256, max(64, hidden_size // 4))
    
    # Detect if we have paired data or need self-align
    # (This is informational - strategy is set in make_craft_datasets)
    has_paired_data = contrastive_dataset is not None
    
    # Estimate dataset sizes for beta tuning
    sft_size = _estimate_dataset_size(sft_dataset)
    contrastive_size = _estimate_dataset_size(contrastive_dataset) if has_paired_data else sft_size
    
    if sft_size is not None and contrastive_size is not None:
        # Auto-compute beta based on dataset ratio
        total = sft_size + contrastive_size
        if total > 0:
            suggested_beta = sft_size / total
            # Clamp to reasonable range
            config["craft_beta"] = max(0.3, min(0.8, suggested_beta))
    
    # Memory optimization based on available memory
    if available_memory_gb is not None:
        if available_memory_gb < 16:
            # Low memory: enable all optimizations
            config["craft_use_gradcache"] = True
            config["craft_gradcache_chunk_size"] = 2
            config["craft_projection_dim"] = min(config.get("craft_projection_dim", 256), 128)
        elif available_memory_gb < 24:
            # Medium memory: some optimizations
            config["craft_use_gradcache"] = True
            config["craft_gradcache_chunk_size"] = 4
        # High memory: defaults are fine
    
    # For large contrastive datasets, consider queue strategy
    if contrastive_size is not None and contrastive_size > 100000:
        config["craft_negative_strategy"] = "queue"
        config["craft_negative_queue_size"] = min(65536, contrastive_size // 2)
    
    return config


def _get_hidden_size(model: "nn.Module") -> Optional[int]:
    """Extract hidden size from model config."""
    # Handle various model wrappers
    model_to_check = model
    
    # Unwrap DDP
    if hasattr(model_to_check, 'module'):
        model_to_check = model_to_check.module
    
    # Unwrap PEFT
    if hasattr(model_to_check, 'get_base_model'):
        model_to_check = model_to_check.get_base_model()
    
    # Get config
    config = getattr(model_to_check, 'config', None)
    if config is None:
        return None
    
    return getattr(config, 'hidden_size', None)


def _estimate_dataset_size(dataset: Optional[Any]) -> Optional[int]:
    """Estimate dataset size."""
    if dataset is None:
        return None
    
    try:
        return len(dataset)
    except (TypeError, AttributeError):
        return None


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
        default="assistant_masks",
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

    # -------------------------------------------------------------------------
    # Gradient Balancing (addresses gradient dominance / task imbalance)
    # -------------------------------------------------------------------------
    craft_gradient_balancing: str = field(
        default="none",
        metadata={
            "help": "Gradient balancing strategy to prevent gradient dominance: "
            "none|gradnorm|uncertainty|pcgrad|loss_scale. "
            "gradnorm: Dynamic gradient normalization (Chen et al., ICML 2018). "
            "uncertainty: Homoscedastic uncertainty weighting (Kendall et al., 2018). "
            "pcgrad: Project conflicting gradients (Yu et al., NeurIPS 2020). "
            "loss_scale: Simple loss normalization by running mean."
        },
    )
    craft_gradnorm_alpha: float = field(
        default=1.5,
        metadata={
            "help": "GradNorm asymmetry hyperparameter. Higher = stronger balancing. "
            "Typical range: 0.5-3.0. Only used if craft_gradient_balancing='gradnorm'."
        },
    )
    craft_loss_scale_momentum: float = field(
        default=0.99,
        metadata={
            "help": "Momentum for running loss mean in loss_scale balancing. "
            "Higher = more stable but slower adaptation."
        },
    )
    craft_gradient_clip_per_task: bool = field(
        default=False,
        metadata={
            "help": "Apply gradient clipping per-task before combining. "
            "Prevents any single task from dominating via extreme gradients."
        },
    )
    craft_gradient_clip_value: float = field(
        default=1.0,
        metadata={
            "help": "Max gradient norm per task when craft_gradient_clip_per_task=True."
        },
    )


@dataclass
class CRAFTSFTConfig(CRAFTConfigMixin, SFTConfig):  # type: ignore[misc]
    """Config for the CRAFT-augmented SFT trainer."""
    
    @classmethod
    def from_preset(
        cls,
        preset: str,
        output_dir: str = "./outputs",
        **overrides,
    ) -> "CRAFTSFTConfig":
        """
        Create config from a preset with optional overrides.
        
        Available presets:
        - "minimal": Just add contrastive with minimal changes
        - "balanced": Good defaults for most cases (recommended)
        - "memory_efficient": For limited GPU memory
        - "large_batch": For 1000+ effective batch size
        - "aggressive": Strong contrastive signal
        
        Example:
            config = CRAFTSFTConfig.from_preset(
                "balanced",
                output_dir="./my_outputs",
                per_device_train_batch_size=4,
            )
        
        Args:
            preset: Preset name
            output_dir: Output directory (required by TRL)
            **overrides: Additional config values to override
            
        Returns:
            Configured CRAFTSFTConfig instance
        """
        preset_values = get_preset(preset)
        preset_values.update(overrides)
        return cls(output_dir=output_dir, **preset_values)
    
    @classmethod
    def auto(
        cls,
        output_dir: str = "./outputs",
        model: Optional["nn.Module"] = None,
        sft_dataset: Optional[Any] = None,
        contrastive_dataset: Optional[Any] = None,
        available_memory_gb: Optional[float] = None,
        **overrides,
    ) -> "CRAFTSFTConfig":
        """
        Auto-detect optimal configuration based on model and data.
        
        This analyzes your model and datasets to determine sensible defaults.
        Any auto-detected values can be overridden via **overrides.
        
        Example:
            config = CRAFTSFTConfig.auto(
                output_dir="./outputs",
                model=my_model,
                sft_dataset=train_data,
                contrastive_dataset=pairs_data,
                per_device_train_batch_size=4,
            )
        
        Args:
            output_dir: Output directory (required by TRL)
            model: Model to be trained (used to detect hidden size)
            sft_dataset: SFT dataset
            contrastive_dataset: Contrastive dataset (None for self-align)
            available_memory_gb: Available GPU memory in GB
            **overrides: Additional config values to override
            
        Returns:
            Configured CRAFTSFTConfig instance
        """
        auto_values = auto_configure(
            model=model,
            sft_dataset=sft_dataset,
            contrastive_dataset=contrastive_dataset,
            available_memory_gb=available_memory_gb,
        )
        auto_values.update(overrides)
        return cls(output_dir=output_dir, **auto_values)


@dataclass
class CRAFTORPOConfig(CRAFTConfigMixin, ORPOConfig):  # type: ignore[misc]
    """Config for CRAFT ORPO trainer."""
    
    @classmethod
    def from_preset(cls, preset: str, output_dir: str = "./outputs", **overrides) -> "CRAFTORPOConfig":
        """Create config from preset. See CRAFTSFTConfig.from_preset for details."""
        preset_values = get_preset(preset)
        preset_values.update(overrides)
        return cls(output_dir=output_dir, **preset_values)
    
    @classmethod
    def auto(cls, output_dir: str = "./outputs", model=None, sft_dataset=None, 
             contrastive_dataset=None, available_memory_gb=None, **overrides) -> "CRAFTORPOConfig":
        """Auto-detect config. See CRAFTSFTConfig.auto for details."""
        auto_values = auto_configure(model=model, sft_dataset=sft_dataset,
                                     contrastive_dataset=contrastive_dataset,
                                     available_memory_gb=available_memory_gb)
        auto_values.update(overrides)
        return cls(output_dir=output_dir, **auto_values)


@dataclass
class CRAFTGRPOConfig(CRAFTConfigMixin, GRPOConfig):  # type: ignore[misc]
    """Config for CRAFT GRPO trainer."""
    
    @classmethod
    def from_preset(cls, preset: str, output_dir: str = "./outputs", **overrides) -> "CRAFTGRPOConfig":
        """Create config from preset. See CRAFTSFTConfig.from_preset for details."""
        preset_values = get_preset(preset)
        preset_values.update(overrides)
        return cls(output_dir=output_dir, **preset_values)
    
    @classmethod
    def auto(cls, output_dir: str = "./outputs", model=None, sft_dataset=None,
             contrastive_dataset=None, available_memory_gb=None, **overrides) -> "CRAFTGRPOConfig":
        """Auto-detect config. See CRAFTSFTConfig.auto for details."""
        auto_values = auto_configure(model=model, sft_dataset=sft_dataset,
                                     contrastive_dataset=contrastive_dataset,
                                     available_memory_gb=available_memory_gb)
        auto_values.update(overrides)
        return cls(output_dir=output_dir, **auto_values)


@dataclass
class CRAFTPPOConfig(CRAFTConfigMixin, PPOConfig):  # type: ignore[misc]
    """Config for CRAFT PPO trainer."""
    
    @classmethod
    def from_preset(cls, preset: str, output_dir: str = "./outputs", **overrides) -> "CRAFTPPOConfig":
        """Create config from preset. See CRAFTSFTConfig.from_preset for details."""
        preset_values = get_preset(preset)
        preset_values.update(overrides)
        return cls(output_dir=output_dir, **preset_values)
    
    @classmethod
    def auto(cls, output_dir: str = "./outputs", model=None, sft_dataset=None,
             contrastive_dataset=None, available_memory_gb=None, **overrides) -> "CRAFTPPOConfig":
        """Auto-detect config. See CRAFTSFTConfig.auto for details."""
        auto_values = auto_configure(model=model, sft_dataset=sft_dataset,
                                     contrastive_dataset=contrastive_dataset,
                                     available_memory_gb=available_memory_gb)
        auto_values.update(overrides)
        return cls(output_dir=output_dir, **auto_values)


@dataclass
class CRAFTDPOConfig(CRAFTConfigMixin, DPOConfig):  # type: ignore[misc]
    """Config for CRAFT DPO trainer."""
    
    @classmethod
    def from_preset(cls, preset: str, output_dir: str = "./outputs", **overrides) -> "CRAFTDPOConfig":
        """Create config from preset. See CRAFTSFTConfig.from_preset for details."""
        preset_values = get_preset(preset)
        preset_values.update(overrides)
        return cls(output_dir=output_dir, **preset_values)
    
    @classmethod
    def auto(cls, output_dir: str = "./outputs", model=None, sft_dataset=None,
             contrastive_dataset=None, available_memory_gb=None, **overrides) -> "CRAFTDPOConfig":
        """Auto-detect config. See CRAFTSFTConfig.auto for details."""
        auto_values = auto_configure(model=model, sft_dataset=sft_dataset,
                                     contrastive_dataset=contrastive_dataset,
                                     available_memory_gb=available_memory_gb)
        auto_values.update(overrides)
        return cls(output_dir=output_dir, **auto_values)
