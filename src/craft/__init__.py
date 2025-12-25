"""CRAFT (Contrastive Representation Aware Fine-Tuning) toolkit."""

from .losses import InfoNCELoss, ProjectionHead, pool_hidden_states
from .config import (
    CRAFTConfigMixin,
    CRAFTSFTConfig,
    CRAFTORPOConfig,
    CRAFTGRPOConfig,
    CRAFTPPOConfig,
    CRAFTDPOConfig,
    CRAFT_PRESETS,
    get_preset,
    auto_configure,
)
from .data import (
    CRAFTDatasetBundle,
    CRAFTCollator,
    CRAFTMixedDataLoader,
    make_craft_datasets,
)
from .metrics import (
    compute_contrastive_accuracy,
    compute_representation_consistency,
    update_representation_reference,
)
from .trainers import (
    CRAFTSFTTrainer,
    CRAFTORPOTrainer,
    CRAFTGRPOTrainer,
    CRAFTPPOTrainer,
    CRAFTDPOTrainer,
    CRAFTTrainerMixin,
)
from .accumulator import (
    CRAFTGradientAccumulator,
    AccumulationScales,
    compute_batch_distribution,
)
from .hooks import (
    LastHiddenStateHook,
    get_backbone,
)
from .gradcache import (
    GradCacheContrastiveLoss,
    GradCacheConfig,
    CachedEmbeddingBank,
)
from .gradient_balancing import (
    GradientBalancer,
    LossScaleBalancer,
    UncertaintyWeightingBalancer,
    GradNormBalancer,
    PCGradBalancer,
    create_gradient_balancer,
)

__all__ = [
    # Losses
    "InfoNCELoss",
    "ProjectionHead",
    "pool_hidden_states",
    # Config
    "CRAFTConfigMixin",
    "CRAFTSFTConfig",
    "CRAFTORPOConfig",
    "CRAFTGRPOConfig",
    "CRAFTPPOConfig",
    "CRAFTDPOConfig",
    "CRAFT_PRESETS",
    "get_preset",
    "auto_configure",
    # Data
    "CRAFTDatasetBundle",
    "CRAFTCollator",
    "CRAFTMixedDataLoader",
    "make_craft_datasets",
    # Metrics
    "compute_contrastive_accuracy",
    "compute_representation_consistency",
    "update_representation_reference",
    # Trainers
    "CRAFTSFTTrainer",
    "CRAFTORPOTrainer",
    "CRAFTGRPOTrainer",
    "CRAFTPPOTrainer",
    "CRAFTDPOTrainer",
    "CRAFTTrainerMixin",
    # Accumulator
    "CRAFTGradientAccumulator",
    "AccumulationScales",
    "compute_batch_distribution",
    # Hooks
    "LastHiddenStateHook",
    "get_backbone",
    # GradCache
    "GradCacheContrastiveLoss",
    "GradCacheConfig",
    "CachedEmbeddingBank",
    # Gradient Balancing
    "GradientBalancer",
    "LossScaleBalancer",
    "UncertaintyWeightingBalancer",
    "GradNormBalancer",
    "PCGradBalancer",
    "create_gradient_balancer",
]

__version__ = "0.4.2"
