"""CRAFT (Contrastive Representation Aware Fine-Tuning) toolkit."""

from .losses import InfoNCELoss, ProjectionHead, pool_hidden_states
from .config import (
    CRAFTConfigMixin,
    CRAFTSFTConfig,
    CRAFTORPOConfig,
    CRAFTGRPOConfig,
    CRAFTPPOConfig,
    CRAFTDPOConfig,
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
]

__version__ = "0.3.1"
