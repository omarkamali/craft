"""CRAFT (Contrastive Representation Aware Fine-Tuning) toolkit."""

from .losses import InfoNCELoss
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
)

__all__ = [
    "InfoNCELoss",
    "CRAFTConfigMixin",
    "CRAFTSFTConfig",
    "CRAFTORPOConfig",
    "CRAFTGRPOConfig",
    "CRAFTPPOConfig",
    "CRAFTDPOConfig",
    "CRAFTDatasetBundle",
    "CRAFTCollator",
    "CRAFTMixedDataLoader",
    "make_craft_datasets",
    "compute_contrastive_accuracy",
    "compute_representation_consistency",
    "update_representation_reference",
    "CRAFTSFTTrainer",
    "CRAFTORPOTrainer",
    "CRAFTGRPOTrainer",
    "CRAFTPPOTrainer",
    "CRAFTDPOTrainer",
]

__version__ = "0.2.2"
