"""
CRAFT Trainers: Contrastive Representation Aware Fine-Tuning.

This module provides trainer mixins and wrappers that combine SFT with
contrastive learning objectives. Key features:

- Accumulation-aware loss scaling for correct gradient ratios
- Single forward pass for self-align (dual pooling)
- GradCache support for memory-efficient paired contrastive
- Hook-based hidden state capture to reduce memory usage

References:
- Raffel et al. "Exploring the Limits of Transfer Learning" (T5), 2020
  - Multi-task gradient accumulation
- Chen et al. "A Simple Framework for Contrastive Learning" (SimCLR), 2020
  - Projection head design
- Gao et al. "Scaling Deep Contrastive Learning Batch Size" (GradCache), 2021
  - Memory-efficient contrastive learning
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .accumulator import CRAFTGradientAccumulator, compute_batch_distribution
from .config import (
    CRAFTSFTConfig,
    CRAFTORPOConfig,
    CRAFTGRPOConfig,
    CRAFTPPOConfig,
    CRAFTDPOConfig,
)
from .data import CRAFTCollator, CRAFTDatasetBundle, CRAFTMixedDataLoader, make_craft_datasets
from .gradcache import CachedEmbeddingBank, GradCacheConfig, GradCacheContrastiveLoss
from .hooks import LastHiddenStateHook, get_backbone
from .losses import InfoNCELoss, pool_hidden_states
from .metrics import (
    compute_contrastive_accuracy,
    compute_representation_consistency,
    update_representation_reference,
)


logger = logging.getLogger(__name__)


class _MissingTRLTrainer:  # type: ignore[too-few-public-methods]
    def __init__(self, *_, **__):
        raise ImportError("CRAFT trainers require TRL; install craft[trl].")

    def get_train_dataloader(self):  # pragma: no cover - defensive
        raise ImportError("CRAFT trainers require TRL; install craft[trl].")

    def compute_loss(self, *_, **__):  # pragma: no cover - defensive
        raise ImportError("CRAFT trainers require TRL; install craft[trl].")

    def log(self, *_, **__):  # pragma: no cover - defensive
        raise ImportError("CRAFT trainers require TRL; install craft[trl].")


try:  # pragma: no cover - optional dependency
    from trl import SFTTrainer as _TRL_SFTTrainer

    _CRAFT_HAS_TRL = True
except ImportError:  # pragma: no cover
    _CRAFT_HAS_TRL = False
    _TRL_SFTTrainer = _MissingTRLTrainer  # type: ignore[assignment]

if _CRAFT_HAS_TRL:
    try:  # pragma: no cover
        from trl import ORPOTrainer as _TRL_ORPOTrainer
    except ImportError:  # pragma: no cover
        _TRL_ORPOTrainer = None

    try:  # pragma: no cover
        from trl import GRPOTrainer as _TRL_GRPOTrainer
    except ImportError:  # pragma: no cover
        _TRL_GRPOTrainer = None

    try:  # pragma: no cover
        from trl import PPOTrainer as _TRL_PPOTrainer
    except ImportError:  # pragma: no cover
        _TRL_PPOTrainer = None

    try:  # pragma: no cover
        from trl import DPOTrainer as _TRL_DPOTrainer
    except ImportError:  # pragma: no cover
        _TRL_DPOTrainer = None
else:  # pragma: no cover
    _TRL_ORPOTrainer = None
    _TRL_GRPOTrainer = None
    _TRL_PPOTrainer = None
    _TRL_DPOTrainer = None


class CRAFTTrainerMixin:
    """
    Mixin providing CRAFT loss + metrics for TRL trainers.

    This mixin implements:
    1. Accumulation-aware loss scaling for correct alpha:(1-alpha) ratio
    2. Single forward pass for self-align with dual pooling
    3. GradCache support for memory-efficient paired contrastive
    4. Hook-based hidden state capture

    The key improvement over naive approaches is that gradient accumulation
    is treated as the combination mechanism, not an afterthought. This ensures
    the accumulated gradient matches the desired loss ratio regardless of
    batch distribution.
    """

    craft_bundle: CRAFTDatasetBundle
    craft_collator: CRAFTCollator
    craft_loss: InfoNCELoss

    _craft_reference_embeddings: Optional[torch.Tensor]
    _craft_latest_logs: Dict[str, float]
    _craft_enable_contrastive: bool
    _craft_accumulator: Optional[CRAFTGradientAccumulator]
    _craft_hidden_hook: Optional[LastHiddenStateHook]
    _craft_negative_bank: Optional[CachedEmbeddingBank]
    
    # Loss accumulation for proper logging (not per-micro-batch)
    _craft_loss_accumulator: Dict[str, float]
    _craft_loss_counts: Dict[str, int]

    def __init__(
        self,
        *args,
        craft_bundle: Optional[CRAFTDatasetBundle] = None,
        contrastive_dataset: Optional[Any] = None,
        craft_strategy: Optional[str] = None,
        craft_collator: Optional[Any] = None,
        craft_sft_loader: Optional[DataLoader] = None,
        craft_contrastive_loader: Optional[DataLoader] = None,
        craft_loader_factory: Optional[
            Callable[["CRAFTTrainerMixin"], Tuple[DataLoader, Optional[DataLoader]]]
        ] = None,
        **kwargs,
    ) -> None:
        if craft_loader_factory is not None and (
            craft_sft_loader is not None or craft_contrastive_loader is not None
        ):
            raise ValueError(
                "Pass either craft_loader_factory or explicit craft_*_loader values, not both."
            )

        self._craft_user_bundle = craft_bundle
        self._craft_user_collator = craft_collator
        self._craft_contrastive_dataset = contrastive_dataset
        self._craft_strategy = craft_strategy

        self._craft_user_sft_loader = craft_sft_loader
        self._craft_user_contrastive_loader = craft_contrastive_loader
        self._craft_loader_factory = craft_loader_factory

        super().__init__(*args, **kwargs)
        self._craft_post_init()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _craft_post_init(self) -> None:
        """Initialize CRAFT components after parent __init__."""
        primary_dataset = self._resolve_craft_primary_dataset()

        if self._craft_user_bundle is not None:
            self.craft_bundle = self._craft_user_bundle
        else:
            strategy = self._craft_strategy
            if strategy is None:
                strategy = (
                    "paired_dataset"
                    if self._craft_contrastive_dataset is not None
                    else "self_align"
                )

            self.craft_bundle = make_craft_datasets(
                primary_dataset,
                contrastive_dataset=self._craft_contrastive_dataset,
                strategy=strategy,
            )

        # Collator for contrastive batches
        if self._craft_user_collator is not None:
            self.craft_collator = self._craft_user_collator
        else:
            self.craft_collator = CRAFTCollator()

        # Get model config
        model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_to_check, 'get_base_model'):
            model_to_check = model_to_check.get_base_model()
        hidden_size = getattr(getattr(model_to_check, "config", None), "hidden_size", None)

        # Initialize InfoNCE loss with improved projection head
        pooling = getattr(self.args, "craft_pooling", "last_token")
        projection_dim = getattr(self.args, "craft_projection_dim", 256)
        dropout = getattr(self.args, "craft_projection_dropout", 0.0)
        learnable_temp = getattr(self.args, "craft_learnable_temperature", False)

        self.craft_loss = InfoNCELoss(
            temperature=self.args.craft_temperature,
            pooling=pooling,
            hidden_size=hidden_size,
            projection_dim=projection_dim,
            learnable_temperature=learnable_temp,
            dropout=dropout,
        ).to(model_to_check.device if hasattr(model_to_check, 'device') else 'cpu')

        # Initialize state
        self._craft_reference_embeddings = None
        self._craft_latest_logs = {}
        self._craft_accumulator = None  # Initialized in get_train_dataloader
        
        # Loss accumulation for proper logging (average over logging_steps, not per-micro-batch)
        self._craft_loss_accumulator = {}
        self._craft_loss_counts = {}
        self._craft_hidden_hook = None
        self._craft_negative_bank = None

        # Check if contrastive training is enabled
        self._craft_enable_contrastive = bool(
            getattr(self.args, "craft_alpha", 1.0) < 1.0
            and (
                self.craft_bundle.contrastive_dataset is not None
                or self.craft_bundle.strategy == "self_align"
            )
        )

        # Initialize hook if enabled
        if self._craft_enable_contrastive and getattr(self.args, "craft_use_hidden_state_hook", True):
            try:
                backbone = get_backbone(self.model)
                self._craft_hidden_hook = LastHiddenStateHook(backbone)
            except Exception as e:
                logger.warning(f"Failed to attach hidden state hook: {e}. Falling back to output_hidden_states.")
                self._craft_hidden_hook = None

        # Initialize negative bank if using queue strategy
        if self._craft_enable_contrastive and getattr(self.args, "craft_negative_strategy", "in_batch") == "queue":
            queue_size = getattr(self.args, "craft_negative_queue_size", 65536)
            proj_dim = getattr(self.args, "craft_projection_dim", 256)
            self._craft_negative_bank = CachedEmbeddingBank(
                embedding_dim=proj_dim,
                bank_size=queue_size,
                device=model_to_check.device if hasattr(model_to_check, 'device') else torch.device('cpu'),
            )

    def _resolve_craft_primary_dataset(self) -> Any:
        """Resolve the primary dataset for CRAFT."""
        if getattr(self, "train_dataset", None) is not None:
            return self.train_dataset
        if getattr(self, "dataset", None) is not None:
            return self.dataset
        raise ValueError("CRAFT trainer requires a train_dataset or dataset to be provided.")

    # ------------------------------------------------------------------
    # Data Loader
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
        """Build mixed SFT/contrastive dataloader with proper accumulator."""
        base_loader: DataLoader = super().get_train_dataloader()

        if not self._craft_enable_contrastive:
            return base_loader

        sft_loader, contrastive_loader, sft_batches, craft_batches = self._craft_build_data_loaders(
            base_loader
        )

        self._craft_validate_self_align_requirements(sft_loader)

        if (
            getattr(self.args, "craft_length_strategy", "oversample") == "error"
            and sft_batches is not None
            and craft_batches is not None
            and sft_batches != craft_batches
        ):
            raise ValueError(
                "CRAFT length strategy set to 'error' but SFT and contrastive batches differ"
            )

        # Compute batch distribution for accumulator
        n_sft, n_con = compute_batch_distribution(
            self.args.craft_beta,
            self.args.gradient_accumulation_steps,
        )

        # Initialize accumulator with proper scaling
        self._craft_accumulator = CRAFTGradientAccumulator(
            alpha=self.args.craft_alpha,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            n_sft=n_sft,
            n_contrastive=n_con,
        )

        if getattr(self.args, "craft_debug", False):
            logger.info(f"CRAFT accumulator: {self._craft_accumulator}")

        return CRAFTMixedDataLoader(
            sft_loader,
            contrastive_loader,
            beta=self.args.craft_beta,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            beta_mode=getattr(self.args, "craft_beta_mode", "fixed"),
            length_strategy=getattr(self.args, "craft_length_strategy", "oversample"),
            total_sft_batches=sft_batches,
            total_craft_batches=craft_batches,
        )

    def _craft_build_data_loaders(
        self,
        base_loader: DataLoader,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[int], Optional[int]]:
        """Build SFT and contrastive data loaders."""
        contrastive_dataset = (
            self.craft_bundle.contrastive_dataset
            if self.craft_bundle.contrastive_dataset is not None
            else self.craft_bundle.sft_dataset
        )

        if self._craft_loader_factory is not None:
            produced = self._craft_loader_factory(self)
            if (
                not isinstance(produced, tuple)
                or len(produced) != 2
                or produced[0] is None
            ):
                raise ValueError("craft_loader_factory must return (sft_loader, contrastive_loader?)")
            sft_loader, contrastive_loader = produced
        else:
            sft_loader = self._craft_user_sft_loader or base_loader
            contrastive_loader = self._craft_user_contrastive_loader

            if sft_loader is None:
                raise ValueError("CRAFT requires an SFT loader when contrastive mixing is enabled.")

            if contrastive_loader is None:
                contrastive_loader = self._craft_create_default_contrastive_loader(
                    base_loader=base_loader,
                    dataset=contrastive_dataset,
                )

        if contrastive_loader is None and contrastive_dataset is not None:
            raise ValueError("Contrastive loader could not be constructed for CRAFT.")

        sft_batches = self._craft_estimate_batches_from_loader(
            sft_loader,
            dataset=self.craft_bundle.sft_dataset,
            fallback_batch_size=getattr(base_loader, "batch_size", None),
        )

        craft_batches = self._craft_estimate_batches_from_loader(
            contrastive_loader,
            dataset=contrastive_dataset,
            fallback_batch_size=(
                self.args.craft_contrastive_batch_size
                or getattr(base_loader, "batch_size", None)
            ),
        )

        return sft_loader, contrastive_loader, sft_batches, craft_batches

    def _craft_create_default_contrastive_loader(
        self,
        *,
        base_loader: DataLoader,
        dataset: Optional[Any],
    ) -> Optional[DataLoader]:
        """Create default contrastive dataloader."""
        if dataset is None:
            return None

        batch_size = getattr(self.args, "craft_contrastive_batch_size", None)
        if batch_size is None:
            batch_size = getattr(base_loader, "batch_size", None)
        if batch_size is None:
            # Fallback to training args batch size (accelerate wraps loaders without batch_size attr)
            batch_size = getattr(self.args, "per_device_train_batch_size", None)
        if batch_size is None:
            raise ValueError(
                "Unable to infer contrastive batch size; please set craft_contrastive_batch_size."
            )

        allow_packed = bool(getattr(self.args, "craft_allow_packed_contrastive", False))
        if not allow_packed and self._craft_collator_may_pack(self.craft_collator):
            raise ValueError(
                "CRAFT detected a likely packing/flattening collator for contrastive batches. "
                "This can break in-batch negatives. Either:\n"
                "  - pass an explicit craft_contrastive_loader with a non-packing collator, or\n"
                "  - pass craft_collator=<your contrastive-safe collator>, or\n"
                "  - set args.craft_allow_packed_contrastive=True to override."
            )

        sampler = self._get_train_sampler(dataset)
        shuffle = sampler is None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.craft_collator,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            drop_last=getattr(base_loader, "drop_last", False),
            persistent_workers=getattr(base_loader, "persistent_workers", False),
        )

    @staticmethod
    def _craft_collator_may_pack(collator: Any) -> bool:
        """Heuristic to detect packing collators."""
        name = type(collator).__name__.lower()
        if "flatten" in name or "packing" in name:
            return True
        if bool(getattr(collator, "return_position_ids", False)):
            return True
        if hasattr(collator, "separator_id"):
            return True
        return False

    def _craft_estimate_batches_from_loader(
        self,
        loader: Optional[DataLoader],
        *,
        dataset: Optional[Any],
        fallback_batch_size: Optional[int],
    ) -> Optional[int]:
        """Estimate number of batches in a loader."""
        if loader is None:
            return None

        try:
            return len(loader)
        except TypeError:
            pass

        batch_size = getattr(loader, "batch_size", None) or fallback_batch_size
        if dataset is None:
            return None
        return self._estimate_batches(dataset, batch_size)

    def _craft_validate_self_align_requirements(self, sft_loader: DataLoader) -> None:
        """Validate that self-align has required data."""
        if self.craft_bundle.strategy != "self_align":
            return

        keys = getattr(self.args, "craft_contrastive_keys", {}) or {}
        label_key = keys.get("anchor_labels", "labels")
        mask_key = getattr(self.args, "craft_assistant_mask_key", "assistant_masks")

        inspected_batches = 0
        for batch in self._craft_iter_sft_batches(sft_loader, limit=2):
            if not isinstance(batch, Mapping):
                continue

            inspected_batches += 1

            has_labels = self._craft_batch_has_valid_labels(batch, label_key)
            has_mask = self._craft_batch_has_assistant_mask(batch, mask_key)

            if has_labels or has_mask:
                return

        if inspected_batches == 0:
            raise ValueError(
                "CRAFT strategy='self_align' requires a readable SFT dataloader to validate assistant spans."
            )

        raise ValueError(
            "CRAFT strategy='self_align' needs either labels (with assistant tokens where labels != -100) "
            f"or an assistant mask column (key='{mask_key}') in the SFT batches."
        )

    def _craft_iter_sft_batches(self, loader: DataLoader, *, limit: int):
        """Iterate over first few batches of a loader."""
        iterator = iter(loader)
        for _ in range(limit):
            try:
                yield next(iterator)
            except StopIteration:
                break

    def _craft_batch_has_valid_labels(
        self,
        batch: Mapping[str, Any],
        label_key: Optional[str],
    ) -> bool:
        """Check if batch has valid labels for self-align."""
        if not label_key or label_key not in batch:
            return False
        tensor = self._craft_as_tensor(batch[label_key])
        if tensor is None or tensor.numel() == 0:
            return False
        return bool(tensor.ne(-100).any().item())

    def _craft_batch_has_assistant_mask(
        self,
        batch: Mapping[str, Any],
        mask_key: Optional[str],
    ) -> bool:
        """Check if batch has assistant mask."""
        if not mask_key or mask_key not in batch:
            return False
        tensor = self._craft_as_tensor(batch[mask_key])
        if tensor is None or tensor.numel() == 0:
            return False
        return bool(tensor.to(dtype=torch.bool).any().item())

    @staticmethod
    def _craft_as_tensor(value: Any) -> Optional[torch.Tensor]:
        """Convert value to tensor if possible."""
        if isinstance(value, torch.Tensor):
            return value.detach()
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Loss Computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        """
        Compute CRAFT loss with proper accumulation-aware scaling.

        For self-align strategy, this uses a single forward pass with dual
        pooling (one for anchor, one for positive based on assistant mask).
        """
        batch_type = inputs.pop("craft_batch_type", None)

        if batch_type != "craft" or not self._craft_enable_contrastive:
            return self._compute_sft_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # For self-align, use efficient single forward pass
        if self.craft_bundle.strategy == "self_align":
            return self._compute_self_align_loss(
                model,
                inputs,
                return_outputs=return_outputs,
            )

        # For paired dataset, use standard or GradCache approach
        return self._compute_paired_contrastive_loss(
            model,
            inputs,
            return_outputs=return_outputs,
        )

    def _compute_sft_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        *,
        return_outputs: bool,
        num_items_in_batch: Optional[int],
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        """Compute SFT loss with proper accumulation scaling."""
        if return_outputs:
            base_loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            base_loss = super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )
            outputs = None

        # Apply accumulation-aware scaling
        if self._craft_accumulator is not None:
            total_loss = self._craft_accumulator.scale_sft_loss(base_loss)
        else:
            total_loss = self.args.craft_alpha * base_loss

        self._log_craft_losses(sft_loss=base_loss, contrastive_loss=None)

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_self_align_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        *,
        return_outputs: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        """
        Compute combined SFT + contrastive loss with single forward pass.

        This is the key optimization for self-align: we compute both losses
        from the same forward pass, using dual pooling (full sequence for
        anchor, assistant tokens only for positive).
        """
        # Single forward pass with hidden states
        use_hook = self._craft_hidden_hook is not None and self._craft_hidden_hook.is_attached

        if use_hook:
            # Hook captures hidden state automatically
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels"),
                return_dict=True,
            )
            hidden = self._craft_hidden_hook.get()
        else:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels"),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = self._extract_last_hidden_state(outputs)

        sft_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=hidden.device)

        # Dual pooling: anchor = full sequence, positive = assistant tokens
        full_mask = inputs["attention_mask"]
        assistant_mask = self._derive_assistant_mask(inputs, full_mask)

        pooling = getattr(self.args, "craft_pooling", "last_token")
        anchor_pooled = pool_hidden_states(hidden, full_mask, pooling)
        positive_pooled = pool_hidden_states(hidden, assistant_mask, pooling)

        # Compute contrastive loss from pooled representations
        contrastive_loss = self.craft_loss.forward_from_pooled(
            anchor_pooled,
            positive_pooled,
            additional_negatives=self._get_additional_negatives(),
        )

        # Apply accumulation-aware scaling
        if self._craft_accumulator is not None:
            # For self-align, we have both losses on every step
            # Scale appropriately for combined loss
            total_loss = (
                self._craft_accumulator.scale_sft_loss(sft_loss) +
                self._craft_accumulator.scale_contrastive_loss(contrastive_loss)
            )
        else:
            total_loss = self.args.craft_alpha * sft_loss + (1 - self.args.craft_alpha) * contrastive_loss

        # Compute and log metrics
        self._compute_and_log_metrics(anchor_pooled, positive_pooled, sft_loss, contrastive_loss)

        # Update negative bank if using queue strategy
        if self._craft_negative_bank is not None:
            with torch.no_grad():
                proj_anchor = self.craft_loss._project(anchor_pooled)
                self._craft_negative_bank.enqueue(proj_anchor)

        # Clear hook state and free hidden states to prevent memory leak
        if use_hook:
            self._craft_hidden_hook.clear()
        
        # Explicitly delete hidden states reference to free memory sooner
        # This is critical when output_hidden_states=True as it holds all layer outputs
        del hidden
        if not use_hook and hasattr(outputs, 'hidden_states'):
            # Clear the hidden_states tuple to free memory
            outputs.hidden_states = None

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_paired_contrastive_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        *,
        return_outputs: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        """Compute contrastive loss for paired dataset."""
        anchor_ids, anchor_mask, positive_ids, positive_mask = self._prepare_contrastive_inputs(
            inputs
        )

        use_gradcache = getattr(self.args, "craft_use_gradcache", False)

        if use_gradcache:
            contrastive_loss = self._compute_gradcache_loss(
                anchor_ids, anchor_mask, positive_ids, positive_mask
            )
        else:
            contrastive_loss = self._compute_standard_contrastive_loss(
                anchor_ids, anchor_mask, positive_ids, positive_mask
            )

        # Apply accumulation-aware scaling
        if self._craft_accumulator is not None:
            total_loss = self._craft_accumulator.scale_contrastive_loss(contrastive_loss)
        else:
            total_loss = (1.0 - self.args.craft_alpha) * contrastive_loss

        self._log_craft_losses(sft_loss=None, contrastive_loss=contrastive_loss)

        if return_outputs:
            return total_loss, None
        return total_loss

    def _compute_standard_contrastive_loss(
        self,
        anchor_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss with standard forward passes."""
        backbone = get_backbone(self.model)
        use_hook = self._craft_hidden_hook is not None and self._craft_hidden_hook.is_attached

        # Forward passes
        if use_hook:
            backbone(
                input_ids=anchor_ids,
                attention_mask=anchor_mask,
                use_cache=False,
                return_dict=True,
            )
            anchor_h = self._craft_hidden_hook.get()
            self._craft_hidden_hook.clear()

            backbone(
                input_ids=positive_ids,
                attention_mask=positive_mask,
                use_cache=False,
                return_dict=True,
            )
            pos_h = self._craft_hidden_hook.get()
            self._craft_hidden_hook.clear()
        else:
            anchor_out = backbone(
                input_ids=anchor_ids,
                attention_mask=anchor_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
            anchor_h = self._extract_last_hidden_state(anchor_out)
            # Free hidden states immediately
            if hasattr(anchor_out, 'hidden_states'):
                anchor_out.hidden_states = None

            pos_out = backbone(
                input_ids=positive_ids,
                attention_mask=positive_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
            pos_h = self._extract_last_hidden_state(pos_out)
            # Free hidden states immediately
            if hasattr(pos_out, 'hidden_states'):
                pos_out.hidden_states = None

        # Compute loss
        need_details = bool(self.args.craft_report_metrics)

        if need_details:
            loss, details = self.craft_loss(
                anchor_h,
                pos_h,
                anchor_mask,
                positive_mask,
                return_details=True,
                additional_negatives=self._get_additional_negatives(),
            )

            self._compute_and_log_metrics_from_embeddings(
                details["anchor_embeddings"],
                details["positive_embeddings"],
                None,
                loss,
            )

            # Update negative bank
            if self._craft_negative_bank is not None:
                self._craft_negative_bank.enqueue(details["anchor_embeddings"])
        else:
            loss = self.craft_loss(
                anchor_h,
                pos_h,
                anchor_mask,
                positive_mask,
                additional_negatives=self._get_additional_negatives(),
            )

        return loss

    def _compute_gradcache_loss(
        self,
        anchor_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        positive_ids: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss using GradCache for memory efficiency."""
        backbone = get_backbone(self.model)
        chunk_size = getattr(self.args, "craft_gradcache_chunk_size", 4)
        pooling = getattr(self.args, "craft_pooling", "last_token")

        def pooling_fn(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return pool_hidden_states(hidden, mask, pooling)

        gradcache = GradCacheContrastiveLoss(
            backbone=backbone,
            projector=self.craft_loss.projector,
            pooling_fn=pooling_fn,
            config=GradCacheConfig(
                chunk_size=chunk_size,
                temperature=self.args.craft_temperature,
            ),
        )

        return gradcache(anchor_ids, anchor_mask, positive_ids, positive_mask)

    def _derive_assistant_mask(
        self,
        inputs: Dict[str, torch.Tensor],
        full_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Derive assistant token mask for self-align."""
        mask_strategy = getattr(self.args, "craft_assistant_mask_strategy", "auto")
        mask_key = getattr(self.args, "craft_assistant_mask_key", "assistant_masks")

        if mask_strategy == "provided" and mask_key in inputs:
            return inputs[mask_key].long() * full_mask

        if mask_strategy == "none":
            return full_mask

        # Auto: use labels != -100
        if "labels" in inputs:
            labels = inputs["labels"]
            return (labels != -100).long() * full_mask

        # Fallback to full mask
        return full_mask

    def _get_additional_negatives(self) -> Optional[torch.Tensor]:
        """Get additional negatives from queue if available."""
        if self._craft_negative_bank is None or self._craft_negative_bank.size == 0:
            return None
        return self._craft_negative_bank.get_negatives()

    def _prepare_contrastive_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare anchor and positive inputs from batch."""
        keys = self.args.craft_contrastive_keys

        try:
            anchor_ids = inputs[keys["anchor_input_ids"]]
            anchor_mask = inputs[keys["anchor_attention_mask"]]
        except KeyError as missing:
            raise ValueError(f"CRAFT contrastive batch missing required anchor key: {missing}") from None

        positive_id_key = keys["positive_input_ids"]
        positive_mask_key = keys["positive_attention_mask"]

        if positive_id_key not in inputs or positive_mask_key not in inputs:
            if self.craft_bundle.strategy != "self_align":
                missing = []
                if positive_id_key not in inputs:
                    missing.append(positive_id_key)
                if positive_mask_key not in inputs:
                    missing.append(positive_mask_key)
                raise ValueError("CRAFT contrastive batch missing keys: " + ", ".join(sorted(missing)))

            inputs.setdefault(positive_id_key, anchor_ids)

            mask_strategy = getattr(self.args, "craft_assistant_mask_strategy", "auto")
            if positive_mask_key not in inputs:
                if mask_strategy == "provided":
                    raise ValueError("craft_assistant_mask_strategy='provided' requires positive mask column")

                candidate_mask = anchor_mask
                if mask_strategy == "auto":
                    label_key = keys.get("anchor_labels")
                    if label_key and label_key in inputs:
                        labels = inputs[label_key]
                        candidate_mask = (labels != -100).long() * anchor_mask

                inputs[positive_mask_key] = candidate_mask.clone()

        positive_ids = inputs[positive_id_key]
        positive_mask = inputs[positive_mask_key]
        return anchor_ids, anchor_mask, positive_ids, positive_mask

    @staticmethod
    def _extract_last_hidden_state(output: Any) -> torch.Tensor:
        """Extract last hidden state from model output."""
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state

        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise AttributeError(
                "Model output missing last_hidden_state and hidden_states; "
                "enable hidden state returns via output_hidden_states=True."
            )

        if isinstance(hidden_states, (list, tuple)):
            if not hidden_states:
                raise AttributeError("hidden_states sequence is empty.")
            return hidden_states[-1]

        if isinstance(hidden_states, torch.Tensor):
            return hidden_states

        raise TypeError(
            f"Unsupported hidden_states type: {type(hidden_states)!r}"
        )

    # ------------------------------------------------------------------
    # Metrics & Logging
    # ------------------------------------------------------------------

    def _compute_and_log_metrics(
        self,
        anchor_pooled: torch.Tensor,
        positive_pooled: torch.Tensor,
        sft_loss: Optional[torch.Tensor],
        contrastive_loss: torch.Tensor,
    ) -> None:
        """Compute and log CRAFT metrics."""
        metrics: Dict[str, float] = {}  # Store floats, not tensors
        report_metrics = getattr(self.args, "craft_report_metrics", [])

        with torch.no_grad():
            # Project for metrics
            proj_anchor = self.craft_loss._project(anchor_pooled)
            proj_positive = self.craft_loss._project(positive_pooled)

            if "contrastive_accuracy" in report_metrics:
                metrics["craft_contrastive_accuracy"] = float(compute_contrastive_accuracy(
                    proj_anchor, proj_positive
                ).item())

            if "representation_consistency" in report_metrics:
                metrics["craft_representation_consistency"] = float(compute_representation_consistency(
                    proj_anchor,
                    self._craft_reference_embeddings,
                ).item())

            if "temperature" in report_metrics:
                metrics["craft_temperature"] = float(self.craft_loss.temperature.item())

            # Update reference embeddings - move to CPU and detach fully
            self._craft_reference_embeddings = update_representation_reference(
                self._craft_reference_embeddings,
                proj_anchor.detach().cpu(),
            )

        self._log_craft_losses(
            sft_loss=sft_loss,
            contrastive_loss=contrastive_loss,
            metrics=metrics,
        )

    def _compute_and_log_metrics_from_embeddings(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        sft_loss: Optional[torch.Tensor],
        contrastive_loss: torch.Tensor,
    ) -> None:
        """Compute metrics from already-projected embeddings."""
        metrics: Dict[str, float] = {}  # Store floats, not tensors
        report_metrics = getattr(self.args, "craft_report_metrics", [])

        with torch.no_grad():
            if "contrastive_accuracy" in report_metrics:
                metrics["craft_contrastive_accuracy"] = float(compute_contrastive_accuracy(
                    anchor_emb, positive_emb
                ).item())

            if "representation_consistency" in report_metrics:
                metrics["craft_representation_consistency"] = float(compute_representation_consistency(
                    anchor_emb,
                    self._craft_reference_embeddings,
                ).item())

            if "temperature" in report_metrics:
                metrics["craft_temperature"] = float(self.craft_loss.temperature.item())

            self._craft_reference_embeddings = update_representation_reference(
                self._craft_reference_embeddings,
                anchor_emb.detach().cpu(),
            )

        self._log_craft_losses(
            sft_loss=sft_loss,
            contrastive_loss=contrastive_loss,
            metrics=metrics,
        )

    def _log_craft_losses(
        self,
        *,
        sft_loss: Optional[torch.Tensor],
        contrastive_loss: Optional[torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Accumulate CRAFT losses for proper logging.
        
        Instead of logging per-micro-batch (which causes wild fluctuations),
        we accumulate losses and only log averaged values when the trainer
        logs its main loss. This matches the behavior of the built-in loss
        logging in Hugging Face Trainer.
        """
        # Accumulate losses (not log immediately)
        if sft_loss is not None:
            val = float(sft_loss.detach().mean().item())
            self._craft_loss_accumulator["loss/craft_sft"] = (
                self._craft_loss_accumulator.get("loss/craft_sft", 0.0) + val
            )
            self._craft_loss_counts["loss/craft_sft"] = (
                self._craft_loss_counts.get("loss/craft_sft", 0) + 1
            )
            
        if contrastive_loss is not None:
            val = float(contrastive_loss.detach().mean().item())
            self._craft_loss_accumulator["loss/craft_contrast"] = (
                self._craft_loss_accumulator.get("loss/craft_contrast", 0.0) + val
            )
            self._craft_loss_counts["loss/craft_contrast"] = (
                self._craft_loss_counts.get("loss/craft_contrast", 0) + 1
            )

        # Accumulate additional metrics
        if metrics:
            for name, value in metrics.items():
                key = f"metrics/{name}"
                if isinstance(value, torch.Tensor):
                    value = float(value.detach().mean().item())
                self._craft_loss_accumulator[key] = (
                    self._craft_loss_accumulator.get(key, 0.0) + float(value)
                )
                self._craft_loss_counts[key] = (
                    self._craft_loss_counts.get(key, 0) + 1
                )

    def _flush_craft_logs(self) -> Dict[str, float]:
        """
        Compute averaged losses and reset accumulators.
        
        Called by the overridden log() method when the trainer logs.
        """
        logs: Dict[str, float] = {}
        
        for key, total in self._craft_loss_accumulator.items():
            count = self._craft_loss_counts.get(key, 1)
            if count > 0:
                logs[key] = total / count
        
        # Compute weighted total if we have both losses
        if "loss/craft_sft" in logs or "loss/craft_contrast" in logs:
            total = 0.0
            if "loss/craft_sft" in logs:
                total += self.args.craft_alpha * logs["loss/craft_sft"]
            if "loss/craft_contrast" in logs:
                total += (1.0 - self.args.craft_alpha) * logs["loss/craft_contrast"]
            logs["loss/craft_total"] = total
        
        # Store for craft_metrics property
        self._craft_latest_logs = logs.copy()
        
        # Reset accumulators
        self._craft_loss_accumulator = {}
        self._craft_loss_counts = {}
        
        return logs

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log to inject accumulated CRAFT metrics.
        
        This ensures CRAFT losses are logged at the same frequency as the
        main training loss, properly averaged over the logging window.
        """
        # Only flush CRAFT logs when the trainer is logging its main loss
        # (indicated by presence of 'loss' key, which is the averaged training loss)
        if "loss" in logs and self._craft_loss_accumulator:
            craft_logs = self._flush_craft_logs()
            logs.update(craft_logs)
        
        super().log(logs, start_time)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def craft_metrics(self) -> Dict[str, float]:
        """Return latest CRAFT metrics."""
        return dict(self._craft_latest_logs)

    @staticmethod
    def _estimate_batches(dataset: Any, batch_size: Optional[int]) -> Optional[int]:
        """Estimate number of batches from dataset and batch size."""
        try:
            length = len(dataset)
        except TypeError:
            return None
        if length is None:
            return None
        if not batch_size or batch_size <= 0:
            return None
        return math.ceil(length / batch_size)


# ------------------------------------------------------------------
# Trainer Classes
# ------------------------------------------------------------------

class CRAFTSFTTrainer(CRAFTTrainerMixin, _TRL_SFTTrainer):
    args: CRAFTSFTConfig  # type: ignore[assignment]


if _TRL_ORPOTrainer is not None:

    class CRAFTORPOTrainer(CRAFTTrainerMixin, _TRL_ORPOTrainer):
        args: CRAFTORPOConfig  # type: ignore[assignment]

else:  # pragma: no cover

    class CRAFTORPOTrainer:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise ImportError("ORPOTrainer is unavailable. Install a newer TRL version.")


if _TRL_GRPOTrainer is not None:

    class CRAFTGRPOTrainer(CRAFTTrainerMixin, _TRL_GRPOTrainer):
        args: CRAFTGRPOConfig  # type: ignore[assignment]

else:  # pragma: no cover

    class CRAFTGRPOTrainer:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise ImportError("GRPOTrainer is unavailable. Install a newer TRL version.")


if _TRL_PPOTrainer is not None:

    class CRAFTPPOTrainer(CRAFTTrainerMixin, _TRL_PPOTrainer):
        args: CRAFTPPOConfig  # type: ignore[assignment]

else:  # pragma: no cover

    class CRAFTPPOTrainer:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise ImportError("PPOTrainer is unavailable. Install a newer TRL version.")


if _TRL_DPOTrainer is not None:

    class CRAFTDPOTrainer(CRAFTTrainerMixin, _TRL_DPOTrainer):
        args: CRAFTDPOConfig  # type: ignore[assignment]

else:  # pragma: no cover

    class CRAFTDPOTrainer:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise ImportError("DPOTrainer is unavailable. Install a newer TRL version.")
