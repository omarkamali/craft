from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .config import (
    CRAFTSFTConfig,
    CRAFTORPOConfig,
    CRAFTGRPOConfig,
    CRAFTPPOConfig,
    CRAFTDPOConfig,
)
from .data import CRAFTCollator, CRAFTDatasetBundle, CRAFTMixedDataLoader, make_craft_datasets
from .losses import InfoNCELoss
from .metrics import (
    compute_contrastive_accuracy,
    compute_representation_consistency,
    update_representation_reference,
)



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
    """Mixin providing CRAFT loss + metrics for TRL trainers."""

    craft_bundle: CRAFTDatasetBundle
    craft_collator: CRAFTCollator
    craft_loss: InfoNCELoss

    _craft_reference_embeddings: Optional[torch.Tensor]
    _craft_latest_logs: Dict[str, float]
    _craft_enable_contrastive: bool

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
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _craft_post_init(self) -> None:
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

        # FIX 1: Do NOT default craft_collator to self.data_collator.
        # self.data_collator is typically tuned for SFT (packing/flattening) and must not
        # be accidentally used for contrastive batches when we auto-build contrastive_loader.
        if self._craft_user_collator is not None:
            self.craft_collator = self._craft_user_collator
        else:
            self.craft_collator = CRAFTCollator()

        pooling = getattr(self.args, "craft_pooling", "last_token")
        # Handle DDP-wrapped models for config access
        model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
        hidden_size = getattr(getattr(model_to_check, "config", None), "hidden_size", None)

        self.craft_loss = InfoNCELoss(
            temperature=self.args.craft_temperature,
            pooling=pooling,
            hidden_size=hidden_size,
        ).to(model_to_check.device)

        self._craft_reference_embeddings = None
        self._craft_latest_logs = {}

        self._craft_enable_contrastive = bool(
            getattr(self.args, "craft_alpha", 1.0) < 1.0
            and (
                self.craft_bundle.contrastive_dataset is not None
                or self.craft_bundle.strategy == "self_align"
            )
        )

    def _resolve_craft_primary_dataset(self) -> Any:
        if getattr(self, "train_dataset", None) is not None:
            return self.train_dataset
        if getattr(self, "dataset", None) is not None:
            return self.dataset
        raise ValueError("CRAFT trainer requires a train_dataset or dataset to be provided.")

    # ------------------------------------------------------------------
    # Data loader integration
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
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
                # auto-build contrastive loader; apply packing-guard here
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

    # FIX 2: Fail-fast against likely "packing/flattening" collators when auto-building
    # the contrastive loader (unless user opts in).
    def _craft_create_default_contrastive_loader(
        self,
        *,
        base_loader: DataLoader,
        dataset: Optional[Any],
    ) -> Optional[DataLoader]:
        if dataset is None:
            return None

        batch_size = getattr(self.args, "craft_contrastive_batch_size", None)
        if batch_size is None:
            batch_size = getattr(base_loader, "batch_size", None)
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
        """
        Heuristic guard: catch common packing/flattening collators (e.g., DataCollatorWithFlattening)
        without importing transformers as a hard dependency.
        """
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
        if self.craft_bundle.strategy != "self_align":
            return

        keys = getattr(self.args, "craft_contrastive_keys", {}) or {}
        label_key = keys.get("anchor_labels", "labels")
        mask_key = getattr(self.args, "craft_assistant_mask_key", "assistant_mask")

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
                "CRAFT strategy='self_align' requires a readable SFT dataloader to validate assistant spans. "
                "Ensure the dataset implements __len__/__iter__ or supply craft_sft_loader/craft_loader_factory."
            )

        raise ValueError(
            "CRAFT strategy='self_align' needs either labels (with assistant tokens where labels != -100) "
            f"or an assistant mask column (key='{mask_key}') in the SFT batches. "
            "Add one of these fields or disable self_align."
        )

    def _craft_iter_sft_batches(self, loader: DataLoader, *, limit: int):
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
        if not mask_key or mask_key not in batch:
            return False
        tensor = self._craft_as_tensor(batch[mask_key])
        if tensor is None or tensor.numel() == 0:
            return False
        return bool(tensor.to(dtype=torch.bool).any().item())

    @staticmethod
    def _craft_as_tensor(value: Any) -> Optional[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return value.detach()
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        batch_type = inputs.pop("craft_batch_type", None)
        if batch_type != "craft" or not self._craft_enable_contrastive:
            return self._compute_craft_sft_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        return self._compute_craft_contrastive_loss(
            model,
            inputs,
            return_outputs=return_outputs,
        )

    def _compute_craft_sft_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        *,
        return_outputs: bool,
        num_items_in_batch: Optional[int],
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
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

        total_loss = self.args.craft_alpha * base_loss
        self._log_craft_losses(sft_loss=base_loss, contrastive_loss=None)

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_craft_contrastive_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        *,
        return_outputs: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        anchor_ids, anchor_mask, positive_ids, positive_mask = self._prepare_contrastive_inputs(
            inputs
        )

        # Debug prints before backbone forwards
        if self.is_world_process_zero():
            torch.cuda.reset_peak_memory_stats()
            print(f"[CRAFT][contrastive] anchor_ids: shape={tuple(anchor_ids.shape)} dtype={anchor_ids.dtype} device={anchor_ids.device}")
            print(f"[CRAFT][contrastive] pos_ids:    shape={tuple(positive_ids.shape)} dtype={positive_ids.dtype} device={positive_ids.device}")
            print(f"[CRAFT][contrastive] mem(before): alloc={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

        # Use backbone-only forwards to bypass LM head
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()
        else:
            base = model

        # Handle DDP-wrapped models
        if hasattr(base, 'module') and hasattr(base, 'model'):
            # DDP wrapper around a model with .model attribute
            backbone = base.module.model
        elif hasattr(base, 'module'):
            # DDP wrapper around a model without .model attribute
            backbone = base.module
        else:
            # Non-DDP model
            backbone = base.model  # HF CausalLM backbone (no lm_head)

        anchor_out = backbone(
            input_ids=anchor_ids,
            attention_mask=anchor_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        pos_out = backbone(
            input_ids=positive_ids,
            attention_mask=positive_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        anchor_h = self._extract_last_hidden_state(anchor_out)
        pos_h = self._extract_last_hidden_state(pos_out)

        # Debug prints after backbone forwards
        if self.is_world_process_zero():
            print(f"[CRAFT][contrastive] anchor_h: shape={tuple(anchor_h.shape)} dtype={anchor_h.dtype}")
            print(f"[CRAFT][contrastive] pos_h:    shape={tuple(pos_h.shape)} dtype={pos_h.dtype}")
            print(f"[CRAFT][contrastive] mem(after fwd): alloc={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            print(f"[CRAFT][contrastive] anchor_out has logits? {hasattr(anchor_out, 'logits')}")

        # Only request details if we need them for metrics
        need_details = (
            "contrastive_accuracy" in self.args.craft_report_metrics
            or "representation_consistency" in self.args.craft_report_metrics
        )
        
        if need_details:
            info_loss, details = self.craft_loss(
                anchor_h,
                pos_h,
                anchor_mask,
                positive_mask,
                return_details=True,
            )
        else:
            info_loss = self.craft_loss(
                anchor_h,
                pos_h,
                anchor_mask,
                positive_mask,
                return_details=False,
            )
            details = None

        total_loss = (1.0 - self.args.craft_alpha) * info_loss

        metrics: Dict[str, torch.Tensor] = {}
        
        if details is not None:
            anchor_emb = details["anchor_embeddings"].detach()
            positive_emb = details["positive_embeddings"].detach()

            if "contrastive_accuracy" in self.args.craft_report_metrics:
                metrics["craft_contrastive_accuracy"] = compute_contrastive_accuracy(
                    anchor_emb, positive_emb
                )

            if "representation_consistency" in self.args.craft_report_metrics:
                metrics["craft_representation_consistency"] = compute_representation_consistency(
                    anchor_emb,
                    self._craft_reference_embeddings,
                )

            # Update reference using detached embeddings - use mean-pooled vector
            self._craft_reference_embeddings = update_representation_reference(
                self._craft_reference_embeddings,
                anchor_emb.cpu(),  # Already detached, move to CPU for storage
            )

        # Clean up large tensors to free memory
        del anchor_out, pos_out, anchor_h, pos_h
        if details is not None:
            del anchor_emb, positive_emb

        self._log_craft_losses(
            sft_loss=None,
            contrastive_loss=info_loss,
            metrics=metrics,
        )

        if return_outputs:
            return total_loss, None
        return total_loss

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_craft_losses(
        self,
        *,
        sft_loss: Optional[torch.Tensor],
        contrastive_loss: Optional[torch.Tensor],
        metrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        logs: Dict[str, float] = {}

        if sft_loss is not None:
            logs["loss/craft_sft"] = float(sft_loss.detach().mean())
        if contrastive_loss is not None:
            logs["loss/craft_contrast"] = float(contrastive_loss.detach().mean())

        if logs:
            total = 0.0
            if sft_loss is not None:
                total += self.args.craft_alpha * float(sft_loss.detach().mean())
            if contrastive_loss is not None:
                total += (1.0 - self.args.craft_alpha) * float(contrastive_loss.detach().mean())
            logs["loss/craft_total"] = total

        if metrics:
            for name, value in metrics.items():
                logs[f"metrics/{name}"] = float(value.detach().mean())

        if logs:
            self._craft_latest_logs = logs
            self.log(logs)

    def _prepare_contrastive_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """
        Return the final hidden state from transformer outputs, even when the
        object lacks a dedicated ``last_hidden_state`` attribute (e.g. some
        ``CausalLMOutputWithPast`` instances when using headless backbones).
        """
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
            f"Unsupported hidden_states type: {type(hidden_states)!r}; expected Tensor or sequence of Tensors."
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def craft_metrics(self) -> Dict[str, float]:
        """Return latest CRAFT metrics that were logged."""
        return dict(self._craft_latest_logs)

    @staticmethod
    def _estimate_batches(dataset: Any, batch_size: Optional[int]) -> Optional[int]:
        try:
            length = len(dataset)
        except TypeError:
            return None
        if length is None:
            return None
        if not batch_size or batch_size <= 0:
            return None
        return math.ceil(length / batch_size)


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
