from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

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

try:  # pragma: no cover - optional dependency
    from trl import SFTTrainer as _TRL_SFTTrainer
    _CRAFT_HAS_TRL = True
except ImportError:  # pragma: no cover
    _CRAFT_HAS_TRL = False

    class _MissingTRLTrainer:  # type: ignore[too-few-public-methods]
        def __init__(self, *_, **__):
            raise ImportError("CRAFT trainers require TRL; install craft[trl].")

        def get_train_dataloader(self):  # pragma: no cover - defensive
            raise ImportError("CRAFT trainers require TRL; install craft[trl].")

        def compute_loss(self, *_, **__):  # pragma: no cover - defensive
            raise ImportError("CRAFT trainers require TRL; install craft[trl].")

        def log(self, *_, **__):  # pragma: no cover - defensive
            raise ImportError("CRAFT trainers require TRL; install craft[trl].")

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
        craft_collator: Optional[CRAFTCollator] = None,
        **kwargs,
    ) -> None:
        self._craft_user_bundle = craft_bundle
        self._craft_user_collator = craft_collator
        self._craft_contrastive_dataset = contrastive_dataset
        self._craft_strategy = craft_strategy
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

        if self._craft_user_collator is not None:
            self.craft_collator = self._craft_user_collator
        else:
            collator = getattr(self, "data_collator", None)
            self.craft_collator = collator or CRAFTCollator()

        pooling = getattr(self.args, "craft_pooling", "last_token")
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
        self.craft_loss = InfoNCELoss(
            temperature=self.args.craft_temperature,
            pooling=pooling,
            hidden_size=hidden_size,
        ).to(self.model.device)

        self._craft_reference_embeddings = None
        self._craft_latest_logs = {}
        self._craft_enable_contrastive = (
            self.args.craft_alpha < 1.0
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
        raise ValueError(
            "CRAFT trainer requires a train_dataset or dataset to be provided."
        )

    # ------------------------------------------------------------------
    # Data loader integration
    # ------------------------------------------------------------------
    def get_train_dataloader(self) -> DataLoader:
        base_loader: DataLoader = super().get_train_dataloader()
        if not self._craft_enable_contrastive:
            return base_loader

        contrastive_dataset = (
            self.craft_bundle.contrastive_dataset
            if self.craft_bundle.contrastive_dataset is not None
            else self.craft_bundle.sft_dataset
        )

        contrastive_batch_size = (
            self.args.craft_contrastive_batch_size
            if getattr(self.args, "craft_contrastive_batch_size", None) is not None
            else base_loader.batch_size
        )
        contrastive_loader = DataLoader(
            contrastive_dataset,
            batch_size=contrastive_batch_size,
            shuffle=True,
            collate_fn=self.craft_collator,
            num_workers=getattr(base_loader, "num_workers", 0),
            pin_memory=getattr(base_loader, "pin_memory", False),
            drop_last=getattr(base_loader, "drop_last", False),
            persistent_workers=getattr(base_loader, "persistent_workers", False),
        )

        sft_batches = self._estimate_batches(self.craft_bundle.sft_dataset, base_loader.batch_size)
        craft_batches = None
        if contrastive_dataset is not None:
            craft_batches = self._estimate_batches(
                contrastive_dataset,
                contrastive_batch_size if contrastive_batch_size else base_loader.batch_size,
            )

        if (
            self.args.craft_length_strategy == "error"
            and sft_batches is not None
            and craft_batches is not None
            and sft_batches != craft_batches
        ):
            raise ValueError(
                "CRAFT length strategy set to 'error' but SFT and contrastive batches differ"
            )

        return CRAFTMixedDataLoader(
            base_loader,
            contrastive_loader,
            beta=self.args.craft_beta,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            beta_mode=getattr(self.args, "craft_beta_mode", "fixed"),
            length_strategy=getattr(self.args, "craft_length_strategy", "oversample"),
            total_sft_batches=sft_batches,
            total_craft_batches=craft_batches,
        )

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
        base_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

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
        anchor_ids, anchor_mask, positive_ids, positive_mask = self._prepare_contrastive_inputs(inputs)

        anchor_outputs = model(
            input_ids=anchor_ids,
            attention_mask=anchor_mask,
            output_hidden_states=True,
        )
        positive_outputs = model(
            input_ids=positive_ids,
            attention_mask=positive_mask,
            output_hidden_states=True,
        )

        info_loss, details = self.craft_loss(
            anchor_outputs.hidden_states[-1],
            positive_outputs.hidden_states[-1],
            anchor_mask,
            positive_mask,
            return_details=True,
        )
        total_loss = (1.0 - self.args.craft_alpha) * info_loss

        metrics: Dict[str, torch.Tensor] = {}
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
            self._craft_reference_embeddings = update_representation_reference(
                self._craft_reference_embeddings,
                anchor_emb,
            )

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
                total += (1.0 - self.args.craft_alpha) * float(
                    contrastive_loss.detach().mean()
                )
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
            raise ValueError(
                f"CRAFT contrastive batch missing required anchor key: {missing}"
            ) from None

        # Handle self-align strategy by synthesising positive columns when absent.
        positive_id_key = keys["positive_input_ids"]
        positive_mask_key = keys["positive_attention_mask"]

        if positive_id_key not in inputs or positive_mask_key not in inputs:
            if self.craft_bundle.strategy != "self_align":
                missing = []
                if positive_id_key not in inputs:
                    missing.append(positive_id_key)
                if positive_mask_key not in inputs:
                    missing.append(positive_mask_key)
                raise ValueError(
                    "CRAFT contrastive batch missing keys: " + ", ".join(sorted(missing))
                )

            inputs.setdefault(positive_id_key, anchor_ids)

            mask_strategy = getattr(self.args, "craft_assistant_mask_strategy", "auto")
            if positive_mask_key not in inputs:
                if mask_strategy == "provided":
                    raise ValueError(
                        "craft_assistant_mask_strategy='provided' requires positive mask column"
                    )

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
