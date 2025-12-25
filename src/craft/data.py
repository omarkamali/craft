from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


@dataclass
class CRAFTDatasetBundle:
    """Description of datasets involved in CRAFT training."""

    sft_dataset: Any
    contrastive_dataset: Optional[Any]
    strategy: str


def make_craft_datasets(
    primary_dataset: Any,
    contrastive_dataset: Optional[Any] = None,
    strategy: str = "paired_dataset",
) -> CRAFTDatasetBundle:
    """Return a bundle describing SFT and contrastive datasets."""

    if strategy not in {"paired_dataset", "self_align"}:
        raise ValueError("strategy must be 'paired_dataset' or 'self_align'")

    if strategy == "paired_dataset" and contrastive_dataset is None:
        raise ValueError(
            "contrastive_dataset must be provided when strategy='paired_dataset'."
        )

    if strategy == "self_align" and contrastive_dataset is not None:
        raise ValueError(
            "contrastive_dataset should be None when strategy='self_align'."
        )

    return CRAFTDatasetBundle(
        sft_dataset=primary_dataset,
        contrastive_dataset=contrastive_dataset,
        strategy=strategy,
    )


class CRAFTCollator:
    """Default collator for CRAFT batches."""

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        
        # Collect all keys
        batch = {}
        for key in features[0].keys():
            values = [f[key] for f in features]
            # Stack tensors or convert lists to tensors
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], (list, tuple)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return batch


class CRAFTMixedDataLoader:
    """Yield SFT/InfoNCE batches according to beta ratio within accumulation window."""

    def __init__(
        self,
        sft_loader: DataLoader,
        contrastive_loader: Optional[DataLoader] = None,
        *,
        beta: float = 0.6,
        gradient_accumulation_steps: int = 1,
        beta_mode: str = "fixed",
        length_strategy: str = "oversample",
        total_sft_batches: Optional[int] = None,
        total_craft_batches: Optional[int] = None,
    ) -> None:
        self.sft_loader = sft_loader
        self.contrastive_loader = contrastive_loader
        self.beta = float(min(max(beta, 0.0), 1.0))
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.beta_mode = beta_mode
        self.length_strategy = length_strategy
        self.total_sft_batches = total_sft_batches
        self.total_craft_batches = total_craft_batches

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        sft_iter = iter(self.sft_loader)
        contrastive_iter: Optional[Iterator] = None
        if self.contrastive_loader is not None:
            contrastive_iter = iter(self.contrastive_loader)

        has_contrastive = contrastive_iter is not None
        pattern = self._build_cycle_pattern(
            has_contrastive=has_contrastive,
            beta=self._resolve_beta(has_contrastive),
        )

        limit_batches: Optional[int] = None
        if self.length_strategy in {"cap", "auto_beta"}:
            if self.total_sft_batches is not None and self.total_craft_batches is not None:
                limit_batches = min(self.total_sft_batches, self.total_craft_batches)

        sft_emitted = 0
        craft_emitted = 0

        for batch_type in cycle(pattern):
            if batch_type == "sft":
                if limit_batches is not None and sft_emitted >= limit_batches:
                    return
                try:
                    batch = next(sft_iter)
                except StopIteration:
                    return
                batch = dict(batch)
                batch["craft_batch_type"] = "sft"
                sft_emitted += 1
                yield batch
            else:
                if contrastive_iter is None:
                    continue
                if limit_batches is not None and craft_emitted >= limit_batches:
                    return
                try:
                    batch = next(contrastive_iter)
                except StopIteration:
                    if self.length_strategy == "oversample":
                        contrastive_iter = iter(self.contrastive_loader)
                        batch = next(contrastive_iter)
                    else:
                        return
                batch = dict(batch)
                batch["craft_batch_type"] = "craft"
                craft_emitted += 1
                yield batch

    def __len__(self) -> int:
        return len(self.sft_loader)

    def _resolve_beta(self, has_contrastive: bool) -> float:
        if not has_contrastive:
            return 1.0
        if self.beta_mode == "auto" or self.length_strategy == "auto_beta":
            if (
                self.total_sft_batches is not None
                and self.total_craft_batches is not None
                and (self.total_sft_batches + self.total_craft_batches) > 0
            ):
                return self.total_sft_batches / (
                    self.total_sft_batches + self.total_craft_batches
                )
        return self.beta

    def _build_cycle_pattern(self, *, has_contrastive: bool, beta: float) -> List[str]:
        if not has_contrastive:
            return ["sft"]

        total = self.gradient_accumulation_steps
        effective_beta = float(min(max(beta, 0.0), 1.0))
        sft_steps = int(round(effective_beta * total))
        sft_steps = max(0, min(total, sft_steps))
        craft_steps = total - sft_steps

        if sft_steps == 0 and effective_beta > 0:
            sft_steps = 1
            craft_steps = max(0, total - sft_steps)
        if craft_steps == 0 and effective_beta < 1:
            craft_steps = 1
            sft_steps = max(0, total - craft_steps)

        pattern: List[str] = ["sft"] * sft_steps
        pattern.extend(["craft"] * craft_steps)

        if not pattern:
            pattern = ["sft"]

        return pattern
