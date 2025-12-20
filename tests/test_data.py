import itertools

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from craft.data import CRAFTCollator, CRAFTMixedDataLoader, make_craft_datasets


class SimpleDataset(Dataset):
    def __init__(self, length: int, with_targets: bool = True):
        self.length = length
        self.with_targets = with_targets

    def __len__(self):  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor([idx, idx + 1]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([idx, idx + 1]),
        }
        if self.with_targets:
            item["input_ids_tgt"] = torch.tensor([idx + 10, idx + 11])
            item["attention_mask_tgt"] = torch.tensor([1, 1])
        return item


def test_make_craft_datasets_requires_contrastive_with_strategy():
    primary = SimpleDataset(4)
    bundle = make_craft_datasets(primary, strategy="paired_dataset", contrastive_dataset=primary)
    assert bundle.contrastive_dataset is primary

    with pytest.raises(ValueError):
        make_craft_datasets(primary, strategy="paired_dataset")

    with pytest.raises(ValueError):
        make_craft_datasets(primary, contrastive_dataset=primary, strategy="self_align")


def test_craft_mixed_dataloader_pattern_counts():
    primary = SimpleDataset(8)
    contrastive = SimpleDataset(8)

    base_loader = DataLoader(primary, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())
    contrastive_loader = DataLoader(contrastive, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())

    mixed = CRAFTMixedDataLoader(
        base_loader,
        contrastive_loader,
        beta=0.75,
        gradient_accumulation_steps=4,
    )

    pattern = list(itertools.islice((batch["craft_batch_type"] for batch in mixed), 8))
    assert pattern.count("sft") == 6
    assert pattern.count("craft") == 2
    # order should follow cycle of 3 SFT then 1 CRAFT
    assert pattern[:4] == ["sft", "sft", "sft", "craft"]


def test_craft_mixed_dataloader_without_contrastive():
    primary = SimpleDataset(3)
    base_loader = DataLoader(primary, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())

    mixed = CRAFTMixedDataLoader(base_loader, beta=0.2, gradient_accumulation_steps=3)
    pattern = [batch["craft_batch_type"] for batch in mixed]
    assert all(kind == "sft" for kind in pattern)


def test_length_strategy_cap_limits_batches():
    sft = SimpleDataset(5)
    craft = SimpleDataset(2)

    sft_loader = DataLoader(sft, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())
    craft_loader = DataLoader(craft, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())

    mixed = CRAFTMixedDataLoader(
        sft_loader,
        craft_loader,
        beta=0.5,
        gradient_accumulation_steps=2,
        beta_mode="fixed",
        length_strategy="cap",
        total_sft_batches=5,
        total_craft_batches=2,
    )

    batches = list(mixed)
    craft_seen = sum(batch["craft_batch_type"] == "craft" for batch in batches)
    sft_seen = sum(batch["craft_batch_type"] == "sft" for batch in batches)
    assert craft_seen == 2
    assert sft_seen == 2  # capped to the smaller dataset length


def test_length_strategy_auto_beta_recomputes_ratio():
    sft = SimpleDataset(60)
    craft = SimpleDataset(40)

    sft_loader = DataLoader(sft, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())
    craft_loader = DataLoader(craft, batch_size=1, shuffle=False, collate_fn=CRAFTCollator())

    mixed = CRAFTMixedDataLoader(
        sft_loader,
        craft_loader,
        beta=0.1,  # should be overridden by auto beta
        gradient_accumulation_steps=5,
        beta_mode="auto",
        length_strategy="auto_beta",
        total_sft_batches=60,
        total_craft_batches=40,
    )

    pattern = list(itertools.islice((batch["craft_batch_type"] for batch in mixed), 5))
    assert pattern == ["sft", "sft", "sft", "craft", "craft"]
