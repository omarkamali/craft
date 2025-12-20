import pytest
import torch
from torch.utils.data import DataLoader

from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.trainers import CRAFTSFTTrainer

pytest.importorskip("trl", reason="CRAFT trainers require TRL")


class SFTOnlyDataset(torch.utils.data.Dataset):
    def __len__(self):  # pragma: no cover - trivial
        return 6

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([idx, idx + 1]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([idx, idx + 1]),
        }


class PairedDataset(torch.utils.data.Dataset):
    def __len__(self):  # pragma: no cover - trivial
        return 6

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([idx, idx + 1]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([idx, idx + 1]),
            "input_ids_tgt": torch.tensor([idx + 10, idx + 11]),
            "attention_mask_tgt": torch.tensor([1, 1]),
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.config.hidden_size)
        hidden[:, -1, 0] = input_ids[:, -1].float()
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5)
        outputs = type("Outputs", (), {"loss": loss, "hidden_states": (hidden,)})()
        return outputs


class MaskOptionalDataset(torch.utils.data.Dataset):
    def __init__(self, include_mask: bool):
        self.include_mask = include_mask

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        batch = {
            "input_ids": torch.tensor([idx, idx + 1]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, -100]),
        }
        if self.include_mask:
            batch["assistant_mask"] = torch.tensor([0, 1])
        return batch


def test_self_align_strategy_adds_positive_columns():
    dataset = SFTOnlyDataset()
    bundle = make_craft_datasets(dataset, strategy="self_align")

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        craft_alpha=0.5,
        craft_assistant_mask_strategy="auto",
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
    )

    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    assert "input_ids_tgt" in batch
    assert "attention_mask_tgt" in batch


def test_contrastive_batch_requires_keys():
    dataset = SFTOnlyDataset()
    bundle = make_craft_datasets(dataset, strategy="self_align")

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        craft_alpha=0.5,
        craft_assistant_mask_strategy="provided",
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
    )

    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    batch.pop("attention_mask_tgt", None)

    with pytest.raises(ValueError):
        trainer.compute_loss(trainer.model, batch)


def test_beta_ratio_cycles_batches():
    dataset = PairedDataset()
    bundle = make_craft_datasets(dataset, strategy="paired_dataset", contrastive_dataset=dataset)

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        craft_beta=0.5,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
    )

    loader = trainer.get_train_dataloader()
    pattern = []
    for _ in range(8):
        batch = next(iter(loader))
        pattern.append(batch["craft_batch_type"])
    assert pattern.count("craft") >= 2
    assert pattern.count("sft") >= 2


def test_length_strategy_error_raises_on_mismatch():
    dataset = PairedDataset()
    short_contrastive = torch.utils.data.Subset(PairedDataset(), range(2))
    bundle = make_craft_datasets(
        dataset,
        strategy="paired_dataset",
        contrastive_dataset=short_contrastive,
    )

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=1,
        craft_beta=0.5,
        craft_length_strategy="error",
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
    )

    with pytest.raises(ValueError):
        trainer.get_train_dataloader()


def test_contrastive_batch_size_override_applied():
    dataset = PairedDataset()
    bundle = make_craft_datasets(
        dataset,
        strategy="paired_dataset",
        contrastive_dataset=dataset,
    )

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        craft_beta=0.5,
        craft_contrastive_batch_size=4,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
    )

    loader = trainer.get_train_dataloader()
    for batch in loader:
        if batch["craft_batch_type"] == "craft":
            assert batch["input_ids"].shape[0] == 4
            break
    else:  # pragma: no cover - defensive
        pytest.fail("Did not encounter a contrastive batch")


def test_custom_loaders_are_respected():
    dataset = PairedDataset()
    bundle = make_craft_datasets(dataset, strategy="paired_dataset", contrastive_dataset=dataset)

    def tagged_collator(tag):
        base = CRAFTCollator()

        def _collate(features):
            batch = base(features)
            batch["collate_tag"] = tag
            return batch

        return _collate

    sft_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=tagged_collator("sft"),
    )
    contrastive_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=tagged_collator("craft"),
    )

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        craft_alpha=0.5,
        craft_beta=0.5,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        craft_sft_loader=sft_loader,
        craft_contrastive_loader=contrastive_loader,
    )

    loader = trainer.get_train_dataloader()
    seen_tags = {"sft": set(), "craft": set()}
    iterator = iter(loader)
    for _ in range(4):
        batch = next(iterator)
        batch_type = batch["craft_batch_type"]
        seen_tags[batch_type].add(batch["collate_tag"])
    assert seen_tags["sft"] == {"sft"}
    assert seen_tags["craft"] == {"craft"}


def test_self_align_validation_requires_labels_or_mask():
    dataset = MaskOptionalDataset(include_mask=False)
    bundle = make_craft_datasets(dataset, strategy="self_align")

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        craft_alpha=0.5,
        craft_assistant_mask_strategy="auto",
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
    )

    with pytest.raises(ValueError):
        trainer.get_train_dataloader()


def test_self_align_validation_accepts_assistant_mask():
    dataset = MaskOptionalDataset(include_mask=True)
    bundle = make_craft_datasets(dataset, strategy="self_align")

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        craft_alpha=0.5,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
    )

    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    assert batch["craft_batch_type"] == "sft"
