import pytest
import torch
from torch.utils.data import DataLoader

from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.losses import InfoNCELoss
from craft.trainers import CRAFTSFTTrainer

pytest.importorskip("trl", reason="CRAFT trainers require TRL")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, include_targets: bool = True):
        self.include_targets = include_targets

    def __len__(self):  # pragma: no cover - constant length
        return 10

    def __getitem__(self, idx):
        data = {
            "input_ids": torch.tensor([idx, idx + 1]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([idx, idx + 1]),
        }
        if self.include_targets:
            data["input_ids_tgt"] = torch.tensor([idx + 1, idx + 2])
            data["attention_mask_tgt"] = torch.tensor([1, 1])
        return data


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.config.hidden_size)
        # simple deterministic hidden states
        hidden[:, -1, 0] = input_ids[:, -1].float()
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5)
        outputs = type("Outputs", (), {"loss": loss, "hidden_states": (hidden,)})()
        return outputs


def test_craft_trainer_self_align_generates_positive_mask():
    dataset = DummyDataset(include_targets=False)
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


def test_craft_trainer_logs_metrics(monkeypatch):
    dataset = DummyDataset()
    bundle = make_craft_datasets(dataset, strategy="paired_dataset", contrastive_dataset=dataset)

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        craft_alpha=0.7,
        craft_beta=0.5,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
    )

    logs = {}

    def fake_log(self, values):
        logs.update(values)

    monkeypatch.setattr(CRAFTSFTTrainer, "log", fake_log, raising=False)

    loader = trainer.get_train_dataloader()
    batches = []
    iterator = iter(loader)
    for _ in range(4):
        batches.append(next(iterator))

    craft_batch = next(b for b in batches if b.get("craft_batch_type") == "craft")
    trainer.compute_loss(trainer.model, craft_batch)

    assert "loss/craft_total" in logs
    assert "metrics/craft_contrastive_accuracy" in logs
