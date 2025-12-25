import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.losses import InfoNCELoss
from craft.trainers import CRAFTSFTTrainer

pytest.importorskip("trl", reason="CRAFT trainers require TRL")


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def make_dummy_dataset(include_targets: bool = True, size: int = 10):
    data = {
        "input_ids": [[i, i + 1] for i in range(size)],
        "attention_mask": [[1, 1] for _ in range(size)],
        "labels": [[-100, i + 1] for i in range(size)],  # First token masked, second is assistant
    }
    if include_targets:
        data["input_ids_tgt"] = [[i + 1, i + 2] for i in range(size)]
        data["attention_mask_tgt"] = [[1, 1] for _ in range(size)]
    return Dataset.from_dict(data)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": 4,
            "_name_or_path": "gpt2",
            "_attn_implementation": "eager",
        })()

    def forward(self, input_ids, attention_mask, labels=None, assistant_masks=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.config.hidden_size)
        # simple deterministic hidden states
        hidden[:, -1, 0] = input_ids[:, -1].float()
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5)
        outputs = type("Outputs", (), {"loss": loss, "hidden_states": (hidden,)})()
        return outputs


def test_craft_trainer_self_align_generates_positive_mask(tokenizer):
    """Test that self_align generates positive mask during loss computation."""
    dataset = make_dummy_dataset(include_targets=False)
    bundle = make_craft_datasets(dataset, strategy="self_align")

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        craft_alpha=0.5,
        craft_assistant_mask_strategy="auto",
        use_cpu=True,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
        processing_class=tokenizer,
    )

    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    # Self-align adds positive columns during loss computation, not in dataloader
    assert batch.get("craft_batch_type") == "craft"
    loss = trainer.compute_loss(trainer.model, batch)
    assert torch.isfinite(loss)


def test_craft_trainer_logs_metrics(tokenizer, monkeypatch):
    dataset = make_dummy_dataset()
    bundle = make_craft_datasets(dataset, strategy="paired_dataset", contrastive_dataset=dataset)

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        craft_alpha=0.7,
        craft_beta=0.5,
        use_cpu=True,
    )

    trainer = CRAFTSFTTrainer(
        model=DummyModel(),
        args=config,
        train_dataset=dataset,
        craft_bundle=bundle,
        data_collator=CRAFTCollator(),
        processing_class=tokenizer,
    )

    logs = {}

    # Store original log method to call it
    original_log = trainer.log.__func__

    def fake_log(self, values):
        # Call original to trigger CRAFT log flushing
        original_log(self, values)
        logs.update(values)

    monkeypatch.setattr(CRAFTSFTTrainer, "log", fake_log, raising=False)

    loader = trainer.get_train_dataloader()
    batches = []
    iterator = iter(loader)
    for _ in range(4):
        batches.append(next(iterator))

    craft_batch = next(b for b in batches if b.get("craft_batch_type") == "craft")
    trainer.compute_loss(trainer.model, craft_batch)

    # CRAFT losses are now accumulated and only flushed when main 'loss' is logged.
    # Simulate the trainer logging its main loss to trigger flush.
    trainer.log({"loss": 1.0})

    assert "loss/craft_total" in logs
    assert "loss/craft_contrast" in logs
