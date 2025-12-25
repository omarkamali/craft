import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.trainers import CRAFTSFTTrainer

pytest.importorskip("trl", reason="CRAFT trainers require TRL")


# Shared tokenizer for all tests (cached after first load)
@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def make_sft_dataset(size=6):
    return Dataset.from_dict({
        "input_ids": [[i, i + 1] for i in range(size)],
        "attention_mask": [[1, 1] for _ in range(size)],
        "labels": [[-100, i + 1] for i in range(size)],  # First token masked, second is assistant
    })


def make_paired_dataset(size=6):
    return Dataset.from_dict({
        "input_ids": [[i, i + 1] for i in range(size)],
        "attention_mask": [[1, 1] for _ in range(size)],
        "labels": [[-100, i + 1] for i in range(size)],  # First token masked, second is assistant
        "input_ids_tgt": [[i + 10, i + 11] for i in range(size)],
        "attention_mask_tgt": [[1, 1] for _ in range(size)],
    })


def make_mask_dataset(include_mask: bool, size=4):
    data = {
        "input_ids": [[i, i + 1] for i in range(size)],
        "attention_mask": [[1, 1] for _ in range(size)],
        "labels": [[-100, -100] for _ in range(size)],
    }
    if include_mask:
        data["assistant_masks"] = [[0, 1] for _ in range(size)]
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
        hidden[:, -1, 0] = input_ids[:, -1].float()
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5)
        outputs = type("Outputs", (), {"loss": loss, "hidden_states": (hidden,)})()
        return outputs


def test_self_align_strategy_adds_positive_columns(tokenizer):
    """Test that self_align strategy generates positive mask during loss computation."""
    dataset = make_sft_dataset()
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
        processing_class=tokenizer,
    )

    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    # Self-align adds positive columns during _prepare_contrastive_inputs, not in dataloader
    # Just verify we get a craft batch and can compute loss
    assert batch.get("craft_batch_type") == "craft"
    loss = trainer.compute_loss(trainer.model, batch)
    assert torch.isfinite(loss)


def test_contrastive_batch_requires_keys(tokenizer):
    """Test that missing required keys raise ValueError."""
    # Use paired_dataset strategy which requires _tgt columns
    dataset = make_paired_dataset()
    bundle = make_craft_datasets(dataset, strategy="paired_dataset", contrastive_dataset=dataset)

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
        data_collator=CRAFTCollator(),
        processing_class=tokenizer,
    )

    loader = trainer.get_train_dataloader()
    # Get a craft batch
    for batch in loader:
        if batch.get("craft_batch_type") == "craft":
            # Remove required key
            batch.pop("attention_mask_tgt", None)
            with pytest.raises(ValueError):
                trainer.compute_loss(trainer.model, batch)
            break


def test_beta_ratio_cycles_batches(tokenizer):
    dataset = make_paired_dataset()
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
        processing_class=tokenizer,
    )

    loader = trainer.get_train_dataloader()
    pattern = []
    iterator = iter(loader)
    for _ in range(8):
        batch = next(iterator)
        pattern.append(batch["craft_batch_type"])
    assert pattern.count("craft") >= 2
    assert pattern.count("sft") >= 2


def test_length_strategy_error_raises_on_mismatch(tokenizer):
    dataset = make_paired_dataset()
    short_contrastive = make_paired_dataset(size=2)
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
        processing_class=tokenizer,
    )

    with pytest.raises(ValueError):
        trainer.get_train_dataloader()


def test_contrastive_batch_size_override_applied(tokenizer):
    dataset = make_paired_dataset()
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
        processing_class=tokenizer,
    )

    loader = trainer.get_train_dataloader()
    for batch in loader:
        if batch["craft_batch_type"] == "craft":
            assert batch["input_ids"].shape[0] == 4
            break
    else:  # pragma: no cover - defensive
        pytest.fail("Did not encounter a contrastive batch")


def test_custom_loaders_are_respected(tokenizer):
    dataset = make_paired_dataset()
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
        processing_class=tokenizer,
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


def test_self_align_validation_requires_labels_or_mask(tokenizer):
    dataset = make_mask_dataset(include_mask=False)
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
        processing_class=tokenizer,
    )

    with pytest.raises(ValueError):
        trainer.get_train_dataloader()


def test_self_align_validation_accepts_assistant_mask(tokenizer):
    dataset = make_mask_dataset(include_mask=True)
    bundle = make_craft_datasets(dataset, strategy="self_align")

    config = CRAFTSFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        craft_alpha=0.5,
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

    # Should not raise - validation passes with assistant_masks column
    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    assert batch["craft_batch_type"] in ("sft", "craft")
