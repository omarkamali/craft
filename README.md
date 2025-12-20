# CRAFT · Contrastive Representation Aware Fine-Tuning

CRAFT is a library that layers a contrastive InfoNCE objective on top of standard SFT and
preference-optimization trainers. It provides:

- **Composable losses** – configurable InfoNCE loss with projection/pooling and weighted
  blending against supervised losses via `craft_alpha`.
- **Mixed data loading** – automated cycling of SFT and contrastive batches according to a
  configurable `craft_beta` ratio, with optional auto-tuning via `craft_beta_mode`.
- **Trainer wrappers** – drop-in replacements for TRL's SFT/ORPO/GRPO/PPO/DPO trainers plus
  utilities for plain `transformers.Trainer` usage.
- **Metrics** – contrastive accuracy, representation consistency, and reference tracking.
- **Dataset utilities** – helpers for paired datasets or self-aligned positives, plus a
  default collator ready for mixed InfoNCE/SFT batches.
- **Flexible length matching** – options to oversample, cap, auto-adjust ratios, or raise
  if SFT and contrastive lengths diverge, alongside per-loader batch size overrides.

## Installation

```bash
# Editable install with testing extras
uv pip install -e '.[test]'

# Optional dependency groups
uv pip install -e '.[trl]'    # TRL trainers
uv pip install -e '.[hf]'     # transformers integration only
uv pip install -e '.[peft]'   # LoRA/PEFT examples
uv pip install -e '.[all]'    # everything
```

## Package layout

```
craft/
  ├── config.py     # CRAFT config mixin + TRL-specific configs
  ├── data.py       # Dataset bundle, collator, mixed dataloader
  ├── losses.py     # InfoNCELoss + loss combination helpers
  ├── metrics.py    # Metric utilities and EMA helpers
  ├── trainers.py   # CRAFT trainer mixin + TRL wrappers
  └── __init__.py   # Public exports
```

## Quick start

```python
from transformers import AutoModelForCausalLM
from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.trainers import CRAFTSFTTrainer

# Assume `sft_dataset` and `contrastive_dataset` are tokenized datasets with the
# appropriate columns (`input_ids`, `attention_mask`, optional *_tgt columns).

bundle = make_craft_datasets(
    sft_dataset,
    contrastive_dataset=contrastive_dataset,
    strategy="paired_dataset",
)

model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

args = CRAFTSFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    craft_alpha=0.6,
    craft_beta=0.5,
)

trainer = CRAFTSFTTrainer(
    model=model,
    args=args,
    train_dataset=sft_dataset,
    craft_bundle=bundle,
    data_collator=CRAFTCollator(),
)

trainer.train()
```

### Length matching & batching strategies

CRAFT lets you control how supervised (SFT) and contrastive datasets are balanced:

- `craft_length_strategy="oversample"` – loop the shorter loader (default).
- `"cap"` – stop when either loader exhausts, keeping epochs perfectly aligned.
- `"auto_beta"` – cap like above **and** recompute `craft_beta` from observed batch counts.
- `"error"` – raise if lengths diverge, useful for deterministic experiments.

Combine this with `craft_contrastive_batch_size` to decouple batch sizes:

```python
config = CRAFTSFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    craft_contrastive_batch_size=4,
    craft_beta=0.5,
    craft_beta_mode="auto",
    craft_length_strategy="auto_beta",
)
```

These knobs are honoured by all `CRAFT*Trainer` classes and the `CRAFTMixedDataLoader`.

## Notebooks

Six notebooks under `packages/craft/notebooks` cover end-to-end workflows:

1. **01-craft-basic-sft** – minimal CRAFTSFTTrainer run with paired datasets.
2. **02-craft-best-practices** – conversation packing, assistant masking, LoRA.
3. **03a-craft-loss-transformers-trainer** – integrate `InfoNCELoss` with vanilla
   `transformers.Trainer`.
4. **03b-craft-trl-sft** – TRL SFTTrainer wrapper with CRAFT metrics.
5. **03c-craft-trl-orpo** – ORPO preference optimisation with contrastive batches.
6. **04-craft-qlora-translation-eval** – QLoRA fine-tune of `unsloth/gemma-3-270M-it`
   on Flores translations, with before/after BLEU, loss curves, and metric plots.

## Testing

CRAFT ships with a pytest suite covering losses, metrics, data utilities, and trainer mixins.

```bash
uv pip install -e '.[test]'
uv run python -m pytest -q
```

## Contributing

1. Add or update tests for new functionality.
2. Run the lint/test suite before submitting patches.
3. Update notebooks and documentation to reflect API changes.
