# CRAFT · Contrastive Representation Aware Fine-Tuning

CRAFT is a library that layers a contrastive InfoNCE objective on top of standard SFT and
preference-optimization trainers. It provides:

- **Composable losses** – configurable InfoNCE loss with projection/pooling and weighted
  blending against supervised losses via `craft_alpha`.
- **Accumulation-aware scaling** – proper gradient ratio regardless of batch distribution,
  ensuring `alpha` means exactly what it says.
- **Memory-efficient training** – hook-based hidden state capture and GradCache support
  for large-batch contrastive learning under memory constraints.
- **Single forward pass** – for self-align strategy, both SFT and contrastive losses are
  computed from one forward pass using dual pooling.
- **Trainer wrappers** – drop-in replacements for TRL's SFT/ORPO/GRPO/PPO/DPO trainers plus
  utilities for plain `transformers.Trainer` usage.
- **Metrics** – contrastive accuracy, representation consistency, and reference tracking.
- **Dataset utilities** – helpers for paired datasets or self-aligned positives, plus a
  default collator ready for mixed InfoNCE/SFT batches.
- **Flexible length matching** – options to oversample, cap, auto-adjust ratios, or raise
  if SFT and contrastive lengths diverge, alongside per-loader batch size overrides.

## Techniques & References

CRAFT incorporates techniques from several influential papers:

| Technique | Reference | Usage in CRAFT |
|-----------|-----------|----------------|
| InfoNCE Loss | Oord et al. "Representation Learning with Contrastive Predictive Coding" (2018) | Core contrastive objective |
| Projection Head | Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020) | 2-layer MLP with GELU for projection |
| Temperature Scaling | Gao et al. "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (2021) | Configurable temperature (0.05 default) |
| Learnable Temperature | Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021) | Optional `craft_learnable_temperature` |
| GradCache | Gao et al. "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup" (2021) | Memory-efficient large-batch training |
| Negative Queue | He et al. "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo, 2020) | Optional `craft_negative_strategy="queue"` |
| Multi-task Accumulation | Raffel et al. "Exploring the Limits of Transfer Learning" (T5, 2020) | Accumulation-aware loss scaling |

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
  ├── config.py       # CRAFT config mixin + TRL-specific configs
  ├── data.py         # Dataset bundle, collator, mixed dataloader
  ├── losses.py       # InfoNCELoss, ProjectionHead, pooling strategies
  ├── metrics.py      # Metric utilities and EMA helpers
  ├── trainers.py     # CRAFT trainer mixin + TRL wrappers
  ├── accumulator.py  # Accumulation-aware loss scaling
  ├── hooks.py        # Memory-efficient hidden state capture
  ├── gradcache.py    # GradCache for large-batch contrastive
  └── __init__.py     # Public exports
```

## What's New

### v0.3.0:

This release introduces significant optimizations for memory efficiency and training correctness:

**Accumulation-Aware Loss Scaling**: The loss scaling now correctly accounts for batch
distribution within gradient accumulation windows. Previously, with `alpha=0.6` and
`beta=0.6`, the effective gradient ratio was ~72:28 instead of the intended 60:40.
Now `alpha` means exactly what it says regardless of `beta`.

**Single Forward Pass for Self-Align**: When using `strategy="self_align"`, CRAFT now
computes both SFT and contrastive losses from a single forward pass using dual pooling.
This eliminates the redundant second forward pass, reducing compute by ~50% for self-align.

**Memory-Efficient Hidden State Capture**: New hook-based hidden state extraction captures
only the final layer output instead of all layers. This reduces memory overhead from
O(num_layers × batch × seq × hidden) to O(batch × seq × hidden).

**GradCache Support**: For paired dataset training with large batches, enable
`craft_use_gradcache=True` to compute contrastive loss with gradient caching.
This allows effective batch sizes of 1000+ even on a single GPU.

**Improved Projection Head**: The projection head now uses a 2-layer MLP with GELU
activation (following SimCLR), replacing the previous single-layer Tanh design.
Output dimension is configurable via `craft_projection_dim`.

```python
config = CRAFTSFTConfig(
    # Memory optimization
    craft_use_gradcache=True,           # Enable GradCache for large batches
    craft_gradcache_chunk_size=8,       # Chunk size for backward pass
    craft_use_hidden_state_hook=True,   # Hook-based hidden state capture

    # Projection head
    craft_projection_dim=256,           # Lower dim = more efficient
    craft_learnable_temperature=True,   # CLIP-style learnable temp

    # Negative sampling
    craft_negative_strategy="queue",    # MoCo-style negative queue
    craft_negative_queue_size=65536,
)
```

### Custom Data Loaders

CRAFT now supports custom PyTorch `DataLoader` instances for both SFT and contrastive training, giving you more control over batching, sampling, and collation logic.

```python
trainer = CRAFTSFTTrainer(
    model=model,
    args=args,
    train_dataset=sft_dataset,  # Still required for length calculations
    craft_bundle=bundle,
    craft_sft_loader=custom_sft_loader,          # Custom SFT loader
    craft_contrastive_loader=custom_contrast_loader  # Custom contrastive loader
)
```

### Enhanced Self-align Validation

When using `strategy="self_align"`, CRAFT now performs additional validation to ensure your data is properly formatted:

- Validates presence of either `labels` or `assistant_mask` in SFT batches
- Ensures at least one token is marked as an assistant token
- Provides clear error messages for common configuration issues

```python
# Example of valid self-align batch
{
    "input_ids": torch.tensor([...]),
    "attention_mask": torch.tensor([...]),
    "labels": torch.tensor([-100, -100, 1234, 5678, -100]),  # Assistant tokens where labels != -100
    # OR
    "assistant_mask": torch.tensor([0, 0, 1, 1, 0])  # 1 marks assistant tokens
}
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
