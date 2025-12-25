# CRAFT Training Guide

> One-page field manual for teams bringing CRAFT into production training loops.

## 1. Why CRAFT
- **Preserve latent geometry while fine-tuning**: layer InfoNCE on top of SFT or preference objectives to keep helpfulness/grounding vectors tight instead of drifting. See `CRAFTTrainerMixin` for integration details @src/craft/trainers.py#72-420.
- **Drop-in for TRL & vanilla Trainer**: configs extend TRL defaults, so existing scripts swap `SFTConfig` → `CRAFTSFTConfig` and add a `CRAFTDatasetBundle`.

## 2. Datasets at a glance
| Role | When to use | Required columns |
| --- | --- | --- |
| **SFT dataset** | Always — supplies supervised batches | `input_ids`, `attention_mask`, optional `labels` |
| **Extra InfoNCE dataset** | When you have explicit positive pairs (human preference tuples, retrieval positives, multilingual alignments). Configure `strategy="paired_dataset"`. | Anchor columns above **plus** `input_ids_tgt`, `attention_mask_tgt` | 
| **Self-align InfoNCE** | When only SFT data exists. Set `strategy="self_align"`; positives are synthesised by reusing anchor tokens and masks (see mask fallback @src/craft/trainers.py#352-398). | Anchor columns; positive columns optional |

**Bundle builder**: `make_craft_datasets(primary, contrastive_dataset, strategy)` validates these combinations @src/craft/data.py#21-45.

## 3. Batch mixing & knobs
- **Custom SFT Data Loaders**: You can provide your own PyTorch `DataLoader` instances for both SFT and contrastive training. This allows for advanced batching, sampling, or collation logic. Pass them to the trainer as `craft_sft_loader` and `craft_contrastive_loader` arguments when initializing the trainer.
- **Self-align Requirements**: When using `strategy="self_align"`, your SFT batches must include either:
  - Standard `labels` tensor (with `-100` for tokens to ignore) from which assistant spans are derived, or
  - An explicit `assistant_mask` boolean tensor (configurable via `craft_assistant_mask_key` in training args)
  The trainer will validate this requirement and raise a clear error if neither is present.
- **craft_alpha** (default 0.6): weight on SFT loss. Lower values emphasise InfoNCE; keep ≥0.4 for stability on small models. Defined in `CRAFTConfigMixin` @src/craft/config.py#34-118.
- **craft_beta**: fraction of gradient steps allocated to SFT batches within an accumulation window. `CRAFTMixedDataLoader` enforces the pattern @src/craft/data.py#55-171. Example: `beta=0.75`, `grad_accum=4` ⇒ 3 SFT + 1 contrastive step loop.
- **craft_beta_mode** / **craft_length_strategy**: handle dataset length mismatch.
  - `oversample` (default): loop shorter loader.
  - `cap`: stop when either loader ends.
  - `auto_beta`: recompute beta from observed lengths, then cap.
  - `error`: fail fast when counts differ (see regression test @tests/test_trainers.py#134-160).
- **craft_contrastive_batch_size**: override positive batches if they are smaller or cheaper to compute.
- **craft_pooling**, **craft_temperature**: govern InfoNCE projection head and logits (implementation @src/craft/losses.py#37-122).

## 4. Minimal workflow
1. **Tokenize datasets** so anchor / positive columns align (± conversation packing).
2. **Build bundle**:
   ```python
   bundle = make_craft_datasets(
       sft_dataset,
       contrastive_dataset=my_pairs,  # None for self-align
       strategy="paired_dataset",
   )
   ```
3. **Instantiate config**: start from your TRL config and add CRAFT knobs:
   ```python
   from craft.config import CRAFTSFTConfig

   args = CRAFTSFTConfig(
       output_dir="./outputs",
       per_device_train_batch_size=2,
       craft_alpha=0.6,
       craft_beta=0.5,
       craft_length_strategy="auto_beta",
   )
   ```
4. **Trainer**: use `CRAFTSFTTrainer`, `CRAFTPPOTrainer`, etc., passing `craft_bundle` + optional `CRAFTCollator`.
5. **Monitor logs**: look for `loss/craft_total`, `loss/craft_contrast`, and metrics specified in `craft_report_metrics` such as `metrics/craft_contrastive_accuracy`.

## 5. Memory optimization (v0.3.0)

CRAFT v0.3.0 introduces expert-level memory optimizations for large-scale training:

### Hook-based Hidden State Capture
Instead of using `output_hidden_states=True` (which stores O(layers×batch×seq×hidden)), CRAFT can now capture only the final layer norm output at O(batch×seq×hidden):

```python
config = CRAFTSFTConfig(
    craft_use_hidden_state_hook=True,  # Enable hook-based capture
)
```

### GradCache for Large Batch Contrastive Learning
Train with effective batch sizes of 1000+ on a single GPU by chunking the forward/backward pass:

```python
config = CRAFTSFTConfig(
    craft_use_gradcache=True,           # Enable GradCache
    craft_gradcache_chunk_size=8,       # Backward chunk size
    per_device_train_batch_size=4,      # Physical batch
)
```

### Learnable Temperature & Projection Tuning
Fine-tune the temperature scaling and projection head output dimension:

```python
config = CRAFTSFTConfig(
    craft_learnable_temperature=True,   # CLIP-style learnable temp
    craft_projection_dim=256,           # Lower dims = more efficient
    craft_projection_dropout=0.1,
)
```

### Negative Queue Strategy
Use MoCo-style memory bank instead of in-batch negatives to reduce effective batch dependencies:

```python
config = CRAFTSFTConfig(
    craft_negative_strategy="queue",    # "inbatch" or "queue"
    craft_negative_queue_size=65536,
)
```

## 6. Operational considerations
- **Throughput planning**: with hook-based capture and GradCache, contrastive overhead is minimal. Use `craft_use_hidden_state_hook=True` by default and `craft_use_gradcache=True` for paired datasets.
- **Dataset hygiene**:
  - Keep anchor/positive sequence lengths similar to avoid InfoNCE temperature collapse.
  - For paired datasets, ensure positives are true semantic matches; noisy positives hurt more than missing ones.
  - For self-align, double-check assistant masking so positives only cover assistant spans (automatic masking derives from `labels != -100`).
- **Scheduling**: favour `auto_beta` when dataset sizes change run-to-run (e.g., filtering pipelines). Use `error` for exact reproducibility sweeps.
- **Evaluation**: log contrastive accuracy (>0.7 indicates decent in-batch discrimination) and representation consistency to detect drift across checkpoints.
- **Fallback mode**: set `craft_alpha=1.0` to disable InfoNCE without code changes if you need to isolate bugs.

## 7. Dataset checklist
- [ ] Anchor tensors: `input_ids`, `attention_mask`, optionally `labels` (for mask derivation / losses).
- [ ] Positive tensors: `_tgt` columns if using paired strategy.
- [ ] Consistent tokenization + padding rules between anchor and positive splits.
- [ ] Metadata documenting source, filtering steps, and license (helps reproduce contrastive wins).

## 8. Customization & Advanced Usage

### Custom Data Loaders
You can inject custom PyTorch `DataLoader` instances for both SFT and contrastive training:

```python
from torch.utils.data import DataLoader

# Create custom loaders with your preferred settings
sft_loader = DataLoader(
    sft_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=my_custom_collator  # Optional custom collation
)

contrastive_loader = DataLoader(
    contrastive_dataset,
    batch_size=8,  # Can differ from SFT batch size
    shuffle=True,
    collate_fn=my_contrastive_collator
)

# Pass to trainer
trainer = CRAFTSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,  # Still required for length calculations
    craft_bundle=bundle,
    craft_sft_loader=sft_loader,           # Custom SFT loader
    craft_contrastive_loader=contrastive_loader  # Custom contrastive loader
)
```

### Self-align Validation
When using `strategy="self_align"`, the trainer performs validation to ensure your data is properly formatted:

1. **Validation Checks**:
   - Verifies that either `labels` or `assistant_mask` exists in SFT batches
   - Ensures at least one token is marked as an assistant token (not all -100 in labels or at least one True in assistant_mask)
   - Validates that the data loader is iterable and provides batches in the expected format

2. **Error Messages**:
   - If no valid batches are found: "CRAFT strategy='self_align' requires a readable SFT dataloader to validate assistant spans..."
   - If neither labels nor assistant_mask is present: "CRAFT strategy='self_align' needs either labels (with assistant tokens where labels != -100) or an assistant mask column..."

3. **Configuration**:
   - Customize the assistant mask key via `args.craft_assistant_mask_key` (default: "assistant_mask")
   - Control label key via `args.craft_contrastive_keys.anchor_labels` (default: "labels")

## 9. Gradient Balancing (v0.4.0)

When training with multiple losses (SFT + contrastive), you may observe **gradient dominance** — tasks with larger gradient magnitudes eclipse smaller ones. Research shows this can vary by 15-33x between tasks.

### The Problem
```
# Without balancing:
SFT gradient norm:         0.1
Contrastive gradient norm: 10.0  ← dominates updates
```

### Available Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `none` | No balancing (default) | When losses are naturally balanced |
| `loss_scale` | Normalize by running mean | Simple, works well in practice |
| `uncertainty` | Learn task uncertainty σ | When task noise varies |
| `gradnorm` | Balance gradient magnitudes | When training rates differ |
| `pcgrad` | Project conflicting gradients | When tasks actively conflict |

### Configuration

```python
config = CRAFTSFTConfig(
    # Choose strategy
    craft_gradient_balancing="loss_scale",  # none|loss_scale|uncertainty|gradnorm|pcgrad
    
    # Strategy-specific options
    craft_loss_scale_momentum=0.99,   # EMA momentum for loss_scale
    craft_gradnorm_alpha=1.5,         # GradNorm asymmetry (higher = stronger)
    
    # Per-task gradient clipping (optional, works with any strategy)
    craft_gradient_clip_per_task=True,
    craft_gradient_clip_value=1.0,
)
```

### Recommendations

1. **Start with `loss_scale`** — it's simple and often sufficient
2. **Use `uncertainty`** when tasks have different noise levels (e.g., noisy contrastive pairs)
3. **Use `gradnorm`** when you observe one task converging much faster than another
4. **Use `pcgrad`** when tasks actively interfere (loss for one goes up when other improves)

### References
- **GradNorm**: Chen et al., ICML 2018 — [arxiv:1711.02257](https://arxiv.org/abs/1711.02257)
- **Uncertainty Weighting**: Kendall et al., CVPR 2018 — [arxiv:1705.07115](https://arxiv.org/abs/1705.07115)
- **PCGrad**: Yu et al., NeurIPS 2020 — [arxiv:2001.06782](https://arxiv.org/abs/2001.06782)

## 10. Presets & Auto-Configuration (v0.4.0)

CRAFT now offers presets and auto-configuration to simplify getting started.

### Presets

Instead of tuning individual parameters, start from a preset:

```python
from craft.config import CRAFTSFTConfig

# Use a preset directly
config = CRAFTSFTConfig.from_preset("balanced", output_dir="./outputs")

# Override specific values
config = CRAFTSFTConfig.from_preset(
    "memory_efficient",
    output_dir="./outputs",
    per_device_train_batch_size=2,
    craft_alpha=0.7,  # Override preset value
)
```

| Preset | Description | When to Use |
|--------|-------------|-------------|
| `minimal` | High alpha (0.8), no balancing | Trying CRAFT with minimal risk |
| `balanced` | Recommended defaults | Starting a new project |
| `memory_efficient` | GradCache + small projection | OOM errors, single GPU |
| `large_batch` | Queue negatives, GradCache | Multiple GPUs, 1000+ batch |
| `aggressive` | Strong contrastive (alpha=0.4) | Representation quality critical |

### Auto-Configuration

Let CRAFT analyze your model and data to suggest settings:

```python
config = CRAFTSFTConfig.auto(
    output_dir="./outputs",
    model=my_model,                    # Detects hidden size → projection dim
    sft_dataset=train_data,            # Estimates size → beta
    contrastive_dataset=pairs_data,    # Large dataset → queue strategy
    available_memory_gb=16,            # Low memory → enables GradCache
)
```

Auto-configuration:
- Scales `craft_projection_dim` based on model hidden size (hidden_size // 4, capped at 256)
- Computes `craft_beta` from dataset size ratio
- Enables GradCache and reduces projection dim for low memory
- Switches to queue negatives for large contrastive datasets (>100k)

### Combining Presets with Auto

You can also override auto-detected values:

```python
config = CRAFTSFTConfig.auto(
    output_dir="./outputs",
    model=model,
    sft_dataset=sft_data,
    # Override auto-detected values
    craft_gradient_balancing="gradnorm",
    num_train_epochs=5,
)
```

## 11. FAQ
1. **Do I need a separate InfoNCE dataset?** No. If you lack explicit pairs, set `strategy="self_align"` and CRAFT reuses SFT batches as both anchor and positive, masking to isolate assistant spans. Dedicated contrastive data usually improves contrastive accuracy by 5–15 points, so prefer it when available.
2. **How do negatives work?** Default is in-batch negatives—every other example in the contrastive batch. Swap in custom queues by overriding `CRAFTTrainerMixin._compute_craft_contrastive_loss` or adding a queue inside your collator.
3. **What about evaluation batches?** Training automatically toggles between SFT and contrastive batches; evaluation remains pure SFT unless you write a custom eval loop.
4. **Can I run preference optimisation (DPO/ORPO/GRPO/PPO) with CRAFT?** Yes—use the matching `CRAFT*Config` + trainer variant. Contrastive batches mix with preference losses exactly like SFT.
5. **How do I resume?** Resumption works like TRL: as long as you checkpoint the trainer, CRAFT restores projection head weights, beta schedule, and representation references.

## 12. DDP Training Script

The `ddp/train_craft.py` script provides a production-ready multi-GPU training setup with:

- **Multi-source contrastive data**: FLORES+ (curated), NLLB (mined bitext), and semantic similarity datasets
- **Stratified sampling**: Preserves language/source distribution when limiting samples
- **QLoRA**: 4-bit quantization with LoRA adapters

### Quick Start

```bash
# Edit paths in launch_craft.sh, then:
cd ddp
./launch_craft.sh
```

### Key v0.3/v0.4 Options

```bash
# Memory optimization (v0.3)
--craft_use_hidden_state_hook true   # Hook-based capture (recommended)
--craft_use_gradcache true           # Large effective batch sizes
--craft_gradcache_chunk_size 4       # Smaller = less memory
--craft_learnable_temperature true   # CLIP-style learnable temp
--craft_projection_dim 256           # Lower = more efficient

# Gradient balancing (v0.4)
--craft_gradient_balancing loss_scale  # Prevents gradient dominance
--craft_loss_scale_momentum 0.99       # EMA momentum

# Alternative: use gradnorm for stronger balancing
--craft_gradient_balancing gradnorm
--craft_gradnorm_alpha 1.5
```

### Contrastive Data Sources

| Source | Description | Quality | Volume |
|--------|-------------|---------|--------|
| FLORES+ | Curated parallel sentences | High | ~1k per lang pair |
| NLLB | Mined bitext (LASER filtered) | Medium-High | Millions |
| all-nli | NLI entailment pairs | High | ~500k |
| quora-duplicates | Duplicate questions | High | ~400k |
| natural-questions | QA pairs | Medium | ~300k |

### Tuning Tips

1. **Start with `loss_scale`** gradient balancing—it's simple and effective
2. **Enable GradCache** if you want larger effective batch sizes without OOM
3. **Use `--nllb_min_laser_score 1.25`** or higher for cleaner bitext
4. **Monitor `loss/craft_contrast`** and `metrics/craft_contrastive_accuracy`—accuracy >0.7 indicates good discrimination

For deeper dives see README quick start and example notebooks under `notebooks/` (especially 02, 03a–c, 04).
