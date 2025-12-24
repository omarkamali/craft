# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-12-24

### Fixed
- Fixed memory leak in `_craft_reference_embeddings` EMA update causing OOM during long training runs
- Fixed hidden states not being freed when using `output_hidden_states=True` fallback path
- Fixed tensor references in metrics keeping computation graph alive

## [0.3.0] - 2025-12-24

### Added
- Memory-efficient hidden state capture via PyTorch hooks (`LastHiddenStateHook`) with auto-detection of final layer norms
- GradCache support (`GradCacheContrastiveLoss`) for 1000+ effective batch sizes on single GPU
- MoCo-style negative queue (`CachedEmbeddingBank`) for memory-efficient negative sampling
- Accumulation-aware loss scaling (`CRAFTGradientAccumulator`) ensuring `alpha` parameter means exactly what it says
- SimCLR-style 2-layer MLP projection head with GELU and L2 normalization
- Learnable temperature parameter (CLIP-style) with `craft_learnable_temperature` flag
- Single forward pass for self-align strategy via dual pooling (50% compute reduction)
- Configurable projection output dimension via `craft_projection_dim`
- Additional pooling strategy: `weighted_mean` for position-aware token weighting
- Support for additional negatives in InfoNCE loss

### Changed
- Projection head now uses 2-layer MLP with GELU instead of single-layer Tanh
- Improved backbone extraction to handle DDP and PEFT wrapping
- Memory overhead reduced from O(layers×batch×seq×hidden) to O(batch×seq×hidden)

### New Configuration Options
- `craft_use_gradcache` – Enable gradient caching for large batches
- `craft_gradcache_chunk_size` – Chunk size for GradCache backward pass
- `craft_use_hidden_state_hook` – Hook-based hidden state capture
- `craft_projection_dim` – Output dimension for projection head (default 256)
- `craft_projection_dropout` – Dropout in projection head
- `craft_learnable_temperature` – Make temperature a trainable parameter
- `craft_negative_strategy` – Choose "inbatch" or "queue" for negatives
- `craft_negative_queue_size` – Size of negative embedding queue

## [0.2.6] - 2025-12-24

### Fixed
- Move reference embedding to CUDA device to prevent device mismatch errors

## [0.2.5] - 2025-12-24

### Fixed
- Fixed AttributeError when models return `CausalLMOutputWithPast` without `last_hidden_state` attribute
- Added `_extract_last_hidden_state` helper method to safely extract hidden states from different model output types

## [0.2.4] - 2025-12-24

### Fixed
- Added proper handling for DDP-wrapped models when accessing backbone, config, and device attributes

## [0.2.3] - 2025-12-24

This update tries to address OOM issues by reducing memory usage in contrastive training.

### Fixed
- Fixed critical OOM issues in contrastive training by eliminating LM head logits generation
- Replaced full model forwards with backbone-only forwards to bypass LM head (~71.79 GiB savings)
- Removed `output_hidden_states=True` from contrastive forwards to reduce memory usage
- Made `return_details=True` conditional - only requested when metrics need it
- Optimized reference storage to use CPU tensors and mean-pooled vectors instead of batch tensors
- Added explicit tensor cleanup with `del` statements after contrastive computations
- Removed logits from loss details dictionary to prevent storing large tensors
- Added `.detach()` to embeddings in details so metrics don't keep computation graph
- Added comprehensive debug prints for memory tracking around contrastive forwards

## [0.2.2] - 2025-12-24

### Fixed
- Fixed TRL import issues

## [0.2.0] - 2025-12-20

### Added
- Support for custom PyTorch `DataLoader` instances for both SFT and contrastive training
- Enhanced self-align validation with clear error messages
- New `craft_assistant_mask_key` configuration option for customizing the assistant mask key
- Added `craft_allow_packed_contrastive` flag to control packing detection in contrastive batches
- Improved documentation with usage examples and best practices

### Changed
- Default collator no longer inherits from the trainer's data_collator to prevent packing issues
- Improved error messages for common configuration mistakes
- More robust handling of missing or invalid configuration
- Updated dependencies to latest stable versions

### Fixed
- Fixed potential issues with contrastive batch handling when using custom loaders
- Improved validation of self-align strategy requirements
- Fixed edge cases in batch processing and loss computation
- Resolved potential race conditions in mixed data loading

## [0.1.0] - 2025-12-20

Initial beta release of CRAFT: Contrastive Representation Aware Fine-Tuning toolkit

### Added
- Core CRAFT training framework with PyTorch backend
- Support for various loss functions including contrastive learning objectives
- Data loading and preprocessing utilities
- Training metrics and monitoring
- Integration with Hugging Face Transformers
- Support for PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning) integration
- Comprehensive test suite
- Documentation and example notebooks
