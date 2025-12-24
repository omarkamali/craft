# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
