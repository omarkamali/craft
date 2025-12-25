"""
Tests for CRAFT presets and auto-configuration.

These tests verify that:
1. All presets are valid and contain expected keys
2. from_preset() creates valid configs
3. auto_configure() produces sensible defaults
4. Overrides work correctly
"""

import pytest
import torch
import torch.nn as nn

from craft.config import (
    CRAFT_PRESETS,
    get_preset,
    auto_configure,
    CRAFTSFTConfig,
    CRAFTORPOConfig,
    CRAFTGRPOConfig,
    CRAFTPPOConfig,
    CRAFTDPOConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Simple model with config for testing."""
    class SimpleConfig:
        hidden_size = 768
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleConfig()
            self.linear = nn.Linear(768, 768)
        
        def forward(self, x):
            return self.linear(x)
    
    return SimpleModel()


@pytest.fixture
def mock_dataset():
    """Mock dataset with __len__."""
    class MockDataset:
        def __init__(self, size: int):
            self._size = size
        
        def __len__(self):
            return self._size
    
    return MockDataset


# =============================================================================
# Preset Tests
# =============================================================================

class TestPresets:
    """Tests for preset definitions."""
    
    def test_all_presets_exist(self):
        """All documented presets should exist."""
        expected = ["minimal", "balanced", "memory_efficient", "large_batch", "aggressive"]
        for name in expected:
            assert name in CRAFT_PRESETS, f"Missing preset: {name}"
    
    def test_presets_have_craft_alpha(self):
        """All presets should define craft_alpha."""
        for name, preset in CRAFT_PRESETS.items():
            assert "craft_alpha" in preset, f"Preset '{name}' missing craft_alpha"
            assert 0 < preset["craft_alpha"] <= 1, f"Invalid craft_alpha in '{name}'"
    
    def test_get_preset_returns_copy(self):
        """get_preset should return a copy, not the original."""
        preset1 = get_preset("balanced")
        preset2 = get_preset("balanced")
        
        preset1["craft_alpha"] = 0.99
        assert preset2["craft_alpha"] != 0.99
    
    def test_get_preset_unknown_raises(self):
        """Unknown preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")
    
    def test_minimal_preset_is_conservative(self):
        """Minimal preset should have high alpha (mostly SFT)."""
        preset = get_preset("minimal")
        assert preset["craft_alpha"] >= 0.7
        assert preset["craft_gradient_balancing"] == "none"
    
    def test_balanced_preset_has_gradient_balancing(self):
        """Balanced preset should enable gradient balancing."""
        preset = get_preset("balanced")
        assert preset["craft_gradient_balancing"] == "loss_scale"
        assert preset["craft_learnable_temperature"] is True
    
    def test_memory_efficient_preset_enables_gradcache(self):
        """Memory efficient preset should enable GradCache."""
        preset = get_preset("memory_efficient")
        assert preset["craft_use_gradcache"] is True
        assert preset["craft_projection_dim"] <= 128
    
    def test_large_batch_preset_uses_queue(self):
        """Large batch preset should use negative queue."""
        preset = get_preset("large_batch")
        assert preset["craft_negative_strategy"] == "queue"
        assert preset["craft_negative_queue_size"] > 0


# =============================================================================
# from_preset Tests
# =============================================================================

class TestFromPreset:
    """Tests for Config.from_preset() class method."""
    
    def test_from_preset_creates_config(self):
        """from_preset should create a valid config."""
        config = CRAFTSFTConfig.from_preset("balanced", output_dir="./test", use_cpu=True)
        
        assert config.output_dir == "./test"
        assert config.craft_alpha == 0.6
        assert config.craft_gradient_balancing == "loss_scale"
    
    def test_from_preset_with_overrides(self):
        """Overrides should take precedence over preset values."""
        config = CRAFTSFTConfig.from_preset(
            "balanced",
            output_dir="./test",
            craft_alpha=0.9,
            per_device_train_batch_size=8,
            use_cpu=True,
        )
        
        assert config.craft_alpha == 0.9  # Overridden
        assert config.craft_gradient_balancing == "loss_scale"  # From preset
        assert config.per_device_train_batch_size == 8  # Override
    
    def test_from_preset_all_config_classes(self):
        """All config classes should support from_preset."""
        configs = [
            CRAFTSFTConfig.from_preset("minimal", output_dir="./test", use_cpu=True),
            CRAFTORPOConfig.from_preset("minimal", output_dir="./test", use_cpu=True),
            CRAFTGRPOConfig.from_preset("minimal", output_dir="./test", use_cpu=True),
            CRAFTPPOConfig.from_preset("minimal", output_dir="./test", use_cpu=True),
            CRAFTDPOConfig.from_preset("minimal", output_dir="./test", use_cpu=True),
        ]
        
        for config in configs:
            assert config.craft_alpha == 0.8  # Minimal preset value
    
    def test_from_preset_unknown_raises(self):
        """Unknown preset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            CRAFTSFTConfig.from_preset("nonexistent", output_dir="./test", use_cpu=True)


# =============================================================================
# auto_configure Tests
# =============================================================================

class TestAutoConfigure:
    """Tests for auto_configure() function."""
    
    def test_auto_configure_defaults(self):
        """auto_configure with no args should return balanced-like defaults."""
        config = auto_configure()
        
        assert "craft_alpha" in config
        assert "craft_gradient_balancing" in config
        assert config["craft_gradient_balancing"] == "loss_scale"
    
    def test_auto_configure_scales_projection_dim(self, simple_model):
        """Projection dim should scale with model hidden size."""
        config = auto_configure(model=simple_model)
        
        # hidden_size=768, so projection_dim should be 768//4 = 192
        assert config["craft_projection_dim"] == 192
    
    def test_auto_configure_large_hidden_size(self):
        """Large hidden size should cap projection dim at 256."""
        class LargeConfig:
            hidden_size = 4096
        
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = LargeConfig()
        
        config = auto_configure(model=LargeModel())
        assert config["craft_projection_dim"] == 256  # Capped
    
    def test_auto_configure_small_hidden_size(self):
        """Small hidden size should have minimum projection dim of 64."""
        class SmallConfig:
            hidden_size = 128
        
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SmallConfig()
        
        config = auto_configure(model=SmallModel())
        assert config["craft_projection_dim"] == 64  # Minimum
    
    def test_auto_configure_computes_beta_from_datasets(self, mock_dataset):
        """Beta should be computed from dataset sizes."""
        sft = mock_dataset(1000)
        contrastive = mock_dataset(1000)
        
        config = auto_configure(sft_dataset=sft, contrastive_dataset=contrastive)
        
        # Equal sizes -> beta should be ~0.5
        assert 0.4 <= config["craft_beta"] <= 0.6
    
    def test_auto_configure_beta_clamped(self, mock_dataset):
        """Beta should be clamped to reasonable range."""
        sft = mock_dataset(100)
        contrastive = mock_dataset(10000)  # Much larger
        
        config = auto_configure(sft_dataset=sft, contrastive_dataset=contrastive)
        
        # Should be clamped to at least 0.3
        assert config["craft_beta"] >= 0.3
    
    def test_auto_configure_low_memory(self):
        """Low memory should enable aggressive optimizations."""
        config = auto_configure(available_memory_gb=12)
        
        assert config["craft_use_gradcache"] is True
        assert config["craft_gradcache_chunk_size"] == 2
        assert config["craft_projection_dim"] <= 128
    
    def test_auto_configure_medium_memory(self):
        """Medium memory should enable some optimizations."""
        config = auto_configure(available_memory_gb=20)
        
        assert config["craft_use_gradcache"] is True
        assert config["craft_gradcache_chunk_size"] == 4
    
    def test_auto_configure_high_memory(self):
        """High memory should use defaults."""
        config = auto_configure(available_memory_gb=48)
        
        # GradCache from balanced preset, but not forced
        # Just verify it doesn't crash
        assert "craft_alpha" in config
    
    def test_auto_configure_large_contrastive_uses_queue(self, mock_dataset):
        """Large contrastive dataset should trigger queue strategy."""
        sft = mock_dataset(10000)
        contrastive = mock_dataset(200000)  # Large
        
        config = auto_configure(sft_dataset=sft, contrastive_dataset=contrastive)
        
        assert config["craft_negative_strategy"] == "queue"
        assert config["craft_negative_queue_size"] <= 100000


# =============================================================================
# Config.auto Tests
# =============================================================================

class TestConfigAuto:
    """Tests for Config.auto() class method."""
    
    def test_auto_creates_config(self):
        """auto() should create a valid config."""
        config = CRAFTSFTConfig.auto(output_dir="./test", use_cpu=True)
        
        assert config.output_dir == "./test"
        assert hasattr(config, "craft_alpha")
    
    def test_auto_with_model(self, simple_model):
        """auto() should use model to configure projection dim."""
        config = CRAFTSFTConfig.auto(
            output_dir="./test",
            model=simple_model,
            use_cpu=True,
        )
        
        assert config.craft_projection_dim == 192  # 768 // 4
    
    def test_auto_with_overrides(self, simple_model):
        """Overrides should take precedence over auto values."""
        config = CRAFTSFTConfig.auto(
            output_dir="./test",
            model=simple_model,
            craft_projection_dim=512,  # Override auto-detected value
            use_cpu=True,
        )
        
        assert config.craft_projection_dim == 512
    
    def test_auto_all_config_classes(self):
        """All config classes should support auto()."""
        configs = [
            CRAFTSFTConfig.auto(output_dir="./test", use_cpu=True),
            CRAFTORPOConfig.auto(output_dir="./test", use_cpu=True),
            CRAFTGRPOConfig.auto(output_dir="./test", use_cpu=True),
            CRAFTPPOConfig.auto(output_dir="./test", use_cpu=True),
            CRAFTDPOConfig.auto(output_dir="./test", use_cpu=True),
        ]
        
        for config in configs:
            assert hasattr(config, "craft_alpha")
            assert hasattr(config, "craft_gradient_balancing")


# =============================================================================
# Integration Tests
# =============================================================================

class TestPresetsIntegration:
    """Integration tests for presets with real-ish scenarios."""
    
    def test_preset_values_are_valid_for_config(self):
        """All preset values should be valid config attributes."""
        for preset_name in CRAFT_PRESETS:
            # Should not raise
            config = CRAFTSFTConfig.from_preset(preset_name, output_dir="./test", use_cpu=True)
            
            # Verify key attributes are set
            assert hasattr(config, "craft_alpha")
            assert hasattr(config, "craft_gradient_balancing")
    
    def test_auto_then_override_workflow(self, simple_model, mock_dataset):
        """Common workflow: auto-detect then override specific values."""
        sft = mock_dataset(5000)
        contrastive = mock_dataset(10000)
        
        config = CRAFTSFTConfig.auto(
            output_dir="./outputs",
            model=simple_model,
            sft_dataset=sft,
            contrastive_dataset=contrastive,
            # User overrides
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            use_cpu=True,
        )
        
        # Auto-detected values
        assert config.craft_projection_dim == 192
        assert 0.3 <= config.craft_beta <= 0.5  # Adjusted for dataset ratio
        
        # User overrides
        assert config.per_device_train_batch_size == 4
        assert config.gradient_accumulation_steps == 8
        assert config.num_train_epochs == 3
