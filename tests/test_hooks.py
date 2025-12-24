"""Tests for the CRAFT hooks module."""

import pytest
import torch
import torch.nn as nn

from craft.hooks import LastHiddenStateHook, get_backbone


class SimpleTransformer(nn.Module):
    """Simple transformer-like model for testing with hookable norm layer."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        x = self.embedding(input_ids)
        hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm(x)
        if output_hidden_states:
            hidden_states.append(x)

        return type("Outputs", (), {
            "last_hidden_state": x,
            "hidden_states": tuple(hidden_states) if output_hidden_states else None,
        })()


class CausalLMModel(nn.Module):
    """Model with backbone structure like HuggingFace CausalLM."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 100):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.model = SimpleTransformer(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
        outputs = self.model(input_ids, attention_mask, output_hidden_states)
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = torch.tensor(0.5)

        return type("CausalLMOutput", (), {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": outputs.last_hidden_state,
            "hidden_states": outputs.hidden_states,
        })()


class TestLastHiddenStateHook:
    """Tests for LastHiddenStateHook."""

    def test_hook_captures_output(self):
        """Test that hook captures hidden state during forward pass."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        input_ids = torch.randint(0, 100, (2, 10))
        outputs = model(input_ids)

        captured = hook.get()
        assert captured is not None
        assert captured.shape == outputs.last_hidden_state.shape
        # Check values match
        assert torch.allclose(captured, outputs.last_hidden_state)

    def test_hook_clear(self):
        """Test that clear removes captured state."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        input_ids = torch.randint(0, 100, (2, 10))
        model(input_ids)

        assert hook.get_optional() is not None
        hook.clear()
        assert hook.get_optional() is None

    def test_hook_remove(self):
        """Test that remove detaches the hook."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        assert hook.is_attached

        hook.remove()
        assert not hook.is_attached

    def test_hook_context_manager(self):
        """Test hook as context manager."""
        model = SimpleTransformer()

        with LastHiddenStateHook(model) as hook:
            assert hook.is_attached
            input_ids = torch.randint(0, 100, (2, 10))
            model(input_ids)
            captured = hook.get()
            assert captured is not None

        # After context, hook should be removed
        assert not hook.is_attached

    def test_get_raises_without_forward(self):
        """Test that get() raises if no forward pass was done."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        with pytest.raises(RuntimeError):
            hook.get()

    def test_hook_on_causal_lm_backbone(self):
        """Test hook attachment on CausalLM-style model."""
        model = CausalLMModel()

        # Hook should attach to the backbone's norm layer
        hook = LastHiddenStateHook(model.model)

        input_ids = torch.randint(0, 100, (2, 10))
        model(input_ids)

        captured = hook.get()
        assert captured is not None
        assert captured.shape == (2, 10, 64)


class TestGetBackbone:
    """Tests for get_backbone function."""

    def test_get_backbone_from_causal_lm(self):
        """Test extracting backbone from CausalLM model."""
        model = CausalLMModel()
        backbone = get_backbone(model)

        # Should return the model.model (the transformer part)
        assert backbone is model.model

    def test_get_backbone_from_plain_transformer(self):
        """Test extracting backbone from plain transformer."""
        model = SimpleTransformer()
        backbone = get_backbone(model)

        # Should return the model itself (no .model attribute)
        assert backbone is model

    def test_get_backbone_handles_ddp_wrapper(self):
        """Test that DDP-wrapped models are handled."""
        model = CausalLMModel()

        # Simulate DDP wrapper
        class FakeDDP:
            def __init__(self, module):
                self.module = module

        ddp_model = FakeDDP(model)
        backbone = get_backbone(ddp_model)

        # Should unwrap DDP and return the backbone
        assert backbone is model.model


class TestHookIntegration:
    """Integration tests for hooks with model forward passes."""

    def test_multiple_forward_passes(self):
        """Test that hook works correctly across multiple forward passes."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        for i in range(3):
            input_ids = torch.randint(0, 100, (2, 10))
            outputs = model(input_ids)
            captured = hook.get()

            assert captured.shape == outputs.last_hidden_state.shape
            assert torch.allclose(captured, outputs.last_hidden_state)

            hook.clear()

    def test_hook_with_varying_batch_sizes(self):
        """Test hook with different batch sizes."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        for batch_size in [1, 4, 8]:
            input_ids = torch.randint(0, 100, (batch_size, 10))
            outputs = model(input_ids)
            captured = hook.get()

            assert captured.shape[0] == batch_size
            hook.clear()

    def test_hook_with_varying_sequence_lengths(self):
        """Test hook with different sequence lengths."""
        model = SimpleTransformer()
        hook = LastHiddenStateHook(model)

        for seq_len in [5, 20, 50]:
            input_ids = torch.randint(0, 100, (2, seq_len))
            outputs = model(input_ids)
            captured = hook.get()

            assert captured.shape[1] == seq_len
            hook.clear()
