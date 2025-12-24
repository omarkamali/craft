"""Tests for CRAFTTrainerMixin._extract_last_hidden_state method."""

import pytest
import torch

from craft.trainers import CRAFTTrainerMixin


class MockOutput:
    """Mock model output with configurable attributes."""
    
    def __init__(self, last_hidden_state=None, hidden_states=None):
        if last_hidden_state is not None:
            self.last_hidden_state = last_hidden_state
        if hidden_states is not None:
            self.hidden_states = hidden_states


def test_extract_last_hidden_state_from_attribute():
    """Test extraction when last_hidden_state attribute exists."""
    hidden = torch.randn(2, 5, 768)
    output = MockOutput(last_hidden_state=hidden)
    
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    assert torch.equal(result, hidden)


def test_extract_last_hidden_state_from_list():
    """Test extraction from hidden_states list/tuple."""
    hidden_layers = [torch.randn(2, 5, 768) for _ in range(12)]
    output = MockOutput(hidden_states=hidden_layers)
    
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    assert torch.equal(result, hidden_layers[-1])


def test_extract_last_hidden_state_from_tuple():
    """Test extraction from hidden_states tuple."""
    hidden_layers = tuple(torch.randn(2, 5, 768) for _ in range(12))
    output = MockOutput(hidden_states=hidden_layers)
    
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    assert torch.equal(result, hidden_layers[-1])


def test_extract_last_hidden_state_from_tensor():
    """Test extraction when hidden_states is a single tensor."""
    hidden = torch.randn(2, 5, 768)
    output = MockOutput(hidden_states=hidden)
    
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    assert torch.equal(result, hidden)


def test_extract_last_hidden_state_prefers_last_hidden_state():
    """Test that last_hidden_state is preferred over hidden_states."""
    hidden_attr = torch.randn(2, 5, 768)
    hidden_list = [torch.randn(2, 5, 768) for _ in range(12)]
    output = MockOutput(last_hidden_state=hidden_attr, hidden_states=hidden_list)
    
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    assert torch.equal(result, hidden_attr)
    assert not torch.equal(result, hidden_list[-1])


def test_extract_last_hidden_state_missing_both():
    """Test error when both attributes are missing."""
    output = MockOutput()
    
    with pytest.raises(AttributeError, match="Model output missing last_hidden_state and hidden_states"):
        CRAFTTrainerMixin._extract_last_hidden_state(output)


def test_extract_last_hidden_state_empty_hidden_states():
    """Test error when hidden_states is empty list."""
    output = MockOutput(hidden_states=[])
    
    with pytest.raises(AttributeError, match="hidden_states sequence is empty"):
        CRAFTTrainerMixin._extract_last_hidden_state(output)


def test_extract_last_hidden_state_none_last_hidden_state():
    """Test extraction when last_hidden_state is None."""
    hidden_layers = [torch.randn(2, 5, 768) for _ in range(12)]
    output = MockOutput(last_hidden_state=None, hidden_states=hidden_layers)
    
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    assert torch.equal(result, hidden_layers[-1])


def test_extract_last_hidden_state_unsupported_type():
    """Test error when hidden_states has unsupported type."""
    output = MockOutput(hidden_states="not_a_tensor")
    
    with pytest.raises(TypeError, match="Unsupported hidden_states type"):
        CRAFTTrainerMixin._extract_last_hidden_state(output)


def test_extract_last_hidden_state_realistic_causallm_output():
    """Test with realistic CausalLMOutputWithPast-like structure."""
    # Simulate a realistic model output
    batch_size, seq_len, hidden_size = 3, 10, 768
    num_layers = 12
    
    # Create hidden states for each layer
    hidden_states = tuple(
        torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers)
    )
    
    # Create output like CausalLMOutputWithPast (no last_hidden_state)
    class CausalLMOutputWithPast:
        def __init__(self):
            self.hidden_states = hidden_states
            self.logits = torch.randn(batch_size, seq_len, 50257)  # vocab size
    
    output = CausalLMOutputWithPast()
    result = CRAFTTrainerMixin._extract_last_hidden_state(output)
    
    assert torch.equal(result, hidden_states[-1])
    assert result.shape == (batch_size, seq_len, hidden_size)
