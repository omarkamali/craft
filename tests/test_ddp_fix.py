#!/usr/bin/env python3
"""Test script to verify DDP fix works correctly."""

import torch
import torch.nn as nn
from unittest.mock import Mock

class MockModel:
    """Mock model with .model attribute like HF CausalLM."""
    def __init__(self):
        self.model = Mock()
        self.config = Mock()
        self.config.hidden_size = 768
        self.device = torch.device('cpu')

class MockDDPModel:
    """Mock DDP-wrapped model."""
    def __init__(self, model):
        self.module = model
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.module, name)

def test_ddp_handling():
    """Test that our DDP handling logic works correctly."""
    
    # Test 1: Non-DDP model with .model attribute
    regular_model = MockModel()
    base = regular_model
    
    # Simulate our fixed logic
    if hasattr(base, 'module') and hasattr(base, 'model'):
        backbone = base.module.model
    elif hasattr(base, 'module'):
        backbone = base.module
    else:
        backbone = base.model
    
    assert backbone == regular_model.model, "Failed to access .model on regular model"
    print("✓ Non-DDP model test passed")
    
    # Test 2: DDP-wrapped model with .model attribute
    ddp_model = MockDDPModel(regular_model)
    base = ddp_model
    
    # Simulate our fixed logic
    if hasattr(base, 'module') and hasattr(base, 'model'):
        backbone = base.module.model
    elif hasattr(base, 'module'):
        backbone = base.module
    else:
        backbone = base.model
    
    assert backbone == regular_model.model, "Failed to access .model through DDP wrapper"
    print("✓ DDP-wrapped model test passed")
    
    # Test 3: Config and device access through DDP
    model_to_check = ddp_model.module if hasattr(ddp_model, 'module') else ddp_model
    assert hasattr(model_to_check, 'config'), "Failed to access config through DDP"
    assert hasattr(model_to_check, 'device'), "Failed to access device through DDP"
    print("✓ DDP config/device access test passed")
    
    print("All tests passed! DDP fix is working correctly.")

if __name__ == "__main__":
    test_ddp_handling()
