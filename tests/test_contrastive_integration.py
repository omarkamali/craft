"""Integration tests for contrastive loss with different model output types."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.trainers import CRAFTTrainerMixin, CRAFTSFTTrainer

pytest.importorskip("trl", reason="CRAFT trainers require TRL")


class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([1, 2, 3, idx % 10 + 1]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, idx % 10 + 1, (idx + 1) % 10 + 1]),
        }


class ModelWithLastHiddenState(nn.Module):
    """Model that outputs last_hidden_state attribute."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(100, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
    
    def forward(self, input_ids, attention_mask, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        
        outputs = type("Outputs", (), {})()
        if output_hidden_states:
            outputs.last_hidden_state = x
            outputs.hidden_states = (x,)  # Also include hidden_states for completeness
        else:
            outputs.last_hidden_state = x
        
        return outputs


class ModelWithOnlyHiddenStates(nn.Module):
    """Model that only outputs hidden_states (like CausalLMOutputWithPast)."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(100, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
    
    def forward(self, input_ids, attention_mask, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        
        # Simulate multiple layers
        layer_outputs = []
        for layer in self.transformer.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
            layer_outputs.append(x)
        
        outputs = type("Outputs", (), {})()
        if output_hidden_states:
            outputs.hidden_states = tuple(layer_outputs)
            # Note: no last_hidden_state attribute
        
        return outputs


class CausalLMOutputWithPast:
    """Mock of transformers.modeling_outputs.CausalLMOutputWithPast."""
    
    def __init__(self, hidden_states, logits=None):
        self.hidden_states = hidden_states
        self.logits = logits


class ModelWithCausalLMOutput(nn.Module):
    """Model that outputs CausalLMOutputWithPast-like structure."""
    
    def __init__(self, hidden_size=64, vocab_size=1000):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        
        # Simulate multiple layers
        layer_outputs = []
        for layer in self.transformer.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
            layer_outputs.append(x)
        
        logits = self.lm_head(x)
        
        outputs = CausalLMOutputWithPast(
            hidden_states=tuple(layer_outputs) if output_hidden_states else None,
            logits=logits
        )
        
        return outputs


class TestContrastiveLossIntegration:
    """Integration tests for contrastive loss with different model output types."""
    
    def test_model_with_last_hidden_state(self):
        """Test contrastive loss with model that has last_hidden_state."""
        dataset = SimpleDataset(size=8)
        bundle = make_craft_datasets(dataset, strategy="self_align")
        
        config = CRAFTSFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            craft_alpha=0.5,
            craft_temperature=0.1,
            craft_contrastive_keys={
                "anchor_input_ids": "input_ids",
                "anchor_attention_mask": "attention_mask",
                "anchor_labels": "labels",
                "positive_input_ids": "input_ids_tgt",
                "positive_attention_mask": "attention_mask_tgt",
            }
        )
        
        model = ModelWithLastHiddenState()
        trainer = CRAFTSFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            craft_bundle=bundle,
            data_collator=CRAFTCollator(),
        )
        
        # Get a contrastive batch
        loader = trainer.get_train_dataloader()
        for batch in loader:
            if batch.get("craft_batch_type") == "craft":
                loss = trainer._compute_craft_contrastive_loss(
                    model, batch, return_outputs=False
                )
                assert torch.isfinite(loss)
                assert loss.item() >= 0
                break
    
    def test_model_with_only_hidden_states(self):
        """Test contrastive loss with model that only has hidden_states."""
        dataset = SimpleDataset(size=8)
        bundle = make_craft_datasets(dataset, strategy="self_align")
        
        config = CRAFTSFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            craft_alpha=0.5,
            craft_temperature=0.1,
            craft_contrastive_keys={
                "anchor_input_ids": "input_ids",
                "anchor_attention_mask": "attention_mask",
                "anchor_labels": "labels",
                "positive_input_ids": "input_ids_tgt",
                "positive_attention_mask": "attention_mask_tgt",
            }
        )
        
        model = ModelWithOnlyHiddenStates()
        trainer = CRAFTSFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            craft_bundle=bundle,
            data_collator=CRAFTCollator(),
        )
        
        # Get a contrastive batch
        loader = trainer.get_train_dataloader()
        for batch in loader:
            if batch.get("craft_batch_type") == "craft":
                loss = trainer._compute_craft_contrastive_loss(
                    model, batch, return_outputs=False
                )
                assert torch.isfinite(loss)
                assert loss.item() >= 0
                break
    
    def test_model_with_causallm_output(self):
        """Test contrastive loss with model that outputs CausalLMOutputWithPast."""
        dataset = SimpleDataset(size=8)
        bundle = make_craft_datasets(dataset, strategy="self_align")
        
        config = CRAFTSFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            craft_alpha=0.5,
            craft_temperature=0.1,
            craft_contrastive_keys={
                "anchor_input_ids": "input_ids",
                "anchor_attention_mask": "attention_mask",
                "anchor_labels": "labels",
                "positive_input_ids": "input_ids_tgt",
                "positive_attention_mask": "attention_mask_tgt",
            }
        )
        
        model = ModelWithCausalLMOutput()
        trainer = CRAFTSFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            craft_bundle=bundle,
            data_collator=CRAFTCollator(),
        )
        
        # Get a contrastive batch
        loader = trainer.get_train_dataloader()
        for batch in loader:
            if batch.get("craft_batch_type") == "craft":
                loss = trainer._compute_craft_contrastive_loss(
                    model, batch, return_outputs=False
                )
                assert torch.isfinite(loss)
                assert loss.item() >= 0
                break
    
    def test_extract_last_hidden_state_with_real_models(self):
        """Test _extract_last_hidden_state directly with different model outputs."""
        batch_size, seq_len, hidden_size = 2, 5, 64
        
        # Test with last_hidden_state
        model1 = ModelWithLastHiddenState()
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output1 = model1(input_ids, attention_mask, output_hidden_states=True)
        hidden1 = CRAFTTrainerMixin._extract_last_hidden_state(output1)
        assert hidden1.shape == (batch_size, seq_len, hidden_size)
        
        # Test with only hidden_states
        model2 = ModelWithOnlyHiddenStates()
        output2 = model2(input_ids, attention_mask, output_hidden_states=True)
        hidden2 = CRAFTTrainerMixin._extract_last_hidden_state(output2)
        assert hidden2.shape == (batch_size, seq_len, hidden_size)
        
        # Test with CausalLMOutputWithPast
        model3 = ModelWithCausalLMOutput()
        output3 = model3(input_ids, attention_mask, output_hidden_states=True)
        hidden3 = CRAFTTrainerMixin._extract_last_hidden_state(output3)
        assert hidden3.shape == (batch_size, seq_len, hidden_size)
    
    def test_contrastive_loss_computations_consistency(self):
        """Test that different output types give consistent results when hidden states are the same."""
        batch_size, seq_len, hidden_size = 2, 4, 32
        
        # Create identical hidden states
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create different output objects with same hidden states
        output1 = type("Output1", (), {"last_hidden_state": hidden})()
        output2 = type("Output2", (), {"hidden_states": (hidden,)})()
        output3 = type("Output3", (), {"hidden_states": hidden})()
        
        # Extract and compare
        h1 = CRAFTTrainerMixin._extract_last_hidden_state(output1)
        h2 = CRAFTTrainerMixin._extract_last_hidden_state(output2)
        h3 = CRAFTTrainerMixin._extract_last_hidden_state(output3)
        
        assert torch.equal(h1, h2)
        assert torch.equal(h2, h3)
        assert torch.equal(h1, h3)
