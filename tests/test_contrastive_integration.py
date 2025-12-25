"""Integration tests for contrastive loss with different model output types."""

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer

from craft.config import CRAFTSFTConfig
from craft.data import CRAFTCollator, make_craft_datasets
from craft.trainers import CRAFTTrainerMixin, CRAFTSFTTrainer

pytest.importorskip("trl", reason="CRAFT trainers require TRL")


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def make_simple_dataset(size=10):
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, i % 10 + 1] for i in range(size)],
        "attention_mask": [[1, 1, 1, 1] for _ in range(size)],
        "labels": [[-100, -100, i % 10 + 1, (i + 1) % 10 + 1] for i in range(size)],
    })


def make_model_config(hidden_size):
    return type("Config", (), {
        "hidden_size": hidden_size,
        "_name_or_path": "gpt2",
        "_attn_implementation": "eager",
    })()


class ModelWithLastHiddenState(nn.Module):
    """Model that outputs last_hidden_state attribute."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = make_model_config(hidden_size)
        self.embed = nn.Embedding(100, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
    
    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        
        outputs = type("Outputs", (), {})()
        outputs.loss = torch.tensor(0.5) if labels is not None else None
        if output_hidden_states:
            outputs.last_hidden_state = x
            outputs.hidden_states = (x,)
        else:
            outputs.last_hidden_state = x
        
        return outputs


class ModelWithOnlyHiddenStates(nn.Module):
    """Model that only outputs hidden_states (like CausalLMOutputWithPast)."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.config = make_model_config(hidden_size)
        self.embed = nn.Embedding(100, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
    
    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        
        layer_outputs = []
        for layer in self.transformer.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
            layer_outputs.append(x)
        
        outputs = type("Outputs", (), {})()
        outputs.loss = torch.tensor(0.5) if labels is not None else None
        if output_hidden_states:
            outputs.hidden_states = tuple(layer_outputs)
        
        return outputs


class CausalLMOutputWithPast:
    """Mock of transformers.modeling_outputs.CausalLMOutputWithPast."""
    
    def __init__(self, hidden_states, logits=None, loss=None):
        self.hidden_states = hidden_states
        self.logits = logits
        self.loss = loss


class ModelWithCausalLMOutput(nn.Module):
    """Model that outputs CausalLMOutputWithPast-like structure."""
    
    def __init__(self, hidden_size=64, vocab_size=1000):
        super().__init__()
        self.config = make_model_config(hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        x = self.embed(input_ids)
        
        layer_outputs = []
        for layer in self.transformer.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
            layer_outputs.append(x)
        
        logits = self.lm_head(x)
        
        outputs = CausalLMOutputWithPast(
            hidden_states=tuple(layer_outputs) if output_hidden_states else None,
            logits=logits,
            loss=torch.tensor(0.5) if labels is not None else None,
        )
        
        return outputs


class TestContrastiveLossIntegration:
    """Integration tests for contrastive loss with different model output types."""
    
    def test_model_with_last_hidden_state(self, tokenizer):
        """Test contrastive loss with model that has last_hidden_state."""
        dataset = make_simple_dataset(size=8)
        bundle = make_craft_datasets(dataset, strategy="self_align")
        
        config = CRAFTSFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            craft_alpha=0.5,
            craft_temperature=0.1,
            use_cpu=True,
        )
        
        model = ModelWithLastHiddenState()
        trainer = CRAFTSFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            craft_bundle=bundle,
            data_collator=CRAFTCollator(),
            processing_class=tokenizer,
        )
        
        loader = trainer.get_train_dataloader()
        for batch in loader:
            if batch.get("craft_batch_type") == "craft":
                loss = trainer.compute_loss(model, batch)
                assert torch.isfinite(loss)
                assert loss.item() >= 0
                break
    
    def test_model_with_only_hidden_states(self, tokenizer):
        """Test contrastive loss with model that only has hidden_states."""
        dataset = make_simple_dataset(size=8)
        bundle = make_craft_datasets(dataset, strategy="self_align")
        
        config = CRAFTSFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            craft_alpha=0.5,
            craft_temperature=0.1,
            use_cpu=True,
        )
        
        model = ModelWithOnlyHiddenStates()
        trainer = CRAFTSFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            craft_bundle=bundle,
            data_collator=CRAFTCollator(),
            processing_class=tokenizer,
        )
        
        loader = trainer.get_train_dataloader()
        for batch in loader:
            if batch.get("craft_batch_type") == "craft":
                loss = trainer.compute_loss(model, batch)
                assert torch.isfinite(loss)
                assert loss.item() >= 0
                break
    
    def test_model_with_causallm_output(self, tokenizer):
        """Test contrastive loss with model that outputs CausalLMOutputWithPast."""
        dataset = make_simple_dataset(size=8)
        bundle = make_craft_datasets(dataset, strategy="self_align")
        
        config = CRAFTSFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            craft_alpha=0.5,
            craft_temperature=0.1,
            use_cpu=True,
        )
        
        model = ModelWithCausalLMOutput()
        trainer = CRAFTSFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            craft_bundle=bundle,
            data_collator=CRAFTCollator(),
            processing_class=tokenizer,
        )
        
        loader = trainer.get_train_dataloader()
        for batch in loader:
            if batch.get("craft_batch_type") == "craft":
                loss = trainer.compute_loss(model, batch)
                assert torch.isfinite(loss)
                assert loss.item() >= 0
                break
    
    def test_extract_last_hidden_state_with_real_models(self):
        """Test _extract_last_hidden_state directly with different model outputs."""
        batch_size, seq_len, hidden_size = 2, 5, 64
        
        model1 = ModelWithLastHiddenState()
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output1 = model1(input_ids, attention_mask, output_hidden_states=True)
        hidden1 = CRAFTTrainerMixin._extract_last_hidden_state(output1)
        assert hidden1.shape == (batch_size, seq_len, hidden_size)
        
        model2 = ModelWithOnlyHiddenStates()
        output2 = model2(input_ids, attention_mask, output_hidden_states=True)
        hidden2 = CRAFTTrainerMixin._extract_last_hidden_state(output2)
        assert hidden2.shape == (batch_size, seq_len, hidden_size)
        
        model3 = ModelWithCausalLMOutput()
        output3 = model3(input_ids, attention_mask, output_hidden_states=True)
        hidden3 = CRAFTTrainerMixin._extract_last_hidden_state(output3)
        assert hidden3.shape == (batch_size, seq_len, hidden_size)
    
    def test_contrastive_loss_computations_consistency(self):
        """Test that different output types give consistent results when hidden states are the same."""
        batch_size, seq_len, hidden_size = 2, 4, 32
        
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        
        output1 = type("Output1", (), {"last_hidden_state": hidden})()
        output2 = type("Output2", (), {"hidden_states": (hidden,)})()
        output3 = type("Output3", (), {"hidden_states": hidden})()
        
        h1 = CRAFTTrainerMixin._extract_last_hidden_state(output1)
        h2 = CRAFTTrainerMixin._extract_last_hidden_state(output2)
        h3 = CRAFTTrainerMixin._extract_last_hidden_state(output3)
        
        assert torch.equal(h1, h2)
        assert torch.equal(h2, h3)
        assert torch.equal(h1, h3)
