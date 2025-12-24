"""
Memory-efficient hidden state capture using PyTorch hooks.

This module provides hook-based alternatives to `output_hidden_states=True`,
which stores all layer outputs and can be memory-prohibitive for large models.

Instead of storing O(num_layers × batch × seq × hidden), we capture only the
final layer output at O(batch × seq × hidden).

References:
- PyTorch hooks documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_hook.html
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn


class LastHiddenStateHook:
    """
    Capture only the last hidden state without storing all layer outputs.

    This hook attaches to the final layer norm of a transformer model and
    captures its output during the forward pass. This is much more memory
    efficient than using output_hidden_states=True.

    Usage:
        hook = LastHiddenStateHook(model)
        outputs = model(input_ids, attention_mask)
        hidden = hook.get()  # [batch, seq, hidden]
        hook.clear()  # Free memory

    Attributes:
        hidden_state: The captured hidden state tensor, or None if not captured.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the hook.

        Args:
            model: The transformer model to hook into.
            layer_name: Optional explicit layer name to hook. If None, auto-detects
                the final layer norm (works for most HuggingFace models).
        """
        self.hidden_state: Optional[torch.Tensor] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._model = model

        target_module = self._find_target_module(model, layer_name)
        if target_module is not None:
            self._handle = target_module.register_forward_hook(self._hook)

    def _find_target_module(
        self,
        model: nn.Module,
        layer_name: Optional[str],
    ) -> Optional[nn.Module]:
        """
        Find the target module to hook.

        Tries common patterns for HuggingFace models:
        - model.model.norm (LLaMA, Mistral, Qwen)
        - model.transformer.ln_f (GPT-2, GPT-Neo)
        - model.model.final_layernorm (Falcon)
        - model.encoder.layer[-1].output.LayerNorm (BERT-style)
        - Direct norm attribute on model (test models)
        """
        if layer_name is not None:
            # Explicit layer name provided
            parts = layer_name.split(".")
            module = model
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    return None
            return module

        # Auto-detect common patterns
        # Handle DDP wrapping
        base = model.module if hasattr(model, "module") else model

        # Try common patterns
        patterns = [
            ("model", "norm"),  # LLaMA, Mistral, Qwen
            ("model", "final_layernorm"),  # Some models
            ("transformer", "ln_f"),  # GPT-2, GPT-Neo
            ("transformer", "norm"),  # Some variants
            ("encoder", "final_layer_norm"),  # Encoder models
        ]

        for container_name, norm_name in patterns:
            container = getattr(base, container_name, None)
            if container is not None:
                norm = getattr(container, norm_name, None)
                if norm is not None:
                    return norm

        # Fallback: try to find any LayerNorm at the model.model level
        backbone = getattr(base, "model", None)
        if backbone is not None:
            for name, module in backbone.named_children():
                if "norm" in name.lower() and isinstance(module, (nn.LayerNorm,)):
                    return module

        # Final fallback: check for norm directly on the model (test models)
        norm = getattr(base, "norm", None)
        if norm is not None and isinstance(norm, (nn.LayerNorm,)):
            return norm

        # Try ln_f for GPT-style
        ln_f = getattr(base, "ln_f", None)
        if ln_f is not None and isinstance(ln_f, (nn.LayerNorm,)):
            return ln_f

        return None

    def _hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        """Hook function that captures the output."""
        if isinstance(output, tuple):
            self.hidden_state = output[0]
        else:
            self.hidden_state = output

    def get(self) -> torch.Tensor:
        """
        Return the captured hidden state.

        Raises:
            RuntimeError: If no hidden state was captured.
        """
        if self.hidden_state is None:
            raise RuntimeError(
                "No hidden state captured. Ensure forward pass completed "
                "and hook was properly attached."
            )
        return self.hidden_state

    def get_optional(self) -> Optional[torch.Tensor]:
        """Return the captured hidden state or None if not captured."""
        return self.hidden_state

    def clear(self) -> None:
        """Clear the captured hidden state to free memory."""
        self.hidden_state = None

    def remove(self) -> None:
        """Remove the hook from the model."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.hidden_state = None

    @property
    def is_attached(self) -> bool:
        """Check if the hook is currently attached."""
        return self._handle is not None

    def __enter__(self) -> "LastHiddenStateHook":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()


class GradientCachingHook:
    """
    Hook for caching activations during gradient checkpointing-aware training.

    This is used by GradCache to capture activations without gradients during
    the first forward pass, then recompute with gradients during backward.

    References:
    - Gao et al. "Scaling Deep Contrastive Learning Batch Size under Memory
      Limited Setup" (GradCache), 2021
    """

    def __init__(self) -> None:
        self._activations: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(self, module: nn.Module) -> None:
        """Register a forward hook on the given module."""
        handle = module.register_forward_hook(self._capture)
        self._handles.append(handle)

    def _capture(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        """Capture the output activation."""
        if isinstance(output, tuple):
            self._activations.append(output[0].detach())
        else:
            self._activations.append(output.detach())

    def get_activations(self) -> list[torch.Tensor]:
        """Return captured activations."""
        return self._activations

    def clear(self) -> None:
        """Clear captured activations."""
        self._activations = []

    def remove(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._activations = []


def get_backbone(model: nn.Module) -> nn.Module:
    """
    Extract the backbone (transformer body) from a causal LM model.

    This returns the transformer without the LM head, suitable for
    extracting hidden states for contrastive learning.

    Handles:
    - DDP-wrapped models
    - PEFT-wrapped models
    - Standard HuggingFace CausalLM models

    Args:
        model: The full model (potentially wrapped).

    Returns:
        The backbone transformer module.
    """
    # Unwrap DDP
    if hasattr(model, "module"):
        model = model.module

    # Handle PEFT models
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()

    # Unwrap DDP again (in case PEFT was DDP-wrapped)
    if hasattr(model, "module"):
        model = model.module

    # Get backbone (the transformer without LM head)
    if hasattr(model, "model"):
        return model.model  # LLaMA, Mistral, etc.
    if hasattr(model, "transformer"):
        return model.transformer  # GPT-2, GPT-Neo
    if hasattr(model, "encoder"):
        return model.encoder  # Encoder models

    # Fallback: return the model itself
    return model
