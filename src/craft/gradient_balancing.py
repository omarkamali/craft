"""
Gradient Balancing for Multi-Task Learning.

This module implements strategies to address gradient dominance (task imbalance),
where tasks with larger gradient magnitudes eclipse smaller ones during training.

Implemented strategies:
- GradNorm: Dynamic gradient normalization (Chen et al., ICML 2018)
- Uncertainty Weighting: Homoscedastic uncertainty (Kendall et al., CVPR 2018)
- PCGrad: Projecting Conflicting Gradients (Yu et al., NeurIPS 2020)
- Loss Scale: Simple loss normalization by running mean

References:
- Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing
  in Deep Multitask Networks" (ICML 2018) https://arxiv.org/abs/1711.02257
- Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
  (CVPR 2018) https://arxiv.org/abs/1705.07115
- Yu et al. "Gradient Surgery for Multi-Task Learning"
  (NeurIPS 2020) https://arxiv.org/abs/2001.06782
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientBalancer:
    """
    Base class for gradient balancing strategies.
    
    Gradient dominance (also called task imbalance) occurs when tasks with
    larger gradient magnitudes dominate the update direction, causing other
    tasks to be under-optimized. This can vary by 15-33x between tasks.
    """
    
    def __init__(self, task_names: List[str], device: torch.device):
        self.task_names = task_names
        self.device = device
        self._step = 0
    
    def compute_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Compute task weights based on current losses.
        
        Args:
            losses: Dict mapping task name to loss tensor
            model: Model (needed for gradient-based methods)
            
        Returns:
            Dict mapping task name to weight multiplier
        """
        raise NotImplementedError
    
    def step(self) -> None:
        """Called after each optimization step."""
        self._step += 1


class LossScaleBalancer(GradientBalancer):
    """
    Simple loss normalization by running mean.
    
    Divides each loss by its exponential moving average, so all losses
    contribute roughly equally regardless of their absolute scale.
    
    This is the simplest approach and often works well in practice.
    """
    
    def __init__(
        self,
        task_names: List[str],
        device: torch.device,
        momentum: float = 0.99,
        warmup_steps: int = 100,
    ):
        super().__init__(task_names, device)
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        
        # Running means for each task
        self._running_means: Dict[str, float] = {name: 1.0 for name in task_names}
        self._initialized: Dict[str, bool] = {name: False for name in task_names}
    
    def compute_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        weights = {}
        
        for name in self.task_names:
            if name not in losses:
                weights[name] = 1.0
                continue
            
            loss_val = losses[name].detach().item()
            
            # Initialize or update running mean
            if not self._initialized[name]:
                self._running_means[name] = loss_val
                self._initialized[name] = True
            else:
                self._running_means[name] = (
                    self.momentum * self._running_means[name] +
                    (1 - self.momentum) * loss_val
                )
            
            # During warmup, use uniform weights
            if self._step < self.warmup_steps:
                weights[name] = 1.0
            else:
                # Normalize by running mean (avoid division by zero)
                mean = max(self._running_means[name], 1e-8)
                weights[name] = 1.0 / mean
        
        # Normalize weights to sum to number of tasks
        total = sum(weights.values())
        if total > 0:
            scale = len(self.task_names) / total
            weights = {k: v * scale for k, v in weights.items()}
        
        return weights


class UncertaintyWeightingBalancer(GradientBalancer):
    """
    Homoscedastic uncertainty weighting (Kendall et al., CVPR 2018).
    
    Learns task-specific log-variance parameters that automatically
    balance losses based on their uncertainty/noise level.
    
    Loss_i is weighted by 1/(2*σ_i²) + log(σ_i), where σ_i is learned.
    This naturally down-weights noisy/high-variance tasks.
    """
    
    def __init__(
        self,
        task_names: List[str],
        device: torch.device,
        initial_log_var: float = 0.0,
    ):
        super().__init__(task_names, device)
        
        # Learnable log-variance for each task (log(σ²))
        # Using log-variance for numerical stability
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_log_var, device=device))
            for name in task_names
        })
    
    def compute_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Compute uncertainty-based weights.
        
        The effective loss becomes: L_i / (2 * exp(log_var_i)) + log_var_i / 2
        Which simplifies to: L_i * exp(-log_var_i) / 2 + log_var_i / 2
        """
        weights = {}
        
        for name in self.task_names:
            if name not in self.log_vars:
                weights[name] = 1.0
                continue
            
            log_var = self.log_vars[name]
            # Weight = 1 / (2 * σ²) = exp(-log_var) / 2
            # We return just the multiplier; regularization term added separately
            weights[name] = float(torch.exp(-log_var).item() / 2)
        
        return weights
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get the regularization term: sum of log_var / 2.
        
        This prevents σ from going to infinity (which would zero out all losses).
        """
        reg = torch.tensor(0.0, device=self.device)
        for log_var in self.log_vars.values():
            reg = reg + log_var / 2
        return reg
    
    def parameters(self):
        """Return learnable parameters for optimizer."""
        return self.log_vars.parameters()


class GradNormBalancer(GradientBalancer):
    """
    GradNorm: Gradient Normalization (Chen et al., ICML 2018).
    
    Dynamically tunes task weights to balance gradient magnitudes,
    ensuring all tasks train at similar rates.
    
    Key idea: Adjust weights so that gradient norms are proportional
    to the inverse training rate (tasks that are behind get larger gradients).
    """
    
    def __init__(
        self,
        task_names: List[str],
        device: torch.device,
        alpha: float = 1.5,
        initial_losses: Optional[Dict[str, float]] = None,
    ):
        super().__init__(task_names, device)
        self.alpha = alpha  # Asymmetry hyperparameter
        
        # Learnable task weights
        self.task_weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1, device=device))
            for name in task_names
        })
        
        # Initial losses for computing relative training rates
        self._initial_losses: Dict[str, float] = initial_losses or {}
        self._current_losses: Dict[str, float] = {}
    
    def compute_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        # Update current losses
        for name, loss in losses.items():
            loss_val = loss.detach().item()
            self._current_losses[name] = loss_val
            
            # Initialize if needed
            if name not in self._initial_losses:
                self._initial_losses[name] = loss_val
        
        # Return current weights (actual balancing happens in backward)
        weights = {}
        for name in self.task_names:
            if name in self.task_weights:
                weights[name] = float(self.task_weights[name].item())
            else:
                weights[name] = 1.0
        
        return weights
    
    def compute_gradnorm_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_params: List[nn.Parameter],
    ) -> torch.Tensor:
        """
        Compute GradNorm loss for updating task weights.
        
        This should be called during training to update the task weights
        based on gradient magnitudes and training rates.
        """
        if len(task_losses) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute gradient norms for each task
        grad_norms = {}
        for name, loss in task_losses.items():
            if name not in self.task_weights:
                continue
            
            # Weighted loss
            weighted_loss = self.task_weights[name] * loss
            
            # Compute gradient w.r.t. shared parameters
            grads = torch.autograd.grad(
                weighted_loss,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            
            # Compute L2 norm of gradients
            grad_norm = torch.tensor(0.0, device=self.device)
            for g in grads:
                if g is not None:
                    grad_norm = grad_norm + g.norm() ** 2
            grad_norms[name] = torch.sqrt(grad_norm)
        
        if not grad_norms:
            return torch.tensor(0.0, device=self.device)
        
        # Average gradient norm
        avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)
        
        # Compute relative inverse training rates
        inverse_rates = {}
        for name in grad_norms:
            if name in self._initial_losses and name in self._current_losses:
                initial = max(self._initial_losses[name], 1e-8)
                current = max(self._current_losses[name], 1e-8)
                inverse_rates[name] = current / initial
            else:
                inverse_rates[name] = 1.0
        
        # Normalize inverse rates
        avg_rate = sum(inverse_rates.values()) / len(inverse_rates)
        relative_rates = {
            name: rate / avg_rate for name, rate in inverse_rates.items()
        }
        
        # GradNorm loss: ||G_i - avg_G * r_i^alpha||
        gradnorm_loss = torch.tensor(0.0, device=self.device)
        for name, grad_norm in grad_norms.items():
            target = avg_grad_norm * (relative_rates[name] ** self.alpha)
            gradnorm_loss = gradnorm_loss + torch.abs(grad_norm - target)
        
        return gradnorm_loss
    
    def parameters(self):
        """Return learnable parameters for optimizer."""
        return self.task_weights.parameters()


class PCGradBalancer(GradientBalancer):
    """
    PCGrad: Projecting Conflicting Gradients (Yu et al., NeurIPS 2020).
    
    When gradients from different tasks conflict (negative cosine similarity),
    project each gradient onto the normal plane of conflicting gradients.
    
    This prevents tasks from "fighting" each other during optimization.
    """
    
    def __init__(self, task_names: List[str], device: torch.device):
        super().__init__(task_names, device)
    
    def compute_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        # PCGrad doesn't use loss-based weights; it modifies gradients directly
        return {name: 1.0 for name in self.task_names}
    
    @staticmethod
    def project_conflicting_gradients(
        gradients: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Project gradients to remove conflicts.
        
        For each gradient g_i, if it conflicts with g_j (negative dot product),
        project g_i onto the normal plane of g_j.
        
        Args:
            gradients: List of gradient tensors (flattened)
            
        Returns:
            List of projected gradient tensors
        """
        if len(gradients) < 2:
            return gradients
        
        projected = [g.clone() for g in gradients]
        
        for i in range(len(projected)):
            for j in range(len(projected)):
                if i == j:
                    continue
                
                g_i = projected[i]
                g_j = gradients[j]  # Use original for projection target
                
                # Check for conflict (negative dot product)
                dot = torch.dot(g_i.flatten(), g_j.flatten())
                
                if dot < 0:
                    # Project g_i onto normal plane of g_j
                    # g_i_proj = g_i - (g_i · g_j / ||g_j||²) * g_j
                    g_j_norm_sq = torch.dot(g_j.flatten(), g_j.flatten())
                    if g_j_norm_sq > 1e-8:
                        projected[i] = g_i - (dot / g_j_norm_sq) * g_j
        
        return projected


def create_gradient_balancer(
    strategy: str,
    task_names: List[str],
    device: torch.device,
    **kwargs,
) -> Optional[GradientBalancer]:
    """
    Factory function to create gradient balancer.
    
    Args:
        strategy: One of 'none', 'loss_scale', 'uncertainty', 'gradnorm', 'pcgrad'
        task_names: List of task names (e.g., ['sft', 'contrastive'])
        device: Torch device
        **kwargs: Strategy-specific arguments
        
    Returns:
        GradientBalancer instance or None if strategy is 'none'
    """
    if strategy == "none":
        return None
    
    if strategy == "loss_scale":
        return LossScaleBalancer(
            task_names,
            device,
            momentum=kwargs.get("momentum", 0.99),
        )
    
    if strategy == "uncertainty":
        return UncertaintyWeightingBalancer(
            task_names,
            device,
        )
    
    if strategy == "gradnorm":
        return GradNormBalancer(
            task_names,
            device,
            alpha=kwargs.get("alpha", 1.5),
        )
    
    if strategy == "pcgrad":
        return PCGradBalancer(task_names, device)
    
    raise ValueError(f"Unknown gradient balancing strategy: {strategy}")
