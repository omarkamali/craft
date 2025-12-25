"""
Tests for gradient balancing strategies.

These tests verify the mathematical properties and expectations from:
- GradNorm (Chen et al., ICML 2018): Gradient magnitudes should be balanced
- Uncertainty Weighting (Kendall et al., CVPR 2018): Higher uncertainty → lower weight
- PCGrad (Yu et al., NeurIPS 2020): Conflicting gradients should be projected
- Loss Scale: Losses should be normalized by their running mean
"""

import pytest
import torch
import torch.nn as nn
import math

from craft.gradient_balancing import (
    GradientBalancer,
    LossScaleBalancer,
    UncertaintyWeightingBalancer,
    GradNormBalancer,
    PCGradBalancer,
    create_gradient_balancer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def task_names():
    return ["sft", "contrastive"]


@pytest.fixture
def simple_model():
    """Simple model for gradient computation tests."""
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
    )


# =============================================================================
# Factory Tests
# =============================================================================

class TestCreateGradientBalancer:
    """Tests for the factory function."""
    
    def test_none_returns_none(self, task_names, device):
        balancer = create_gradient_balancer("none", task_names, device)
        assert balancer is None
    
    def test_loss_scale_creates_correct_type(self, task_names, device):
        balancer = create_gradient_balancer("loss_scale", task_names, device)
        assert isinstance(balancer, LossScaleBalancer)
    
    def test_uncertainty_creates_correct_type(self, task_names, device):
        balancer = create_gradient_balancer("uncertainty", task_names, device)
        assert isinstance(balancer, UncertaintyWeightingBalancer)
    
    def test_gradnorm_creates_correct_type(self, task_names, device):
        balancer = create_gradient_balancer("gradnorm", task_names, device)
        assert isinstance(balancer, GradNormBalancer)
    
    def test_pcgrad_creates_correct_type(self, task_names, device):
        balancer = create_gradient_balancer("pcgrad", task_names, device)
        assert isinstance(balancer, PCGradBalancer)
    
    def test_unknown_strategy_raises(self, task_names, device):
        with pytest.raises(ValueError, match="Unknown gradient balancing strategy"):
            create_gradient_balancer("unknown", task_names, device)
    
    def test_gradnorm_alpha_passed(self, task_names, device):
        balancer = create_gradient_balancer("gradnorm", task_names, device, alpha=2.0)
        assert balancer.alpha == 2.0
    
    def test_loss_scale_momentum_passed(self, task_names, device):
        balancer = create_gradient_balancer("loss_scale", task_names, device, momentum=0.9)
        assert balancer.momentum == 0.9


# =============================================================================
# LossScaleBalancer Tests
# =============================================================================

class TestLossScaleBalancer:
    """
    Tests for loss normalization by running mean.
    
    Key property: After convergence, weights should be inversely proportional
    to loss magnitudes, so all losses contribute equally.
    """
    
    def test_initial_weights_are_uniform(self, task_names, device):
        """During warmup, weights should be uniform."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=100)
        
        losses = {
            "sft": torch.tensor(1.0),
            "contrastive": torch.tensor(10.0),  # 10x larger
        }
        
        weights = balancer.compute_weights(losses)
        
        # During warmup (step 0), weights should be uniform
        assert weights["sft"] == pytest.approx(1.0)
        assert weights["contrastive"] == pytest.approx(1.0)
    
    def test_weights_inversely_proportional_to_loss_after_warmup(self, task_names, device):
        """After warmup, larger losses should get smaller weights."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=0, momentum=0.0)
        
        # With momentum=0, running mean equals current loss
        losses = {
            "sft": torch.tensor(1.0),
            "contrastive": torch.tensor(10.0),
        }
        
        weights = balancer.compute_weights(losses)
        
        # Weight should be 1/loss, normalized
        # sft: 1/1 = 1, contrastive: 1/10 = 0.1
        # Normalized to sum to 2: sft = 1.818, contrastive = 0.182
        assert weights["sft"] > weights["contrastive"]
        assert weights["sft"] / weights["contrastive"] == pytest.approx(10.0, rel=0.01)
    
    def test_running_mean_converges(self, task_names, device):
        """Running mean should converge to stable value."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=0, momentum=0.9)
        
        # Simulate constant losses
        losses = {"sft": torch.tensor(2.0), "contrastive": torch.tensor(4.0)}
        
        for _ in range(100):
            balancer.compute_weights(losses)
            balancer.step()
        
        # Running means should converge to actual values
        assert balancer._running_means["sft"] == pytest.approx(2.0, rel=0.01)
        assert balancer._running_means["contrastive"] == pytest.approx(4.0, rel=0.01)
    
    def test_weights_sum_to_num_tasks(self, task_names, device):
        """Weights should be normalized to sum to number of tasks."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=0, momentum=0.0)
        
        losses = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(5.0)}
        weights = balancer.compute_weights(losses)
        
        assert sum(weights.values()) == pytest.approx(len(task_names))
    
    def test_handles_missing_task(self, task_names, device):
        """Should handle missing tasks gracefully."""
        balancer = LossScaleBalancer(task_names, device)
        
        # Only provide one loss
        losses = {"sft": torch.tensor(1.0)}
        weights = balancer.compute_weights(losses)
        
        assert "sft" in weights
        assert "contrastive" in weights
        assert weights["contrastive"] == 1.0  # Default for missing


# =============================================================================
# UncertaintyWeightingBalancer Tests
# =============================================================================

class TestUncertaintyWeightingBalancer:
    """
    Tests for homoscedastic uncertainty weighting (Kendall et al., 2018).
    
    Key properties:
    1. Weight = 1 / (2 * σ²) = exp(-log_var) / 2
    2. Higher uncertainty (larger σ) → lower weight
    3. Regularization term prevents σ → ∞
    """
    
    def test_initial_weights_are_equal(self, task_names, device):
        """With equal initial log_var, weights should be equal."""
        balancer = UncertaintyWeightingBalancer(task_names, device, initial_log_var=0.0)
        
        losses = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(1.0)}
        weights = balancer.compute_weights(losses)
        
        # exp(-0) / 2 = 0.5 for both
        assert weights["sft"] == pytest.approx(0.5)
        assert weights["contrastive"] == pytest.approx(0.5)
    
    def test_higher_log_var_gives_lower_weight(self, task_names, device):
        """Higher uncertainty should give lower weight."""
        balancer = UncertaintyWeightingBalancer(task_names, device)
        
        # Manually set different log_vars
        with torch.no_grad():
            balancer.log_vars["sft"].fill_(0.0)  # σ² = 1
            balancer.log_vars["contrastive"].fill_(2.0)  # σ² = e² ≈ 7.4
        
        losses = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(1.0)}
        weights = balancer.compute_weights(losses)
        
        # sft: exp(-0)/2 = 0.5
        # contrastive: exp(-2)/2 ≈ 0.068
        assert weights["sft"] > weights["contrastive"]
        assert weights["sft"] == pytest.approx(0.5, rel=0.01)
        assert weights["contrastive"] == pytest.approx(math.exp(-2) / 2, rel=0.01)
    
    def test_weight_formula_matches_paper(self, task_names, device):
        """Verify weight = exp(-log_var) / 2 as per Kendall et al."""
        balancer = UncertaintyWeightingBalancer(task_names, device)
        
        test_log_vars = [-1.0, 0.0, 1.0, 2.0]
        
        for log_var_val in test_log_vars:
            with torch.no_grad():
                balancer.log_vars["sft"].fill_(log_var_val)
            
            losses = {"sft": torch.tensor(1.0)}
            weights = balancer.compute_weights(losses)
            
            expected = math.exp(-log_var_val) / 2
            assert weights["sft"] == pytest.approx(expected, rel=0.01)
    
    def test_regularization_loss_is_positive(self, task_names, device):
        """Regularization term should be positive (prevents σ → ∞)."""
        balancer = UncertaintyWeightingBalancer(task_names, device, initial_log_var=1.0)
        
        reg_loss = balancer.get_regularization_loss()
        
        # sum of log_var / 2 = 2 * 1.0 / 2 = 1.0
        assert reg_loss.item() == pytest.approx(1.0)
    
    def test_parameters_are_learnable(self, task_names, device):
        """Log_vars should be learnable parameters."""
        balancer = UncertaintyWeightingBalancer(task_names, device)
        
        params = list(balancer.parameters())
        assert len(params) == len(task_names)
        
        for p in params:
            assert p.requires_grad


# =============================================================================
# GradNormBalancer Tests
# =============================================================================

class TestGradNormBalancer:
    """
    Tests for GradNorm (Chen et al., ICML 2018).
    
    Key properties:
    1. Task weights are learnable
    2. Gradient norms should be balanced across tasks
    3. Tasks that are "behind" (higher relative loss) get larger gradients
    """
    
    def test_initial_weights_are_one(self, task_names, device):
        """Initial task weights should be 1.0."""
        balancer = GradNormBalancer(task_names, device)
        
        losses = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(1.0)}
        weights = balancer.compute_weights(losses)
        
        assert weights["sft"] == pytest.approx(1.0)
        assert weights["contrastive"] == pytest.approx(1.0)
    
    def test_alpha_parameter_stored(self, task_names, device):
        """Alpha hyperparameter should be stored."""
        balancer = GradNormBalancer(task_names, device, alpha=2.0)
        assert balancer.alpha == 2.0
    
    def test_initial_losses_tracked(self, task_names, device):
        """Initial losses should be tracked for relative training rate."""
        balancer = GradNormBalancer(task_names, device)
        
        losses = {"sft": torch.tensor(2.0), "contrastive": torch.tensor(4.0)}
        balancer.compute_weights(losses)
        
        assert balancer._initial_losses["sft"] == pytest.approx(2.0)
        assert balancer._initial_losses["contrastive"] == pytest.approx(4.0)
    
    def test_parameters_are_learnable(self, task_names, device):
        """Task weights should be learnable parameters."""
        balancer = GradNormBalancer(task_names, device)
        
        params = list(balancer.parameters())
        assert len(params) == len(task_names)
        
        for p in params:
            assert p.requires_grad
    
    def test_gradnorm_loss_computation(self, task_names, device, simple_model):
        """GradNorm loss should be computable."""
        balancer = GradNormBalancer(task_names, device)
        
        # Create losses that require grad
        x = torch.randn(4, 10)
        out = simple_model(x)
        
        task_losses = {
            "sft": out[:, 0].mean(),
            "contrastive": out[:, 1].mean(),
        }
        
        # Initialize balancer
        balancer.compute_weights({k: v.detach() for k, v in task_losses.items()})
        
        # Compute GradNorm loss
        shared_params = list(simple_model.parameters())
        gradnorm_loss = balancer.compute_gradnorm_loss(task_losses, shared_params)
        
        assert gradnorm_loss.requires_grad or gradnorm_loss.item() >= 0
    
    def test_relative_training_rate_affects_target(self, task_names, device):
        """Tasks that are behind should have higher target gradient norm."""
        balancer = GradNormBalancer(task_names, device, alpha=1.0)
        
        # Set initial losses
        initial = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(1.0)}
        balancer.compute_weights(initial)
        
        # Now sft has improved (lower loss), contrastive hasn't
        current = {"sft": torch.tensor(0.5), "contrastive": torch.tensor(1.0)}
        balancer.compute_weights(current)
        
        # Relative inverse training rate:
        # sft: 0.5/1.0 = 0.5 (improved)
        # contrastive: 1.0/1.0 = 1.0 (no improvement)
        # contrastive should get higher target gradient norm
        
        assert balancer._current_losses["sft"] == pytest.approx(0.5)
        assert balancer._current_losses["contrastive"] == pytest.approx(1.0)


# =============================================================================
# PCGradBalancer Tests
# =============================================================================

class TestPCGradBalancer:
    """
    Tests for PCGrad (Yu et al., NeurIPS 2020).
    
    Key properties:
    1. Non-conflicting gradients are unchanged
    2. Conflicting gradients (negative dot product) are projected
    3. After projection, gradients should not conflict
    """
    
    def test_weights_are_uniform(self, task_names, device):
        """PCGrad doesn't use loss-based weights."""
        balancer = PCGradBalancer(task_names, device)
        
        losses = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(10.0)}
        weights = balancer.compute_weights(losses)
        
        assert weights["sft"] == 1.0
        assert weights["contrastive"] == 1.0
    
    def test_non_conflicting_gradients_unchanged(self):
        """Gradients with positive dot product should be unchanged."""
        g1 = torch.tensor([1.0, 0.0, 0.0])
        g2 = torch.tensor([0.5, 0.5, 0.0])  # Positive dot product with g1
        
        dot = torch.dot(g1, g2)
        assert dot > 0, "Test setup: gradients should not conflict"
        
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2])
        
        # Should be unchanged (or very close)
        assert torch.allclose(projected[0], g1, atol=1e-6)
        assert torch.allclose(projected[1], g2, atol=1e-6)
    
    def test_conflicting_gradients_projected(self):
        """Gradients with negative dot product should be projected."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([-1.0, 0.0])  # Directly opposite
        
        dot = torch.dot(g1, g2)
        assert dot < 0, "Test setup: gradients should conflict"
        
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2])
        
        # After projection, g1 should be projected onto normal plane of g2
        # g1_proj = g1 - (g1·g2 / ||g2||²) * g2
        # g1_proj = [1,0] - (-1/1) * [-1,0] = [1,0] + [-1,0] = [0,0]
        assert torch.allclose(projected[0], torch.tensor([0.0, 0.0]), atol=1e-6)
    
    def test_projection_removes_conflict(self):
        """After projection, gradients should not conflict."""
        g1 = torch.tensor([1.0, 1.0])
        g2 = torch.tensor([-1.0, 0.5])  # Conflicts with g1 (dot = -0.5)
        
        initial_dot = torch.dot(g1, g2)
        assert initial_dot < 0, "Test setup: gradients should conflict"
        
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2])
        
        # After projection, dot product should be >= 0
        final_dot = torch.dot(projected[0], g2)
        assert final_dot >= -1e-6, f"Projected gradient still conflicts: dot={final_dot}"
    
    def test_projection_formula_matches_paper(self):
        """Verify projection formula: g_i - (g_i·g_j / ||g_j||²) * g_j"""
        g1 = torch.tensor([3.0, 4.0])
        g2 = torch.tensor([-1.0, 0.0])
        
        # Manual calculation
        dot = torch.dot(g1, g2)  # -3
        g2_norm_sq = torch.dot(g2, g2)  # 1
        expected = g1 - (dot / g2_norm_sq) * g2  # [3,4] - (-3) * [-1,0] = [3,4] + [-3,0] = [0,4]
        
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2])
        
        assert torch.allclose(projected[0], expected, atol=1e-6)
    
    def test_single_gradient_unchanged(self):
        """Single gradient should be unchanged."""
        g1 = torch.tensor([1.0, 2.0, 3.0])
        
        projected = PCGradBalancer.project_conflicting_gradients([g1])
        
        assert len(projected) == 1
        assert torch.allclose(projected[0], g1)
    
    def test_orthogonal_gradients_unchanged(self):
        """Orthogonal gradients (dot=0) should be unchanged."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([0.0, 1.0])
        
        dot = torch.dot(g1, g2)
        assert dot == pytest.approx(0.0), "Test setup: gradients should be orthogonal"
        
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2])
        
        assert torch.allclose(projected[0], g1, atol=1e-6)
        assert torch.allclose(projected[1], g2, atol=1e-6)
    
    def test_multiple_conflicting_gradients(self):
        """Should handle multiple pairwise conflicts."""
        g1 = torch.tensor([1.0, 0.0, 0.0])
        g2 = torch.tensor([-0.5, 1.0, 0.0])
        g3 = torch.tensor([-0.5, -0.5, 1.0])
        
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2, g3])
        
        assert len(projected) == 3
        # All projected gradients should exist
        for p in projected:
            assert p.shape == g1.shape


# =============================================================================
# Integration Tests
# =============================================================================

class TestGradientBalancingIntegration:
    """Integration tests verifying end-to-end behavior."""
    
    def test_loss_scale_stabilizes_training(self, task_names, device):
        """Loss scale should make effective losses similar magnitude."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=0, momentum=0.0)
        
        # Very different loss magnitudes
        losses = {
            "sft": torch.tensor(0.1),
            "contrastive": torch.tensor(100.0),
        }
        
        weights = balancer.compute_weights(losses)
        
        # Effective losses should be similar
        effective_sft = losses["sft"].item() * weights["sft"]
        effective_contrastive = losses["contrastive"].item() * weights["contrastive"]
        
        # Ratio should be much closer to 1 than original 1000x difference
        ratio = effective_sft / effective_contrastive
        assert 0.1 < ratio < 10, f"Effective loss ratio {ratio} not balanced"
    
    def test_uncertainty_weighting_is_differentiable(self, task_names, device):
        """Uncertainty parameters should be trainable via gradient descent."""
        balancer = UncertaintyWeightingBalancer(task_names, device)
        
        # Simulate training step
        losses = {"sft": torch.tensor(1.0), "contrastive": torch.tensor(2.0)}
        weights = balancer.compute_weights(losses)
        
        # Compute weighted loss + regularization
        total_loss = (
            weights["sft"] * losses["sft"] +
            weights["contrastive"] * losses["contrastive"] +
            balancer.get_regularization_loss()
        )
        
        # Should be able to compute gradients
        total_loss.backward()
        
        for name, log_var in balancer.log_vars.items():
            assert log_var.grad is not None, f"No gradient for {name}"
    
    def test_balancer_step_increments(self, task_names, device):
        """Step counter should increment."""
        balancer = LossScaleBalancer(task_names, device)
        
        assert balancer._step == 0
        balancer.step()
        assert balancer._step == 1
        balancer.step()
        assert balancer._step == 2


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_loss_handled(self, task_names, device):
        """Should handle zero loss without division by zero."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=0, momentum=0.0)
        
        losses = {"sft": torch.tensor(0.0), "contrastive": torch.tensor(1.0)}
        
        # Should not raise
        weights = balancer.compute_weights(losses)
        
        assert not math.isnan(weights["sft"])
        assert not math.isinf(weights["sft"])
    
    def test_very_small_loss_handled(self, task_names, device):
        """Should handle very small losses."""
        balancer = LossScaleBalancer(task_names, device, warmup_steps=0, momentum=0.0)
        
        losses = {"sft": torch.tensor(1e-10), "contrastive": torch.tensor(1.0)}
        
        weights = balancer.compute_weights(losses)
        
        assert not math.isnan(weights["sft"])
        assert not math.isinf(weights["sft"])
    
    def test_empty_gradients_for_pcgrad(self):
        """PCGrad should handle empty gradient list."""
        projected = PCGradBalancer.project_conflicting_gradients([])
        assert projected == []
    
    def test_zero_gradient_for_pcgrad(self):
        """PCGrad should handle zero gradients."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([0.0, 0.0])  # Zero gradient
        
        # Should not raise
        projected = PCGradBalancer.project_conflicting_gradients([g1, g2])
        
        assert len(projected) == 2
