import torch

from craft.metrics import (
    compute_contrastive_accuracy,
    compute_representation_consistency,
    update_representation_reference,
)


def test_contrastive_accuracy_top1():
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    acc = compute_contrastive_accuracy(anchor, positives)
    assert torch.isclose(acc, torch.tensor(1.0))


def test_representation_consistency_with_reference():
    current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sim = compute_representation_consistency(current, reference)
    assert torch.isclose(sim, torch.tensor(1.0))


def test_update_representation_reference():
    prev = torch.tensor([[0.0, 0.0]])
    current = torch.tensor([[1.0, 1.0]])
    updated = update_representation_reference(prev, current, momentum=0.5)
    assert torch.allclose(updated, torch.tensor([[0.5, 0.5]]))
