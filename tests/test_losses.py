import torch
import torch.nn as nn

from craft.losses import InfoNCELoss, _pool_hidden_states, combine_craft_losses


def make_batch(batch_size: int, seq_len: int, hidden: int):
    hidden_anchor = torch.zeros(batch_size, seq_len, hidden)
    hidden_pos = torch.zeros_like(hidden_anchor)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    for i in range(batch_size):
        idx = seq_len - 1
        hidden_anchor[i, idx, i % hidden] = 1.0
        hidden_pos[i, idx, i % hidden] = 1.0

    return hidden_anchor, hidden_pos, mask.clone(), mask.clone()


def test_pool_hidden_states_last_token():
    hidden_anchor, _, mask_anchor, _ = make_batch(2, 4, 6)
    pooled = _pool_hidden_states(hidden_anchor, mask_anchor, "last_token")
    assert pooled.shape == (2, 6)
    assert torch.allclose(pooled[0], hidden_anchor[0, -1])


def test_infonce_forward_identity_projector():
    bsz, seq, dim = 4, 3, 8
    h_a, h_p, m_a, m_p = make_batch(bsz, seq, dim)

    loss_fn = InfoNCELoss(temperature=0.1, hidden_size=dim)
    loss_fn.projector = nn.Sequential(nn.Identity())

    loss = loss_fn(h_a, h_p, m_a, m_p)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_infonce_alignment_vs_mismatch():
    bsz, seq, dim = 6, 3, 10
    h_a, h_p, m_a, m_p = make_batch(bsz, seq, dim)

    loss_fn = InfoNCELoss(temperature=0.1, hidden_size=dim)
    loss_fn.projector = nn.Sequential(nn.Identity())

    aligned = loss_fn(h_a, h_p, m_a, m_p).item()
    perm = torch.randperm(bsz)
    mismatched = loss_fn(h_a, h_p[perm], m_a, m_p[perm]).item()

    assert aligned < mismatched


def test_infonce_return_details_contains_embeddings():
    h_a, h_p, m_a, m_p = make_batch(3, 4, 5)
    loss_fn = InfoNCELoss(temperature=0.2, hidden_size=5)
    loss_fn.projector = nn.Sequential(nn.Identity())

    loss, details = loss_fn(h_a, h_p, m_a, m_p, return_details=True)
    assert torch.isfinite(loss)
    assert "anchor_embeddings" in details
    assert details["anchor_embeddings"].shape == (3, 5)


def test_combine_craft_losses_balances_alpha():
    sft = torch.tensor(2.0)
    contrastive = torch.tensor(1.0)

    result = combine_craft_losses(sft_loss=sft, contrastive_loss=contrastive, alpha=0.25)
    assert torch.isclose(result.total_loss, torch.tensor(1.25))
    assert result.sft_loss is sft
    assert result.contrastive_loss is contrastive
