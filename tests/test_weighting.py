"""Residual-Based Attention (RBA) weighting: bounded per-point self-adaptive weights.

The gradient-norm rebalancer these replace is unbounded and ratchets physics weights
to the 1e3 cap over long runs, collapsing the data fit. RBA's attention is bounded by
construction (eta/(1-gamma)); these tests pin that bound (the regression) and that it
up-weights the high-residual points while carrying no gradient.
"""

from __future__ import annotations

import pytest
import torch


def test_rba_attention_stays_bounded_over_a_long_run():
    """The ratchet regression: however large the residuals, the attention can never
    exceed eta/(1-gamma) -- unlike the gradnorm rebalancer that climbed to the 1e3 cap."""
    from naviernet.physics.weighting import rba_bound, rba_update

    gamma, eta = 0.999, 0.01
    attention = torch.zeros(8, 1)
    # Worst case for growth: one point is always the peak (norm 1.0) every step.
    residual_sq = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
    for _ in range(20000):
        attention = rba_update(attention, residual_sq, gamma, eta)

    bound = rba_bound(gamma, eta)
    assert bound == pytest.approx(10.0)
    assert attention.max().item() <= bound + 1e-6, "attention must stay under eta/(1-gamma)"
    assert attention.max().item() == pytest.approx(bound, rel=1e-3), (
        "peak point saturates the bound"
    )


def test_rba_attention_concentrates_on_high_residual_points():
    """Attention grows where the residual is large, so those points get up-weighted."""
    from naviernet.physics.weighting import rba_update

    attention = torch.zeros(3, 1)
    residual_sq = torch.tensor([[1e-4], [1.0], [100.0]])  # low, mid, high
    for _ in range(500):
        attention = rba_update(attention, residual_sq, 0.999, 0.01)

    a = attention.squeeze()
    assert a[2] > a[1] > a[0], "larger residual -> larger attention"


def test_rba_update_carries_no_gradient():
    """Attention steers which residuals matter; it must not be part of the autograd
    graph (it is updated from detached residual magnitudes)."""
    from naviernet.physics.weighting import rba_update

    residual_sq = (torch.rand(5, 1, requires_grad=True) + 0.1) ** 2
    updated = rba_update(torch.zeros(5, 1), residual_sq, 0.999, 0.01)
    assert not updated.requires_grad, "attention must be detached"


def test_rba_weighted_mean_reduces_to_the_plain_mean_at_zero_attention():
    """At attention=0 the multiplier (1+lambda) is 1, so an unattended term is exactly
    today's mean-squared residual -- the no-op init that keeps a fresh run == baseline."""
    from naviernet.physics.weighting import rba_weighted_mean

    residual_sq = torch.rand(16, 1)
    assert rba_weighted_mean(residual_sq, torch.zeros(16, 1)).item() == pytest.approx(
        residual_sq.mean().item(), rel=1e-6
    )


def test_rba_weighted_mean_up_weights_high_attention_points_and_keeps_gradient():
    """The weighted mean lifts the loss on high-attention points (vs the plain mean)
    and stays differentiable in the residual (so it still trains the model)."""
    from naviernet.physics.weighting import rba_weighted_mean

    residual_sq = torch.rand(8, 1, requires_grad=True)
    attention = torch.zeros(8, 1)
    attention[0] = 5.0  # attend hard to point 0

    weighted = rba_weighted_mean(residual_sq, attention)
    assert weighted.item() > residual_sq.mean().item(), "attention raises the loss"
    assert weighted.requires_grad, "the weighted loss still trains the model"


def test_rba_bound_guards_against_an_all_zero_residual():
    """A degenerate all-zero residual (peak 0) must not divide by zero -- attention just
    decays toward zero."""
    from naviernet.physics.weighting import rba_update

    updated = rba_update(torch.ones(4, 1), torch.zeros(4, 1), 0.999, 0.01)
    assert torch.all(torch.isfinite(updated))
    assert torch.all(updated < 1.0), "with no residual, attention decays (gamma < 1)"
