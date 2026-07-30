"""Residual-Based Attention (RBA) weighting (Anagnostopoulos et al., arXiv:2307.00379).

A per-collocation-point, per-term multiplicative weight (``attention``) that grows
where the PDE residual is large, so the optimiser attends to the hard regions (the
moving interface, stiff late times) instead of averaging them away. Unlike the
gradient-norm rebalancer it replaces -- which is unbounded and ratchets the physics
weights to the 1e3 cap over a long run, collapsing the data fit -- RBA is **bounded by
construction**: the update is a decaying EMA of the max-normalised residual, so the
attention can never exceed ``eta / (1 - gamma)`` no matter how large the residuals get.

The attention is **detached** (it steers which residuals matter, carrying no gradient);
the gradient still flows through the residual itself in :func:`rba_weighted_mean`.
"""

from __future__ import annotations

import torch

# Floors the max-normalisation so an all-zero residual batch (peak 0) decays the
# attention toward zero instead of dividing by zero. A numerical guard, not a tunable.
_PEAK_FLOOR = 1e-8


def rba_bound(gamma: float, eta: float) -> float:
    """The asymptotic upper bound on the attention, ``eta / (1 - gamma)``.

    The fixed point of ``lambda = gamma*lambda + eta*n`` with the normalised residual
    ``n`` in ``[0, 1]`` is ``lambda* = eta*n / (1 - gamma) <= eta / (1 - gamma)``; from a
    zero start the attention never exceeds it, which is the whole point of RBA.
    """
    return eta / (1.0 - gamma)


def rba_update(
    attention: torch.Tensor, residual_sq: torch.Tensor, gamma: float, eta: float
) -> torch.Tensor:
    """One bounded EMA step of the per-point attention (detached).

    ``lambda_i <- gamma*lambda_i + eta * |r_i| / max_j |r_j|`` where ``|r_i|`` is the
    residual magnitude (``sqrt(residual_sq)``) and the max is over the current fixed
    collocation pool. Returns the updated attention (same shape), carrying no gradient.
    """
    with torch.no_grad():
        magnitude = residual_sq.sqrt()
        peak = magnitude.max().clamp_min(_PEAK_FLOOR)
        return gamma * attention + eta * (magnitude / peak)


def rba_weighted_mean(residual_sq: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """The collocation loss for one term under RBA: ``mean((1 + lambda_i) * r_i^2)``.

    The ``1 +`` keeps a unit baseline, so at ``attention = 0`` this is exactly the plain
    mean-squared residual (a fresh run == the pre-RBA objective); attention only ever
    *lifts* the loss on the hard points. Differentiable in ``residual_sq`` -- the
    attention is a detached multiplier, not part of the graph.
    """
    return ((1.0 + attention) * residual_sq).mean()
