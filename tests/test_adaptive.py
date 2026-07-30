"""Residual-adaptive collocation (RAD, Wu et al. 2023): resample points toward where
the PDE residual is largest, keeping a uniform base for global coverage.

These pin the pure selection math: high-residual candidates get more probability, the
resampled pool concentrates in a synthetic high-residual band while still keeping the
base fraction elsewhere, and the whole thing is seed-deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_rad_probabilities_concentrate_on_high_residual():
    """Probability is proportional to residual magnitude (plus a uniform floor), so a
    high-residual point is sampled more than a low-residual one, and it is a valid
    distribution (non-negative, sums to 1)."""
    from naviernet.data.adaptive import rad_probabilities

    mag = torch.tensor([0.0, 1.0, 10.0, 100.0])
    p = rad_probabilities(mag)

    assert p.shape == mag.shape
    assert torch.all(p >= 0)
    assert p.sum().item() == pytest.approx(1.0)
    assert p[3] > p[2] > p[1] > p[0], "more probability on larger residual"
    # The uniform floor (c) keeps even the zero-residual point reachable.
    assert p[0] > 0


def test_rad_probabilities_are_uniform_when_residuals_are_equal():
    """With equal residuals the distribution is uniform -- no region is preferred."""
    from naviernet.data.adaptive import rad_probabilities

    p = rad_probabilities(torch.ones(20))
    assert torch.allclose(p, torch.full((20,), 1.0 / 20), atol=1e-6)


def test_rad_resample_biases_toward_the_high_residual_band_and_keeps_a_base():
    """Given candidates whose high-residual points sit in a spatial band, the resampled
    pool concentrates there far above the uniform expectation -- while the base fraction
    still comes from the global (uniform) sampler, so no region is abandoned."""
    from naviernet.data.adaptive import rad_resample

    # 1000 candidates on [0,1]; those with x in [0.4, 0.6] carry a huge residual.
    cand_x = torch.rand(1000, 3)
    in_band = (cand_x[:, 0] > 0.4) & (cand_x[:, 0] < 0.6)
    mag = torch.where(in_band, torch.tensor(100.0), torch.tensor(0.01))
    base_x = torch.rand(1000, 3)  # global uniform base

    def band_frac(pool):
        return ((pool[:, 0] > 0.4) & (pool[:, 0] < 0.6)).float().mean().item()

    residual_weighted = rad_resample(cand_x, mag, base_x, 200, 0.5, np.random.default_rng(0))
    # Control: flat residuals -> RAD is uniform, so the band sits at its ~20% area share.
    flat = rad_resample(cand_x, torch.ones(1000), base_x, 200, 0.5, np.random.default_rng(0))

    assert residual_weighted.shape == (200, 3)
    assert band_frac(residual_weighted) > 1.5 * band_frac(flat), (
        "residual weighting must concentrate in the high-residual band vs flat weighting"
    )
    assert band_frac(residual_weighted) < 0.85, (
        "the uniform base keeps coverage outside the band"
    )


def test_rad_resample_preserves_the_base_fraction_and_count():
    """The pool is exactly `n` points: round(n*fraction) adaptive + the rest from the
    base sampler."""
    from naviernet.data.adaptive import rad_resample

    rng = np.random.default_rng(0)
    cand_x = torch.rand(500, 3)
    mag = torch.rand(500)
    base_x = torch.full((500, 3), -1.0)  # sentinel: base points are identifiable

    pool = rad_resample(cand_x, mag, base_x, n=100, fraction=0.5, rng=rng)

    assert pool.shape == (100, 3)
    n_base = int((pool == -1.0).all(dim=1).sum())
    assert n_base == 50, "half the pool is the uniform base at fraction=0.5"


def test_rad_resample_handles_the_fraction_edges():
    """fraction=0 -> the pool is entirely the uniform base; fraction=1 with fewer
    candidates than requested clamps to the candidate count (no over-draw), the rest
    filled from the base."""
    from naviernet.data.adaptive import rad_resample

    rng = np.random.default_rng(0)
    base_x = torch.full((100, 3), -1.0)  # sentinel base

    all_base = rad_resample(
        torch.rand(50, 3), torch.rand(50), base_x, n=40, fraction=0.0, rng=rng
    )
    assert all_base.shape == (40, 3)
    assert bool((all_base == -1.0).all()), "fraction=0 -> pool is entirely the base"

    # fraction=1 asks for 40 adaptive but only 10 candidates exist -> 10 adaptive + 30 base.
    small_cand = torch.rand(10, 3)
    clamped = rad_resample(small_cand, torch.rand(10), base_x, n=40, fraction=1.0, rng=rng)
    assert clamped.shape == (40, 3)
    assert int((clamped == -1.0).all(dim=1).sum()) == 30, (
        "the shortfall is filled from the base"
    )


def test_rad_resample_is_seed_deterministic():
    """Same seed -> same pool, so golden/reproducibility hold."""
    from naviernet.data.adaptive import rad_resample

    cand_x, mag, base_x = torch.rand(300, 3), torch.rand(300), torch.rand(300, 3)
    a = rad_resample(cand_x, mag, base_x, 80, 0.5, np.random.default_rng(7))
    b = rad_resample(cand_x, mag, base_x, 80, 0.5, np.random.default_rng(7))
    assert torch.equal(a, b)
