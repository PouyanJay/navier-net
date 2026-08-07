"""Time-binned collocation.

A construction that rebuilds the interface from a field pays that cost once per
distinct TIME in the batch. Independent per-point times make that unaffordable
(3072 rebuilds a step); binning makes it routine. These tests hold the binning to
the two things that make it safe to use: it must not bias the sample, and it must
leave the default path untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import staged_run as _staged_run


def _dataset(tmp_path):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.utils.paths import RunPaths

    cfg, _ = _staged_run(tmp_path)
    return BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")


def test_binning_off_is_the_independent_draw_it_always_was(tmp_path):
    """The default must be untouched, or every existing run's sampling changes."""
    data = _dataset(tmp_path)
    a = data.sample_collocation(256, np.random.default_rng(0))
    b = data.sample_collocation(256, np.random.default_rng(0), time_bins=0)
    assert torch.equal(a, b)


@pytest.mark.parametrize("bins", [1, 4, 16])
def test_binning_collapses_the_batch_onto_that_many_instants(tmp_path, bins):
    """The whole point: the number of distinct times is what the rebuild cost is
    proportional to."""
    data = _dataset(tmp_path)
    points = data.sample_collocation(512, np.random.default_rng(1), time_bins=bins)
    assert len(torch.unique(points[:, 2])) <= bins


def test_the_instants_are_stratified_across_the_window(tmp_path):
    """One instant per equal stratum, so the window stays covered. Clustered
    times would bias the residual toward whatever part of the run they landed in
    -- the failure this must not have."""
    data = _dataset(tmp_path)
    d = data.domain
    edges = np.linspace(d.t_min, d.t_max, 9)
    times = torch.unique(
        data.sample_collocation(2048, np.random.default_rng(2), time_bins=8)[:, 2].detach()
    ).numpy()

    assert len(times) == 8
    for lo, hi, t in zip(edges[:-1], edges[1:], np.sort(times), strict=True):
        assert lo <= t <= hi, f"instant {t} escaped its stratum [{lo}, {hi}]"


def test_binning_leaves_the_points_where_they_were_in_space(tmp_path):
    """Only the time coordinate is binned. The near-interface half was chosen at
    a frame, and moving it in x or y would take it off the interface it exists to
    sample."""
    data = _dataset(tmp_path)
    plain = data.sample_collocation(512, np.random.default_rng(3))
    binned = data.sample_collocation(512, np.random.default_rng(3), time_bins=8)
    assert torch.equal(plain[:, :2], binned[:, :2])


def test_every_time_lands_on_its_nearest_instant(tmp_path):
    """Snapped to the NEAREST instant, not reassigned to an arbitrary one: a point
    keeps the moment it was drawn for, as closely as the bins allow. Asserted as
    the contract itself rather than as a distance bound -- the worst-case distance
    depends on where the random instants fell, so a bound would either be loose
    enough to pass anything or tight enough to flake."""
    data = _dataset(tmp_path)
    plain = data.sample_collocation(1024, np.random.default_rng(4))[:, 2].detach().numpy()
    binned = data.sample_collocation(1024, np.random.default_rng(4), time_bins=8)[:, 2]
    binned = binned.detach().numpy()
    instants = np.unique(binned)

    nearest = instants[np.abs(plain[:, None] - instants[None, :]).argmin(axis=1)]
    assert np.array_equal(binned, nearest)


def test_binned_points_still_carry_gradient(tmp_path):
    """Collocation points are differentiated for every PDE residual; a rebuilt
    tensor that dropped requires_grad would fail deep in the loss instead."""
    data = _dataset(tmp_path)
    assert data.sample_collocation(64, np.random.default_rng(5), time_bins=4).requires_grad


def test_the_trainer_honours_the_binning_config(tmp_path):
    """The knob has to reach the sampler, or a run trains unbinned while the
    config says otherwise -- and the whole point is the cost of the times the
    trainer actually draws."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.training import train

    cfg, paths = _staged_run(tmp_path, ["training.collocation_time_bins=4"])
    assert cfg.training.collocation_time_bins == 4

    seen: list[int] = []
    original = BubbleDataset.sample_collocation

    def spy(self, n, rng, time_bins=0):
        seen.append(time_bins)
        return original(self, n, rng, time_bins)

    BubbleDataset.sample_collocation = spy
    try:
        train(cfg, paths)
    finally:
        BubbleDataset.sample_collocation = original

    assert seen, "the trainer never sampled collocation points"
    assert set(seen) == {4}, f"the trainer sampled with time_bins={set(seen)}"


def test_a_run_trains_end_to_end_with_binned_times(tmp_path):
    """The walking skeleton for this half: binning is a real training path, not
    just a sampler unit."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, ["training.collocation_time_bins=4"])
    train(cfg, paths)
    model, _, _ = load_model(cfg, paths)
    assert model is not None
