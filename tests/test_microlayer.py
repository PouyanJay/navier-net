"""Microlayer wetting of the footprint: the bubble leaves a film on the floor it
covers, the floor's flux evaporates it, and a spot makes vapour only while its
film lasts, so the vapour follows newly covered floor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from naviernet import training
from naviernet.data.dataset import BubbleDataset
from naviernet.models.pinn import BubblePINN
from naviernet.physics.groups import compute_groups
from naviernet.physics.heating import NEVER_COVERED, cover_times
from naviernet.training import _geometry_priors, bind_cover_times

from .conftest import make_config, staged_capsule_run

MICROLAYER = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "model.footprint_evaporation=true",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "experiment.footprint_wetting=microlayer",
    "experiment.microlayer_um=0.5",
]


def _frames(alpha: list[list[float]], times: list[float]):
    """A one-row strip of pixels over a few frames."""
    stack = np.array(alpha, dtype=float)[:, None, :]
    return SimpleNamespace(
        alpha=stack, valid=np.ones_like(stack), t=np.array(times), n_event=len(times)
    )


def test_cover_times_read_the_first_crossing_between_frames():
    data = _frames([[1.0, 0.0, 0.0], [1.0, 0.25, 0.0], [1.0, 0.75, 0.0]], [0.0, 1.0, 2.0])

    times = cover_times(data, age_star=0.3)[0]

    assert times[0] == pytest.approx(-0.3), "the first frame's footprint is as old as declared"
    assert times[1] == pytest.approx(1.5), "half way between the frames around the crossing"
    assert times[2] == NEVER_COVERED


def test_the_film_life_is_its_thickness_over_the_floor_flux():
    groups = compute_groups(
        make_config(
            [*MICROLAYER, "experiment.q_wall_W_cm2=2.0", "experiment.footprint_age_ms=0.5"]
        )
    )
    cfg = make_config(MICROLAYER)
    fluid, t_ref = cfg.fluid, cfg.scales.L_ref_um * 1e-6 / cfg.scales.U_ref

    assert groups["microlayer_tau_star_per_um"] == pytest.approx(
        fluid.rho_l * fluid.h_lv * 1e-6 / 2.0e4 / t_ref
    )
    assert groups["footprint_age_star"] == pytest.approx(0.5e-3 / t_ref)


def _bound_model(tmp_path, overrides):
    cfg, paths = staged_capsule_run(tmp_path, [*MICROLAYER, *overrides])
    data = BubbleDataset(cfg, paths)
    torch.manual_seed(0)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))
    bind_cover_times(cfg, model, data)
    return cfg, data, model


def test_a_spot_is_wet_while_its_film_lasts_and_dry_after(tmp_path):
    cfg, data, model = _bound_model(tmp_path, [])
    xs, ys, times = model._cover
    row, col = np.unravel_index(int(torch.argmin(times)), times.shape)
    covered_at = float(times[row, col])
    life = compute_groups(cfg)["microlayer_tau_star_per_um"] * 0.5
    spot = lambda t: torch.tensor([[float(xs[col]), float(ys[row]), t]])  # noqa: E731
    local = torch.ones(1, 1)

    fresh = float(model._wetness(spot(covered_at), local))
    spent = float(model._wetness(spot(covered_at + 2.0 * life), local))
    weak_flux = float(model._wetness(spot(covered_at + 2.0 * life), 0.25 * local))

    assert fresh > 0.99
    assert spent < 0.01
    assert weak_flux > 0.99, "a quarter of the flux makes the film last four times longer"


def test_the_fitted_film_starts_at_its_declared_value_and_takes_a_gradient(tmp_path):
    _, data, model = _bound_model(tmp_path, ["experiment.microlayer_mode=fit"])
    t = float(data.t[len(data.t) // 2])
    x = torch.rand(256, 3) * torch.tensor([float(data.x.max()), float(data.y.max()), 0.0])
    x[:, 2] = t

    assert float(model.microlayer_um.detach()) == pytest.approx(0.5)
    model.footprint_source(x).sum().backward()
    assert model._log_microlayer_raw.grad is not None


def test_the_source_needs_the_cover_times(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, MICROLAYER)
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, BubbleDataset(cfg, paths)))

    with pytest.raises(RuntimeError, match="bind_cover_times"):
        model.footprint_source(torch.zeros(4, 3))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("experiment.microlayer_um=0", "microlayer_um"),
        ("experiment.footprint_age_ms=-1", "footprint_age_ms"),
        ("experiment.footprint_wetting=patchy", "footprint_wetting"),
    ],
)
def test_an_impossible_microlayer_is_refused(override, message):
    with pytest.raises(ValueError, match=message):
        compute_groups(make_config([*MICROLAYER, override]))


def test_a_microlayer_run_trains_and_reloads(tmp_path):
    cfg, paths = staged_capsule_run(
        tmp_path,
        [*MICROLAYER, "experiment.microlayer_mode=fit", "training.steps=2"],
    )
    training.train(cfg, paths)

    model, _, _ = training.load_model(cfg, paths)

    assert model._cover is not None
    assert float(model.microlayer_um.detach()) > 0.0
