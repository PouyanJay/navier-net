"""The depth bypass (``model.depth_bypass``, physics/depth.py): the outline
displaces the liquid a bubble riding on films with rounded sides pushes aside,
not a full-depth slab's worth."""

from __future__ import annotations

import math

import pytest
import torch

from naviernet import training, warmup_cache
from naviernet.physics import depth, mass_balance
from naviernet.physics.groups import compute_groups
from naviernet.physics.residuals import kinematic_residual

from .conftest import staged_walled_capsule_run

EXACT = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.walls_at=physical",
    "model.hard_walls=true",
    "model.mirror_symmetry=true",
    "model.outlet_condition=zero_gradient",
    "model.exact_continuity=true",
    "model.exact_continuity_body_sources=32",
    "model.exact_continuity_cap_sources=8",
    "training.stage_b_warmup_steps=1",
    "training.freeze_geometry=true",
    "training.edge_transport_after_freeze=false",
    "training.steps=3",
    "training.log_every=1",
]


def _run(tmp_path, extra=()):
    return staged_walled_capsule_run(tmp_path, [*EXACT, "model.depth_bypass=true", *extra])


def test_a_bubble_as_wide_as_the_channel_is_deep_has_a_circular_section():
    nose, body = depth.cross_section_fractions(width=1.0, depth=1.0, film=0.0)

    assert nose == pytest.approx(math.pi / 4) and body == 1.0


def test_a_very_wide_bubble_displaces_almost_the_whole_slab():
    nose, body = depth.cross_section_fractions(width=100.0, depth=1.0, film=0.0)

    assert nose == pytest.approx(1.0, abs=3e-3) and body == 1.0


def test_the_films_take_their_share_of_the_depth_everywhere():
    nose, body = depth.cross_section_fractions(width=3.0, depth=1.0, film=0.05)

    assert body == pytest.approx(0.9)
    assert nose < body


def test_the_series_1_w300_bubble_leaves_a_sixth_of_the_slab_behind_its_head():
    # 234 um wide, 150 um deep, a 4.9 um film, in units of 300 um.
    nose, body = depth.cross_section_fractions(234 / 300, 150 / 300, 4.9 / 300)

    assert 0.80 < nose < 0.85
    assert 0.93 < body < 0.95


def test_the_bypass_scales_the_front_speed_in_the_kinematic_condition():
    class Front:
        points = torch.zeros(2, 3)
        normal = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        normal_speed = torch.tensor([[1.0], [2.0]])

    class Model:
        def velocity(self, x, c=None):
            return torch.tensor([[0.8], [1.6]]), torch.zeros(2, 1)

    residual = kinematic_residual(Model(), Front(), bypass=torch.tensor([[0.8], [0.8]]))

    assert torch.allclose(residual, torch.zeros(2, 1))


def test_a_run_engages_the_bypass_at_the_freeze_and_the_liquid_follows_it(tmp_path):
    cfg, paths = _run(tmp_path)

    model, data, _ = training.train(cfg, paths)

    assert model.bypass_engaged
    table = model._bypass_table
    assert bool((table.nose < table.body).all()) and bool((table.body < 1.0).all())
    t = torch.tensor([[float(data.t[2])]], requires_grad=True)
    front = model.front(t, 64, 16)
    u, v = model.velocity(front.points.detach())
    advance = (u * front.normal[:, 0:1] + v * front.normal[:, 1:2]).detach()
    speed = front.normal_speed.detach()
    reduced = speed * depth.on_front(model, front)
    assert float((speed - reduced).abs().mean()) > 1e-3, "the test must tell the two apart"
    assert float((advance - reduced).abs().mean()) < float((advance - speed).abs().mean())
    loaded, _, _ = training.load_model(cfg, paths)
    assert loaded.bypass_engaged and loaded._bypass_table is not None


def test_the_mass_balance_takes_out_the_liquid_the_outline_passes_over(tmp_path):
    cfg, paths = _run(tmp_path)
    model, data, _ = training.train(cfg, paths)
    plan = mass_balance.build_plan(
        data.domain, (data.y_lo, data.y_hi), compute_groups(cfg), cfg.training, "cpu"
    )

    sink = depth.mass_balance_sink(model, plan)
    with_bypass = depth.flux_defect(model, plan)
    same_flow = mass_balance.axial_flux_defect(model, plan)
    model.bypass_engaged = False
    off = depth.flux_defect(model, plan)

    assert sink.shape == (plan.n_times, plan.n_stations)
    assert torch.allclose(sink[:, 0], torch.zeros(plan.n_times)), "nothing upstream of x_min"
    assert float(sink.abs().max()) > 0.0
    # An absolute tolerance: a sink of 1e-7 is below float32's resolution of the defects.
    assert torch.allclose(with_bypass - same_flow, sink / plan.q_in, atol=1e-5, rtol=1e-4)
    assert torch.equal(off, mass_balance.axial_flux_defect(model, plan)), "plain when off"


def test_the_warm_up_key_ignores_the_bypass(tmp_path):
    on, paths = _run(tmp_path)
    off, _ = _run(tmp_path, ["model.depth_bypass=false"])
    keys = ("brinkman",)

    assert (
        warmup_cache.warmup_key(on, paths, keys)[0]
        == warmup_cache.warmup_key(off, paths, keys)[0]
    )


def test_the_bypass_needs_the_exact_liquid_flow(tmp_path):
    cfg, paths = staged_walled_capsule_run(
        tmp_path, [*EXACT, "model.exact_continuity=false", "model.depth_bypass=true"]
    )

    with pytest.raises(ValueError, match="exact_continuity"):
        training.train(cfg, paths)
