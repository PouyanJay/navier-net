"""The film pressure: the one the depth-averaged model cannot carry.

At the bubble's SIDES the pressure balancing the capillary pressure is not the
bulk liquid pressure but the pressure inside the Bretherton film between bubble
and gap wall -- ~4.9 um in a 150 um gap, so ~10^3x the drag and a large pressure
difference sustained over a tiny distance. That film lies in the gap direction,
which a depth-averaged model integrates away: one pressure per (x, y) cannot be
both the film's and the bulk's.

Measured, that shows up as a body-to-cap capillary offset of 1.25-1.48 that is
near-CONSTANT across every frame and across two independently trained runs,
while the variation along the body (0.21-0.82, and shrinking) is the neck. So the
correction is one scalar: enough to absorb the offset, structurally unable to
absorb the neck.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]
FILM = [*SHARP, "model.film_pressure=true"]


def _model(tmp_path, overrides):
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, overrides)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    return BubblePINN(cfg, geometry=_geometry_priors(cfg, data)), data, cfg


def test_film_pressure_is_off_by_default(tmp_path):
    """Opt-in: without it the jump is exactly the condition R4 shipped."""
    model, _, cfg = _model(tmp_path, SHARP)
    assert cfg.model.film_pressure is False
    assert not hasattr(model, "p_film_raw")


def test_film_pressure_requires_the_sharp_interface(tmp_path):
    """It corrects the Young-Laplace jump; without that condition there is
    nothing for it to correct."""
    from naviernet.models.pinn import BubblePINN

    cfg, _ = _staged_run(tmp_path, ["model=stage_b", "model.film_pressure=true"])
    with pytest.raises(ValueError, match="sharp_interface"):
        BubblePINN(cfg)


def test_the_offset_applies_to_the_body_and_never_to_the_caps(tmp_path):
    """The caps face bulk liquid -- there is no film there, so the model's own
    pressure is already the right one to compare against. Applying the offset
    there would be inventing a correction for a place that does not need it."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import laplace_jump_residual

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    front = model.nets["phi"].front(torch.tensor([[0.2], [0.8]]), n_body=16, n_cap=8)

    with torch.no_grad():
        model.p_film_raw.fill_(0.0)
        baseline = laplace_jump_residual(model, front, groups).clone()
        model.p_film_raw.fill_(0.7)
        shifted = laplace_jump_residual(model, front, groups)

    moved = (shifted - baseline).abs().squeeze(1)
    on_cap = front.on_cap.squeeze(1) == 1
    assert torch.allclose(moved[on_cap], torch.zeros_like(moved[on_cap]), atol=1e-7)
    assert (moved[~on_cap] > 0.5).all(), "the body must feel the whole offset"


def test_the_offset_cannot_absorb_variation_along_the_body(tmp_path):
    """The design constraint. The offset is ONE scalar, so it shifts every body
    sample equally and leaves the differences between them untouched -- those
    differences are the capillary gradient that drains the neck. An offset with
    spatial freedom would swallow the neck along with the discrepancy."""
    from naviernet.physics.groups import compute_groups
    from naviernet.physics.residuals import laplace_jump_residual

    model, data, cfg = _model(tmp_path, FILM)
    groups = compute_groups(cfg)
    front = model.nets["phi"].front(torch.tensor([[0.5]]), n_body=24, n_cap=6)
    body = front.on_cap.squeeze(1) == 0

    def body_variation():
        with torch.no_grad():
            r = laplace_jump_residual(model, front, groups).squeeze(1)[body]
        return r - r.mean()

    with torch.no_grad():
        model.p_film_raw.fill_(0.0)
        before = body_variation()
        model.p_film_raw.fill_(-1.3)
        after = body_variation()
    assert torch.allclose(before, after, atol=1e-6), (
        "a constant offset must leave the body's internal variation -- the neck "
        "signal -- exactly as it was"
    )


def test_the_offset_trains_and_is_recorded(tmp_path):
    """End to end: it is an inferred closure like the interfacial resistance, so
    it has to actually move and travel in the checkpoint."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, FILM)
    model, _, _ = train(cfg, paths)
    assert float(model.p_film_raw.abs()) > 0.0, "the film pressure never trained"

    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert "p_film_raw" in ckpt["model"]
    assert ckpt["film_pressure"] is True

    reloaded, _, _ = load_model(cfg, paths)
    assert float(reloaded.p_film_raw) == pytest.approx(float(model.p_film_raw))


def test_checkpoint_refuses_a_film_pressure_mismatch(tmp_path):
    """It adds a parameter and changes the objective, like every other
    architectural flag."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(tmp_path, FILM)
    train(cfg, paths)
    plain, _ = _staged_run(tmp_path, SHARP)
    with pytest.raises(ValueError, match="film_pressure"):
        load_model(plain, paths)


def test_the_diagnostic_scores_the_same_condition_the_trainer_optimises(tmp_path):
    """If the jump error were measured without the offset the run was trained
    with, the metric and the objective would be different conditions -- and the
    gate would be reading the wrong one."""
    from naviernet.physics.diagnostics import interface_diagnostics

    model, data, _ = _model(tmp_path, FILM)
    # The mutation is under no_grad; the diagnostic itself is not -- the front
    # sampler differentiates its own parameterization to get curvature and speed.
    with torch.no_grad():
        model.p_film_raw.fill_(0.0)
    without = interface_diagnostics(model, data).laplace_error_front
    with torch.no_grad():
        model.p_film_raw.fill_(-1.3)
    with_offset = interface_diagnostics(model, data).laplace_error_front
    assert without != pytest.approx(with_offset), (
        "the diagnostic ignored the film offset the residual applies"
    )
