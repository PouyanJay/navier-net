"""Interface sharpening: annealing alpha_eps so the neck is resolvable.

`alpha = sigmoid(phi / alpha_eps)` puts the interface's transition over roughly
4 * alpha_eps. At the shipped 0.05 that is ~0.2 in y* -- and the measured
frame-11 neck is 0.224 wide IN TOTAL. The feature the physics is supposed to
produce is the same size as the blur it is produced through, so it is annealed
down over training instead of being fixed at a value chosen for Stage A.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import staged_run as _staged_run

SHARP = ["model=stage_b", "model.front_geometry=true", "model.sharp_interface=true"]


def test_annealing_is_off_by_default(tmp_path):
    """Opt-in: alpha_eps holds at its configured value unless asked otherwise."""
    from naviernet.training import annealed_alpha_eps

    cfg, _ = _staged_run(tmp_path)
    assert cfg.training.alpha_eps_anneal_steps == 0
    for step in (1, 10, 1000):
        assert annealed_alpha_eps(step, cfg) == pytest.approx(cfg.model.alpha_eps)


def test_annealing_walks_geometrically_to_the_final_value(tmp_path):
    """Geometric, not linear: alpha_eps is a scale, so equal FRACTIONAL steps are
    what keep the sharpening gradual near the end where it bites hardest."""
    from naviernet.training import annealed_alpha_eps

    cfg, _ = _staged_run(
        tmp_path,
        [
            "model.alpha_eps=0.04",
            "training.alpha_eps_final=0.01",
            "training.alpha_eps_anneal_steps=100",
        ],
    )
    assert annealed_alpha_eps(1, cfg) == pytest.approx(0.04, rel=0.05)
    assert annealed_alpha_eps(50, cfg) == pytest.approx(0.02, rel=0.05)
    assert annealed_alpha_eps(100, cfg) == pytest.approx(0.01, rel=1e-6)
    # and it holds there afterwards rather than continuing to collapse
    assert annealed_alpha_eps(500, cfg) == pytest.approx(0.01, rel=1e-6)


def test_annealing_rejects_a_widening_or_non_positive_target(tmp_path):
    """alpha_eps must stay positive (it divides phi) and the schedule sharpens."""
    from naviernet.training import _validate_training_config

    for bad in ("training.alpha_eps_final=0.0", "training.alpha_eps_final=0.5"):
        cfg, _ = _staged_run(
            tmp_path, ["model.alpha_eps=0.05", bad, "training.alpha_eps_anneal_steps=10"]
        )
        with pytest.raises(ValueError, match="alpha_eps_final"):
            _validate_training_config(cfg.training, cfg.model.fields, cfg.model.alpha_eps)


def test_a_run_ends_sharper_and_evaluates_at_the_sharpened_value(tmp_path):
    """The value the run ENDED at travels in the checkpoint. Without that, the
    model would be scored through a blur it was not trained with."""
    from naviernet.training import load_model, train

    cfg, paths = _staged_run(
        tmp_path,
        [
            *SHARP,
            "model.alpha_eps=0.05",
            "training.alpha_eps_final=0.02",
            "training.alpha_eps_anneal_steps=2",
            "training.steps=2",
        ],
    )
    model, _, _ = train(cfg, paths)
    assert model.eps == pytest.approx(0.02)

    reloaded, _, _ = load_model(cfg, paths)
    assert reloaded.eps == pytest.approx(0.02), (
        "evaluation must use the interface thickness the run ended at"
    )
    ckpt = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["alpha_eps"] == pytest.approx(0.02)


def test_sharpening_narrows_the_predicted_interface(tmp_path):
    """The point of the exercise: a smaller alpha_eps really does put the
    alpha transition into a narrower band, so a neck can be told from a blur."""
    from naviernet.data.dataset import BubbleDataset
    from naviernet.models.pinn import BubblePINN
    from naviernet.training import _geometry_priors
    from naviernet.utils.paths import RunPaths

    cfg, paths = _staged_run(tmp_path, SHARP)
    data = BubbleDataset(cfg, RunPaths.from_config(cfg), device="cpu")
    model = BubblePINN(cfg, geometry=_geometry_priors(cfg, data))

    d = data.domain
    column = torch.stack(
        [
            torch.full((400,), 0.5 * (d.x_min + d.x_max)),
            torch.linspace(d.y_min, d.y_max, 400),
            torch.full((400,), 0.5 * d.t_max),
        ],
        dim=1,
    )

    def transition_width():
        with torch.no_grad():
            a = model.alpha(column).squeeze()
        inside = (a > 0.1) & (a < 0.9)
        y = column[:, 1][inside]
        return float(y.max() - y.min()) if inside.any() else 0.0

    model.eps = 0.05
    blurred = transition_width()
    model.eps = 0.01
    sharpened = transition_width()
    assert sharpened < 0.5 * blurred, f"{blurred:.4f} -> {sharpened:.4f}"


def test_a_joint_run_also_evaluates_at_the_sharpened_value(tmp_path):
    """`load_model` restores the annealed thickness; `load_joint` must too, or a
    joint run is silently scored through a blur it was never trained with."""
    from naviernet.training import load_joint, train
    from tests.conftest import staged_joint_run

    cfg, paths = staged_joint_run(
        tmp_path,
        ["model.front_geometry=true", "model.alpha_eps=0.05",
         "training.alpha_eps_final=0.02", "training.alpha_eps_anneal_steps=2"],
    )
    train(cfg, paths)

    model, _, _ = load_joint(cfg, paths)
    assert model.eps == pytest.approx(0.02), (
        "a joint run must be evaluated at the thickness it ended on"
    )
