"""Sequential temperature solve: after ``training.temperature_stage_from`` the
outline and the flow hold still and only the temperature net trains, on the
energy equation and the thermal conditions, from a restarted learning rate."""

from __future__ import annotations

import pytest
import torch

from naviernet import training

from .conftest import staged_capsule_run

THREE_STAGE = [
    "model=stage_b",
    "model.front_geometry=true",
    "model.sharp_interface=true",
    "model.thermal_model=sharp",
    "experiment.T_inlet_C=56.6",
    "experiment.T_init_C=58.0",
    "training.stage_b_warmup_steps=2",
    "training.freeze_geometry=true",
    "training.temperature_stage_from=3",
    "training.log_every=1",
]


def _params(model, keep) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if keep(k)}


def _changed(before: dict, after: dict) -> list[str]:
    return [k for k in before if not torch.equal(before[k], after[k])]


def _is_temperature(key: str) -> bool:
    return key.startswith("nets.T.")


def test_only_the_temperature_trains_after_the_stage_starts(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*THREE_STAGE, "training.steps=3"])
    flow_stage, _, _ = training.train(cfg, paths)
    temperature = _params(flow_stage, _is_temperature)
    rest = _params(flow_stage, lambda k: not _is_temperature(k))

    cfg, paths = staged_capsule_run(tmp_path, [*THREE_STAGE, "training.steps=2"])
    trained, _, state = training.train(cfg, paths)

    assert state["done"] == 5
    assert _changed(rest, _params(trained, lambda k: not _is_temperature(k))) == [], (
        "the outline, the flow and the scalars are locked"
    )
    assert _changed(temperature, _params(trained, _is_temperature)), "the temperature trains"


def test_the_stage_logs_the_temperature_terms_alone_at_a_restarted_rate(tmp_path):
    cfg, paths = staged_capsule_run(tmp_path, [*THREE_STAGE, "training.steps=5"])
    _, _, state = training.train(cfg, paths)

    last = state["hist"][-1]
    assert last["step"] == 5
    assert last["lr"] == pytest.approx(cfg.training.lr), "the clock restarts at the stage"
    assert {"energy_liquid", "theta_bc"} <= set(last)
    assert not {"data", "brinkman", "div"} & set(last), "the flow terms are not evaluated"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["training.freeze_geometry=false"], "freeze_geometry"),
        (["training.temperature_stage_from=2"], "after the warm-up"),
        (["training.temperature_stage_from=-1"], ">= 0"),
    ],
)
def test_a_temperature_stage_without_its_preconditions_is_refused(tmp_path, overrides, message):
    cfg, paths = staged_capsule_run(tmp_path, [*THREE_STAGE, *overrides])

    with pytest.raises(ValueError, match=message):
        training.train(cfg, paths)
