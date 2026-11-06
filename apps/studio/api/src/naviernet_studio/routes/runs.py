"""Runs: the roster, one run's summary and frame roster, and the artifacts the
pipeline already wrote."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from naviernet_studio import ids
from naviernet_studio.routes.common import scene_or_404
from naviernet_studio.scene import Scene
from naviernet_studio.settings import Settings, get_settings

router = APIRouter(prefix="/api/runs", tags=["runs"])

ArtifactName = Literal["audit", "metrics", "trajectory", "front_velocity", "config"]


def _summary(settings: Settings, run_id: str) -> dict:
    directory = settings.outputs_dir / run_id
    metrics = ids.read_json(directory / "metrics.json") or {}
    audit = ids.read_json(directory / "audit.json") or {}
    conditions = _conditions(directory)
    return {
        "id": run_id,
        "dataset": metrics.get("dataset") or ids.dataset_of(directory),
        "iou_mean": metrics.get("iou_mean"),
        "iou_holdout": metrics.get("iou_holdout"),
        "nose_speed_mm_s": metrics.get("nose_speed_mm_s"),
        "n_frames": len(metrics.get("iou_per_frame", []) or []),
        "verdicts": {k: v.get("pass") for k, v in (audit.get("verdicts") or {}).items()},
        **conditions,
    }


def _conditions(directory) -> dict:
    snapshot = directory / ".hydra" / "config.yaml"
    if not snapshot.is_file():
        return {"flow_rate_mL_hr": None, "T_init_C": None, "has_config": False}
    from omegaconf import OmegaConf

    raw = OmegaConf.load(snapshot)
    experiment = raw.get("experiment", {})
    return {
        "flow_rate_mL_hr": experiment.get("flow_rate_mL_hr"),
        "T_init_C": experiment.get("T_init_C"),
        "q_wall_W_cm2": experiment.get("q_wall_W_cm2"),
        "has_config": True,
    }


@router.get("")
def list_runs(settings: Settings = Depends(get_settings)) -> list[dict]:
    """Every run with a checkpoint, newest first, with its headline numbers."""
    return [_summary(settings, run_id) for run_id in ids.list_run_ids(settings)]


@router.get("/{run}")
def run_detail(scene: Scene = Depends(scene_or_404)) -> dict:
    """What this run is and what the studio can ask of it."""
    cfg = scene.cfg
    return {
        "id": scene.run_id,
        "dataset": str(cfg.dataset),
        "fields": scene.fields,
        "has_front": scene.has_front,
        "has_footprint": scene.has_footprint,
        "sharp_thermal": scene.sharp_thermal,
        "equations": scene.equation_ids,
        "inertia": scene.inertia,
        "frames": [asdict(f) for f in scene.frames()],
        "scales": scene.scales.as_dict(),
        "groups": {k: float(v) for k, v in scene.groups.items() if isinstance(v, (int, float))},
        "conditions": {
            "flow_rate_mL_hr": float(cfg.experiment.flow_rate_mL_hr),
            "T_init_C": _optional_float(cfg.experiment.T_init_C),
            "T_inlet_C": _optional_float(cfg.experiment.T_inlet_C),
            "T_sat_C": float(cfg.experiment.T_sat_C),
            "q_wall_W_cm2": float(cfg.experiment.q_wall_W_cm2),
            "channel_width_um": float(cfg.experiment.channel_width_um),
            "channel_height_um": float(cfg.experiment.channel_height_um),
            "fluid": str(cfg.fluid.name),
        },
        "domain_um": {
            "x": [
                float(scene.data.domain.x_min) * scene.scales.length_um,
                float(scene.data.domain.x_max) * scene.scales.length_um,
            ],
            "y_walls": [
                float(scene.data.y_lo) * scene.scales.length_um,
                float(scene.data.y_hi) * scene.scales.length_um,
            ],
            "y_image": [
                float(scene.data.domain.y_min) * scene.scales.length_um,
                float(scene.data.domain.y_max) * scene.scales.length_um,
            ],
            "centreline": 0.5
            * float(scene.data.y_lo + scene.data.y_hi)
            * scene.scales.length_um,
        },
    }


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


@router.get("/{run}/artifacts/{name}")
def artifact(name: ArtifactName, scene: Scene = Depends(scene_or_404)) -> dict:
    """A pipeline artifact verbatim: audit.json, metrics.json, trajectory.json,
    front_velocity.json or the resolved config snapshot."""
    if name == "config":
        from omegaconf import OmegaConf

        return OmegaConf.to_container(scene.cfg, resolve=True)  # type: ignore[return-value]
    path = {
        "audit": scene.paths.audit_json,
        "metrics": scene.paths.metrics_json,
        "trajectory": scene.paths.trajectory_json,
        "front_velocity": scene.paths.front_velocity_json,
    }[name]
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{scene.run_id} wrote no {path.name}")
    return json.loads(path.read_text())
