"""The catalogue of characterised working fluids.

Each fluid is a Hydra config group under ``configs/fluid/<id>.yaml``; this service
enumerates those files and composes each so the API never re-states a property
value the pipeline already owns. The set of stems here is also the allow-list of
fluid choices a series may select (see ``datasets.save_conditions``).
"""

from __future__ import annotations

from functools import lru_cache

from omegaconf import DictConfig

from naviernet.config import config_dir
from naviernet_api.models import Fluid
from naviernet_api.services.config_service import compose_cfg

# The abstract schema node registered in code, not a selectable fluid file.
_TEMPLATE_STEM = "base_fluid"

# Any dataset composes here: the fluid properties depend only on the selected
# fluid group, not on `dataset` (which feeds path interpolation, not `cfg.fluid`).
_COMPOSE_DATASET = "highest_t"


@lru_cache(maxsize=1)
def available_fluid_ids() -> tuple[str, ...]:
    """Sorted config-group stems under ``configs/fluid/`` (the selection
    allow-list). Cached: the config directory is fixed for the process."""
    fluid_dir = config_dir() / "fluid"
    stems = sorted(p.stem for p in fluid_dir.glob("*.yaml") if p.stem != _TEMPLATE_STEM)
    return tuple(stems)


def is_known_fluid(fluid_id: str) -> bool:
    """Whether ``fluid_id`` names a selectable fluid config group."""
    return fluid_id in available_fluid_ids()


def _fluid_from_cfg(fluid_id: str, cfg: DictConfig) -> Fluid:
    fluid_cfg = cfg.fluid
    return Fluid(
        id=fluid_id,
        name=fluid_cfg.name,
        T_sat_C=fluid_cfg.T_sat_C,
        rho_l=fluid_cfg.rho_l,
        rho_v=fluid_cfg.rho_v,
        mu_l=fluid_cfg.mu_l,
        mu_v=fluid_cfg.mu_v,
        k_l=fluid_cfg.k_l,
        k_v=fluid_cfg.k_v,
        cp_l=fluid_cfg.cp_l,
        cp_v=fluid_cfg.cp_v,
        sigma=fluid_cfg.sigma,
        h_lv=fluid_cfg.h_lv,
    )


def list_fluids() -> list[Fluid]:
    """Every characterised fluid with its saturated properties, composed from
    its config group. The dataset is immaterial here (only ``cfg.fluid`` is
    read), so the pipeline's default experiment is used to compose."""
    fluids: list[Fluid] = []
    for fluid_id in available_fluid_ids():
        cfg = compose_cfg(_COMPOSE_DATASET, overrides=[f"fluid={fluid_id}"])
        fluids.append(_fluid_from_cfg(fluid_id, cfg))
    return fluids
