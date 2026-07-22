"""Governing physics: dimensionless groups and autodiff PDE residuals."""

from naviernet.physics.groups import (
    compute_groups,
    hydraulic_diameter_m,
    inlet_velocity_m_s,
    reference_time_ms,
    save_groups,
)
from naviernet.physics.residuals import (
    StageAResiduals,
    boundary_losses,
    gradients,
    interface_indicator,
    source_penalty,
    stage_a_residuals,
)

__all__ = [
    "StageAResiduals",
    "boundary_losses",
    "compute_groups",
    "gradients",
    "hydraulic_diameter_m",
    "inlet_velocity_m_s",
    "interface_indicator",
    "reference_time_ms",
    "save_groups",
    "source_penalty",
    "stage_a_residuals",
]
