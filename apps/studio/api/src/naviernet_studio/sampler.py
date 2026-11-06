"""Evaluate catalogued point quantities at arbitrary (x*, y*, t*) points.

One pass builds every intermediate a batch of quantities shares (the velocity,
its gradients, the pressure, the superheat), then each quantity is read from
that bag and converted with the run's scales. Autograd is enabled only when a
requested quantity needs it; points of one instant should be batched together
because the exact-continuity potentials are solved per distinct instant.
"""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np
import torch

from naviernet.models.pinn import RATE_STEP
from naviernet.physics.heating import wall_flux_ratio
from naviernet.physics.residuals import (
    _laplacian,
    _momentum_phase,
    confinement_drag,
    gradients,
    mixture,
    stage_a_residuals,
    vapour_potential_residual,
)
from naviernet_studio import catalogue
from naviernet_studio.scene import Scene

CHUNK = 2048
# The depth-profile factor of the convective inertia under a parabolic gap flow.
DEPTH_PROFILE_CONVECTION = 1.2
VAPOUR_LEVEL = 0.5
GRAD_KINDS = (catalogue.KIND_RESIDUAL, catalogue.KIND_TERM, catalogue.KIND_DERIVED)


class _Bag:
    """Lazily computed intermediates for one chunk of points."""

    def __init__(self, scene: Scene, x: torch.Tensor):
        self.scene = scene
        self.model = scene.model
        self.groups = scene.groups
        self.x = x

    @cached_property
    def alpha(self) -> torch.Tensor:
        return self.model.alpha(self.x)

    @cached_property
    def phi(self) -> torch.Tensor:
        if getattr(self.model, "front_geometry", False):
            return self.model._phi_with_frozen_geometry(self.x, None)
        return self.model.phi(self.x)

    @cached_property
    def liquid(self) -> torch.Tensor:
        """1 in the liquid, 0 in the vapour (the exact side when the run has it)."""
        if getattr(self.model, "exact_liquid", False):
            return self.model.liquid_side(self.x).unsqueeze(1).float()
        return (self.alpha < VAPOUR_LEVEL).float()

    @cached_property
    def phase(self) -> torch.Tensor:
        """The vapour fraction the momentum law takes its properties from: the
        explicit front's own side under momentum_phases=sharp, else alpha."""
        return _momentum_phase(self.model, self.x, None)

    @cached_property
    def velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.velocity(self.x)

    @cached_property
    def u_grad(self):
        return gradients(self.velocity[0], self.x)

    @cached_property
    def v_grad(self):
        return gradients(self.velocity[1], self.x)

    @cached_property
    def u_t(self) -> torch.Tensor:
        """The liquid velocity's time derivative by a central difference: under
        exact continuity the potentials carry no time graph, so autograd in t
        reads only the vapour networks."""
        if not getattr(self.model, "exact_liquid", False):
            return self.u_grad[2]
        with torch.no_grad():
            ahead = self.x.detach().clone()
            behind = self.x.detach().clone()
            ahead[:, 2] += RATE_STEP
            behind[:, 2] -= RATE_STEP
            u_ahead, _ = self.model.velocity(ahead)
            u_behind, _ = self.model.velocity(behind)
        return (u_ahead - u_behind) / (2.0 * RATE_STEP)

    @cached_property
    def v_t(self) -> torch.Tensor:
        if not getattr(self.model, "exact_liquid", False):
            return self.v_grad[2]
        with torch.no_grad():
            ahead = self.x.detach().clone()
            behind = self.x.detach().clone()
            ahead[:, 2] += RATE_STEP
            behind[:, 2] -= RATE_STEP
            _, v_ahead = self.model.velocity(ahead)
            _, v_behind = self.model.velocity(behind)
        return (v_ahead - v_behind) / (2.0 * RATE_STEP)

    @cached_property
    def p_liquid(self) -> torch.Tensor:
        return self.model.liquid_pressure(self.x)

    @cached_property
    def p_vapour(self) -> torch.Tensor:
        return self.model.p_vapor(self.x[:, 2:3])

    @cached_property
    def pressure(self) -> torch.Tensor:
        return self.model.pressure(self.x)

    @cached_property
    def p_grad(self):
        return gradients(self.p_liquid, self.x)

    @cached_property
    def theta(self) -> torch.Tensor:
        return self.model.temperature(self.x)

    @cached_property
    def theta_grad(self):
        return gradients(self.theta, self.x)

    @cached_property
    def source(self) -> torch.Tensor:
        return self.model.source(self.x)

    @cached_property
    def footprint(self) -> torch.Tensor:
        if self.scene.has_footprint:
            return self.model.footprint_source(self.x)
        return torch.zeros_like(self.alpha)

    @cached_property
    def stage_a(self):
        return stage_a_residuals(self.model, self.x)

    @cached_property
    def momentum_terms(self) -> dict[str, torch.Tensor]:
        """The x-momentum terms of the trained Brinkman law, unweighted."""
        u, v = self.velocity
        u_x, u_y, _ = self.u_grad
        drag = confinement_drag(self.phase, self.groups)
        viscosity = mixture(self.phase, self.groups["mu_ratio"]) / self.groups["Re"]
        density = mixture(self.phase, self.groups["rho_ratio"])
        terms = {
            "mom_dpdx": self.p_grad[0],
            "mom_friction": drag * u,
            "mom_viscous": -viscosity * _laplacian(u, self.x),
        }
        if self.scene.inertia:
            terms["mom_unsteady"] = density * self.u_t
            terms["mom_convective"] = density * DEPTH_PROFILE_CONVECTION * (u * u_x + v * u_y)
        else:
            terms["mom_unsteady"] = torch.zeros_like(u)
            terms["mom_convective"] = torch.zeros_like(u)
        return terms

    @cached_property
    def momentum_y(self) -> torch.Tensor:
        u, v = self.velocity
        v_x, v_y, _ = self.v_grad
        drag = confinement_drag(self.phase, self.groups)
        viscosity = mixture(self.phase, self.groups["mu_ratio"]) / self.groups["Re"]
        residual = self.p_grad[1] + drag * v - viscosity * _laplacian(v, self.x)
        if self.scene.inertia:
            density = mixture(self.phase, self.groups["rho_ratio"])
            residual = residual + density * (
                self.v_t + DEPTH_PROFILE_CONVECTION * (u * v_x + v * v_y)
            )
        return residual

    @cached_property
    def energy_terms(self) -> dict[str, torch.Tensor]:
        u, v = self.velocity
        t_x, t_y, t_t = self.theta_grad
        heat = (
            self.groups["q_wall_star"]
            * wall_flux_ratio(self.x, self.groups)
            * (1.0 - self.alpha)
        )
        return {
            "en_rate": t_t,
            "en_advection": u * t_x + v * t_y,
            "en_diffusion": -_laplacian(self.theta, self.x) / self.groups["Pe"],
            "en_heating": -heat,
        }


def _masked(value: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """NaN where ``keep`` is 0: the studio draws nulls as gaps, never as zeros."""
    return torch.where(keep > 0.5, value, torch.full_like(value, float("nan")))


def _read(bag: _Bag, quantity_id: str) -> torch.Tensor:  # noqa: PLR0911, PLR0912
    """The quantity in MODEL units at the bag's points, ``(N, 1)``."""
    q = quantity_id
    if q == "alpha":
        return bag.alpha
    if q == "phi":
        return bag.phi
    if q == "u":
        return bag.velocity[0]
    if q == "v":
        return bag.velocity[1]
    if q == "speed":
        return torch.hypot(*bag.velocity)
    if q == "p":
        return bag.pressure
    if q == "p_rel":
        return bag.pressure - bag.p_vapour
    if q == "p_liquid":
        return bag.p_liquid
    if q == "p_vapour":
        return bag.p_vapour
    if q in ("T", "theta"):
        return bag.theta
    if q == "s":
        return bag.source
    if q == "s_footprint":
        return bag.footprint
    if q == "s_thermal":
        return bag.source - bag.footprint
    if q == "vorticity":
        return bag.v_grad[0] - bag.u_grad[1]
    if q == "strain":
        u_x, u_y, _ = bag.u_grad
        v_x, v_y, _ = bag.v_grad
        return torch.sqrt(2.0 * u_x**2 + 2.0 * v_y**2 + (u_y + v_x) ** 2)
    if q == "shear_xy":
        return bag.u_grad[1] + bag.v_grad[0]
    if q == "friction_x":
        return -confinement_drag(bag.phase, bag.groups) * bag.velocity[0]
    if q == "dpdx":
        return bag.p_grad[0]
    if q == "dpdy":
        return bag.p_grad[1]
    if q == "grad_T":
        return torch.hypot(bag.theta_grad[0], bag.theta_grad[1])
    if q == "res_vof":
        return bag.stage_a.vof
    if q == "res_div":
        return (
            _masked(bag.stage_a.div, 1.0 - bag.liquid)
            if bag.model.exact_liquid
            else bag.stage_a.div
        )
    if q in catalogue.MOMENTUM_TERMS:
        return _masked(bag.momentum_terms[q], bag.liquid)
    if q == "res_brinkman_x":
        return _masked(sum(bag.momentum_terms.values()), bag.liquid)
    if q == "res_brinkman_y":
        return _masked(bag.momentum_y, bag.liquid)
    if q == "res_brinkman":
        return _masked(
            torch.hypot(sum(bag.momentum_terms.values()), bag.momentum_y), bag.liquid
        )
    if q == "res_vapour_potential":
        return _masked(vapour_potential_residual(bag.model, bag.x), 1.0 - bag.liquid)
    if q in catalogue.ENERGY_TERMS:
        return _masked(bag.energy_terms[q], bag.liquid)
    if q == "res_energy":
        return _masked(sum(bag.energy_terms.values()), bag.liquid)
    raise KeyError(q)


def _convert(scene: Scene, quantity_id: str, value: np.ndarray) -> np.ndarray:
    q = catalogue.BY_ID[quantity_id]
    scales = scene.scales
    if quantity_id == "T":
        return scales.saturation_C + scales.superheat_K * value
    if q.scale:
        return value * getattr(scales, q.scale)
    return value


def evaluate(
    scene: Scene, points_star: np.ndarray, quantity_ids: list[str]
) -> dict[str, np.ndarray]:
    """Each requested quantity in physical units at every point, ``(N,)``."""
    unknown = catalogue.unknown(quantity_ids)
    if unknown:
        raise KeyError(f"unknown quantities: {', '.join(unknown)}")
    # Residuals and their terms share one autograd bag even when the requested
    # term itself needs no derivative, so any of them turns gradients on.
    needs_grad = any(
        catalogue.BY_ID[i].needs_grad or catalogue.BY_ID[i].kind in GRAD_KINDS
        for i in quantity_ids
    )
    device = next(scene.model.parameters()).device
    parts: dict[str, list[np.ndarray]] = {i: [] for i in quantity_ids}
    n_chunks = max(1, math.ceil(len(points_star) / CHUNK))
    with scene.lock, torch.enable_grad():
        for chunk in np.array_split(points_star.astype(np.float32), n_chunks):
            if len(chunk) == 0:
                continue
            x = torch.tensor(chunk, dtype=torch.float32, device=device)
            if needs_grad:
                x.requires_grad_(True)
            bag = _Bag(scene, x)
            for i in quantity_ids:
                parts[i].append(_read(bag, i).detach().cpu().numpy().ravel())
    return {i: _convert(scene, i, np.concatenate(parts[i])) for i in quantity_ids}


def points_at(xs_star: np.ndarray, ys_star: np.ndarray, t_star: float) -> np.ndarray:
    """``(N, 3)`` rows from paired coordinates at one instant."""
    return np.column_stack([xs_star, ys_star, np.full(len(xs_star), t_star)]).astype(np.float32)
