"""The equation registry: one declarative table of the governing PDE terms.

The trainer builds its active loss set and its rebalancing group from this table
rather than hardcoding them, and the API/UI read the same metadata (stage, TeX,
the fields each equation uses or unlocks, its weight key) so the front end never
hardcodes physics.

An equation is *enabled* for a run when it is implemented and every field it
requires is present in ``model.fields``. With the Stage-A field set
``(phi, u, v, s)`` this yields exactly ``vof, div, src, bc`` in that order, with
``vof, div, bc`` rebalanced and ``src`` a fixed penalty — reproducing the
pre-registry trainer byte-for-byte. Stage-B equations (``mom``, ``energy``) are
declared here for the UI but stay ``implemented=False`` until their residuals land.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from naviernet.physics.residuals import (
    StageAResiduals,
    boundary_losses,
    source_penalty,
    stage_a_residuals,
)


class LossContext:
    """Everything a per-equation loss term needs, sampled once per step.

    Residual bundles are computed lazily and cached so terms sharing them (e.g.
    ``vof`` and ``div``) reuse a single autograd graph, exactly as the original
    trainer did with one ``stage_a_residuals`` call.
    """

    def __init__(self, model, x_coll, inlet, walls, u_inlet: float, groups: dict) -> None:
        self.model = model
        self.x_coll = x_coll
        self.inlet = inlet
        self.walls = walls
        self.u_inlet = u_inlet
        self.groups = groups
        self._res_a: StageAResiduals | None = None

    @property
    def res_a(self) -> StageAResiduals:
        if self._res_a is None:
            self._res_a = stage_a_residuals(self.model, self.x_coll)
        return self._res_a


# --- Stage-A loss terms (byte-for-byte identical to the original trainer) ----


def _vof_term(ctx: LossContext) -> torch.Tensor:
    return (ctx.res_a.vof**2).mean()


def _div_term(ctx: LossContext) -> torch.Tensor:
    return (ctx.res_a.div**2).mean()


def _src_term(ctx: LossContext) -> torch.Tensor:
    return source_penalty(ctx.res_a)


def _bc_term(ctx: LossContext) -> torch.Tensor:
    return boundary_losses(ctx.model, ctx.inlet, ctx.walls, ctx.u_inlet)


@dataclass(frozen=True)
class Equation:
    """One governing equation and how it enters the objective."""

    id: str
    stage: str  # "A" | "B"
    name: str
    tex: str
    weight_key: str
    fields_required: tuple[str, ...]
    fields_added: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()  # dimensionless groups entering the equation
    rebalanced: bool = True
    implemented: bool = True
    term: Callable[[LossContext], torch.Tensor] | None = field(default=None, repr=False)


REGISTRY: tuple[Equation, ...] = (
    Equation(
        id="vof",
        stage="A",
        name="Interface transport",
        tex=r"\frac{\partial \alpha}{\partial t} + u\,\frac{\partial \alpha}{\partial x}"
        r" + v\,\frac{\partial \alpha}{\partial y} = 0",
        weight_key="vof",
        fields_required=("phi", "u", "v"),
        term=_vof_term,
    ),
    Equation(
        id="div",
        stage="A",
        name="Continuity",
        tex=r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = s(x,y,t)",
        weight_key="div",
        fields_required=("u", "v", "s"),
        term=_div_term,
    ),
    Equation(
        id="src",
        stage="A",
        name="Source penalty",
        tex=r"s \to 0 \quad \text{away from the interface}",
        weight_key="src",
        fields_required=("s",),
        rebalanced=False,  # a deliberate soft penalty, held where it is put
        term=_src_term,
    ),
    Equation(
        id="bc",
        stage="A",
        name="Boundary conditions",
        tex=r"u = U_\text{in}\ \text{(inlet)}; \quad u = 0\ \text{(walls)}",
        weight_key="bc",
        fields_required=("u", "v"),
        term=_bc_term,
    ),
    Equation(
        id="mom",
        stage="B",
        name="Momentum",
        tex=r"\tilde{\rho}(\alpha)\!\left(\frac{\partial u}{\partial t} + u\cdot\nabla u\right)"
        r" = -\nabla p + \frac{1}{\mathrm{Re}}\nabla^2 u"
        r" - C_\text{HS}\,\tilde{\mu}(\alpha)\,u + \frac{1}{\mathrm{We}}\kappa\,\nabla\alpha",
        weight_key="mom",
        fields_required=("phi", "u", "v", "p"),
        fields_added=("p",),
        groups=("Re", "We", "hele_shaw"),
        implemented=False,  # residuals land in a later task
    ),
    Equation(
        id="energy",
        stage="B",
        name="Energy + evaporation",
        tex=r"\frac{\partial T}{\partial t} + u\cdot\nabla T = \frac{1}{\mathrm{Pe}}\nabla^2 T"
        r" + \dot{q}_\text{wall} - \mathrm{St}\,j\,\delta_\text{int}",
        weight_key="energy",
        fields_required=("u", "v", "T"),
        fields_added=("T",),
        groups=("Pe", "Pr"),
        implemented=False,
    ),
)


def enabled_equations(fields: Sequence[str]) -> list[Equation]:
    """The equations active for a model with the given fields, in registry order."""
    present = set(fields)
    return [e for e in REGISTRY if e.implemented and set(e.fields_required) <= present]


def rebalanced_terms(equations: Sequence[Equation]) -> tuple[str, ...]:
    """The weight keys the gradient-norm rebalancer adjusts, in order."""
    return tuple(e.weight_key for e in equations if e.rebalanced)
