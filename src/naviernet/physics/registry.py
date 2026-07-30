"""The equation registry: one declarative table of the governing PDE terms.

The trainer builds its active loss set and its rebalancing group from this table
rather than hardcoding them, and the API/UI read the same metadata (stage, TeX,
the fields each equation uses or unlocks, its weight key) so the front end never
hardcodes physics.

An equation is *enabled* for a run when it is implemented and every field it
requires is present in ``model.fields``. With the Stage-A field set
``(phi, u, v, s)`` this yields exactly ``vof, div, src, bc`` in that order, with
``vof, div, bc`` rebalanced and ``src`` a fixed penalty — reproducing the
pre-registry trainer byte-for-byte. The Stage-A equations are ``core`` (always on;
the UI locks them). Adding ``p`` unlocks momentum; adding ``T`` unlocks energy and
the evaporation mass closure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from naviernet.physics.residuals import (
    EnergyResiduals,
    MomentumResiduals,
    StageAResiduals,
    boundary_losses,
    energy_residuals,
    momentum_residuals,
    source_penalty_sq,
    stage_a_residuals,
)


class LossContext:
    """Everything a per-equation loss term needs, sampled once per step.

    Residual bundles are computed lazily and cached so terms sharing them (e.g.
    ``vof`` and ``div``) reuse a single autograd graph, exactly as the original
    trainer did with one ``stage_a_residuals`` call.

    The boundary batches (``inlet``, ``walls``, ``u_inlet``) are only read by the
    ``bc`` term and may be omitted for a collocation-only context -- the per-bin
    contexts the causal temporal weighting builds evaluate only the interior PDE
    terms (see :func:`collocation_equations`).
    """

    def __init__(
        self,
        model,
        x_coll: torch.Tensor,
        inlet: torch.Tensor | None = None,
        walls: torch.Tensor | None = None,
        u_inlet: float = 0.0,
        groups: dict[str, float] | None = None,
        c: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.x_coll = x_coll
        self.inlet = inlet
        self.walls = walls
        self.u_inlet = u_inlet
        self.groups = groups
        # The dataset's conditioning row for this batch (None when unconditioned).
        # Every residual bundle below is evaluated with it, so one joint step is
        # a sum of per-dataset LossContexts, each carrying its own `c`.
        self.c = c
        self._res_a: StageAResiduals | None = None
        self._mom: MomentumResiduals | None = None
        self._energy: EnergyResiduals | None = None

    @property
    def res_a(self) -> StageAResiduals:
        if self._res_a is None:
            self._res_a = stage_a_residuals(self.model, self.x_coll, self.c)
        return self._res_a

    @property
    def mom_res(self) -> MomentumResiduals:
        if self._mom is None:
            self._mom = momentum_residuals(self.model, self.x_coll, self.groups, c=self.c)
        return self._mom

    @property
    def energy_res(self) -> EnergyResiduals:
        if self._energy is None:
            self._energy = energy_residuals(
                self.model, self.x_coll, self.groups, self.model.r_int_star, c=self.c
            )
        return self._energy


# --- Collocation loss terms ---------------------------------------------------
#
# Each collocation term is the mean of a per-point squared residual. The pointwise
# function exposes that per-point residual (shape ``(n_coll, 1)``, non-negative) so
# per-point schemes -- RBA attention (Phase 2) and residual-adaptive resampling
# (Phase 3) -- can read it, and ``term = mean(pointwise)`` keeps the scalar objective
# identical to the original trainer (byte-for-byte for the golden Stage-A terms; the
# Stage-B mom term now sums per point before the mean, mathematically but not bitwise
# the same as the old sum-of-means).


def _mean(
    pointwise: Callable[[LossContext], torch.Tensor],
) -> Callable[[LossContext], torch.Tensor]:
    """The scalar loss term for a per-point squared residual: its mean."""
    return lambda ctx: pointwise(ctx).mean()


def _vof_sq(ctx: LossContext) -> torch.Tensor:
    return ctx.res_a.vof**2


def _div_sq(ctx: LossContext) -> torch.Tensor:
    return ctx.res_a.div**2


def _src_sq(ctx: LossContext) -> torch.Tensor:
    return source_penalty_sq(ctx.res_a)


def _bc_term(ctx: LossContext) -> torch.Tensor:
    return boundary_losses(ctx.model, ctx.inlet, ctx.walls, ctx.u_inlet, ctx.c)


def _mom_sq(ctx: LossContext) -> torch.Tensor:
    res = ctx.mom_res
    return res.mom_x**2 + res.mom_y**2


def _energy_sq(ctx: LossContext) -> torch.Tensor:
    return ctx.energy_res.energy**2


def _evap_sq(ctx: LossContext) -> torch.Tensor:
    return ctx.energy_res.src_closure**2


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
    core: bool = False  # always-on (the Stage-A objective); the UI locks it on
    # Evaluated on the interior collocation points ``x_coll`` (True) rather than the
    # boundary batches (False, i.e. the ``bc`` term). Causal temporal weighting
    # reweights only the collocation terms, so it selects on this flag.
    on_collocation: bool = True
    term: Callable[[LossContext], torch.Tensor] | None = field(default=None, repr=False)
    # Per-point squared residual (shape ``(n_coll, 1)``, non-negative) for the
    # collocation terms; ``term`` is its mean. ``None`` for boundary terms (bc), which
    # are not evaluated on the collocation points. Read by the per-point RBA attention
    # and residual-adaptive resampling.
    pointwise: Callable[[LossContext], torch.Tensor] | None = field(default=None, repr=False)


REGISTRY: tuple[Equation, ...] = (
    Equation(
        id="vof",
        stage="A",
        name="Interface transport",
        tex=r"\frac{\partial \alpha}{\partial t} + u\,\frac{\partial \alpha}{\partial x}"
        r" + v\,\frac{\partial \alpha}{\partial y} = 0",
        weight_key="vof",
        fields_required=("phi", "u", "v"),
        core=True,
        pointwise=_vof_sq,
        term=_mean(_vof_sq),
    ),
    Equation(
        id="div",
        stage="A",
        name="Continuity",
        tex=r"\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = s(x,y,t)",
        weight_key="div",
        fields_required=("u", "v", "s"),
        core=True,
        pointwise=_div_sq,
        term=_mean(_div_sq),
    ),
    Equation(
        id="src",
        stage="A",
        name="Source penalty",
        tex=r"s \to 0 \quad \text{away from the interface}",
        weight_key="src",
        fields_required=("s",),
        rebalanced=False,  # a deliberate soft penalty, held where it is put
        core=True,
        pointwise=_src_sq,
        term=_mean(_src_sq),
    ),
    Equation(
        id="bc",
        stage="A",
        name="Boundary conditions",
        tex=r"u = U_\text{in}\ \text{(inlet)}; \quad u = 0\ \text{(walls)}",
        weight_key="bc",
        fields_required=("u", "v"),
        core=True,
        on_collocation=False,  # evaluated on the inlet/wall batches, not x_coll
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
        pointwise=_mom_sq,
        term=_mean(_mom_sq),
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
        pointwise=_energy_sq,
        term=_mean(_energy_sq),
    ),
    Equation(
        id="evap",
        stage="B",
        name="Evaporation mass closure",
        tex=r"s = (1 - \rho_v/\rho_\ell)\,\mathrm{St}\,j\,\delta_\text{int}",
        weight_key="evap",
        fields_required=("s", "T"),
        groups=("Ja",),
        rebalanced=False,  # a soft consistency penalty, ramped by the curriculum
        pointwise=_evap_sq,
        term=_mean(_evap_sq),
    ),
)


def enabled_equations(fields: Sequence[str]) -> list[Equation]:
    """The equations active for a model with the given fields, in registry order."""
    present = set(fields)
    return [e for e in REGISTRY if e.implemented and set(e.fields_required) <= present]


def collocation_equations(equations: Sequence[Equation]) -> list[Equation]:
    """The equations evaluated on the interior collocation points ``x_coll`` -- every
    governing PDE term except the boundary conditions -- in registry order.

    Causal temporal weighting (Wang et al., arXiv:2203.07404) reweights this set by
    time; the boundary and supervised (``data``) terms are already time-anchored and
    stay uniform.
    """
    return [e for e in equations if e.on_collocation]


def rebalanced_terms(equations: Sequence[Equation]) -> tuple[str, ...]:
    """The weight keys the gradient-norm rebalancer adjusts, in order."""
    return tuple(e.weight_key for e in equations if e.rebalanced)


def stage_b_terms(equations: Sequence[Equation]) -> tuple[str, ...]:
    """The weight keys of the Stage-B equations, in order.

    These are gated off during the warm-up (`training.stage_b_warmup_steps`) so
    the interface converges on the Stage-A objective before the momentum, energy,
    and evaporation physics engages.
    """
    return tuple(e.weight_key for e in equations if e.stage == "B")
