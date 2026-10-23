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

from naviernet.models.geometry import FrontSamples
from naviernet.physics.film import (
    advancing_mask,
    depletion_residual,
    deposition_residual,
    film_source_residual,
)
from naviernet.physics.residuals import (
    EnergyResiduals,
    MomentumResiduals,
    NucleationPulse,
    StageAResiduals,
    ThermalBatches,
    boundary_losses,
    brinkman_residuals,
    darcy_residuals,
    energy_liquid_balance,
    energy_residuals,
    evap_flux_closure,
    footprint_source_penalty_sq,
    interface_mass_flux,
    kinematic_residual,
    laplace_jump_residual,
    momentum_residuals,
    source_penalty_sq,
    stage_a_residuals,
    thermal_boundary_losses,
    vapour_potential_residual,
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
        u_inlet: float | torch.Tensor = 0.0,
        groups: dict[str, float] | None = None,
        c: torch.Tensor | None = None,
        front_times: torch.Tensor | None = None,
        outlet: torch.Tensor | None = None,
        wall_condition: str = "no_slip",
        initial: torch.Tensor | None = None,
        outlet_pressure: float | None = None,
    ) -> None:
        self.model = model
        self.x_coll = x_coll
        self.inlet = inlet
        self.walls = walls
        self.u_inlet = u_inlet
        # The outlet batch (None when no outlet condition is imposed) and the
        # side-wall condition the `bc` term applies (see residuals.boundary_losses).
        self.outlet = outlet
        self.outlet_pressure = outlet_pressure
        self.wall_condition = wall_condition
        # Points at the first instant, for the sharp thermal model's initial
        # temperature condition (None when that model is off).
        self.initial = initial
        self.groups = groups
        # Times the interface conditions are sampled at (sharp-interface runs only);
        # the front itself is built lazily below, once per step, and shared by every
        # term that reads it -- the same caching the residual bundles use.
        self.front_times = front_times
        self._front: FrontSamples | None = None
        # The dataset's conditioning row for this batch (None when unconditioned).
        # Every residual bundle below is evaluated with it, so one joint step is
        # a sum of per-dataset LossContexts, each carrying its own `c`.
        self.c = c
        self._res_a: StageAResiduals | None = None
        self._mom: MomentumResiduals | None = None
        self._darcy: MomentumResiduals | None = None
        self._brinkman: MomentumResiduals | None = None
        self._vapour_potential: torch.Tensor | None = None
        self._energy: EnergyResiduals | None = None
        self._energy_liquid: tuple[torch.Tensor, torch.Tensor] | None = None
        self._mass_flux: torch.Tensor | None = None

    @property
    def front(self) -> FrontSamples:
        """Points on the explicit interface at this step's front times."""
        if self.front_times is None:
            raise RuntimeError(
                "a sharp-interface term was evaluated without front times -- build "
                "the LossContext with front_times=..., as the trainer does."
            )
        if self._front is None:
            model_cfg = self.model.cfg.model
            # Through the model, not `nets["phi"]`: a joint run's bound view has
            # to contribute its own conditioning row and anchors, and reaching
            # past it would sample the same dataset's front for every condition.
            self._front = self.model.front(
                self.front_times,
                n_body=int(model_cfg.front_body_samples),
                n_cap=int(model_cfg.front_cap_samples),
            )
        return self._front

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
    def darcy_res(self) -> MomentumResiduals:
        if self._darcy is None:
            self._darcy = darcy_residuals(self.model, self.x_coll, self.groups, c=self.c)
        return self._darcy

    @property
    def brinkman_res(self) -> MomentumResiduals:
        if self._brinkman is None:
            self._brinkman = brinkman_residuals(
                self.model,
                self.x_coll,
                self.groups,
                c=self.c,
                inertia=bool(getattr(self.model.cfg.model, "momentum_inertia", False)),
            )
        return self._brinkman

    @property
    def vapour_potential_res(self) -> torch.Tensor:
        if self._vapour_potential is None:
            self._vapour_potential = vapour_potential_residual(
                self.model, self.x_coll, c=self.c
            )
        return self._vapour_potential

    @property
    def energy_liquid_res(self) -> torch.Tensor:
        return self._energy_liquid_balance()[0]

    @property
    def energy_liquid_scale(self) -> torch.Tensor:
        """The size of the liquid energy equation's terms at each point, detached."""
        return self._energy_liquid_balance()[1]

    def _energy_liquid_balance(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._energy_liquid is None:
            self._energy_liquid = energy_liquid_balance(
                self.model,
                self.x_coll,
                self.groups,
                c=self.c,
                one_way=bool(getattr(self.model.cfg.training, "one_way_advection", False)),
            )
        return self._energy_liquid

    @property
    def mass_flux(self) -> torch.Tensor | None:
        """The liquid-side kinematic correction on this step's front under the
        sharp thermal model; ``None`` otherwise."""
        if getattr(self.model, "thermal_model", "legacy") != "sharp":
            return None
        if self._mass_flux is None:
            self._mass_flux = interface_mass_flux(self.model, self.front, self.groups)
        return self._mass_flux

    @property
    def energy_res(self) -> EnergyResiduals:
        if self._energy is None:
            self._energy = energy_residuals(
                self.model,
                self.x_coll,
                self.groups,
                self.model.r_int_star,
                c=self.c,
                pulse=self._nucleation_pulse(),
                two_way_closure=bool(
                    getattr(self.model.cfg.training, "evap_closure_two_way", False)
                ),
            )
        return self._energy

    def _nucleation_pulse(self) -> NucleationPulse | None:
        """The localized nucleation pulse for this batch, or ``None`` when the feature is
        off. Location/width/timing are fixed priors the trainer put in ``groups`` (in the
        model's non-dimensional units); only the magnitude is the model's learnable one."""
        if not getattr(self.model, "has_nucleation_pulse", False):
            return None
        return NucleationPulse(
            x_pin=self.groups["x_pin_star"],
            t0=self.groups["pulse_t0"],
            sigma=self.groups["pulse_sigma"],
            q_pulse=self.model.q_pulse_star,
        )


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
    # Which region is forbidden to the source depends on the film: without it,
    # mass off the interface band is unphysical; with it (the 1-D film field, or
    # the footprint's film evaporation), the films evaporate across the whole
    # footprint and only the outside liquid stays forbidden.
    if getattr(ctx.model, "liquid_film", False) or getattr(
        ctx.model, "footprint_evaporation", False
    ):
        with torch.no_grad():
            c = None if ctx.c is None else ctx.c.expand(ctx.x_coll.shape[0], -1)
            vapour = ctx.model.alpha(ctx.x_coll, c)
        return footprint_source_penalty_sq(ctx.res_a, vapour)
    return source_penalty_sq(ctx.res_a)


def _bc_term(ctx: LossContext) -> torch.Tensor:
    # The inlet temperature is anchored only when the superheat can leave it --
    # otherwise theta_in is already the field's floor and the condition is vacuous.
    theta_in = ctx.model.theta_in if getattr(ctx.model, "depletable_superheat", False) else None
    return boundary_losses(
        ctx.model,
        ctx.inlet,
        ctx.walls,
        ctx.u_inlet,
        ctx.c,
        theta_in,
        wall_condition=ctx.wall_condition,
        outlet_x=ctx.outlet,
        outlet_pressure=ctx.outlet_pressure,
    )


def _mom_sq(ctx: LossContext) -> torch.Tensor:
    res = ctx.mom_res
    return res.mom_x**2 + res.mom_y**2


def _darcy_sq(ctx: LossContext) -> torch.Tensor:
    res = ctx.darcy_res
    return res.mom_x**2 + res.mom_y**2


def _brinkman_sq(ctx: LossContext) -> torch.Tensor:
    res = ctx.brinkman_res
    return res.mom_x**2 + res.mom_y**2


def _vapour_potential_sq(ctx: LossContext) -> torch.Tensor:
    return ctx.vapour_potential_res**2


def _kinematic_term(ctx: LossContext) -> torch.Tensor:
    return (
        kinematic_residual(ctx.model, ctx.front, ctx.c, mass_flux=ctx.mass_flux) ** 2
    ).mean()


def _energy_liquid_sq(ctx: LossContext) -> torch.Tensor:
    return ctx.energy_liquid_res**2


def _energy_liquid_objective(ctx: LossContext) -> torch.Tensor:
    """What training minimises for the liquid energy equation: its squared
    residual, or under ``training.energy_relative`` that residual over the size
    of its own terms (floored at ``training.energy_relative_floor``, capped at
    ``training.energy_relative_cap``), so the
    open liquid, whose terms are some 40 times smaller than at the interface,
    counts as much as the interface does. The reported residual stays absolute."""
    tcfg = ctx.model.cfg.training
    if not bool(getattr(tcfg, "energy_relative", False)):
        return _energy_liquid_sq(ctx)
    floor = float(getattr(tcfg, "energy_relative_floor", ENERGY_RELATIVE_FLOOR))
    cap = float(getattr(tcfg, "energy_relative_cap", ENERGY_RELATIVE_CAP))
    size = ctx.energy_liquid_scale.clamp(max=cap)
    relative = (ctx.energy_liquid_res / (size + floor)) ** 2
    if str(getattr(tcfg, "energy_relative_mode", "relative")) == "mixed":
        return _energy_liquid_sq(ctx) + relative
    return relative


# The relative energy objective's floor on the terms' size, dimensionless: about
# the heating term over the upstream heater zone (q* times its 0.225 share on
# Series-1-w300), so a point where every term vanishes is not divided by zero.
ENERGY_RELATIVE_FLOOR = 0.02
# Its cap: above it a point counts at its absolute error, so a point cannot
# lower its own weight by growing its terms (schema: energy_relative_cap).
ENERGY_RELATIVE_CAP = 5.0


def training_pointwise(eq: Equation, ctx: LossContext) -> torch.Tensor:
    """The per-point quantity training minimises for ``eq``: its own objective
    when it has one, else its squared residual (what the audit reports)."""
    return (eq.objective or eq.pointwise)(ctx)


def _theta_bc_term(ctx: LossContext) -> torch.Tensor:
    batches = ThermalBatches(ctx.inlet, ctx.initial, ctx.walls)
    return thermal_boundary_losses(ctx.model, ctx.groups, batches, ctx.c)


def _evap_flux_sq(ctx: LossContext) -> torch.Tensor:
    return evap_flux_closure(ctx.model, ctx.x_coll, ctx.c) ** 2


def _laplace_term(ctx: LossContext) -> torch.Tensor:
    residual = laplace_jump_residual(ctx.model, ctx.front, ctx.groups)
    if str(getattr(ctx.model.cfg.model, "laplace_condition", "pointwise")) == "mean":
        residual = per_instant_mean(residual, ctx.front.points[:, 2])
    return (residual**2).mean()


def per_instant_mean(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """``values`` ``(N, 1)`` averaged over the samples sharing each instant of
    ``times`` ``(N,)``: one mean per instant, ``(K, 1)``."""
    _, which = torch.unique(times.detach(), return_inverse=True)
    count = torch.zeros(int(which.max()) + 1, 1, device=values.device, dtype=values.dtype)
    total = torch.zeros_like(count).index_add(0, which, values)
    count = count.index_add(0, which, torch.ones_like(values))
    return total / count


def _film_term(ctx: LossContext) -> torch.Tensor:
    # Mean over the ADVANCING samples only -- the ones deposition constrains.
    # Averaging over the receding rest (zeroed in the residual) would scale the
    # term by an arbitrary advancing-to-receding ratio that changes as the
    # bubble evolves.
    residual = deposition_residual(ctx.model, ctx.front, ctx.groups, ctx.c)
    anchored = advancing_mask(ctx.front).sum().clamp(min=1.0)
    return (residual**2).sum() / anchored


def _film_source_sq(ctx: LossContext) -> torch.Tensor:
    return film_source_residual(ctx.model, ctx.x_coll, ctx.groups, ctx.c) ** 2


def _film_depletion_term(ctx: LossContext) -> torch.Tensor:
    residual = depletion_residual(
        ctx.model, ctx.front_times, ctx.groups, ctx.model.film_stations, ctx.c
    )
    return (residual**2).mean()


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
    # Which interface treatment this equation belongs to. ``any`` = both (the
    # transport, continuity and energy physics is the same either way); ``diffuse``
    # = only when the shape is driven by bulk residuals over a smeared alpha;
    # ``sharp`` = only under ``model.sharp_interface``, where the conditions are
    # imposed on the explicit front. Selecting here keeps the trainer, the API and
    # the UI reading one table instead of each branching on the flag.
    mode: str = "any"
    # The model flag that opts this equation in, or ``None`` for the mode/field
    # gating alone. The film term exists only when the film field does; encoding
    # that here keeps `enabled_equations` the single place activation is decided.
    flag: str | None = None
    # The model flag that opts this equation OUT: the film's source closure
    # REPLACES the |grad alpha| one, and two closures on the same field with
    # different distributions would fight. ``None`` = never displaced.
    unless_flag: str | None = None
    # Several displacing flags, for an equation more than one feature replaces;
    # ``unless_flag`` is the single-flag shorthand and stays for the UI.
    unless_flags: tuple[str, ...] = ()
    # The momentum closure (``model.momentum_closure``) this equation belongs to:
    # Darcy and Brinkman are two laws for the same balance, so exactly one is
    # active. ``None`` = independent of the closure.
    closure: str | None = None
    term: Callable[[LossContext], torch.Tensor] | None = field(default=None, repr=False)
    # What training minimises per point when it differs from ``pointwise``
    # (which the audit and the log report); None is ``pointwise`` itself.
    objective: Callable[[LossContext], torch.Tensor] | None = field(default=None, repr=False)
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
        # A thermal source lives on the front by construction; penalising it off
        # the interface would score the kernel's own tails.
        unless_flag="thermal_source",
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
        mode="diffuse",
        pointwise=_mom_sq,
        term=_mean(_mom_sq),
    ),
    Equation(
        id="darcy",
        stage="B",
        name="Darcy (depth-averaged momentum)",
        tex=r"\nabla p = -C_\text{HS}\,\tilde{\mu}(\alpha)\,\mathbf{u}",
        weight_key="darcy",
        fields_required=("phi", "u", "v", "p"),
        fields_added=("p",),
        groups=("hele_shaw", "mu_ratio"),
        mode="sharp",
        closure="darcy",
        pointwise=_darcy_sq,
        term=_mean(_darcy_sq),
    ),
    Equation(
        id="brinkman",
        stage="B",
        name="Brinkman (depth-averaged momentum with wall shear)",
        tex=r"\nabla p = -C_\text{HS}\,\tilde{\mu}(\alpha)\,\mathbf{u}"
        r" + \frac{\tilde{\mu}(\alpha)}{\mathrm{Re}}\nabla^2\mathbf{u}",
        weight_key="brinkman",
        fields_required=("phi", "u", "v", "p"),
        fields_added=("p",),
        groups=("hele_shaw", "mu_ratio", "Re"),
        mode="sharp",
        closure="brinkman",
        pointwise=_brinkman_sq,
        term=_mean(_brinkman_sq),
    ),
    Equation(
        id="vapour_potential",
        stage="B",
        name="Curl-free vapour velocity",
        tex=r"\alpha\left(\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right) = 0",
        weight_key="vapour_potential",
        fields_required=("phi", "u", "v"),
        mode="sharp",
        # Only under the two-phase pressure, where the uniform vapour pressure
        # makes the Darcy law inside the bubble say nothing about its velocity.
        flag="two_phase_pressure",
        pointwise=_vapour_potential_sq,
        term=_mean(_vapour_potential_sq),
    ),
    Equation(
        id="kinematic",
        stage="B",
        name="Kinematic condition",
        tex=r"v_n = \mathbf{u}\cdot\mathbf{n} \quad \text{on } \Gamma",
        weight_key="kinematic",
        fields_required=("phi", "u", "v"),
        mode="sharp",
        # A condition ON the front, not an interior residual -- like `bc`, so it
        # stays out of the causal/RBA collocation reweighting.
        on_collocation=False,
        term=_kinematic_term,
    ),
    Equation(
        id="laplace",
        stage="B",
        name="Young-Laplace jump",
        tex=r"p_v(t) - p_\ell(\Gamma) = \frac{1}{\mathrm{We}}"
        r"\left(\kappa_\parallel + \kappa_\perp\right)",
        weight_key="laplace",
        fields_required=("phi", "p"),
        # Ca and H_star reach it through the gap curvature's Bretherton term --
        # a first-order effect on the jump, so the UI must show the dependency.
        groups=("We", "Ca", "H_star"),
        mode="sharp",
        # Evaluated on the front samples, not the interior collocation batch --
        # like `bc`, and for the same reason: it is a boundary condition.
        on_collocation=False,
        # NOT rebalanced. The gradient-norm rebalancer drives every term it owns
        # to parity with the data term -- measured, it holds this one at ~0.85.
        # Parity is the wrong target here: this condition's whole job is to
        # overrule the pixel fit where the two disagree about the interface's
        # shape, and a term pinned to the data term's gradient can never do that.
        # Its weight is the configured one, so the trade is the user's to set.
        rebalanced=False,
        term=_laplace_term,
    ),
    Equation(
        id="film",
        stage="B",
        name="Film deposition",
        tex=r"\delta(x, t) = \frac{H^*}{2}\,"
        r"\frac{P\,(3\,\mathrm{Ca}\,v_n)^{2/3}}{1 + P\,Q\,(3\,\mathrm{Ca}\,v_n)^{2/3}}"
        r" \quad \text{where } v_n > 0",
        weight_key="film",
        fields_required=("phi",),
        # Ca and H_star set the deposited thickness through the local capillary
        # number, exactly as they set the Bretherton pressure correction.
        groups=("Ca", "H_star"),
        mode="sharp",
        flag="liquid_film",
        # A condition ON the front, like `kinematic` -- not an interior residual,
        # so it stays out of the causal/RBA collocation reweighting.
        on_collocation=False,
        # NOT rebalanced: the film net's parameters are disjoint from every other
        # field's, so this term is the ONLY gradient they receive -- parity with
        # the data term's gradient norm is meaningless for it.
        rebalanced=False,
        term=_film_term,
    ),
    Equation(
        id="film_depletion",
        stage="B",
        name="Film depletion",
        tex=r"\frac{\partial \delta}{\partial t}\Big|_{x}"
        r" = -\frac{E_\text{film}\,\theta}{\delta + R_\gamma^*}"
        r"\,g(\delta) \quad \text{behind } \Gamma",
        weight_key="film_depletion",
        fields_required=("phi", "T"),
        # E_film = Ja/Pe; the kinetic resistance R_gamma* comes from the fluid.
        groups=("Ja", "Pe"),
        mode="sharp",
        flag="liquid_film",
        # Evaluated on the film's own footprint quadrature, not x_coll -- like
        # the front conditions, it stays out of the collocation reweighting.
        on_collocation=False,
        # NOT rebalanced, same reason as the deposition term: the film net's
        # parameters are disjoint from every other field's.
        rebalanced=False,
        term=_film_depletion_term,
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
        # Displaced by the sharp thermal model's liquid energy equation.
        unless_flag="sharp_thermal",
        pointwise=_energy_sq,
        term=_mean(_energy_sq),
    ),
    Equation(
        id="energy_liquid",
        stage="B",
        name="Liquid energy, interface at saturation",
        tex=r"(1-\alpha)\left[\frac{\partial\theta}{\partial t} + u\cdot\nabla\theta"
        r" - \frac{1}{\mathrm{Pe}}\nabla^2\theta - q^*(x,y,t)(1-\alpha)\right] = 0",
        weight_key="energy_liquid",
        fields_required=("phi", "u", "v", "T"),
        fields_added=("T",),
        groups=("Pe", "q_wall_star"),
        mode="sharp",
        flag="sharp_thermal",
        pointwise=_energy_liquid_sq,
        term=_mean(_energy_liquid_sq),
        objective=_energy_liquid_objective,
    ),
    Equation(
        id="theta_bc",
        stage="B",
        name="Temperature conditions",
        tex=r"\theta = \theta_\text{in}\ \text{(inlet)};\quad \theta = \theta_0\ (t_0);"
        r"\quad \partial_y\theta = 0\ \text{or}\ \theta = \theta_w\ \text{(walls)}",
        weight_key="theta_bc",
        fields_required=("phi", "T"),
        groups=("theta_inlet",),
        mode="sharp",
        flag="sharp_thermal",
        on_collocation=False,  # evaluated on the inlet, initial and wall batches
        # NOT rebalanced. These are the series' measured temperatures (the
        # liquid at 58 C before the first frame, the inlet), as much data as the
        # masks are. Pinned to gradient parity, the first full-chain benchmark
        # let the physics erase the initial superheat (the condition's loss sat
        # at 0.76 while the temperature flattened to saturation everywhere).
        # Their weight is the configured one, held at the data term's.
        rebalanced=False,
        term=_theta_bc_term,
    ),
    Equation(
        id="evap_flux",
        stage="B",
        name="Source from the interface flux",
        tex=r"s = C_m \oint \frac{\partial\theta}{\partial n}\, K_w(\mathbf{x}-\mathbf{X})\,d\ell",
        weight_key="evap_flux",
        fields_required=("phi", "s", "T"),
        groups=("mass_flux_to_source",),
        mode="sharp",
        flag="sharp_thermal",
        # With a thermal source there is no free source to pull.
        unless_flag="thermal_source",
        rebalanced=False,  # a soft consistency closure, like the one it replaces
        pointwise=_evap_flux_sq,
        term=_mean(_evap_flux_sq),
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
        # Displaced by the film's source closure: same field, different
        # distribution, and the distribution is the point of the film. Also
        # displaced by the sharp thermal model, whose flux closure replaces it.
        unless_flag="liquid_film",
        unless_flags=("liquid_film", "sharp_thermal"),
        pointwise=_evap_sq,
        term=_mean(_evap_sq),
    ),
    Equation(
        id="film_source",
        stage="B",
        name="Film evaporation source",
        tex=r"s = \frac{2\,(\rho_\ell/\rho_v - 1)}{H^*}\,"
        r"\frac{E_\text{film}\,\theta}{\delta + R_\gamma^*}\,g(\delta)\,\alpha",
        weight_key="film_source",
        fields_required=("s", "T"),
        groups=("Ja", "Pe", "rho_ratio"),
        mode="sharp",
        flag="liquid_film",
        # A soft consistency penalty on `s`, like the closure it replaces.
        rebalanced=False,
        pointwise=_film_source_sq,
        term=_mean(_film_source_sq),
    ),
)


def enabled_equations(
    fields: Sequence[str],
    sharp_interface: bool = False,
    liquid_film: bool = False,
    momentum_closure: str = "darcy",
    two_phase_pressure: bool = False,
    sharp_thermal: bool = False,
    thermal_source: bool = False,
) -> list[Equation]:
    """The equations active for a model with the given fields, in registry order.

    ``sharp_interface`` selects the interface treatment (``model.sharp_interface``):
    it admits the equations imposed on the explicit front and excludes the ones
    that only make sense over a smeared alpha. ``liquid_film`` opts in the film
    term (``model.liquid_film``), which exists only when the film field does.
    ``momentum_closure`` picks the sharp-mode momentum law (``model.momentum_
    closure``): Darcy or Brinkman, never both. ``two_phase_pressure`` opts in the
    curl-free vapour velocity (``model.pressure_representation=two_phase``).
    ``sharp_thermal`` swaps the legacy energy and evaporation pair for the liquid
    energy equation, the temperature conditions and the flux closure
    (``model.thermal_model=sharp``); ``thermal_source`` then drops the free
    source's closure and its off-interface penalty (``model.source_mode=thermal``).
    Every default keeps existing call sites composing byte-for-byte.
    """
    present = set(fields)
    allowed = {"any", "sharp" if sharp_interface else "diffuse"}
    active = {
        name
        for name, on in (
            ("liquid_film", liquid_film),
            ("two_phase_pressure", two_phase_pressure),
            ("sharp_thermal", sharp_thermal),
            ("thermal_source", thermal_source),
        )
        if on
    }
    return [
        e
        for e in REGISTRY
        if e.implemented
        and e.mode in allowed
        and e.flag in ({None} | active)
        and e.unless_flag not in active
        and not (set(e.unless_flags) & active)
        and e.closure in (None, momentum_closure)
        and set(e.fields_required) <= present
    ]


def flags_from(model_cfg) -> dict[str, object]:
    """The :func:`enabled_equations` keyword arguments a model config implies, so
    every caller (trainer, evaluation, API) gates the registry the same way."""
    return {
        "sharp_interface": bool(getattr(model_cfg, "sharp_interface", False)),
        "liquid_film": bool(getattr(model_cfg, "liquid_film", False)),
        "momentum_closure": str(getattr(model_cfg, "momentum_closure", "darcy")),
        "two_phase_pressure": (
            str(getattr(model_cfg, "pressure_representation", "single")) == "two_phase"
        ),
        "sharp_thermal": str(getattr(model_cfg, "thermal_model", "legacy")) == "sharp",
        "thermal_source": str(getattr(model_cfg, "source_mode", "free")) == "thermal",
    }


# The configuration choice behind each registry flag, as the Hydra override that
# switches it on, and the one that keeps it off; what the API reports as an
# equation's ``enabled_by``.
_FLAG_CHOICES = {
    "liquid_film": "model.liquid_film=true",
    "two_phase_pressure": "model.pressure_representation=two_phase",
    "sharp_thermal": "model.thermal_model=sharp",
    "thermal_source": "model.source_mode=thermal",
}
_FLAG_DEFAULTS = {
    "liquid_film": "model.liquid_film=false",
    "two_phase_pressure": "model.pressure_representation=single",
    "sharp_thermal": "model.thermal_model=legacy",
    "thermal_source": "model.source_mode=free",
}


def enabled_by(equation: Equation) -> str | None:
    """The configuration choices that admit ``equation``, as comma-separated
    Hydra overrides (composing them, with its fields present, enables it), or
    None for a term gated only by the fields the model trains."""
    choices: list[str] = []
    if equation.mode == "sharp":
        choices.append("model.sharp_interface=true")
    elif equation.mode == "diffuse":
        choices.append("model.sharp_interface=false")
    if equation.closure is not None:
        choices.append(f"model.momentum_closure={equation.closure}")
    if equation.flag is not None:
        choices.append(_FLAG_CHOICES[equation.flag])
    excluded = {equation.unless_flag, *equation.unless_flags} - {None}
    choices.extend(_FLAG_DEFAULTS[flag] for flag in sorted(excluded))
    return ", ".join(choices) or None


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
