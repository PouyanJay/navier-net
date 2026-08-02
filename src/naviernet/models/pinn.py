"""The field-ensemble PINN: one Fourier-feature MLP per physical field.

Separate networks per field (rather than one multi-head network) keep the
fields' spectral requirements independent -- the interface field needs far more
high-frequency capacity than the velocity components -- and let Stage B add
pressure and temperature without disturbing anything Stage A learned.

The volume fraction is *never* predicted directly. A network outputs a
level-set-like field ``phi``, and ``alpha = sigmoid(phi / eps)``. This bounds
alpha in (0, 1) by construction, needs no clamping or penalty to stay physical,
and makes the interface half-thickness ``eps`` an explicit, annealable
parameter rather than an emergent property of the fit.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from naviernet.models.geometry import (
    GeometricInterface,
    GeometryContext,
    GeometryPriors,
)
from naviernet.models.layers import AdaptiveTanh, FourierFeatures

# Hidden layout of the vapour-pressure net. A module constant, not config: p_v is
# one smooth scalar curve in time, and capacity beyond this buys nothing but the
# freedom to wobble -- which would undo the near-isobaric constraint it encodes.
VAPOR_HIDDEN = 32
VAPOR_DEPTH = 2


def _resolve(arch, key: str, fallback):
    """A per-field override value if set, otherwise the global default."""
    if arch is None:
        return fallback
    value = getattr(arch, key, None)
    return fallback if value is None else value


class FieldNet(nn.Module):
    """Fourier-feature MLP with adaptive-tanh activations for a single field.

    ``arch`` is an optional per-field override; any attribute it leaves ``None``
    falls back to the global ``cfg.model`` value, so passing ``None`` reproduces
    the global architecture exactly.
    """

    def __init__(self, cfg, out_dim: int = 1, arch=None, n_cond: int = 0):
        super().__init__()
        model_cfg = cfg.model
        hidden = _resolve(arch, "hidden", model_cfg.hidden)
        depth = _resolve(arch, "layers", model_cfg.layers)
        self.n_cond = int(n_cond)
        self.ff = FourierFeatures(
            in_dim=3,
            n_feats=_resolve(arch, "fourier_feats", model_cfg.fourier_feats),
            scale=_resolve(arch, "fourier_scale", model_cfg.fourier_scale),
        )

        # The conditioning vector is appended to the Fourier features, not fed
        # through them: it is a per-dataset constant, so only the (x, y, t)
        # coordinates get positional encoding and only they are differentiated in
        # the PDE residuals. n_cond=0 reproduces the unconditioned architecture.
        dims = [self.ff.out_dim + self.n_cond] + [hidden] * depth
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [
                nn.Linear(d_in, d_out),
                AdaptiveTanh(d_out, model_cfg.nodewise_activation),
            ]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.ff(x)
        if c is not None:
            feats = torch.cat([feats, c], dim=-1)
        return self.mlp(feats)


class BubblePINN(nn.Module):
    """Ensemble of field networks.

    Stage A uses ``phi`` (yielding alpha), the velocity components ``u`` and
    ``v``, and the inferred dilatation source ``s``. Stage B adds ``p`` and
    ``T`` by listing them in ``cfg.model.fields``.

    Every accessor takes points ``x`` of shape ``(N, 3)`` ordered ``(x, y, t)``
    and returns columns of shape ``(N, 1)``. When the model is conditioned
    (``n_cond > 0``), each accessor also takes the per-point context ``c`` of
    shape ``(N, n_cond)`` -- the point's dataset's conditioning vector. ``c`` is
    ``None`` for an unconditioned (single-dataset) model.
    """

    def __init__(
        self,
        cfg,
        fields: Sequence[str] | None = None,
        n_cond: int = 0,
        pin: tuple[float, float] | None = None,
        geometry: GeometryPriors | None = None,
    ):
        # NB over the usual argument cap: ``fields``/``n_cond``/``pin``/``geometry``
        # are orthogonal opt-ins (field set, joint conditioning, hard pin, front
        # geometry) and bundling them into a carrier object would churn every
        # construction site for no clarity gain -- a reviewed, deliberate deviation.
        super().__init__()
        self.cfg = cfg
        self.eps = float(cfg.model.alpha_eps)
        self.n_cond = int(n_cond)
        names = list(fields if fields is not None else cfg.model.fields)
        self._validate_front_geometry(cfg, geometry)
        self._validate_sharp_interface(cfg, names)
        self._validate_film_pressure(cfg)
        self._init_hard_pin(cfg, pin)

        per_field = getattr(cfg.model, "per_field", None) or {}
        self.nets = nn.ModuleDict(
            {
                name: GeometricInterface(
                    geometry, allow_pinch=self.allow_pinch, n_cond=self.n_cond
                )
                if name == "phi" and self.front_geometry
                else FieldNet(cfg, arch=per_field.get(name), n_cond=self.n_cond)
                for name in names
            }
        )
        self._init_inverse_unknowns(cfg, names)
        self._init_vapor_pressure()

    def _validate_sharp_interface(self, cfg, names: list[str]) -> None:
        """Reject a sharp-interface composition that cannot work, before any net
        is built: there is no front to sample without the front geometry, and no
        jump to impose without a liquid pressure."""
        self.sharp_interface = bool(getattr(cfg.model, "sharp_interface", False))
        if not self.sharp_interface:
            return
        if not self.front_geometry:
            raise ValueError(
                "model.sharp_interface imposes the interface conditions ON the "
                "explicit front, so it requires model.front_geometry=true -- enable "
                "the front geometry, or disable model.sharp_interface."
            )
        if "p" not in names:
            raise ValueError(
                "model.sharp_interface reads the liquid pressure at the interface, so "
                "it requires the 'p' field in model.fields (the Stage-B field set); "
                f"this model has {names}."
            )

    def _validate_film_pressure(self, cfg) -> None:
        """The film pressure corrects the Young-Laplace jump; without that
        condition there is nothing for it to correct."""
        self.film_pressure = bool(getattr(cfg.model, "film_pressure", False))
        if self.film_pressure and not self.sharp_interface:
            raise ValueError(
                "model.film_pressure corrects the Young-Laplace jump on the front, "
                "so it requires model.sharp_interface=true -- enable it, or turn "
                "model.film_pressure off."
            )

    def _init_vapor_pressure(self) -> None:
        """``p_v(t)``: the vapour interior's pressure, one scalar per time.

        The bubble is very nearly isobaric inside -- ``mu_l/mu_v ~ 37``, so the
        viscous pressure drop along the vapour is negligible beside the liquid's.
        Making that STRUCTURAL rather than hoped-for is the whole point: with
        ``p_v`` uniform in space, the Young-Laplace jump forces the total
        curvature to be nearly uniform along the entire front, which is the
        mechanism that inflates the fast nose cap and drains the slower mid-body
        into a neck. A free 2-D pressure field asserts nothing of the kind.
        """
        if not self.sharp_interface:
            return
        if self.film_pressure:
            # ONE scalar, deliberately. It has to absorb the body-to-cap offset
            # the depth-averaged pressure cannot represent, and it must NOT be
            # able to absorb variation ALONG the body -- that variation is the
            # capillary gradient which drains the neck. A spatially constant
            # offset can do the first and structurally cannot do the second.
            # Free-signed: the film sits behind an advancing meniscus and is
            # measured to be at LOWER pressure than the bulk.
            self.p_film_raw = nn.Parameter(torch.zeros(()))
        dims = [1] + [VAPOR_HIDDEN] * VAPOR_DEPTH
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d_in, d_out), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], 1))
        self.vapor_pressure = nn.Sequential(*layers)

    def film_offset(self, on_cap: torch.Tensor) -> torch.Tensor:
        """The film-to-bulk pressure offset at each front sample.

        Zero on the caps: there is no film there -- the meniscus faces bulk
        liquid, so the model's own pressure is already the right one to compare
        against, and an offset would be inventing a correction where none is
        needed. Zero everywhere when the feature is off.
        """
        if not self.film_pressure:
            return torch.zeros_like(on_cap)
        return self.p_film_raw * (1.0 - on_cap)

    def p_vapor(self, t: torch.Tensor) -> torch.Tensor:
        """Vapour-interior pressure at times ``t`` of shape ``(N, 1)``.

        Space-independent by construction (see :meth:`_init_vapor_pressure`). The
        gauge is fixed by the jump condition itself, which ties ``p_v`` to the
        liquid pressure at the front, so no positivity or offset constraint is
        needed or wanted.
        """
        if not self.sharp_interface:
            raise RuntimeError(
                "p_vapor is only defined for a sharp-interface model; this one was "
                "built with model.sharp_interface=false."
            )
        return self.vapor_pressure(t)

    def _validate_front_geometry(self, cfg, geometry: GeometryPriors | None) -> None:
        """Reject unusable front-geometry compositions loudly, before any net is
        built: the geometry pins exactly (hard_pin is redundant and conflicting),
        priors are required, and a per-field phi override would be silently
        ignored.

        Joint conditioning is supported: the nets take the dataset's
        conditioning row and each dataset's measured anchors are bound per call
        (``bound(c, pin=, geometry=)``), so one construction serves every
        condition."""
        self.front_geometry = bool(getattr(cfg.model, "front_geometry", False))
        self.allow_pinch = bool(getattr(cfg.model, "allow_pinch", False))
        if self.allow_pinch and not self.front_geometry:
            raise ValueError(
                "model.allow_pinch relaxes the front geometry's own topology and "
                "monotonicity guarantees, so it requires model.front_geometry=true; "
                "a free level set has no such guarantees to relax."
            )
        if not self.front_geometry:
            return
        if getattr(cfg.model, "hard_pin", False):
            raise ValueError(
                "model.front_geometry already pins the root exactly by construction; "
                "it is mutually exclusive with model.hard_pin -- disable one."
            )
        if geometry is None:
            raise ValueError(
                "model.front_geometry=true needs the dataset's geometry priors: "
                "construct the model with geometry=GeometryPriors(...), as "
                "train()/load_model() do."
            )
        if (getattr(cfg.model, "per_field", None) or {}).get("phi") is not None:
            raise ValueError(
                "model.per_field.phi has no effect when model.front_geometry=true "
                "(the phi net is the geometric construction) -- remove the override."
            )

    def _init_hard_pin(self, cfg, pin: tuple[float, float] | None) -> None:
        """Hard root pin: phi = ell(x, y) * N, so the interface (alpha = 0.5) passes
        through the dataset's measured root anchor at EVERY time -- the constraint is
        architectural, so it keeps holding beyond the training window, unlike a loss
        term. The anchor is data-derived (never config) and deliberately NOT
        persisted: the checkpoint format is unchanged and the anchor is re-measured
        from the same dataset on every construction."""
        self.hard_pin = bool(getattr(cfg.model, "hard_pin", False))
        self.pin_d_ref = float(cfg.model.pin_d_ref) if self.hard_pin else None
        self.pin_anchor: torch.Tensor | None
        if self.hard_pin:
            if self.pin_d_ref <= 0:
                raise ValueError(f"model.pin_d_ref must be > 0, got {self.pin_d_ref}")
            if pin is None and self.n_cond == 0:
                # A conditioned (joint) model legitimately has no single anchor --
                # each dataset binds its own per call (see ``bound``); an unbound
                # call then fails loudly in ``phi``. A single-dataset model has
                # exactly one dataset, so a missing anchor is a wiring bug.
                raise ValueError(
                    "model.hard_pin=true needs the dataset's root anchor: construct "
                    "the model with pin=(x0, y0), as train()/load_model() do."
                )
        if self.hard_pin and pin is not None:
            self.register_buffer(
                "pin_anchor", torch.tensor(pin, dtype=torch.float32), persistent=False
            )
        else:
            self.pin_anchor = None

    def _init_inverse_unknowns(self, cfg, names: list[str]) -> None:
        """Stage-B inverse unknowns, present only when temperature is modelled:
        the interfacial resistance closing evaporation and the inlet superheat.
        Both are inferred, not measured (the dataset calls them unknowns). The
        nucleation pulse adds its learnable magnitude (the heater power is unknown;
        location/width/timing are fixed priors supplied by the trainer)."""
        if "T" in names:
            self._log_r_int = nn.Parameter(torch.zeros(1))
            self._theta_in_raw = nn.Parameter(torch.zeros(1))
        self.has_nucleation_pulse = bool(getattr(cfg.model, "nucleation_pulse", False))
        if self.has_nucleation_pulse:
            if "T" not in names:
                raise ValueError(
                    "model.nucleation_pulse=true heats the energy equation, so it "
                    "requires the 'T' field in model.fields."
                )
            self._log_q_pulse = nn.Parameter(torch.zeros(1))

    @property
    def fields(self) -> list[str]:
        return list(self.nets.keys())

    def phi(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        pin: torch.Tensor | None = None,
        geometry: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """Level-set field; its zero contour is the interface. With the hard pin
        on, the zero contour is anchored to the root for all t (see __init__).
        ``pin`` is the per-call anchor a joint run's bound view supplies; a
        single-dataset model carries its own. ``geometry`` is the same idea for
        the front geometry: a joint run binds each dataset's own measured
        anchors, so one shared construction lands on each condition's own root,
        front and channel."""
        raw = (
            self.nets["phi"](x, c, priors=geometry)
            if self.front_geometry
            else self.nets["phi"](x, c)
        )
        if not self.hard_pin:
            return raw
        anchor = pin if pin is not None else self.pin_anchor
        if anchor is None:
            raise RuntimeError(
                "hard_pin model evaluated without a root anchor -- a joint model "
                "must be bound per dataset first: model.bound(c, pin=context.pin)."
            )
        return self._pin_gate(x, anchor) * raw

    def _pin_gate(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """``tanh(d^2/d_ref^2)``: exactly 0 at the anchor, saturating to 1 beyond
        ~2*d_ref so the far field is untouched. The squared-distance argument keeps
        the gate C-infinity: the raw Euclidean distance has a cusp at the anchor
        whose *second* derivative diverges like 1/d, and the Stage-B surface-tension
        term differentiates alpha twice (curvature) -- measured, kappa*grad(alpha)
        blows up ~3e6 near the anchor with a linear gate but stays bounded (~60)
        with this form, because grad(alpha) vanishes as fast as kappa grows."""
        # A per-call anchor (joint bound view) is a plain CPU tensor; align it
        # with the batch (a no-op for the registered single-dataset buffer).
        d_sq = ((x[:, :2] - anchor.to(x.device)) ** 2).sum(dim=-1, keepdim=True)
        return torch.tanh(d_sq / self.pin_d_ref**2)

    def alpha(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        pin: torch.Tensor | None = None,
        geometry: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """Volume fraction, bounded in (0, 1) by construction."""
        return torch.sigmoid(self.phi(x, c, pin=pin, geometry=geometry) / self.eps)

    def front(
        self,
        t: torch.Tensor,
        n_body: int,
        n_cap: int,
        c: torch.Tensor | None = None,
        geometry: GeometryPriors | None = None,
    ):
        """The explicit interface at times ``t`` -- the object the sharp-interface
        conditions are imposed on.

        Routed through the model rather than reached for via ``nets["phi"]``
        because a joint run's per-dataset view binds its own conditioning row and
        anchors: going around it would hand every dataset the SAME front, and
        every jump residual after the first would be scored against another
        condition's interface.
        """
        if not self.front_geometry:
            raise RuntimeError(
                "front() needs the explicit front: this model was built with "
                "model.front_geometry=false, so there is no parameterized "
                "interface to sample."
            )
        return self.nets["phi"].front(t, n_body, n_cap, GeometryContext(c, geometry))

    def velocity(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.nets["u"](x, c), self.nets["v"](x, c)

    def source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Inferred volumetric dilatation from phase change."""
        return self.nets["s"](x, c)

    def pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Stage B. Raises if the pressure field was not configured."""
        return self._require("p")(x, c)

    @property
    def r_int_star(self) -> torch.Tensor:
        """Non-dimensional interfacial resistance (>0), a trainable unknown."""
        return nn.functional.softplus(self._log_r_int) + 1e-3

    @property
    def theta_in(self) -> torch.Tensor:
        """Inlet superheat in (0, 1) -- saturation to wall -- a trainable unknown."""
        return torch.sigmoid(self._theta_in_raw)

    @property
    def q_pulse_star(self) -> torch.Tensor:
        """Non-dimensional nucleation-pulse magnitude (>0), a trainable unknown (the
        heater's power is not known, so only its strength is fit)."""
        return nn.functional.softplus(self._log_q_pulse)

    def temperature(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Non-dimensional superheat, bounded to (theta_in, 1) so temperature stays
        between the inlet and the wall. Stage B; raises if T was not configured."""
        raw = self._require("T")(x, c)
        return self.theta_in + (1.0 - self.theta_in) * torch.sigmoid(raw)

    def _require(self, name: str) -> nn.Module:
        if name not in self.nets:
            raise KeyError(
                f"field {name!r} is not in this model (has: {self.fields}). "
                f"Add it to cfg.model.fields and retrain."
            )
        return self.nets[name]

    def bound(
        self,
        c: torch.Tensor,
        pin: tuple[float, float] | None = None,
        geometry: GeometryPriors | None = None,
    ) -> BoundPINN:
        """This model with one dataset's conditioning row -- and, for a hard-pin
        or front-geometry run, that dataset's measured anchors -- bound (joint
        training/eval/viz)."""
        return BoundPINN(self, c, pin=pin, geometry=geometry)


class BoundPINN:
    """A conditioned model with one dataset's conditioning row bound.

    Joint checkpoints need a per-dataset context on every call; binding it once
    lets every single-dataset consumer (figures, video, reconstruction) render
    a joint model unchanged. For a hard-pin run the dataset's root anchor is
    bound alongside, so the pin gate uses the right root per dataset. Anything
    not overridden falls through to the underlying model (``eps``, trainable
    unknowns, ``fields``...).
    """

    def __init__(
        self,
        model: BubblePINN,
        c: torch.Tensor,
        pin: tuple[float, float] | None = None,
        geometry: GeometryPriors | None = None,
    ):
        self._model = model
        self._c = c
        self._pin = None if pin is None else torch.as_tensor(pin, dtype=torch.float32)
        self._geometry = geometry

    def _ctx(self, x: torch.Tensor) -> torch.Tensor:
        return self._c.expand(x.shape[0], -1)

    def phi(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.phi(
            x, c if c is not None else self._ctx(x), pin=self._pin, geometry=self._geometry
        )

    def alpha(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.alpha(
            x, c if c is not None else self._ctx(x), pin=self._pin, geometry=self._geometry
        )

    def velocity(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._model.velocity(x, c if c is not None else self._ctx(x))

    def front(self, t: torch.Tensor, n_body: int, n_cap: int):
        """This dataset's own front: its conditioning row and its measured
        anchors, never the raw model's."""
        return self._model.front(t, n_body, n_cap, self._c, self._geometry)

    def source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.source(x, c if c is not None else self._ctx(x))

    def pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.pressure(x, c if c is not None else self._ctx(x))

    def temperature(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.temperature(x, c if c is not None else self._ctx(x))

    def __getattr__(self, name: str):
        return getattr(self._model, name)
