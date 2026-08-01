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

from naviernet.models.geometry import GeometricInterface, GeometryPriors
from naviernet.models.layers import AdaptiveTanh, FourierFeatures


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
        self._validate_front_geometry(cfg, geometry)
        self._init_hard_pin(cfg, pin)

        names = list(fields if fields is not None else cfg.model.fields)
        per_field = getattr(cfg.model, "per_field", None) or {}
        self.nets = nn.ModuleDict(
            {
                name: GeometricInterface(geometry)
                if name == "phi" and self.front_geometry
                else FieldNet(cfg, arch=per_field.get(name), n_cond=self.n_cond)
                for name in names
            }
        )
        self._init_inverse_unknowns(cfg, names)

    def _validate_front_geometry(self, cfg, geometry: GeometryPriors | None) -> None:
        """Reject unusable front-geometry compositions loudly, before any net is
        built: the geometry pins exactly (hard_pin is redundant and conflicting),
        joint conditioning is unsupported, priors are required, and a per-field
        phi override would be silently ignored."""
        self.front_geometry = bool(getattr(cfg.model, "front_geometry", False))
        if not self.front_geometry:
            return
        if getattr(cfg.model, "hard_pin", False):
            raise ValueError(
                "model.front_geometry already pins the root exactly by construction; "
                "it is mutually exclusive with model.hard_pin -- disable one."
            )
        if self.n_cond > 0:
            raise NotImplementedError(
                "model.front_geometry is not yet supported for joint (multi-dataset) "
                "runs; disable it for joint runs for now."
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
    ) -> torch.Tensor:
        """Level-set field; its zero contour is the interface. With the hard pin
        on, the zero contour is anchored to the root for all t (see __init__).
        ``pin`` is the per-call anchor a joint run's bound view supplies; a
        single-dataset model carries its own."""
        raw = self.nets["phi"](x, c)
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
    ) -> torch.Tensor:
        """Volume fraction, bounded in (0, 1) by construction."""
        return torch.sigmoid(self.phi(x, c, pin=pin) / self.eps)

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

    def bound(self, c: torch.Tensor, pin: tuple[float, float] | None = None) -> BoundPINN:
        """This model with one dataset's conditioning row -- and, for a hard-pin
        run, that dataset's root anchor -- bound (joint training/eval/viz)."""
        return BoundPINN(self, c, pin=pin)


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
        self, model: BubblePINN, c: torch.Tensor, pin: tuple[float, float] | None = None
    ):
        self._model = model
        self._c = c
        self._pin = None if pin is None else torch.as_tensor(pin, dtype=torch.float32)

    def _ctx(self, x: torch.Tensor) -> torch.Tensor:
        return self._c.expand(x.shape[0], -1)

    def phi(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.phi(x, c if c is not None else self._ctx(x), pin=self._pin)

    def alpha(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.alpha(x, c if c is not None else self._ctx(x), pin=self._pin)

    def velocity(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._model.velocity(x, c if c is not None else self._ctx(x))

    def source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.source(x, c if c is not None else self._ctx(x))

    def pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.pressure(x, c if c is not None else self._ctx(x))

    def temperature(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.temperature(x, c if c is not None else self._ctx(x))

    def __getattr__(self, name: str):
        return getattr(self._model, name)
