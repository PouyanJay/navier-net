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
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.func import functional_call

from naviernet.models.attention import SelfAdaptiveWeights
from naviernet.models.film import LiquidFilm
from naviernet.models.geometry import (
    GeometricInterface,
    GeometryContext,
    GeometryPriors,
)
from naviernet.models.layers import AdaptiveTanh, FourierFeatures, inverse_softplus, logit

# A deliberate exception to the usual physics -> models dependency direction
# (the only one): the film net's channel bound and data-anchored start are
# DERIVED from the config's dimensionless groups, exactly as the trainer
# derives them -- duplicating that arithmetic here would let the two drift.
# Neither module imports back into naviernet.models, so no cycle is possible.
from naviernet.physics.film import deposited_thickness
from naviernet.physics.groups import compute_groups
from naviernet.physics.heating import DEPLETION_SMOOTHING, wall_flux_ratio

# What training.self_adaptive weights: nothing, the liquid energy equation, or
# every collocation equation.
SELF_ADAPTIVE_MODES = ("off", "energy", "all")

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
            scales=_resolve(arch, "fourier_scales", None),
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
        self._validate_receding_cap(cfg)
        self._validate_root_attachment(cfg)
        self._validate_pressure_shape(cfg)
        self._validate_film_pressure(cfg)
        self._validate_liquid_film(cfg, names)
        self._validate_pressure_representation(cfg, names)
        self._validate_thermal(cfg, names)
        self._validate_mirror_symmetry(cfg, geometry)
        self._init_hard_kinematics(cfg)
        self._init_hard_walls(cfg, geometry)
        self._field_cache: dict | None = None
        # Read on every thermal-source call; the groups are fixed for the model.
        self._source_coefficient = float(compute_groups(cfg)["mass_flux_to_source"])
        self._init_hard_pin(cfg, pin)

        per_field = getattr(cfg.model, "per_field", None) or {}
        self.nets = nn.ModuleDict(
            {
                name: GeometricInterface(
                    geometry,
                    allow_pinch=self.allow_pinch,
                    n_cond=self.n_cond,
                    evolving_width=self.evolving_width,
                    cap_freedom=self.cap_freedom,
                    cap_delta=self.cap_delta,
                    pressure_shape=self.pressure_shape,
                    shape_delta=self.shape_delta,
                    nucleation_root=self.nucleation_root,
                    root_delta=self.root_delta,
                    nose_shape=self.nose_shape,
                    nose_delta=self.nose_delta,
                    mirror_axis=self.mirror_axis,
                    smooth_apex=bool(getattr(cfg.model, "smooth_apex", False)),
                )
                if name == "phi" and self.front_geometry
                else FieldNet(cfg, arch=per_field.get(name), n_cond=self.n_cond)
                for name in names
            }
        )
        self._init_inverse_unknowns(cfg, names)
        self._init_vapor_pressure()
        # LAST, deliberately: the film net must not consume RNG before any of
        # the shared fields, so a seeded run with the flag off starts from
        # bit-identical weights (regression-tested).
        self._init_liquid_film(cfg)
        # After the film for tidiness; consumes no RNG (zero-init scalars).
        self._init_root_attachment()
        self._init_film_evaporation(cfg)
        # After every field, so a run with it off builds bit-identical nets.
        self._init_self_adaptive(cfg)

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

    def _validate_receding_cap(self, cfg) -> None:
        """The receding Bretherton branch modifies the gap curvature the jump
        condition consumes; without the sharp-interface conditions nothing in
        training reads it, and a flag that silently did nothing would be the
        undriven-knob trap wearing a different hat."""
        self.receding_cap = bool(getattr(cfg.model, "receding_cap", False))
        if self.receding_cap and not self.sharp_interface:
            raise ValueError(
                "model.receding_cap extends the gap curvature the Young-Laplace "
                "jump consumes, so it requires model.sharp_interface=true -- "
                "enable it, or turn model.receding_cap off."
            )

    def _validate_root_attachment(self, cfg) -> None:
        """The attachment pressure blends the jump condition's gap term over the
        patch; without the root DOF there is nothing for the lowered pressure to
        shape, and without the sharp interface there is no jump to blend."""
        self.root_attachment = bool(getattr(cfg.model, "root_attachment", False))
        if not self.root_attachment:
            return
        if not self.nucleation_root:
            raise ValueError(
                "model.root_attachment drives the Hugelschaffer root DOF through "
                "the jump condition, so it requires model.nucleation_root=true -- "
                "enable it, or turn model.root_attachment off."
            )
        if not self.sharp_interface:
            raise ValueError(
                "model.root_attachment blends the Young-Laplace jump's gap term "
                "over the patch, so it requires model.sharp_interface=true -- "
                "enable it, or turn model.root_attachment off."
            )

    def _validate_pressure_shape(self, cfg) -> None:
        """The pressure cannot determine a shape it is not being compared against:
        without the sharp-interface conditions there is no Young-Laplace jump on
        the front, and so nothing to solve the width modes from."""
        if self.pressure_shape and not self.sharp_interface:
            raise ValueError(
                "model.pressure_shape solves the width modes FROM the Young-Laplace "
                "jump, which only exists under model.sharp_interface=true -- enable "
                "it, or turn model.pressure_shape off."
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

    def _validate_pressure_representation(self, cfg, names: list[str]) -> None:
        """The two-phase pressure blends the vapour scalar with the liquid net, so
        it needs both: the sharp-interface unknown ``p_v(t)`` and the ``p`` field."""
        self.pressure_representation = str(
            getattr(cfg.model, "pressure_representation", "single")
        )
        if self.pressure_representation not in ("single", "two_phase"):
            raise ValueError(
                "model.pressure_representation must be 'single' or 'two_phase', got "
                f"{self.pressure_representation!r}."
            )
        if self.pressure_representation == "single":
            return
        if not self.sharp_interface or "p" not in names:
            raise ValueError(
                "model.pressure_representation=two_phase blends the vapour pressure "
                "p_v(t) with the liquid pressure network, so it requires "
                "model.sharp_interface=true and the 'p' field."
            )

    def _validate_thermal(self, cfg, names: list[str]) -> None:
        """The sharp thermal model's composition, rejected before any net is built.

        It gates theta by the distance into the liquid and reads the evaporation
        flux on the explicit front, so it needs the front geometry, the sharp
        interface and the T field. The kinetic interface and the free
        parameterisation are declared but not yet built under it (the legacy
        model carries the kinetic route), and are refused by name.
        """
        model_cfg = cfg.model
        self.thermal_model = str(getattr(model_cfg, "thermal_model", "legacy"))
        self.interface_temperature = str(
            getattr(model_cfg, "interface_temperature", "saturation")
        )
        self.theta_parameterization = str(getattr(model_cfg, "theta_parameterization", "gated"))
        self.theta_gate_width = float(getattr(model_cfg, "theta_gate_width", 0.005))
        self.theta_gate = str(getattr(model_cfg, "theta_gate", "linear"))
        self.theta_gate_length = float(getattr(model_cfg, "theta_gate_length", 0.04))
        if self.theta_gate_length <= 0.0:
            raise ValueError(
                f"model.theta_gate_length={self.theta_gate_length} must be > 0 (a length in L_ref)"
            )
        self.source_mode = str(getattr(model_cfg, "source_mode", "free"))
        # Evaluation and every run without a warm-up read the thermal source from
        # the first step; the trainer clears this through a warm-up (see source).
        self.thermal_source_engaged = True
        kernel = getattr(model_cfg, "source_kernel_width", None)
        self.source_kernel_width = float(kernel if kernel is not None else model_cfg.alpha_eps)
        for name, value, allowed in (
            ("thermal_model", self.thermal_model, ("legacy", "sharp")),
            ("interface_temperature", self.interface_temperature, ("saturation", "kinetic")),
            ("theta_parameterization", self.theta_parameterization, ("gated", "free")),
            ("theta_gate", self.theta_gate, ("linear", "saturating")),
            ("source_mode", self.source_mode, ("free", "thermal")),
        ):
            if value not in allowed:
                raise ValueError(f"model.{name} must be one of {allowed}, got {value!r}")
        if self.thermal_model == "legacy":
            if self.source_mode == "thermal":
                raise ValueError(
                    "model.source_mode=thermal takes the source from the interface flux "
                    "of the sharp thermal model, so it requires model.thermal_model=sharp."
                )
            return
        if "T" not in names or not self.front_geometry or not self.sharp_interface:
            raise ValueError(
                "model.thermal_model=sharp gates theta by the distance into the liquid and "
                "reads the evaporation flux on the explicit front, so it requires the 'T' "
                "field, model.front_geometry=true and model.sharp_interface=true."
            )
        if self.interface_temperature != "saturation":
            raise NotImplementedError(
                "model.interface_temperature=kinetic is not yet built under the sharp "
                "thermal model; the legacy model carries the kinetic (resistance) route."
            )
        if self.theta_parameterization != "gated":
            raise NotImplementedError(
                "model.theta_parameterization=free is not yet built under the sharp "
                "thermal model; use the gated parameterisation."
            )

    def _validate_mirror_symmetry(self, cfg, geometry: GeometryPriors | None) -> None:
        """The mirror axis is the centre of the run's wall span, which the front
        geometry's priors carry; without them there is no axis to reflect about."""
        self.mirror_symmetry = bool(getattr(cfg.model, "mirror_symmetry", False))
        self.mirror_axis: float | None = None
        if not self.mirror_symmetry:
            return
        if not self.front_geometry or geometry is None:
            raise ValueError(
                "model.mirror_symmetry reflects every field about the channel centreline "
                "of the run's wall span, which the front geometry carries, so it requires "
                "model.front_geometry=true."
            )
        self.mirror_axis = 0.5 * (float(geometry.y_min) + float(geometry.y_max))

    @contextmanager
    def caching_fields(self):
        """Within the block, each field is computed once per point batch and
        shared: the equations of a training step read the velocity, the volume
        fraction, the temperature and the pressure at the same collocation
        points, and recomputing them per equation doubled the step's graph (and
        its backward pass) for no change in any value. Entries are keyed on the
        batch tensor itself, so a different batch is never served another's."""
        self._field_cache = {}
        try:
            yield
        finally:
            self._field_cache = None

    def _shared(self, name: str, x: torch.Tensor, c, compute):
        """``compute()``, or its value from this block's cache for the same
        batch ``x`` and context ``c``."""
        cache = self._field_cache
        if cache is None:
            return compute()
        key = (name, id(x), id(c))
        hit = cache.get(key)
        if hit is not None and hit[0] is x and hit[1] is c:
            return hit[2]
        value = compute()
        cache[key] = (x, c, value)  # holding x and c keeps their ids unique
        return value

    def _init_hard_walls(self, cfg, geometry: GeometryPriors | None) -> None:
        """No flow through the side walls by construction (``model.hard_walls``):
        the walls are the physical ones the front geometry's priors carry."""
        self.hard_walls = bool(getattr(cfg.model, "hard_walls", False))
        self.wall_span: tuple[float, float] | None = None
        if not self.hard_walls:
            return
        if str(getattr(cfg.model, "walls_at", "crop")) != "physical" or geometry is None:
            raise ValueError(
                "model.hard_walls holds the flow at the channel's side walls, so it needs "
                "model.walls_at=physical and the front geometry that carries them"
            )
        self.wall_span = (float(geometry.y_min), float(geometry.y_max))

    def _wall_factor(self, x: torch.Tensor) -> torch.Tensor:
        """``4 (y - y_lo)(y_hi - y) / (y_hi - y_lo)^2``: zero at both walls, one
        on the centreline, even about it."""
        lo, hi = self.wall_span
        y = x[:, 1:2]
        return 4.0 * (y - lo) * (hi - y) / (hi - lo) ** 2

    def _init_hard_kinematics(self, cfg) -> None:
        """The exact kinematic condition (``model.hard_kinematics``): it reads the
        interface's normal and speed off the explicit geometry, so it needs one."""
        self.hard_kinematics = bool(getattr(cfg.model, "hard_kinematics", False))
        # Evaluation and every run without a warm-up apply it from the start;
        # the trainer holds it off through the warm-up, as it does the thermal
        # source, so the warm-up stays the outline fit alone.
        self.kinematics_engaged = True
        self.kinematic_blend_length = float(getattr(cfg.model, "kinematic_blend_length", 0.04))
        self.kinematic_side = str(getattr(cfg.model, "kinematic_side", "liquid"))
        if self.kinematic_side not in ("both", "liquid"):
            raise ValueError(
                f"model.kinematic_side must be 'both' or 'liquid', got {self.kinematic_side!r}"
            )
        if not self.hard_kinematics:
            return
        if not self.front_geometry:
            raise ValueError(
                "model.hard_kinematics reads the interface's normal and speed off the explicit "
                "geometry, so it requires model.front_geometry=true"
            )
        if self.kinematic_blend_length <= 0.0:
            raise ValueError(
                f"model.kinematic_blend_length={self.kinematic_blend_length} must be > 0"
            )

    def _reflect(self, x: torch.Tensor) -> torch.Tensor:
        """``(x, y, t) -> (x, 2 y_c - y, t)``: the point's mirror image across the axis."""
        return torch.cat([x[:, :1], 2.0 * self.mirror_axis - x[:, 1:2], x[:, 2:]], dim=1)

    def _even(self, net, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """The net's Reynolds average over the mirror group: exactly even about the axis."""
        if self.mirror_axis is None:
            return net(x, c)
        return 0.5 * (net(x, c) + net(self._reflect(x), c))

    def _odd(self, net, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """The net's odd projection: exactly odd about the axis (the transverse velocity)."""
        if self.mirror_axis is None:
            return net(x, c)
        return 0.5 * (net(x, c) - net(self._reflect(x), c))

    def _init_self_adaptive(self, cfg) -> None:
        """The self-adaptive weight map (``training.self_adaptive``), or None."""
        mode = str(getattr(cfg.training, "self_adaptive", "off"))
        self.attention: SelfAdaptiveWeights | None = None
        if mode == "off":
            return
        from naviernet.physics import registry

        equations = registry.enabled_equations(
            cfg.model.fields, **registry.flags_from(cfg.model)
        )
        keys = [e.weight_key for e in registry.collocation_equations(equations)]
        if mode == "energy":
            keys = [k for k in keys if k == "energy_liquid"]
        if mode not in SELF_ADAPTIVE_MODES or not keys:
            raise ValueError(
                f"training.self_adaptive={mode!r} needs one of {SELF_ADAPTIVE_MODES} "
                "and an equation to weight (energy needs the liquid energy equation)"
            )
        self.attention = SelfAdaptiveWeights(
            tuple(keys), float(cfg.training.self_adaptive_ratio)
        )

    def _init_film_evaporation(self, cfg) -> None:
        """Film evaporation under the footprint (``model.footprint_evaporation``):
        the source coefficient from the groups, and the wet fraction, declared
        or ONE bounded trained scalar that starts exactly at its declared value
        (no RNG)."""
        self.footprint_evaporation = bool(getattr(cfg.model, "footprint_evaporation", False))
        if not self.footprint_evaporation:
            return
        if self.thermal_model != "sharp":
            raise ValueError(
                "model.footprint_evaporation splits the series' wall flux by phase with the "
                "sharp thermal model's energy equation (the liquid's share heats the liquid, "
                "the footprint's share evaporates the film), so it requires "
                "model.thermal_model=sharp."
            )
        groups = compute_groups(cfg)
        self._footprint_source_star = float(groups["footprint_source_star"])
        self._heater_zones = {
            key: float(groups[key])
            for key in ("q_upstream_ratio", "heater_edge_star")
            if key in groups
        }
        self._wet_fraction_init = float(groups["wet_fraction_init"])
        self.wet_fraction_mode = str(getattr(cfg.experiment, "wet_fraction_mode", "fixed"))
        if self.wet_fraction_mode == "fit":
            self._wet_fraction_raw = nn.Parameter(torch.zeros(1))
        self.footprint_wetting = str(getattr(cfg.experiment, "footprint_wetting", "uniform"))
        if self.footprint_wetting == "microlayer":
            self._tau_per_um = float(groups["microlayer_tau_star_per_um"])
            self._microlayer_init = float(groups["microlayer_um_init"])
            self.microlayer_mode = str(cfg.experiment.microlayer_mode)
            if self.microlayer_mode == "fit":
                self._log_microlayer_raw = nn.Parameter(torch.zeros(1))
            self._cover: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def bind_cover_times(self, xs, ys, times) -> None:
        """The instant the images first show each pixel of the floor covered
        (``heating.cover_times``, on the dataset's uniform pixel grid ``xs`` by
        ``ys``), which the microlayer wetting reads."""
        device = next(self.parameters()).device
        self._cover = tuple(
            torch.as_tensor(a, dtype=torch.float32, device=device) for a in (xs, ys, times)
        )

    @property
    def microlayer_um(self):
        """The film the bubble leaves on the floor, microns: the declared value
        when fixed; when fitted, a trained positive scalar starting there."""
        if self.microlayer_mode == "fixed":
            return self._microlayer_init
        return self._microlayer_init * torch.exp(self._log_microlayer_raw)

    def _wetness(self, x: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """The share of the floor still wet at ``x``: the uniform wet share, or
        under the microlayer wetting one while the film the bubble left when it
        covered the spot lasts (its life is the film over the local flux) and
        zero after, ``(N, 1)``."""
        if self.footprint_wetting == "uniform":
            return self.wet_fraction
        if self._cover is None:
            raise RuntimeError(
                "the microlayer wetting reads when the images first cover each spot of the "
                "floor: bind them with bind_cover_times before evaluating the source"
            )
        xs, ys, times = self._cover
        col = ((x[:, 0] - xs[0]) / (xs[1] - xs[0])).round().long().clamp(0, len(xs) - 1)
        row = ((x[:, 1] - ys[0]) / (ys[1] - ys[0])).round().long().clamp(0, len(ys) - 1)
        covered_at = times[row, col].unsqueeze(1)
        life = self._tau_per_um * self.microlayer_um / local.clamp_min(1e-6)
        return torch.sigmoid((covered_at + life - x[:, 2:3]) / (DEPLETION_SMOOTHING * life))

    @property
    def wet_fraction(self):
        """The wet share of the footprint: the declared value when fixed; when
        fitted, a trained scalar sigmoid-bounded in (0, 1) and shifted so the
        zero-init raw lands exactly on the declared start."""
        if not self.footprint_evaporation:
            raise RuntimeError(
                "wet_fraction needs the footprint source: this model was built with "
                "model.footprint_evaporation=false."
            )
        if self.wet_fraction_mode == "fixed":
            return self._wet_fraction_init
        return torch.sigmoid(self._wet_fraction_raw + logit(self._wet_fraction_init))

    def footprint_source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """``footprint_source_star * f_wet * alpha``: the vapour the wall flux
        under the bubble creates by evaporating the film, ``(N, 1)``; zero when
        the feature is off.

        The indicator is detached: the footprint reads the geometry and must not
        reshape it through this side door. The images, the interface conditions
        and the measured front velocity shape the front; the source creates the
        volume the velocity field then has to carry through it.
        """
        if not self.footprint_evaporation:
            return torch.zeros_like(x[:, :1])
        local = wall_flux_ratio(x, self._heater_zones)
        return (
            self._footprint_source_star
            * local
            * self._wetness(x, local)
            * self.alpha(x, c).detach()
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

    def _validate_liquid_film(self, cfg, names: list[str]) -> None:
        """The film is scored on the explicit front, which the trainer only
        samples in sharp mode -- and its later stages feed the jump condition,
        which only exists there. Depletion is driven by the local superheat, so
        the temperature field must exist for the film to have a drive."""
        self.liquid_film = bool(getattr(cfg.model, "liquid_film", False))
        if not self.liquid_film:
            return
        if not self.sharp_interface:
            raise ValueError(
                "model.liquid_film rides on the explicit front the sharp-interface "
                "conditions sample, so it requires model.sharp_interface=true -- "
                "enable it, or turn model.liquid_film off."
            )
        if "T" not in names:
            raise ValueError(
                "model.liquid_film depletes by evaporation driven by the local "
                "superheat, so it requires the 'T' field in model.fields (the "
                f"Stage-B field set); this model has {names}."
            )

    def _init_liquid_film(self, cfg) -> None:
        """``delta(u, t)``: the liquid film's thickness over the front's spine.

        The channel bound and the data-anchored start both come from the
        dimensionless groups, computed here from the config exactly as the
        trainer computes them -- the film's geometry is the channel's, not a
        per-run tunable.
        """
        if not self.liquid_film:
            return
        groups = compute_groups(cfg)
        if "R_gamma_star" not in groups:
            raise ValueError(
                "model.liquid_film needs fluid.molar_mass (the Schrage kinetic "
                "resistance is computed from it); this config predates the field "
                "-- add molar_mass to the fluid config, as configs/fluid/*.yaml do."
            )
        delta_ref = float(deposited_thickness(torch.ones(1, 1), groups))
        self.film = LiquidFilm(0.5 * groups["H_star"], delta_ref, n_cond=self.n_cond)
        # Cached like `liquid_film` itself, so consumers read a model attribute
        # instead of chaining through cfg.
        self.film_stations = int(cfg.model.film_stations)
        # The accommodation coefficient's magnitude is the admitted unknown
        # (measured values span 1e-3 to 1), so the kinetic resistance carries a
        # trained log-scale, `r_int_star`'s film analogue: exp(0) = 1 starts at
        # the literature value, and the fluid-to-fluid RATIO stays computed.
        self._log_film_resistance = nn.Parameter(torch.zeros(1))

    # The patch footprint's starting extent as a fraction of the root cap, its
    # starting growth rate (fractions of the cap per unit t*), and the smooth
    # gate's width as a fraction of the patch length. Inits, not physics: the
    # dry-spot literature has no FC-72 measurement to anchor them (stated in
    # the plan), so they only place the unknowns somewhere trainable -- the
    # mechanism gate is that both MOVE off these.
    PATCH_FRAC_INIT = 0.25
    PATCH_RATE_INIT = 0.5
    PATCH_GATE_FRACTION = 0.25

    def _init_root_attachment(self) -> None:
        """The attachment's two unknowns, both zero-init scalars (no RNG):
        the apparent contact angle (see :meth:`theta_app_deg`) and the patch
        footprint's monotone extent (see :meth:`patch_extent`)."""
        if not self.root_attachment:
            return
        self._theta_app_raw = nn.Parameter(torch.zeros(1))
        self._patch_frac_raw = nn.Parameter(torch.zeros(1))
        self._patch_rate_raw = nn.Parameter(torch.zeros(1))

    def theta_app_deg(self, groups: dict[str, float]) -> torch.Tensor:
        """The apparent contact angle over the patch, in degrees -- ONE trained
        scalar, sigmoid-bounded in the fluid's own window and shifted so the
        zero-init raw lands exactly on the fluid's init (the evaporating-angle
        law at this run's superheat scale, or the advancing angle on water's
        hysteresis path). An inverse unknown like ``r_int_star``: the apparent
        angle under evaporation is emergent, never an FC-72 literal in code.
        """
        if not self.root_attachment:
            raise RuntimeError(
                "theta_app_deg needs the attachment: this model was built with "
                "model.root_attachment=false."
            )
        try:
            lo = float(groups["theta_app_min_deg"])
            hi = float(groups["theta_app_max_deg"])
            init = float(groups["theta_app_init_deg"])
        except KeyError as missing:
            raise ValueError(
                "model.root_attachment needs the per-fluid wetting groups "
                "(theta_app_min/init/max_deg), which this config does not derive "
                "-- it predates the fluid wetting fields. Add theta_e_deg / "
                "theta_adv_deg / theta_rec_deg to the fluid config, as "
                "configs/fluid/*.yaml do."
            ) from missing
        span = max(hi - lo, 1e-6)
        shift = logit((init - lo) / span)
        return lo + span * torch.sigmoid(self._theta_app_raw + shift)

    def patch_extent(self, t: torch.Tensor) -> torch.Tensor:
        """The patch footprint's axial extent from the anchor at times ``t`` of
        shape ``(N, 1)``: a trained monotone fraction of the root cap radius.

        ``sigmoid(logit(f0) + raw + softplus(rate) * (t - t_min))`` -- monotone
        in t by construction (the nose-rate treatment: a dry patch does not
        rewet while the bubble grows) and bounded by the cap. The cap radius is
        DETACHED: the patch reads the geometry, it must not reshape the cap
        through this side door (the jump condition is its one honest channel).
        """
        if not self.root_attachment:
            raise RuntimeError(
                "patch_extent needs the attachment: this model was built with "
                "model.root_attachment=false."
            )
        geometry = self.nets["phi"]
        rate = torch.nn.functional.softplus(
            self._patch_rate_raw + inverse_softplus(self.PATCH_RATE_INIT)
        )
        grown = (t - float(geometry.priors.t_min)).clamp(min=0.0) * rate
        fraction = torch.sigmoid(logit(self.PATCH_FRAC_INIT) + self._patch_frac_raw + grown)
        return fraction * geometry.frame(t).r_root.detach().abs()

    def attachment_fraction(self, points: torch.Tensor) -> torch.Tensor:
        """The attachment field ``a(x, t)`` at front samples ``(N, 3)``: 1 over
        the patch footprint, 0 beyond it, smoothly gated on the axial distance
        from the measured anchor. The gate width scales with the patch (a
        fixed fraction), so the transition stays resolvable as the patch grows.
        """
        if not self.root_attachment:
            raise RuntimeError(
                "attachment_fraction needs the attachment: this model was built "
                "with model.root_attachment=false."
            )
        geometry = self.nets["phi"]
        extent = self.patch_extent(points[:, 2:3])
        distance = (points[:, 0:1] - float(geometry.priors.x_root)).clamp(min=0.0)
        gate = self.PATCH_GATE_FRACTION * extent + 1e-6
        return torch.sigmoid((extent - distance) / gate)

    def film_resistance(self, groups: dict[str, float]) -> torch.Tensor:
        """The effective kinetic resistance ``R_gamma* exp(w)``, an inverse
        unknown scaled off the literature value (see ``_init_liquid_film``)."""
        if not self.liquid_film:
            raise RuntimeError(
                "film_resistance needs the liquid film: this model was built "
                "with model.liquid_film=false."
            )
        return groups["R_gamma_star"] * torch.exp(self._log_film_resistance)

    def film_thickness(self, front, c: torch.Tensor | None = None) -> torch.Tensor:
        """The film thickness at each front sample's axial position, ``(N, 1)``.

        The coordinates are DETACHED: the film field is a record attached to
        the wall, and no term that reads it through this accessor may move the
        front by gradient through the film net's inputs. (Terms that need the
        film's own derivatives -- depletion's ``d delta/dt`` -- build their own
        leaf tensors and call the net directly.)
        """
        if not self.liquid_film:
            raise RuntimeError(
                "film_thickness needs the liquid film: this model was built with "
                "model.liquid_film=false, so there is no film net to evaluate."
            )
        return self.film(front.points[:, 0:1].detach(), front.points[:, 2:3].detach(), c)

    def film_root(self) -> tuple[float, float]:
        """The measured root anchor ``(x, y)`` the film's quadrature starts
        from -- the upstream end of the footprint, and the height the driving
        superheat is read at."""
        priors = self.nets["phi"].priors
        return priors.x_root, priors.y_root

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
        self.evolving_width = bool(getattr(cfg.model, "evolving_width", False))
        self.cap_freedom = bool(getattr(cfg.model, "cap_freedom", False))
        self.cap_delta = float(getattr(cfg.model, "cap_delta", 0.2))
        self.pressure_shape = bool(getattr(cfg.model, "pressure_shape", False))
        self.shape_delta = float(getattr(cfg.model, "shape_delta", 0.3))
        self.nucleation_root = bool(getattr(cfg.model, "nucleation_root", False))
        self.root_delta = float(getattr(cfg.model, "root_delta", 0.6))
        self.nose_shape = bool(getattr(cfg.model, "nose_shape", False))
        self.nose_delta = float(getattr(cfg.model, "nose_delta", 0.6))
        if self.nucleation_root and not self.front_geometry:
            raise ValueError(
                "model.nucleation_root reshapes the front geometry's ROOT CAP, so "
                "it requires model.front_geometry=true; a free level set has no "
                "root cap to reshape."
            )
        if self.nose_shape and not self.front_geometry:
            raise ValueError(
                "model.nose_shape reshapes the front geometry's NOSE CAP, so it "
                "requires model.front_geometry=true; a free level set has no nose "
                "cap to reshape."
            )
        if self.evolving_width and not self.front_geometry:
            raise ValueError(
                "model.evolving_width reparameterises the front geometry's width "
                "profile, so it requires model.front_geometry=true."
            )
        if self.pressure_shape and not self.front_geometry:
            raise ValueError(
                "model.pressure_shape solves the front geometry's own width modes "
                "from the pressure, so it requires model.front_geometry=true; a free "
                "level set has no modes to solve for."
            )
        if self.cap_freedom and not self.front_geometry:
            raise ValueError(
                "model.cap_freedom frees the front geometry's END CAPS, so it "
                "requires model.front_geometry=true; a free level set has no caps "
                "to free -- its interface is already unconstrained."
            )
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
        self.depletable_superheat = bool(getattr(cfg.model, "depletable_superheat", False))
        if self.depletable_superheat and "T" not in names:
            raise ValueError(
                "model.depletable_superheat relaxes the temperature's lower bound, so "
                "it requires the 'T' field in model.fields (the Stage-B field set); "
                f"this model has {names}."
            )
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
        if pin is not None or geometry is not None:
            return torch.sigmoid(self.phi(x, c, pin=pin, geometry=geometry) / self.eps)
        return self._shared("alpha", x, c, lambda: torch.sigmoid(self.phi(x, c) / self.eps))

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

    def apex(
        self,
        t: torch.Tensor,
        c: torch.Tensor | None = None,
        geometry: GeometryPriors | None = None,
    ) -> torch.Tensor:
        """The nose apex ``(x, y)`` at times ``t`` of shape ``(N, 1)``.

        Routed through the model for the same reason as :meth:`front`: a joint
        run's per-dataset view binds its own conditioning row and anchors, and
        going around it would report one dataset's apex for every condition.
        """
        if not self.front_geometry:
            raise RuntimeError(
                "apex() needs the explicit front: this model was built with "
                "model.front_geometry=false, so there is no parameterized nose "
                "to locate."
            )
        return self.nets["phi"].apex(t, GeometryContext(c, geometry))

    def velocity(
        self, x: torch.Tensor, c: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._shared("velocity", x, c, lambda: self._velocity(x, c))

    def _velocity(
        self, x: torch.Tensor, c: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = self._even(self.nets["u"], x, c), self._odd(self.nets["v"], x, c)
        # The wall factor goes on the network's velocity, before the kinematic
        # correction, so the correction stays exact on the interface; what it
        # adds at a wall is its blend's tail, sech^2 of the gap over the blend
        # length (about 1.5% of the interface's normal-speed mismatch across a
        # 34 um gap at the default 12 um blend).
        if self.hard_walls:
            v = v * self._wall_factor(x)
        if not (self.hard_kinematics and self.kinematics_engaged):
            return u, v
        return self._kinematic_correction(x, c, u, v)

    # Below this |grad phi| the normal is undefined (the medial axis deep inside
    # the vapour, where the blend has long faded); the floor only keeps the
    # division finite there.
    _GRAD_FLOOR = 1e-6
    # Points sampled on the interface sit within rounding of it (3e-8 measured),
    # on either side; they belong to the liquid side the condition is set on.
    _ON_INTERFACE = 1e-6

    def _kinematic_correction(
        self, x: torch.Tensor, c: torch.Tensor | None, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``u + (V_n - u.n) n psi``: the network's velocity with its normal part
        replaced by the interface's speed ``V_n`` where ``psi = sech^2(phi / l)``
        is one (the interface), blending back to the network over the length
        ``l``. The tangential part is untouched.

        ``n = -grad phi / |grad phi|`` (out of the vapour) and ``V_n = phi_t /
        |grad phi|`` are the level set's own normal and normal speed, exact on the
        interface and its natural extension off it; ``phi`` is read with the
        geometry's parameters detached, so no flow term can move the outline
        through this. Both are even about a mirror axis, so the correction keeps
        ``u`` even and ``v`` odd.
        """
        grad_on = torch.is_grad_enabled()
        with torch.enable_grad():
            leaf = x if x.requires_grad else x.detach().requires_grad_(True)
            phi = self._phi_with_frozen_geometry(leaf, c)
            phi_x, phi_y, phi_t = torch.autograd.grad(
                phi, leaf, torch.ones_like(phi), create_graph=True
            )[0].split(1, dim=1)
            norm = torch.sqrt(phi_x**2 + phi_y**2).clamp_min(self._GRAD_FLOOR)
            n_x, n_y = -phi_x / norm, -phi_y / norm
            speed = phi_t / norm
            blend = 1.0 - torch.tanh(phi / self.kinematic_blend_length) ** 2
            if self.kinematic_side == "liquid":
                # The vapour (phi > 0) keeps the network's velocity: a step at
                # the interface, where the real velocity jumps too.
                blend = torch.where(phi <= self._ON_INTERFACE, blend, torch.zeros_like(blend))
            shortfall = blend * (speed - (u * n_x + v * n_y))
            u, v = u + shortfall * n_x, v + shortfall * n_y
        if not grad_on:
            return u.detach(), v.detach()
        return u, v

    def source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """The volumetric dilatation from phase change: the free ``s`` network, or
        under ``model.source_mode=thermal`` the interface flux spread over the
        band (:meth:`thermal_source`).

        The thermal source is held back while :attr:`thermal_source_engaged` is
        False: the trainer clears it through the Stage-A warm-up, when the
        energy equation and the temperature conditions are gated off, so the
        continuity residual cannot shape the temperature on its own through the
        flux (the first full-chain benchmark drove it to the thousands that way
        before the thermal physics ever engaged). The free ``s`` carries the
        warm-up instead, exactly as the legacy Stage A does.
        """
        if self.source_mode == "thermal" and self.thermal_source_engaged:
            return self.thermal_source(x, c)
        return self._even(self.nets["s"], x, c)

    def thermal_source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """``s(x, t) = C_m sum_i g_i dl_i K_w(x - X_i(t))``: the liquid-side flux
        through the front at each point's instant, carried into the interface
        band by a normalised Gaussian, so the source integrates to exactly
        ``C_m`` times the flux integrated along the front (the Hardt-Wondra
        distribution with a kernel in place of the Helmholtz solve).

        The front's points and arclength weights are detached: the geometry is
        shaped by the images and the interface conditions, never through this
        side door. The flux itself keeps its gradient, so the temperature learns
        from the growth the source has to carry. Points are grouped by their
        instant, so a batch whose times are binned costs one front per bin.

        Under ``model.footprint_evaporation`` the film's source
        (:meth:`footprint_source`) is added: the wall flux under the bubble
        creates vapour across the footprint, the outline flux along the front.
        """
        coefficient = self._source_coefficient
        sigma = self.source_kernel_width
        source = torch.zeros_like(x[:, :1])
        instants, inverse = torch.unique(x[:, 2].detach(), return_inverse=True)
        for k, instant in enumerate(instants):
            members = inverse == k
            flux, weights, points = self._front_flux(instant, c)
            offsets = x[members, :2].unsqueeze(1) - points.unsqueeze(0)
            kernel = torch.exp(-(offsets**2).sum(dim=-1) / (2.0 * sigma**2))
            kernel = kernel / (2.0 * torch.pi * sigma**2)
            source[members] = coefficient * (kernel * (flux * weights).T).sum(
                dim=1, keepdim=True
            )
        return source + self.footprint_source(x, c)

    def interface_flux_at(self, front, c: torch.Tensor | None = None) -> torch.Tensor:
        """``g = d theta/dn`` on the liquid side at the front samples, ``(N, 1)``;
        the gradient flows to the temperature net only.

        Under the gated parameterisation ``theta = G(d) N`` with ``d`` the
        smoothed distance into the liquid, the flux is read analytically as
        ``G'(0) N (-d phi/dn)``: the slope the gate has just inside the liquid.
        Differentiating ``theta`` at the front itself would read the middle of
        the smoothing, where ``d`` has only half its liquid-side slope, and
        measured on the trusted recipe it returned 0.54 to 0.59 of the gradient
        a few smoothing widths into the liquid. Any other parameterisation is
        differentiated directly.
        """
        leaf = front.points.detach().clone().requires_grad_(True)
        normal = front.normal.detach()
        if self.thermal_model == "sharp" and self.theta_parameterization == "gated":
            phi = self._phi_with_frozen_geometry(leaf, c)
            grad_phi = torch.autograd.grad(phi.sum(), leaf, create_graph=True)[0][:, :2]
            into_liquid = -(grad_phi * normal).sum(dim=1, keepdim=True)
            net = self._even(self._require("T"), leaf, c)
            return self._gate_slope() * net * into_liquid.detach()
        theta = self.temperature(leaf, c)
        gradient = torch.autograd.grad(theta.sum(), leaf, create_graph=True)[0][:, :2]
        return (gradient * normal).sum(dim=1, keepdim=True)

    def _gate_slope(self) -> float:
        """The temperature gate's slope just inside the liquid, per unit distance:
        1 for the linear gate, ``1 / theta_gate_length`` for the saturating one."""
        return 1.0 / self.theta_gate_length if self.theta_gate == "saturating" else 1.0

    def _front_flux(
        self, instant: torch.Tensor, c: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(flux, arclength weights, xy)`` of the front at one instant."""
        model_cfg = self.cfg.model
        n_body, n_cap = int(model_cfg.front_body_samples), int(model_cfg.front_cap_samples)
        front = self.front(instant.reshape(1, 1), n_body, n_cap, c)
        points = front.points.detach()[:, :2]
        return (
            self.interface_flux_at(front, c),
            _arclength_weights(points, n_body, n_cap),
            points,
        )

    def pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """The pressure field. Stage B; raises if the pressure field was not configured.

        Under the two-phase representation this is the physical field: the uniform
        vapour pressure inside the bubble, the liquid network outside, blended by
        the DETACHED indicator so the blend can never reshape the interface. The
        momentum law and the jump condition read :meth:`liquid_pressure` instead.
        """
        liquid = self.liquid_pressure(x, c)
        if self.pressure_representation == "single":
            return liquid
        vapour_fraction = self.alpha(x, c).detach()
        return vapour_fraction * self.p_vapor(x[:, 2:3]) + (1.0 - vapour_fraction) * liquid

    def liquid_pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._shared("liquid_pressure", x, c, lambda: self._liquid_pressure(x, c))

    def _liquid_pressure(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """The liquid-side pressure: the ``p`` network itself. Identical to
        :meth:`pressure` under the single representation."""
        return self._even(self._require("p"), x, c)

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
        return self._shared("temperature", x, c, lambda: self._temperature(x, c))

    def _temperature(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """Non-dimensional superheat. Stage B; raises if T was not configured.

        Bounded to ``(theta_in, 1)`` -- inlet to wall -- by default. Under
        ``model.depletable_superheat`` the floor becomes SATURATION instead:
        liquid that has just boiled a bubble is colder than the inlet it arrived
        at, and that depletion is what throttles evaporation as the bubble
        blankets the wall. With the inlet as a floor the field has a stable place
        to sit and measurably does: it collapsed to a constant (std 4e-07) and
        evaporation became a fixed multiple of interfacial area, which can only
        grow. ``theta_in`` then anchors the inlet as a boundary condition rather
        than flooring the whole field.
        """
        if self.thermal_model == "sharp":
            return self._temperature_gate(x, c) * self._even(self._require("T"), x, c)
        raw = torch.sigmoid(self._even(self._require("T"), x, c))
        if self.depletable_superheat:
            return raw
        return self.theta_in + (1.0 - self.theta_in) * raw

    def _temperature_gate(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """The factor the T network is multiplied by under the sharp model: zero
        in the vapour and on the front; the distance into the liquid itself
        (``model.theta_gate=linear``), or that distance saturating at one beyond
        a few ``theta_gate_length`` (``saturating``), so the far field reads the
        network directly and does not move with the bubble."""
        distance = self._liquid_distance(x, c)
        if self.theta_gate == "saturating":
            return torch.tanh(distance / self.theta_gate_length)
        return distance

    def _liquid_distance(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """The (smoothed) distance into the liquid, ``0.5 (sqrt(phi^2 + d^2) - phi)``:
        zero in the vapour and on the front, ``-phi`` in the liquid. The gate the
        sharp thermal model multiplies the T net by, so theta is exactly saturated
        at the interface and may fall below it in the liquid."""
        phi = self._phi_with_frozen_geometry(x, c)
        width = self.theta_gate_width
        return 0.5 * (torch.sqrt(phi**2 + width**2) - phi)

    def _phi_with_frozen_geometry(
        self, x: torch.Tensor, c: torch.Tensor | None
    ) -> torch.Tensor:
        return self._shared("phi_frozen", x, c, lambda: self._phi_frozen(x, c))

    def _phi_frozen(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """``phi(x)`` with the geometry's parameters detached but its dependence on
        ``x`` kept: the temperature's spatial derivatives see the interface, while
        no thermal term can move it through the gate."""
        geometry = self.nets["phi"]
        frozen = {name: value.detach() for name, value in geometry.named_parameters()}
        frozen.update(dict(geometry.named_buffers()))
        return functional_call(geometry, frozen, (x, c))

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


def _arclength_weights(points: torch.Tensor, n_body: int, n_cap: int) -> torch.Tensor:
    """Trapezoid weights along the front, ``(N, 1)``, from the sampler's layout:
    the upper and lower profiles then the two caps, each ordered along its own
    curve. An open end carries half a spacing beyond its last sample, which is
    how the profiles (which exclude the seams) reach the caps (which include them)."""
    weights = []
    start = 0
    for count in (n_body, n_body, n_cap, n_cap):
        segment = points[start : start + count]
        start += count
        spacing = (segment[1:] - segment[:-1]).norm(dim=1)
        weight = torch.zeros(count, device=points.device, dtype=points.dtype)
        weight[:-1] = weight[:-1] + 0.5 * spacing
        weight[1:] = weight[1:] + 0.5 * spacing
        weight[0] = weight[0] + 0.5 * spacing[0]
        weight[-1] = weight[-1] + 0.5 * spacing[-1]
        weights.append(weight)
    return torch.cat(weights).unsqueeze(1)


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

    def apex(self, t: torch.Tensor):
        """This dataset's own nose apex -- bound exactly like :meth:`front`."""
        return self._model.apex(t, self._c, self._geometry)

    def film_root(self) -> tuple[float, float]:
        """This dataset's own measured root anchor, never the reference
        dataset's -- the same binding rule as :meth:`front`."""
        if self._geometry is not None:
            return self._geometry.x_root, self._geometry.y_root
        return self._model.film_root()

    def film(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None):
        """The film net with this dataset's conditioning row bound: the film
        residuals call the net directly (they need their own leaf tensors for
        its derivatives), so the bound view must interpose here too or a joint
        run would evaluate an unconditioned film."""
        return self._model.film(x, t, c if c is not None else self._ctx(x))

    def source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.source(x, c if c is not None else self._ctx(x))

    def pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.pressure(x, c if c is not None else self._ctx(x))

    def liquid_pressure(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.liquid_pressure(x, c if c is not None else self._ctx(x))

    def thermal_source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.thermal_source(x, c if c is not None else self._ctx(x))

    def footprint_source(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.footprint_source(x, c if c is not None else self._ctx(x))

    def interface_flux_at(self, front, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.interface_flux_at(front, c if c is not None else self._c)

    def temperature(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._model.temperature(x, c if c is not None else self._ctx(x))

    def __getattr__(self, name: str):
        return getattr(self._model, name)
