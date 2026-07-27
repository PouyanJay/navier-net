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

    def __init__(self, cfg, fields: Sequence[str] | None = None, n_cond: int = 0):
        super().__init__()
        self.cfg = cfg
        self.eps = float(cfg.model.alpha_eps)
        self.n_cond = int(n_cond)
        names = list(fields if fields is not None else cfg.model.fields)
        per_field = getattr(cfg.model, "per_field", None) or {}
        self.nets = nn.ModuleDict(
            {name: FieldNet(cfg, arch=per_field.get(name), n_cond=self.n_cond) for name in names}
        )
        # Stage-B inverse unknowns, present only when temperature is modelled:
        # the interfacial resistance closing evaporation and the inlet superheat.
        # Both are inferred, not measured (the dataset calls them unknowns).
        if "T" in names:
            self._log_r_int = nn.Parameter(torch.zeros(1))
            self._theta_in_raw = nn.Parameter(torch.zeros(1))

    @property
    def fields(self) -> list[str]:
        return list(self.nets.keys())

    def phi(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Raw level-set field; its zero contour is the interface."""
        return self.nets["phi"](x, c)

    def alpha(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Volume fraction, bounded in (0, 1) by construction."""
        return torch.sigmoid(self.phi(x, c) / self.eps)

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
