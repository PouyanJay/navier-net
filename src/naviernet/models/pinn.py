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


class FieldNet(nn.Module):
    """Fourier-feature MLP with adaptive-tanh activations for a single field."""

    def __init__(self, cfg, out_dim: int = 1):
        super().__init__()
        model_cfg = cfg.model
        self.ff = FourierFeatures(
            in_dim=3,
            n_feats=model_cfg.fourier_feats,
            scale=model_cfg.fourier_scale,
        )

        dims = [self.ff.out_dim] + [model_cfg.hidden] * model_cfg.layers
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [
                nn.Linear(d_in, d_out),
                AdaptiveTanh(d_out, model_cfg.nodewise_activation),
            ]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ff(x))


class BubblePINN(nn.Module):
    """Ensemble of field networks.

    Stage A uses ``phi`` (yielding alpha), the velocity components ``u`` and
    ``v``, and the inferred dilatation source ``s``. Stage B adds ``p`` and
    ``T`` by listing them in ``cfg.model.fields``.

    Every accessor takes points ``x`` of shape ``(N, 3)`` ordered ``(x, y, t)``
    and returns columns of shape ``(N, 1)``.
    """

    def __init__(self, cfg, fields: Sequence[str] | None = None):
        super().__init__()
        self.cfg = cfg
        self.eps = float(cfg.model.alpha_eps)
        names = list(fields if fields is not None else cfg.model.fields)
        self.nets = nn.ModuleDict({name: FieldNet(cfg) for name in names})

    @property
    def fields(self) -> list[str]:
        return list(self.nets.keys())

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Raw level-set field; its zero contour is the interface."""
        return self.nets["phi"](x)

    def alpha(self, x: torch.Tensor) -> torch.Tensor:
        """Volume fraction, bounded in (0, 1) by construction."""
        return torch.sigmoid(self.phi(x) / self.eps)

    def velocity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.nets["u"](x), self.nets["v"](x)

    def source(self, x: torch.Tensor) -> torch.Tensor:
        """Inferred volumetric dilatation from phase change."""
        return self.nets["s"](x)

    def pressure(self, x: torch.Tensor) -> torch.Tensor:
        """Stage B. Raises if the pressure field was not configured."""
        return self._require("p")(x)

    def temperature(self, x: torch.Tensor) -> torch.Tensor:
        """Stage B. Raises if the temperature field was not configured."""
        return self._require("T")(x)

    def _require(self, name: str) -> nn.Module:
        if name not in self.nets:
            raise KeyError(
                f"field {name!r} is not in this model (has: {self.fields}). "
                f"Add it to cfg.model.fields and retrain."
            )
        return self.nets[name]
