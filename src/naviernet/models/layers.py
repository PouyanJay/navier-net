"""Building blocks for the field networks.

Both layers address the same difficulty: a plain tanh MLP on raw ``(x, y, t)``
is spectrally biased towards smooth functions, and a bubble interface is
anything but smooth. Fourier features supply high-frequency content up front;
adaptive activations let the network tune its own nonlinearity to use it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Gaussian Fourier embedding ``x -> [sin(xB), cos(xB)]``.

    The frequency matrix ``B`` is drawn once and held fixed as a buffer, so it
    travels with the checkpoint and a reloaded model sees exactly the same
    embedding it was trained with.

    Args:
        in_dim: Input dimensionality (3 for ``x, y, t``).
        n_feats: Number of random frequencies; output width is ``2 * n_feats``.
        scale: Standard deviation of ``B``. Larger values represent sharper
            gradients but make optimisation harder.
    """

    def __init__(self, in_dim: int = 3, n_feats: int = 64, scale: float = 3.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_feats) * scale)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.B
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


class AdaptiveTanh(nn.Module):
    """``tanh(a * x)`` with a trainable slope, after Jagtap et al. (2020).

    With ``nodewise=True`` each neuron owns its slope, which is the variant used
    in the reference PINN literature; with ``nodewise=False`` the layer shares a
    single scalar.
    """

    def __init__(self, width: int, nodewise: bool = True):
        super().__init__()
        self.a = nn.Parameter(torch.ones((width,) if nodewise else (1,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a * x)


# Hidden layout of the small nets that parameterize low-frequency curves -- a
# rate, a width profile, a centerline, a film thickness. Deliberately module
# constants, not config: those nets exist to represent smooth 1-D functions,
# and capacity beyond this re-admits the wiggles their representations forbid.
CURVE_HIDDEN = 32
CURVE_DEPTH = 2


def mlp(in_dim: int, out_bias: float = 0.0) -> nn.Sequential:
    """A small tanh MLP whose LAST layer starts at zero weights and the given
    bias: the net begins as the constant ``out_bias`` and learns deviations --
    the data-anchored initialization the curve nets' priors provide."""
    layers: list[nn.Module] = []
    dims = [in_dim] + [CURVE_HIDDEN] * CURVE_DEPTH
    for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(d_in, d_out), nn.Tanh()]
    last = nn.Linear(dims[-1], 1)
    # Small (not zero: exact zeros would block gradient into the hidden layers)
    # so the net starts within a hair of the constant and can still learn.
    nn.init.normal_(last.weight, std=0.01)
    nn.init.constant_(last.bias, out_bias)
    layers.append(last)
    return nn.Sequential(*layers)


def logit(p: float) -> float:
    """The pre-sigmoid value landing exactly on ``p``, clamped away from 0/1 --
    how a curve net's output bias is anchored at a measured starting value."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(p)))


def inverse_softplus(value: float) -> float:
    """The pre-softplus value landing exactly on ``value`` -- ``logit``'s
    sibling for rate-like quantities anchored at a measured positive start."""
    value = max(value, 1e-6)
    return float(torch.log(torch.expm1(torch.tensor(value))))
