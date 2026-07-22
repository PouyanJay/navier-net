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
