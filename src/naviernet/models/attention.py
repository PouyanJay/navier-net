"""Self-adaptive weights over space and time (``training.self_adaptive``).

Self-adaptive PINNs (McClenny & Braga-Neto 2023) give every collocation point a
trainable weight that rises where the residual stays high: the networks descend
the weighted loss while the weights ascend it. Their points are fixed; this
trainer draws fresh points every step, so the weight here is a map over
``(x, y, t)``, a small network, which is the continuous form the same paper
fits with a Gaussian process.

The map's output passes a gradient reversal, so one backward pass gives the
fields their descent direction and the map its ascent direction, and one
optimiser steps both. Each term's weights are normalised to a mean of 1 over
the batch and bounded so the largest is at most ``ratio`` times the smallest:
the map can move a term's emphasis between places and times, not inflate it,
and the configured weights between terms keep their meaning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from naviernet.models.layers import FourierFeatures

# The map's size. Deliberately small: it only has to say where the residual
# stays high, a smooth pattern, not resolve the fields.
ATTENTION_FEATS = 16
ATTENTION_SCALE = 3.0
ATTENTION_HIDDEN = 32
ATTENTION_DEPTH = 2


class _ReverseGradient(torch.autograd.Function):
    """Identity forward, negated gradient backward: what the map is trained
    with, so minimising the loss for the fields maximises it for the map."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return -grad


class SelfAdaptiveWeights(nn.Module):
    """One weight map per loss term in ``keys``, read at the collocation points.

    Starts uniform (the last layer at zero), so a run with it on begins as the
    run without it.
    """

    def __init__(self, keys: tuple[str, ...], ratio: float):
        super().__init__()
        if ratio <= 1.0:
            raise ValueError(f"training.self_adaptive_ratio must be > 1, got {ratio}")
        self.keys = tuple(keys)
        # exp(a) / mean, with |a| <= bound: the largest weight over the smallest
        # is at most exp(2 bound) = ratio.
        self.bound = 0.5 * math.log(ratio)
        self.ff = FourierFeatures(3, ATTENTION_FEATS, ATTENTION_SCALE)
        dims = [self.ff.out_dim] + [ATTENTION_HIDDEN] * ATTENTION_DEPTH
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d_in, d_out), nn.Tanh()]
        last = nn.Linear(dims[-1], len(self.keys))
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.mlp = nn.Sequential(*layers, last)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{key: weights}``, each ``(N, 1)`` with mean 1 over the ``N``
        points. The coordinates are detached: the map reads where a point is,
        and nothing of the fields' derivatives flows through it."""
        logits = _ReverseGradient.apply(self.mlp(self.ff(x.detach())))
        raw = torch.exp(self.bound * torch.tanh(logits))
        weights = raw / raw.mean(dim=0, keepdim=True)
        return {key: weights[:, i : i + 1] for i, key in enumerate(self.keys)}
