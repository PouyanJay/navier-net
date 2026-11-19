"""The root's depth law (``model.depth_drain``).

The edge rule takes the bubble as flat across the channel's depth, so the
depth's part of the pressure step is the channel's own. Near the root that
does not hold: the tail narrows there, its section rounds, and the pressure
step the flat rule asks for is off by an amount that depends on the distance
from the root tip and not on the frame.

The law here is that correction to the pressure step, in the run's pressure
units: values at equally spaced distances from the root tip joined by straight
lines, zero at its reach and beyond. It is one curve for every instant, and it
is not trained. A run that takes the flag on fits it once to the pressure step
it already has (:meth:`DepthRootCurve.fit`) and holds it from there: a curve
the solver could move would be a place to put any error it did not want to
answer for.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The state-dict prefix of the law: the trainer recognises a checkpoint saved
# without it (a run continued with the flag newly on).
DEPTH_PREFIX = "depth_root."

# The fit's ridge, as a share of the mean weight a knot carries: a knot no
# sample reaches stays at zero.
FIT_RIDGE = 1e-6


class DepthRootCurve(nn.Module):
    """The correction to the edge's pressure step against the distance from
    the root tip, ``knots`` values over ``reach`` (the run's length units),
    held fixed once fitted.

    ``starts_at`` is the first image's time, bound from the dataset: the law
    is a statement about the imaged bubble, and is not applied before it.
    """

    knot_values: torch.Tensor

    def __init__(self, reach: float, knots: int):
        super().__init__()
        if not reach > 0.0:
            raise ValueError(f"the root law's reach must be positive, got {reach}")
        if knots < 2:
            raise ValueError(f"the root law needs at least two knots, got {knots}")
        self.reach = float(reach)
        self.spacing = self.reach / knots
        # A buffer, not a parameter: saved with the run, never trained.
        self.register_buffer("knot_values", torch.zeros(knots))
        self._starts_at: float | None = None

    def bind_first_image(self, time: float) -> None:
        """The first image's time, from the dataset."""
        self._starts_at = float(time)

    @property
    def starts_at(self) -> float:
        """The time the law, and the split it belongs to, applies from."""
        if self._starts_at is None:
            raise RuntimeError(
                "model.depth_drain acts from the first image on and its time is not "
                "bound: bind it with bind_first_image before evaluating the edge rule"
            )
        return self._starts_at

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """The correction at each ``distance`` from the root tip ``(N, 1)``, ``(N, 1)``."""
        return self._basis(distance) @ self.knot_values.reshape(-1, 1)

    def _basis(self, distance: torch.Tensor) -> torch.Tensor:
        """Each knot's share at each distance, ``(N, knots)``: one at its own
        knot, falling to zero at its neighbours. The knot at the reach is not
        among them, which is what holds the law at zero there."""
        position = distance.reshape(-1, 1).clamp(min=0.0) / self.spacing
        knots = torch.arange(
            len(self.knot_values), device=distance.device, dtype=position.dtype
        )
        return torch.relu(1.0 - (position - knots).abs())

    @torch.no_grad()
    def fit(self, distance: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> None:
        """Set the law to the weighted least-squares fit of ``target`` at the
        samples ``distance`` from the root tip (all ``(N, 1)``)."""
        basis = self._basis(distance).double()
        weighted = basis * weights.double().reshape(-1, 1)
        normal = weighted.T @ basis
        ridge = FIT_RIDGE * normal.diagonal().mean().clamp(min=1e-30)
        normal = normal + ridge * torch.eye(
            len(normal), dtype=normal.dtype, device=normal.device
        )
        solution = torch.linalg.solve(normal, weighted.T @ target.double().reshape(-1, 1))
        self.knot_values.copy_(solution.reshape(-1).to(self.knot_values.dtype))
