"""Solving the interface's width modes FROM the pressure.

The Young-Laplace condition says what curvature the pressure demands at every
point of the front. Everywhere else in this codebase that is used as a PENALTY:
the shape is learned, and the condition scores it afterwards. Measured, that
trade is real -- raising `training.weights.laplace` halves the jump error and
costs ~0.015 IoU, because a penalty at weight 10 against data at 10 is an
argument, and the shape loses some of it.

Here the condition DETERMINES the shape instead. One Gauss-Newton step per
instant picks the width modes whose curvature best matches what the pressure
demands, and a bounded learned correction (`model.shape_delta`) carries only what
the depth-averaged model cannot.
"""

from __future__ import annotations

import torch

from naviernet.models.geometry import SHAPE_MODES
from naviernet.physics.residuals import (
    IN_PLANE_WEIGHT,
    gap_curvature,
    pressure_implied_curvature,
)

# Finite-difference step for the Jacobian, in coefficient units. The modulation is
# `tanh` of a sum of modes, so its curvature response is smooth and O(1) here;
# small enough to be a derivative, large enough to stay clear of float32 noise in
# a curvature that is itself a second derivative.
JACOBIAN_STEP = 1e-2

# Levenberg-style damping, as a fraction of the normal matrix's own scale. Modes
# the front cannot distinguish (a short bubble resolves fewer than four) make
# `J^T J` near-singular, and an undamped solve would answer with an enormous
# coefficient in a direction the data never constrained.
DAMPING = 1e-2


def solve_shape_modes(
    model,
    times: torch.Tensor,
    groups: dict[str, float],
    n_body: int,
    n_cap: int,
    damping: float = DAMPING,
) -> torch.Tensor:
    """The width modes the pressure demands at each instant, ``(M, SHAPE_MODES)``.

    One Gauss-Newton step from the current shape: linearise the front's curvature
    in the coefficients, then solve the normal equations against the curvature the
    pressure implies.

    **The Jacobian is detached; the residual is not.** That is the whole gradient
    design. `J` is how the SHAPE responds to its own coefficients -- a geometric
    fact, and differentiating through it would only add a second-order term. The
    residual carries the pressure nets, so the data loss reaches them THROUGH the
    solved shape: the images get to tell the pressure what shape it ought to
    imply. Freezing `J` is the standard Gauss-Newton treatment and keeps one
    linear solve in the graph instead of a Jacobian's worth of second derivatives.
    """
    geometry = model.nets["phi"]
    device = times.device
    n_modes = SHAPE_MODES
    zeros = torch.zeros(times.shape[0], n_modes, device=device)

    geometry.bind_shape(times, zeros)
    try:
        front = geometry.front(times, n_body=n_body, n_cap=n_cap)
        carried = front.kappa_par
        # In-plane only: the gap term is the channel's, and no shape can change it,
        # so asking the modes to account for it would be asking the impossible.
        # Both sides in the SAME units the jump condition uses: the demanded total
        # curvature less the gap part leaves what the in-plane term must supply,
        # and the in-plane term enters the condition weighted by pi/4.
        demanded = pressure_implied_curvature(model, front, groups) - gap_curvature(
            front.normal_speed, groups
        )
        residual = demanded - IN_PLANE_WEIGHT * carried

        jacobian = _curvature_jacobian(geometry, times, carried.detach(), n_body, n_cap)
    finally:
        # Solved or thrown, the geometry must not be left wearing a probe's shape.
        geometry.bind_shape(None, None)

    instant = _instant_index(front.points[:, 2], times)
    return _normal_equations(jacobian, residual, instant, times.shape[0], n_modes, damping)


def _curvature_jacobian(
    geometry, times: torch.Tensor, carried: torch.Tensor, n_body: int, n_cap: int
) -> torch.Tensor:
    """``d kappa / d a`` per front sample, ``(N, SHAPE_MODES)``, by differences.

    One front evaluation per mode, not per (mode, instant): perturbing the SAME
    mode at every instant at once is still a one-column probe, because each
    instant's shape reads only its own row of coefficients. So the whole Jacobian
    costs ``SHAPE_MODES`` evaluations however many instants are bound.
    """
    columns = []
    for mode in range(SHAPE_MODES):
        probe = torch.zeros(times.shape[0], SHAPE_MODES, device=times.device)
        probe[:, mode] = JACOBIAN_STEP
        geometry.bind_shape(times, probe)
        # NOT under `no_grad`: the front takes its own normal speed by
        # differentiating position with respect to t, so a disabled graph breaks
        # the evaluation itself. Detached per column instead -- the Jacobian is
        # meant to be frozen, and this is where that happens.
        perturbed = geometry.front(times, n_body=n_body, n_cap=n_cap).kappa_par
        columns.append(((perturbed - carried) / JACOBIAN_STEP).detach())
    return torch.cat(columns, dim=1)


def _instant_index(sample_times: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Which bound instant each front sample belongs to, ``(N,)``.

    The front lays its four segments out t-major and concatenates them, so an
    instant's samples are not contiguous and cannot be recovered by reshaping.
    """
    return (sample_times.reshape(-1, 1) - times.reshape(1, -1)).abs().argmin(dim=1)


def _normal_equations(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    instant: torch.Tensor,
    n_times: int,
    n_modes: int,
    damping: float,
) -> torch.Tensor:
    """Per-instant damped least squares, ``(M, K)``.

    Accumulated by scatter rather than by looping the instants: each sample
    contributes to exactly one instant's normal matrix, so one `index_add` builds
    all of them.
    """
    device = jacobian.device
    outer = jacobian.unsqueeze(2) * jacobian.unsqueeze(1)  # (N, K, K)
    rhs = jacobian * residual  # (N, K)

    normal = torch.zeros(n_times, n_modes, n_modes, device=device).index_add(0, instant, outer)
    target = torch.zeros(n_times, n_modes, device=device).index_add(0, instant, rhs)

    # Damping scaled by each instant's own conditioning, so one short frame with a
    # tiny normal matrix is not swamped by a fixed absolute floor.
    scale = torch.diagonal(normal, dim1=1, dim2=2).mean(dim=1).clamp(min=1e-12)
    eye = torch.eye(n_modes, device=device).expand(n_times, n_modes, n_modes)
    damped = normal + damping * scale.reshape(-1, 1, 1) * eye
    return torch.linalg.solve(damped, target.unsqueeze(2)).squeeze(2)


def bind_solved_shape(model, times: torch.Tensor, groups: dict[str, float], cfg) -> None:
    """Solve the width modes at ``times`` and leave them bound on the geometry.

    The single entry point, used by the trainer once a step AND by `load_model`
    once at load, because those two have to agree. If training used the solved
    shape and everything downstream fell back to the learned one, every figure,
    IoU and diagnostic would describe a bubble the model never trained -- the
    quietest possible way for this feature to look like it worked.

    A no-op unless the feature is on, so callers need no branch of their own.
    """
    if not bool(getattr(cfg.model, "pressure_shape", False)):
        return
    coeffs = solve_shape_modes(
        model,
        times,
        groups,
        n_body=int(cfg.model.front_body_samples),
        n_cap=int(cfg.model.front_cap_samples),
    )
    model.nets["phi"].bind_shape(times, coeffs)
