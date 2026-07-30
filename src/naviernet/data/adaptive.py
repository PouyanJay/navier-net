"""Residual-adaptive collocation resampling (RAD; Wu et al., arXiv:2207.10289).

Uniform collocation under-resolves the moving interface and the late times where the
PDE residual is largest. RAD periodically redraws the collocation set from a
distribution proportional to the residual magnitude, so points concentrate where the
solution is hardest -- while a uniform base fraction is always kept, so no region is
abandoned (the "no silent caps" guardrail).

Pure sampling utilities here; the trainer supplies the candidate pool, its residual
magnitudes, and a uniform base pool, and calls :func:`rad_resample`.
"""

from __future__ import annotations

import numpy as np
import torch

# RAD distribution p(x) ∝ (r(x)^k / mean(r^k)) + c (Wu et al., eq. 2). k sharpens the
# concentration; c is a uniform floor so no candidate is ever unreachable. The paper's
# default is c=1 (a 50/50 residual/uniform mix) for when RAD is the *only* sampler; here
# `resample_fraction` already keeps a separate uniform base, so a smaller floor lets the
# adaptive fraction concentrate strongly on the residual while c>0 still guards the
# degenerate all-equal-residual case. Fixed algorithm constants, not physical tunables.
_RAD_K = 1.0
_RAD_C = 0.1


def rad_probabilities(mag: torch.Tensor, k: float = _RAD_K, c: float = _RAD_C) -> torch.Tensor:
    """The RAD sampling probability over candidates with residual magnitude ``mag``:
    ``p ∝ mag^k / mean(mag^k) + c``, normalised to sum to 1 (uniform when all equal)."""
    weight = mag.detach() ** k
    weight = weight / weight.mean().clamp_min(1e-12) + c
    return weight / weight.sum()


def rad_resample(
    cand_x: torch.Tensor,
    mag: torch.Tensor,
    base_x: torch.Tensor,
    n: int,
    fraction: float,
    rng: np.random.Generator,
    k: float = _RAD_K,
    c: float = _RAD_C,
) -> torch.Tensor:
    """A refreshed collocation pool of ``n`` points.

    ``round(n*fraction)`` points are drawn (without replacement) from ``cand_x`` with the
    RAD probability -- concentrating on high-residual regions -- and the remaining points
    are taken from ``base_x`` (a freshly sampled uniform pool) for global coverage.
    Seed-deterministic via ``rng``.
    """
    n_adaptive = min(cand_x.shape[0], round(n * fraction))
    n_base = n - n_adaptive
    probs = rad_probabilities(mag, k, c).cpu().numpy()
    idx = rng.choice(cand_x.shape[0], size=n_adaptive, replace=False, p=probs)
    adaptive = cand_x[torch.as_tensor(idx, device=cand_x.device)]
    return torch.cat([adaptive, base_x[:n_base]], dim=0)
