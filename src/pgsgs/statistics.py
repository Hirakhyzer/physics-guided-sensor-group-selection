from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    pairs: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci95_lower: float
    ci95_upper: float
    candidate_win_rate: float


def paired_bootstrap_delta(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
    *,
    lower_is_better: bool = True,
    bootstrap_samples: int = 5000,
    seed: int = 2026,
) -> PairedComparison:
    """Estimate uncertainty of a matched candidate-minus-baseline difference.

    Pairing is required: values at each index must come from the same domain,
    scenario, seed, and protocol configuration. The bootstrap is descriptive
    evidence and does not repair an invalid experimental design.
    """
    left = np.asarray(baseline, dtype=float)
    right = np.asarray(candidate, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right) or len(left) == 0:
        raise ValueError("baseline and candidate must be non-empty matched one-dimensional samples")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired samples must be finite")

    delta = right - left
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    utility_delta = -delta if lower_is_better else delta

    return PairedComparison(
        pairs=len(delta),
        baseline_mean=float(left.mean()),
        candidate_mean=float(right.mean()),
        mean_delta=float(delta.mean()),
        ci95_lower=float(lower),
        ci95_upper=float(upper),
        candidate_win_rate=float(np.mean(utility_delta > 0.0)),
    )
