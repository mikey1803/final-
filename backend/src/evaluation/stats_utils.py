"""Shared statistics helpers for the Phase 3 rigor upgrades.

Bootstrap confidence intervals, Holm-Bonferroni multiple-comparison
correction, and Cliff's delta — used across the main harness, the
randomized reward-matrix sweep, and the non-stationarity experiment.
"""

from typing import Callable, Sequence

import numpy as np

MC_VARIATION_NOTE = (
    "Confidence intervals reflect Monte Carlo variation across simulation "
    "seeds, not sampling variability from a population of real agent "
    "deployments."
)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    statistic: Callable = np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for *statistic* over *values*.

    See ``MC_VARIATION_NOTE`` — this quantifies variation across the
    simulation's own trial seeds, not real-world sampling variability.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_stats = statistic(arr[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(resample_stats, alpha)), float(np.quantile(resample_stats, 1.0 - alpha))


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta, a non-parametric effect size in [-1, 1].

    +1 => every value in x exceeds every value in y; -1 => the reverse;
    0 => no stochastic dominance either way. Vectorized O(n*m) comparison.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    diff = xa[:, None] - ya[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(xa) * len(ya)))


def cliffs_delta_magnitude(delta: float) -> str:
    """Romano et al. (2006) magnitude thresholds for Cliff's delta."""
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    return "large"


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the same order as the input, each
    enforced to be monotonically non-decreasing across ranks and capped
    at 1.0 (the standard Holm procedure).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        if np.isnan(val):
            val = 1.0
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()
