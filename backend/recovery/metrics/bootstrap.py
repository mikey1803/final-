"""Bootstrap confidence intervals and non-parametric effect sizes.

Fully vectorized NumPy resampling throughout (no Python-level per-resample
loops) -- ``bootstrap_ci`` with 10,000 resamples over 50 observations must
complete in under 100ms. No p-values, t-tests, Holm correction, or
significance testing here or anywhere in this package -- see
docs/statistics.md for why seed-based p-values are not inferential.
"""

from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "bootstrap_ci",
    "paired_bootstrap_diff",
    "cohens_d",
    "cliffs_delta",
    "magnitude",
]


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
    statistic: Callable = np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic`` over ``values``.

    Quantifies Monte Carlo uncertainty across resamples of THIS finite
    sample -- not inferential uncertainty about a population of real
    deployments. See docs/statistics.md.

    Raises
    ------
    ValueError
        If ``values`` has fewer than 2 observations (nothing to resample a
        meaningful distribution from).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        raise ValueError(f"bootstrap_ci requires at least 2 observations, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resample_stats = statistic(arr[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(resample_stats, alpha)), float(np.quantile(resample_stats, 1.0 - alpha))


def paired_bootstrap_diff(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired mean difference ``mean(x) - mean(y)``.

    ``x`` and ``y`` must be equal-length and index-aligned (e.g. per-seed
    results for two methods run under matching seeds/CRN). Unlike
    ``bootstrap_ci``, which resamples one independent sample, this
    resamples PAIRS jointly (the same resampled indices applied to both
    ``x`` and ``y``), preserving within-pair correlation in every resample.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths, or fewer than 2 pairs.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise ValueError(f"x and y must be equal-length paired samples, got shapes {xa.shape} and {ya.shape}")
    n = xa.size
    if n < 2:
        raise ValueError(f"paired_bootstrap_diff requires at least 2 paired observations, got {n}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diff = xa[idx].mean(axis=1) - ya[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(diff, alpha)), float(np.quantile(diff, 1.0 - alpha))


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d: ``(mean(x) - mean(y)) / pooled_std``.

    ``pooled_std`` uses the standard pooled sample standard deviation
    (``ddof=1`` per sample).

    Raises
    ------
    ValueError
        If either sample has fewer than 2 observations, or the pooled
        standard deviation is ~0 (< 1e-12; d is undefined).
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    nx, ny = xa.size, ya.size
    if nx < 2 or ny < 2:
        raise ValueError(f"cohens_d requires at least 2 observations in each sample, got {nx} and {ny}")
    var_x = xa.var(ddof=1)
    var_y = ya.var(ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std < 1e-12:
        raise ValueError(f"pooled standard deviation is ~0 ({pooled_std!r}); Cohen's d is undefined")
    return float((xa.mean() - ya.mean()) / pooled_std)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta, a non-parametric effect size in [-1, 1].

    +1: every value in ``x`` exceeds every value in ``y``. -1: the
    reverse. 0: no stochastic dominance either way. Vectorized
    ``O(|x| * |y|)`` pairwise comparison.

    Raises
    ------
    ValueError
        If either sample is empty.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size == 0 or ya.size == 0:
        raise ValueError("cliffs_delta requires at least one observation in each sample")
    diff = xa[:, None] - ya[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (xa.size * ya.size))


def magnitude(delta: float) -> str:
    """Romano et al. (2006) magnitude thresholds for Cliff's delta."""
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"
