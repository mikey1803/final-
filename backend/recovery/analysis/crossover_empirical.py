"""Empirical crossover-point detection for adaptive-vs-static curves.

Given a learner's metric curve (one value per horizon ``T``, e.g.
late-stage rate or normalized cumulative regret) and a static baseline's
curve over the SAME horizon grid, locate where the learner overtakes the
baseline for good. Operates purely on already-computed curves (plain
float sequences) -- no simulation, no log access -- so this module needs
nothing beyond ``numpy``/``math``.
"""

import math
from typing import Sequence

import numpy as np

__all__ = ["crossover_horizon", "crossover_distribution"]


def crossover_horizon(
    learner_curve: Sequence[float],
    static_curve: Sequence[float],
    horizons: Sequence[float],
) -> float:
    """Smallest ``T`` where ``learner_curve >= static_curve`` and stays
    there for every larger ``T`` in the grid, refined by log-linear
    interpolation between the bracketing grid points.

    Parameters
    ----------
    learner_curve, static_curve : sequence of float
        Equal-length, index-aligned metric values, one per horizon in
        ``horizons``.
    horizons : sequence of float
        The horizon grid, strictly increasing, same length as the curves
        (e.g. ``[100, 300, 1000, 3000, 10000, 30000, 100000]``).

    Returns
    -------
    float
        The (possibly fractional) crossing horizon, log-linearly
        interpolated between the two grid points bracketing where
        ``learner - static`` changes sign, or ``math.inf`` if the learner
        never overtakes and stays ahead through the end of the grid.

    Raises
    ------
    ValueError
        If the three sequences have mismatched lengths, are empty, or
        ``horizons`` is not strictly increasing.
    """
    learner = np.asarray(learner_curve, dtype=np.float64)
    static = np.asarray(static_curve, dtype=np.float64)
    T = np.asarray(horizons, dtype=np.float64)
    if not (learner.shape == static.shape == T.shape):
        raise ValueError(
            f"learner_curve, static_curve, and horizons must have equal length, got "
            f"{learner.shape}, {static.shape}, {T.shape}"
        )
    if T.size == 0:
        raise ValueError("horizons must have at least one point")
    if np.any(np.diff(T) <= 0):
        raise ValueError("horizons must be strictly increasing")

    above = learner >= static
    n = T.size
    for i in range(n):
        if not (above[i] and np.all(above[i:])):
            continue
        if i == 0:
            return float(T[0])
        # The smallest valid i (i > 0) is necessarily preceded by a point
        # BELOW the baseline (see docs/experiments/horizon.md, "Prefix
        # evaluation" is unrelated -- see the crossover section there for
        # why i-1 can't itself be "above" here): interpolate the ZERO of
        # the (learner - static) difference curve, linear in log(T) since
        # the horizon grid itself is geometrically spaced.
        diff_prev = learner[i - 1] - static[i - 1]
        diff_curr = learner[i] - static[i]
        if diff_curr == diff_prev:
            return float(T[i])
        log_t_prev, log_t_curr = math.log(T[i - 1]), math.log(T[i])
        frac = diff_prev / (diff_prev - diff_curr)
        frac = min(max(frac, 0.0), 1.0)
        log_t_cross = log_t_prev + frac * (log_t_curr - log_t_prev)
        return float(math.exp(log_t_cross))
    return math.inf


def crossover_distribution(per_seed_crossings: Sequence[float]) -> dict:
    """Summarize a batch of ``crossover_horizon`` results (e.g. one per seed).

    Never reports the mean: crossover times are heavy-tailed and
    right-censored by ``math.inf`` (a seed whose learner never crosses).
    Averaging censored data either drops those seeds (survivorship bias)
    or corrupts the mean (treating ``inf`` as a huge-but-finite number).
    ``median``/IQR are robust to both problems as long as fewer than half
    the seeds are censored; ``fraction_never_crossed`` reports the
    censoring itself instead of hiding it. See
    docs/experiments/horizon.md, "How to interpret inf crossovers".

    Parameters
    ----------
    per_seed_crossings : sequence of float
        ``crossover_horizon`` outputs, one per seed (each a finite T or
        ``math.inf``).

    Returns
    -------
    dict
        ``{"median": float, "iqr_low": float, "iqr_high": float,
        "min": float, "max": float, "fraction_never_crossed": float}``.
        The first five keys are all ``math.inf`` if every entry is
        ``math.inf`` (nothing finite to summarize).

    Raises
    ------
    ValueError
        If ``per_seed_crossings`` is empty.
    """
    arr = np.asarray(per_seed_crossings, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("crossover_distribution requires at least one crossover_horizon result")
    finite = arr[np.isfinite(arr)]
    fraction_never_crossed = float(np.mean(~np.isfinite(arr)))
    if finite.size == 0:
        return {
            "median": math.inf,
            "iqr_low": math.inf,
            "iqr_high": math.inf,
            "min": math.inf,
            "max": math.inf,
            "fraction_never_crossed": 1.0,
        }
    return {
        "median": float(np.median(finite)),
        "iqr_low": float(np.percentile(finite, 25)),
        "iqr_high": float(np.percentile(finite, 75)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "fraction_never_crossed": fraction_never_crossed,
    }
