"""Time-to-parity: when a learner's cumulative reward overtakes a baseline's.

A comparison is only meaningful if both logs faced the SAME realized
sequence of environment conditions -- which, per the kernel's Common
Random Numbers design (docs/architecture.md, "RNG stream design"), is
guaranteed only when both logs were produced by ``run_episodes`` calls
with the SAME top-level ``seed``. This module cannot verify that
structurally (metrics/ may not import the kernel, and ``EpisodeLog``
itself does not record the seed it was built from), so callers must pass
each log's seed explicitly and a mismatch raises ``ValueError``.
"""

import math
from typing import Sequence

import numpy as np

__all__ = ["time_to_parity", "summarize_parity"]


def time_to_parity(
    learner_log,
    learner_seed: int,
    baseline_log,
    baseline_seed: int,
    stability_window: int = 50,
) -> float:
    """Smallest episode where learner cumulative reward first exceeds the
    baseline's and remains above for ``stability_window`` consecutive
    episodes.

    Parameters
    ----------
    learner_log, baseline_log : EpisodeLog-like
        Must expose ``.reward`` (shape ``(n_episodes,)``) and have the same
        ``n_episodes``.
    learner_seed, baseline_seed : int
        The top-level seed each log was produced with. Must match.
    stability_window : int, default=50
        Number of consecutive episodes the lead must hold for.

    Returns
    -------
    float
        The (integer-valued) crossing episode, or ``math.inf`` if parity
        never occurs (including if it occurs but cannot hold for a full
        ``stability_window`` before the log ends).

    Raises
    ------
    ValueError
        If ``learner_seed != baseline_seed``, if either log is empty, or
        if the two logs have different ``n_episodes``.
    """
    if learner_seed != baseline_seed:
        raise ValueError(
            f"time_to_parity requires both logs from the same seed, got "
            f"learner_seed={learner_seed!r}, baseline_seed={baseline_seed!r}"
        )
    learner_reward = np.asarray(learner_log.reward, dtype=np.float64)
    baseline_reward = np.asarray(baseline_log.reward, dtype=np.float64)
    if learner_reward.size == 0 or baseline_reward.size == 0:
        raise ValueError("time_to_parity requires both logs to have at least one episode")
    if learner_reward.size != baseline_reward.size:
        raise ValueError(
            f"time_to_parity requires equal-length logs, got learner "
            f"n_episodes={learner_reward.size}, baseline n_episodes={baseline_reward.size}"
        )

    n = learner_reward.size
    learner_cum = np.cumsum(learner_reward)
    baseline_cum = np.cumsum(baseline_reward)
    above = learner_cum > baseline_cum

    for episode in range(n):
        if not above[episode]:
            continue
        window_end = episode + stability_window
        if window_end > n:
            break  # cannot verify a full stability window before the horizon ends
        if np.all(above[episode:window_end]):
            return float(episode)
    return math.inf


def summarize_parity(times: Sequence[float], horizon: int) -> dict:
    """Summarize a batch of ``time_to_parity`` results (e.g. one per seed).

    Never silently discards ``math.inf`` entries: ``fraction_never_crossed``
    is always reported alongside ``median_finite`` so a caller cannot
    mistake "median over the runs that happened to cross" for "median
    outcome" -- see docs/statistics.md, "Time-to-parity".

    Parameters
    ----------
    times : sequence of float
        ``time_to_parity`` outputs (each either a finite episode index or
        ``math.inf``).
    horizon : int
        The episode horizon these runs were evaluated over (recorded
        as-is, for context -- not derived from ``times``).

    Returns
    -------
    dict
        ``{"median_finite": float, "fraction_never_crossed": float, "horizon": int}``.
        ``median_finite`` is ``math.inf`` if every run in ``times`` never
        crossed.

    Raises
    ------
    ValueError
        If ``times`` is empty.
    """
    arr = np.asarray(times, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("summarize_parity requires at least one time_to_parity result")
    finite = arr[np.isfinite(arr)]
    fraction_never_crossed = float(np.mean(~np.isfinite(arr)))
    median_finite = float(np.median(finite)) if finite.size > 0 else math.inf
    return {
        "median_finite": median_finite,
        "fraction_never_crossed": fraction_never_crossed,
        "horizon": horizon,
    }
