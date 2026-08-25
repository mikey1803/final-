"""Time-varying normalization and recovery statistics for drifted runs.

After a drift event the oracle rate and the random-policy floor both
change (they are properties of the ACTIVE matrix), so a single global
``recovery.theory.analytic.normalize`` call over a whole drifted run is
mathematically invalid: it would score post-shift episodes against a
pre-shift ceiling/floor that no longer describes what is achievable. This
module normalizes every episode against the matrix that was ACTUALLY
active for it, reconstructed from a ``recovery.envs.drift.
DriftingEnvironment.matrix_history()``-shaped compact history -- never
against the run's pre-shift endpoints.

Package-level import constraint (see ``recovery/metrics/__init__.py``):
this module imports only ``numpy``, ``pandas``, and
``recovery.theory.analytic`` -- reusing ``oracle_rate``/``random_rate``
from there rather than recomputing them, so there is exactly one
implementation of "oracle ceiling" / "random floor" in this codebase.
Nothing here imports ``recovery.envs`` -- ``matrix_history`` is consumed
structurally, as a sequence of ``(episode, matrix)`` pairs (a plain list
of ``recovery.envs.drift.DriftEvent`` works, and so does a plain list of
2-tuples), matching this package's established duck-typing convention.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from recovery.theory.analytic import oracle_rate, random_rate

__all__ = [
    "matrix_segments",
    "time_varying_normalize",
    "post_shift_window_score",
    "recovery_delta",
    "adaptation_half_life",
]


def matrix_segments(
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    n_episodes: int,
) -> list:
    """Turn a compact ``(episode, matrix)`` history into a list of
    ``(start, end, matrix)`` segments covering ``[0, n_episodes)`` --
    the ONE shared primitive every time-varying metric in this package
    (normalization here, regret in ``recovery.metrics.drift_regret``)
    builds on, so "which matrix was active when" is computed exactly
    once, not reimplemented per metric.

    ``end`` of the last segment is ``n_episodes`` (extends to the end of
    the run); every other segment's ``end`` is the NEXT history entry's
    episode. Segments starting at or after ``n_episodes`` are dropped;
    the last segment before ``n_episodes`` is clipped to it.

    Raises
    ------
    ValueError
        If ``matrix_history`` is empty, or has no entry starting before
        ``n_episodes``.
    """
    events = sorted(matrix_history, key=lambda e: e[0])
    if not events:
        raise ValueError("matrix_history must contain at least one entry")
    segments = []
    for i, (episode, matrix) in enumerate(events):
        start = int(episode)
        if start >= n_episodes:
            break
        end = int(events[i + 1][0]) if i + 1 < len(events) else n_episodes
        end = min(end, n_episodes)
        if end <= start:
            continue
        segments.append((start, end, np.asarray(matrix, dtype=np.float64)))
    if not segments:
        raise ValueError(
            f"no matrix_history entry starts before n_episodes={n_episodes}; cannot normalize"
        )
    return segments


def _segments(
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    n_episodes: int,
    p: Optional[np.ndarray],
) -> list:
    """``matrix_segments`` plus each segment's ``(oracle_rate,
    random_rate)`` -- the normalization-specific extension of the shared
    primitive above. Returns ``(start, end, oracle_rate, random_rate)``.
    """
    return [
        (start, end, oracle_rate(matrix, p), random_rate(matrix, p))
        for start, end, matrix in matrix_segments(matrix_history, n_episodes)
    ]


def time_varying_normalize(
    reward: np.ndarray,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    p: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize every episode's reward against the matrix active AT THAT
    EPISODE, never against a single global (e.g. pre-shift or final)
    matrix.

    ``normalized[i] = (reward[i] - random_rate(P_i, p)) / (oracle_rate(P_i, p) - random_rate(P_i, p))``
    where ``P_i`` is whichever ``matrix_history`` entry was active at
    episode ``i``. Reduces to
    ``recovery.theory.analytic.normalize`` applied uniformly when
    ``matrix_history`` has exactly one entry (the stationary case).

    Parameters
    ----------
    reward : numpy.ndarray, shape (n_episodes,)
        Per-episode 0/1 (or continuous) reward.
    matrix_history : sequence of (int, numpy.ndarray)
        Compact change-event history, e.g.
        ``recovery.envs.drift.DriftingEnvironment.matrix_history()``.
        Must contain at least one entry starting at or before episode 0.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform (passed through to
        ``oracle_rate``/``random_rate``).

    Returns
    -------
    numpy.ndarray, shape (n_episodes,)
        Per-episode normalized score (0.0 = that episode's active
        random floor, 1.0 = that episode's active oracle ceiling).

    Raises
    ------
    ValueError
        If ``matrix_history`` is empty, has no entry before
        ``len(reward)``, or if any segment's matrix is degenerate
        (``oracle_rate - random_rate < 1e-12``, mirroring
        ``recovery.theory.analytic.normalize``'s own guard).
    """
    reward = np.asarray(reward, dtype=np.float64)
    n = reward.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for start, end, o, r in _segments(matrix_history, n, p):
        denom = o - r
        if denom < 1e-12:
            raise ValueError(
                f"segment [{start}, {end}) has oracle_rate - random_rate = {denom!r} < 1e-12; "
                f"cannot normalize (degenerate matrix for this segment)"
            )
        out[start:end] = (reward[start:end] - r) / denom
    return out


def post_shift_window_score(
    reward: np.ndarray,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    shift_episode: int,
    window: int,
    p: Optional[np.ndarray] = None,
) -> float:
    """Mean time-varying-normalized score over
    ``[shift_episode, shift_episode + window)``.

    Every episode in the window is normalized against WHICHEVER matrix
    was active for it (via ``time_varying_normalize``) -- correct even if
    a further shift occurs inside the window itself.

    Parameters
    ----------
    reward : numpy.ndarray, shape (n_episodes,)
        Per-episode reward.
    matrix_history : sequence of (int, numpy.ndarray)
        Compact change-event history.
    shift_episode : int
        Start of the window (typically the episode a shift took effect).
    window : int
        Window length in episodes (clipped to the available log length).
    p : numpy.ndarray, optional
        Type prior.

    Returns
    -------
    float
        Mean normalized score over the (possibly clipped) window.

    Raises
    ------
    ValueError
        If the window is empty (``shift_episode >= len(reward)`` or
        ``window <= 0``).
    """
    normalized = time_varying_normalize(reward, matrix_history, p)
    n = normalized.shape[0]
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window!r}")
    start = int(shift_episode)
    end = min(start + int(window), n)
    if start >= n or end <= start:
        raise ValueError(
            f"post-shift window [{start}, {end}) is empty for a log of length {n}"
        )
    return float(np.nanmean(normalized[start:end]))


def recovery_delta(
    reward: np.ndarray,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    shift_episode: int,
    pre_window: int,
    post_window: int,
    p: Optional[np.ndarray] = None,
) -> float:
    """``pre_score - post_score``: how much normalized performance dropped
    across the shift.

    ``pre_score`` is the mean time-varying-normalized score over
    ``[shift_episode - pre_window, shift_episode)`` (using whichever
    matrix -- always pre-shift, by construction -- was active there);
    ``post_score`` is the same over
    ``[shift_episode, shift_episode + post_window)`` (post-shift matrix).
    Positive values mean the shift hurt (performance fell); negative
    values mean it helped (e.g. ``GradualRotationDrift`` toward a
    strictly better target). This is the run-level building block
    ``experiments/drift_analysis.py`` bootstraps a confidence interval
    over (across seeds) -- this function itself returns one number for
    one run, no resampling here (see
    ``recovery.metrics.bootstrap.paired_bootstrap_diff``, reused
    unchanged at the analysis layer, never reimplemented here).

    Parameters
    ----------
    reward : numpy.ndarray, shape (n_episodes,)
        Per-episode reward.
    matrix_history : sequence of (int, numpy.ndarray)
        Compact change-event history.
    shift_episode : int
        The episode the shift being measured took effect.
    pre_window, post_window : int
        Number of episodes to average immediately before/after the shift.
    p : numpy.ndarray, optional
        Type prior.

    Returns
    -------
    float
        ``pre_score - post_score``.

    Raises
    ------
    ValueError
        If either window is empty (not enough pre- or post-shift data).
    """
    normalized = time_varying_normalize(reward, matrix_history, p)
    n = normalized.shape[0]
    shift_episode = int(shift_episode)
    pre_start = max(0, shift_episode - int(pre_window))
    if pre_start >= shift_episode:
        raise ValueError(
            f"pre-shift window [{pre_start}, {shift_episode}) is empty -- not enough "
            f"pre-shift episodes for pre_window={pre_window!r}"
        )
    pre_score = float(np.nanmean(normalized[pre_start:shift_episode]))

    post_end = min(shift_episode + int(post_window), n)
    if post_end <= shift_episode:
        raise ValueError(
            f"post-shift window [{shift_episode}, {post_end}) is empty for a log of length {n}"
        )
    post_score = float(np.nanmean(normalized[shift_episode:post_end]))
    return pre_score - post_score


def adaptation_half_life(
    reward: np.ndarray,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    shift_episode: int,
    baseline_window: int = 50,
    rolling_window: int = 50,
    target_fraction: float = 0.5,
    p: Optional[np.ndarray] = None,
) -> float:
    """Episodes-since-shift until the ROLLING time-varying-normalized
    score recovers to within ``target_fraction`` of the pre-shift
    baseline.

    Method: compute ``baseline`` as the mean normalized score over the
    ``baseline_window`` episodes immediately before ``shift_episode``.
    Compute a ``rolling_window``-wide rolling mean of the post-shift
    normalized series (``pandas.Series.rolling(..., min_periods=rolling_window)``
    -- a FULL window is required before any rolling estimate counts, so a
    single lucky/unlucky early episode can never masquerade as
    "recovered"). Find the RUNNING MINIMUM (the trough) of that rolling
    series, then search FORWARD FROM THE TROUGH ONLY for the first point
    at or above ``target = trough + target_fraction * (baseline - trough)``.
    Searching from the trough (not from episode 0 of the post-shift
    window) avoids a degenerate near-zero "recovery time" from noise
    before performance has genuinely bottomed out.

    Parameters
    ----------
    reward : numpy.ndarray, shape (n_episodes,)
        Per-episode reward.
    matrix_history : sequence of (int, numpy.ndarray)
        Compact change-event history.
    shift_episode : int
        The episode the shift being measured took effect.
    baseline_window : int, default=50
        Episodes before ``shift_episode`` averaged for the baseline.
    rolling_window : int, default=50
        Width of the rolling mean applied to the post-shift series.
    target_fraction : float, default=0.5
        Fraction of the ``baseline - trough`` gap that counts as
        "recovered" (0.5 = classic half-life; other values give e.g.
        90%-recovery time at ``target_fraction=0.9``).
    p : numpy.ndarray, optional
        Type prior.

    Returns
    -------
    float
        Number of post-shift episodes (0-indexed from ``shift_episode``)
        until recovery, or ``float("inf")`` if recovery never occurs
        within the available post-shift data (including when there are
        fewer than ``rolling_window`` post-shift episodes to even form
        one rolling estimate). If the shift never caused a genuine dip
        (``baseline <= trough``, e.g. a rotation toward a better matrix),
        returns ``0.0`` -- already at or above target at the trough
        itself, by convention (see module docs for the "recovery"
        framing this metric assumes).

    Raises
    ------
    ValueError
        If ``shift_episode < baseline_window`` (not enough pre-shift data
        to form a baseline), or ``rolling_window <= 0``.
    """
    if rolling_window <= 0:
        raise ValueError(f"rolling_window must be > 0, got {rolling_window!r}")
    shift_episode = int(shift_episode)
    if shift_episode < baseline_window:
        raise ValueError(
            f"shift_episode={shift_episode} < baseline_window={baseline_window}; not enough "
            f"pre-shift episodes for a baseline"
        )
    normalized = time_varying_normalize(reward, matrix_history, p)
    baseline = float(np.nanmean(normalized[shift_episode - baseline_window : shift_episode]))

    post = normalized[shift_episode:]
    if post.shape[0] < rolling_window:
        return float("inf")

    rolling = pd.Series(post).rolling(window=rolling_window, min_periods=rolling_window).mean().to_numpy()
    valid_idx = np.flatnonzero(~np.isnan(rolling))
    if valid_idx.size == 0:
        return float("inf")

    trough_idx = valid_idx[int(np.argmin(rolling[valid_idx]))]
    trough = float(rolling[trough_idx])

    if baseline <= trough:
        return 0.0

    target = trough + target_fraction * (baseline - trough)
    for idx in valid_idx[valid_idx >= trough_idx]:
        if rolling[idx] >= target:
            return float(idx)
    return float("inf")
