"""Prefix evaluation: treat any horizon ``T`` within a run's length as its
own completed run, without re-simulating.

The kernel (``recovery.core.episode.run_episodes``) draws every RNG
stream sequentially, starting from state derived only from ``seed`` --
never from the planned ``n_episodes``. The first ``T`` rows of a
length-``T_max`` run are therefore IDENTICAL to a fresh length-``T`` run
at the same seed (see ``tests/test_horizon.py``'s prefix-equivalence
test and docs/experiments/horizon.md, "Prefix evaluation"). This lets a
horizon sweep evaluate every ``T`` in a grid from ONE ``T_max``
trajectory per (method, seed) cell instead of one simulation per
(method, seed, T) -- and makes every horizon's curve PAIRED across the
same random draws, with no cross-horizon noise.
"""

from typing import Optional

import numpy as np

__all__ = ["PrefixLog", "prefix_view", "late_stage_rate"]


class PrefixLog:
    """Read-only view of the first ``n_episodes`` rows of a longer log.

    Duck-type compatible with every function in ``recovery.metrics`` that
    reads ``.true_type``/``.action``/``.reward`` (and, where present,
    ``.predicted_type``) -- nothing downstream needs to know this is a
    slice rather than a freshly simulated ``EpisodeLog``.
    """

    __slots__ = ("true_type", "predicted_type", "action", "reward", "n_episodes")

    def __init__(
        self,
        true_type: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        predicted_type: Optional[np.ndarray] = None,
    ) -> None:
        self.true_type = true_type
        self.predicted_type = predicted_type
        self.action = action
        self.reward = reward
        self.n_episodes = int(reward.shape[0])


def prefix_view(log, T: int) -> PrefixLog:
    """Return the first ``T`` episodes of ``log`` as a ``PrefixLog``.

    Parameters
    ----------
    log : EpisodeLog-like
        Exposes ``.true_type``/``.action``/``.reward`` (and optionally
        ``.predicted_type``) as equal-length arrays or array-likes (a
        ``pandas.DataFrame``'s columns work too).
    T : int
        Horizon to slice to, in ``[1, len(log)]``.

    Returns
    -------
    PrefixLog
        A view over the first ``T`` rows -- element-for-element identical
        to what a fresh ``T``-episode run at the same seed would produce
        (this is a slice, not a recomputation).

    Raises
    ------
    ValueError
        If ``T < 1``, or ``T`` exceeds ``log``'s own length. A horizon
        grid point larger than what was actually simulated must fail
        loudly, never silently truncate to whatever's available.
    """
    reward = np.asarray(log.reward)
    n_available = reward.shape[0]
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if T > n_available:
        raise ValueError(
            f"requested horizon T={T} exceeds this log's length ({n_available} "
            f"episodes); prefix_view never silently truncates"
        )
    predicted_type = getattr(log, "predicted_type", None)
    return PrefixLog(
        true_type=np.asarray(log.true_type)[:T],
        action=np.asarray(log.action)[:T],
        reward=np.asarray(log.reward)[:T],
        predicted_type=np.asarray(predicted_type)[:T] if predicted_type is not None else None,
    )


def late_stage_rate(log, late_stage_frac: float = 0.2) -> float:
    """Mean reward over the final ``late_stage_frac`` fraction of ``log``.

    Pair with ``prefix_view(full_log, T)`` to get "late-stage rate of the
    first T episodes" -- what a horizon sweep needs per horizon. Passing
    a log's full length instead gives the late-stage rate of the whole
    run (what ``recovery.analysis.tables`` uses for the stationary
    tables).

    Raises
    ------
    ValueError
        If ``late_stage_frac`` is outside ``(0, 1]``, or ``log`` is empty.
    """
    if not (0.0 < late_stage_frac <= 1.0):
        raise ValueError(f"late_stage_frac must be in (0, 1], got {late_stage_frac}")
    reward = np.asarray(log.reward, dtype=np.float64)
    n = reward.shape[0]
    if n == 0:
        raise ValueError("log has zero episodes; late_stage_rate requires at least one")
    start = int((1.0 - late_stage_frac) * n)
    window = reward[start:]
    return float(window.mean()) if window.size > 0 else float("nan")
