"""Time-varying regret for drifted runs (Task 8A, Correction 2).

``recovery.metrics.regret`` (Task 5, unmodified) always measures regret
against ONE fixed matrix -- correct for stationary runs, but WRONG for a
drifted run if that one matrix is (as Task 8's first implementation did)
the FINAL matrix: an episode played early in a run, under the ORIGINAL
matrix, would be scored against a ceiling/floor that did not exist yet
when that episode happened. Regret at episode ``e`` must always be
computed against the oracle of the matrix ACTIVE at episode ``e``.

This module reuses ``recovery.metrics.drift_metrics.matrix_segments``
(the same "which matrix was active when" primitive
``time_varying_normalize`` is built on) rather than re-deriving segment
boundaries a second way, and reuses nothing from
``recovery.metrics.regret`` beyond the underlying mathematical
definition it documents (per-episode regret = the active oracle's best
value minus the value of the action actually played) -- the two modules
compute genuinely different things (one global matrix vs. one matrix per
segment), so this is not a duplicate implementation of the same metric,
it is the drift-aware GENERALIZATION of it. For a single-segment
(stationary) ``matrix_history``, every function here is bit-identical to
its ``recovery.metrics.regret`` counterpart (see
``tests/test_drift.py::test_drift_regret_equals_ordinary_regret_when_stationary``).

Same package-level import constraint as ``drift_metrics.py`` (see
``recovery/metrics/__init__.py``): only ``numpy``, ``pandas``, and other
``recovery.metrics``/``recovery.theory.analytic`` modules.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from recovery.metrics.drift_metrics import matrix_segments

__all__ = [
    "instantaneous_drift_regret",
    "cumulative_drift_regret",
    "normalized_drift_regret",
]


def _true_type_and_action(log) -> Tuple[np.ndarray, np.ndarray]:
    true_type = np.asarray(log.true_type)
    action = np.asarray(log.action)
    if true_type.size == 0:
        raise ValueError("log has zero episodes; drift regret metrics require at least one")
    return true_type, action


def instantaneous_drift_regret(
    log,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
) -> np.ndarray:
    """Per-episode regret, shape ``(n_episodes,)``, against the matrix
    ACTIVE at each episode -- NEVER the final matrix.

    ``regret[i] = max_k P_i[true_type[i], k] - P_i[true_type[i], action[i]]``
    where ``P_i`` is whichever ``matrix_history`` entry was active at
    episode ``i``. Always ``>= 0``, for the same reason
    ``recovery.metrics.regret.instantaneous_regret`` is: ``max_k P_i[t,
    k]`` upper-bounds every entry in that matrix's row ``t``.

    Parameters
    ----------
    log : EpisodeLog-like
        Exposes ``.true_type``/``.action`` (shape ``(n_episodes,)``).
    matrix_history : sequence of (int, numpy.ndarray)
        Compact change-event history, e.g.
        ``recovery.envs.drift.DriftingEnvironment.matrix_history()``.

    Returns
    -------
    numpy.ndarray, shape (n_episodes,)

    Raises
    ------
    ValueError
        If the log is empty, or ``matrix_history`` has no entry before
        ``len(log)``.
    """
    true_type, action = _true_type_and_action(log)
    n = true_type.shape[0]
    regret = np.empty(n, dtype=np.float64)
    for start, end, matrix in matrix_segments(matrix_history, n):
        best = matrix.max(axis=1)
        seg_type = true_type[start:end]
        seg_action = action[start:end]
        regret[start:end] = best[seg_type] - matrix[seg_type, seg_action]
    return regret


def cumulative_drift_regret(
    log,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
) -> np.ndarray:
    """Running cumulative time-varying regret, shape ``(n_episodes,)``.

    ``cumulative_drift_regret[i]`` is the total time-varying regret
    incurred through episode ``i`` inclusive. Monotonically
    non-decreasing, since every term of ``instantaneous_drift_regret``
    is ``>= 0``.

    IMPORTANT: changing only a FUTURE segment of ``matrix_history``
    (e.g. a later shift's target matrix) cannot change any value of this
    array at or before that shift's episode -- each entry only ever
    reads the matrix segment active at its own index (see
    ``tests/test_drift.py::test_changing_future_matrix_does_not_alter_past_regret``).
    """
    return np.cumsum(instantaneous_drift_regret(log, matrix_history))


def normalized_drift_regret(
    log,
    matrix_history: Sequence[Tuple[int, np.ndarray]],
    p: Optional[np.ndarray] = None,
) -> float:
    """Final cumulative time-varying regret, normalized so 0.0 = oracle,
    1.0 = random -- the time-varying generalization of
    ``recovery.metrics.regret.normalized_cumulative_regret``.

    The denominator is likewise the TIME-VARYING random-policy regret:
    for each segment, the random-policy's expected per-type regret
    (``max_k P[t, k] - mean_k P[t, k]``) evaluated against that
    segment's OWN matrix and its own realized true-type sequence, summed
    across segments -- never a single global denominator computed from
    one matrix.

    Parameters
    ----------
    log : EpisodeLog-like
        Exposes ``.true_type``/``.action``.
    matrix_history : sequence of (int, numpy.ndarray)
        Compact change-event history.
    p : numpy.ndarray, optional
        Unused by the numerator/denominator formula below (both are
        computed directly from the REALIZED true-type sequence, exactly
        like ``normalized_cumulative_regret``'s own denominator) --
        accepted only for signature symmetry with
        ``time_varying_normalize``/other functions in this package that
        DO take an explicit type prior.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the log is empty, ``matrix_history`` has no entry before
        ``len(log)``, or the total denominator is ~0 (< 1e-12 -- every
        strategy equally good for every type actually realized, across
        every segment).
    """
    del p  # see docstring
    true_type, _action = _true_type_and_action(log)
    n = true_type.shape[0]
    final = float(cumulative_drift_regret(log, matrix_history)[-1])

    denom = 0.0
    for start, end, matrix in matrix_segments(matrix_history, n):
        best = matrix.max(axis=1)
        random_expected = matrix.mean(axis=1)
        seg_type = true_type[start:end]
        denom += float(np.sum(best[seg_type] - random_expected[seg_type]))

    if abs(denom) < 1e-12:
        raise ValueError(
            f"time-varying random-policy regret over this log's realized type sequence is "
            f"~0 ({denom!r}); cannot normalize"
        )
    return final / denom
