"""Regret metrics: the realized cost of not knowing the true failure type.

Regret is always measured against the ANALYTIC oracle (``P.max(axis=1)``,
the same quantity ``recovery.theory.analytic.oracle_rate`` is built from),
never against the best action a policy happened to observe in one run --
an empirical-best comparison would be biased by which actions that
specific policy tried, and would not be comparable across runs/policies.
"""

import numpy as np

__all__ = [
    "instantaneous_regret",
    "cumulative_regret",
    "final_cumulative_regret",
    "normalized_cumulative_regret",
]


def _true_type_and_action(log) -> tuple[np.ndarray, np.ndarray]:
    true_type = np.asarray(log.true_type)
    action = np.asarray(log.action)
    if true_type.size == 0:
        raise ValueError("log has zero episodes; regret metrics require at least one")
    return true_type, action


def instantaneous_regret(log, P: np.ndarray) -> np.ndarray:
    """Per-episode regret, shape ``(n_episodes,)``.

    ``regret[i] = max(P[true_type[i]]) - P[true_type[i], action[i]]``.
    Always >= 0, since ``max(P[t])`` upper-bounds every entry in row ``t``.
    """
    true_type, action = _true_type_and_action(log)
    best = P.max(axis=1)
    return best[true_type] - P[true_type, action]


def cumulative_regret(log, P: np.ndarray) -> np.ndarray:
    """Running cumulative regret, shape ``(n_episodes,)``.

    ``cumulative_regret[i]`` is the total regret incurred through episode
    ``i`` inclusive. Monotonically non-decreasing, since every term of
    ``instantaneous_regret`` is >= 0.
    """
    return np.cumsum(instantaneous_regret(log, P))


def final_cumulative_regret(log, P: np.ndarray) -> float:
    """Total regret over the whole log (the last entry of ``cumulative_regret``)."""
    return float(cumulative_regret(log, P)[-1])


def normalized_cumulative_regret(log, P: np.ndarray) -> float:
    """``final_cumulative_regret`` normalized so 0.0 = oracle, 1.0 = random.

    The denominator is the regret a UNIFORM RANDOM policy would incur over
    the SAME realized true-type sequence (``log.true_type``) -- not a
    fresh simulated random-policy run (metrics must never simulate; see
    the package docstring). A random policy's expected reward for true
    type ``t`` is the closed form ``mean_k P[t, k]``, so the denominator is
    ``sum_i (max(P[true_type[i]]) - mean_k P[true_type[i], k])``: the
    analytic random-policy regret evaluated deterministically over the
    realized type sequence, giving numerator and denominator identical
    conditions to compare under.

    Raises
    ------
    ValueError
        If the log is empty, or if the denominator is ~0 (< 1e-12) --
        every strategy equally good for every type actually realized in
        this log, so the [oracle, random] scale is degenerate.
    """
    true_type, _action = _true_type_and_action(log)
    best = P.max(axis=1)
    random_expected = P.mean(axis=1)
    denom = float(np.sum(best[true_type] - random_expected[true_type]))
    if abs(denom) < 1e-12:
        raise ValueError(
            f"random-policy regret over this log's realized type sequence is "
            f"~0 ({denom!r}); cannot normalize (every strategy is equally good "
            f"for every type this log actually realized)"
        )
    return final_cumulative_regret(log, P) / denom
