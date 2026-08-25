"""Cost-aware metrics: Table II's execution costs applied to completed logs.

``recovery.envs.canonical`` defines ``STRATEGY_NAMES`` in the canonical
order, but this package may not import ``recovery.envs`` (see the package
docstring: metrics/ imports only numpy, pandas,
``recovery.theory.analytic``). ``STRATEGY_COSTS`` below is therefore a
manually pinned table, aligned BY POSITION to that ordering
(``["retry", "replan", "switch_tool", "re_ground", "verify", "clarify"]``)
and cross-checked against it in ``tests/test_metrics.py`` (tests have no
such import restriction), not derived by importing it.
"""

from typing import Optional

import numpy as np

__all__ = [
    "STRATEGY_COSTS",
    "total_cost",
    "cost_normalized_reward",
    "cost_aware_oracle_rate",
    "cost_aware_ceiling",
    "cost_aware_floor",
]

# Position-aligned with recovery.envs.canonical.STRATEGY_NAMES:
#   0 retry        1.0
#   1 replan       3.0
#   2 switch_tool  2.0
#   3 re_ground    2.5
#   4 verify       1.5
#   5 clarify      2.0
STRATEGY_COSTS: np.ndarray = np.array([1.0, 3.0, 2.0, 2.5, 1.5, 2.0], dtype=np.float64)


def _prior(n_types: int, p: Optional[np.ndarray]) -> np.ndarray:
    if p is None:
        return np.full(n_types, 1.0 / n_types, dtype=np.float64)
    prior = np.asarray(p, dtype=np.float64)
    if prior.shape != (n_types,):
        raise ValueError(f"p must have shape ({n_types},), got {prior.shape}")
    return prior


def _validate_P_costs(P: np.ndarray, costs: np.ndarray) -> None:
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D, got shape {P.shape}")
    n_strats = P.shape[1]
    if costs.shape != (n_strats,):
        raise ValueError(f"costs must have shape ({n_strats},), got {costs.shape}")


def total_cost(log, costs: np.ndarray = STRATEGY_COSTS) -> float:
    """Sum of ``costs[action]`` over every episode in the log."""
    action = np.asarray(log.action)
    if action.size == 0:
        raise ValueError("log has zero episodes; total_cost requires at least one")
    return float(costs[action].sum())


def cost_normalized_reward(log, costs: np.ndarray = STRATEGY_COSTS) -> float:
    """``successes / total_cost``.

    Returns 0.0 (never NaN) when total successes are zero, regardless of
    total cost.
    """
    reward = np.asarray(log.reward, dtype=np.float64)
    if reward.size == 0:
        raise ValueError("log has zero episodes; cost_normalized_reward requires at least one")
    successes = float(reward.sum())
    if successes == 0.0:
        return 0.0
    cost = total_cost(log, costs)
    if cost <= 0.0:
        raise ValueError(f"total cost must be positive to compute cost_normalized_reward, got {cost!r}")
    return successes / cost


def cost_aware_oracle_rate(
    P: np.ndarray, costs: np.ndarray = STRATEGY_COSTS, p: Optional[np.ndarray] = None
) -> float:
    """Expected success rate of the per-type ``argmax_k(P[t,k]/costs[k])`` policy.

    This is NOT the same oracle as ``recovery.theory.analytic.oracle_rate``
    (which maximizes raw success probability, ignoring cost) -- picking the
    cheapest-per-unit-success action can trade away some success
    probability. The two oracles must never be mixed in the same
    normalization: ``recovery.theory.analytic.normalize`` uses the
    cost-blind oracle_rate/random_rate scale; ``cost_aware_ceiling``/
    ``cost_aware_floor`` below define a SEPARATE cost-aware scale for
    normalizing ``cost_normalized_reward``.
    """
    _validate_P_costs(P, costs)
    n_types = P.shape[0]
    prior = _prior(n_types, p)
    best_actions = np.argmax(P / costs[None, :], axis=1)
    achieved = P[np.arange(n_types), best_actions]
    return float(prior @ achieved)


def _cost_aware_extreme_rate(
    P: np.ndarray, costs: np.ndarray, p: Optional[np.ndarray], pick: str
) -> float:
    """Shared implementation for ``cost_aware_ceiling``/``cost_aware_floor``.

    Both are a RATIO OF EXPECTATIONS (expected success / expected cost) for
    the per-type best-or-worst success-per-unit-cost policy -- this matches
    what ``cost_normalized_reward`` estimates from a finite log in the
    large-``n_episodes`` limit (mean(reward)/mean(cost) -> E[reward]/E[cost]
    by the law of large numbers), not a mean of per-episode ratios.
    """
    _validate_P_costs(P, costs)
    n_types = P.shape[0]
    prior = _prior(n_types, p)
    per_unit = P / costs[None, :]
    actions = np.argmax(per_unit, axis=1) if pick == "max" else np.argmin(per_unit, axis=1)
    success_rate = float(prior @ P[np.arange(n_types), actions])
    expected_cost = float(prior @ costs[actions])
    if expected_cost <= 0.0:
        raise ValueError(f"expected cost must be positive, got {expected_cost!r}")
    return success_rate / expected_cost


def cost_aware_ceiling(
    P: np.ndarray, costs: np.ndarray = STRATEGY_COSTS, p: Optional[np.ndarray] = None
) -> float:
    """Analytic upper bound for ``cost_normalized_reward``: the expected
    success-per-unit-cost of the ``argmax_k(P[t,k]/costs[k])`` policy (the
    same policy family as ``cost_aware_oracle_rate``)."""
    return _cost_aware_extreme_rate(P, costs, p, pick="max")


def cost_aware_floor(
    P: np.ndarray, costs: np.ndarray = STRATEGY_COSTS, p: Optional[np.ndarray] = None
) -> float:
    """Analytic lower bound for ``cost_normalized_reward``: the expected
    success-per-unit-cost of the ``argmin_k(P[t,k]/costs[k])`` policy (the
    worst success-per-unit-cost choice for every type)."""
    return _cost_aware_extreme_rate(P, costs, p, pick="min")
