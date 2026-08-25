"""Per-type performance metrics: results broken out by TRUE failure type.

A single mean success rate (or mean regret) can hide a policy that is
excellent on six types and catastrophic on a seventh -- the aggregate
number looks fine while one failure mode goes unaddressed. These
functions surface that by reporting per-type breakdowns and the worst one.

``recovery.core.log.EpisodeLog`` already has a ``per_type_rate()`` method
with equivalent per-type-mean-reward logic, but this package may not
import ``recovery.core`` (see the package docstring), so
``per_type_rates`` below is a from-scratch, numpy-only reimplementation
against ``log.true_type``/``log.reward`` directly -- a metrics-layer
computation over already-completed observations, not a duplication of any
simulation logic.
"""

import numpy as np

from recovery.metrics.regret import instantaneous_regret

__all__ = ["per_type_rates", "worst_type_rate", "per_type_regret"]


def per_type_rates(log, n_types: int) -> np.ndarray:
    """Mean reward broken down by TRUE type, shape ``(n_types,)``.

    Types never observed as a true type in this log get ``nan``.
    """
    true_type = np.asarray(log.true_type)
    reward = np.asarray(log.reward, dtype=np.float64)
    if true_type.size == 0:
        raise ValueError("log has zero episodes; per_type_rates requires at least one")
    rates = np.full(n_types, np.nan, dtype=np.float64)
    for t in range(n_types):
        mask = true_type == t
        if np.any(mask):
            rates[t] = float(reward[mask].mean())
    return rates


def worst_type_rate(log, n_types: int) -> float:
    """Lowest per-type success rate among observed types (unobserved types
    -- ``nan`` in ``per_type_rates`` -- are ignored, not treated as 0)."""
    rates = per_type_rates(log, n_types)
    observed = rates[~np.isnan(rates)]
    if observed.size == 0:
        raise ValueError("no observed types in this log; cannot compute worst_type_rate")
    return float(observed.min())


def per_type_regret(log, P: np.ndarray) -> np.ndarray:
    """Mean instantaneous regret broken down by TRUE type, shape ``(P.shape[0],)``.

    Same rationale as the module docstring: mean overall regret can hide a
    single type whose regret is much larger than the rest.
    """
    n_types = P.shape[0]
    true_type = np.asarray(log.true_type)
    if true_type.size == 0:
        raise ValueError("log has zero episodes; per_type_regret requires at least one")
    regret = instantaneous_regret(log, P)
    rates = np.full(n_types, np.nan, dtype=np.float64)
    for t in range(n_types):
        mask = true_type == t
        if np.any(mask):
            rates[t] = float(regret[mask].mean())
    return rates
