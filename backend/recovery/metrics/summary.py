"""``summarize``: the one reusable summary dictionary future experiments
should build from a completed ``EpisodeLog``. No file I/O -- the caller
serializes the returned dict if/when it wants to (out of scope here)."""

from typing import Optional, Sequence

import numpy as np

from recovery.metrics.bootstrap import bootstrap_ci, cliffs_delta, cohens_d, magnitude
from recovery.metrics.cost import (
    STRATEGY_COSTS,
    cost_aware_oracle_rate,
    cost_normalized_reward,
    total_cost,
)
from recovery.metrics.pertype import per_type_rates, worst_type_rate
from recovery.metrics.regret import (
    cumulative_regret,
    final_cumulative_regret,
    instantaneous_regret,
    normalized_cumulative_regret,
)
from recovery.theory.analytic import normalize

__all__ = ["summarize"]

DEFAULT_LATE_STAGE_FRAC = 0.2


def summarize(
    log,
    P: np.ndarray,
    n_types: Optional[int] = None,
    costs: np.ndarray = STRATEGY_COSTS,
    p: Optional[np.ndarray] = None,
    late_stage_frac: float = DEFAULT_LATE_STAGE_FRAC,
    baseline_values: Optional[Sequence[float]] = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict:
    """Build the standard metrics summary for one completed log.

    Parameters
    ----------
    log : EpisodeLog-like
        Exposes ``.reward``, ``.true_type``, ``.action`` (shape
        ``(n_episodes,)``).
    P : numpy.ndarray, shape (N, K)
        The ground-truth success matrix.
    n_types : int, optional
        Number of true types, for the per-type breakdown. Defaults to
        ``P.shape[0]``.
    costs : numpy.ndarray, shape (K,), default=STRATEGY_COSTS
        Per-strategy execution cost.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    late_stage_frac : float, default=0.2
        Fraction of the log's tail used for ``late_stage_rate`` and the
        bootstrap CI (matches the paper's "late-stage success" window and
        ``recovery.analysis.tables.LATE_STAGE_FRAC``).
    baseline_values : sequence of float, optional
        A comparison sample (e.g. another policy's late-stage per-episode
        rewards, or its per-seed rates) for the effect-size keys. If
        omitted, ``cliffs_delta``/``cohens_d``/``effect_magnitude`` are
        ``None`` in the returned dict (the keys are always present).
    bootstrap_resamples, bootstrap_seed : int
        Passed through to ``bootstrap_ci`` over the late-stage reward
        window (an intra-run bootstrap -- Monte Carlo uncertainty from
        this ONE run's finite episode count, not a cross-seed bootstrap;
        see docs/statistics.md).

    Returns
    -------
    dict
        Every metric this package implements, as one flat dictionary (see
        module code for the exact key set). No file I/O is performed.

    Raises
    ------
    ValueError
        If the log is empty, or if any underlying metric's own edge-case
        validation fails (e.g. ``normalized_cumulative_regret`` on a
        degenerate ``P``).
    """
    reward = np.asarray(log.reward, dtype=np.float64)
    n_episodes = reward.size
    if n_episodes == 0:
        raise ValueError("log has zero episodes; summarize requires at least one")
    n_types_resolved = P.shape[0] if n_types is None else n_types

    start = int((1.0 - late_stage_frac) * n_episodes)
    late_reward = reward[start:]
    late_stage_rate = float(late_reward.mean()) if late_reward.size > 0 else float("nan")
    normalized_score = normalize(late_stage_rate, P, p)

    if late_reward.size >= 2:
        ci = bootstrap_ci(late_reward, n_resamples=bootstrap_resamples, seed=bootstrap_seed)
    else:
        ci = (float("nan"), float("nan"))

    if baseline_values is not None:
        delta = cliffs_delta(late_reward, baseline_values)
        effect_cohens_d = cohens_d(late_reward, baseline_values)
        effect_magnitude = magnitude(delta)
    else:
        delta = None
        effect_cohens_d = None
        effect_magnitude = None

    return {
        "late_stage_rate": late_stage_rate,
        "normalized_score": normalized_score,
        "instantaneous_regret": instantaneous_regret(log, P),
        "cumulative_regret": cumulative_regret(log, P),
        "final_cumulative_regret": final_cumulative_regret(log, P),
        "normalized_cumulative_regret": normalized_cumulative_regret(log, P),
        "total_cost": total_cost(log, costs),
        "cost_normalized_reward": cost_normalized_reward(log, costs),
        "cost_aware_oracle_rate": cost_aware_oracle_rate(P, costs, p),
        "per_type_rates": per_type_rates(log, n_types_resolved),
        "worst_type_rate": worst_type_rate(log, n_types_resolved),
        "bootstrap_ci": ci,
        "cliffs_delta": delta,
        "cohens_d": effect_cohens_d,
        "effect_magnitude": effect_magnitude,
    }
