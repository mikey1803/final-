"""Reusable research metrics layer (Task 5).

Simulation generates observations; this package interprets them. Every
function here is pure and deterministic: it consumes a completed
``recovery.core.log.EpisodeLog`` (or the raw ``numpy`` arrays it exposes --
``true_type``, ``action``, ``reward``, all shape ``(n_episodes,)``) plus a
success matrix ``P``, and returns numbers. Nothing in this package
simulates, samples rewards, updates policies, classifies failures, modifies
environments, does file I/O, or prints.

This package may import only ``numpy``, ``pandas``, and
``recovery.theory.analytic`` -- never ``recovery.core``, ``recovery.envs``,
``recovery.policies``, ``recovery.classify``, or anything under
``experiments/``. Functions therefore type ``log`` structurally (duck
typing: any object exposing the right arrays works) rather than importing
``EpisodeLog`` to check against it.
"""

from recovery.metrics.bootstrap import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    magnitude,
    paired_bootstrap_diff,
)
from recovery.metrics.cost import (
    STRATEGY_COSTS,
    cost_aware_ceiling,
    cost_aware_floor,
    cost_aware_oracle_rate,
    cost_normalized_reward,
    total_cost,
)
from recovery.metrics.drift_metrics import (
    adaptation_half_life,
    matrix_segments,
    post_shift_window_score,
    recovery_delta,
    time_varying_normalize,
)
from recovery.metrics.drift_regret import (
    cumulative_drift_regret,
    instantaneous_drift_regret,
    normalized_drift_regret,
)
from recovery.metrics.parity import summarize_parity, time_to_parity
from recovery.metrics.pertype import per_type_rates, per_type_regret, worst_type_rate
from recovery.metrics.prefix import PrefixLog, late_stage_rate, prefix_view
from recovery.metrics.regret import (
    cumulative_regret,
    final_cumulative_regret,
    instantaneous_regret,
    normalized_cumulative_regret,
)
from recovery.metrics.summary import summarize

__all__ = [
    "bootstrap_ci",
    "cliffs_delta",
    "cohens_d",
    "magnitude",
    "paired_bootstrap_diff",
    "STRATEGY_COSTS",
    "cost_aware_ceiling",
    "cost_aware_floor",
    "cost_aware_oracle_rate",
    "cost_normalized_reward",
    "total_cost",
    "summarize_parity",
    "time_to_parity",
    "per_type_rates",
    "per_type_regret",
    "worst_type_rate",
    "PrefixLog",
    "late_stage_rate",
    "prefix_view",
    "cumulative_regret",
    "final_cumulative_regret",
    "instantaneous_regret",
    "normalized_cumulative_regret",
    "summarize",
    "time_varying_normalize",
    "matrix_segments",
    "post_shift_window_score",
    "recovery_delta",
    "adaptation_half_life",
    "instantaneous_drift_regret",
    "cumulative_drift_regret",
    "normalized_drift_regret",
]
