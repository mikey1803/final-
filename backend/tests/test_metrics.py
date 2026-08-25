"""Tests for the Task 5 metrics layer (``recovery/metrics/``).

These tests DO use the kernel (``recovery.core.episode.run_episodes``) and
native policies/classifiers to produce real ``EpisodeLog`` instances to
feed into the metrics under test -- that restriction (numpy/pandas/
analytic only) applies to ``recovery/metrics/`` itself, not to its tests.
Some tests use a minimal ``_FakeLog`` (just the ``.reward``/``.action``/
``.true_type`` arrays a metrics function actually reads) to isolate metric
logic from the kernel entirely.
"""

import inspect
import math

import numpy as np
import pytest

from recovery.classify.synthetic import PerfectClassifier
from recovery.core.episode import run_episodes
from recovery.envs.canonical import BLIND_RULE, MATRIX, STRATEGY_NAMES
from recovery.envs.matrix_env import MatrixEnvironment
from recovery.policies.adapters import OraclePolicy, StaticRulePolicy, UniformRandomPolicy
from recovery.theory.analytic import oracle_rate

from recovery.metrics import bootstrap as bootstrap_module
from recovery.metrics import cost as cost_module
from recovery.metrics import parity as parity_module
from recovery.metrics import pertype as pertype_module
from recovery.metrics import regret as regret_module
from recovery.metrics import summary as summary_module
from recovery.metrics.bootstrap import bootstrap_ci, cliffs_delta
from recovery.metrics.cost import STRATEGY_COSTS, cost_normalized_reward
from recovery.metrics.parity import summarize_parity, time_to_parity
from recovery.metrics.pertype import per_type_rates
from recovery.metrics.regret import (
    cumulative_regret,
    final_cumulative_regret,
    instantaneous_regret,
    normalized_cumulative_regret,
)
from recovery.metrics.summary import summarize


class _FakeLog:
    """Minimal stand-in exposing only the arrays a metrics function reads."""

    def __init__(self, reward=None, action=None, true_type=None):
        if reward is not None:
            self.reward = np.asarray(reward, dtype=np.float64)
        if action is not None:
            self.action = np.asarray(action)
        if true_type is not None:
            self.true_type = np.asarray(true_type)


# ---------------------------------------------------------------------------
# Architecture constraints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [bootstrap_module, cost_module, parity_module, pertype_module, regret_module, summary_module],
)
def test_metrics_modules_do_not_import_kernel_envs_policies_or_experiments(module):
    """Checks actual `import`/`from ... import` lines only -- the modules'
    own docstrings legitimately DISCUSS why recovery.core/envs/etc. are
    off-limits, which would false-positive a plain substring search."""
    import re

    source = inspect.getsource(module)
    import_pattern = re.compile(r"^\s*(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import\b)")
    imported_modules = [
        m.group(1) or m.group(2)
        for line in source.splitlines()
        if (m := import_pattern.match(line))
    ]
    for forbidden in ("recovery.core", "recovery.envs", "recovery.policies", "recovery.classify", "experiments"):
        offending = [mod for mod in imported_modules if mod.startswith(forbidden)]
        assert not offending, f"{module.__name__} has a forbidden import: {offending}"


def test_strategy_costs_aligned_with_canonical_strategy_names():
    """STRATEGY_COSTS can't import recovery.envs.canonical (see cost.py's
    module docstring), so this test is the cross-check that its manually
    pinned ordering actually matches."""
    assert len(STRATEGY_COSTS) == len(STRATEGY_NAMES)
    expected = dict(zip(STRATEGY_NAMES, [1.0, 3.0, 2.0, 2.5, 1.5, 2.0]))
    for i, name in enumerate(STRATEGY_NAMES):
        assert STRATEGY_COSTS[i] == expected[name]


# ---------------------------------------------------------------------------
# Regret
# ---------------------------------------------------------------------------


def test_oracle_instantaneous_regret_is_exactly_zero():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(env, OraclePolicy(MATRIX), PerfectClassifier(), n_episodes=2000, seed=1)
    regret = instantaneous_regret(log, MATRIX)
    assert np.all(regret == 0.0)


def test_worst_arm_policy_mean_regret_equals_oracle_minus_row_minimum():
    worst_rule = MATRIX.argmin(axis=1)
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(
        env, StaticRulePolicy(worst_rule), PerfectClassifier(), n_episodes=20_000, seed=2
    )
    mean_regret = float(instantaneous_regret(log, MATRIX).mean())
    expected = oracle_rate(MATRIX) - float(MATRIX.min(axis=1).mean())
    assert mean_regret == pytest.approx(expected, abs=0.02)


def test_cumulative_regret_is_monotonically_non_decreasing():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(
        env, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=1000, seed=3
    )
    cum = cumulative_regret(log, MATRIX)
    assert np.all(np.diff(cum) >= 0.0)


def test_uniform_random_normalized_cumulative_regret_near_one():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(
        env, UniformRandomPolicy(seed=7), PerfectClassifier(), n_episodes=100_000, seed=4
    )
    ncr = normalized_cumulative_regret(log, MATRIX)
    assert abs(ncr - 1.0) < 0.05


def test_final_cumulative_regret_is_last_entry_of_cumulative_regret():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(
        env, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=500, seed=5
    )
    assert final_cumulative_regret(log, MATRIX) == pytest.approx(cumulative_regret(log, MATRIX)[-1])


def test_instantaneous_regret_raises_on_empty_log():
    empty = _FakeLog(reward=[], action=[], true_type=[])
    with pytest.raises(ValueError):
        instantaneous_regret(empty, MATRIX)


def test_normalized_cumulative_regret_raises_on_degenerate_matrix():
    degenerate = np.full_like(MATRIX, 0.5)  # every strategy equally good for every type
    log = _FakeLog(reward=[1, 0, 1], action=[0, 1, 2], true_type=[0, 1, 2])
    with pytest.raises(ValueError):
        normalized_cumulative_regret(log, degenerate)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_all_retry_policy_cost_normalized_reward_equals_success_rate():
    reward = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    log = _FakeLog(reward=reward, action=np.zeros(5, dtype=int))  # 0 == retry, cost 1.0
    assert cost_normalized_reward(log) == pytest.approx(float(reward.mean()))


def test_cost_normalized_reward_returns_zero_not_nan_when_all_rewards_zero():
    log = _FakeLog(reward=[0.0, 0.0, 0.0], action=[0, 1, 2])
    result = cost_normalized_reward(log)
    assert result == 0.0
    assert not math.isnan(result)


def test_total_cost_raises_on_empty_log():
    empty = _FakeLog(reward=[], action=[])
    with pytest.raises(ValueError):
        cost_module.total_cost(empty)


def test_cost_aware_oracle_differs_from_analytic_oracle_when_costs_vary():
    """Sanity check the two oracles are genuinely different concepts (not
    an accidental alias) -- see cost.py's cost_aware_oracle_rate docstring."""
    rate_cost_blind = oracle_rate(MATRIX)
    rate_cost_aware = cost_module.cost_aware_oracle_rate(MATRIX, STRATEGY_COSTS)
    # Cost-aware oracle picks best success-per-cost, not best raw success,
    # so its raw success rate is <= the cost-blind oracle's.
    assert rate_cost_aware <= rate_cost_blind + 1e-9


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_time_to_parity_returns_inf_when_parity_never_occurs():
    learner = _FakeLog(reward=np.zeros(200))
    baseline = _FakeLog(reward=np.ones(200))
    result = time_to_parity(learner, 42, baseline, 42, stability_window=50)
    assert result == math.inf


def test_time_to_parity_finds_crossing_when_learner_overtakes_and_stays_ahead():
    n = 300
    baseline_reward = np.full(n, 0.3)
    learner_reward = np.full(n, 0.3)
    learner_reward[50:] = 0.9  # learner jumps ahead after episode 50 and stays there
    learner = _FakeLog(reward=learner_reward)
    baseline = _FakeLog(reward=baseline_reward)
    result = time_to_parity(learner, 1, baseline, 1, stability_window=50)
    assert result != math.inf
    assert result >= 50


def test_time_to_parity_mismatched_seeds_raise_value_error():
    learner = _FakeLog(reward=np.ones(100))
    baseline = _FakeLog(reward=np.zeros(100))
    with pytest.raises(ValueError):
        time_to_parity(learner, 1, baseline, 2)


def test_time_to_parity_mismatched_lengths_raise_value_error():
    learner = _FakeLog(reward=np.ones(100))
    baseline = _FakeLog(reward=np.zeros(50))
    with pytest.raises(ValueError):
        time_to_parity(learner, 1, baseline, 1)


def test_summarize_parity_reports_fraction_never_crossed_without_discarding():
    times = [10.0, math.inf, 20.0, math.inf]
    result = summarize_parity(times, horizon=1000)
    assert result["fraction_never_crossed"] == pytest.approx(0.5)
    assert result["median_finite"] == pytest.approx(15.0)
    assert result["horizon"] == 1000


def test_summarize_parity_all_infinite_reports_inf_median_not_nan():
    result = summarize_parity([math.inf, math.inf], horizon=500)
    assert result["fraction_never_crossed"] == 1.0
    assert result["median_finite"] == math.inf


# ---------------------------------------------------------------------------
# Per-type
# ---------------------------------------------------------------------------


def test_per_type_rates_unobserved_type_is_nan():
    log = _FakeLog(reward=[1.0, 0.0], true_type=[0, 0])
    rates = per_type_rates(log, n_types=3)
    assert rates[0] == pytest.approx(0.5)
    assert math.isnan(rates[1])
    assert math.isnan(rates[2])


def test_worst_type_rate_ignores_nan_unobserved_types():
    log = _FakeLog(reward=[1.0, 0.0, 1.0], true_type=[0, 1, 1])
    worst = pertype_module.worst_type_rate(log, n_types=5)
    assert worst == pytest.approx(0.5)  # type 1's rate; type 0 is 1.0; types 2-4 unobserved


# ---------------------------------------------------------------------------
# Bootstrap and effect sizes
# ---------------------------------------------------------------------------


def test_bootstrap_ci_bernoulli_contains_true_rate_and_has_expected_width():
    rng = np.random.default_rng(0)
    samples = rng.binomial(1, 0.7, size=10_000).astype(np.float64)
    lo, hi = bootstrap_ci(samples, n_resamples=10_000, seed=1)
    assert lo <= 0.7 <= hi
    analytic_se = math.sqrt(0.7 * 0.3 / len(samples))
    analytic_width = 2 * 1.96 * analytic_se
    observed_width = hi - lo
    assert observed_width == pytest.approx(analytic_width, rel=0.2)


def test_bootstrap_ci_single_observation_raises_value_error():
    with pytest.raises(ValueError):
        bootstrap_ci([0.5])


def test_bootstrap_ci_is_fast_for_10000_resamples_over_50_observations():
    import time

    values = np.random.default_rng(0).random(50)
    t0 = time.perf_counter()
    bootstrap_ci(values, n_resamples=10_000)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"bootstrap_ci took {elapsed * 1000:.1f}ms (must be < 100ms)"


def test_cliffs_delta_completely_separated_samples():
    x = [10.0, 11.0, 12.0]
    y = [1.0, 2.0, 3.0]
    assert cliffs_delta(x, y) == 1.0


def test_cliffs_delta_identical_samples_is_zero():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cliffs_delta(x, x) == pytest.approx(0.0)


def test_magnitude_thresholds():
    assert bootstrap_module.magnitude(0.1) == "negligible"
    assert bootstrap_module.magnitude(0.2) == "small"
    assert bootstrap_module.magnitude(0.4) == "medium"
    assert bootstrap_module.magnitude(0.9) == "large"


def test_cohens_d_requires_at_least_two_observations_each():
    with pytest.raises(ValueError):
        bootstrap_module.cohens_d([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


def test_summarize_returns_every_implemented_metric():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(
        env, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=500, seed=9
    )
    result = summarize(log, MATRIX)
    expected_keys = {
        "late_stage_rate",
        "normalized_score",
        "instantaneous_regret",
        "cumulative_regret",
        "final_cumulative_regret",
        "normalized_cumulative_regret",
        "total_cost",
        "cost_normalized_reward",
        "cost_aware_oracle_rate",
        "per_type_rates",
        "worst_type_rate",
        "bootstrap_ci",
        "cliffs_delta",
        "cohens_d",
        "effect_magnitude",
    }
    assert set(result.keys()) == expected_keys
    assert result["cliffs_delta"] is None  # no baseline_values given
    assert result["cohens_d"] is None
    assert result["effect_magnitude"] is None


def test_summarize_with_baseline_values_populates_effect_sizes():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(
        env, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=500, seed=10
    )
    baseline = np.zeros(100)  # a deliberately much-worse "baseline" sample
    result = summarize(log, MATRIX, baseline_values=baseline)
    assert result["cliffs_delta"] is not None
    assert result["cohens_d"] is not None
    assert result["effect_magnitude"] in {"negligible", "small", "medium", "large"}


def test_summarize_performs_no_file_io():
    source = inspect.getsource(summary_module)
    for forbidden in ("open(", "to_csv", "to_parquet", "Path(", "print("):
        assert forbidden not in source


def test_summarize_raises_on_empty_log():
    empty = _FakeLog(reward=[])
    with pytest.raises(ValueError):
        summarize(empty, MATRIX)
