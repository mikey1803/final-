"""Tests for the Task 6 horizon sweep (recovery/core/parallel.py,
recovery/metrics/prefix.py, recovery/analysis/crossover_empirical.py,
experiments/horizon_sweep.py, experiments/horizon_analysis.py).
"""

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.crossover_empirical import crossover_distribution, crossover_horizon
from recovery.classify.synthetic import PerfectClassifier
from recovery.core.episode import run_episodes
from recovery.core.parallel import RunConfig, RunStore, run_grid
from recovery.envs.canonical import BLIND_RULE, MATRIX
from recovery.envs.matrix_env import MatrixEnvironment
from recovery.metrics.prefix import late_stage_rate, prefix_view
from recovery.policies.adapters import StaticRulePolicy, ThompsonPolicy, UCB1Policy

HORIZONS = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]

SMOKE_TIMEOUT_SECONDS = 60.0
SUBPROCESS_TIMEOUT_SECONDS = 120.0


# ======================================================================
# 1. Prefix equivalence
# ======================================================================


def test_prefix_equivalence_static_policy():
    """A T_max=5000 trajectory's first 1000 rows must be byte-identical to
    a fresh 1000-episode run at the same seed -- this IS what makes
    prefix evaluation valid (see recovery/metrics/prefix.py)."""
    env_full = MatrixEnvironment(MATRIX)
    log_full = run_episodes(
        env_full, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=5000, seed=123
    )
    prefix = prefix_view(log_full, 1000)

    env_fresh = MatrixEnvironment(MATRIX)
    log_fresh = run_episodes(
        env_fresh, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=1000, seed=123
    )

    assert np.array_equal(prefix.true_type, log_fresh.true_type)
    assert np.array_equal(prefix.predicted_type, log_fresh.predicted_type)
    assert np.array_equal(prefix.action, log_fresh.action)
    assert np.array_equal(prefix.reward, log_fresh.reward)


def test_prefix_equivalence_stateful_policy():
    """Same equivalence, but for a STATEFUL (learning) policy -- Thompson
    Sampling only ever conditions on its OWN past (ctx, action, reward)
    history, which is identical for episodes 0..999 whether or not the
    run continues past episode 1000."""
    env_full = MatrixEnvironment(MATRIX)
    log_full = run_episodes(env_full, ThompsonPolicy(seed=11), PerfectClassifier(), n_episodes=5000, seed=456)
    prefix = prefix_view(log_full, 1000)

    env_fresh = MatrixEnvironment(MATRIX)
    log_fresh = run_episodes(env_fresh, ThompsonPolicy(seed=11), PerfectClassifier(), n_episodes=1000, seed=456)

    assert np.array_equal(prefix.true_type, log_fresh.true_type)
    assert np.array_equal(prefix.action, log_fresh.action)
    assert np.array_equal(prefix.reward, log_fresh.reward)


def test_prefix_view_raises_when_horizon_exceeds_log_length():
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(env, StaticRulePolicy(BLIND_RULE), PerfectClassifier(), n_episodes=100, seed=1)
    with pytest.raises(ValueError):
        prefix_view(log, 200)


# ======================================================================
# 2. Determinism under parallelism
# ======================================================================


def _tiny_grid(store_root: Path, n_cells_per_method: int = 2) -> list[RunConfig]:
    cells = []
    for method, factory in (
        ("static_blind", lambda: StaticRulePolicy(BLIND_RULE, name="static_blind")),
        ("thompson", None),  # built per-seed below (needs the seed closed over)
    ):
        for seed_idx in range(n_cells_per_method):
            seed = 7000 + seed_idx
            if method == "thompson":
                policy_factory = (lambda s=seed: ThompsonPolicy(seed=s, name="thompson"))
            else:
                policy_factory = factory
            cells.append(
                RunConfig(
                    method=method, env_id="canonical", matrix=MATRIX, seed=seed, n_episodes=500,
                    policy_factory=policy_factory, classifier_factory=lambda: PerfectClassifier(),
                )
            )
    return cells


def test_run_grid_deterministic_regardless_of_n_jobs(tmp_path):
    cells_a = _tiny_grid(tmp_path)
    cells_b = _tiny_grid(tmp_path)  # separate RunConfig objects, same content_key()s

    store_a = tmp_path / "store_n1"
    store_b = tmp_path / "store_n4"
    run_grid(cells_a, store_root=store_a, n_jobs=1, show_progress=False)
    run_grid(cells_b, store_root=store_b, n_jobs=4, show_progress=False)

    reader_a, reader_b = RunStore(store_a), RunStore(store_b)
    for ca, cb in zip(cells_a, cells_b):
        assert ca.content_key() == cb.content_key()
        df_a = reader_a.read(ca)
        df_b = reader_b.read(cb)
        assert np.array_equal(df_a["true_type"].to_numpy(), df_b["true_type"].to_numpy())
        assert np.array_equal(df_a["action"].to_numpy(), df_b["action"].to_numpy())
        assert np.array_equal(df_a["reward"].to_numpy(), df_b["reward"].to_numpy())


def test_run_grid_skips_cells_already_in_the_store(tmp_path):
    cells = _tiny_grid(tmp_path, n_cells_per_method=1)
    store_root = tmp_path / "store"
    paths_first = run_grid(cells, store_root=store_root, n_jobs=1, show_progress=False)
    mtimes_first = [p.stat().st_mtime_ns for p in paths_first]

    paths_second = run_grid(cells, store_root=store_root, n_jobs=1, show_progress=False)
    mtimes_second = [p.stat().st_mtime_ns for p in paths_second]

    assert paths_first == paths_second
    assert mtimes_first == mtimes_second  # files were never rewritten the second time


# ======================================================================
# 3-5. crossover_empirical
# ======================================================================


def test_crossover_horizon_returns_inf_when_never_crosses():
    learner = [0.1, 0.1, 0.1, 0.1]
    static = [0.5, 0.5, 0.5, 0.5]
    horizons = [100, 1000, 10_000, 100_000]
    assert crossover_horizon(learner, static, horizons) == math.inf


def test_crossover_horizon_finds_correct_t_for_synthetic_monotone_crossing():
    horizons = [10, 100]
    learner = [-1.0, 1.0]
    static = [0.0, 0.0]
    # log-linear interpolation: zero crossing of (learner - static) at the
    # geometric midpoint of the bracketing grid points (diff goes -1 -> +1
    # linearly in log(T), crossing zero exactly halfway through log-space).
    expected = math.sqrt(10 * 100)
    result = crossover_horizon(learner, static, horizons)
    assert result == pytest.approx(expected, rel=1e-9)


def test_crossover_horizon_immediate_crossing_at_first_point():
    horizons = [100, 1000]
    learner = [1.0, 1.0]
    static = [0.0, 0.0]
    assert crossover_horizon(learner, static, horizons) == 100.0


def test_crossover_horizon_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        crossover_horizon([1.0, 2.0], [1.0], [10, 100])


def test_crossover_distribution_reports_fraction_never_crossed_no_mean_over_inf():
    crossings = [500.0, math.inf, 1500.0, math.inf, math.inf]
    result = crossover_distribution(crossings)
    assert result["fraction_never_crossed"] == pytest.approx(0.6)
    assert result["median"] == pytest.approx(1000.0)  # median of [500, 1500] only
    assert math.isfinite(result["median"])


def test_crossover_distribution_all_never_crossed_reports_inf_not_nan():
    result = crossover_distribution([math.inf, math.inf])
    assert result["fraction_never_crossed"] == 1.0
    assert result["median"] == math.inf


# ======================================================================
# 6. Validation cell: a q=7/7 (oracle-equivalent) static rule is never
#    crossed by Thompson within 1e5 episodes.
# ======================================================================


def test_q7_of_7_static_rule_never_crossed_by_thompson_within_1e5():
    """``perfect_rule = MATRIX.argmax(axis=1)`` plays the best action for
    EVERY true type -- i.e. it achieves ``oracle_rate(MATRIX)`` exactly,
    the hard ceiling no policy (adaptive or not) can exceed in
    expectation (recovery.theory.analytic.oracle_rate's docstring).
    Thompson Sampling approaches but never exceeds this ceiling in
    expectation; empirically, for the seed pinned here, its late-stage
    rate never overtakes and stays ahead of the perfect rule's anywhere
    in the horizon grid."""
    perfect_rule = MATRIX.argmax(axis=1)
    seed = 0

    env_static = MatrixEnvironment(MATRIX)
    log_static = run_episodes(
        env_static, StaticRulePolicy(perfect_rule, name="perfect_static"), PerfectClassifier(),
        n_episodes=100_000, seed=seed,
    )
    env_thompson = MatrixEnvironment(MATRIX)
    log_thompson = run_episodes(
        env_thompson, ThompsonPolicy(seed=seed, name="thompson"), PerfectClassifier(),
        n_episodes=100_000, seed=seed,
    )

    static_curve = [late_stage_rate(prefix_view(log_static, T)) for T in HORIZONS]
    thompson_curve = [late_stage_rate(prefix_view(log_thompson, T)) for T in HORIZONS]

    crossing = crossover_horizon(thompson_curve, static_curve, HORIZONS)
    assert crossing == math.inf


# ======================================================================
# 7. Smoke run of the full experiment
# ======================================================================

_WARMED_UP = False


def _warm_up_horizon_sweep(tmp_path: Path) -> None:
    """One untimed --smoke run to absorb loky worker-pool cold-start cost
    (same rationale as tests/test_stationary_experiments.py's _warm_up)."""
    global _WARMED_UP
    if _WARMED_UP:
        return
    warmup_output = tmp_path / "warmup.parquet"
    warmup_store = tmp_path / "warmup_store"
    subprocess.run(
        [
            sys.executable, str(_EXPERIMENTS_DIR / "horizon_sweep.py"), "--smoke",
            "--output", str(warmup_output), "--store-root", str(warmup_store),
        ],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, cwd=str(_BACKEND_DIR),
    )
    _WARMED_UP = True


def test_horizon_sweep_smoke_end_to_end(tmp_path):
    _warm_up_horizon_sweep(tmp_path)

    output_path = tmp_path / "horizon_sweep_smoke.parquet"
    store_root = tmp_path / "smoke_store"
    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, str(_EXPERIMENTS_DIR / "horizon_sweep.py"), "--smoke",
            "--output", str(output_path), "--store-root", str(store_root),
        ],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, cwd=str(_BACKEND_DIR),
    )
    elapsed = time.time() - t0

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert elapsed < SMOKE_TIMEOUT_SECONDS, f"--smoke took {elapsed:.2f}s (must be < 60s)"

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) > 0
    expected_columns = {
        "method", "env_id", "classifier_id", "seed", "T", "late_stage_rate", "normalized",
        "cumulative_regret", "normalized_cumulative_regret", "time_to_parity", "worst_type_rate",
    }
    assert set(df.columns) == expected_columns
    assert set(df["method"].unique()) == {"static_blind", "thompson", "ucb1"}


def test_horizon_analysis_smoke_end_to_end(tmp_path):
    _warm_up_horizon_sweep(tmp_path)

    sweep_output = tmp_path / "sweep_for_analysis.parquet"
    store_root = tmp_path / "analysis_store"
    sweep_result = subprocess.run(
        [
            sys.executable, str(_EXPERIMENTS_DIR / "horizon_sweep.py"), "--smoke",
            "--output", str(sweep_output), "--store-root", str(store_root),
        ],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, cwd=str(_BACKEND_DIR),
    )
    assert sweep_result.returncode == 0, sweep_result.stderr

    crossover_output = tmp_path / "crossover.parquet"
    analysis_result = subprocess.run(
        [
            sys.executable, str(_EXPERIMENTS_DIR / "horizon_analysis.py"),
            "--input", str(sweep_output), "--output", str(crossover_output),
        ],
        capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, cwd=str(_BACKEND_DIR),
    )
    assert analysis_result.returncode == 0, analysis_result.stderr
    assert crossover_output.exists()

    df = pd.read_parquet(crossover_output)
    expected_columns = {
        "method", "env_id", "classifier_id", "criterion", "n_seeds",
        "median", "iqr_low", "iqr_high", "min", "max", "fraction_never_crossed",
    }
    assert set(df.columns) == expected_columns
    assert set(df["criterion"].unique()) == {"asymptotic", "cumulative"}
    # static_blind is the baseline (excluded from crossover rows) --
    # nothing is ever crossed against itself.
    assert "static_blind" not in set(df["method"].unique())
    assert "Horizon crossover summary" in analysis_result.stdout
