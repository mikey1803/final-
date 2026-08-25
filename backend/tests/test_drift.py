"""Tests for the Task 8 drift taxonomy (``recovery.envs.drift``) and its
time-varying normalization layer (``recovery.metrics.drift_metrics``),
plus the ``Context.drift_signal`` kernel wiring.

No dependency on ``hypothesis`` (not installed in this environment, and
not used anywhere else in this repository) -- "property-based sanity
tests" below are randomized loops over many independently-drawn
configurations instead, following this file's own stress-test
conventions rather than introducing a new test dependency for one file.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.episode import run_episodes  # noqa: E402
from recovery.core.protocols import Context  # noqa: E402
from recovery.envs.canonical import MATRIX  # noqa: E402
from recovery.envs.drift import (  # noqa: E402
    AnnouncedDrift,
    DriftingEnvironment,
    GradualRotationDrift,
    MonotoneDecayDrift,
    PartialReorderDrift,
    PermutationDrift,
)
from recovery.envs.matrix_env import MatrixEnvironment  # noqa: E402
from recovery.metrics.drift_metrics import (  # noqa: E402
    adaptation_half_life,
    post_shift_window_score,
    recovery_delta,
    time_varying_normalize,
)
from recovery.policies.adapters import (  # noqa: E402
    StaticRulePolicy,
    ThompsonPolicy,
    UniformRandomPolicy,
)
from recovery.metrics.regret import (  # noqa: E402
    cumulative_regret,
    instantaneous_regret,
    normalized_cumulative_regret,
)
from recovery.metrics.drift_regret import (  # noqa: E402
    cumulative_drift_regret,
    instantaneous_drift_regret,
    normalized_drift_regret,
)
from recovery.theory.analytic import normalize as analytic_normalize  # noqa: E402

N_TYPES, N_STRATS = MATRIX.shape


def _random_matrix(rng: np.random.Generator, n_types: int, n_strats: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n_types, n_strats))


def _run_to_episode(env, upto: int) -> None:
    """Advance ``env`` through ``step(0), ..., step(upto - 1)`` -- the
    same sequence ``run_episodes`` would issue before episode ``upto``'s
    ``sample_reward`` call."""
    for i in range(upto):
        env.step(i)


# ==========================================================================
# 1. Matrix shape preserved
# ==========================================================================


def test_matrix_shape_preserved_across_all_five_models():
    rng = np.random.default_rng(1)
    models = [
        PermutationDrift(MATRIX, schedule=[10, 30], affected_rows=[0, 1], rng=rng),
        MonotoneDecayDrift(MATRIX, strategy=0, rows=[0, 1], half_life=20, schedule=[10, 30], floor=0.05),
        PartialReorderDrift(MATRIX, rows=[0, 1], schedule=[10, 30]),
        GradualRotationDrift(MATRIX, MATRIX[::-1].copy(), start_episode=10, window=40),
    ]
    for env in models:
        _run_to_episode(env, 60)
        assert env.current_matrix().shape == MATRIX.shape
        assert env.base_matrix().shape == MATRIX.shape
        for event in env.matrix_history():
            assert event.matrix.shape == MATRIX.shape


# ==========================================================================
# 2. Values always remain inside [0, 1]
# ==========================================================================


def test_values_always_in_unit_interval_stress_100_random_matrices():
    """Randomized stress test over >= 100 random matrices/configurations."""
    rng = np.random.default_rng(2)
    for trial in range(120):
        n_types = int(rng.integers(2, 6))
        n_strats = int(rng.integers(2, 6))
        base = _random_matrix(rng, n_types, n_strats)
        model_kind = trial % 4
        schedule = sorted(int(x) for x in rng.integers(0, 100, size=int(rng.integers(1, 4))))
        rows = sorted(set(int(x) for x in rng.integers(0, n_types, size=int(rng.integers(1, n_types + 1)))))

        if model_kind == 0:
            env = PermutationDrift(base, schedule=schedule, affected_rows=rows, rng=rng)
        elif model_kind == 1:
            strategy = int(rng.integers(0, n_strats))
            floor = float(min(base[r, strategy] for r in rows)) * rng.uniform(0.0, 1.0)
            env = MonotoneDecayDrift(
                base, strategy=strategy, rows=rows, half_life=float(rng.uniform(5, 50)),
                schedule=schedule, floor=floor,
            )
        elif model_kind == 2:
            env = PartialReorderDrift(base, rows=rows, schedule=schedule)
        else:
            target = _random_matrix(rng, n_types, n_strats)
            env = GradualRotationDrift(base, target, start_episode=int(rng.integers(0, 20)), window=int(rng.integers(1, 60)))

        for i in range(120):
            env.step(i)
            cur = env.current_matrix()
            assert np.all(cur >= 0.0) and np.all(cur <= 1.0), f"trial {trial} kind {model_kind}: out-of-range value"
        for event in env.matrix_history():
            assert np.all(event.matrix >= 0.0) and np.all(event.matrix <= 1.0)


# ==========================================================================
# 3. Base matrix never changes
# ==========================================================================


def test_base_matrix_never_mutates():
    rng = np.random.default_rng(3)
    env = PermutationDrift(MATRIX, schedule=[5, 15, 25], affected_rows=[0, 1, 2], rng=rng)
    original = MATRIX.copy()

    # (a) the caller's own MATRIX array is untouched by construction/use.
    _run_to_episode(env, 40)
    assert np.array_equal(MATRIX, original)

    # (b) base_matrix() returns a snapshot; mutating it must not affect
    # the environment's internal state.
    snapshot = env.base_matrix()
    snapshot[:] = -1.0
    assert np.array_equal(env.base_matrix(), original)

    # (c) the first matrix_history() entry is also the untouched base.
    assert np.array_equal(env.matrix_history()[0].matrix, original)


# ==========================================================================
# 4. Gradual interpolation endpoints correct
# ==========================================================================


def test_gradual_interpolation_endpoints_correct():
    target = MATRIX[::-1].copy()
    env = GradualRotationDrift(MATRIX, target, start_episode=50, window=100)

    _run_to_episode(env, 50)
    assert np.allclose(env.current_matrix(), MATRIX)

    _run_to_episode(env, 100)  # halfway through the window (episode 100 -> alpha=0.5)
    midpoint_expected = MATRIX + (target - MATRIX) * 0.5
    assert np.allclose(env.current_matrix(), midpoint_expected)

    _run_to_episode(env, 151)  # 50 + 100 + 1 -> fully past the window
    assert np.allclose(env.current_matrix(), target)

    # Before start_episode, the matrix is exactly the base -- no drift yet.
    env2 = GradualRotationDrift(MATRIX, target, start_episode=50, window=100)
    _run_to_episode(env2, 10)
    assert np.array_equal(env2.current_matrix(), MATRIX)


# ==========================================================================
# 5. Partial reorder changes argmax only
# ==========================================================================


def test_partial_reorder_changes_argmax_only():
    env = PartialReorderDrift(MATRIX, rows=[0, 2, 4], schedule=[10])
    old = MATRIX.copy()
    old_argmax = old.argmax(axis=1)

    _run_to_episode(env, 15)
    new = env.current_matrix()
    new_argmax = new.argmax(axis=1)

    for t in range(N_TYPES):
        if t in (0, 2, 4):
            assert sorted(new[t]) == pytest.approx(sorted(old[t]))
            assert new_argmax[t] != old_argmax[t]
            # the new argmax must be the OLD second-best strategy.
            second_best = np.argsort(old[t], kind="stable")[::-1][1]
            assert new_argmax[t] == second_best
        else:
            assert np.array_equal(new[t], old[t])
            assert new_argmax[t] == old_argmax[t]


def test_partial_reorder_requires_at_least_two_strategies():
    with pytest.raises(ValueError):
        PartialReorderDrift(np.array([[0.5]]), rows=[0], schedule=[10])


# ==========================================================================
# 6/7. Monotone decay: always monotonic, respects floor
# ==========================================================================


def test_monotone_decay_always_monotonic_and_respects_floor():
    rng = np.random.default_rng(4)
    for trial in range(20):
        strategy = int(rng.integers(0, N_STRATS))
        rows = sorted(set(int(x) for x in rng.integers(0, N_TYPES, size=3)))
        half_life = float(rng.uniform(5, 80))
        base_val = float(min(MATRIX[r, strategy] for r in rows))
        floor = base_val * float(rng.uniform(0.0, 0.9))
        schedule = list(range(0, 300, int(rng.integers(10, 40))))

        env = MonotoneDecayDrift(MATRIX, strategy=strategy, rows=rows, half_life=half_life, schedule=schedule, floor=floor)
        values = []
        for i in range(300):
            env.step(i)
            values.append(env.current_matrix()[rows[0], strategy])

        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9, f"trial {trial}: decay increased at step {i}"
        assert all(v >= floor - 1e-9 for v in values), f"trial {trial}: value fell below floor"


def test_monotone_decay_floor_above_base_value_raises():
    with pytest.raises(ValueError):
        MonotoneDecayDrift(MATRIX, strategy=0, rows=[0], half_life=10, schedule=[5], floor=0.99)


# ==========================================================================
# 8. Announcement only on shift episodes
# ==========================================================================


def test_announcement_only_on_shift_episodes():
    rng = np.random.default_rng(5)
    inner = PermutationDrift(MATRIX, schedule=[10, 25, 40], affected_rows=[0, 1], rng=rng)
    env = AnnouncedDrift(inner)

    observed_true_at = []
    for i in range(50):
        if env.drift_signal:
            observed_true_at.append(i)
        env.step(i)

    assert observed_true_at == [10, 25, 40]


def test_multiple_announcements_repeated_shift():
    rng = np.random.default_rng(6)
    schedule = [20, 40, 60, 80]
    inner = PermutationDrift(MATRIX, schedule=schedule, affected_rows=[0], rng=rng)
    env = AnnouncedDrift(inner)

    signal_episodes = []
    for i in range(100):
        if env.drift_signal:
            signal_episodes.append(i)
        env.step(i)

    assert signal_episodes == schedule
    assert env.shift_signal == env.drift_signal  # alias always agrees


def test_edge_case_shift_at_episode_zero_is_announced_immediately():
    rng = np.random.default_rng(7)
    inner = PermutationDrift(MATRIX, schedule=[0], affected_rows=[0, 1], rng=rng)
    env = AnnouncedDrift(inner)
    # No step() has been called yet -- this is exactly what the kernel
    # reads for episode 0's Context.
    assert env.drift_signal is True
    assert not np.array_equal(env.current_matrix(), MATRIX)  # already shifted


def test_edge_case_overlapping_shifts_same_episode():
    rng = np.random.default_rng(8)
    inner = PermutationDrift(MATRIX, schedule=[30, 30], affected_rows=[0], rng=rng)
    env = AnnouncedDrift(inner)

    signal_episodes = []
    for i in range(50):
        if env.drift_signal:
            signal_episodes.append(i)
        env.step(i)

    # Two schedule entries due at the same episode collapse into ONE
    # announced episode (a bool cannot represent event count).
    assert signal_episodes == [30]
    episodes_in_history = [e.episode for e in inner.matrix_history()]
    assert episodes_in_history == [0, 30, 30]  # both events still individually recorded


# ==========================================================================
# 9. Same seed -> same history
# ==========================================================================


def test_same_seed_produces_same_history():
    def build():
        rng = np.random.default_rng(123)
        env = PermutationDrift(MATRIX, schedule=[10, 20, 30], affected_rows=[0, 1, 2], rng=rng)
        _run_to_episode(env, 40)
        return env

    a, b = build(), build()
    hist_a, hist_b = a.matrix_history(), b.matrix_history()
    assert len(hist_a) == len(hist_b)
    for ea, eb in zip(hist_a, hist_b):
        assert ea.episode == eb.episode
        assert np.array_equal(ea.matrix, eb.matrix)


def test_deterministic_behavior_identical_seeds_full_kernel_run():
    def build_and_run():
        env = AnnouncedDrift(
            PermutationDrift(MATRIX, schedule=[50], affected_rows=[0, 1], rng=np.random.default_rng(999))
        )
        return run_episodes(env, ThompsonPolicy(seed=11), PerfectClassifier(), n_episodes=200, seed=42)

    log_a = build_and_run()
    log_b = build_and_run()
    assert np.array_equal(log_a.action, log_b.action)
    assert np.array_equal(log_a.reward, log_b.reward)
    assert np.array_equal(log_a.true_type, log_b.true_type)


# ==========================================================================
# 10. Stationary normalization identical to Task 5
# ==========================================================================


def test_stationary_normalization_identical_to_task5():
    rng = np.random.default_rng(9)
    reward = rng.uniform(0.0, 1.0, size=500)
    history = [(0, MATRIX)]

    got = time_varying_normalize(reward, history)
    expected = np.array([analytic_normalize(r, MATRIX) for r in reward])
    assert np.allclose(got, expected)


def test_time_varying_normalize_uses_active_matrix_not_preshift_endpoints():
    """The core scientific requirement: post-shift episodes must be
    normalized against the POST-shift matrix's own oracle/random rates,
    never the pre-shift matrix's."""
    rng = np.random.default_rng(10)
    base = MATRIX.copy()
    target = np.clip(base + 0.25, 0.0, 1.0)  # uniformly easier matrix
    history = [(0, base), (100, target)]
    reward = np.full(200, 0.9)  # a fixed raw rate, pre- and post-shift

    normalized = time_varying_normalize(reward, history)
    pre_naive = analytic_normalize(0.9, base)
    post_naive_wrong = analytic_normalize(0.9, base)  # what NOT using time-varying norm would give
    post_correct = analytic_normalize(0.9, target)

    assert np.allclose(normalized[:100], pre_naive)
    assert np.allclose(normalized[100:], post_correct)
    assert not np.allclose(normalized[100:], post_naive_wrong)  # would be wrong if endpoints weren't updated


# ==========================================================================
# 11. Half-life returns infinity when recovery never occurs
# ==========================================================================


def test_half_life_returns_infinity_when_recovery_never_occurs():
    n = 300
    shift_episode = 150
    reward = np.empty(n, dtype=np.float64)
    reward[:shift_episode] = 1.0  # perfect pre-shift
    reward[shift_episode:] = 0.0  # collapses to zero and NEVER recovers
    history = [(0, MATRIX), (shift_episode, MATRIX)]  # matrix identity irrelevant to this synthetic reward

    hl = adaptation_half_life(reward, history, shift_episode=shift_episode, baseline_window=50, rolling_window=50)
    assert hl == float("inf")


def test_half_life_returns_infinity_with_insufficient_post_shift_data():
    n = 160
    shift_episode = 150  # only 10 post-shift episodes, fewer than rolling_window
    reward = np.ones(n, dtype=np.float64)
    history = [(0, MATRIX)]
    hl = adaptation_half_life(reward, history, shift_episode=shift_episode, baseline_window=50, rolling_window=50)
    assert hl == float("inf")


def test_half_life_finite_when_recovery_occurs():
    n = 400
    shift_episode = 100
    reward = np.empty(n, dtype=np.float64)
    reward[:shift_episode] = 1.0
    # drop, then linearly recover back to 1.0 over the next 100 episodes, then stay recovered.
    ramp = np.linspace(0.0, 1.0, 100)
    reward[shift_episode:shift_episode + 100] = ramp
    reward[shift_episode + 100:] = 1.0
    history = [(0, MATRIX)]

    hl = adaptation_half_life(reward, history, shift_episode=shift_episode, baseline_window=50, rolling_window=20)
    assert np.isfinite(hl)
    assert 0 <= hl <= 200


# ==========================================================================
# 12. Permutation reproduces original paper setup
# ==========================================================================


def test_permutation_reproduces_original_paper_setup():
    """Cross-checks against ``experiments.prior_grid.DriftEnvironment``
    (Task 7's local, single-shift/repeated-shift drift class) -- both
    implement the SAME mathematical drift mechanism (permute affected
    rows' strategy assignment) and the SAME kernel-timing convention
    (``episode + 1 >= shift_episode``). Their permutation VALUES differ
    (independent RNG draws), so this checks mathematical/timing
    equivalence, not bit-identical output.
    """
    sys.path.insert(0, str(_EXPERIMENTS_DIR.parent))
    from experiments.prior_grid import DriftEnvironment as LegacyDriftEnvironment

    shift_episode = 250
    affected = [1, 3]

    legacy = LegacyDriftEnvironment(MATRIX, shift_episodes=[shift_episode], affected_types=affected, shift_seed=1)
    new = PermutationDrift(MATRIX, schedule=[shift_episode], affected_rows=affected, rng=np.random.default_rng(1))

    for env in (legacy, new):
        for i in range(shift_episode - 1):
            env.step(i)
        assert np.array_equal(env.success_matrix(), MATRIX), "pre-shift matrix must equal the base matrix"

    for i in range(shift_episode - 1, shift_episode + 10):
        legacy.step(i)
        new.step(i)

    legacy_post, new_post = legacy.success_matrix(), new.success_matrix()
    for env_name, post in (("legacy", legacy_post), ("new", new_post)):
        for t in range(N_TYPES):
            if t in affected:
                assert sorted(post[t]) == pytest.approx(sorted(MATRIX[t])), f"{env_name}: row {t} value-set changed"
            else:
                assert np.array_equal(post[t], MATRIX[t]), f"{env_name}: unaffected row {t} changed"


# ==========================================================================
# Matrix replay reconstructs every drift exactly
# ==========================================================================


def test_matrix_history_replay_reconstructs_every_drift_exactly():
    rng = np.random.default_rng(11)
    schedule = [5, 15, 33, 70]

    def build():
        return PermutationDrift(MATRIX, schedule=schedule, affected_rows=[0, 2, 4], rng=np.random.default_rng(42))

    reference = build()
    _run_to_episode(reference, 100)
    history = reference.matrix_history()

    probe_episodes = [0, 4, 5, 14, 15, 32, 33, 69, 70, 99]
    for probe in probe_episodes:
        fresh = build()
        _run_to_episode(fresh, probe)
        expected = fresh.current_matrix()

        # reconstruct "matrix active at `probe`" from the compact history alone
        active = None
        for event in history:
            if event.episode <= probe:
                active = event.matrix
            else:
                break
        assert active is not None
        assert np.array_equal(active, expected), f"replay mismatch at episode {probe}"


# ==========================================================================
# No NaN/Inf ever appears
# ==========================================================================


def test_no_nan_or_inf_values_ever_appear():
    rng = np.random.default_rng(12)
    for trial in range(100):
        n_types = int(rng.integers(2, 6))
        n_strats = int(rng.integers(2, 6))
        base = _random_matrix(rng, n_types, n_strats)
        schedule = sorted(int(x) for x in rng.integers(0, 80, size=3))

        kind = trial % 4
        if kind == 0:
            env = PermutationDrift(base, schedule=schedule, affected_rows=list(range(n_types)), rng=rng)
        elif kind == 1:
            strategy = int(rng.integers(0, n_strats))
            floor = float(base[:, strategy].min()) * 0.1
            env = MonotoneDecayDrift(base, strategy=strategy, rows=list(range(n_types)), half_life=15, schedule=schedule, floor=floor)
        elif kind == 2:
            env = PartialReorderDrift(base, rows=list(range(n_types)), schedule=schedule)
        else:
            target = _random_matrix(rng, n_types, n_strats)
            env = GradualRotationDrift(base, target, start_episode=5, window=40)

        for i in range(90):
            env.step(i)
            cur = env.current_matrix()
            assert np.all(np.isfinite(cur)), f"trial {trial} kind {kind}: non-finite value at step {i}"


# ==========================================================================
# Edge cases: magnitude == n_types, window longer than run
# ==========================================================================


def test_edge_case_magnitude_equals_n_types_permutation():
    rng = np.random.default_rng(13)
    env = PermutationDrift(MATRIX, schedule=[10], affected_rows=list(range(N_TYPES)), rng=rng)
    _run_to_episode(env, 20)
    post = env.current_matrix()
    for t in range(N_TYPES):
        assert sorted(post[t]) == pytest.approx(sorted(MATRIX[t]))


def test_edge_case_window_longer_than_run():
    target = MATRIX[::-1].copy()
    env = GradualRotationDrift(MATRIX, target, start_episode=0, window=10_000)
    for i in range(500):
        env.step(i)
    cur = env.current_matrix()
    assert np.all(cur >= 0.0) and np.all(cur <= 1.0)
    assert not np.allclose(cur, MATRIX)
    assert not np.allclose(cur, target)  # nowhere near reaching target by episode 500
    # Task 8A, Correction 1: history now gains one entry per ramp episode
    # (exact normalization), so it grows with episodes actually simulated
    # inside the ramp -- still O(window), never O(n_episodes) beyond what
    # was actually run, and the target is still never reached.
    assert len(env.matrix_history()) == 1 + 500
    assert env.matrix_history()[-1].episode == 500


def test_gradual_rotation_validates_shape_start_and_window():
    with pytest.raises(ValueError):
        GradualRotationDrift(MATRIX, np.zeros((2, 2)), start_episode=0, window=10)
    with pytest.raises(ValueError):
        GradualRotationDrift(MATRIX, MATRIX, start_episode=-1, window=10)
    with pytest.raises(ValueError):
        GradualRotationDrift(MATRIX, MATRIX, start_episode=0, window=0)


def test_permutation_out_of_range_row_raises():
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError):
        PermutationDrift(MATRIX, schedule=[10], affected_rows=[N_TYPES], rng=rng)


def test_drifting_environment_base_class_requires_subclass_next_matrix():
    with pytest.raises(NotImplementedError):
        DriftingEnvironment(MATRIX, schedule=[5]).step(4)


# ==========================================================================
# Context.drift_signal kernel wiring
# ==========================================================================


def test_context_drift_signal_defaults_false_and_is_backward_compatible():
    ctx = Context(predicted_type=0)
    assert ctx.drift_signal is False
    ctx2 = Context(predicted_type=1, episode=5)
    assert ctx2.drift_signal is False


def test_stationary_environment_never_sets_drift_signal():
    recorded = []

    class RecordingPolicy:
        name = "recorder"

        def select(self, ctx, rng):
            recorded.append(ctx.drift_signal)
            return 0

        def update(self, ctx, action, reward):
            pass

        def reset(self):
            pass

    env = MatrixEnvironment(MATRIX)
    run_episodes(env, RecordingPolicy(), PerfectClassifier(), n_episodes=100, seed=1)
    assert all(s is False for s in recorded)


def test_kernel_context_receives_drift_signal_from_announced_drift():
    rng = np.random.default_rng(15)
    env = AnnouncedDrift(PermutationDrift(MATRIX, schedule=[30, 60], affected_rows=[0, 1], rng=rng))

    recorded = []

    class RecordingPolicy:
        name = "recorder"

        def select(self, ctx, rng):
            recorded.append((ctx.episode, ctx.drift_signal))
            return 0

        def update(self, ctx, action, reward):
            pass

        def reset(self):
            pass

    run_episodes(env, RecordingPolicy(), PerfectClassifier(), n_episodes=100, seed=1)
    true_episodes = [ep for ep, sig in recorded if sig]
    assert true_episodes == [30, 60]


# ==========================================================================
# AnnouncedDrift is a pure decorator: no other behavior changes
# ==========================================================================


def test_announced_drift_does_not_change_sampling_behavior():
    rng_a = np.random.default_rng(16)
    rng_b = np.random.default_rng(16)
    inner = PermutationDrift(MATRIX, schedule=[20], affected_rows=[0, 1], rng=rng_a)
    wrapped = AnnouncedDrift(PermutationDrift(MATRIX, schedule=[20], affected_rows=[0, 1], rng=rng_b))

    reward_rng_1 = np.random.default_rng(0)
    reward_rng_2 = np.random.default_rng(0)
    for i in range(40):
        for t in range(N_TYPES):
            for a in range(N_STRATS):
                r1 = inner.sample_reward(t, a, np.random.default_rng(i * 100 + t * 10 + a))
                r2 = wrapped.sample_reward(t, a, np.random.default_rng(i * 100 + t * 10 + a))
                assert r1 == r2
        inner.step(i)
        wrapped.step(i)
    assert np.array_equal(inner.current_matrix(), wrapped.current_matrix())


# ==========================================================================
# drift_metrics: post_shift_window_score / recovery_delta basic sanity
# ==========================================================================


def test_post_shift_window_score_and_recovery_delta_basic_sanity():
    n = 300
    shift_episode = 150
    reward = np.empty(n)
    reward[:shift_episode] = 0.8
    reward[shift_episode:] = 0.3
    history = [(0, MATRIX)]  # single matrix -> both windows normalize against the same P

    score = post_shift_window_score(reward, history, shift_episode=shift_episode, window=50)
    expected_score = analytic_normalize(0.3, MATRIX)
    assert score == pytest.approx(expected_score, abs=1e-9)

    delta = recovery_delta(reward, history, shift_episode=shift_episode, pre_window=50, post_window=50)
    expected_delta = analytic_normalize(0.8, MATRIX) - analytic_normalize(0.3, MATRIX)
    assert delta == pytest.approx(expected_delta, abs=1e-9)
    assert delta > 0  # performance dropped after the (synthetic) shift


# ==========================================================================
# Smoke experiment (drift_taxonomy.py --smoke, < 90s)
# ==========================================================================


@pytest.mark.slow
def test_drift_taxonomy_smoke_end_to_end(tmp_path):
    output_path = tmp_path / "drift_taxonomy.smoke.parquet"
    result = subprocess.run(
        [sys.executable, str(_EXPERIMENTS_DIR / "drift_taxonomy.py"), "--smoke", "--output", str(output_path)],
        capture_output=True, text=True, timeout=180, cwd=str(_BACKEND_DIR),
    )
    assert result.returncode == 0, f"drift_taxonomy.py --smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) > 0


# ==========================================================================
# Static policies report N/A for adaptation statistics (never fabricated)
# ==========================================================================


def test_static_policies_report_na_for_adaptation_statistics(tmp_path):
    """End-to-end: run a tiny drift-taxonomy grid + its analysis, and
    confirm ``static_rule``/``uniform_random`` (``NON_ADAPTIVE_METHODS``)
    get the literal text "N/A" for recovery_delta/adaptation_half_life in
    the analysis CSVs -- never a fabricated number, and never silently
    dropped either (the rows must still be present)."""
    sys.path.insert(0, str(_EXPERIMENTS_DIR))
    from drift_taxonomy import NON_ADAPTIVE_METHODS, build_grid, run_drift_grid
    from drift_analysis import adaptation_half_life_summary, per_cell_metrics, recovery_delta_with_ci

    assert {"static_rule", "uniform_random"} <= NON_ADAPTIVE_METHODS

    store_root = tmp_path / "store"
    cells = build_grid(
        drift_types=["permutation"], frequencies=[20], magnitudes=[2],
        methods=["static_rule", "uniform_random", "stock_thompson"],
        n_seeds=2, base_seed=555, n_episodes=200,
    )
    metadata_df = run_drift_grid(cells, store_root=store_root, show_progress=False)
    cell_df = per_cell_metrics(metadata_df, store_root)

    assert cell_df.loc[cell_df["method"] == "static_rule", "is_adaptive"].eq(False).all()
    assert cell_df.loc[cell_df["method"] == "stock_thompson", "is_adaptive"].eq(True).all()
    assert cell_df.loc[cell_df["method"] == "static_rule", "recovery_delta"].isna().all()

    delta_summary = recovery_delta_with_ci(cell_df)
    hl_summary = adaptation_half_life_summary(cell_df)
    for df in (delta_summary, hl_summary):
        static_rows = df[df["method"] == "static_rule"]
        assert len(static_rows) > 0, "static_rule rows must still be present, never dropped"
        value_cols = [c for c in df.columns if c not in ("drift_type", "method", "n_cells", "frequency", "magnitude")]
        for col in value_cols:
            assert (static_rows[col] == "N/A").all(), f"{col} must be literal 'N/A' for static_rule"


# ==========================================================================
# TASK 8A -- Scientific Corrections: required regression tests
# ==========================================================================


# --- Correction 1: GradualRotationDrift's normalization must be EXACT ---


def test_gradual_rotation_history_equals_true_matrix_at_every_episode():
    """Required test 1: for every single episode of a ramp (not just a
    handful of probes), the compact ``matrix_history()`` must reconstruct
    EXACTLY the same matrix a fresh, independently-stepped instance
    reports via ``current_matrix()`` at that episode -- Correction 1's
    whole point."""
    target = MATRIX[::-1].copy()
    start, window, n_episodes = 20, 60, 120

    reference = GradualRotationDrift(MATRIX, target, start_episode=start, window=window)
    for i in range(n_episodes - 1):
        reference.step(i)
    history = reference.matrix_history()

    for probe in range(n_episodes):
        fresh = GradualRotationDrift(MATRIX, target, start_episode=start, window=window)
        _run_to_episode(fresh, probe)
        expected = fresh.current_matrix()

        active = None
        for event in history:
            if event.episode <= probe:
                active = event.matrix
            else:
                break
        assert active is not None
        assert np.array_equal(active, expected), f"episode {probe}: history reconstruction not exact"


# --- Correction 2: time-varying regret ---


def test_drift_regret_equals_ordinary_regret_when_stationary():
    """Required test 2: for a single-matrix history, every
    ``drift_regret`` function must be bit-identical to its
    ``recovery.metrics.regret`` counterpart."""
    env = MatrixEnvironment(MATRIX)
    log = run_episodes(env, ThompsonPolicy(seed=3), PerfectClassifier(), n_episodes=1500, seed=3)
    history = [(0, MATRIX)]

    assert np.array_equal(instantaneous_regret(log, MATRIX), instantaneous_drift_regret(log, history))
    assert np.array_equal(cumulative_regret(log, MATRIX), cumulative_drift_regret(log, history))
    assert normalized_cumulative_regret(log, MATRIX) == pytest.approx(normalized_drift_regret(log, history))


def test_changing_future_matrix_does_not_alter_past_regret():
    """Required test 3: two histories that agree up to episode T but
    diverge afterward must produce IDENTICAL cumulative drift regret for
    every episode < T -- regret at episode e depends only on the matrix
    segment active at e, never on what happens later."""
    rng = np.random.default_rng(21)
    env = PermutationDrift(MATRIX, schedule=[400], affected_rows=[0, 1], rng=rng)
    log = run_episodes(env, ThompsonPolicy(seed=4), PerfectClassifier(), n_episodes=800, seed=4)
    history = env.matrix_history()
    assert len(history) == 2  # base + one shift at episode 400

    real_future = history[1].matrix
    alternate_future = np.clip(MATRIX + 0.35, 0.0, 1.0)
    assert not np.allclose(real_future, alternate_future)

    history_alt = [history[0], (400, alternate_future)]
    cum_real = cumulative_drift_regret(log, history)
    cum_alt = cumulative_drift_regret(log, history_alt)

    assert np.array_equal(cum_real[:400], cum_alt[:400])
    assert not np.array_equal(cum_real[400:], cum_alt[400:])


# --- Correction 4: stationary digest equality ---


def _pre_task8_run_episodes(env, policy, classifier, n_episodes, seed):
    """Byte-for-byte replica of ``recovery.core.episode.run_episodes`` AS
    IT EXISTED BEFORE TASK 8 -- the ONLY difference from today's version
    is the omitted ``drift_signal=getattr(...)`` kwarg in ``Context``
    construction below. Kept local to this test (not imported from
    production code) precisely because it must NOT track future changes
    to ``recovery/core/episode.py`` -- it is a frozen reference for this
    one comparison."""
    from recovery.core.log import EpisodeLog
    from recovery.core.rng import from_seed

    streams = from_seed(seed)
    log = EpisodeLog(n_episodes)
    policy.reset()
    types_rng, classifier_rng, reward_rng, policy_rng = (
        streams.types, streams.classifier, streams.reward, streams.policy
    )
    n_types = env.n_types
    for i in range(n_episodes):
        true_type = int(types_rng.integers(0, n_types))
        predicted = classifier.predict(true_type, classifier_rng)
        ctx = Context(predicted_type=predicted, episode=i)  # no drift_signal kwarg -- pre-Task-8
        action = policy.select(ctx, policy_rng)
        reward = env.sample_reward(true_type, action, reward_rng)
        policy.update(ctx, action, reward)
        env.step(i)
        log.true_type[i] = true_type
        log.predicted_type[i] = predicted
        log.action[i] = action
        log.reward[i] = reward
    return log


def _log_digest(log) -> str:
    import hashlib

    h = hashlib.sha256()
    for arr in (log.episode, log.true_type, log.predicted_type, log.action, log.reward):
        h.update(np.asarray(arr).tobytes())
    return h.hexdigest()


def test_stationary_digest_equality_before_and_after_task8():
    """Required test 4: run the SAME stationary configuration through
    the pre-Task-8 kernel loop and today's kernel, and require every
    ``EpisodeLog`` array to be element-wise identical, and their SHA256
    content digests to match exactly. See
    ``docs/experiments/drift_validation.md`` for the full digest values
    reported from this same comparison at a larger episode count."""
    env_before, env_after = MatrixEnvironment(MATRIX), MatrixEnvironment(MATRIX)
    log_before = _pre_task8_run_episodes(env_before, ThompsonPolicy(seed=1), PerfectClassifier(), 5000, 77)
    log_after = run_episodes(env_after, ThompsonPolicy(seed=1), PerfectClassifier(), n_episodes=5000, seed=77)

    assert np.array_equal(log_before.episode, log_after.episode)
    assert np.array_equal(log_before.true_type, log_after.true_type)
    assert np.array_equal(log_before.predicted_type, log_after.predicted_type)
    assert np.array_equal(log_before.action, log_after.action)
    assert np.array_equal(log_before.reward, log_after.reward)

    digest_before, digest_after = _log_digest(log_before), _log_digest(log_after)
    assert digest_before == digest_after


# --- Matrix history replay reconstructs every matrix exactly (dense case) ---


def test_matrix_history_replay_reconstructs_gradual_rotation_exactly():
    """Required test 5 (companion to
    ``test_matrix_history_replay_reconstructs_every_drift_exactly``,
    which already covers the event-driven models): every episode of a
    dense ``GradualRotationDrift`` history, not just discrete shift
    points, must replay exactly."""
    target = MATRIX[::-1].copy()
    env = GradualRotationDrift(MATRIX, target, start_episode=10, window=30)
    for i in range(59):
        env.step(i)
    history = env.matrix_history()

    for probe in range(60):
        fresh = GradualRotationDrift(MATRIX, target, start_episode=10, window=30)
        _run_to_episode(fresh, probe)
        expected = fresh.current_matrix()
        active = next((e.matrix for e in reversed(history) if e.episode <= probe), None)
        assert active is not None and np.array_equal(active, expected)
