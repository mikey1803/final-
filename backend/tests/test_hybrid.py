"""Tests for Prior-Guided Thompson Sampling (PGTS) --
``recovery.policies.hybrid.PriorInitializedThompson``/``DiscountedPriorThompson``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betaln, digamma

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.synthetic import PerfectClassifier
from recovery.core.episode import run_episodes
from recovery.core.protocols import Context
from recovery.envs.canonical import MATRIX, STRATEGY_NAMES
from recovery.envs.matrix_env import MatrixEnvironment
from recovery.policies.adapters import StaticRulePolicy
from recovery.policies.hybrid import DiscountedPriorThompson, PriorInitializedThompson

N_TYPES = MATRIX.shape[0]
N_STRATEGIES = MATRIX.shape[1]
BAD_RULE = np.array([(k + 1) % N_STRATEGIES for k in range(N_TYPES)], dtype=np.int64)  # a fixed, arbitrary table


def _static_env_and_prior_table():
    return MATRIX, BAD_RULE


# ======================================================================
# 1. strength=0 => the table has no effect => bitwise identical to
#    (native, vectorized) stock Thompson Sampling.
# ======================================================================


def test_strength_zero_is_table_independent_and_matches_stock_thompson():
    """At strength=0, ``alpha0=beta0=1`` everywhere REGARDLESS of the
    table (see ``PriorInitializedThompson._build_prior``) -- so this IS
    exactly stock (uninformative-prior) Thompson Sampling's math, native
    and vectorized. Proven here two ways:

    (a) two PGTS instances built from DIFFERENT tables, strength=0, same
        seed, produce BITWISE-IDENTICAL action/reward sequences -- the
        table cannot be influencing anything.
    (b) a from-scratch, minimal vectorized Thompson Sampling
        implementation (literally ``Beta(1,1)`` everywhere, updated by
        reward -- i.e. what "stock Thompson Sampling" mathematically IS)
        produces the SAME sequence too, given the same RNG draws.
    """
    table_a = np.zeros(N_TYPES, dtype=np.int64)
    table_b = (np.arange(N_TYPES) % N_STRATEGIES).astype(np.int64)
    assert not np.array_equal(table_a, table_b)

    env_a = MatrixEnvironment(MATRIX)
    pgts_a = PriorInitializedThompson(table_a, N_STRATEGIES, strength=0.0, prior_mean=0.7)
    log_a = run_episodes(env_a, pgts_a, PerfectClassifier(), n_episodes=3000, seed=101)

    env_b = MatrixEnvironment(MATRIX)
    pgts_b = PriorInitializedThompson(table_b, N_STRATEGIES, strength=0.0, prior_mean=0.9)
    log_b = run_episodes(env_b, pgts_b, PerfectClassifier(), n_episodes=3000, seed=101)

    assert np.array_equal(log_a.action, log_b.action)
    assert np.array_equal(log_a.reward, log_b.reward)

    # (b) manual from-scratch vectorized Beta(1,1) Thompson Sampling, run
    # through the SAME kernel/seed, must match too.
    class _ManualStockThompson:
        name = "manual_stock_thompson"

        def __init__(self):
            self._alpha = np.ones((N_TYPES, N_STRATEGIES))
            self._beta = np.ones((N_TYPES, N_STRATEGIES))

        def select(self, ctx: Context, rng: np.random.Generator) -> int:
            t = ctx.predicted_type
            theta = rng.beta(self._alpha[t], self._beta[t])
            return int(np.argmax(theta))

        def update(self, ctx: Context, action: int, reward: int) -> None:
            t = ctx.predicted_type
            self._alpha[t, action] += reward
            self._beta[t, action] += 1 - reward

        def reset(self) -> None:
            self._alpha[:] = 1.0
            self._beta[:] = 1.0

    env_c = MatrixEnvironment(MATRIX)
    log_c = run_episodes(env_c, _ManualStockThompson(), PerfectClassifier(), n_episodes=3000, seed=101)
    assert np.array_equal(log_a.action, log_c.action)
    assert np.array_equal(log_a.reward, log_c.reward)


# ======================================================================
# 2. strength=1e6 behaves like the static rule during early/sustained
#    learning.
# ======================================================================


def test_very_large_strength_behaves_like_static_rule():
    """A very large (but finite) strength should track
    ``StaticRulePolicy`` closely -- NOT bitwise-identical from episode 0,
    which is not actually guaranteed by the algorithm as specified (see
    docs/experiments/hybrid.md, "A subtlety in the large-strength
    limit"): every non-recommended arm still starts at literal
    ``Beta(1,1)`` = Uniform(0,1), which has a small but nonzero chance of
    outsampling the recommended arm's tightly concentrated draw on any
    given episode purely by chance, before that arm's own posterior
    shifts down from real evidence. With ``prior_mean`` close to 1 this
    chance is small per-episode and the two policies' action sequences
    match with high fidelity -- checked here at >=97% overall and >=99%
    over the run's final quarter (well past the brief early-exploration
    window where any divergence concentrates).
    """
    matrix, table = _static_env_and_prior_table()
    n_episodes = 5000
    seed = 0

    env_pgts = MatrixEnvironment(matrix)
    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=1e6, prior_mean=0.99)
    log_pgts = run_episodes(env_pgts, pgts, PerfectClassifier(), n_episodes=n_episodes, seed=seed)

    env_static = MatrixEnvironment(matrix)
    log_static = run_episodes(
        env_static, StaticRulePolicy(table), PerfectClassifier(), n_episodes=n_episodes, seed=seed
    )

    overall_match = np.mean(log_pgts.action == log_static.action)
    late_quarter = n_episodes * 3 // 4
    late_match = np.mean(log_pgts.action[late_quarter:] == log_static.action[late_quarter:])

    assert overall_match >= 0.97, f"overall action match {overall_match:.4f} < 0.97"
    assert late_match >= 0.99, f"late-quarter action match {late_match:.4f} < 0.99"


# ======================================================================
# 3. reset() restores the ORIGINAL informed prior, not Beta(1,1).
# ======================================================================


def test_reset_restores_original_informed_prior_not_uniform():
    table = np.array([2, 0, 1, 3, 5, 4, 2], dtype=np.int64)[:N_TYPES]
    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=15.0, prior_mean=0.65)

    alpha0 = pgts.posterior_mean().copy()  # captures the informed prior's shape via the mean
    original_alpha_full = pgts._alpha.copy()
    original_beta_full = pgts._beta.copy()

    # informed arm should NOT be uniform Beta(1,1)
    rec_mean = pgts.posterior_mean()[0, table[0]]
    assert rec_mean != pytest.approx(0.5)

    rng = np.random.default_rng(5)
    env = MatrixEnvironment(MATRIX)
    for i in range(200):
        ctx = Context(predicted_type=int(rng.integers(0, N_TYPES)), episode=i)
        action = pgts.select(ctx, rng)
        reward = int(rng.random() < 0.5)
        pgts.update(ctx, action, reward)

    assert not np.array_equal(pgts._alpha, original_alpha_full)  # state actually changed

    pgts.reset()
    assert np.array_equal(pgts._alpha, original_alpha_full)
    assert np.array_equal(pgts._beta, original_beta_full)
    assert not np.allclose(pgts._alpha, 1.0)  # confirms it's the INFORMED prior, not Beta(1,1)


# ======================================================================
# 4. Updates use the PREDICTED type, never the true type.
# ======================================================================


def test_update_only_touches_predicted_types_row():
    """Directly calling ``update`` with a fixed ``predicted_type`` must
    only ever change that type's row -- ``Context`` structurally has no
    true-type field for anything to leak through even if it wanted to
    (see ``recovery.core.protocols.Context``), and this checks the
    POLICY's own update logic honors that (only ``ctx.predicted_type`` is
    read, confirmed by the fact that no other row moves)."""
    table = np.zeros(N_TYPES, dtype=np.int64)
    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=10.0, prior_mean=0.7)
    before = pgts._alpha.copy(), pgts._beta.copy()

    target_type = 3
    ctx = Context(predicted_type=target_type, episode=0)
    pgts.update(ctx, action=2, reward=1)

    after_alpha, after_beta = pgts._alpha, pgts._beta
    for t in range(N_TYPES):
        if t == target_type:
            assert after_alpha[t, 2] == before[0][t, 2] + 1
            unchanged_mask = np.ones(N_STRATEGIES, dtype=bool)
            unchanged_mask[2] = False
            assert np.array_equal(after_alpha[t, unchanged_mask], before[0][t, unchanged_mask])
            assert np.array_equal(after_beta[t], before[1][t])  # reward=1 -> beta unaffected
        else:
            assert np.array_equal(after_alpha[t], before[0][t])
            assert np.array_equal(after_beta[t], before[1][t])


# ======================================================================
# 5. Discounting decays the posterior toward Beta(1,1).
# ======================================================================


def test_discount_decays_posterior_toward_beta_1_1():
    table = np.zeros(N_TYPES, dtype=np.int64)
    pgts = DiscountedPriorThompson(table, N_STRATEGIES, strength=100.0, prior_mean=0.7, gamma=0.5)
    t = 0
    recommended_arm = int(table[t])
    other_arm = (recommended_arm + 1) % N_STRATEGIES

    alpha0_rec = pgts._alpha[t, recommended_arm]
    beta0_rec = pgts._beta[t, recommended_arm]
    assert alpha0_rec == pytest.approx(1.0 + 100.0 * 0.7)

    # Repeatedly update a DIFFERENT arm (`other_arm`) -- every update()
    # discounts the WHOLE row first, so `recommended_arm`'s elevated
    # prior decays purely from discounting, never replenished by direct
    # evidence.
    for i in range(30):
        ctx = Context(predicted_type=t, episode=i)
        pgts.update(ctx, action=other_arm, reward=1)

    assert pgts._alpha[t, recommended_arm] == pytest.approx(1.0, abs=1e-3)
    assert pgts._beta[t, recommended_arm] == pytest.approx(1.0, abs=1e-3)
    # sanity: it MOVED (didn't start at 1.0)
    assert alpha0_rec > 1.5


def test_base_class_gamma_one_is_a_no_op_discount():
    """gamma=1.0 (the PriorInitializedThompson default) must be an exact
    identity -- no discount drift at all, confirming DiscountedPriorThompson
    is purely an added behavior, not a different code path."""
    table = np.zeros(N_TYPES, dtype=np.int64)
    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=50.0, prior_mean=0.7, gamma=1.0)
    t, other_arm = 0, 1
    rec_before = pgts._alpha[t, int(table[t])], pgts._beta[t, int(table[t])]
    for i in range(50):
        ctx = Context(predicted_type=t, episode=i)
        pgts.update(ctx, action=other_arm, reward=1)
    rec_after = pgts._alpha[t, int(table[t])], pgts._beta[t, int(table[t])]
    assert rec_after == rec_before


# ======================================================================
# 6. Validation errors raise ValueError.
# ======================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(strength=-1.0),
        dict(strength=float("inf")),
        dict(strength=float("nan")),
        dict(prior_mean=0.0),
        dict(prior_mean=1.0),
        dict(prior_mean=1.5),
        dict(prior_mean=-0.1),
        dict(gamma=0.0),
        dict(gamma=1.5),
        dict(gamma=-0.1),
    ],
)
def test_invalid_constructor_arguments_raise_value_error(kwargs):
    table = np.zeros(N_TYPES, dtype=np.int64)
    base_kwargs = dict(strength=20.0, prior_mean=0.7, gamma=1.0)
    base_kwargs.update(kwargs)
    with pytest.raises(ValueError):
        PriorInitializedThompson(table, N_STRATEGIES, **base_kwargs)


def test_out_of_range_table_index_raises_value_error():
    bad_table = np.array([0, 1, 2, 3, 4, N_STRATEGIES, 0], dtype=np.int64)[:N_TYPES]
    with pytest.raises(ValueError):
        PriorInitializedThompson(bad_table, N_STRATEGIES, strength=10.0)


def test_empty_table_raises_value_error():
    with pytest.raises(ValueError):
        PriorInitializedThompson(np.array([], dtype=np.int64), N_STRATEGIES, strength=10.0)


def test_extreme_strength_raises_value_error_on_overflow():
    """A pathologically large but finite strength (far beyond anything
    the experiment grid uses) doesn't overflow at construction (alpha0/
    beta0 stay finite -- see ``_build_prior``'s docstring), but DOES
    overflow ``posterior_variance()``'s squaring step -- caught there and
    raised as ValueError rather than silently returned as inf/nan."""
    table = np.zeros(N_TYPES, dtype=np.int64)
    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=1e300, prior_mean=0.99)
    with pytest.raises(ValueError):
        pgts.posterior_variance()


# ======================================================================
# 7. Higher strength never hurts when quality is perfect and the
#    environment is stationary (within Monte Carlo noise).
# ======================================================================


def test_higher_strength_never_hurts_with_perfect_quality_stationary():
    perfect_table = MATRIX.argmax(axis=1)
    n_seeds = 15
    n_episodes = 2000
    strengths = [0.0, 5.0, 20.0, 100.0]

    mean_rates = []
    for strength in strengths:
        rates = []
        for seed_idx in range(n_seeds):
            seed = 3000 + seed_idx
            env = MatrixEnvironment(MATRIX)
            pgts = PriorInitializedThompson(perfect_table, N_STRATEGIES, strength=strength, prior_mean=0.7)
            log = run_episodes(env, pgts, PerfectClassifier(), n_episodes=n_episodes, seed=seed)
            rates.append(log.reward[-int(n_episodes * 0.2):].mean())
        mean_rates.append(float(np.mean(rates)))

    # A generous tolerance absorbs Monte Carlo noise -- the claim is "no
    # meaningful harm", not "strictly monotonic every step".
    tolerance = 0.03
    for i in range(1, len(mean_rates)):
        assert mean_rates[i] >= mean_rates[0] - tolerance, (
            f"strength={strengths[i]} mean rate {mean_rates[i]:.4f} fell more than "
            f"{tolerance} below strength=0's {mean_rates[0]:.4f}"
        )


# ======================================================================
# 8. Negative transfer exists for low-quality, high-strength priors
#    under drift.
# ======================================================================


def test_negative_transfer_exists_for_bad_prior_under_drift():
    """If no negative-transfer region is found, this test fails -- a
    confidently WRONG prior under drift must be demonstrably worse than
    an uninformed one somewhere in this parameter space, or the
    implementation contradicts PGTS's intended algorithmic behaviour
    (see docs/experiments/hybrid.md, "Negative transfer")."""
    from experiments.prior_grid import DriftEnvironment, _pick_shift_types

    n_seeds = 20
    n_episodes = 1000
    shift_episode = 500
    strength = 500.0
    prior_mean = 0.8

    bad_rates, stock_rates = [], []
    for seed_idx in range(n_seeds):
        seed = 2000 + seed_idx
        bad_table = PriorInitializedThompson.from_quality(MATRIX, 0, seed)  # fully wrong table

        shift_rng = np.random.default_rng(seed)
        affected = _pick_shift_types(shift_rng, N_TYPES, 2)

        env_bad = DriftEnvironment(MATRIX, [shift_episode], affected, shift_seed=seed)
        pgts_bad = PriorInitializedThompson(bad_table, N_STRATEGIES, strength=strength, prior_mean=prior_mean)
        log_bad = run_episodes(env_bad, pgts_bad, PerfectClassifier(), n_episodes=n_episodes, seed=seed)

        env_stock = DriftEnvironment(MATRIX, [shift_episode], affected, shift_seed=seed)
        pgts_stock = PriorInitializedThompson(bad_table, N_STRATEGIES, strength=0.0, prior_mean=prior_mean)
        log_stock = run_episodes(env_stock, pgts_stock, PerfectClassifier(), n_episodes=n_episodes, seed=seed)

        # early post-shift window: the confidently-wrong prior has not
        # yet been overridden by evidence here.
        window = slice(shift_episode, shift_episode + 150)
        bad_rates.append(log_bad.reward[window].mean())
        stock_rates.append(log_stock.reward[window].mean())

    bad_rates, stock_rates = np.array(bad_rates), np.array(stock_rates)
    assert bad_rates.mean() < stock_rates.mean(), (
        f"expected negative transfer (high-strength wrong prior worse than stock "
        f"Thompson under drift), got bad={bad_rates.mean():.4f} >= "
        f"stock={stock_rates.mean():.4f}"
    )
    fraction_worse = float(np.mean(bad_rates < stock_rates))
    assert fraction_worse >= 0.6, f"negative transfer only in {fraction_worse:.2f} of seeds, expected >= 0.6"


# ======================================================================
# 9. Posterior diagnostics match analytic Beta moments.
# ======================================================================


def test_posterior_diagnostics_match_analytic_beta_moments():
    table = np.array([1, 3, 5, 0, 2, 4, 1], dtype=np.int64)[:N_TYPES]
    strength, prior_mean = 30.0, 0.6
    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=strength, prior_mean=prior_mean)

    rng = np.random.default_rng(9)
    for i in range(50):
        ctx = Context(predicted_type=int(rng.integers(0, N_TYPES)), episode=i)
        action = int(rng.integers(0, N_STRATEGIES))
        reward = int(rng.random() < 0.5)
        pgts.update(ctx, action, reward)

    a, b = pgts._alpha, pgts._beta
    expected_mean = a / (a + b)
    expected_var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    expected_entropy = betaln(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) + (a + b - 2.0) * digamma(a + b)
    expected_confidence = a + b
    expected_prior_strength = pgts._alpha0 + pgts._beta0 - 2.0

    np.testing.assert_allclose(pgts.posterior_mean(), expected_mean)
    np.testing.assert_allclose(pgts.posterior_variance(), expected_var)
    np.testing.assert_allclose(pgts.posterior_entropy(), expected_entropy)
    np.testing.assert_allclose(pgts.posterior_confidence(), expected_confidence)
    np.testing.assert_allclose(pgts.effective_prior_strength(), expected_prior_strength)

    # sanity: the recommended arm's ORIGINAL prior strength equals `strength`
    for t in range(N_TYPES):
        assert pgts.effective_prior_strength()[t, table[t]] == pytest.approx(strength)


def test_diagnostics_do_not_affect_policy_behavior():
    """Calling every diagnostic method must not change subsequent
    select()/update() behavior (read-only, per the module's contract)."""
    table = np.zeros(N_TYPES, dtype=np.int64)
    pgts_a = PriorInitializedThompson(table, N_STRATEGIES, strength=10.0, prior_mean=0.7)
    pgts_b = PriorInitializedThompson(table, N_STRATEGIES, strength=10.0, prior_mean=0.7)

    # exercise every diagnostic on pgts_a only
    _ = pgts_a.posterior_mean()
    _ = pgts_a.posterior_variance()
    _ = pgts_a.posterior_entropy()
    _ = pgts_a.effective_prior_strength()
    _ = pgts_a.posterior_confidence()

    env_a = MatrixEnvironment(MATRIX)
    env_b = MatrixEnvironment(MATRIX)
    log_a = run_episodes(env_a, pgts_a, PerfectClassifier(), n_episodes=500, seed=17)
    log_b = run_episodes(env_b, pgts_b, PerfectClassifier(), n_episodes=500, seed=17)
    assert np.array_equal(log_a.action, log_b.action)
    assert np.array_equal(log_a.reward, log_b.reward)


# ======================================================================
# 10. Posterior entropy monotonically decreases during sustained
#     learning in a stationary environment.
# ======================================================================


def test_posterior_entropy_monotonically_decreases_under_sustained_learning():
    """Tracks the entropy of the arm a moderate-strength PGTS is expected
    to exploit heavily (its own recommended, and objectively good, arm
    for a frequently-occurring type) at coarse checkpoints -- individual
    episodes can occasionally tick entropy up (a surprising outcome
    widens the posterior slightly before it narrows again), but checked
    at 250-episode granularity over a long stationary run, the trend is
    strictly decreasing."""
    table = MATRIX.argmax(axis=1)  # a good (perfect) table -> heavily exploited
    strength = 5.0
    tracked_type = 0
    tracked_arm = int(table[tracked_type])

    pgts = PriorInitializedThompson(table, N_STRATEGIES, strength=strength, prior_mean=0.7)
    env = MatrixEnvironment(MATRIX)
    rng = np.random.default_rng(0)

    n_total, checkpoint_every = 5000, 250
    checkpoints = []
    for i in range(n_total):
        true_type = int(rng.integers(0, N_TYPES))
        ctx = Context(predicted_type=true_type, episode=i)
        action = pgts.select(ctx, rng)
        reward = int(rng.random() < MATRIX[true_type, action])
        pgts.update(ctx, action, reward)
        if (i + 1) % checkpoint_every == 0:
            checkpoints.append(float(pgts.posterior_entropy()[tracked_type, tracked_arm]))

    diffs = np.diff(checkpoints)
    assert np.all(diffs <= 1e-9), f"entropy increased between checkpoints: {checkpoints}"
    assert checkpoints[-1] < checkpoints[0]
