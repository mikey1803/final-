"""Unit tests for the shared simulation kernel (recovery.core.episode).

Covers determinism, RNG stream independence, the two synthetic
classifiers, the reward-path regression guard for the Table VII anomaly,
and reset correctness.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.classify.synthetic import ConfusionClassifier, PerfectClassifier
from recovery.core.episode import run_episodes
from recovery.core.protocols import Context
from recovery.envs.canonical import BLIND_RULE, MATRIX
from recovery.envs.matrix_env import MatrixEnvironment
from recovery.policies.adapters import (
    OraclePolicy,
    StaticRulePolicy,
    ThompsonPolicy,
    UniformRandomPolicy,
)
from recovery.theory.analytic import policy_rate_under_confusion


# ======================================================================
# 1. Determinism
# ======================================================================


def test_determinism_same_seed_identical_logs():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()

    log_a = run_episodes(env, StaticRulePolicy(BLIND_RULE), clf, n_episodes=5000, seed=123)
    log_b = run_episodes(env, StaticRulePolicy(BLIND_RULE), clf, n_episodes=5000, seed=123)

    assert np.array_equal(log_a.episode, log_b.episode)
    assert np.array_equal(log_a.true_type, log_b.true_type)
    assert np.array_equal(log_a.predicted_type, log_b.predicted_type)
    assert np.array_equal(log_a.action, log_b.action)
    assert np.array_equal(log_a.reward, log_b.reward)


# ======================================================================
# 2. Stream independence
# ======================================================================


def test_stream_independence_true_type_sequence_unaffected_by_policy():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()

    log_static = run_episodes(env, StaticRulePolicy(BLIND_RULE), clf, n_episodes=5000, seed=99)
    log_random = run_episodes(env, UniformRandomPolicy(seed=7), clf, n_episodes=5000, seed=99)
    log_oracle = run_episodes(env, OraclePolicy(MATRIX), clf, n_episodes=5000, seed=99)

    assert np.array_equal(log_static.true_type, log_random.true_type)
    assert np.array_equal(log_static.true_type, log_oracle.true_type)
    # Actions legitimately differ -- policies are different -- but the
    # environment's true-type sequence must be byte-identical regardless.
    assert not np.array_equal(log_static.action, log_random.action)


# ======================================================================
# 3. PerfectClassifier correctness
# ======================================================================


def test_perfect_classifier_predicted_equals_true_for_all_episodes():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()
    log = run_episodes(env, StaticRulePolicy(BLIND_RULE), clf, n_episodes=10_000, seed=1)
    assert np.array_equal(log.true_type, log.predicted_type)


# ======================================================================
# 4. ConfusionClassifier empirical accuracy
# ======================================================================


def test_confusion_classifier_empirical_diagonal_matches_accuracy():
    n = MATRIX.shape[0]
    accuracy = 0.7
    off_diagonal = (1.0 - accuracy) / (n - 1)
    C = np.full((n, n), off_diagonal, dtype=np.float64)
    np.fill_diagonal(C, accuracy)

    env = MatrixEnvironment(MATRIX)
    clf = ConfusionClassifier(C)
    log = run_episodes(env, StaticRulePolicy(BLIND_RULE), clf, n_episodes=200_000, seed=2)

    empirical_accuracy = float(np.mean(log.true_type == log.predicted_type))
    assert empirical_accuracy == pytest.approx(accuracy, abs=0.01)


# ======================================================================
# 5. Reward-path regression guard (the Table VII anomaly)
# ======================================================================


def test_reward_path_guard_reward_uses_true_type_not_predicted_type():
    """Regression test for the Table VII anomaly (docs/anomalies.md).

    Builds a deliberately asymmetric 2x2 matrix and a classifier that
    ALWAYS predicts type 0, regardless of the true type. Because the
    policy is a constant function of the predicted type (always 0 here),
    the ACTION played is the same every episode -- so this isolates
    exactly one thing: whether ``sample_reward`` was evaluated against the
    TRUE type's row (correct) or the PREDICTED type's row (the historical
    bug class this kernel is designed to make structurally impossible).

    If ``run_episodes`` ever regresses to sampling reward from the
    predicted type, the empirical rate collapses to a CONSTANT
    ``P[0, action]`` (0.9 here) regardless of the true-type mix, sharply
    diverging from the correct analytic prediction (0.5) computed via
    ``policy_rate_under_confusion`` from the TRUE-type-indexed matrix.
    This test would then fail loudly.
    """
    P = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    always_predicts_zero = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    action = 0
    pi = np.array([action, action])  # constant policy: always plays `action`

    analytic_prediction = policy_rate_under_confusion(P, pi, always_predicts_zero)
    assert analytic_prediction == pytest.approx(0.5)

    env = MatrixEnvironment(P)
    clf = ConfusionClassifier(always_predicts_zero)
    policy = StaticRulePolicy(pi)
    log = run_episodes(env, policy, clf, n_episodes=100_000, seed=3)

    # Sanity: the classifier really does always predict 0.
    assert np.all(log.predicted_type == 0)
    # Sanity: the true type still varies (else this test proves nothing).
    assert log.true_type.min() == 0 and log.true_type.max() == 1

    empirical_rate = log.success_rate()
    assert empirical_rate == pytest.approx(analytic_prediction, abs=0.01)
    # If the reward path were bugged (predicted-type row instead of
    # true-type row), the empirical rate would instead sit near
    # P[0, action] = 0.9, far outside this tolerance.
    assert abs(empirical_rate - 0.9) > 0.05


# ======================================================================
# 6. Reset correctness
# ======================================================================


def test_reset_two_consecutive_runs_same_policy_object_identical():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()
    policy = ThompsonPolicy(seed=11)  # stateful, learns across episodes

    log_a = run_episodes(env, policy, clf, n_episodes=2000, seed=42)
    log_b = run_episodes(env, policy, clf, n_episodes=2000, seed=42)

    assert np.array_equal(log_a.action, log_b.action)
    assert np.array_equal(log_a.reward, log_b.reward)
    assert np.array_equal(log_a.true_type, log_b.true_type)
    assert np.array_equal(log_a.predicted_type, log_b.predicted_type)


# ======================================================================
# Context immutability (structural guarantee, not just convention)
# ======================================================================


def test_context_is_frozen():
    ctx = Context(predicted_type=0)
    with pytest.raises(Exception):
        ctx.predicted_type = 1  # type: ignore[misc]
