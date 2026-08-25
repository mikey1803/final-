"""Integration tests: does the shared simulation kernel converge to the
closed-form predictions in recovery.theory.analytic?

These run 200_000-episode simulations and are marked ``slow``. Run the
fast suite only with ``pytest -m "not slow"``; run everything (including
these) with plain ``pytest``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.classify.synthetic import ConfusionClassifier, PerfectClassifier
from recovery.core.episode import run_episodes
from recovery.envs.canonical import BLIND_RULE, MATRIX
from recovery.envs.matrix_env import MatrixEnvironment
from recovery.policies.adapters import OraclePolicy, StaticRulePolicy, UniformRandomPolicy
from recovery.theory.analytic import (
    oracle_rate,
    policy_rate,
    policy_rate_under_confusion,
    random_rate,
    uniform_confusion,
)

N_EPISODES = 200_000


@pytest.mark.slow
def test_static_rule_perfect_classifier_matches_policy_rate_ceiling():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()
    policy = StaticRulePolicy(BLIND_RULE)

    log = run_episodes(env, policy, clf, n_episodes=N_EPISODES, seed=0)
    empirical = log.success_rate()
    expected = policy_rate(MATRIX, BLIND_RULE)

    print(f"\n  StaticRulePolicy(BLIND_RULE) + PerfectClassifier: "
          f"empirical={empirical:.6f}  analytic={expected:.6f}  diff={empirical - expected:+.6f}")
    assert empirical == pytest.approx(expected, abs=0.003)


@pytest.mark.slow
def test_oracle_policy_matches_oracle_rate():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()
    policy = OraclePolicy(MATRIX)

    log = run_episodes(env, policy, clf, n_episodes=N_EPISODES, seed=0)
    empirical = log.success_rate()
    expected = oracle_rate(MATRIX)

    print(f"\n  OraclePolicy + PerfectClassifier: "
          f"empirical={empirical:.6f}  analytic={expected:.6f}  diff={empirical - expected:+.6f}")
    assert empirical == pytest.approx(expected, abs=0.003)


@pytest.mark.slow
def test_uniform_random_policy_matches_random_rate():
    env = MatrixEnvironment(MATRIX)
    clf = PerfectClassifier()
    policy = UniformRandomPolicy(seed=123)

    log = run_episodes(env, policy, clf, n_episodes=N_EPISODES, seed=0)
    empirical = log.success_rate()
    expected = random_rate(MATRIX)

    print(f"\n  UniformRandomPolicy + PerfectClassifier: "
          f"empirical={empirical:.6f}  analytic={expected:.6f}  diff={empirical - expected:+.6f}")
    assert empirical == pytest.approx(expected, abs=0.003)


@pytest.mark.slow
def test_static_rule_confusion_classifier_matches_policy_rate_under_confusion():
    n = MATRIX.shape[0]
    C = uniform_confusion(n, 0.939)

    env = MatrixEnvironment(MATRIX)
    clf = ConfusionClassifier(C)
    policy = StaticRulePolicy(BLIND_RULE)

    log = run_episodes(env, policy, clf, n_episodes=N_EPISODES, seed=0)
    empirical = log.success_rate()
    expected = policy_rate_under_confusion(MATRIX, BLIND_RULE, C)

    print(f"\n  StaticRulePolicy(BLIND_RULE) + ConfusionClassifier(uniform_confusion(7, 0.939)): "
          f"empirical={empirical:.6f}  analytic={expected:.6f}  diff={empirical - expected:+.6f}")
    assert empirical == pytest.approx(expected, abs=0.004)
