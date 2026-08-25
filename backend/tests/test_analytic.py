"""Tests for the independent analytic reference layer (recovery.theory.analytic).

Three layers: exact hand-computed constants against the canonical
environment, general mathematical properties, and input-validation errors.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.envs.canonical import BLIND_RULE, MATRIX
from recovery.theory.analytic import (
    normalize,
    oracle_rate,
    policy_rate,
    policy_rate_under_confusion,
    random_rate,
    uniform_confusion,
)

# ======================================================================
# Layer 1 — exact constants (hand-computed against the canonical matrix)
# ======================================================================


def test_oracle_rate_exact():
    assert oracle_rate(MATRIX) == pytest.approx(5.30 / 7)


def test_random_rate_exact():
    assert random_rate(MATRIX) == pytest.approx(18.35 / 42)


def test_policy_rate_exact():
    assert policy_rate(MATRIX, BLIND_RULE) == pytest.approx(5.15 / 7)


def test_normalize_exact():
    rate = policy_rate(MATRIX, BLIND_RULE)
    assert normalize(rate, MATRIX) == pytest.approx(0.9331, abs=1e-4)


# ======================================================================
# Layer 2 — properties
# ======================================================================


def test_identity_confusion_equals_policy_rate():
    n = MATRIX.shape[0]
    identity = np.eye(n, dtype=np.float64)
    assert policy_rate_under_confusion(MATRIX, BLIND_RULE, identity) == pytest.approx(
        policy_rate(MATRIX, BLIND_RULE)
    )


@pytest.mark.parametrize("accuracy", [0.99, 0.95, 0.90, 0.80, 0.50, 1.0 / 7.0])
def test_confused_rate_never_exceeds_ceiling(accuracy):
    n = MATRIX.shape[0]
    C = uniform_confusion(n, accuracy)
    confused = policy_rate_under_confusion(MATRIX, BLIND_RULE, C)
    ceiling = policy_rate(MATRIX, BLIND_RULE)
    assert confused <= ceiling + 1e-9


def test_uniform_confusion_at_chance_accuracy_erases_policy_identity():
    """At accuracy=1/n, uniform_confusion makes every row of C identical
    (uniform), so the predicted type carries no information about the true
    type. Two policies that are permutations of each other (same multiset
    of strategy indices, different assignment to types) then yield
    identical rates, since the action is effectively drawn uniformly from
    the policy's value multiset regardless of the true type.
    """
    n = MATRIX.shape[0]
    C = uniform_confusion(n, 1.0 / n)
    rule_a = BLIND_RULE
    rule_b = BLIND_RULE[::-1].copy()  # a permutation: same multiset, different assignment
    assert not np.array_equal(rule_a, rule_b)
    rate_a = policy_rate_under_confusion(MATRIX, rule_a, C)
    rate_b = policy_rate_under_confusion(MATRIX, rule_b, C)
    assert rate_a == pytest.approx(rate_b)


def test_policy_rate_with_argmax_policy_equals_oracle():
    argmax_policy = MATRIX.argmax(axis=1)
    assert policy_rate(MATRIX, argmax_policy) == pytest.approx(oracle_rate(MATRIX))


def test_random_policies_never_exceed_oracle():
    rng = np.random.default_rng(0)
    n_types, n_strats = MATRIX.shape
    ceiling = oracle_rate(MATRIX)
    for _ in range(200):
        pi = rng.integers(0, n_strats, size=n_types)
        assert policy_rate(MATRIX, pi) <= ceiling + 1e-9


def test_nonuniform_prior_sums_correctly_and_within_bounds():
    n = MATRIX.shape[0]
    rng = np.random.default_rng(42)
    raw = rng.random(n)
    p = raw / raw.sum()
    assert p.sum() == pytest.approx(1.0)

    lo = random_rate(MATRIX, p)
    hi = oracle_rate(MATRIX, p)
    rate = policy_rate(MATRIX, BLIND_RULE, p)
    assert lo <= rate <= hi + 1e-9


# ======================================================================
# Layer 3 — validation
# ======================================================================


def test_transposed_P_raises():
    with pytest.raises(ValueError):
        policy_rate(MATRIX.T, BLIND_RULE)


def test_bad_pi_index_raises():
    bad_pi = BLIND_RULE.copy()
    bad_pi[0] = MATRIX.shape[1]  # one past the last valid strategy index
    with pytest.raises(ValueError):
        policy_rate(MATRIX, bad_pi)


def test_non_stochastic_C_raises():
    n = MATRIX.shape[0]
    C = np.full((n, n), 1.0 / n, dtype=np.float64)
    C[0, 0] += 0.5  # break row-stochasticity
    with pytest.raises(ValueError):
        policy_rate_under_confusion(MATRIX, BLIND_RULE, C)


def test_prior_not_summing_to_one_raises():
    n = MATRIX.shape[0]
    p = np.full(n, 1.0 / n, dtype=np.float64)
    p[0] += 0.1
    with pytest.raises(ValueError):
        policy_rate(MATRIX, BLIND_RULE, p)


# ======================================================================
# Anomaly documentation
# ======================================================================


def test_reported_0741_exceeds_perfect_classifier_ceiling_and_uniform_noise_prediction():
    """Documents the Table VII anomaly this task investigates.

    The paper reports 0.741 for the blind 6/7-correct rule at 93.9%
    classifier accuracy (see docs/anomalies.md). A uniform-noise confusion
    model at that same accuracy predicts ~0.7146, and the noiseless ceiling
    for this policy is policy_rate(MATRIX, BLIND_RULE) ~= 0.735714 — 0.741
    exceeds both.
    """
    reported_value = 0.741
    ceiling = policy_rate(MATRIX, BLIND_RULE)
    uniform_noise_prediction = policy_rate_under_confusion(
        MATRIX, BLIND_RULE, uniform_confusion(MATRIX.shape[0], 0.939)
    )

    assert uniform_noise_prediction == pytest.approx(0.7146, abs=1e-3)
    assert reported_value > ceiling
