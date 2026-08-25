"""Tests for the runtime invariant guard layer (recovery.guards.invariants).

Covers the Table VII regression, positive checks against Task 2's measured
values, Naive Retry's valid-negative-normalized-score case, tolerance
scaling with sample size, hard unit-interval bounds, run_all_guards'
conditional-check selection, required exception-message content, and
input-validation (ValueError) edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recovery.guards.invariants as inv
from recovery.envs.canonical import BLIND_RULE, MATRIX
from recovery.guards.invariants import (
    InvariantViolation,
    check_above_worst_case,
    check_bounded_by_oracle,
    check_confusion_prediction,
    check_normalized_range,
    check_rate_in_unit_interval,
    check_static_policy_ceiling,
    run_all_guards,
)
from recovery.theory.analytic import policy_rate, policy_rate_under_confusion, uniform_confusion

# The historical episode count (500 x 50 trials) is too small to
# distinguish 0.741 from noise at 3-sigma; per the resolved design
# decision, this regression test uses Task 2's standard 200_000-episode
# scale, at which point 0.741 is unambiguously outside tolerance.
REGRESSION_N_EPISODES = 200_000


# ======================================================================
# Regression test: the historical Table VII anomaly
# ======================================================================


def test_static_policy_ceiling_regression_table_vii():
    """docs/anomalies.md: paper reported 0.741 for BLIND_RULE; ceiling is 0.735714."""
    with pytest.raises(InvariantViolation) as exc_info:
        check_static_policy_ceiling(0.741, MATRIX, BLIND_RULE, n_episodes=REGRESSION_N_EPISODES)
    message = str(exc_info.value)
    assert "0.741" in message
    assert "0.7357" in message


# ======================================================================
# Positive tests: Task 2's measured values must all pass
# ======================================================================


def test_task2_measured_values_pass_static_rule_perfect():
    run_all_guards(0.735915, MATRIX, 200_000, pi=BLIND_RULE)


def test_task2_measured_values_pass_oracle():
    run_all_guards(0.757070, MATRIX, 200_000, pi=MATRIX.argmax(axis=1))


def test_task2_measured_values_pass_uniform_random():
    # UniformRandomPolicy has no fixed pi (it is not a deterministic
    # mapping), so `pi` is intentionally omitted -- only the always-
    # applicable oracle/worst-case/unit-interval checks run.
    run_all_guards(0.437740, MATRIX, 200_000)


def test_task2_measured_values_pass_confusion():
    C = uniform_confusion(MATRIX.shape[0], 0.939)
    run_all_guards(0.715005, MATRIX, 200_000, pi=BLIND_RULE, C=C)


# ======================================================================
# Naive Retry: negative normalized scores are valid
# ======================================================================


def test_naive_retry_negative_normalized_score_passes():
    check_normalized_range(-0.3663)


def test_normalized_range_rejects_too_negative():
    with pytest.raises(InvariantViolation):
        check_normalized_range(-1.5)


def test_normalized_range_rejects_too_high():
    with pytest.raises(InvariantViolation):
        check_normalized_range(1.2)


# ======================================================================
# Tolerance scaling with n_episodes
# ======================================================================


def test_tolerance_shrinks_with_n_episodes():
    ceiling = policy_rate(MATRIX, BLIND_RULE)
    rate = ceiling + 0.004

    # At n=1000 the slack (~0.047) comfortably covers a +0.004 excess.
    check_static_policy_ceiling(rate, MATRIX, BLIND_RULE, n_episodes=1000)

    # At n=200_000 the slack (~0.0034) no longer covers it.
    with pytest.raises(InvariantViolation):
        check_static_policy_ceiling(rate, MATRIX, BLIND_RULE, n_episodes=200_000)


# ======================================================================
# Hard unit-interval bound
# ======================================================================


def test_rate_in_unit_interval_accepts_valid_rate():
    check_rate_in_unit_interval(0.5)


def test_rate_in_unit_interval_rejects_above_one():
    with pytest.raises(InvariantViolation):
        check_rate_in_unit_interval(1.01)


def test_rate_in_unit_interval_rejects_below_zero():
    with pytest.raises(InvariantViolation):
        check_rate_in_unit_interval(-0.01)


# ======================================================================
# run_all_guards: conditional check selection
# ======================================================================


def test_run_all_guards_skips_static_ceiling_and_confusion_when_pi_is_none(monkeypatch):
    calls = {"oracle": 0, "worst_case": 0, "unit_interval": 0, "static": 0, "confusion": 0}

    def spy(name, original):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(inv, "check_bounded_by_oracle", spy("oracle", inv.check_bounded_by_oracle))
    monkeypatch.setattr(
        inv, "check_above_worst_case", spy("worst_case", inv.check_above_worst_case)
    )
    monkeypatch.setattr(
        inv, "check_rate_in_unit_interval", spy("unit_interval", inv.check_rate_in_unit_interval)
    )
    monkeypatch.setattr(
        inv, "check_static_policy_ceiling", spy("static", inv.check_static_policy_ceiling)
    )
    monkeypatch.setattr(
        inv, "check_confusion_prediction", spy("confusion", inv.check_confusion_prediction)
    )

    inv.run_all_guards(0.5, MATRIX, n_episodes=200_000, pi=None, C=None)

    assert calls["oracle"] == 1
    assert calls["worst_case"] == 1
    assert calls["unit_interval"] == 1
    assert calls["static"] == 0
    assert calls["confusion"] == 0


def test_run_all_guards_runs_static_ceiling_but_not_confusion_when_only_pi_given(monkeypatch):
    calls = {"static": 0, "confusion": 0}
    monkeypatch.setattr(
        inv,
        "check_static_policy_ceiling",
        lambda *a, **kw: calls.__setitem__("static", calls["static"] + 1),
    )
    monkeypatch.setattr(
        inv,
        "check_confusion_prediction",
        lambda *a, **kw: calls.__setitem__("confusion", calls["confusion"] + 1),
    )

    inv.run_all_guards(0.5, MATRIX, n_episodes=200_000, pi=BLIND_RULE, C=None)

    assert calls["static"] == 1
    assert calls["confusion"] == 0


def test_run_all_guards_runs_confusion_when_both_pi_and_C_given(monkeypatch):
    calls = {"confusion": 0}
    monkeypatch.setattr(
        inv,
        "check_confusion_prediction",
        lambda *a, **kw: calls.__setitem__("confusion", calls["confusion"] + 1),
    )

    C = uniform_confusion(MATRIX.shape[0], 0.9)
    inv.run_all_guards(0.5, MATRIX, n_episodes=200_000, pi=BLIND_RULE, C=C)

    assert calls["confusion"] == 1


# ======================================================================
# check_confusion_prediction (two-sided, sigmas=4 default)
# ======================================================================


def test_confusion_prediction_accepts_close_value():
    C = uniform_confusion(MATRIX.shape[0], 0.939)
    check_confusion_prediction(0.715005, MATRIX, BLIND_RULE, C, n_episodes=200_000)


def test_confusion_prediction_rejects_far_value():
    C = uniform_confusion(MATRIX.shape[0], 0.939)
    prediction = policy_rate_under_confusion(MATRIX, BLIND_RULE, C)
    with pytest.raises(InvariantViolation):
        check_confusion_prediction(prediction + 0.05, MATRIX, BLIND_RULE, C, n_episodes=200_000)


def test_confusion_prediction_two_sided_rejects_far_below_too():
    C = uniform_confusion(MATRIX.shape[0], 0.939)
    prediction = policy_rate_under_confusion(MATRIX, BLIND_RULE, C)
    with pytest.raises(InvariantViolation):
        check_confusion_prediction(prediction - 0.05, MATRIX, BLIND_RULE, C, n_episodes=200_000)


# ======================================================================
# Exception message content: check name, observed, bound, tolerance, n_episodes
# ======================================================================


def test_exception_message_contains_required_fields_unit_interval():
    with pytest.raises(InvariantViolation) as exc_info:
        check_rate_in_unit_interval(1.5, n_episodes=100)
    message = str(exc_info.value)
    assert "check_rate_in_unit_interval" in message
    assert "1.5" in message
    assert "100" in message
    assert "tolerance" in message


def test_exception_message_contains_required_fields_oracle():
    with pytest.raises(InvariantViolation) as exc_info:
        check_bounded_by_oracle(0.99, MATRIX, n_episodes=200_000)
    message = str(exc_info.value)
    assert "check_bounded_by_oracle" in message
    assert "0.99" in message
    assert "200000" in message
    assert "tolerance" in message
    assert "oracle_rate" in message


def test_exception_message_contains_required_fields_worst_case():
    with pytest.raises(InvariantViolation) as exc_info:
        check_above_worst_case(0.01, MATRIX, n_episodes=200_000)
    message = str(exc_info.value)
    assert "check_above_worst_case" in message
    assert "0.01" in message
    assert "200000" in message
    assert "tolerance" in message


# ======================================================================
# Edge cases
# ======================================================================


def test_n_episodes_one_gives_extremely_weak_tolerance():
    # slack at sigmas=3, n_episodes=1 is 1.5 -- larger than any possible
    # rate deviation, so even a rate far from the ceiling still passes.
    # This documents the (correct) weakness of the guard at tiny n, not a bug.
    check_bounded_by_oracle(0.99, MATRIX, n_episodes=1)


def test_degenerate_matrix_oracle_equals_worst_case():
    P = np.full((3, 2), 0.5, dtype=np.float64)
    check_bounded_by_oracle(0.5, P, n_episodes=1000)
    check_above_worst_case(0.5, P, n_episodes=1000)


def test_invalid_policy_index_raises_value_error_not_invariant_violation():
    bad_pi = np.array([0, 1, 99, 3, 4, 5, 0])  # 99 is out of range
    with pytest.raises(ValueError):
        check_static_policy_ceiling(0.5, MATRIX, bad_pi, n_episodes=1000)


# ======================================================================
# Input validation: ValueError for malformed input, never InvariantViolation
# ======================================================================


def test_invalid_n_episodes_raises_value_error():
    with pytest.raises(ValueError):
        check_bounded_by_oracle(0.5, MATRIX, n_episodes=0)


def test_invalid_sigmas_raises_value_error():
    with pytest.raises(ValueError):
        check_bounded_by_oracle(0.5, MATRIX, n_episodes=1000, sigmas=-1.0)


def test_invalid_matrix_values_raise_value_error():
    bad_P = MATRIX * 2.0  # values > 1
    with pytest.raises(ValueError):
        check_above_worst_case(0.5, bad_P, n_episodes=1000)


def test_invalid_matrix_shape_raises_value_error():
    bad_P = MATRIX.reshape(-1)  # 1-D, not 2-D
    with pytest.raises(ValueError):
        check_above_worst_case(0.5, bad_P, n_episodes=1000)


def test_prior_not_summing_to_one_raises_value_error():
    bad_p = np.full(MATRIX.shape[0], 1.0 / MATRIX.shape[0])
    bad_p[0] += 0.1
    with pytest.raises(ValueError):
        check_above_worst_case(0.5, MATRIX, n_episodes=1000, p=bad_p)
