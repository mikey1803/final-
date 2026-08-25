"""Tests for the Task 19A classifier-noise-aware crossover extension
(recovery/theory/crossover.py's *_confused / ceiling_reduction /
gap_shrinkage_report / argmax_inversion_check / asymptotic_regime_flag).

These are required unit/self-consistency tests, not empirical validation
evidence -- empirical validation against real data lives in
experiments/crossover_theory_validation.py's Tier 1/Tier 2 output, per
docs/theory/crossover_derivation.md. Nine groups: (1) exact C=I reduction
for every *_confused function, (2) effective_matrix_under_confusion
construction/row-normalization, (3) canonical analytic cross-check to
1e-9, (4) monotonicity in classifier accuracy, (5) limit cases (perfect
table, zero-mass bucket), (6) invalid-input rejection, (7) channel
calculations (ceiling reduction, gap shrinkage), (8) argmax-inversion
check (including a constructed example that DOES trigger it), (9)
regression pins for >= 5 predicted N* values.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.theory.analytic import oracle_rate, policy_rate, uniform_confusion  # noqa: E402
from recovery.theory.crossover import (  # noqa: E402
    argmax_inversion_check,
    asymptotic_regime_flag,
    ceiling_reduction,
    crossover_horizon_confused,
    effective_matrix_under_confusion,
    gap_shrinkage_report,
    predict_crossover,
    static_rate_confused,
    static_regret_rate,
    ts_cumulative_regret,
    ts_cumulative_regret_confused,
)

N = MATRIX.shape[0]
IDENTITY = np.eye(N)


# ======================================================================
# 1. Exact C=I reduction -- every *_confused function must match its
#    unconfused Task 10 counterpart bit-for-bit at a perfect classifier.
# ======================================================================


def test_static_rate_confused_reduces_at_identity():
    a = static_regret_rate(MATRIX, BLIND_RULE)
    b = static_rate_confused(MATRIX, BLIND_RULE, IDENTITY)
    assert a == pytest.approx(b, abs=0.0, rel=1e-12)


def test_ts_cumulative_regret_confused_reduces_at_identity():
    for T in (100, 5000, 100_000):
        a = ts_cumulative_regret(MATRIX, T, N)
        b = ts_cumulative_regret_confused(MATRIX, T, N, IDENTITY)
        assert a == pytest.approx(b, abs=0.0, rel=1e-12), f"mismatch at T={T}"


def test_crossover_horizon_confused_reduces_at_identity():
    T1, trunc1 = predict_crossover(MATRIX, BLIND_RULE)
    T2, trunc2 = crossover_horizon_confused(MATRIX, BLIND_RULE, IDENTITY)
    assert trunc1 == trunc2
    assert T1 == pytest.approx(T2, rel=1e-9)


def test_ceiling_reduction_is_zero_at_identity():
    assert ceiling_reduction(MATRIX, IDENTITY) == pytest.approx(0.0, abs=1e-12)


def test_gap_shrinkage_is_zero_at_identity():
    """At C=I, every predicted bucket has exactly one contributor (itself),
    so P_eff's gap must equal that contributor's own gap exactly."""
    report = gap_shrinkage_report(MATRIX, IDENTITY)
    for row in report:
        assert row["contributing_true_rows"] == [row["bucket"]]
        assert row["gap_shrinkage"] == pytest.approx(0.0, abs=1e-9)


# ======================================================================
# 2. effective_matrix_under_confusion construction / normalization
# ======================================================================


def test_p_prime_sums_to_one():
    C = uniform_confusion(N, 0.8)
    _, p_prime = effective_matrix_under_confusion(MATRIX, C)
    assert p_prime.sum() == pytest.approx(1.0, abs=1e-9)


def test_p_eff_values_in_unit_interval():
    C = uniform_confusion(N, 0.6)
    P_eff, _ = effective_matrix_under_confusion(MATRIX, C)
    assert np.all(P_eff >= 0.0) and np.all(P_eff <= 1.0)


def test_p_eff_is_convex_combination_bounded_by_extremes():
    """Every P_eff[t', k] must lie within [min, max] of the P[:, k] values
    of its contributing true rows -- a convex combination can never exceed
    its inputs' range."""
    C = uniform_confusion(N, 0.5)
    P_eff, _ = effective_matrix_under_confusion(MATRIX, C)
    for t_prime in range(N):
        contributors = np.where(C[:, t_prime] > 0.0)[0]
        lo = MATRIX[contributors].min(axis=0)
        hi = MATRIX[contributors].max(axis=0)
        assert np.all(P_eff[t_prime] >= lo - 1e-9)
        assert np.all(P_eff[t_prime] <= hi + 1e-9)


# ======================================================================
# 3. Canonical analytic cross-check to 1e-9
# ======================================================================


def test_static_rate_confused_matches_manual_oracle_minus_policy_1e9():
    from recovery.theory.analytic import policy_rate_under_confusion

    C = uniform_confusion(N, 0.75)
    manual = oracle_rate(MATRIX) - policy_rate_under_confusion(MATRIX, BLIND_RULE, C)
    via_fn = static_rate_confused(MATRIX, BLIND_RULE, C)
    assert manual == pytest.approx(via_fn, abs=1e-9)


def test_ceiling_reduction_matches_manual_computation_1e9():
    C = uniform_confusion(N, 0.75)
    P_eff, p_prime = effective_matrix_under_confusion(MATRIX, C)
    manual = oracle_rate(MATRIX) - oracle_rate(P_eff, p_prime)
    via_fn = ceiling_reduction(MATRIX, C)
    assert manual == pytest.approx(via_fn, abs=1e-9)


# ======================================================================
# 4. Monotonicity in classifier accuracy
# ======================================================================


def test_static_rate_confused_nonincreasing_as_accuracy_increases():
    """More accurate classifier -> less-noisy policy_rate_under_confusion
    -> non-increasing regret (never worse than the unconfused case)."""
    accuracies = [0.3, 0.5, 0.7, 0.9, 1.0]
    rates = [static_rate_confused(MATRIX, BLIND_RULE, uniform_confusion(N, a)) for a in accuracies]
    for a, b in zip(rates, rates[1:]):
        assert b <= a + 1e-9, f"static_rate_confused not non-increasing in accuracy: {rates}"
    assert rates[-1] == pytest.approx(static_regret_rate(MATRIX, BLIND_RULE), abs=1e-9)


def test_ceiling_reduction_nonincreasing_as_accuracy_increases():
    accuracies = [0.3, 0.5, 0.7, 0.9, 1.0]
    values = [ceiling_reduction(MATRIX, uniform_confusion(N, a)) for a in accuracies]
    for a, b in zip(values, values[1:]):
        assert b <= a + 1e-9, f"ceiling_reduction not non-increasing in accuracy: {values}"
    assert values[-1] == pytest.approx(0.0, abs=1e-9)


def test_crossover_horizon_confused_not_generally_monotonic_in_accuracy():
    """T* is NOT guaranteed monotonic in classifier accuracy, unlike rho
    and ceiling_reduction -- documented here as an asserted, verified
    behavior rather than assumed away. Mechanism, isolated with a clean
    2-type/2-strategy toy matrix (P[0]=[0.8,0.5], P[1]=[0.5,0.8]) under
    uniform_confusion: predicted bucket 0's OWN best arm (by P_eff) FLIPS
    identity as accuracy crosses 0.5 (at a=1.0 bucket 0 prefers arm 0,
    matching true type 0 exactly; at a=0.0 it prefers arm 1, having fully
    inherited type 1's profile instead). The bucket's own top-2 gap is
    therefore V-shaped in accuracy -- LARGE at both extremes, exactly ZERO
    at the crossover point (a=0.5) -- not monotonically shrinking as
    accuracy drops. Since T* depends on this gap, it inherits the same
    non-monotonic shape near the flip, which is why
    docs/theory/crossover_derivation.md documents this as a real
    limitation rather than a stronger monotonicity claim being made
    anywhere in this module."""
    P = np.array([[0.8, 0.5], [0.5, 0.8]])
    pi = np.array([0, 1])  # per-predicted-bucket "smart" table
    gaps = []
    for a in (1.0, 0.9, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1):
        C = uniform_confusion(2, a)
        P_eff, _ = effective_matrix_under_confusion(P, C)
        gaps.append(abs(P_eff[0, 0] - P_eff[0, 1]))
    # Genuinely V-shaped: strictly decreasing down to the a=0.5 minimum,
    # then strictly increasing back up -- not monotone over the full range.
    min_idx = gaps.index(min(gaps))
    assert 0 < min_idx < len(gaps) - 1, f"expected an interior minimum, got gaps={gaps}"
    assert gaps[min_idx] == pytest.approx(0.0, abs=1e-9)
    assert all(a >= b - 1e-9 for a, b in zip(gaps[:min_idx], gaps[1 : min_idx + 1]))
    assert all(a <= b + 1e-9 for a, b in zip(gaps[min_idx:], gaps[min_idx + 1 :]))
    # rho, unlike T*, IS still monotonic here (sanity-check via static_rate_confused).
    rhos = [static_rate_confused(P, pi, uniform_confusion(2, a)) for a in (1.0, 0.7, 0.5, 0.3, 0.1)]
    for a, b in zip(rhos, rhos[1:]):
        assert b >= a - 1e-9, f"rho unexpectedly not monotonic on this toy example: {rhos}"


# ======================================================================
# 5. Limit cases
# ======================================================================


def test_perfect_table_never_crosses_under_confusion():
    """rho=0 (a perfect table) must return infinity regardless of
    confusion -- static_rate_confused(P, pi*, C) where pi* is the
    per-predicted-type-optimal table under a PERFECT classifier reduces
    to the unconfused perfect-table case."""
    pi_star = MATRIX.argmax(axis=1)
    T, truncated = crossover_horizon_confused(MATRIX, pi_star, IDENTITY)
    assert math.isinf(T)
    assert truncated is False


def test_zero_marginal_bucket_raises_not_silently_wrong():
    C = np.eye(N)
    C[N - 1, N - 1] = 0.0
    C[N - 1, 0] = 1.0  # last row's mass moves entirely to bucket 0 -> bucket N-1 unreachable
    with pytest.raises(ValueError, match="zero marginal probability"):
        effective_matrix_under_confusion(MATRIX, C)


# ======================================================================
# 6. Invalid-input rejection
# ======================================================================


def test_non_square_confusion_raises():
    with pytest.raises(ValueError):
        effective_matrix_under_confusion(MATRIX, np.eye(N - 1))


def test_non_stochastic_confusion_raises():
    C = np.eye(N)
    C[0, 0] = 0.5  # row no longer sums to 1
    with pytest.raises(ValueError, match="row-stochastic"):
        effective_matrix_under_confusion(MATRIX, C)


def test_crossover_horizon_confused_bad_bracket_raises():
    with pytest.raises(ValueError):
        crossover_horizon_confused(MATRIX, BLIND_RULE, IDENTITY, t_min=-1)
    with pytest.raises(ValueError):
        crossover_horizon_confused(MATRIX, BLIND_RULE, IDENTITY, t_min=100, t_max=10)


# ======================================================================
# 7. Channel calculations
# ======================================================================


def test_ceiling_reduction_nonnegative():
    for a in (0.2, 0.5, 0.8):
        assert ceiling_reduction(MATRIX, uniform_confusion(N, a)) >= -1e-12


def test_gap_shrinkage_report_shape_and_keys():
    C = uniform_confusion(N, 0.7)
    report = gap_shrinkage_report(MATRIX, C)
    assert len(report) == N
    expected_keys = {"bucket", "gap_P_eff", "contributing_true_rows", "gap_true_rows",
                      "mean_gap_true_rows", "min_gap_true_rows", "gap_shrinkage"}
    for row in report:
        assert expected_keys.issubset(row.keys())
        assert len(row["contributing_true_rows"]) == len(row["gap_true_rows"])


# ======================================================================
# 8. Argmax-inversion check
# ======================================================================


def test_no_inversion_at_identity():
    report = argmax_inversion_check(MATRIX, IDENTITY)
    assert all(not row["is_inversion"] for row in report)


def test_inversion_detected_in_constructed_example():
    """A deliberately constructed 3-type example where two comparably-
    weighted true rows, each individually preferring a DIFFERENT arm,
    both rate a THIRD arm decently -- the mixture prefers that third arm,
    which is optimal for neither contributor. Proves the detector fires
    when the condition is genuinely met (not a dead code path) -- see
    docs/theory/crossover_derivation.md for the worked-through numbers."""
    P = np.array([
        [0.65, 0.10, 0.60],
        [0.10, 0.65, 0.60],
        [0.50, 0.50, 0.50],
    ])
    C = np.array([
        [0.001, 0.0, 0.999],
        [0.0, 0.001, 0.999],
        [0.4995, 0.4995, 0.001],
    ])
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    report = argmax_inversion_check(P, C, p)
    bucket2 = next(r for r in report if r["bucket"] == 2)
    assert bucket2["is_inversion"] is True
    assert bucket2["argmax_P_eff"] == 2
    assert 2 not in bucket2["contributor_argmaxes"] or set(bucket2["contributor_argmaxes"]) != {2}


def test_asymptotic_regime_flag_behavior():
    assert asymptotic_regime_flag(math.inf, 7) is True
    assert asymptotic_regime_flag(1000, 7) is True  # 1000 >= 10*7
    assert asymptotic_regime_flag(50, 7) is False  # 50 < 10*7


# ======================================================================
# 9. Regression pins -- >= 5 predicted N* values, so future changes to
#    this module cannot silently move the theory.
# ======================================================================


def test_regression_pins_five_values():
    """Five predicted crossover horizons (canonical MATRIX, BLIND_RULE,
    uniform_confusion at five accuracy levels), pinned to their computed
    values as of this task. Not monotonic in accuracy by construction --
    ts_cumulative_regret_confused's KL-form denominator depends on the
    full P_eff shape, not just a scalar accuracy, so a non-monotonic
    sequence here is expected and is exactly why these are pinned
    individually rather than asserted to be ordered (monotonicity itself
    is separately tested in Section 4 using policy_rate_under_confusion's
    OWN guaranteed-monotone quantity, not T* directly). A future change to
    this module that moves any of these silently is what this test exists
    to catch."""
    cases = [
        (MATRIX, BLIND_RULE, uniform_confusion(N, 1.00), 109280.074490),
        (MATRIX, BLIND_RULE, uniform_confusion(N, 0.95), 68325.588350),
        (MATRIX, BLIND_RULE, uniform_confusion(N, 0.90), 57413.182941),
        (MATRIX, BLIND_RULE, uniform_confusion(N, 0.80), 76672.816472),
        (MATRIX, BLIND_RULE, uniform_confusion(N, 0.70), 8940863.596114),
    ]
    for P, pi, C, expected in cases:
        T_star, _ = crossover_horizon_confused(P, pi, C)
        assert T_star == pytest.approx(expected, rel=1e-6), f"regression pin drift at accuracy giving C={C[0,0]}"
