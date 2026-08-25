"""Tests for the analytic crossover predictor (recovery/theory/crossover.py).

Eight groups, matching Task 10's required test list exactly: (1)
monotonicity across quality/gap/K/N, (2) perfect table -> infinity, (3)
completely wrong table -> small finite crossover, (4)
``static_regret_rate`` == ``oracle_rate - policy_rate`` exactly, (5)
solver correctness (``R_TS(T*) ~= rho*T*`` within 1%), (6) bracketing
failures never raise, (7) KL vs. gap form agreement/divergence by gap
size, (8) the calibration pipeline's output schema.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from recovery.theory.analytic import oracle_rate, policy_rate  # noqa: E402
from recovery.theory.crossover import (  # noqa: E402
    predict_crossover,
    predict_crossover_grid,
    static_regret_rate,
    ts_cumulative_regret,
)

# ======================================================================
# 1. Monotonicity
# ======================================================================


def _perfect_table_matrix(n_types: int, n_strats: int = 3) -> np.ndarray:
    """A matrix where column 0 is always the best arm by a fixed 0.3 gap,
    identical in every row -- lets tests control quality/gap/K/N by
    construction rather than random sampling."""
    row = np.array([0.80] + [0.50] * (n_strats - 1), dtype=np.float64)
    return np.tile(row, (n_types, 1))


def test_monotonic_in_quality():
    """More correct rows (lower rho) -> non-decreasing T*."""
    n_types = 6
    P = _perfect_table_matrix(n_types)
    T_stars = []
    for n_correct in range(0, n_types + 1):
        pi = np.array([0 if t < n_correct else 1 for t in range(n_types)])
        T_star, _ = predict_crossover(P, pi)
        T_stars.append(T_star)
    for a, b in zip(T_stars, T_stars[1:]):
        assert b >= a - 1e-6, f"T* not non-decreasing in quality: {T_stars}"
    # the fully-correct (q=n_types) table must be infinite; a strictly
    # worse one must be finite -- otherwise the ordering above is vacuous.
    assert math.isinf(T_stars[-1])
    assert math.isfinite(T_stars[0])


def test_monotonic_in_gap():
    """Larger optimal-arm gap on the (single) wrong row -> larger rho ->
    non-increasing T*."""
    n_types = 5
    gaps = [0.05, 0.10, 0.20, 0.30, 0.40]
    T_stars = []
    for gap in gaps:
        P = np.tile(np.array([0.50 + gap, 0.50]), (n_types, 1))
        pi = np.array([0] * (n_types - 1) + [1])  # last row wrong, gap=`gap`
        T_star, _ = predict_crossover(P, pi)
        T_stars.append(T_star)
    for a, b in zip(T_stars, T_stars[1:]):
        assert b <= a + 1e-6, f"T* not non-increasing in gap: {T_stars}"
    assert T_stars[0] > T_stars[-1]  # genuinely different, not a flat tie


def test_monotonic_in_K():
    """Appending extra, strictly-suboptimal arms (never touching the
    optimal value or the chosen wrong-row action) leaves rho unchanged but
    adds terms to the regret sum -> non-decreasing T*."""
    n_types = 5
    base = np.tile(np.array([0.80, 0.50]), (n_types, 1))  # col 0 optimal
    pi = np.array([0] * (n_types - 1) + [1])  # last row plays col 1 (wrong)

    T_stars = []
    P = base.copy()
    rho_ref = static_regret_rate(P, pi)
    for extra_col_value in [0.40, 0.30, 0.20, 0.10]:
        new_col = np.full((n_types, 1), extra_col_value)
        P = np.hstack([P, new_col])
        assert static_regret_rate(P, pi) == pytest.approx(rho_ref)  # rho held fixed
        T_star, _ = predict_crossover(P, pi)
        T_stars.append(T_star)

    for a, b in zip(T_stars, T_stars[1:]):
        assert b >= a - 1e-6, f"T* not non-decreasing in K: {T_stars}"
    assert T_stars[-1] > T_stars[0]


def test_monotonic_in_N():
    """Replicating an identical (type, table-quality) block leaves rho
    unchanged (uniform average of identical numbers) while N grows ->
    non-decreasing T*.

    ``n_repeats`` is kept strictly below the default ``t_min=10``: once
    ``N >= t_min``, ``log(t_min/N)`` clips to 0 (see
    ``ts_cumulative_regret``'s "Small T" edge case), and the search floor
    itself sits outside the asymptotic regime -- ``predict_crossover``
    then floors to ``t_min`` regardless of table quality, by design (see
    the module docstring's "N must stay below t_min" note), which is not
    a meaningful regime to assert monotonicity in.
    """
    block = np.array([[0.80, 0.50]])  # 1 type, optimal=col0
    pi_block = np.array([1])  # this one type plays the wrong arm

    T_stars = []
    for n_repeats in [1, 2, 4, 8]:
        P = np.tile(block, (n_repeats, 1))
        pi = np.tile(pi_block, n_repeats)
        assert static_regret_rate(P, pi) == pytest.approx(static_regret_rate(block, pi_block))
        T_star, _ = predict_crossover(P, pi)
        T_stars.append(T_star)

    for a, b in zip(T_stars, T_stars[1:]):
        assert b >= a - 1e-6, f"T* not non-decreasing in N: {T_stars}"
    assert T_stars[-1] > T_stars[0]


def test_monotonicity_grid_via_predict_crossover_grid():
    """Same qualitative checks, but exercised through the public
    ``predict_crossover_grid`` sweep (sampled matrices) rather than
    hand-built ones -- averaged over many draws per cell to smooth
    per-sample sampling noise."""
    rows = predict_crossover_grid(
        quality_levels=[3, 4, 5, 6, 7], gaps=[0.05, 0.20], K=6, N=7, n_matrices=40, seed=123
    )
    df = pd.DataFrame(rows)

    for gap in [0.05, 0.20]:
        sub = df[df["gap"] == gap]
        means = []
        for q in [3, 4, 5, 6, 7]:
            vals = sub[sub["quality"] == q]["T_star"].to_numpy()
            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite)) if finite.size else math.inf)
        for a, b in zip(means, means[1:]):
            assert b >= a - 1.0, f"gap={gap}: mean T* not non-decreasing in quality: {means}"

    for q in [3, 5]:
        sub = df[df["quality"] == q]
        means = []
        for gap in [0.05, 0.20]:
            vals = sub[sub["gap"] == gap]["T_star"].to_numpy()
            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite)) if finite.size else math.inf)
        assert means[1] <= means[0] + 1.0, f"quality={q}: mean T* not non-increasing in gap: {means}"


# ======================================================================
# 2. Perfect table -> infinity
# ======================================================================


def test_perfect_table_returns_infinity():
    n_types = 7
    P = _perfect_table_matrix(n_types)
    pi = np.zeros(n_types, dtype=np.int64)  # always plays the (unique) best arm
    assert static_regret_rate(P, pi) == pytest.approx(0.0, abs=1e-12)
    T_star, truncated = predict_crossover(P, pi)
    assert math.isinf(T_star)
    assert truncated is False


# ======================================================================
# 3. Completely wrong table -> small finite crossover
# ======================================================================


def test_completely_wrong_table_returns_small_finite_crossover():
    n_types = 7
    P = np.tile(np.array([0.90, 0.10]), (n_types, 1))
    pi = np.ones(n_types, dtype=np.int64)  # always plays the worst arm, huge gap
    rho = static_regret_rate(P, pi)
    assert rho == pytest.approx(0.80)
    T_star, truncated = predict_crossover(P, pi, t_min=10, t_max=1e9)
    assert math.isfinite(T_star)
    assert T_star == pytest.approx(10.0)  # already past crossover at t_min
    assert truncated is False

    # contrast: a much better table must predict a much larger crossover
    good_pi = np.zeros(n_types, dtype=np.int64)
    good_pi[0] = 1  # only one wrong row
    T_star_good, _ = predict_crossover(P, good_pi)
    assert T_star_good > T_star


# ======================================================================
# 4. static_regret_rate must equal oracle_rate - policy_rate exactly
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_static_regret_rate_matches_analytic_functions_exactly(seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.05, 0.95, size=(7, 6))
    pi = rng.integers(0, 6, size=7)
    expected = oracle_rate(P) - policy_rate(P, pi)
    assert static_regret_rate(P, pi) == expected  # bit-identical, not just approx


# ======================================================================
# 5. Solver correctness: R_TS(T*) ~= rho * T* within 1%
# ======================================================================


@pytest.mark.parametrize("form", ["kl", "gap"])
@pytest.mark.parametrize("n_correct", [2, 4, 5])
def test_solver_satisfies_crossover_equation_within_1_percent(form, n_correct):
    n_types = 7
    P = _perfect_table_matrix(n_types)
    pi = np.array([0 if t < n_correct else 1 for t in range(n_types)])
    rho = static_regret_rate(P, pi)
    T_star, truncated = predict_crossover(P, pi, form=form)
    assert not truncated
    assert math.isfinite(T_star)
    assert T_star > 10.0  # a genuine in-bracket root, not the t_min edge case

    lhs = ts_cumulative_regret(P, T_star, n_types, form=form)
    rhs = rho * T_star
    assert lhs == pytest.approx(rhs, rel=0.01)


# ======================================================================
# 6. Bracketing failures never raise
# ======================================================================


def test_bracketing_never_raises_extremely_poor_table():
    n_types = 5
    P = np.tile(np.array([0.95, 0.05]), (n_types, 1))
    pi = np.ones(n_types, dtype=np.int64)
    T_star, truncated = predict_crossover(P, pi, t_min=10, t_max=1e9)
    assert T_star == pytest.approx(10.0)
    assert truncated is False


def test_bracketing_never_raises_never_crosses_within_range():
    n_types = 7
    P = _perfect_table_matrix(n_types)
    pi = np.array([0] * (n_types - 1) + [1])  # single, barely-wrong row
    # a tiny bracket that cannot possibly contain the (huge) true root
    T_star, truncated = predict_crossover(P, pi, t_min=10, t_max=11)
    assert T_star == pytest.approx(11.0)
    assert truncated is True


def test_predict_crossover_never_raises_on_degenerate_matrix():
    """Every row ties (no non-optimal arm at all) -- rho == 0 exactly, must
    resolve to infinity, not a division-by-zero exception."""
    n_types = 4
    P = np.full((n_types, 3), 0.5)
    pi = np.zeros(n_types, dtype=np.int64)
    T_star, truncated = predict_crossover(P, pi)
    assert math.isinf(T_star)
    assert truncated is False


def test_predict_crossover_rejects_invalid_bracket_but_does_not_crash_solver():
    n_types = 5
    P = np.tile(np.array([0.80, 0.50]), (n_types, 1))
    pi = np.array([0] * (n_types - 1) + [1])
    with pytest.raises(ValueError):
        predict_crossover(P, pi, t_min=0)
    with pytest.raises(ValueError):
        predict_crossover(P, pi, t_min=100, t_max=10)


# ======================================================================
# 7. KL vs. gap form: agree at large gaps, diverge at small gaps
# ======================================================================


def test_kl_and_gap_forms_agree_within_3x_at_large_gap():
    n_types = 7
    gap = 0.30
    P = np.tile(np.array([0.50 + gap, 0.50]), (n_types, 1))
    pi = np.array([0] * (n_types - 1) + [1])
    T_kl, _ = predict_crossover(P, pi, form="kl")
    T_gap, _ = predict_crossover(P, pi, form="gap")
    assert math.isfinite(T_kl) and math.isfinite(T_gap)
    ratio = max(T_kl, T_gap) / min(T_kl, T_gap)
    assert ratio <= 3.0, f"KL/gap forms disagree by {ratio:.2f}x at gap={gap} (expected <= 3x)"


def test_kl_and_gap_forms_diverge_materially_at_small_gap():
    n_types = 7
    gap = 0.05
    P = np.tile(np.array([0.50 + gap, 0.50]), (n_types, 1))
    pi = np.array([0] * (n_types - 1) + [1])
    T_kl, _ = predict_crossover(P, pi, form="kl")
    T_gap, _ = predict_crossover(P, pi, form="gap")
    assert math.isfinite(T_kl) and math.isfinite(T_gap)
    ratio = max(T_kl, T_gap) / min(T_kl, T_gap)
    # documented, expected behaviour (see ts_cumulative_regret's docstring):
    # KL(p, p*) ~ gap^2 / (2 p*(1-p*)) for small gaps, so the two forms'
    # 1/denominator terms diverge by roughly a factor of gap / (2 p*(1-p*)).
    assert ratio > 3.0, f"expected material divergence at gap={gap}, got ratio={ratio:.2f}x"


# ======================================================================
# 8. Calibration pipeline schema
# ======================================================================


def test_calibration_pipeline_schema_on_synthetic_data(tmp_path):
    import crossover_calibration as cc

    n_types, n_strats = 7, 6
    rng = np.random.default_rng(7)
    synthetic_rows = []
    for i in range(12):
        P = rng.uniform(0.15, 0.85, size=(n_types, n_strats))
        best = P.argmax(axis=1)
        n_correct = rng.integers(2, n_types + 1)
        correct_rows = set(rng.choice(n_types, size=int(n_correct), replace=False).tolist())
        pi = np.array(
            [
                best[t] if t in correct_rows else int((best[t] + 1) % n_strats)
                for t in range(n_types)
            ]
        )
        rho = static_regret_rate(P, pi)
        T_true, truncated = predict_crossover(P, pi, form="kl", c=2.5)
        observed_T = T_true if math.isfinite(T_true) else 1e9
        synthetic_rows.append(
            {
                "env_id": f"synthetic_{i:02d}",
                "P": P,
                "pi": pi,
                "gap": float(rng.choice([0.05, 0.10, 0.20, 0.30])),
                "quality": int(n_correct),
                "N": n_types,
                "K": n_strats,
                "rho": rho,
                "observed_T": float(observed_T),
                "fraction_never_crossed": 0.0,
                "n_seeds": 20,
            }
        )

    summary = cc.build_calibration_summary(synthetic_rows, form="kl")
    per_cell = cc.predicted_ratio_table(synthetic_rows, form="kl", c=1.0)

    assert "ratio" in per_cell.columns
    for col in [
        "median",
        "iqr",
        "within_factor2",
        "within_factor3",
        "best_fit_c",
        "validated_region",
    ]:
        assert col in summary.columns, f"missing required calibration summary column {col!r}"
    assert len(summary) > 0
    assert (summary["best_fit_c"] > 0).all()
