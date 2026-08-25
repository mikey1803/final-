"""Tests for the Task 13 real-sample significance module
(``recovery.analysis.significance``): Cochran's Q, the per-type-vs-rest
binomial post-hoc test, Cohen's h, and distinct-mapping counting."""

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.significance import (  # noqa: E402
    cochrans_q_test,
    cohens_h,
    n_distinct_mappings,
    per_type_vs_rest_binomial_tests,
)


# ======================================================================
# cochrans_q_test
# ======================================================================


def test_cochrans_q_uniform_data_is_not_significant():
    """All conditions equally likely (p=0.5 coin flips per cell, fixed
    seed) -- Q should be small and p-value should NOT reject uniformity."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 5)).astype(bool)
    result = cochrans_q_test(X)
    assert result["df"] == 4
    assert result["n"] == 200
    assert result["k"] == 5
    assert not result["degenerate"]
    assert result["p_value"] > 0.05


def test_cochrans_q_detects_a_real_per_column_effect():
    """Column 0 always succeeds, columns 1-4 always fail -- an extreme,
    obvious per-condition effect must be detected (tiny p-value)."""
    X = np.zeros((30, 5), dtype=bool)
    X[:, 0] = True
    result = cochrans_q_test(X)
    assert result["p_value"] < 1e-6
    assert result["effect_size_v"] > 0.5


def test_cochrans_q_all_identical_rows_is_degenerate_not_fabricated():
    """Every subject succeeds on every condition -- Q's denominator is
    zero (no within-subject variance at all); must report NaN/degenerate,
    never a fabricated p-value."""
    X = np.ones((10, 4), dtype=bool)
    result = cochrans_q_test(X)
    assert result["degenerate"] is True
    assert np.isnan(result["p_value"])
    assert np.isnan(result["Q"])


def test_cochrans_q_requires_at_least_2_subjects_and_2_conditions():
    with pytest.raises(ValueError):
        cochrans_q_test(np.ones((1, 5), dtype=bool))
    with pytest.raises(ValueError):
        cochrans_q_test(np.ones((5, 1), dtype=bool))


# ======================================================================
# cohens_h
# ======================================================================


def test_cohens_h_zero_when_proportions_equal():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)


def test_cohens_h_matches_known_reference_value():
    # Cohen's h between 0.5 and 0.7 -- textbook value ~= -0.4115 (sign
    # convention: h(p1, p2) with p1 < p2 is negative here).
    h = cohens_h(0.5, 0.7)
    assert h == pytest.approx(-0.4115168, abs=1e-4)


def test_cohens_h_antisymmetric():
    assert cohens_h(0.3, 0.8) == pytest.approx(-cohens_h(0.8, 0.3))


# ======================================================================
# per_type_vs_rest_binomial_tests
# ======================================================================


def test_per_type_vs_rest_flags_the_one_deficient_column():
    """Column 0 succeeds 20% of the time, columns 1-5 succeed 90% of the
    time (5 "normal" columns dilute the pooling effect of the one
    outlier) -- column 0 must have the most extreme p-value and Cohen's h
    of all columns, and a negative h (below its pooled rest rate)."""
    rng = np.random.default_rng(1)
    n = 60
    k = 6
    X = np.zeros((n, k), dtype=bool)
    X[:, 0] = rng.random(n) < 0.2
    for j in range(1, k):
        X[:, j] = rng.random(n) < 0.9
    rows = per_type_vs_rest_binomial_tests(X, ["deficient", "b", "c", "d", "e", "f"])
    assert len(rows) == k
    deficient = next(r for r in rows if r["type_name"] == "deficient")
    others = [r for r in rows if r["type_name"] != "deficient"]
    assert deficient["p_value"] < 0.01
    assert deficient["cohens_h"] < 0
    assert deficient["p_value"] < min(r["p_value"] for r in others)
    assert abs(deficient["cohens_h"]) > max(abs(r["cohens_h"]) for r in others)


def test_per_type_vs_rest_undefined_when_pooled_rest_rate_is_boundary():
    """If every OTHER column is always correct, the pooled reference rate
    for the remaining column is exactly 1.0 -- binomtest is undefined at
    that boundary; must report NaN, not a fabricated p-value."""
    X = np.ones((10, 3), dtype=bool)
    X[:, 0] = False  # column 0 always fails; columns 1-2 always succeed
    rows = per_type_vs_rest_binomial_tests(X, ["a", "b", "c"])
    row_a = next(r for r in rows if r["type_name"] == "a")
    assert row_a["pooled_rest_rate"] == pytest.approx(1.0)
    assert np.isnan(row_a["p_value"])


def test_per_type_vs_rest_column_count_mismatch_raises():
    X = np.ones((5, 3), dtype=bool)
    with pytest.raises(ValueError):
        per_type_vs_rest_binomial_tests(X, ["a", "b"])


# ======================================================================
# n_distinct_mappings
# ======================================================================


def test_n_distinct_mappings_counts_unique_tuples():
    mappings = [[1, 2, 3], [1, 2, 3], [1, 2, 4], [1, 2, 3]]
    assert n_distinct_mappings(mappings) == 2


def test_n_distinct_mappings_all_identical_is_one():
    mappings = [[0, 0, 0]] * 21
    assert n_distinct_mappings(mappings) == 1


def test_n_distinct_mappings_all_unique():
    mappings = [[i, 0, 0] for i in range(5)]
    assert n_distinct_mappings(mappings) == 5


def test_n_distinct_mappings_matches_google_gemini_real_case():
    """Real Task 11 pattern: 15 identical + 6 differing in exactly one
    position -> 2 distinct mappings."""
    base = [0, 1, 2, 3, 4, 0, 5]
    variant = [0, 1, 2, 3, 1, 0, 5]  # differs only at malformed_output's index
    mappings = [base] * 15 + [variant] * 6
    assert n_distinct_mappings(mappings) == 2
