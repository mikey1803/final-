"""Tests for the Task 12 matrix-perturbation utilities
(``recovery.envs.matrix_perturbation``): Gaussian noise, row-entry
permutation (rank-preserving vs. rank-breaking), the ceiling-hardest-type
helper, and the three hand-authored structured matrices (Scheme 3)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.envs.canonical import BLIND_RULE, MATRIX, TYPE_NAMES  # noqa: E402
from recovery.envs.matrix_perturbation import (  # noqa: E402
    gaussian_perturb,
    hardest_type_by_ceiling,
    permute_row_entries,
    structured_malformed_not_hardest,
    structured_single_dominant_strategy,
    structured_uniform_difficulty,
)
from recovery.theory.analytic import oracle_rate, policy_rate  # noqa: E402


# ======================================================================
# gaussian_perturb
# ======================================================================


def test_gaussian_perturb_sigma_zero_is_identity():
    rng = np.random.default_rng(0)
    out = gaussian_perturb(MATRIX, sigma=0.0, rng=rng)
    assert np.array_equal(out, MATRIX)
    assert out is not MATRIX  # defensive copy, not the same object


def test_gaussian_perturb_stays_in_unit_interval():
    rng = np.random.default_rng(1)
    for sigma in (0.05, 0.10, 0.15, 0.20, 0.30, 1.0):
        out = gaussian_perturb(MATRIX, sigma=sigma, rng=rng)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        assert out.shape == MATRIX.shape


def test_gaussian_perturb_does_not_mutate_input():
    rng = np.random.default_rng(2)
    original = MATRIX.copy()
    gaussian_perturb(MATRIX, sigma=0.2, rng=rng)
    assert np.array_equal(MATRIX, original)


def test_gaussian_perturb_negative_sigma_raises():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        gaussian_perturb(MATRIX, sigma=-0.1, rng=rng)


def test_gaussian_perturb_deterministic_given_seeded_rng():
    out1 = gaussian_perturb(MATRIX, sigma=0.1, rng=np.random.default_rng(42))
    out2 = gaussian_perturb(MATRIX, sigma=0.1, rng=np.random.default_rng(42))
    assert np.array_equal(out1, out2)


def test_gaussian_perturb_large_sigma_produces_boundary_clipping():
    rng = np.random.default_rng(4)
    out = gaussian_perturb(MATRIX, sigma=2.0, rng=rng)
    # With sigma this large relative to [0,1], some entries must clip.
    assert np.any(out == 0.0) or np.any(out == 1.0)


# ======================================================================
# permute_row_entries
# ======================================================================


def test_permute_row_entries_fraction_zero_is_noop():
    rng = np.random.default_rng(0)
    out = permute_row_entries(MATRIX, fraction=0.0, mode="rank_breaking", rng=rng)
    assert np.array_equal(out, MATRIX)


def test_permute_row_entries_preserves_row_multiset():
    rng = np.random.default_rng(5)
    for mode in ("rank_preserving", "rank_breaking"):
        out = permute_row_entries(MATRIX, fraction=1.0, mode=mode, rng=rng)
        for t in range(MATRIX.shape[0]):
            assert sorted(out[t].tolist()) == pytest.approx(sorted(MATRIX[t].tolist()))


def test_permute_row_entries_rank_preserving_never_moves_the_argmax_value():
    rng = np.random.default_rng(6)
    for fraction in (0.1, 0.25, 0.5, 1.0):
        out = permute_row_entries(MATRIX, fraction=fraction, mode="rank_preserving", rng=rng)
        for t in range(MATRIX.shape[0]):
            original_argmax = int(np.argmax(MATRIX[t]))
            assert out[t, original_argmax] == MATRIX[t, original_argmax], (
                f"rank_preserving must never move row {t}'s argmax value "
                f"(fraction={fraction})"
            )


def test_permute_row_entries_low_fraction_is_noop_at_k_equals_6():
    """Documented edge case: round(0.1 * 6) = 1 selected position, and a
    single position can't be permuted against anything -- fraction=0.1 is
    a de facto no-op for this project's K=6 action space."""
    rng = np.random.default_rng(7)
    out = permute_row_entries(MATRIX, fraction=0.1, mode="rank_breaking", rng=rng)
    assert np.array_equal(out, MATRIX)


def test_permute_row_entries_fraction_quarter_actually_permutes_something():
    rng = np.random.default_rng(8)
    # round(0.25 * 6) = 2 -- enough to swap. Not guaranteed to differ from
    # MATRIX for every seed (a random 2-permutation can be the identity
    # with probability 1/2), so try a few seeds and require at least one
    # to differ.
    changed = False
    for seed in range(20):
        out = permute_row_entries(MATRIX, fraction=0.25, mode="rank_breaking", rng=np.random.default_rng(seed))
        if not np.array_equal(out, MATRIX):
            changed = True
            break
    assert changed


def test_permute_row_entries_rank_breaking_can_change_the_optimal_strategy():
    """Rank-breaking permutation, at fraction=1.0, must be ABLE to change
    which strategy is optimal for at least one row across many draws --
    this is the entire point of the rank_preserving/rank_breaking
    distinction (see module docstring)."""
    changed_argmax = False
    for seed in range(50):
        out = permute_row_entries(MATRIX, fraction=1.0, mode="rank_breaking", rng=np.random.default_rng(seed))
        if not np.array_equal(np.argmax(out, axis=1), np.argmax(MATRIX, axis=1)):
            changed_argmax = True
            break
    assert changed_argmax


def test_permute_row_entries_invalid_fraction_raises():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        permute_row_entries(MATRIX, fraction=1.5, mode="rank_breaking", rng=rng)
    with pytest.raises(ValueError):
        permute_row_entries(MATRIX, fraction=-0.1, mode="rank_breaking", rng=rng)


def test_permute_row_entries_invalid_mode_raises():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError):
        permute_row_entries(MATRIX, fraction=0.5, mode="not_a_real_mode", rng=rng)


def test_permute_row_entries_does_not_mutate_input():
    rng = np.random.default_rng(11)
    original = MATRIX.copy()
    permute_row_entries(MATRIX, fraction=1.0, mode="rank_breaking", rng=rng)
    assert np.array_equal(MATRIX, original)


# ======================================================================
# hardest_type_by_ceiling
# ======================================================================


def test_hardest_type_by_ceiling_on_canonical_matrix_is_tool_timeout_not_malformed_output():
    """The key clarifying finding this module exists to surface: under
    the ceiling definition, malformed_output is NOT the canonical
    matrix's hardest type -- tool_timeout is (ceiling 0.70, the lowest
    row max in MATRIX)."""
    idx, name = hardest_type_by_ceiling(MATRIX, type_names=TYPE_NAMES)
    assert name == "tool_timeout"
    assert idx == TYPE_NAMES.index("tool_timeout")
    assert name != "malformed_output"


def test_hardest_type_by_ceiling_matches_manual_argmin():
    idx = hardest_type_by_ceiling(MATRIX)
    assert idx == int(np.argmin(MATRIX.max(axis=1)))


# ======================================================================
# Scheme 3: structured matrices
# ======================================================================


def test_structured_uniform_difficulty_every_row_has_identical_ceiling():
    P = structured_uniform_difficulty()
    assert P.shape == MATRIX.shape
    ceilings = P.max(axis=1)
    assert np.allclose(ceilings, ceilings[0])


def test_structured_uniform_difficulty_every_row_is_a_permutation_of_the_same_multiset():
    P = structured_uniform_difficulty()
    reference = sorted(P[0].tolist())
    for t in range(P.shape[0]):
        assert sorted(P[t].tolist()) == pytest.approx(reference)


def test_structured_single_dominant_strategy_strategy_4_wins_every_row():
    P = structured_single_dominant_strategy()
    assert P.shape == MATRIX.shape
    assert np.all(np.argmax(P, axis=1) == 4)


def test_structured_malformed_not_hardest_shape_and_bounds():
    P = structured_malformed_not_hardest()
    assert P.shape == MATRIX.shape
    assert np.all(P >= 0.0) and np.all(P <= 1.0)


def test_structured_malformed_not_hardest_actually_achieves_its_stated_property():
    P = structured_malformed_not_hardest()
    idx, name = hardest_type_by_ceiling(P, type_names=TYPE_NAMES)
    assert name != "malformed_output"
    assert name == "ambiguous_instruction"
    malformed_idx = TYPE_NAMES.index("malformed_output")
    assert P[malformed_idx].max() > P[idx].max()  # malformed_output's ceiling is comfortably higher


def test_structured_malformed_not_hardest_leaves_five_rows_unchanged():
    P = structured_malformed_not_hardest()
    for t in range(MATRIX.shape[0]):
        if TYPE_NAMES[t] in ("malformed_output", "ambiguous_instruction"):
            continue
        assert np.array_equal(P[t], MATRIX[t])


# ======================================================================
# Sanity: every structured/perturbed matrix is a valid input to the
# EXISTING analytic layer (recovery.theory.analytic) and BLIND_RULE
# still applies (shape compatibility) -- this is the actual reuse
# contract Task 12's experiment script depends on.
# ======================================================================


@pytest.mark.parametrize(
    "P",
    [
        structured_uniform_difficulty(),
        structured_single_dominant_strategy(),
        structured_malformed_not_hardest(),
    ],
)
def test_structured_matrices_are_valid_analytic_inputs(P):
    o = oracle_rate(P)
    r = policy_rate(P, BLIND_RULE)
    assert 0.0 <= r <= o <= 1.0


def test_perturbed_matrices_remain_valid_analytic_inputs():
    rng = np.random.default_rng(123)
    for sigma in (0.05, 0.10, 0.15, 0.20, 0.30):
        P = gaussian_perturb(MATRIX, sigma=sigma, rng=rng)
        o = oracle_rate(P)
        r = policy_rate(P, BLIND_RULE)
        assert 0.0 <= r <= o <= 1.0
    for fraction in (0.1, 0.25, 0.5, 1.0):
        for mode in ("rank_preserving", "rank_breaking"):
            P = permute_row_entries(MATRIX, fraction=fraction, mode=mode, rng=rng)
            o = oracle_rate(P)
            r = policy_rate(P, BLIND_RULE)
            assert 0.0 <= r <= o <= 1.0
