"""Tests for the Task 12 matrix-robustness experiment
(``experiments/matrix_robustness.py``): ``evaluate_world``'s correctness
against the existing analytic layer, deterministic world-spec
construction, and the aggregation/breakdown-threshold helpers.

Deliberately small-scale (few seeds, few episodes are NOT reduced --
``evaluate_world`` always uses the module's real ``N_EPISODES=500`` --
only the NUMBER of worlds/seeds tested here is small) so these tests run
in well under a second, unlike the full sweep itself (a real ~10-20
minute run, invoked directly via
``python experiments/matrix_robustness.py``, not through pytest -- see
``docs/experiments/matrix_robustness.md``).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from experiments.matrix_robustness import (  # noqa: E402
    BASE_SEED,
    build_noise_floor_specs,
    build_world_specs,
    breakdown_threshold,
    evaluate_world,
    hardest_type_summary,
    ranking_preservation_summary,
    t_star_summary,
)
from experiments.study_analysis import load_calibrated_c  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.theory.analytic import oracle_rate, policy_rate, random_rate  # noqa: E402
from recovery.theory.crossover import predict_crossover  # noqa: E402

_C = load_calibrated_c()


# ======================================================================
# evaluate_world
# ======================================================================


def test_evaluate_world_analytic_fields_match_the_analytic_layer_exactly():
    """Oracle/BlindRule/UniformRandom/T* must come out IDENTICAL to
    calling recovery.theory.analytic/crossover directly -- evaluate_world
    must not re-derive these, only call through."""
    result = evaluate_world(MATRIX, "canonical", "canonical", 0, BASE_SEED, n_seeds=2, calibrated_c=_C)
    assert result["oracle_rate"] == pytest.approx(oracle_rate(MATRIX))
    assert result["blind_rule_rate"] == pytest.approx(policy_rate(MATRIX, BLIND_RULE))
    assert result["uniform_random_rate"] == pytest.approx(random_rate(MATRIX))
    expected_t_star, expected_truncated = predict_crossover(MATRIX, BLIND_RULE, form="kl", c=_C)
    assert result["T_star"] == pytest.approx(expected_t_star)
    assert result["T_star_truncated"] == expected_truncated


def test_evaluate_world_deterministic_given_same_seed():
    r1 = evaluate_world(MATRIX, "canonical", "canonical", 0, 999, n_seeds=3, calibrated_c=_C)
    r2 = evaluate_world(MATRIX, "canonical", "canonical", 0, 999, n_seeds=3, calibrated_c=_C)
    assert r1["thompson_rate"] == r2["thompson_rate"]
    assert r1["linucb_rate"] == r2["linucb_rate"]
    assert r1["ranking_order"] == r2["ranking_order"]


def test_evaluate_world_different_seeds_give_different_simulated_rates():
    r1 = evaluate_world(MATRIX, "canonical", "canonical", 0, 1, n_seeds=2, calibrated_c=_C)
    r2 = evaluate_world(MATRIX, "canonical", "canonical", 0, 2, n_seeds=2, calibrated_c=_C)
    # Not a strict guarantee for ANY pair of seeds, but true for these two
    # (checked once, not asserted as a mathematical certainty).
    assert (r1["thompson_rate"], r1["linucb_rate"]) != (r2["thompson_rate"], r2["linucb_rate"])


def test_evaluate_world_ranking_order_contains_all_five_methods():
    result = evaluate_world(MATRIX, "canonical", "canonical", 0, BASE_SEED, n_seeds=2, calibrated_c=_C)
    for method in ("Oracle", "BlindRule", "LinUCB", "Thompson", "UniformRandom"):
        assert method in result["ranking_order"]
    assert result["ranking_order"].count(">") == 4


def test_evaluate_world_oracle_is_always_first_in_ranking():
    """Oracle >= every fixed/adaptive policy's rate is a mathematical
    certainty (recovery.theory.analytic.oracle_rate's docstring) -- must
    hold for every world, not just the canonical one."""
    rng = np.random.default_rng(0)
    from recovery.envs.matrix_perturbation import gaussian_perturb

    for sigma in (0.05, 0.2):
        P = gaussian_perturb(MATRIX, sigma=sigma, rng=rng)
        result = evaluate_world(P, "gaussian", f"sigma={sigma}", 0, BASE_SEED, n_seeds=2, calibrated_c=_C)
        assert result["ranking_order"].startswith("Oracle")


def test_evaluate_world_hardest_type_name_is_a_canonical_type_name():
    from recovery.envs.canonical import TYPE_NAMES

    result = evaluate_world(MATRIX, "canonical", "canonical", 0, BASE_SEED, n_seeds=2, calibrated_c=_C)
    assert result["hardest_type_name"] in TYPE_NAMES


# ======================================================================
# build_world_specs / build_noise_floor_specs: determinism, no accidental
# seed collisions
# ======================================================================


def test_build_world_specs_is_deterministic():
    specs1 = build_world_specs(n_matrices_scheme1=3, n_matrices_scheme2=2, n_seeds=2)
    specs2 = build_world_specs(n_matrices_scheme1=3, n_matrices_scheme2=2, n_seeds=2)
    assert len(specs1) == len(specs2)
    for s1, s2 in zip(specs1, specs2):
        assert s1["seed"] == s2["seed"]
        assert np.array_equal(s1["matrix"], s2["matrix"])
        assert s1["scheme"] == s2["scheme"]
        assert s1["magnitude"] == s2["magnitude"]


def test_build_world_specs_expected_count():
    specs = build_world_specs(n_matrices_scheme1=3, n_matrices_scheme2=2, n_seeds=2)
    # 1 canonical + 5*3 (scheme1) + 4*2*2 (scheme2) + 3 (scheme3)
    assert len(specs) == 1 + 5 * 3 + 4 * 2 * 2 + 3


def test_build_world_specs_no_duplicate_seeds():
    specs = build_world_specs(n_matrices_scheme1=5, n_matrices_scheme2=5, n_seeds=3)
    seeds = [s["seed"] for s in specs]
    assert len(seeds) == len(set(seeds)), "every world must get a distinct base seed"


def test_build_noise_floor_specs_no_overlapping_seed_blocks():
    n_seeds = 4
    specs = build_noise_floor_specs(n_reps=20, n_seeds=n_seeds)
    used_seeds = set()
    for spec in specs:
        block = set(range(spec["seed"], spec["seed"] + n_seeds))
        assert used_seeds.isdisjoint(block), "noise-floor replicate seed blocks must not overlap"
        used_seeds |= block


def test_build_noise_floor_specs_all_use_the_unperturbed_canonical_matrix():
    specs = build_noise_floor_specs(n_reps=5, n_seeds=2)
    for spec in specs:
        assert np.array_equal(spec["matrix"], MATRIX)
        assert spec["scheme"] == "noise_floor"


# ======================================================================
# Aggregation helpers, on small synthetic DataFrames (no simulation)
# ======================================================================


def _synthetic_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "scheme": "gaussian", "magnitude": "sigma=0.10", "T_star": 100.0,
        "hardest_type_name": "tool_timeout", "full_order_preserved": True,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_ranking_preservation_summary_basic_fraction():
    df = _synthetic_df([
        {"headline_preserved": True}, {"headline_preserved": True},
        {"headline_preserved": False}, {"headline_preserved": True},
    ])
    out = ranking_preservation_summary(df)
    assert len(out) == 1
    assert out.iloc[0]["fraction_headline_preserved"] == pytest.approx(0.75)
    assert out.iloc[0]["n"] == 4
    assert 0.0 <= out.iloc[0]["headline_ci95_lower"] <= out.iloc[0]["fraction_headline_preserved"]
    assert out.iloc[0]["fraction_headline_preserved"] <= out.iloc[0]["headline_ci95_upper"] <= 1.0


def test_ranking_preservation_summary_groups_by_scheme_and_magnitude():
    df = _synthetic_df([
        {"scheme": "gaussian", "magnitude": "sigma=0.05", "headline_preserved": True},
        {"scheme": "gaussian", "magnitude": "sigma=0.30", "headline_preserved": False},
    ])
    out = ranking_preservation_summary(df)
    assert len(out) == 2
    assert set(out["magnitude"]) == {"sigma=0.05", "sigma=0.30"}


def test_t_star_summary_handles_all_infinite():
    df = _synthetic_df([{"T_star": float("inf")}, {"T_star": float("inf")}])
    out = t_star_summary(df)
    assert out.iloc[0]["fraction_infinite"] == 1.0
    assert out.iloc[0]["n_infinite"] == 2
    assert np.isnan(out.iloc[0]["T_star_finite_mean"])
    assert out.iloc[0]["T_star_ci95_lower"] is None


def test_t_star_summary_mixed_finite_and_infinite():
    df = _synthetic_df([
        {"T_star": 50.0}, {"T_star": 150.0}, {"T_star": float("inf")},
    ])
    out = t_star_summary(df)
    assert out.iloc[0]["n_infinite"] == 1
    assert out.iloc[0]["fraction_infinite"] == pytest.approx(1 / 3)
    assert out.iloc[0]["T_star_finite_mean"] == pytest.approx(100.0)


def test_hardest_type_summary_reports_proportions():
    df = _synthetic_df([
        {"hardest_type_name": "tool_timeout"},
        {"hardest_type_name": "tool_timeout"},
        {"hardest_type_name": "malformed_output"},
    ])
    out = hardest_type_summary(df)
    row = out.iloc[0]
    assert row["tool_timeout"] == pytest.approx(2 / 3)
    assert row["malformed_output"] == pytest.approx(1 / 3)


def test_breakdown_threshold_finds_first_crossing_below_half():
    ranking_df = pd.DataFrame([
        {"scheme": "gaussian", "magnitude": "sigma=0.05", "fraction_headline_preserved": 1.0},
        {"scheme": "gaussian", "magnitude": "sigma=0.10", "fraction_headline_preserved": 0.9},
        {"scheme": "gaussian", "magnitude": "sigma=0.15", "fraction_headline_preserved": 0.4},
        {"scheme": "gaussian", "magnitude": "sigma=0.20", "fraction_headline_preserved": 0.1},
    ])
    order = ["sigma=0.05", "sigma=0.10", "sigma=0.15", "sigma=0.20"]
    assert breakdown_threshold(ranking_df, "gaussian", order) == "sigma=0.15"


def test_breakdown_threshold_returns_none_if_never_drops_below_half():
    ranking_df = pd.DataFrame([
        {"scheme": "gaussian", "magnitude": "sigma=0.05", "fraction_headline_preserved": 1.0},
        {"scheme": "gaussian", "magnitude": "sigma=0.10", "fraction_headline_preserved": 0.9},
    ])
    order = ["sigma=0.05", "sigma=0.10"]
    assert breakdown_threshold(ranking_df, "gaussian", order) is None
