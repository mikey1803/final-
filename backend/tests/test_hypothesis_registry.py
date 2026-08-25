"""Integration test for the Task 13 hypothesis-registry driver
(``experiments/hypothesis_registry.py``): checks the assembled per-family
table's shape/columns on a small synthetic LLM dataframe, without
re-testing logic already covered by ``tests/test_significance.py`` or
Task 11's ``tests/test_study.py``."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from experiments.hypothesis_registry import run_family_tests  # noqa: E402
from recovery.envs.canonical import MATRIX, TYPE_NAMES  # noqa: E402


def _synthetic_llm_df(n_per_family: int = 25) -> pd.DataFrame:
    """A tiny, fully-synthetic (never presented as real) LLM dataframe
    with the same schema as ``llm_designers.parquet`` -- enough to
    exercise ``run_family_tests`` end to end without touching real data."""
    rng = np.random.default_rng(7)
    rows = []
    for family in ("fam_a", "fam_b"):
        for i in range(n_per_family):
            mapping = [int(rng.integers(0, MATRIX.shape[1])) for _ in range(MATRIX.shape[0])]
            rows.append({
                "family": family, "sample_index": i, "parsed_mapping": mapping,
                "parse_status": "ok",
            })
        # a couple of parse failures, to confirm they're excluded but counted
        rows.append({"family": family, "sample_index": n_per_family, "parsed_mapping": None, "parse_status": "parse_failure"})
    return pd.DataFrame(rows)


def test_run_family_tests_shape_and_columns():
    df = _synthetic_llm_df()
    out = run_family_tests(df, MATRIX)
    assert set(out["family"].unique()) == {"fam_a", "fam_b"}
    assert len(out) == 2 * len(TYPE_NAMES)
    for col in ("n_api_calls", "n_parsed_ok", "n_distinct_mappings", "bootstrap_ci_degenerate",
                "accuracy", "accuracy_ci95_lower", "accuracy_ci95_upper", "p_value_raw",
                "p_value_holm", "significant_after_correction", "omnibus_cochrans_Q", "omnibus_p_value"):
        assert col in out.columns


def test_run_family_tests_n_api_calls_includes_parse_failures():
    df = _synthetic_llm_df(n_per_family=10)
    out = run_family_tests(df, MATRIX)
    # 10 ok + 1 parse_failure per family = 11 API calls, but only 10 parsed OK
    fam_a = out[out["family"] == "fam_a"].iloc[0]
    assert fam_a["n_api_calls"] == 11
    assert fam_a["n_parsed_ok"] == 10


def test_run_family_tests_holm_corrected_p_never_less_than_raw():
    df = _synthetic_llm_df()
    out = run_family_tests(df, MATRIX)
    valid = out.dropna(subset=["p_value_raw"])
    assert (valid["p_value_holm"] >= valid["p_value_raw"] - 1e-12).all()


def test_run_family_tests_degenerate_flag_matches_distinct_count():
    df = _synthetic_llm_df()
    out = run_family_tests(df, MATRIX)
    assert ((out["n_distinct_mappings"] <= 3) == out["bootstrap_ci_degenerate"]).all()
