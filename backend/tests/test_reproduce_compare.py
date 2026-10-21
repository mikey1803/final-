"""reproduce.py's comparison helpers: machine-specific environment captures
are compared for schema only (Stage 6's stated design); everything else keeps
the full value comparison."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reproduce  # noqa: E402


def _write(df, path):
    df.to_parquet(path, index=False)
    return path


def test_schema_only_passes_on_value_and_row_changes(tmp_path):
    prod = _write(pd.DataFrame({"package": ["a", "b"], "version": ["1", "2"]}), tmp_path / "p.parquet")
    cand = _write(pd.DataFrame({"package": ["a", "b", "c"], "version": ["1", "3", "9"]}), tmp_path / "c.parquet")
    r = reproduce.compare_schema_only(prod, cand)
    assert r.status == "schema_match"
    assert "values differ" in r.detail  # the difference stays visible


def test_schema_only_fails_on_schema_change(tmp_path):
    prod = _write(pd.DataFrame({"os_version": ["10.0.26200"]}), tmp_path / "p.parquet")
    renamed = _write(pd.DataFrame({"os_build": ["10.0.26300"]}), tmp_path / "c1.parquet")
    retyped = _write(pd.DataFrame({"os_version": [10]}), tmp_path / "c2.parquet")
    assert reproduce.compare_schema_only(prod, renamed).status == "mismatch"
    assert reproduce.compare_schema_only(prod, retyped).status == "mismatch"


def test_schema_only_fails_when_candidate_missing(tmp_path):
    prod = _write(pd.DataFrame({"x": [1]}), tmp_path / "p.parquet")
    assert reproduce.compare_schema_only(prod, tmp_path / "absent.parquet").status == "mismatch"


def test_only_the_two_machine_captures_are_schema_only():
    assert reproduce.SCHEMA_ONLY_ARTIFACTS == {"environment_hardware.parquet", "environment_packages.parquet"}


def test_full_comparison_still_catches_value_changes(tmp_path):
    prod = _write(pd.DataFrame({"rate": [0.5]}), tmp_path / "p.parquet")
    cand = _write(pd.DataFrame({"rate": [0.6]}), tmp_path / "c.parquet")
    assert reproduce.compare_table(prod, cand).status == "mismatch"
