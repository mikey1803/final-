"""Tests for Task 10B's presentation/validation layer (prediction_status
classification, the lookup table's extended schema, and figure
generation) -- ``experiments/crossover_reliability.py``,
``experiments/crossover_extension.py``, ``experiments/crossover_figures.py``.

None of Task 10B's code touches ``recovery/theory/crossover.py`` or the
fitted calibration constant -- these tests check the CLASSIFICATION and
PRESENTATION layer only, never re-deriving or re-fitting anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

import crossover_extension as ce  # noqa: E402
import crossover_figures as cf  # noqa: E402
from crossover_calibration import predicted_ratio_table  # noqa: E402
from crossover_reliability import (  # noqa: E402
    STATUS_TO_CONFIDENCE,
    build_calibration_cell_summary,
    build_calibration_context,
    classify_status,
)

VALID_STATUSES = {"validated", "partially_validated", "unsupported", "extrapolated"}
VALID_CONFIDENCES = {"High", "Medium", "Low"}


# ======================================================================
# 1. Every lookup row has a valid prediction_status (and confidence_level)
# ======================================================================


def test_every_lookup_row_has_valid_prediction_status():
    lookup = ce.build_lookup_table(n_list=[7, 14], gaps=[0.05, 0.20], n_matrices=3)
    assert len(lookup) > 0
    assert set(lookup["prediction_status"].unique()) <= VALID_STATUSES
    assert set(lookup["confidence_level"].unique()) <= VALID_CONFIDENCES
    assert lookup["prediction_status"].notna().all()
    assert lookup["confidence_level"].notna().all()


# ======================================================================
# 2. Validated rows are never marked extrapolated
# ======================================================================


def test_validated_rows_never_marked_extrapolated():
    lookup = ce.build_lookup_table(n_list=[7, 14, 28], gaps=[0.05, 0.10, 0.20, 0.30], n_matrices=3)

    validated_rows = lookup[lookup["prediction_status"] == "validated"]
    assert not validated_rows.empty, "expected at least one validated row in this sweep"
    assert (validated_rows["is_extrapolation"] == False).all()  # noqa: E712
    assert (validated_rows["validated_region"] == True).all()  # noqa: E712
    assert (validated_rows["confidence_level"] == "High").all()

    extrapolated_rows = lookup[lookup["prediction_status"] == "extrapolated"]
    assert not extrapolated_rows.empty
    assert (extrapolated_rows["is_extrapolation"] == True).all()  # noqa: E712
    assert (extrapolated_rows["validated_region"] == False).all()  # noqa: E712

    # N != 7 must always be extrapolated -- never validated or partially_validated
    non_n7 = lookup[lookup["N"] != 7]
    assert (non_n7["prediction_status"] == "extrapolated").all()


def test_classify_status_confidence_mapping_is_consistent():
    """confidence_level must be derived DIRECTLY from prediction_status --
    the same status must never map to two different confidence levels."""
    _rows, _summary, context, _c = build_calibration_context()
    seen: dict = {}
    for N in (7, 14):
        for gap in (0.05, 0.10, 0.20, 0.30, None):
            for quality in range(0, 5):
                status, confidence = classify_status(N, gap, quality, context)
                assert status in VALID_STATUSES
                assert confidence == STATUS_TO_CONFIDENCE[status]
                if status in seen:
                    assert seen[status] == confidence
                seen[status] = confidence


# ======================================================================
# 3 & 4. Figure generation succeeds
# ======================================================================


def _synthetic_per_cell() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20
    gaps = rng.choice([0.05, 0.10, 0.20, 0.30], size=n)
    observed = rng.uniform(10, 5000, size=n)
    predicted = observed * rng.uniform(0.2, 5.0, size=n)
    statuses = rng.choice(
        ["validated", "partially_validated", "unsupported", "extrapolated"], size=n
    )
    return pd.DataFrame(
        {
            "env_id": [f"synthetic_{i}" for i in range(n)],
            "gap": gaps,
            "quality": rng.integers(0, 7, size=n),
            "N": 7,
            "rho": rng.uniform(0.01, 0.5, size=n),
            "observed_T": observed,
            "predicted_T": predicted,
            "prediction_truncated": False,
            "ratio": predicted / observed,
            "form": "kl",
            "c": 0.00452,
            "prediction_status": statuses,
            "confidence_level": [STATUS_TO_CONFIDENCE[s] for s in statuses],
        }
    )


def _synthetic_cell_summary() -> pd.DataFrame:
    rows = []
    for gap in [0.05, 0.10, 0.20, 0.30]:
        for quality in range(0, 4):
            rows.append(
                {
                    "gap": gap,
                    "quality": quality,
                    "samples": 5,
                    "MAE": 50.0,
                    "RMSE": 60.0,
                    "median_ratio": 1.2,
                    "within_factor2": 0.4,
                    "within_factor3": 0.6,
                    "within_factor5": 0.8,
                    "validated_region": gap == 0.05,
                    "extrapolation": False,
                    "prediction_status": "validated" if gap == 0.05 else "unsupported",
                    "confidence_level": "High" if gap == 0.05 else "Low",
                }
            )
    return pd.DataFrame(rows)


def test_reliability_heatmap_generation_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "FIGURES_DIR", tmp_path)
    cell_summary = _synthetic_cell_summary()
    cf.plot_reliability_heatmap(cell_summary)
    assert (tmp_path / "crossover_reliability_heatmap.png").exists()
    assert (tmp_path / "crossover_reliability_heatmap.pdf").exists()
    assert (tmp_path / "crossover_reliability_heatmap.png").stat().st_size > 0
    assert (tmp_path / "crossover_reliability_heatmap.pdf").stat().st_size > 0


def test_scatter_plot_generation_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "FIGURES_DIR", tmp_path)
    per_cell = _synthetic_per_cell()
    metrics = cf.plot_prediction_scatter(per_cell)
    assert (tmp_path / "crossover_prediction_scatter.png").exists()
    assert (tmp_path / "crossover_prediction_scatter.pdf").exists()
    assert (tmp_path / "crossover_prediction_scatter.png").stat().st_size > 0
    assert {"r2_log10", "mae_log10", "rmse_log10"} <= set(metrics.keys())


def test_error_histogram_and_calibration_curve_generation_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "FIGURES_DIR", tmp_path)
    per_cell = _synthetic_per_cell()
    cf.plot_error_histogram(per_cell)
    cf.plot_calibration_curve(per_cell)
    for stem in ["crossover_error_histogram", "crossover_calibration_curve"]:
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()


# ======================================================================
# 5. Calibration summary metrics match the source data
# ======================================================================


def test_calibration_summary_metrics_match_source_data():
    """Every MAE/RMSE/median_ratio/within-Nx value in
    build_calibration_cell_summary's output must be independently
    reproducible by recomputing directly from predicted_ratio_table for
    that exact (gap, quality) group -- not just internally consistent,
    but traceable back to the same source rows."""
    rows, summary, context, best_fit_c = build_calibration_context()
    cell_summary = build_calibration_cell_summary(rows, context, best_fit_c)
    assert len(cell_summary) > 0

    per_cell = predicted_ratio_table(rows, form="kl", c=best_fit_c)

    checked = 0
    for _, expected in cell_summary.iterrows():
        sub = per_cell[np.isclose(per_cell["gap"], expected["gap"]) & (per_cell["quality"] == expected["quality"])]
        assert len(sub) == expected["samples"]

        errors = (sub["predicted_T"] - sub["observed_T"]).to_numpy()
        recomputed_mae = float(np.mean(np.abs(errors)))
        recomputed_rmse = float(np.sqrt(np.mean(errors ** 2)))
        recomputed_median_ratio = float(np.median(sub["ratio"].to_numpy()))

        assert recomputed_mae == pytest.approx(expected["MAE"], rel=1e-9)
        assert recomputed_rmse == pytest.approx(expected["RMSE"], rel=1e-9)
        assert recomputed_median_ratio == pytest.approx(expected["median_ratio"], rel=1e-9)

        log_ratio = np.abs(np.log(sub["ratio"].to_numpy()))
        assert float(np.mean(log_ratio <= np.log(2.0))) == pytest.approx(expected["within_factor2"], rel=1e-9)
        assert float(np.mean(log_ratio <= np.log(3.0))) == pytest.approx(expected["within_factor3"], rel=1e-9)
        assert float(np.mean(log_ratio <= np.log(5.0))) == pytest.approx(expected["within_factor5"], rel=1e-9)
        checked += 1

    assert checked == len(cell_summary)


def test_calibration_summary_extrapolation_always_false_for_tested_cells():
    """Every row in crossover_calibration_summary.parquet describes an
    ACTUALLY-TESTED (gap, quality) combination -- extrapolation (a
    combination never measured) cannot apply to the calibration grid's
    own cells by construction."""
    rows, summary, context, best_fit_c = build_calibration_context()
    cell_summary = build_calibration_cell_summary(rows, context, best_fit_c)
    assert (cell_summary["extrapolation"] == False).all()  # noqa: E712
