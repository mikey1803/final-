#!/usr/bin/env python
"""Task 10B: shared reliability classification for the crossover predictor.

Presentation/validation layer ONLY -- this module does not touch
``recovery/theory/crossover.py``, does not re-fit ``c``, and does not
change any predicted value. It reads the calibration data and the
already-fitted constant produced by Task 10's (unmodified)
``crossover_calibration.py`` and derives, from that alone, a per-point
reliability classification that both ``crossover_extension.py`` (the
lookup table) and ``crossover_figures.py`` (the plots) share -- so a
point is never classified one way in a figure and a different way in the
lookup table.

Four-way classification
------------------------
- ``"validated"``: BOTH this point's gap-bucket AND quality-bucket met
  the calibration's within-3x reliability bar (see
  ``crossover_calibration.VALIDATED_WITHIN_FACTOR3_THRESHOLD``).
- ``"partially_validated"``: exactly ONE of the two did.
- ``"unsupported"``: this exact (gap, quality) combination WAS present
  in the calibration grid, but neither its gap-bucket nor its
  quality-bucket met the reliability bar -- tested and shown unreliable,
  not merely untested.
- ``"extrapolated"``: this point falls entirely outside what the
  calibration grid ever measured -- a taxonomy size other than 7, a gap
  never sampled, or a (gap, quality) combination that specific gap never
  happened to produce (e.g. quality=5/7 or 7/7 at any gap: the
  calibration grid's random sampling never drew a table that good at any
  gap -- see ``docs/theory/crossover.md``'s practitioner example, which
  is exactly this case).

``confidence_level`` collapses the four statuses to three: ``validated``
-> High, ``partially_validated`` -> Medium, ``unsupported``/``extrapolated``
-> Low (both mean "don't trust it," for different reasons -- see
``docs/theory/practitioner_guidelines.md``).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from crossover_calibration import (  # noqa: E402
    VALIDATED_WITHIN_FACTOR3_THRESHOLD,
    build_calibration_summary,
    load_calibration_rows,
    predicted_ratio_table,
)

VALIDATED_TAXONOMY_SIZE = 7

STATUS_TO_CONFIDENCE = {
    "validated": "High",
    "partially_validated": "Medium",
    "unsupported": "Low",
    "extrapolated": "Low",
}


def build_reliability_context(rows: list[dict], summary: pd.DataFrame) -> dict:
    """Everything ``classify_status`` needs, derived ONLY from Task 10's
    already-computed calibration data -- no re-fitting, no new
    statistics beyond simple set/dict lookups.

    Returns
    -------
    dict with keys:
        ``tested_gaps`` : set of float gaps the calibration grid sampled.
        ``tested_quality_by_gap`` : dict[gap -> set of quality values
            actually observed at that gap] (canonical excluded -- it has
            no ``gap``).
        ``gap_validated`` : dict[gap -> bool], from the CALIBRATED,
            per-gap rows of ``summary``.
        ``quality_validated`` : dict[quality -> bool], from the
            CALIBRATED, per-quality rows of ``summary``.
    """
    tested_gaps: set = set()
    tested_quality_by_gap: dict = {}
    for row in rows:
        gap = row["gap"]
        if gap is None or (isinstance(gap, float) and np.isnan(gap)):
            continue  # canonical -- no gap axis
        g = round(float(gap), 2)
        tested_gaps.add(g)
        tested_quality_by_gap.setdefault(g, set()).add(int(row["quality"]))

    calibrated = summary[summary["calibrated"]]
    gap_validated = {
        round(float(r["group"].removeprefix("gap=")), 2): bool(r["validated_region"])
        for _, r in calibrated[calibrated["group"].str.startswith("gap=")].iterrows()
    }
    quality_validated = {
        int(r["group"].removeprefix("quality=")): bool(r["validated_region"])
        for _, r in calibrated[calibrated["group"].str.startswith("quality=")].iterrows()
    }

    return {
        "tested_gaps": tested_gaps,
        "tested_quality_by_gap": tested_quality_by_gap,
        "gap_validated": gap_validated,
        "quality_validated": quality_validated,
    }


def classify_status(N: int, gap: Optional[float], quality: int, context: dict) -> tuple:
    """Classify one (N, gap, quality) point. Returns ``(status, confidence)``.

    Deterministic, pure lookup against ``context`` -- see the module
    docstring for the exact four-way rule.
    """
    if N != VALIDATED_TAXONOMY_SIZE:
        return "extrapolated", STATUS_TO_CONFIDENCE["extrapolated"]
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        # The canonical environment: a genuinely-tested N=7 calibration
        # point, but it has no "gap" (it is not one of the sampled-matrix
        # cells) and so cannot fall in any gap-bucket -- "unsupported"
        # (tested, no coverage), never "extrapolated" (which implies it
        # was never measured at all -- it was).
        return "unsupported", STATUS_TO_CONFIDENCE["unsupported"]

    g = round(float(gap), 2)
    if g not in context["tested_gaps"]:
        return "extrapolated", STATUS_TO_CONFIDENCE["extrapolated"]
    if quality not in context["tested_quality_by_gap"].get(g, set()):
        return "extrapolated", STATUS_TO_CONFIDENCE["extrapolated"]

    gap_ok = context["gap_validated"].get(g, False)
    quality_ok = context["quality_validated"].get(quality, False)
    if gap_ok and quality_ok:
        status = "validated"
    elif gap_ok or quality_ok:
        status = "partially_validated"
    else:
        status = "unsupported"
    return status, STATUS_TO_CONFIDENCE[status]


def build_calibration_context(form: str = "kl") -> tuple:
    """Convenience one-shot: load calibration rows, build the (unmodified)
    Task 10 calibration summary, and derive the reliability context --
    everything downstream (figures, lookup table, tests) needs from a
    single call. Returns ``(rows, summary, context, best_fit_c)``."""
    rows = load_calibration_rows()
    summary = build_calibration_summary(rows, form=form)
    context = build_reliability_context(rows, summary)
    best_fit_c = float(summary["best_fit_c"].iloc[0])
    return rows, summary, context, best_fit_c


def annotate_status(df: pd.DataFrame, context: dict, n_col: str = "N", gap_col: str = "gap", quality_col: str = "quality") -> pd.DataFrame:
    """Add ``prediction_status``/``confidence_level`` columns to any
    dataframe carrying ``N``/``gap``/``quality`` columns (used by both
    the per-cell calibration summary and the lookup table -- one
    classification rule, applied identically everywhere)."""
    statuses, confidences = [], []
    for _, row in df.iterrows():
        status, confidence = classify_status(int(row[n_col]), row[gap_col], int(row[quality_col]), context)
        statuses.append(status)
        confidences.append(confidence)
    out = df.copy()
    out["prediction_status"] = statuses
    out["confidence_level"] = confidences
    return out


def build_calibration_cell_summary(rows: list[dict], context: dict, best_fit_c: float, form: str = "kl") -> pd.DataFrame:
    """Per-(gap, quality) calibration cell summary: MAE/RMSE/median ratio/
    within-2x/3x/5x, plus ``validated_region``/``extrapolation`` flags
    from ``classify_status`` -- ``results/tables/crossover_calibration_summary.parquet``.

    Every row here describes an ACTUALLY-TESTED (gap, quality)
    combination (the calibration grid), so ``extrapolation`` is always
    ``False`` in this table by construction -- extrapolation only
    applies to points the grid never measured (the lookup table). MAE
    and RMSE are computed on the RAW episode scale (not log-space) --
    with observed crossovers spanning ~10 to ~8,470 episodes, both are
    dominated by the largest cells in any group that contains one; see
    ``docs/theory/crossover.md`` for why this is reported alongside,
    never instead of, the scale-free ``median ratio``/within-Nx metrics.
    """
    per_cell = predicted_ratio_table(rows, form=form, c=best_fit_c)
    per_cell["N"] = VALIDATED_TAXONOMY_SIZE

    out_rows = []
    for (gap, quality), sub in per_cell.groupby(["gap", "quality"], dropna=False):
        if isinstance(gap, float) and np.isnan(gap):
            continue  # canonical: not part of the (gap, quality) grid this table describes
        errors = (sub["predicted_T"] - sub["observed_T"]).to_numpy()
        finite_errors = errors[np.isfinite(errors)]
        ratios = sub["ratio"].to_numpy()
        finite_ratios = ratios[np.isfinite(ratios)]
        log_ratio = np.abs(np.log(finite_ratios)) if finite_ratios.size else np.array([])

        status, confidence = classify_status(VALIDATED_TAXONOMY_SIZE, gap, int(quality), context)
        out_rows.append(
            {
                "gap": float(gap),
                "quality": int(quality),
                "samples": int(len(sub)),
                "MAE": float(np.mean(np.abs(finite_errors))) if finite_errors.size else float("nan"),
                "RMSE": float(np.sqrt(np.mean(finite_errors ** 2))) if finite_errors.size else float("nan"),
                "median_ratio": float(np.median(finite_ratios)) if finite_ratios.size else float("nan"),
                "within_factor2": float(np.mean(log_ratio <= np.log(2.0))) if log_ratio.size else float("nan"),
                "within_factor3": float(np.mean(log_ratio <= np.log(3.0))) if log_ratio.size else float("nan"),
                "within_factor5": float(np.mean(log_ratio <= np.log(5.0))) if log_ratio.size else float("nan"),
                "validated_region": status == "validated",
                "extrapolation": False,
                "prediction_status": status,
                "confidence_level": confidence,
            }
        )
    return pd.DataFrame(out_rows).sort_values(["gap", "quality"]).reset_index(drop=True)


def split_validated_vs_rest(rows: list[dict], context: dict, best_fit_c: float, form: str = "kl") -> pd.DataFrame:
    """Overall metrics computed SEPARATELY for validated vs. non-validated
    points (task 10's original report collapsed this into one misleading
    "41.1% overall" figure -- see docs/theory/crossover.md, Section 4).

    Returns a 2-row DataFrame: ``region in {"validated", "not_validated"}``,
    plus ``n``, ``MAE``, ``RMSE``, ``median_ratio``, ``within_factor2/3/5``.
    """
    per_cell = predicted_ratio_table(rows, form=form, c=best_fit_c)
    statuses = []
    for _, row in per_cell.iterrows():
        gap = row["gap"]
        quality = int(row["quality"])
        status, _ = classify_status(VALIDATED_TAXONOMY_SIZE, gap, quality, context)
        statuses.append(status)
    per_cell = per_cell.copy()
    per_cell["prediction_status"] = statuses
    per_cell["region"] = np.where(per_cell["prediction_status"] == "validated", "validated", "not_validated")

    out_rows = []
    for region, sub in per_cell.groupby("region"):
        errors = (sub["predicted_T"] - sub["observed_T"]).to_numpy()
        finite_errors = errors[np.isfinite(errors)]
        ratios = sub["ratio"].to_numpy()
        finite_ratios = ratios[np.isfinite(ratios)]
        log_ratio = np.abs(np.log(finite_ratios)) if finite_ratios.size else np.array([])
        out_rows.append(
            {
                "region": region,
                "n": int(len(sub)),
                "MAE": float(np.mean(np.abs(finite_errors))) if finite_errors.size else float("nan"),
                "RMSE": float(np.sqrt(np.mean(finite_errors ** 2))) if finite_errors.size else float("nan"),
                "median_ratio": float(np.median(finite_ratios)) if finite_ratios.size else float("nan"),
                "within_factor2": float(np.mean(log_ratio <= np.log(2.0))) if log_ratio.size else float("nan"),
                "within_factor3": float(np.mean(log_ratio <= np.log(3.0))) if log_ratio.size else float("nan"),
                "within_factor5": float(np.mean(log_ratio <= np.log(5.0))) if log_ratio.size else float("nan"),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    rows, summary, context, best_fit_c = build_calibration_context()

    cell_summary = build_calibration_cell_summary(rows, context, best_fit_c)
    out_path = _BACKEND_DIR / "results" / "tables" / "crossover_calibration_summary.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_summary.to_parquet(out_path, index=False)
    print(f"Wrote {len(cell_summary)} rows to {out_path}")
    print(cell_summary.to_string(index=False, float_format="{:.3f}".format))

    split = split_validated_vs_rest(rows, context, best_fit_c)
    print("\n=== Validated vs. not-validated region (calibrated predictor) ===")
    print(split.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
