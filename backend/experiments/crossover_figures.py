#!/usr/bin/env python
"""Task 10B: publication-quality figures for the crossover predictor.

Reads only already-computed calibration data (via
``crossover_reliability.build_calibration_context``, which itself only
calls Task 10's unmodified ``crossover_calibration.py``) -- no
simulation, no re-fitting, no change to any predicted value. Every
figure is saved as both PNG (dpi=300, for viewing) and PDF (vector, for
print/LaTeX inclusion).

Figures
-------
1. ``crossover_prediction_scatter.{png,pdf}`` -- predicted vs. observed
   crossover, log-log, identity line, colored by reliability status,
   marker shape by gap, with R^2/MAE/RMSE annotated.
2. ``crossover_reliability_heatmap.{png,pdf}`` -- fraction within 3x by
   (quality, gap), sample count overlaid per cell.
3. ``crossover_error_histogram.{png,pdf}`` -- distribution of signed
   log10(ratio) (over- vs. under-prediction), split by validated/not.
4. ``crossover_calibration_curve.{png,pdf}`` -- ratio vs. observed
   horizon (log-log): does the predictor's bias depend on scale?
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from crossover_calibration import predicted_ratio_table  # noqa: E402
from crossover_reliability import (  # noqa: E402
    build_calibration_cell_summary,
    build_calibration_context,
    classify_status,
)

FIGURES_DIR = _BACKEND_DIR / "results" / "figures"

STATUS_COLORS = {
    "validated": "#16a34a",
    "partially_validated": "#f59e0b",
    "unsupported": "#dc2626",
    "extrapolated": "#6b7280",
}
STATUS_ORDER = ["validated", "partially_validated", "unsupported", "extrapolated"]
GAP_MARKERS = {0.05: "o", 0.10: "s", 0.20: "^", 0.30: "D"}


def _save(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / {stem}.pdf")


def _annotated_per_cell(rows, context, best_fit_c) -> pd.DataFrame:
    df = predicted_ratio_table(rows, form="kl", c=best_fit_c)
    df["N"] = 7
    statuses, confidences = [], []
    for _, row in df.iterrows():
        status, confidence = classify_status(7, row["gap"], int(row["quality"]), context)
        statuses.append(status)
        confidences.append(confidence)
    df["prediction_status"] = statuses
    df["confidence_level"] = confidences
    return df


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """R^2, MAE, RMSE in LOG10 space (raw-episode-scale R^2 is meaningless
    here -- a handful of large-T cells like ``canonical`` would dominate
    the sum of squares completely; log space is the standard choice for
    quantities spanning orders of magnitude, and matches the ratio-based
    metrics used everywhere else in this project)."""
    log_obs = np.log10(observed)
    log_pred = np.log10(predicted)
    residuals = log_pred - log_obs
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2_log10": r2, "mae_log10": mae, "rmse_log10": rmse}


def plot_prediction_scatter(per_cell: pd.DataFrame) -> dict:
    finite = per_cell[np.isfinite(per_cell["predicted_T"]) & np.isfinite(per_cell["observed_T"])]
    finite = finite[(finite["predicted_T"] > 0) & (finite["observed_T"] > 0)]

    metrics = _regression_metrics(finite["observed_T"].to_numpy(), finite["predicted_T"].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 7))
    lo = min(finite["observed_T"].min(), finite["predicted_T"].min()) * 0.5
    hi = max(finite["observed_T"].max(), finite["predicted_T"].max()) * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="y = x (perfect prediction)", zorder=1)

    for status in STATUS_ORDER:
        for gap, marker in GAP_MARKERS.items():
            sub = finite[(finite["prediction_status"] == status) & (np.isclose(finite["gap"], gap))]
            if sub.empty:
                continue
            ax.scatter(
                sub["observed_T"], sub["predicted_T"], marker=marker, color=STATUS_COLORS[status],
                s=70, alpha=0.85, edgecolor="black", linewidth=0.4, zorder=3,
            )
        # canonical (gap=NaN) points for this status, if any -- drawn with an 'x'
        sub_nan = finite[(finite["prediction_status"] == status) & finite["gap"].isna()]
        if not sub_nan.empty:
            ax.scatter(
                sub_nan["observed_T"], sub_nan["predicted_T"], marker="x", color=STATUS_COLORS[status],
                s=110, linewidth=2.2, zorder=4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Empirical crossover horizon T (episodes, log scale)", fontsize=12)
    ax.set_ylabel("Predicted crossover horizon T* (episodes, log scale)", fontsize=12)
    ax.set_title("Predicted vs. empirical crossover (calibrated KL predictor)", fontsize=13)
    ax.grid(True, which="both", alpha=0.25)

    status_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=STATUS_COLORS[s], markersize=9,
                    markeredgecolor="black", label=s.replace("_", " "))
        for s in STATUS_ORDER if (finite["prediction_status"] == s).any()
    ]
    gap_handles = [
        plt.Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=9, label=f"gap={g:.2f}")
        for g, m in GAP_MARKERS.items()
    ]
    identity_handle = [plt.Line2D([0], [0], color="k", linestyle="--", label="y = x")]
    ax.legend(handles=status_handles + gap_handles + identity_handle, fontsize=8, loc="upper left", ncol=1)

    text = (
        f"n = {len(finite)}\n"
        f"R² (log10) = {metrics['r2_log10']:.3f}\n"
        f"MAE (log10) = {metrics['mae_log10']:.3f}\n"
        f"RMSE (log10) = {metrics['rmse_log10']:.3f}"
    )
    ax.text(
        0.97, 0.03, text, transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    fig.tight_layout()
    _save(fig, "crossover_prediction_scatter")
    return metrics


def plot_reliability_heatmap(cell_summary: pd.DataFrame) -> None:
    qualities = sorted(cell_summary["quality"].unique())
    gaps = sorted(cell_summary["gap"].unique())
    grid = np.full((len(qualities), len(gaps)), np.nan)
    counts = np.zeros((len(qualities), len(gaps)), dtype=int)

    for _, row in cell_summary.iterrows():
        qi = qualities.index(row["quality"])
        gi = gaps.index(row["gap"])
        grid[qi, gi] = row["within_factor3"]
        counts[qi, gi] = row["samples"]

    fig, ax = plt.subplots(figsize=(7, 6))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad="#e5e7eb")
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for qi in range(len(qualities)):
        for gi in range(len(gaps)):
            if np.isnan(grid[qi, gi]):
                continue
            ax.text(
                gi, qi, f"{grid[qi, gi]:.2f}\n(n={counts[qi, gi]})", ha="center", va="center",
                fontsize=9, color="black",
            )

    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels([f"{g:.2f}" for g in gaps])
    ax.set_yticks(range(len(qualities)))
    ax.set_yticklabels([f"{int(q)}/7" for q in qualities])
    ax.set_xlabel("Optimal-arm gap", fontsize=12)
    ax.set_ylabel("Static table quality (correct rows / 7)", fontsize=12)
    ax.set_title("Reliability: fraction of predictions within 3x (calibrated KL)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction within 3x")

    fig.tight_layout()
    _save(fig, "crossover_reliability_heatmap")


def plot_error_histogram(per_cell: pd.DataFrame) -> None:
    finite = per_cell[np.isfinite(per_cell["ratio"]) & (per_cell["ratio"] > 0)]
    log_ratio = np.log10(finite["ratio"].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(log_ratio.min() - 0.2, log_ratio.max() + 0.2, 25)

    validated_mask = finite["prediction_status"].to_numpy() == "validated"
    ax.hist(log_ratio[~validated_mask], bins=bins, color="#dc2626", alpha=0.6, label="not validated", edgecolor="black")
    ax.hist(log_ratio[validated_mask], bins=bins, color="#16a34a", alpha=0.8, label="validated", edgecolor="black")

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="perfect prediction (ratio = 1)")
    ax.set_xlabel("log10(predicted / observed)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction error distribution (calibrated KL predictor)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, "crossover_error_histogram")


def plot_calibration_curve(per_cell: pd.DataFrame) -> None:
    finite = per_cell[np.isfinite(per_cell["ratio"]) & (per_cell["ratio"] > 0) & (per_cell["observed_T"] > 0)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for status in STATUS_ORDER:
        sub = finite[finite["prediction_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["observed_T"], sub["ratio"], color=STATUS_COLORS[status], s=60, alpha=0.8,
            edgecolor="black", linewidth=0.4, label=status.replace("_", " "),
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="ratio = 1 (perfect)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Empirical crossover horizon (episodes, log scale)", fontsize=12)
    ax.set_ylabel("Predicted / observed ratio (log scale)", fontsize=12)
    ax.set_title("Does prediction bias depend on horizon scale?", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    _save(fig, "crossover_calibration_curve")


def main() -> None:
    rows, summary, context, best_fit_c = build_calibration_context()
    per_cell = _annotated_per_cell(rows, context, best_fit_c)
    cell_summary = build_calibration_cell_summary(rows, context, best_fit_c)

    print(f"Generating figures from {len(per_cell)} calibration points (best_fit_c={best_fit_c:.5f})...")
    scatter_metrics = plot_prediction_scatter(per_cell)
    plot_reliability_heatmap(cell_summary)
    plot_error_histogram(per_cell)
    plot_calibration_curve(per_cell)

    print("\nScatter plot regression metrics (log10 space):")
    for k, v in scatter_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
