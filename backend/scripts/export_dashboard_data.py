"""Exports selected backend/results tables (parquet) and figures (png) into
frontend/dashboard-react/public/, in the plain CSV/JSON the React dashboard's
hand-rolled loaders expect (src/lib/csv.js, src/hooks/useJson.js). There is no
parquet support in the frontend, so this conversion has to happen here.

Run from repo root:  .venv/Scripts/python.exe backend/scripts/export_dashboard_data.py

Re-run whenever backend/results/ is regenerated (e.g. after `reproduce.py`) and
a dashboard tab depends on it. Each entry below names the exact tab it feeds.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TABLES = _REPO_ROOT / "backend" / "results" / "tables"
_FIGURES = _REPO_ROOT / "backend" / "results" / "figures"
_DATA_OUT = _REPO_ROOT / "frontend" / "dashboard-react" / "public" / "data"
_FIGURES_OUT = _REPO_ROOT / "frontend" / "dashboard-react" / "public" / "figures"

# (source parquet filename, output json filename, dashboard tab)
TABLE_EXPORTS = [
    ("crossover_calibration.parquet", "crossover_calibration.json", "Crossover Predictor"),
    ("drift_summary.parquet", "drift_summary.json", "Drift Taxonomy"),
    ("drift_effect_sizes.parquet", "drift_effect_sizes.json", "Drift Taxonomy"),
    ("human_study_llm_by_family.parquet", "blind_design_by_family.json", "Blind-Design Study"),
    ("matrix_robustness_hardest_type.parquet", "matrix_robustness_hardest_type.json", "Matrix-Family Sensitivity"),
    ("matrix_robustness_tstar.parquet", "matrix_robustness_tstar.json", "Matrix-Family Sensitivity"),
    ("prior_optimal_strength.parquet", "prior_optimal_strength.json", "Prior-Guided Learning"),
    ("prior_negative_transfer_region.parquet", "prior_negative_transfer_region.json", "Prior-Guided Learning"),
    ("llm_cost_accounting_summary.parquet", "llm_cost_accounting_summary.json", "Runtime & Cost"),
    ("environment_hardware.parquet", "environment_hardware.json", "Runtime & Cost"),
    ("horizon_crossover.parquet", "horizon_crossover.json", "Crossover Predictor"),
]

# (source png filename, dashboard tab)
FIGURE_EXPORTS = [
    ("crossover_calibration_curve.png", "Crossover Predictor"),
    ("crossover_error_histogram.png", "Crossover Predictor"),
    ("crossover_prediction_scatter.png", "Crossover Predictor"),
    ("crossover_reliability_heatmap.png", "Crossover Predictor"),
    ("drift_taxonomy.png", "Drift Taxonomy"),
    ("adaptation_curves.png", "Drift Taxonomy"),
    ("recovery_half_life.png", "Drift Taxonomy"),
    ("llm_study_per_type_accuracy_by_family.png", "Blind-Design Study"),
    ("llm_study_q_by_family.png", "Blind-Design Study"),
    ("robustness_ranking_preservation.png", "Matrix-Family Sensitivity"),
    ("robustness_tstar_distribution.png", "Matrix-Family Sensitivity"),
    ("runtime_quality_tradeoff.png", "Runtime & Cost"),
]


def export_tables() -> None:
    _DATA_OUT.mkdir(parents=True, exist_ok=True)
    for src_name, out_name, tab in TABLE_EXPORTS:
        src = _TABLES / src_name
        df = pd.read_parquet(src)
        # Plain JSON has no Infinity/NaN literal; some tables (e.g. an
        # unbounded "optimal prior strength") contain real +-inf. Null them
        # out rather than crash the fetch, and let the tab render "no limit".
        df = df.replace([np.inf, -np.inf], None)
        out_path = _DATA_OUT / out_name
        df.to_json(out_path, orient="records", indent=2)
        print(f"[{tab}] {src_name} ({df.shape[0]} rows) -> {out_path.relative_to(_REPO_ROOT)}")


def export_figures() -> None:
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    for name, tab in FIGURE_EXPORTS:
        src = _FIGURES / name
        dst = _FIGURES_OUT / name
        shutil.copyfile(src, dst)
        print(f"[{tab}] {name} -> {dst.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    export_tables()
    export_figures()
    print("Done.")
