"""One-off export: turn backend/results/phase4b/nonstationarity_raw.csv (the
per-episode, per-trial raw data behind the two Non-Stationarity matplotlib
figures) into two compact JSON files the React dashboard can chart live,
instead of shipping a static PNG. Reproduces the exact rolling-window
statistics from src/evaluation/nonstationarity.py's
`_rolling_success_rate` / `plot_nonstationarity_shifted_type_only` so the
interactive chart matches the paper figure's numbers.

Run once from backend/: .venv/Scripts/python.exe scripts/export_nonstationarity_curves.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "results" / "phase4b" / "nonstationarity_raw.csv"
OUT_DIR = ROOT.parent / "frontend" / "dashboard-react" / "public" / "data"

METHOD_LABELS = {
    "thompson_sampling": "Stock TS",
    "discounted_thompson_sampling": "Discounted TS (gamma=0.99)",
    "fixed_rule": "Fixed Rule (blind)",
}
NUM_EPISODES = 500
SHIFT_EPISODE = 250


def rolling_success_rate(df: pd.DataFrame, method: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["method"] == method]
    pivot = sub.pivot_table(index="episode", columns="trial", values="success", aggfunc="mean")
    pivot = pivot.reindex(range(1, NUM_EPISODES + 1))
    rolling = pivot.rolling(window=window, min_periods=1).mean()
    return rolling.mean(axis=1).values, rolling.std(axis=1).values


def export_recovery_curve(df: pd.DataFrame, window: int = 30) -> dict:
    rows = []
    for ep in range(1, NUM_EPISODES + 1):
        rows.append({"episode": ep})
    curves = {}
    for method in METHOD_LABELS:
        mean, std = rolling_success_rate(df, method, window)
        curves[method] = (mean, std)
    for i, row in enumerate(rows):
        for method, (mean, std) in curves.items():
            row[method] = round(float(mean[i]), 4)
            row[f"{method}__range"] = [round(float(mean[i] - std[i]), 4), round(float(mean[i] + std[i]), 4)]
    return {
        "window": window,
        "shiftEpisode": SHIFT_EPISODE,
        "methods": list(METHOD_LABELS.keys()),
        "methodLabels": METHOD_LABELS,
        "rows": rows,
    }


def export_shifted_type_only(df: pd.DataFrame, window: int = 15) -> dict:
    shift_occurrences = []
    per_method: dict[str, pd.Series] = {}
    per_method_std: dict[str, pd.Series] = {}
    max_occ = 0

    for method in METHOD_LABELS:
        sub = df[(df["method"] == method) & (df["is_shifted_type"])].sort_values(["trial", "episode"]).copy()
        sub["occurrence_idx"] = sub.groupby("trial").cumcount() + 1
        pivot = sub.pivot_table(index="occurrence_idx", columns="trial", values="success")
        rolling = pivot.rolling(window=window, min_periods=1).mean()
        mean_curve = rolling.mean(axis=1)
        std_curve = rolling.std(axis=1)
        per_method[method] = mean_curve
        per_method_std[method] = std_curve
        max_occ = max(max_occ, int(mean_curve.index.max()))

        shift_occ = sub[sub["episode"] >= SHIFT_EPISODE].groupby("trial")["occurrence_idx"].min().mean()
        if not np.isnan(shift_occ):
            shift_occurrences.append(shift_occ)

    rows = []
    for occ in range(1, max_occ + 1):
        row = {"occurrence": occ}
        for method in METHOD_LABELS:
            mean_curve = per_method[method]
            std_curve = per_method_std[method]
            if occ in mean_curve.index and not np.isnan(mean_curve.loc[occ]):
                m = float(mean_curve.loc[occ])
                s = float(std_curve.loc[occ]) if not np.isnan(std_curve.loc[occ]) else 0.0
                row[method] = round(m, 4)
                row[f"{method}__range"] = [round(m - s, 4), round(m + s, 4)]
        rows.append(row)

    return {
        "window": window,
        "meanShiftOccurrence": round(float(np.mean(shift_occurrences)), 2) if shift_occurrences else None,
        "methods": list(METHOD_LABELS.keys()),
        "methodLabels": METHOD_LABELS,
        "rows": rows,
    }


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recovery = export_recovery_curve(df)
    (OUT_DIR / "nonstationarity_recovery_curve.json").write_text(json.dumps(recovery), encoding="utf-8")
    print(f"  Wrote {OUT_DIR / 'nonstationarity_recovery_curve.json'} ({len(recovery['rows'])} rows)")

    shifted = export_shifted_type_only(df)
    (OUT_DIR / "nonstationarity_shifted_type_only.json").write_text(json.dumps(shifted), encoding="utf-8")
    print(f"  Wrote {OUT_DIR / 'nonstationarity_shifted_type_only.json'} ({len(shifted['rows'])} rows)")


if __name__ == "__main__":
    main()
