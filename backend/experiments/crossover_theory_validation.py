#!/usr/bin/env python
"""Task 19A: validate the classifier-noise-aware crossover theory
(``recovery/theory/crossover.py``'s ``*_confused``/``ceiling_reduction``/
``gap_shrinkage_report``/``argmax_inversion_check`` additions) against
already-committed, real data. No new simulation is run anywhere in this
script -- every empirical number comes from reading
``results/tables/horizon_sweep.parquet`` and ``results/tables/table_vii.parquet``
(both already on disk), or from calling ``compute_empirical_confusion_matrix``
(a deterministic classifier-inference pass over the already-trained model,
not an episode simulation).

Tiers (see docs/theory/crossover_derivation.md for the full validation-grid
rationale)
-----------------------------------------------------------------------------
TIER 1 (primary): every ``method="thompson"`` cell actually present in
``horizon_sweep.parquet``, at BOTH ``classifier_id in {"perfect","real"}``.
The committed file contains 3 environments (``canonical``,
``gap0.20_idx00``, ``gap0.20_idx01``) x 2 classifiers = 6 cells -- NOT the
full 26-environment grid ``horizon_sweep.py``'s own defaults would produce
(``DEFAULT_N_GAP_MATRICES=25``); the other 23 gap-matrix environments are
absent from the committed artifact (a documented, cost-gated reduced run --
see docs/experiments/instrumentation.md). Every expected-but-missing cell
is named explicitly below, never silently dropped. C=I (perfect) and C!=I
(real) results are ALWAYS reported and aggregated separately -- pooling
them into one headline number would hide exactly the comparison this tier
exists to make.

TIER 2 (secondary, static-side only): ``table_vii.parquet``'s 6 noise
levels validate ``static_rate_confused`` against real simulated rates.
Table VII contains no adaptive-method trajectory, so it cannot and does not
validate the crossover-horizon prediction -- stated explicitly in the
output, never implied otherwise.

Unit/self-consistency checks (monotonicity, exact C=I reduction, canonical
analytic agreement, regression pins) live in
``tests/test_crossover_theory.py``, not here -- they are required tests,
not empirical validation evidence, per the same distinction Tier 1/2 draws
against the synthetic sweep this script deliberately does NOT include.

Usage
-----
    python experiments/crossover_theory_validation.py [--smoke]
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from horizon_sweep import DEFAULT_BASE_SEED as HORIZON_BASE_SEED  # noqa: E402
from horizon_sweep import DEFAULT_GAP, DEFAULT_N_GAP_MATRICES  # noqa: E402
from horizon_sweep import build_grid as build_horizon_grid  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.theory.analytic import oracle_rate, uniform_confusion  # noqa: E402
from recovery.theory.crossover import (  # noqa: E402
    argmax_inversion_check,
    asymptotic_regime_flag,
    ceiling_reduction,
    crossover_horizon_confused,
    gap_shrinkage_report,
    static_rate_confused,
)
from src.evaluation.classifier_context import (  # noqa: E402
    compute_empirical_confusion_matrix,
    interpolated_confusion_matrix,
    load_or_train_classifier,
)

DEFAULT_OUTPUT_DIR = _BACKEND_DIR / "results" / "phase10" / "crossover_theory"
SMOKE_OUTPUT_DIR = _BACKEND_DIR / "results" / "phase10" / "crossover_theory_smoke"
HORIZON_SWEEP_PATH = _BACKEND_DIR / "results" / "tables" / "horizon_sweep.parquet"
TABLE_VII_PATH = _BACKEND_DIR / "results" / "tables" / "table_vii.parquet"

ASYMPTOTIC_MULTIPLE = 10.0  # see recovery.theory.crossover.asymptotic_regime_flag


# ======================================================================
# TIER 1 -- primary empirical validation (horizon_sweep.parquet)
# ======================================================================


def load_tier1_env_matrices() -> dict:
    """Reconstruct env_id -> matrix EXACTLY as horizon_sweep.py's own
    build_grid built them (same base_seed, same gap-sampling sequence) --
    no new simulation, no re-derivation of the sampling logic, a direct
    reuse of the existing function with t_max/n_seeds set to the cheapest
    values that still produce every RunConfig's .matrix attribute."""
    cells = build_horizon_grid(
        methods=["static_blind"], classifiers=["perfect"], n_seeds=1,
        base_seed=HORIZON_BASE_SEED, t_max=1, n_gap_matrices=DEFAULT_N_GAP_MATRICES, gap=DEFAULT_GAP,
    )
    lookup = {}
    for c in cells:
        lookup.setdefault(c.env_id, c.matrix)
    return lookup


def tier1_verify_availability(horizon_sweep_df: pd.DataFrame) -> tuple:
    """For every (env_id, classifier_id) the full default grid WOULD have
    produced, check whether it is actually present in the committed
    horizon_sweep.parquet, with a non-NaN time_to_parity for method=thompson.

    Returns
    -------
    (available, missing) : tuple[list[dict], list[dict]]
        ``available`` entries carry the exact cached seed count.
        ``missing`` entries name the cell and the exact reason it was
        excluded -- never silently dropped.
    """
    expected_envs = ["canonical"] + [f"gap0.20_idx{i:02d}" for i in range(DEFAULT_N_GAP_MATRICES)]
    expected = [(env, clf) for env in expected_envs for clf in ("perfect", "real")]

    thompson = horizon_sweep_df[horizon_sweep_df["method"] == "thompson"]

    available, missing = [], []
    for env_id, classifier_id in expected:
        cell = thompson[(thompson["env_id"] == env_id) & (thompson["classifier_id"] == classifier_id)]
        if cell.empty:
            missing.append({
                "env_id": env_id, "classifier_id": classifier_id,
                "reason": "env_id/classifier_id combination absent from committed horizon_sweep.parquet",
            })
            continue
        if cell["time_to_parity"].isna().all():
            missing.append({
                "env_id": env_id, "classifier_id": classifier_id,
                "reason": "present but time_to_parity is entirely NaN for this cell",
            })
            continue
        n_seeds = int(cell["seed"].nunique())
        available.append({"env_id": env_id, "classifier_id": classifier_id, "n_seeds": n_seeds})
    return available, missing


def tier1_predict_and_compare(
    available: list, horizon_sweep_df: pd.DataFrame, env_matrix_lookup: dict, confusion_real: np.ndarray,
) -> pd.DataFrame:
    thompson = horizon_sweep_df[horizon_sweep_df["method"] == "thompson"]
    N = MATRIX.shape[0]
    rows = []
    for cell in available:
        env_id, classifier_id, n_seeds = cell["env_id"], cell["classifier_id"], cell["n_seeds"]
        P = env_matrix_lookup[env_id]
        confusion = np.eye(N) if classifier_id == "perfect" else confusion_real
        quality = int(np.sum(BLIND_RULE == P.argmax(axis=1)))

        T_pred, truncated = crossover_horizon_confused(P, BLIND_RULE, confusion)

        cell_rows = thompson[(thompson["env_id"] == env_id) & (thompson["classifier_id"] == classifier_id)]
        per_seed_crossing = cell_rows.groupby("seed")["time_to_parity"].first()
        observed_T = float(per_seed_crossing.median())

        if math.isfinite(T_pred) and math.isfinite(observed_T) and observed_T > 0:
            ratio = T_pred / observed_T
            log_ratio = math.log(ratio)
            abs_rel_error = abs(ratio - 1.0)
        else:
            ratio = log_ratio = abs_rel_error = float("nan")

        rows.append({
            "env_id": env_id, "classifier_id": classifier_id, "n_seeds": n_seeds,
            "quality": quality, "N": N, "K": P.shape[1],
            "predicted_T": T_pred, "prediction_truncated": truncated,
            "observed_T": observed_T, "ratio": ratio, "log_ratio": log_ratio,
            "abs_rel_error": abs_rel_error,
            "asymptotic_defensible": asymptotic_regime_flag(T_pred, N, ASYMPTOTIC_MULTIPLE),
        })
    return pd.DataFrame(rows)


def tier1_summary(tier1_df: pd.DataFrame) -> pd.DataFrame:
    """Separate summaries for classifier_id='perfect' (C=I, backward-
    compatibility sanity check) and 'real' (C!=I, the actual scientific
    validation). NEVER pooled into one headline number."""
    out = []
    for classifier_id, g in tier1_df.groupby("classifier_id"):
        finite = g[np.isfinite(g["ratio"])]
        out.append({
            "classifier_id": classifier_id,
            "n_cells": len(g),
            "n_finite_ratio": len(finite),
            "median_abs_rel_error": float(finite["abs_rel_error"].median()) if len(finite) else float("nan"),
            "mean_log_ratio": float(finite["log_ratio"].mean()) if len(finite) else float("nan"),
            "median_log_ratio": float(finite["log_ratio"].median()) if len(finite) else float("nan"),
            "signed_direction": (
                "over-predicts (T_pred > observed on average)" if len(finite) and finite["log_ratio"].mean() > 0
                else "under-predicts (T_pred < observed on average)" if len(finite)
                else "n/a"
            ),
        })
    return pd.DataFrame(out)


# ======================================================================
# TIER 2 -- secondary static-side validation (table_vii.parquet)
# ======================================================================


def tier2_static_validation(table_vii_df: pd.DataFrame, c_emp: np.ndarray) -> pd.DataFrame:
    """Validates static_rate_confused ONLY. Table VII has no adaptive
    trajectory (method column is 'Fixed Rule' exclusively) -- this tier
    cannot and does not validate crossover_horizon_confused."""
    rows = []
    for noise_level, g in table_vii_df.groupby("noise_level"):
        C = interpolated_confusion_matrix(c_emp, noise_level)
        predicted_rate = oracle_rate(MATRIX) - static_rate_confused(MATRIX, BLIND_RULE, C)
        empirical_rate = float(g["raw_rate"].mean())
        n_seeds = int(g["seed"].nunique())
        rows.append({
            "noise_level": noise_level, "n_seeds": n_seeds,
            "predicted_rate": predicted_rate, "empirical_rate": empirical_rate,
            "abs_error": abs(predicted_rate - empirical_rate),
        })
    return pd.DataFrame(rows).sort_values("noise_level")


# ======================================================================
# Amendment B -- two noise channels + argmax-inversion check
# ======================================================================


def channel_analysis(confusion_real: np.ndarray) -> dict:
    ceiling = ceiling_reduction(MATRIX, confusion_real)
    gaps = gap_shrinkage_report(MATRIX, confusion_real)
    gap_df = pd.DataFrame(gaps)
    return {
        "ceiling_reduction": ceiling,
        "gap_shrinkage_per_bucket": gap_df,
        "mean_gap_shrinkage": float(gap_df["gap_shrinkage"].mean()),
    }


def argmax_inversion_analysis(confusion_real: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(argmax_inversion_check(MATRIX, confusion_real))


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                         help="Use a synthetic uniform_confusion in place of the real classifier's "
                              "empirical confusion matrix, skipping classifier load/inference time.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    if args.smoke:
        print("--smoke: using uniform_confusion(N, 0.9) in place of the real trained classifier "
              "(same convention as experiments/classifier_noise_sweep.py's own --smoke path)")
        c_emp = uniform_confusion(MATRIX.shape[0], 0.9)
    else:
        print("Loading trained classifier and computing its empirical confusion matrix "
              "(deterministic classifier inference, not a new simulation)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = load_or_train_classifier()
            c_emp = compute_empirical_confusion_matrix(clf)
    print(f"Real/effective confusion matrix, mean diagonal (accuracy) = {np.diag(c_emp).mean():.4f}")

    # --- Tier 1 ---
    print("\n=== Tier 1: primary empirical validation (horizon_sweep.parquet) ===")
    horizon_sweep_df = pd.read_parquet(HORIZON_SWEEP_PATH)
    available, missing = tier1_verify_availability(horizon_sweep_df)
    print(f"Available cells: {len(available)} / 52 expected (26 environments x 2 classifiers)")
    if missing:
        print(f"MISSING cells ({len(missing)}), named explicitly, excluded from Tier 1:")
        for m in missing[:5]:
            print(f"  {m['env_id']} / {m['classifier_id']}: {m['reason']}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more (full list in tier1_missing_cells.parquet)")
    pd.DataFrame(missing).to_parquet(output_dir / "tier1_missing_cells.parquet", index=False)
    pd.DataFrame(available).to_parquet(output_dir / "tier1_available_cells.parquet", index=False)

    env_matrix_lookup = load_tier1_env_matrices()
    tier1_df = tier1_predict_and_compare(available, horizon_sweep_df, env_matrix_lookup, c_emp)
    tier1_df.to_parquet(output_dir / "tier1_predictions.parquet", index=False)
    tier1_sum = tier1_summary(tier1_df)
    tier1_sum.to_parquet(output_dir / "tier1_summary.parquet", index=False)
    print(tier1_sum.to_string(index=False))
    print("\nPer-cell detail:")
    print(tier1_df[["env_id", "classifier_id", "n_seeds", "quality", "predicted_T",
                     "observed_T", "ratio", "asymptotic_defensible"]].to_string(index=False))

    # --- Amendment C: asymptotic-regime diagnostic ---
    print("\n=== Amendment C: asymptotic-regime diagnostic ===")
    finite = tier1_df[np.isfinite(tier1_df["abs_rel_error"])]
    if len(finite) >= 3:
        corr = float(finite["observed_T"].corr(finite["abs_rel_error"]))
        corr_log = float(np.log(finite["observed_T"]).corr(finite["abs_rel_error"]))
        print(f"corr(observed_T, abs_rel_error) = {corr:.4f}")
        print(f"corr(log(observed_T), abs_rel_error) = {corr_log:.4f}")
    else:
        corr = corr_log = float("nan")
        print(f"Only {len(finite)} finite-ratio cells -- correlation not meaningfully computable, reporting NaN, "
              f"not fabricating a number from an underpowered sample.")
    print(finite[["env_id", "classifier_id", "observed_T", "asymptotic_defensible", "abs_rel_error"]]
          .sort_values("observed_T").to_string(index=False))
    pd.DataFrame([{"corr_T_vs_error": corr, "corr_logT_vs_error": corr_log, "n": len(finite)}]).to_parquet(
        output_dir / "asymptotic_diagnostic.parquet", index=False
    )

    # --- Tier 2 ---
    print("\n=== Tier 2: secondary static-side validation (table_vii.parquet) ===")
    print("NOTE: Table VII contains no adaptive-method trajectory -- this tier validates "
          "static_rate_confused ONLY, never crossover_horizon_confused.")
    table_vii_df = pd.read_parquet(TABLE_VII_PATH)
    tier2_df = tier2_static_validation(table_vii_df, c_emp)
    tier2_df.to_parquet(output_dir / "tier2_static_validation.parquet", index=False)
    print(tier2_df.to_string(index=False))
    print(f"Tier 2 median absolute error: {tier2_df['abs_error'].median():.4f}")

    # --- Amendment B: channels + argmax inversion ---
    print("\n=== Amendment B: classifier-noise channels ===")
    channels = channel_analysis(c_emp)
    print(f"Channel 1 (ceiling reduction): {channels['ceiling_reduction']:.4f}")
    print(f"Channel 2 (mean gap shrinkage across predicted buckets): {channels['mean_gap_shrinkage']:.4f}")
    channels["gap_shrinkage_per_bucket"].to_parquet(output_dir / "gap_shrinkage_per_bucket.parquet", index=False)
    pd.DataFrame([{"ceiling_reduction": channels["ceiling_reduction"],
                   "mean_gap_shrinkage": channels["mean_gap_shrinkage"]}]).to_parquet(
        output_dir / "channel_summary.parquet", index=False
    )

    print("\n=== Amendment B: argmax-inversion check ===")
    inv_df = argmax_inversion_analysis(c_emp)
    inv_df.to_parquet(output_dir / "argmax_inversion.parquet", index=False)
    n_inversions = int(inv_df["is_inversion"].sum())
    print(f"Inversions found at the real classifier's empirical confusion: {n_inversions} / {len(inv_df)} buckets")
    print(inv_df.to_string(index=False))

    print(f"\nAll outputs written to {output_dir}")


if __name__ == "__main__":
    main()
