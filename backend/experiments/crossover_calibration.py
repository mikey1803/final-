#!/usr/bin/env python
"""Task 10: calibrate the analytic crossover predictor against Task 6 data.

Reads ``results/tables/horizon_crossover.parquet`` (see
``crossover_calibration_grid.py`` for how -- and why -- that file was
regenerated for this task), reconstructs each row's success-probability
matrix ``P`` and the ``static_blind`` table ``BLIND_RULE`` it was measured
against, and compares ``recovery.theory.crossover.predict_crossover``'s
prediction to the empirically measured crossover.

Which empirical crossover this predicts
-----------------------------------------
``predict_crossover`` solves ``R_TS(T) = rho * T`` -- CUMULATIVE TS regret
equal to CUMULATIVE static regret. Equal cumulative regret against the
same oracle means equal cumulative reward, which is exactly Task 6's
``criterion == "cumulative"`` column (``recovery.metrics.parity.time_to_parity``),
not ``"asymptotic"`` (a per-episode-RATE crossing, a different question --
see ``docs/experiments/horizon.md``, "Two crossover notions"). This
script therefore calibrates against ``criterion == "cumulative"`` only;
using ``"asymptotic"`` here would be comparing the predictor to a
quantity it was never derived to predict.

Usage
-----
    python experiments/crossover_calibration.py
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from crossover_calibration_grid import build_calibration_grid  # noqa: E402
from recovery.envs.canonical import BLIND_RULE  # noqa: E402
from recovery.theory.crossover import predict_crossover, static_regret_rate  # noqa: E402

DEFAULT_INPUT = _BACKEND_DIR / "results" / "tables" / "horizon_crossover.parquet"
DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "crossover_calibration.parquet"

# A group is reported "validated" if at least this fraction of its
# CALIBRATED predictions land within 3x of the observed crossover -- an
# explicit, documented threshold (see docs/theory/crossover.md), not a
# claim of exactness.
VALIDATED_WITHIN_FACTOR3_THRESHOLD = 0.5

C_SEARCH_LOG_BOUNDS = (math.log(1e-3), math.log(1e4))


def _env_matrix_lookup() -> dict:
    """env_id -> P, from the SAME deterministic build used to generate
    ``horizon_crossover.parquet`` (see ``crossover_calibration_grid.py``)
    -- never re-derived independently, so a reconstructed matrix is
    guaranteed identical to the one actually simulated."""
    cells = build_calibration_grid()
    lookup: dict = {}
    for cell in cells:
        lookup.setdefault(cell.env_id, cell.matrix)
    return lookup


def load_calibration_rows(input_path: Path = DEFAULT_INPUT) -> list[dict]:
    """Build one dict per (env_id) with everything ``predict_crossover``
    and the calibration summary need: ``P``, ``pi`` (``BLIND_RULE``),
    ``rho``, ``quality``, ``gap``, ``N``, ``K``, and the empirically
    observed cumulative-crossover ``observed_T`` (its median across
    seeds), plus ``fraction_never_crossed``/``n_seeds`` for transparency.

    Rows whose median crossover is ``inf`` (the learner never caught up
    within the T_max=100,000 simulated) are KEPT, with
    ``observed_T=inf`` -- excluded from ratio statistics later (a ratio
    against infinity is undefined), never silently dropped here.
    """
    df = pd.read_parquet(input_path)
    df = df[(df["method"] == "thompson") & (df["criterion"] == "cumulative")]
    if df.empty:
        raise ValueError(
            f"no method='thompson', criterion='cumulative' rows found in {input_path} -- "
            f"cannot calibrate against an empty table"
        )

    matrix_lookup = _env_matrix_lookup()
    rows: list[dict] = []
    for _, row in df.iterrows():
        env_id = row["env_id"]
        if env_id not in matrix_lookup:
            raise ValueError(f"env_id={env_id!r} has no matching sampled matrix -- grid mismatch")
        P = matrix_lookup[env_id]
        pi = BLIND_RULE
        n_types = P.shape[0]
        quality = int(np.sum(pi == P.argmax(axis=1)))
        gap = float("nan") if env_id == "canonical" else float(env_id.split("_idx")[0].replace("gap", ""))
        rows.append(
            {
                "env_id": env_id,
                "P": P,
                "pi": pi,
                "gap": gap,
                "quality": quality,
                "N": n_types,
                "K": P.shape[1],
                "rho": static_regret_rate(P, pi),
                "observed_T": float(row["median"]),
                "fraction_never_crossed": float(row["fraction_never_crossed"]),
                "n_seeds": int(row["n_seeds"]),
            }
        )
    return rows


def predicted_ratio_table(rows: list[dict], form: str = "kl", c: float = 1.0) -> pd.DataFrame:
    """Per-cell predicted-vs-observed table, with a ``ratio`` column
    (``predicted_T / observed_T``). ``ratio`` is ``NaN`` wherever either
    side is non-finite (observed never crossed, or the prediction was
    truncated at ``t_max`` -- see ``prediction_truncated``) -- never a
    fabricated finite number standing in for "unknown."
    """
    out = []
    for row in rows:
        T_pred, truncated = predict_crossover(row["P"], row["pi"], form=form, c=c)
        observed = row["observed_T"]
        if math.isfinite(T_pred) and math.isfinite(observed) and observed > 0:
            ratio = T_pred / observed
        else:
            ratio = float("nan")
        out.append(
            {
                "env_id": row["env_id"],
                "gap": row["gap"],
                "quality": row["quality"],
                "N": row["N"],
                "K": row["K"],
                "rho": row["rho"],
                "observed_T": observed,
                "predicted_T": T_pred,
                "prediction_truncated": truncated,
                "ratio": ratio,
                "form": form,
                "c": c,
            }
        )
    return pd.DataFrame(out)


def fit_best_c(rows: list[dict], form: str = "kl") -> float:
    """Fit the multiplicative constant ``c`` minimizing mean squared
    log-space prediction error, ``mean((log(predicted_T(c)) - log(observed_T))^2)``,
    via bounded 1-D minimization over ``log(c)`` (``c`` enters
    ``ts_cumulative_regret`` linearly and ``predict_crossover``'s
    solution is monotonic in ``c`` -- see ``recovery.theory.crossover``'s
    docstrings -- so this is a well-behaved scalar optimization). Cells
    with a non-finite observed or predicted value are excluded from the
    loss (never coerced to a finite stand-in).

    Returns
    -------
    float
        The best-fit ``c``. NOT a claim that the true regret bound's
        constant equals this value -- only that it minimizes THIS
        empirical log-space error against Task 10's specific calibration
        grid (see docs/theory/crossover.md's honesty caveats).
    """
    finite_observed = [r for r in rows if math.isfinite(r["observed_T"]) and r["observed_T"] > 0]
    if not finite_observed:
        raise ValueError("no finite observed crossovers to calibrate against")

    def loss(log_c: float) -> float:
        c = math.exp(log_c)
        sq_err = 0.0
        n = 0
        for row in finite_observed:
            T_pred, _truncated = predict_crossover(row["P"], row["pi"], form=form, c=c)
            if not math.isfinite(T_pred) or T_pred <= 0:
                continue
            sq_err += (math.log(T_pred) - math.log(row["observed_T"])) ** 2
            n += 1
        return sq_err / n if n > 0 else float("inf")

    result = minimize_scalar(loss, bounds=C_SEARCH_LOG_BOUNDS, method="bounded")
    return float(math.exp(result.x))


def summarize_ratios(per_cell_df: pd.DataFrame) -> dict:
    """Median/IQR/factor-2/factor-3 summary of a ``ratio`` column.
    Non-finite ratios are excluded from every statistic (their COUNT is
    reported separately, never silently dropped without a trace)."""
    all_n = len(per_cell_df)
    finite = per_cell_df["ratio"].to_numpy()
    finite = finite[np.isfinite(finite)]
    n_excluded = all_n - finite.size
    if finite.size == 0:
        return {
            "n": 0,
            "n_excluded_non_finite": n_excluded,
            "median": float("nan"),
            "iqr_low": float("nan"),
            "iqr_high": float("nan"),
            "iqr": float("nan"),
            "within_factor2": float("nan"),
            "within_factor3": float("nan"),
        }
    log_ratio = np.abs(np.log(finite))
    within2 = float(np.mean(log_ratio <= math.log(2.0)))
    within3 = float(np.mean(log_ratio <= math.log(3.0)))
    iqr_low, iqr_high = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
    return {
        "n": int(finite.size),
        "n_excluded_non_finite": int(n_excluded),
        "median": float(np.median(finite)),
        "iqr_low": iqr_low,
        "iqr_high": iqr_high,
        "iqr": iqr_high - iqr_low,
        "within_factor2": within2,
        "within_factor3": within3,
    }


def build_calibration_summary(rows: list[dict], form: str = "kl") -> pd.DataFrame:
    """Build the full calibration report: uncalibrated (``c=1``) vs.
    calibrated (best-fit ``c``) side by side, overall and broken down by
    gap and by quality -- reliability is never collapsed to one number
    that could hide a sub-regime where the predictor fails.

    Returns
    -------
    pandas.DataFrame
        One row per (group, calibration) pair, columns: ``group``, ``n``,
        ``n_excluded_non_finite``, ``median``, ``iqr_low``, ``iqr_high``,
        ``iqr``, ``within_factor2``, ``within_factor3``, ``c_used``,
        ``best_fit_c``, ``calibrated``, ``validated_region``, ``form``.
    """
    best_c = fit_best_c(rows, form=form)
    uncalibrated = predicted_ratio_table(rows, form=form, c=1.0)
    calibrated = predicted_ratio_table(rows, form=form, c=best_c)

    def _rows_for(df: pd.DataFrame, c_used: float, calibrated_flag: bool) -> list[dict]:
        groups: list[tuple] = [("overall", df)]
        for gap in sorted(df["gap"].dropna().unique()):
            groups.append((f"gap={gap:.2f}", df[df["gap"] == gap]))
        for q in sorted(df["quality"].unique()):
            groups.append((f"quality={q}", df[df["quality"] == q]))

        out = []
        for label, sub in groups:
            stats = summarize_ratios(sub)
            validated = (
                bool(stats["within_factor3"] >= VALIDATED_WITHIN_FACTOR3_THRESHOLD)
                if stats["n"] > 0
                else False
            )
            out.append(
                {
                    "group": label,
                    **stats,
                    "c_used": c_used,
                    "best_fit_c": best_c,
                    "calibrated": calibrated_flag,
                    "validated_region": validated,
                    "form": form,
                }
            )
        return out

    summary_rows = _rows_for(uncalibrated, 1.0, False) + _rows_for(calibrated, best_c, True)
    return pd.DataFrame(summary_rows)


def print_report(summary: pd.DataFrame) -> None:
    print("\n=== Crossover predictor calibration ===")
    for calibrated_flag, label in [(False, "UNCALIBRATED (c=1.0)"), (True, "CALIBRATED (best-fit c)")]:
        sub = summary[summary["calibrated"] == calibrated_flag]
        print(f"\n--- {label} ---")
        print(
            sub[
                ["group", "n", "n_excluded_non_finite", "median", "within_factor2", "within_factor3", "validated_region"]
            ].to_string(index=False, float_format="{:.3f}".format)
        )
    best_c = float(summary["best_fit_c"].iloc[0])
    print(f"\nBest-fit c = {best_c:.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--form", type=str, default="kl", choices=["kl", "gap"])
    args = parser.parse_args()

    rows = load_calibration_rows(args.input)
    print(f"Loaded {len(rows)} calibration cells (method='thompson', criterion='cumulative') from {args.input}")

    summary = build_calibration_summary(rows, form=args.form)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output, index=False)
    print(f"Wrote {len(summary)} rows to {args.output}")

    print_report(summary)


if __name__ == "__main__":
    main()
