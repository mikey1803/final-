#!/usr/bin/env python
"""Task 10: practitioner-facing lookup tables beyond the simulated grid.

Task 6's empirical grid (and therefore ``crossover_calibration.py``'s
validation) only ever measured a 7-type, 6-strategy taxonomy
(``recovery.envs.canonical.MATRIX``'s shape, reused by every sampled gap
matrix in ``horizon_sweep.py``). This script sweeps taxonomy sizes a real
deployment might actually have (``N in {7, 14, 28, 56}``) using
``recovery.theory.crossover.predict_crossover_grid`` -- but NEVER
presents an N != 7 prediction as validated: every row is explicitly
flagged ``is_extrapolation`` (taxonomy size, or a quality/gap combination
outside what the calibration grid actually observed), and
``validated_region`` is ``True`` ONLY for rows that are both
non-extrapolated AND fall in a gap-bucket ``crossover_calibration.py``
itself marked reliable (read from ``results/tables/crossover_calibration.parquet``,
never re-derived independently).

Task 10B additionally attaches ``prediction_status`` (the finer four-way
``{validated, partially_validated, unsupported, extrapolated}``
classification -- see ``crossover_reliability.classify_status``, the
SAME function ``crossover_figures.py`` uses, so a row is never
classified one way in a figure and another way here) and
``confidence_level`` (``{High, Medium, Low}``, collapsed directly from
``prediction_status``). ``is_extrapolation``/``validated_region`` are
kept for backward compatibility with the rest of Task 10 -- they are the
coarse two-way projection of the same underlying classification, not a
second, independently-derived judgment.

Usage
-----
    python experiments/crossover_extension.py
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from recovery.theory.crossover import predict_crossover_grid  # noqa: E402
from crossover_reliability import build_calibration_context, classify_status  # noqa: E402

DEFAULT_CALIBRATION_INPUT = _BACKEND_DIR / "results" / "tables" / "crossover_calibration.parquet"
DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "crossover_lookup.parquet"

EXTENSION_N: list[int] = [7, 14, 28, 56]
EXTENSION_GAPS: list[float] = [0.05, 0.10, 0.20, 0.30]
EXTENSION_K = 6  # not swept by Task 10's spec; fixed at the canonical taxonomy's strategy count
N_MATRICES_PER_CELL = 15
SEED = 20261 - 1979  # arbitrary, fixed for reproducibility
FORM = "kl"
# c is NOT fixed here -- build_lookup_table defaults to the CALIBRATED
# best-fit c read from crossover_calibration.parquet (see its docstring).

# t_min must exceed the largest N swept (see recovery.theory.crossover's
# module docstring, "t_min must exceed the taxonomy size N") -- 56 is the
# largest N here, so 100 leaves comfortable margin.
T_MIN = 100
T_MAX = 1e9

VALIDATED_TAXONOMY_SIZE = 7  # the only N Task 6's empirical grid ever measured


def build_lookup_table(
    calibration_path: Path = DEFAULT_CALIBRATION_INPUT,
    n_list: list = EXTENSION_N,
    gaps: list = EXTENSION_GAPS,
    K: int = EXTENSION_K,
    n_matrices: int = N_MATRICES_PER_CELL,
    seed: int = SEED,
    form: str = FORM,
    c: Optional[float] = None,
) -> pd.DataFrame:
    """Build the practitioner-facing lookup table.

    ``c`` defaults to ``None``, meaning "use the CALIBRATED best-fit
    constant from ``crossover_calibration.parquet``" -- the whole point
    of calibrating is to make ``prediction`` the best available estimate,
    not the raw ``c=1`` formula value (which Section 4 of
    ``docs/theory/crossover.md`` shows is off by 1-2 orders of magnitude
    on this project's own calibration grid). Pass an explicit ``c`` to
    override (e.g. ``c=1.0`` to reproduce the uncalibrated formula for
    comparison).

    Note: ``calibration_path`` is accepted for backward-compatible
    signature purposes but the actual reliability classification is
    driven by ``crossover_reliability.build_calibration_context()``,
    which reads Task 10's calibration data directly (the same source
    ``calibration_path`` points at) -- both always agree by construction.
    """
    _rows, _summary, context, best_fit_c = build_calibration_context(form=form)
    c_used = best_fit_c if c is None else c

    rows: list[dict] = []
    for N in n_list:
        quality_levels = list(range(0, N + 1))  # "entire requested range" for this N
        raw = predict_crossover_grid(
            quality_levels=quality_levels,
            gaps=gaps,
            K=K,
            N=N,
            n_matrices=n_matrices,
            seed=seed + N,  # decorrelate draws across N
            form=form,
            c=c_used,
            t_min=T_MIN,
            t_max=T_MAX,
        )
        raw_df = pd.DataFrame(raw)

        for (gap, quality), sub in raw_df.groupby(["gap", "quality"]):
            T_vals = sub["T_star"].to_numpy()
            finite = T_vals[np.isfinite(T_vals)]
            prediction = float(np.median(finite)) if finite.size else math.inf
            n_inf = int(np.sum(~np.isfinite(T_vals)))
            trunc_frac = float(sub["prediction_truncated"].mean())

            status, confidence = classify_status(N, gap, int(quality), context)
            is_extrapolation = status == "extrapolated"
            validated = status == "validated"

            rows.append(
                {
                    "N": N,
                    "K": K,
                    "gap": gap,
                    "quality": quality,
                    "n_matrices": int(len(sub)),
                    "n_never_crossed": n_inf,
                    "prediction": prediction,
                    "prediction_truncated": trunc_frac,
                    "is_extrapolation": is_extrapolation,
                    "validated_region": validated,
                    "prediction_status": status,
                    "confidence_level": confidence,
                    "form": form,
                    "c": c_used,
                    "calibrated_best_fit_c": best_fit_c,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-input", type=Path, default=DEFAULT_CALIBRATION_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    lookup_df = build_lookup_table(calibration_path=args.calibration_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.to_parquet(args.output, index=False)
    print(f"Wrote {len(lookup_df)} rows to {args.output}")

    n_extrap = int(lookup_df["is_extrapolation"].sum())
    n_validated = int(lookup_df["validated_region"].sum())
    print(f"  {n_extrap}/{len(lookup_df)} rows are extrapolations")
    print(f"  {n_validated}/{len(lookup_df)} rows fall in a validated region")
    print(f"\n  prediction_status counts:\n{lookup_df['prediction_status'].value_counts().to_string()}")
    print(f"\n  confidence_level counts:\n{lookup_df['confidence_level'].value_counts().to_string()}")
    print("\nSample (N=7 rows only):")
    print(
        lookup_df[lookup_df["N"] == VALIDATED_TAXONOMY_SIZE]
        .sample(min(10, len(lookup_df[lookup_df["N"] == VALIDATED_TAXONOMY_SIZE])), random_state=0)
        .sort_values(["gap", "quality"])
        .to_string(index=False, float_format="{:.2f}".format)
    )


if __name__ == "__main__":
    main()
