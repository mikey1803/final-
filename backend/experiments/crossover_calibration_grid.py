#!/usr/bin/env python
"""Regenerate the Task 6 crossover-measurement input Task 10 calibrates against.

WHY THIS SCRIPT EXISTS
-----------------------
Task 10's calibration step (``crossover_calibration.py``) is specified to
read ``results/tables/horizon_crossover.parquet`` as an already-existing
Task 6 deliverable. At the time Task 10 started, that file did not exist
in the repository -- only a small 6-row smoke/validation artifact
(``horizon_crossover_validation.parquet``, canonical environment only, 3
methods) had ever been persisted. ``docs/experiments/horizon.md`` itself
documents why: the full default grid (11 methods x 2 classifiers x 30
seeds x 26 environments x T_max=100,000) is "a multi-hour undertaking,"
dominated almost entirely by the ``real`` (sklearn) classifier condition
(measured here at ``--estimate``: ~78.6 hours total, ~4699 of those
minutes from ``classifier_id='real'`` alone vs. ~17.7 minutes for
``classifier_id='perfect'``).

This script produces the missing artifact WITHOUT modifying
``horizon_sweep.py`` or ``horizon_analysis.py`` -- it imports and calls
their exported, unmodified functions (``build_grid``, ``run_grid``,
``evaluate_horizons``, ``compute_crossover_table``) with a scope chosen
specifically for Task 10's needs, and documents that scope honestly
rather than silently narrowing Task 6's own default grid:

1. **classifier_id='perfect' only.** Task 10's theory (``recovery.theory.crossover``)
   models a noiseless Bernoulli bandit against a static table -- it has no
   notion of classifier/confusion noise. Calibrating against the ``real``
   classifier's confusion-corrupted rates would test a variable the
   predictor doesn't even model, while being ~1,300x more expensive.
   Excluding it is a scope decision driven by what the theory can be
   validated against, not merely a runtime shortcut -- documented here and
   in ``docs/theory/crossover.md``.
2. **methods=['static_blind', 'thompson'] only.** The crossover predictor
   is specifically a Thompson-Sampling regret bound; the other 8 adapted
   policies in Task 6's default grid (UCB1, LinUCB, epsilon-greedy, ...)
   are out of scope for what this predictor claims to predict.
3. **Gap is swept explicitly.** ``horizon_sweep.build_grid`` applies ONE
   fixed ``gap`` to every one of its ``n_gap_matrices`` samples in a
   single call; Task 10 needs several distinct gaps to test the
   predictor's gap-dependent reliability. This script therefore calls
   ``build_grid`` four times (once per gap in ``CALIBRATION_GAPS``, each
   with its own ``gap_seed`` so the four groups' matrix draws are
   independent, not merely gap-rescaled copies of the same base draws)
   and concatenates the resulting cell lists before a single ``run_grid``
   call. The ``gap=0.20`` group reuses Task 6's own defaults exactly
   (``n_gap_matrices=25``, ``n_seeds=30``, plus the canonical environment)
   so it doubles as a faithful (if perfect-classifier-only) reproduction
   of Task 6's own headline condition; the other three gaps use a smaller
   ``n_gap_matrices=10``, ``n_seeds=20`` to keep total runtime modest.

Total: 2,760 cells, ~276,000,000 episodes, ~6-8 minutes at 8-way
parallelism on the profiling machine (see ``--estimate``).

Usage
-----
    python experiments/crossover_calibration_grid.py --estimate
    python experiments/crossover_calibration_grid.py
"""

import argparse
import sys
import time
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_EXPERIMENTS_DIR = _BACKEND_DIR / "experiments"
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from horizon_analysis import compute_crossover_table, print_summary  # noqa: E402
from horizon_sweep import build_grid, evaluate_horizons  # noqa: E402
from recovery.core.parallel import run_grid  # noqa: E402

DEFAULT_SWEEP_OUTPUT = _BACKEND_DIR / "results" / "tables" / "horizon_sweep_crossover_calibration.parquet"
DEFAULT_CROSSOVER_OUTPUT = _BACKEND_DIR / "results" / "tables" / "horizon_crossover.parquet"
DEFAULT_STORE_ROOT = _BACKEND_DIR / "results" / "runs" / "crossover_calibration"

CALIBRATION_METHODS = ["static_blind", "thompson"]
CALIBRATION_CLASSIFIERS = ["perfect"]
HORIZONS = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
T_MAX = 100_000
BASE_SEED = 4000

# (gap, n_gap_matrices, n_seeds, include_canonical)
CALIBRATION_GAPS: list[tuple] = [
    (0.20, 25, 30, True),   # matches horizon_sweep.py's own defaults exactly
    (0.05, 10, 20, False),
    (0.10, 10, 20, False),
    (0.30, 10, 20, False),
]


def build_calibration_grid():
    cells = []
    for gap, n_gap_matrices, n_seeds, include_canonical in CALIBRATION_GAPS:
        gap_seed = BASE_SEED + 999_999 + int(round(gap * 1000))  # decorrelate matrix draws across gaps
        cells.extend(
            build_grid(
                methods=CALIBRATION_METHODS,
                classifiers=CALIBRATION_CLASSIFIERS,
                n_seeds=n_seeds,
                base_seed=BASE_SEED,
                t_max=T_MAX,
                include_canonical=include_canonical,
                n_gap_matrices=n_gap_matrices,
                gap=gap,
                gap_seed=gap_seed,
            )
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--sweep-output", type=Path, default=DEFAULT_SWEEP_OUTPUT)
    parser.add_argument("--crossover-output", type=Path, default=DEFAULT_CROSSOVER_OUTPUT)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="loky")
    args = parser.parse_args()

    cells = build_calibration_grid()
    print(f"Crossover calibration grid: {len(cells)} cells (perfect classifier only; see module docstring)")

    if args.estimate:
        from horizon_sweep import estimate_wall_clock

        estimate_wall_clock(cells, n_jobs=args.n_jobs)
        return

    t0 = time.time()
    run_grid(cells, store_root=args.store_root, n_jobs=args.n_jobs, backend=args.backend)
    run_elapsed = time.time() - t0
    print(f"run_grid: {len(cells)} cells ready in {run_elapsed:.2f}s")

    sweep_df = evaluate_horizons(cells, [], HORIZONS, args.store_root)
    args.sweep_output.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_parquet(args.sweep_output, index=False)
    print(f"Wrote {len(sweep_df)} rows to {args.sweep_output}")

    crossover_df = compute_crossover_table(sweep_df)
    args.crossover_output.parent.mkdir(parents=True, exist_ok=True)
    crossover_df.to_parquet(args.crossover_output, index=False)
    total_elapsed = time.time() - t0
    print(f"Wrote {len(crossover_df)} rows to {args.crossover_output} in {total_elapsed:.2f}s total")

    print_summary(crossover_df)


if __name__ == "__main__":
    main()
