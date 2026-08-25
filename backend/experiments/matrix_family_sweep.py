#!/usr/bin/env python
"""Table VI: matrix-family robustness sweep, regenerated through the
Task 2 shared simulation kernel.

Thompson Sampling vs UCB1 across a family of randomized reward matrices
(not just the single canonical matrix), at four optimal-arm gaps. Matrix
generation reuses ``src.evaluation.matrix_family.sample_matrix`` (pure
numpy, not kernel/analytic logic) -- everything downstream of a sampled
matrix flows through ``recovery.analysis.tables.run_trial``, the sole
caller of ``recovery.core.episode.run_episodes`` and
``recovery.guards.invariants.run_all_guards`` in this task. See
docs/reconciliation_stationary.md for how these numbers compare to the
historical `results/phase3a/matrix_family_summary.csv`.

Usage
-----
    python experiments/matrix_family_sweep.py [--smoke] [--output PATH]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.tables import aggregate_and_check, run_trial, write_tidy_table  # noqa: E402
from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.protocols import Policy  # noqa: E402
from recovery.policies.adapters import ThompsonPolicy, UCB1Policy  # noqa: E402
from src.evaluation.matrix_family import sample_matrix  # noqa: E402

DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "table_vi.parquet"
DEFAULT_GAPS = [0.05, 0.10, 0.20, 0.30]
DEFAULT_N_MATRICES = 100
DEFAULT_N_SEEDS = 20
DEFAULT_N_EPISODES = 150
DEFAULT_BASE_SEED = 2026

SMOKE_GAPS = [0.10, 0.30]
SMOKE_N_MATRICES = 3
SMOKE_N_SEEDS = 3
SMOKE_N_EPISODES = DEFAULT_N_EPISODES  # keep episode count (guard validity), reduce breadth

_ALGORITHMS: dict[str, tuple[int, Callable[[int], Policy]]] = {
    "Adaptive (Thompson)": (500, lambda s: ThompsonPolicy(seed=s)),
    "Adaptive (UCB1)": (0, lambda s: UCB1Policy(seed=s)),
}


def run_table_vi(
    gaps: list[float], n_matrices: int, n_seeds: int, n_episodes: int, base_seed: int
) -> list[dict]:
    rows: list[dict] = []
    matrix_rng = np.random.default_rng(base_seed)

    for gap in gaps:
        for matrix_idx in range(n_matrices):
            matrix = sample_matrix(gap=gap, rng=matrix_rng)
            matrix_rows: list[dict] = []

            for algo_name, (offset, factory) in _ALGORITHMS.items():
                for seed_idx in range(n_seeds):
                    env_seed = base_seed + 100_000 * matrix_idx + 1_000 * seed_idx
                    policy = factory(env_seed + offset)
                    result = run_trial(
                        matrix, policy, PerfectClassifier(), n_episodes=n_episodes, seed=env_seed
                    )
                    row = {
                        "method": algo_name, "gap": gap, "matrix_idx": matrix_idx,
                        "seed": seed_idx, **result,
                    }
                    rows.append(row)
                    matrix_rows.append(row)

            # Guard-check this matrix's aggregate (across seeds, per algorithm)
            # against ITS OWN oracle/random/worst-case bounds -- see
            # aggregate_and_check's docstring: every row passed to one call
            # must share a single ground-truth matrix.
            aggregate_and_check(pd.DataFrame(matrix_rows), ["method"], matrix)

        print(f"  gap={gap:.2f}: done ({n_matrices} matrices x {n_seeds} seeds x {len(_ALGORITHMS)} algorithms)")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast validation run (<10s).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        # See stationary_comparison.py's identical fix note: --smoke must
        # never default to the same path as a full run (Task 14 incident).
        args.output = (
            DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_smoke.parquet")
            if args.smoke else DEFAULT_OUTPUT
        )

    if args.smoke:
        gaps, n_matrices, n_seeds, n_episodes = SMOKE_GAPS, SMOKE_N_MATRICES, SMOKE_N_SEEDS, SMOKE_N_EPISODES
    else:
        gaps, n_matrices, n_seeds, n_episodes = DEFAULT_GAPS, DEFAULT_N_MATRICES, DEFAULT_N_SEEDS, DEFAULT_N_EPISODES

    print(f"Table VI: matrix-family sweep ({'smoke' if args.smoke else 'full'} mode)")
    print(f"  gaps={gaps}  n_matrices={n_matrices}  n_seeds={n_seeds}  n_episodes={n_episodes}")

    t0 = time.time()
    rows = run_table_vi(gaps, n_matrices, n_seeds, n_episodes, DEFAULT_BASE_SEED)
    elapsed = time.time() - t0

    df = write_tidy_table(rows, args.output)
    print(f"\nWrote {len(df)} rows to {args.output} in {elapsed:.2f}s")
    print(
        df.groupby(["gap", "method"])[["raw_rate", "normalized"]]
        .mean()
        .to_string(float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
