#!/usr/bin/env python
"""Table VII: classifier-noise sweep, regenerated through the Task 2
shared simulation kernel -- this is the paper's anomaly experiment (see
docs/anomalies.md).

For the blind 6/7-correct rule (``recovery.envs.canonical.BLIND_RULE``),
sweeps beta in {1.0, 0.8, 0.5, 0.3, 0.1, 0.0} interpolating between the
REAL fitted classifier's empirical confusion matrix (beta=1.0) and pure
uniform noise (beta=0.0) -- reusing
``src.evaluation.classifier_context.interpolated_confusion_matrix`` and
``compute_empirical_confusion_matrix`` rather than recomputing them. Every
trial's empirical rate is compared against the analytic prediction from
``recovery.theory.analytic.policy_rate_under_confusion`` (via
``recovery.guards.invariants.check_confusion_prediction``, run inside
``recovery.analysis.tables.run_trial`` -- the sole caller of
``run_episodes``/``run_all_guards`` in this task). Stores empirical
success, analytic prediction, and signed difference for every level. See
docs/reconciliation_stationary.md, "Table VII Final Verdict".

Usage
-----
    python experiments/classifier_noise_sweep.py [--smoke] [--output PATH]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.tables import aggregate_and_check, run_trial, write_tidy_table  # noqa: E402
from recovery.classify.synthetic import ConfusionClassifier  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.policies.adapters import StaticRulePolicy  # noqa: E402
from recovery.theory.analytic import policy_rate_under_confusion, uniform_confusion  # noqa: E402
from src.evaluation.classifier_context import (  # noqa: E402
    compute_empirical_confusion_matrix,
    interpolated_confusion_matrix,
    load_or_train_classifier,
)

DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "table_vii.parquet"
DEFAULT_NOISE_LEVELS = [1.0, 0.8, 0.5, 0.3, 0.1, 0.0]
DEFAULT_N_EPISODES = 500
DEFAULT_N_SEEDS = 50
DEFAULT_BASE_SEED = 123

SMOKE_NOISE_LEVELS = [1.0, 0.0]
SMOKE_N_SEEDS = 2
SMOKE_N_EPISODES = DEFAULT_N_EPISODES  # keep episode count (guard validity), reduce breadth


def run_table_vii(
    noise_levels: list[float],
    n_episodes: int,
    n_seeds: int,
    base_seed: int,
    use_real_classifier: bool = True,
) -> list[dict]:
    if use_real_classifier:
        print("  Loading/training classifier and computing empirical confusion matrix...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn version-mismatch warning on the pickled artifact
            clf = load_or_train_classifier()
            c_emp = compute_empirical_confusion_matrix(clf)
    else:
        # --smoke: skip disk I/O + sklearn deserialization of the trained
        # classifier artifact entirely (its load time depends on external
        # disk/OS conditions, not on anything this experiment script
        # controls, and is irrelevant to what --smoke validates: pipeline
        # wiring, guard integration, and output schema). A synthetic
        # uniform-confusion matrix exercises every downstream code path
        # (interpolation, ConfusionClassifier, check_confusion_prediction)
        # identically; normal mode always uses the real classifier.
        c_emp = uniform_confusion(MATRIX.shape[0], 0.9)

    rows: list[dict] = []
    for beta in noise_levels:
        C = interpolated_confusion_matrix(c_emp, beta)
        analytic_prediction = policy_rate_under_confusion(MATRIX, BLIND_RULE, C)

        level_rows: list[dict] = []
        for seed_idx in range(n_seeds):
            env_seed = base_seed + seed_idx * 1000
            policy = StaticRulePolicy(BLIND_RULE)
            classifier = ConfusionClassifier(C)
            result = run_trial(
                MATRIX, policy, classifier, n_episodes=n_episodes, seed=env_seed,
                pi=BLIND_RULE, C=C,
            )
            row = {
                "method": "Fixed Rule", "noise_level": beta, "seed": seed_idx,
                **result,
                "analytic_prediction": analytic_prediction,
                "signed_diff": result["raw_rate"] - analytic_prediction,
            }
            rows.append(row)
            level_rows.append(row)

        aggregate_and_check(
            pd.DataFrame(level_rows), ["method"], MATRIX, C=C,
            pi_by_group={"Fixed Rule": BLIND_RULE},
        )

        mean_rate = sum(r["raw_rate"] for r in level_rows) / len(level_rows)
        print(
            f"  beta={beta:.1f}: empirical={mean_rate:.4f}  analytic={analytic_prediction:.4f}  "
            f"diff={mean_rate - analytic_prediction:+.4f}"
        )

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
        noise_levels, n_seeds, n_episodes = SMOKE_NOISE_LEVELS, SMOKE_N_SEEDS, SMOKE_N_EPISODES
    else:
        noise_levels, n_seeds, n_episodes = DEFAULT_NOISE_LEVELS, DEFAULT_N_SEEDS, DEFAULT_N_EPISODES

    print(f"Table VII: classifier-noise sweep ({'smoke' if args.smoke else 'full'} mode)")
    print(f"  noise_levels={noise_levels}  n_episodes={n_episodes}  n_seeds={n_seeds}")

    t0 = time.time()
    rows = run_table_vii(
        noise_levels, n_episodes, n_seeds, DEFAULT_BASE_SEED, use_real_classifier=not args.smoke
    )
    elapsed = time.time() - t0

    df = write_tidy_table(rows, args.output)
    print(f"\nWrote {len(df)} rows to {args.output} in {elapsed:.2f}s")
    print(
        df.groupby("noise_level")[["raw_rate", "analytic_prediction", "signed_diff"]]
        .mean()
        .to_string(float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
