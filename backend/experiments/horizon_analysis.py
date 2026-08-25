#!/usr/bin/env python
"""Task 6: crossover analysis over the horizon-sweep output.

Reads ``results/tables/horizon_sweep.parquet`` (see ``horizon_sweep.py``)
and computes, for every adaptive method (everything except its own
``static_blind`` baseline and the ``oracle`` normalizer) x environment x
classifier condition, two DIFFERENT crossover notions -- see
docs/experiments/horizon.md, "Two crossover notions":

- **Asymptotic crossover**: where the per-episode success rate (evaluated
  as ``late_stage_rate`` at each horizon T) overtakes ``static_blind``'s,
  found via ``recovery.analysis.crossover_empirical.crossover_horizon``
  (log-linear interpolation on the horizon grid).
- **Cumulative crossover**: where TOTAL accumulated reward overtakes
  ``static_blind``'s and stays there -- this is exactly
  ``recovery.metrics.parity.time_to_parity``, already computed once per
  (method, env_id, classifier_id, seed) in ``horizon_sweep.py`` from the
  full T_max trajectories (per-episode granularity, not interpolated on
  the horizon grid -- a different, complementary algorithm from the
  asymptotic one above, not a re-derivation of it).

All crossover logic itself lives in ``recovery.analysis.crossover_empirical``
-- this script only groups/reshapes the sweep table and calls it.

Usage
-----
    python experiments/horizon_analysis.py [--input PATH] [--output PATH]
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.crossover_empirical import crossover_distribution, crossover_horizon  # noqa: E402

DEFAULT_INPUT = _BACKEND_DIR / "results" / "tables" / "horizon_sweep.parquet"
DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "horizon_crossover.parquet"

_EXCLUDED_FROM_CROSSOVER = {"static_blind", "oracle"}


def _fmt_t(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:,.0f}"


def compute_crossover_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-(method, env_id, classifier_id, criterion) crossover table.

    Parameters
    ----------
    df : pandas.DataFrame
        ``horizon_sweep.py``'s tidy output: one row per
        (method, env_id, classifier_id, seed, T), with columns
        ``late_stage_rate`` and ``time_to_parity`` among others.

    Returns
    -------
    pandas.DataFrame
        One row per (method, env_id, classifier_id, criterion) with
        ``median``, ``iqr_low``, ``iqr_high``, ``min``, ``max``,
        ``fraction_never_crossed``, ``n_seeds``.
    """
    horizons = sorted(df["T"].unique())
    rows: list[dict] = []

    groups = df[["method", "env_id", "classifier_id"]].drop_duplicates()
    for _, group_key in groups.iterrows():
        method, env_id, classifier_id = group_key["method"], group_key["env_id"], group_key["classifier_id"]
        if method in _EXCLUDED_FROM_CROSSOVER:
            continue

        static_sub = df[
            (df["method"] == "static_blind") & (df["env_id"] == env_id) & (df["classifier_id"] == classifier_id)
        ]
        method_sub = df[
            (df["method"] == method) & (df["env_id"] == env_id) & (df["classifier_id"] == classifier_id)
        ]
        if static_sub.empty:
            raise ValueError(
                f"no static_blind rows found for (env_id={env_id!r}, classifier_id={classifier_id!r}); "
                f"cannot compute crossover for method={method!r}"
            )

        # --- asymptotic crossover: per-seed crossover_horizon() over the late_stage_rate curve ---
        asymptotic_crossings: list[float] = []
        for seed in sorted(method_sub["seed"].unique()):
            learner_curve = (
                method_sub[method_sub["seed"] == seed].set_index("T").loc[horizons, "late_stage_rate"].to_numpy()
            )
            static_curve = (
                static_sub[static_sub["seed"] == seed].set_index("T").loc[horizons, "late_stage_rate"].to_numpy()
            )
            asymptotic_crossings.append(crossover_horizon(learner_curve, static_curve, horizons))
        asymptotic_dist = crossover_distribution(asymptotic_crossings)

        # --- cumulative crossover: time_to_parity, one value per seed (already
        # computed in horizon_sweep.py; identical across that seed's T rows) ---
        cumulative_crossings = (
            method_sub.drop_duplicates(subset=["seed"])["time_to_parity"].astype(float).tolist()
        )
        cumulative_dist = crossover_distribution(cumulative_crossings)

        for criterion, dist, n in (
            ("asymptotic", asymptotic_dist, len(asymptotic_crossings)),
            ("cumulative", cumulative_dist, len(cumulative_crossings)),
        ):
            rows.append(
                {
                    "method": method,
                    "env_id": env_id,
                    "classifier_id": classifier_id,
                    "criterion": criterion,
                    "n_seeds": n,
                    **dist,
                }
            )

    return pd.DataFrame(rows)


def print_summary(crossover_df: pd.DataFrame) -> None:
    """Print, for each adaptive method, the asymptotic crossover T, the
    cumulative crossover T, and the ratio between them (to stdout)."""
    print("\n=== Horizon crossover summary ===")
    keys = crossover_df[["method", "env_id", "classifier_id"]].drop_duplicates()
    for _, key in keys.iterrows():
        method, env_id, classifier_id = key["method"], key["env_id"], key["classifier_id"]
        sub = crossover_df[
            (crossover_df["method"] == method)
            & (crossover_df["env_id"] == env_id)
            & (crossover_df["classifier_id"] == classifier_id)
        ]
        asymptotic = sub[sub["criterion"] == "asymptotic"].iloc[0]
        cumulative = sub[sub["criterion"] == "cumulative"].iloc[0]
        a_med, c_med = asymptotic["median"], cumulative["median"]

        if math.isfinite(a_med) and math.isfinite(c_med) and a_med > 0:
            ratio_str = f"{c_med / a_med:.2f}x"
        else:
            ratio_str = "N/A (one or both crossovers never occur within the tested horizon)"

        print(
            f"{method} [{env_id}/{classifier_id}]: "
            f"asymptotic T={_fmt_t(a_med)} (never-crossed frac={asymptotic['fraction_never_crossed']:.2f}), "
            f"cumulative T={_fmt_t(c_med)} (never-crossed frac={cumulative['fraction_never_crossed']:.2f}), "
            f"cumulative/asymptotic ratio={ratio_str}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    crossover_df = compute_crossover_table(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    crossover_df.to_parquet(args.output, index=False)
    print(f"Wrote {len(crossover_df)} rows to {args.output}")

    print_summary(crossover_df)


if __name__ == "__main__":
    main()
