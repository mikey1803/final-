#!/usr/bin/env python
"""Task 14 addendum: a publication-quality figure showing the trade-off
between per-episode computational cost (Part 1) and policy quality, with
the Pareto frontier highlighted.

This does NOT invent a new composite metric. It plots two metrics that
already exist in this repository, produced by two already-validated
pipelines, for the subset of policies both pipelines share:

- X-axis: ``kernel_us_per_episode`` from
  ``results/tables/runtime_benchmark_kernel_validation.parquet`` --
  Task 14 Part 1's real per-episode wall-clock time measured on the
  actual, unmodified ``recovery.core.episode.run_episodes`` kernel (not
  the isolated timing-loop estimate; the kernel-validated number is used
  here because it is the one directly measured on the same code path
  every other paper table is generated from).
- Y-axis: the ``normalized`` column of ``results/tables/table_v.parquet``
  (Table V, produced by ``experiments/stationary_comparison.py``), averaged
  over that table's 50 seeds per method -- this is the project's existing,
  already-validated normalized performance score for the canonical
  stationary environment (500 episodes/trial), not something computed here.

Matching policies across the two tables
----------------------------------------
The two scripts name policies differently (e.g. runtime_benchmark's
``"UCB1"`` vs. Table V's ``"Adaptive (UCB1)"``) but a policy is only
included in this figure if it is the *exact same class, constructed with
the exact same hyperparameters* in both scripts -- verified by reading
``experiments/stationary_comparison.py``'s ``_METHODS`` dict and
``recovery/policies/adapters.py``'s factory default values directly
(not assumed from naming similarity). See ``_METHOD_MAP`` below for the
verified 10-way mapping, and ``_EXCLUDED`` for the 4 runtime-benchmarked
policies with no same-protocol Table V counterpart and the documented
reason each is left out rather than guessed at.

Usage
-----
    python experiments/pareto_tradeoff_figure.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

TABLES_DIR = _BACKEND_DIR / "results" / "tables"
FIGURES_DIR = _BACKEND_DIR / "results" / "figures"

RUNTIME_PATH = TABLES_DIR / "runtime_benchmark_kernel_validation.parquet"
TABLE_V_PATH = TABLES_DIR / "table_v.parquet"

# runtime_benchmark.py policy name -> table_v.parquet method name.
# Verified identical class + identical (explicit-or-default) hyperparameters
# in both experiments/stationary_comparison.py and
# recovery/policies/adapters.py -- e.g. EpsilonGreedyPolicy's default
# epsilon=0.15 matches stationary_comparison's explicit epsilon=0.15;
# LinUCBPolicy's default alpha=1.0 matches its explicit alpha=1.0;
# ContextualCostAwarePolicy's defaults (lambda_cost=0.3, alpha=0.25) match
# its explicit values. Only the RNG seed differs between the two scripts,
# which affects realized outcomes (already captured by Table V's own
# 50-seed averaging) but not per-decision computational cost.
_METHOD_MAP: dict[str, str] = {
    "FixedRule (blind static, 6/7)": "Fixed Rule",
    "Oracle": "Oracle",
    "UniformRandom": "UniformRandom",
    "NaiveRetry": "Naive Retry",
    "UCB1": "Adaptive (UCB1)",
    "EpsilonGreedy": "Adaptive (e-greedy)",
    "DecayingEpsilonGreedy": "Adaptive (Decaying e)",
    "Thompson": "Adaptive (Thompson)",
    "LinUCB": "LinUCB",
    "ContextualCostAware": "Contextual Cost-Aware",
}

# Runtime-benchmarked policies with NO matching same-protocol Table V entry.
# Left out of the figure rather than merged by assumption or estimated.
_EXCLUDED: dict[str, str] = {
    "StaticRule (BLIND_RULE, native)": (
        "Decision-equivalent to 'Fixed Rule' (both apply the same BLIND_RULE "
        "table) but implemented as a different adapter class (StaticRulePolicy "
        "vs. FixedRulePolicy) with no independent Table V row of its own -- "
        "not merged with Fixed Rule's score without a direct empirical check."
    ),
    "DiscountedThompson": (
        "Not one of Table V's 11 methods; its normalized-performance analogue "
        "would come from the prior/discounting experiments (prior_grid.py), "
        "not the stationary Table V protocol."
    ),
    "PGTS (PriorInitializedThompson, strength=20)": (
        "Introduced for the Task 9 prior-sensitivity study, not benchmarked "
        "in Table V's stationary protocol. Its performance data lives in "
        "prior_grid.parquet, which at the time this figure was generated was "
        "still being regenerated at full scale (see docs/PROJECT_STATUS.md) "
        "-- no valid same-protocol number was available."
    ),
    "DiscountedPGTS": (
        "Same reason as PGTS above: a prior_grid.py-family policy, not part "
        "of Table V's stationary protocol."
    ),
}


def build_tradeoff_table() -> pd.DataFrame:
    runtime = pd.read_parquet(RUNTIME_PATH)[["policy", "kernel_us_per_episode"]]
    table_v = pd.read_parquet(TABLE_V_PATH)

    perf = (
        table_v.groupby("method")["normalized"]
        .agg(normalized_performance_mean="mean", normalized_performance_std="std")
        .reset_index()
    )

    rows = []
    for rb_name, tv_name in _METHOD_MAP.items():
        rt_row = runtime.loc[runtime["policy"] == rb_name]
        perf_row = perf.loc[perf["method"] == tv_name]
        if rt_row.empty or perf_row.empty:
            raise ValueError(f"Missing data for mapped policy: {rb_name!r} -> {tv_name!r}")
        rows.append({
            "policy": rb_name,
            "table_v_method": tv_name,
            "runtime_us_per_episode": rt_row["kernel_us_per_episode"].iloc[0],
            "normalized_performance_mean": perf_row["normalized_performance_mean"].iloc[0],
            "normalized_performance_std": perf_row["normalized_performance_std"].iloc[0],
        })
    df = pd.DataFrame(rows)

    # Pareto frontier: minimize runtime, maximize normalized performance.
    # A point is dominated if another point is at least as good on both
    # axes and strictly better on at least one.
    is_efficient = []
    for i, row in df.iterrows():
        dominated = (
            (
                (df["runtime_us_per_episode"] <= row["runtime_us_per_episode"])
                & (df["normalized_performance_mean"] >= row["normalized_performance_mean"])
                & (
                    (df["runtime_us_per_episode"] < row["runtime_us_per_episode"])
                    | (df["normalized_performance_mean"] > row["normalized_performance_mean"])
                )
            )
            .drop(index=i)
            .any()
        )
        is_efficient.append(not dominated)
    df["is_pareto_efficient"] = is_efficient
    return df.sort_values("runtime_us_per_episode").reset_index(drop=True)


def plot_tradeoff(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(7, 5))

    frontier = df.loc[df["is_pareto_efficient"]].sort_values("runtime_us_per_episode")
    dominated = df.loc[~df["is_pareto_efficient"]]

    ax.errorbar(
        dominated["runtime_us_per_episode"], dominated["normalized_performance_mean"],
        yerr=dominated["normalized_performance_std"], fmt="o", color="#9a9a9a",
        ecolor="#c8c8c8", elinewidth=1, capsize=2, markersize=6, zorder=2,
        label="Dominated",
    )
    ax.errorbar(
        frontier["runtime_us_per_episode"], frontier["normalized_performance_mean"],
        yerr=frontier["normalized_performance_std"], fmt="D", color="#1f6f43",
        ecolor="#9fcbb0", elinewidth=1.2, capsize=2, markersize=7, zorder=3,
        label="Pareto-efficient",
    )
    ax.plot(
        frontier["runtime_us_per_episode"], frontier["normalized_performance_mean"],
        color="#1f6f43", linestyle="--", linewidth=1.2, zorder=2.5,
    )

    # Stagger label vertical offset for points close together in log-x so
    # nearby labels (e.g. Oracle/FixedRule) don't overlap each other or the
    # other point's marker.
    import numpy as np

    sorted_df = df.sort_values("runtime_us_per_episode").reset_index(drop=True)
    log_x = np.log10(sorted_df["runtime_us_per_episode"])
    offsets = [(6, 4)]
    for i in range(1, len(sorted_df)):
        if log_x[i] - log_x[i - 1] < 0.08:
            prev_dy = offsets[-1][1]
            offsets.append((6, -16 if prev_dy > 0 else 4))
        else:
            offsets.append((6, 4))

    for (_, row), (dx, dy) in zip(sorted_df.iterrows(), offsets):
        ax.annotate(
            row["policy"].split(" (")[0],
            (row["runtime_us_per_episode"], row["normalized_performance_mean"]),
            textcoords="offset points", xytext=(dx, dy), fontsize=7.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Runtime per episode (microseconds, log scale) — kernel-measured")
    ax.set_ylabel("Normalized performance score (Table V, canonical stationary world)")
    ax.set_title("Policy quality vs. computational overhead (stationary environment)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "runtime_quality_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "runtime_quality_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_tradeoff_table()

    out_path = TABLES_DIR / "runtime_quality_tradeoff.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

    plot_tradeoff(df)
    print(f"Wrote {FIGURES_DIR / 'runtime_quality_tradeoff.png'}")
    print(f"Wrote {FIGURES_DIR / 'runtime_quality_tradeoff.pdf'}")

    print("\n=== Trade-off table ===")
    print(df.to_string(index=False))

    frontier = df.loc[df["is_pareto_efficient"], "policy"].tolist()
    dominated = df.loc[~df["is_pareto_efficient"], "policy"].tolist()
    print(f"\nPareto-efficient ({len(frontier)}): {frontier}")
    print(f"Dominated ({len(dominated)}): {dominated}")

    print(f"\nExcluded from this figure ({len(_EXCLUDED)} policies benchmarked in Part 1 "
          "but with no same-protocol Table V performance number):")
    for name, reason in _EXCLUDED.items():
        print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
