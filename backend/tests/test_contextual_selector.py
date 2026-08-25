"""Simulation demonstrating the cost-efficiency gain of ContextualCostAwareSelector.

Runs a head-to-head comparison between ContextualCostAwareSelector and
ThompsonSamplingSelector over many episodes, tracking both cumulative
success rate AND average per-episode recovery cost.

Expected outcome (the "headline result"):
  - Both selectors achieve similar late-stage success rates.
  - ContextualCostAwareSelector achieves it at a **lower average cost**
    because its reward function penalises expensive strategies.
"""

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.failures.taxonomy import FailureEvent, FailureType
from src.recovery.contextual_selector import ContextualCostAwareSelector
from src.recovery.selector import SelectorFactory
from src.recovery.strategies import (
    RecoveryStrategy,
    execute_strategy,
    get_action_cost,
)


# ======================================================================
# Simulation
# ======================================================================


def _build_context(
    episode: int,
    retry_count: int,
    tool_name: str = "web_search",
) -> dict:
    return {
        "retry_count": retry_count,
        "tool_name": tool_name,
        "time_since_last_failure": episode % 10,
    }


def run_single_trial(
    selector,
    failure_types: list[FailureType],
    base_seed: int,
    method_name: str = "",
) -> tuple[list[bool], list[float]]:
    """Run one trial and return (successes, costs) per episode."""
    outcome_rng = random.Random(base_seed + 7)
    successes: list[bool] = []
    costs: list[float] = []

    for ep, ft in enumerate(failure_types, start=1):
        retry_count = outcome_rng.randint(0, 3)
        context = _build_context(ep, retry_count)

        strategy = selector.select_strategy(ft, context=context)
        event = FailureEvent(failure_type=ft, raw_error="sim")
        ctx = {"tool_name": "web_search", "step_index": ep, "goal": "cost_test"}
        outcome = execute_strategy(strategy, event, ctx, rng=outcome_rng)

        actual_cost = outcome.get("cost", get_action_cost(strategy))
        successes.append(outcome["success"])
        costs.append(actual_cost)

        selector.update(
            ft, strategy, success=outcome["success"],
            cost=actual_cost, context=context,
        )

    return successes, costs


def run_comparison(
    n_episodes: int = 500,
    n_trials: int = 10,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run multiple trials comparing Thompson vs ContextualCostAware.

    Returns a DataFrame with columns:
        trial, episode, method, success, cost, cumulative_success_rate, average_cost
    """
    rows = []

    for trial_idx in range(n_trials):
        trial_seed = base_seed + trial_idx * 1000

        # Shared failure-type sequence for fair comparison
        type_rng = random.Random(trial_seed)
        fts = [type_rng.choice(list(FailureType)) for _ in range(n_episodes)]

        # --- Thompson Sampling ---
        ts_sel = SelectorFactory.create("thompson_sampling", seed=trial_seed)
        ts_successes, ts_costs = run_single_trial(
            ts_sel, fts, trial_seed, method_name="Thompson Sampling",
        )

        # --- Contextual Cost-Aware ---
        cc_sel = ContextualCostAwareSelector(
            lambda_cost=0.3, alpha=0.25, seed=trial_seed + 500,
        )
        cc_successes, cc_costs = run_single_trial(
            cc_sel, fts, trial_seed + 7, method_name="Contextual Cost-Aware",
        )

        for ep in range(n_episodes):
            for method_name, succ_list, cost_list in [
                ("Thompson Sampling", ts_successes, ts_costs),
                ("Contextual Cost-Aware", cc_successes, cc_costs),
            ]:
                successes_sofar = succ_list[:ep + 1]
                costs_sofar = cost_list[:ep + 1]
                cum_rate = sum(successes_sofar) / len(successes_sofar)
                avg_cost = sum(costs_sofar) / len(costs_sofar)

                rows.append({
                    "trial": trial_idx,
                    "episode": ep + 1,
                    "method": method_name,
                    "success": succ_list[ep],
                    "cost": cost_list[ep],
                    "cumulative_success_rate": cum_rate,
                    "average_cost": avg_cost,
                })

    return pd.DataFrame(rows)


# ======================================================================
# Plotting
# ======================================================================


def plot_results(df: pd.DataFrame, n_episodes: int, save_dir: Path):
    """Plot success rate and cost curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["Thompson Sampling", "Contextual Cost-Aware"]
    colors = {"Thompson Sampling": "#16a34a", "Contextual Cost-Aware": "#e11d48"}
    episodes = np.arange(1, n_episodes + 1)

    for method in methods:
        sub = df[df["method"] == method]
        pivot_rate = sub.pivot_table(index="episode", columns="trial", values="cumulative_success_rate")
        pivot_cost = sub.pivot_table(index="episode", columns="trial", values="average_cost")

        mean_rate = pivot_rate.mean(axis=1).values
        std_rate = pivot_rate.std(axis=1).values
        mean_cost = pivot_cost.mean(axis=1).values
        std_cost = pivot_cost.std(axis=1).values

        # Success rate
        ax1.plot(episodes, mean_rate, label=method, color=colors[method], linewidth=2)
        ax1.fill_between(episodes, mean_rate - std_rate, mean_rate + std_rate,
                         color=colors[method], alpha=0.12)

        # Average cost
        ax2.plot(episodes, mean_cost, label=method, color=colors[method], linewidth=2)
        ax2.fill_between(episodes, mean_cost - std_cost, mean_cost + std_cost,
                         color=colors[method], alpha=0.12)

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Cumulative Success Rate", fontsize=12)
    ax1.set_title("Success Rate", fontsize=13)
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Average Per-Episode Cost", fontsize=12)
    ax2.set_title("Average Recovery Cost", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / "cost_efficiency_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Cost efficiency plot saved to {save_dir / 'cost_efficiency_comparison.png'}")


# ======================================================================
# Main
# ======================================================================


def main():
    n_episodes = 500
    n_trials = 20

    print(f"\n  Running cost-efficiency comparison: {n_episodes} eps x {n_trials} trials ...")
    df = run_comparison(n_episodes=n_episodes, n_trials=n_trials)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    plot_results(df, n_episodes, data_dir)

    # --- Final summary ---
    print(f"\n  {'='*70}")
    print(f"  Final comparison at episode {n_episodes}")
    print(f"  {'='*70}")

    summary_rows = []
    for method in df["method"].unique():
        sub = df[(df["method"] == method) & (df["episode"] == n_episodes)]
        late_sub = df[(df["method"] == method) & (df["episode"] > int(n_episodes * 0.8))]

        mean_rate = sub["cumulative_success_rate"].mean()
        std_rate = sub["cumulative_success_rate"].std()
        mean_cost = sub["average_cost"].mean()
        std_cost = sub["average_cost"].std()

        # Late-stage (last 20%) success rate computed from per-episode success
        late_success_rates = late_sub.groupby("trial")["success"].mean()
        late_success_mean = late_success_rates.mean()
        late_success_std = late_success_rates.std()

        summary_rows.append({
            "Method": method,
            "Final Success Rate": f"{mean_rate:.2%} +/- {std_rate:.2%}",
            "Late-Stage Success Rate": f"{late_success_mean:.2%} +/- {late_success_std:.2%}",
            "Average Cost": f"{mean_cost:.3f} +/- {std_cost:.3f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()

    # --- Headline: cost difference at similar success rates ---
    thompson = df[(df["method"] == "Thompson Sampling") & (df["episode"] == n_episodes)]
    contextual = df[(df["method"] == "Contextual Cost-Aware") & (df["episode"] == n_episodes)]

    ts_rate = thompson["cumulative_success_rate"].mean()
    ts_cost = thompson["average_cost"].mean()
    cc_rate = contextual["cumulative_success_rate"].mean()
    cc_cost = contextual["average_cost"].mean()

    print(f"  {'='*70}")
    print(f"  HEADLINE RESULT")
    print(f"  {'='*70}")
    print(f"  Thompson Sampling:         success={ts_rate:.2%}, avg cost={ts_cost:.3f}")
    print(f"  Contextual Cost-Aware:     success={cc_rate:.2%}, avg cost={cc_cost:.3f}")
    cost_reduction = (1 - cc_cost / ts_cost) * 100 if ts_cost > 0 else 0.0
    print(f"  Cost reduction:            {cost_reduction:.1f}% "
          f"(Cost-Aware is {'cheaper' if cost_reduction > 0 else 'more expensive'})")
    print()

    # Save CSV
    csv_path = data_dir / "cost_efficiency_results.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Raw data saved to {csv_path}")
    print()


if __name__ == "__main__":
    main()
