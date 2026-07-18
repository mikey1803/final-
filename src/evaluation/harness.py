"""Evaluation harness: compare all recovery-selection methods.

``run_comparison`` runs many independent trials of the same episode
sequence across every method, tracks cumulative success rate and
cumulative regret over time, and produces:
  - A summary DataFrame of cumulative / late-stage success rates and
    cumulative regret.
  - A line-chart plot with shaded std bands (``data/method_comparison.png``).
  - A cumulative-regret plot (``data/regret_curve.png``).
  - A full raw-results CSV (``data/evaluation_results.csv``).
"""

import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.baselines import (
    BaseSelector,
    FixedRuleBaseline,
    NaiveRetryBaseline,
    OneShotLLMBaseline,
    RandomBaseline,
)
from src.failures.taxonomy import FailureEvent, FailureType
from src.recovery.strategies import SUCCESS_MATRIX, RecoveryStrategy, execute_strategy
from src.recovery.selector import AdaptiveSelector

# Colour / style map for each method
_METHOD_STYLES: dict[str, dict] = {
    "Adaptive (UCB1)":          {"color": "#2563eb", "linestyle": "-",  "marker": "o"},
    "Adaptive (Thompson)":      {"color": "#16a34a", "linestyle": "-",  "marker": "P"},
    "Naive Retry":              {"color": "#9ca3af", "linestyle": ":",  "marker": "x"},
    "Fixed Rule":               {"color": "#f59e0b", "linestyle": "--", "marker": "s"},
    "Random":                   {"color": "#a855f7", "linestyle": "-.", "marker": "D"},
    "One-Shot LLM":             {"color": "#ef4444", "linestyle": "--", "marker": "^"},
}

# Ground-truth best success probability per failure type
_OPTIMAL_RATE: dict[FailureType, float] = {
    ft: max(rates.values()) for ft, rates in SUCCESS_MATRIX.items()
}


def _make_selectors(trial_seed: int) -> dict[str, BaseSelector]:
    """Create a fresh set of all selectors for one trial."""
    return {
        "Adaptive (UCB1)":      AdaptiveSelector(algorithm="ucb1", seed=trial_seed),
        "Adaptive (Thompson)":  AdaptiveSelector(algorithm="thompson_sampling", seed=trial_seed + 500),
        "Naive Retry":          NaiveRetryBaseline(),
        "Fixed Rule":           FixedRuleBaseline(),
        "Random":               RandomBaseline(seed=trial_seed),
        "One-Shot LLM":         OneShotLLMBaseline(seed=trial_seed),
    }


def _simulate_episode(
    selector: BaseSelector,
    failure_type: FailureType,
    rng: random.Random,
    episode_number: int,
) -> bool:
    """Run one recovery episode: select strategy, simulate, update, return success."""
    strategy = selector.select_strategy(failure_type)
    event = FailureEvent(failure_type=failure_type, raw_error="simulated")
    ctx = {"tool_name": "web_search", "step_index": episode_number, "goal": "eval"}
    outcome = execute_strategy(strategy, event, ctx, rng=rng)
    if isinstance(selector, AdaptiveSelector):
        selector.update(failure_type, strategy, success=outcome["success"], episode_number=episode_number)
    return outcome["success"]


def _build_significance_table(
    late_stage_rates: dict[str, list[float]],
    adaptive_method: str = "Adaptive (UCB1)",
) -> pd.DataFrame:
    """Compare adaptive late-stage rates against each baseline with paired t-tests."""
    adaptive_rates = late_stage_rates[adaptive_method]
    rows = []

    for baseline, baseline_rates in late_stage_rates.items():
        if baseline == adaptive_method:
            continue

        t_statistic, p_value = ttest_rel(adaptive_rates, baseline_rates)
        mean_difference = float(np.mean(np.array(adaptive_rates) - np.array(baseline_rates)))
        rows.append({
            "baseline": baseline,
            "mean_difference": mean_difference,
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "significant": "YES (p < 0.05)" if p_value < 0.05 else "no",
        })

    return pd.DataFrame(rows).set_index("baseline")


def run_comparison(
    num_episodes: int = 500,
    num_trials: int = 30,
    base_seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a full multi-trial comparison of all methods.

    Returns:
        (summary_df, raw_df) where ``summary_df`` has rows = method
        names and columns for cumulative / late-stage success rates,
        cumulative regret, and their standard deviations.  ``raw_df``
        has per-episode cumulative success rate for every method and
        trial.
    """
    method_names = list(_METHOD_STYLES.keys())
    late_stage_start_idx = min(max(int(num_episodes * 0.8), 0), num_episodes - 1)

    all_cumulative_rates: dict[str, list[float]] = {m: [] for m in method_names}
    all_late_stage_rates: dict[str, list[float]] = {m: [] for m in method_names}
    all_total_successes: dict[str, list[int]] = {m: [] for m in method_names}
    all_final_regret: dict[str, list[float]] = {m: [] for m in method_names}
    raw_rows: list[dict] = []

    for trial_idx in range(num_trials):
        trial_seed = base_seed + trial_idx * 1000

        # Shared failure-type sequence for this trial
        type_rng = random.Random(trial_seed)
        fts = [type_rng.choice(list(FailureType)) for _ in range(num_episodes)]

        # Outcome RNG (independent of type selection)
        outcome_rng = random.Random(trial_seed + 7)

        selectors = _make_selectors(trial_seed)
        curves: dict[str, np.ndarray] = {m: np.zeros(num_episodes) for m in method_names}
        # Cumulative regret: sum of per-episode instant regret
        cum_regret: dict[str, np.ndarray] = {m: np.zeros(num_episodes) for m in method_names}
        episode_successes: dict[str, list[bool]] = {m: [] for m in method_names}

        for ep in range(num_episodes):
            ft = fts[ep]
            optimal_r = _OPTIMAL_RATE[ft]

            for mname, sel in selectors.items():
                success = _simulate_episode(sel, ft, outcome_rng, ep + 1)
                episode_successes[mname].append(success)

                # Cumulative success rate
                prev = curves[mname][ep - 1] if ep > 0 else 0.0
                prev_count = prev * ep if ep > 0 else 0.0
                new_count = prev_count + (1.0 if success else 0.0)
                curves[mname][ep] = new_count / (ep + 1)

                # Instant regret = optimal_rate - actual_reward
                instant_regret = optimal_r - (1.0 if success else 0.0)
                prev_regret = cum_regret[mname][ep - 1] if ep > 0 else 0.0
                cum_regret[mname][ep] = prev_regret + instant_regret

        for mname in method_names:
            c = curves[mname]
            cr = cum_regret[mname]
            late_successes = episode_successes[mname][late_stage_start_idx:]
            all_cumulative_rates[mname].append(float(c[-1]))
            all_late_stage_rates[mname].append(float(np.mean(late_successes)))
            all_total_successes[mname].append(int(round(c[-1] * num_episodes)))
            all_final_regret[mname].append(float(cr[-1]))

            for ep_idx in range(num_episodes):
                raw_rows.append({
                    "trial": trial_idx,
                    "episode": ep_idx + 1,
                    "method": mname,
                    "cumulative_success_rate": float(c[ep_idx]),
                    "cumulative_regret": float(cr[ep_idx]),
                })

    # --- summary ---
    summary_rows = []
    for mname in method_names:
        summary_rows.append({
            "method": mname,
            "mean_cumulative_success_rate": float(np.mean(all_cumulative_rates[mname])),
            "std_cumulative_success_rate": float(np.std(all_cumulative_rates[mname])),
            "mean_late_stage_success_rate": float(np.mean(all_late_stage_rates[mname])),
            "std_late_stage_success_rate": float(np.std(all_late_stage_rates[mname])),
            "mean_total_successes": float(np.mean(all_total_successes[mname])),
            "mean_final_cumulative_regret": float(np.mean(all_final_regret[mname])),
            "std_final_cumulative_regret": float(np.std(all_final_regret[mname])),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("method")
    summary_df.attrs["late_stage_significance"] = _build_significance_table(all_late_stage_rates)

    raw_df = pd.DataFrame(raw_rows)
    return summary_df, raw_df


def plot_comparison(
    raw_df: pd.DataFrame,
    num_episodes: int,
    save_dir: Path,
    show_late_stage_marker: bool = True,
) -> None:
    """Plot mean cumulative success rate with shaded std band per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)
    late_stage_start_episode = int(num_episodes * 0.8) + 1

    for mname, style in _METHOD_STYLES.items():
        sub = raw_df[raw_df["method"] == mname]
        pivot = sub.pivot_table(index="episode", columns="trial", values="cumulative_success_rate")
        mean_curve = pivot.mean(axis=1).values
        std_curve = pivot.std(axis=1).values

        ax.plot(episodes, mean_curve, label=mname, color=style["color"],
                linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                        color=style["color"], alpha=0.12)

    if show_late_stage_marker:
        ax.axvline(late_stage_start_episode, color="#111827", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(late_stage_start_episode, 0.03, "Late stage begins",
                rotation=90, va="bottom", ha="right", fontsize=9, color="#111827")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Success Rate", fontsize=12)
    ax.set_title("Recovery-Strategy Selection - Method Comparison", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / "method_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Comparison plot saved to {save_dir / 'method_comparison.png'}")


def plot_regret(
    raw_df: pd.DataFrame,
    num_episodes: int,
    save_dir: Path,
) -> None:
    """Plot mean cumulative regret with shaded std band per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)

    for mname, style in _METHOD_STYLES.items():
        sub = raw_df[raw_df["method"] == mname]
        pivot = sub.pivot_table(index="episode", columns="trial", values="cumulative_regret")
        mean_curve = pivot.mean(axis=1).values
        std_curve = pivot.std(axis=1).values

        ax.plot(episodes, mean_curve, label=mname, color=style["color"],
                linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                        color=style["color"], alpha=0.10)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("Cumulative Regret - Method Comparison", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / "regret_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Regret plot saved to {save_dir / 'regret_curve.png'}")


def run_and_report(
    num_episodes: int = 500,
    num_trials: int = 10,
    data_dir: Optional[Path] = None,
    show_late_stage_marker: bool = True,
) -> pd.DataFrame:
    """Run the full comparison, plot, save CSV, and print the summary table.

    Returns the summary DataFrame for further use.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running comparison: {num_episodes} episodes x {num_trials} trials ...")
    summary_df, raw_df = run_comparison(num_episodes=num_episodes, num_trials=num_trials)

    # Save raw CSV
    csv_path = data_dir / "evaluation_results.csv"
    raw_df.to_csv(str(csv_path), index=False)
    print(f"  Raw results saved to {csv_path}")

    # Plots
    plot_comparison(raw_df, num_episodes, data_dir, show_late_stage_marker=show_late_stage_marker)
    plot_regret(raw_df, num_episodes, data_dir)

    # Print summary
    print(f"\n  {'='*80}")
    print(f"  Summary (averaged over {num_trials} trials x {num_episodes} episodes)")
    print(f"  {'='*80}")
    print(summary_df.to_string(float_format="{:.4f}".format))
    print()

    significance_df = summary_df.attrs.get("late_stage_significance")
    if significance_df is not None:
        print(f"  {'='*80}")
        print("  Late-stage significance vs Adaptive (UCB1)")
        print(f"  {'='*80}")
        print(significance_df.to_string(float_format="{:.6f}".format))
        print()

    return summary_df
