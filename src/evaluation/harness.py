"""Evaluation harness: compare all five recovery-selection methods.

``run_comparison`` runs many independent trials of the same episode
sequence across every method, tracks cumulative success rate over time,
and produces:
  - A summary DataFrame of cumulative and late-stage success rates.
  - A line-chart plot with shaded std bands (``data/method_comparison.png``).
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
from src.recovery.strategies import execute_strategy
from src.recovery.selector import AdaptiveSelector

# Colour / style map for each method (order matters)
_METHOD_STYLES: dict[str, dict] = {
    "Adaptive (UCB1)":      {"color": "#2563eb", "linestyle": "-",  "marker": "o"},
    "Naive Retry":          {"color": "#9ca3af", "linestyle": ":",  "marker": "x"},
    "Fixed Rule":           {"color": "#f59e0b", "linestyle": "--", "marker": "s"},
    "Random":               {"color": "#a855f7", "linestyle": "-.", "marker": "D"},
    "One-Shot LLM":         {"color": "#ef4444", "linestyle": "--", "marker": "^"},
}


def _make_selectors(trial_seed: int) -> dict[str, BaseSelector]:
    """Create a fresh set of all five selectors for one trial."""
    return {
        "Adaptive (UCB1)": AdaptiveSelector(algorithm="ucb1", seed=trial_seed),
        "Naive Retry":     NaiveRetryBaseline(),
        "Fixed Rule":      FixedRuleBaseline(),
        "Random":          RandomBaseline(seed=trial_seed),
        "One-Shot LLM":    OneShotLLMBaseline(seed=trial_seed),
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
    """Run a full multi-trial comparison of all five methods.

    Returns:
        (summary_df, raw_df) where ``summary_df`` has rows = method
        names and columns for cumulative and late-stage success rates
        plus their standard deviations, and ``raw_df`` has every
        per-episode cumulative success rate for every method and trial.
    """
    method_names = list(_METHOD_STYLES.keys())
    late_stage_start_idx = int(num_episodes * 0.8)
    late_stage_start_idx = min(max(late_stage_start_idx, 0), num_episodes - 1)
    all_cumulative_rates: dict[str, list[float]] = {m: [] for m in method_names}
    all_late_stage_rates: dict[str, list[float]] = {m: [] for m in method_names}
    all_total_successes: dict[str, list[int]] = {m: [] for m in method_names}
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
        episode_successes: dict[str, list[bool]] = {m: [] for m in method_names}

        for ep in range(num_episodes):
            for mname, sel in selectors.items():
                success = _simulate_episode(sel, fts[ep], outcome_rng, ep + 1)
                episode_successes[mname].append(success)
                prev = curves[mname][ep - 1] if ep > 0 else 0.0
                # Running average: previous cumulative successes / (ep + 1)
                prev_count = prev * ep if ep > 0 else 0.0
                new_count = prev_count + (1.0 if success else 0.0)
                curves[mname][ep] = new_count / (ep + 1)

        for mname in method_names:
            c = curves[mname]
            late_successes = episode_successes[mname][late_stage_start_idx:]
            all_cumulative_rates[mname].append(float(c[-1]))
            all_late_stage_rates[mname].append(float(np.mean(late_successes)))
            total_succ = int(round(c[-1] * num_episodes))
            all_total_successes[mname].append(total_succ)
            for ep_idx in range(num_episodes):
                raw_rows.append({
                    "trial": trial_idx,
                    "episode": ep_idx + 1,
                    "method": mname,
                    "cumulative_success_rate": float(c[ep_idx]),
                })

    # --- summary ---
    summary_rows = []
    for mname in method_names:
        cumulative_rates = all_cumulative_rates[mname]
        late_stage_rates = all_late_stage_rates[mname]
        successes = all_total_successes[mname]
        summary_rows.append({
            "method": mname,
            "mean_cumulative_success_rate": float(np.mean(cumulative_rates)),
            "std_cumulative_success_rate": float(np.std(cumulative_rates)),
            "mean_late_stage_success_rate": float(np.mean(late_stage_rates)),
            "std_late_stage_success_rate": float(np.std(late_stage_rates)),
            "mean_total_successes": float(np.mean(successes)),
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
        # Pivot: rows=episode, cols=trials
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
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / "method_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Comparison plot saved to {save_dir / 'method_comparison.png'}")


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

    # Plot
    plot_comparison(raw_df, num_episodes, data_dir, show_late_stage_marker=show_late_stage_marker)

    # Print summary
    print(f"\n  {'='*72}")
    print(f"  Summary (averaged over {num_trials} trials x {num_episodes} episodes)")
    print(f"  {'='*72}")
    print(summary_df.to_string(float_format="{:.4f}".format))
    print()

    significance_df = summary_df.attrs.get("late_stage_significance")
    if significance_df is not None:
        print(f"  {'='*72}")
        print("  Late-stage significance vs Adaptive (UCB1)")
        print(f"  {'='*72}")
        print(significance_df.to_string(float_format="{:.6f}".format))
        print()

    return summary_df
