"""Evaluation harness: compare all recovery-selection methods.

``run_comparison`` runs many independent trials of the same episode
sequence across every method, tracks cumulative success rate, cumulative
regret, and **average cost per episode** over time, and produces:
  - A summary DataFrame of cumulative / late-stage success rates
    **with 95 % confidence intervals**, cumulative regret, and average cost.
  - Cohen's d effect sizes between Thompson Sampling and each other method.
  - A line-chart plot with shaded std bands (``data/method_comparison.png``).
  - A cumulative-regret plot (``data/regret_curve.png``) with convergence
    slope annotation for Thompson Sampling.
  - A average-cost plot (``data/cost_curve.png``).
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
    LinUCBBaseline,
    NaiveRetryBaseline,
    OneShotLLMBaseline,
    OraclePolicy,
    RandomBaseline,
)
from src.failures.taxonomy import FailureEvent, FailureType
from src.recovery.strategies import SUCCESS_MATRIX, RecoveryStrategy, execute_strategy, get_action_cost
from src.recovery.selector import BanditSelector, SelectorFactory
from src.recovery.contextual_selector import ContextualCostAwareSelector
from src.evaluation.classifier_context import (
    compute_empirical_confusion_matrix,
    effective_accuracy,
    interpolated_confusion_matrix,
    load_or_train_classifier,
    sample_synthetic_context,
)
from src.evaluation.stats_utils import (
    MC_VARIATION_NOTE,
    bootstrap_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    holm_bonferroni,
)

# Colour / style map for each method
_METHOD_STYLES: dict[str, dict] = {
    "Adaptive (Thompson)":               {"color": "#16a34a", "linestyle": "-",  "marker": "P"},
    "Adaptive (UCB1)":                   {"color": "#2563eb", "linestyle": "-",  "marker": "o"},
    "Adaptive (e-greedy)":               {"color": "#0891b2", "linestyle": "-",  "marker": "v"},
    "Adaptive (Decaying e)":             {"color": "#7c3aed", "linestyle": "-",  "marker": "h"},
    "Contextual Cost-Aware":             {"color": "#e11d48", "linestyle": "-",  "marker": "D"},
    "Naive Retry":                       {"color": "#9ca3af", "linestyle": ":",  "marker": "x"},
    "Fixed Rule":                        {"color": "#f59e0b", "linestyle": "--", "marker": "s"},
    "UniformRandom":                     {"color": "#a855f7", "linestyle": "-.", "marker": "D"},
    "One-Shot LLM":                      {"color": "#ef4444", "linestyle": "--", "marker": "^"},
    "LinUCB":                            {"color": "#0d9488", "linestyle": "-",  "marker": "^"},
    "Oracle":                            {"color": "#111827", "linestyle": ":",  "marker": "*"},
}

# Ground-truth best success probability per failure type
_OPTIMAL_RATE: dict[FailureType, float] = {
    ft: max(rates.values()) for ft, rates in SUCCESS_MATRIX.items()
}


def _make_selectors(trial_seed: int) -> dict[str, BaseSelector]:
    """Create a fresh set of all selectors for one trial."""
    return {
        "Adaptive (UCB1)":       SelectorFactory.create("ucb1", seed=trial_seed),
        "Adaptive (Thompson)":   SelectorFactory.create("thompson_sampling", seed=trial_seed + 500),
        "Adaptive (e-greedy)":   SelectorFactory.create("epsilon_greedy", epsilon=0.15, seed=trial_seed + 600),
        "Adaptive (Decaying e)": SelectorFactory.create("decaying_epsilon_greedy", seed=trial_seed + 700),
        "Contextual Cost-Aware": ContextualCostAwareSelector(
            lambda_cost=0.3, alpha=0.25, seed=trial_seed + 1000,
        ),
        "Naive Retry":           NaiveRetryBaseline(),
        "Fixed Rule":            FixedRuleBaseline(),
        "UniformRandom":         RandomBaseline(seed=trial_seed),
        "One-Shot LLM":          OneShotLLMBaseline(seed=trial_seed),
        "LinUCB":                LinUCBBaseline(alpha=1.0, seed=trial_seed + 1500),
        "Oracle":                OraclePolicy(),
    }


def _build_context(
    method_name: str,
    selector,
    failure_type: FailureType,
    episode_number: int,
    retry_count: int,
    tool_name: str,
) -> Optional[dict]:
    """Build a context dict for contextual selectors; returns None for plain bandits."""
    if not isinstance(selector, (ContextualCostAwareSelector, LinUCBBaseline)):
        return None
    # Simulate time_since_last_failure as a function of episode_number
    time_since = min(episode_number % 10, 100)
    return {
        "retry_count": retry_count,
        "tool_name": tool_name,
        "time_since_last_failure": time_since,
        "episode_number": episode_number,
    }


def _simulate_episode(
    selector: BaseSelector,
    failure_type: FailureType,
    rng: random.Random,
    episode_number: int,
    method_name: str = "",
    retry_count: int = 0,
    observed_failure_type: Optional[FailureType] = None,
) -> tuple[bool, float]:
    """Run one recovery episode: select strategy, simulate, update, return (success, cost).

    ``observed_failure_type``, if given, is the type the *selector*
    conditions its select_strategy/update calls on — e.g. a classifier
    prediction or synthetic-noise draw (see ``classifier_context.py``).
    The actual outcome is always sampled from ``SUCCESS_MATRIX[failure_type]``
    (ground truth), so a mismatch between the two models a selector acting
    on imperfect information about which failure actually occurred. When
    omitted, this is equivalent to context_mode="oracle" (perfect information).
    """
    obs_type = failure_type if observed_failure_type is None else observed_failure_type
    ctx = {"tool_name": "web_search", "step_index": episode_number, "goal": "eval"}
    context = _build_context(method_name, selector, obs_type, episode_number, retry_count, "web_search")

    strategy = selector.select_strategy(obs_type, context=context)
    event = FailureEvent(failure_type=failure_type, raw_error="simulated")
    outcome = execute_strategy(strategy, event, ctx, rng=rng)

    actual_cost = outcome.get("cost", get_action_cost(strategy))
    if isinstance(selector, (BanditSelector, LinUCBBaseline)):
        selector.update(obs_type, strategy, success=outcome["success"], cost=actual_cost, context=context, episode_number=episode_number)
    return outcome["success"], actual_cost


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

    df = pd.DataFrame(rows).set_index("baseline")
    # Holm-Bonferroni correction across this family of pairwise comparisons
    df["p_value_holm"] = holm_bonferroni(df["p_value"].tolist())
    df["significant_holm"] = np.where(df["p_value_holm"] < 0.05, "YES (p < 0.05)", "no")
    return df


def _build_effect_size_table(
    late_stage_rates: dict[str, list[float]],
    reference_method: str = "Adaptive (Thompson)",
) -> pd.DataFrame:
    """Compute Cohen's d effect size between *reference_method* and every other method.

    Also includes paired t-test p-values for convenience.

    Cohen's d thresholds: |d| < 0.2 → negligible, 0.2–0.5 → small,
    0.5–0.8 → medium, ≥ 0.8 → large.
    """
    ref_rates = np.array(late_stage_rates[reference_method])
    rows = []

    for method, method_rates_list in late_stage_rates.items():
        if method == reference_method:
            continue

        method_rates = np.array(method_rates_list)

        # Pooled standard deviation (Cohen's d)
        n1, n2 = len(ref_rates), len(method_rates)
        var1, var2 = np.var(ref_rates, ddof=1), np.var(method_rates, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        mean_diff = float(np.mean(ref_rates) - np.mean(method_rates))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Magnitude label
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        t_statistic, p_value = ttest_rel(ref_rates, method_rates)
        delta = cliffs_delta(ref_rates, method_rates)

        rows.append({
            "method": method,
            "mean_difference": mean_diff,
            "cohens_d": float(cohens_d),
            "effect_magnitude": magnitude,
            "cliffs_delta": delta,
            "cliffs_delta_magnitude": cliffs_delta_magnitude(delta),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "significant": "YES (p < 0.05)" if p_value < 0.05 else "no",
        })

    df = pd.DataFrame(rows).set_index("method")
    df["p_value_holm"] = holm_bonferroni(df["p_value"].tolist())
    df["significant_holm"] = np.where(df["p_value_holm"] < 0.05, "YES (p < 0.05)", "no")
    return df


def _build_cost_effect_size_table(
    avg_costs: dict[str, list[float]],
    reference_method: str = "Contextual Cost-Aware",
) -> pd.DataFrame:
    """Compute Cohen's d on average recovery cost: *reference_method* vs every other.

    A **positive** d means the reference method is *cheaper* (lower cost).
    This uses the same pooled-SD formula as the success-rate effect sizes.
    """
    ref_costs = np.array(avg_costs[reference_method])
    rows = []

    for method, method_costs_list in avg_costs.items():
        if method == reference_method:
            continue

        method_costs = np.array(method_costs_list)

        # Pooled standard deviation
        n1, n2 = len(ref_costs), len(method_costs)
        var1, var2 = np.var(ref_costs, ddof=1), np.var(method_costs, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Positive mean_diff = other method is more expensive than reference
        mean_diff = float(np.mean(method_costs) - np.mean(ref_costs))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        t_statistic, p_value = ttest_rel(method_costs, ref_costs)
        delta = cliffs_delta(method_costs, ref_costs)

        rows.append({
            "method": method,
            "ref_mean_cost": float(np.mean(ref_costs)),
            "method_mean_cost": float(np.mean(method_costs)),
            "cost_difference": mean_diff,
            "cohens_d": float(cohens_d),
            "effect_magnitude": magnitude,
            "cliffs_delta": delta,
            "cliffs_delta_magnitude": cliffs_delta_magnitude(delta),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "significant": "YES (p < 0.05)" if p_value < 0.05 else "no",
        })

    df = pd.DataFrame(rows).set_index("method")
    df["p_value_holm"] = holm_bonferroni(df["p_value"].tolist())
    df["significant_holm"] = np.where(df["p_value_holm"] < 0.05, "YES (p < 0.05)", "no")
    return df


def _add_normalized_scores(
    summary_df: pd.DataFrame,
    random_key: str = "UniformRandom",
    oracle_key: str = "Oracle",
) -> pd.DataFrame:
    """Add oracle/random-normalized success-rate columns to *summary_df*.

    normalized = (method_rate - random_rate) / (oracle_rate - random_rate)

    0.0 == UniformRandom floor, 1.0 == Oracle ceiling. Mutates and returns
    *summary_df* with two new columns; leaves every existing column intact.
    """
    for raw_col, norm_col in [
        ("mean_cumulative_success_rate", "normalized_cumulative_score"),
        ("mean_late_stage_success_rate", "normalized_late_stage_score"),
    ]:
        random_rate = summary_df.loc[random_key, raw_col]
        oracle_rate = summary_df.loc[oracle_key, raw_col]
        denom = oracle_rate - random_rate
        summary_df[norm_col] = (
            (summary_df[raw_col] - random_rate) / denom if denom != 0 else float("nan")
        )
    return summary_df


def run_comparison(
    num_episodes: int = 500,
    num_trials: int = 50,
    base_seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a full multi-trial comparison of all methods.

    Returns:
        (summary_df, raw_df) where ``summary_df`` has rows = method
        names and columns for cumulative / late-stage success rates,
        cumulative regret, average cost, and their standard deviations.
        ``raw_df`` has per-episode cumulative success rate, cumulative
        regret, and average cost for every method and trial.
    """
    method_names = list(_METHOD_STYLES.keys())
    late_stage_start_idx = min(max(int(num_episodes * 0.8), 0), num_episodes - 1)

    all_cumulative_rates: dict[str, list[float]] = {m: [] for m in method_names}
    all_late_stage_rates: dict[str, list[float]] = {m: [] for m in method_names}
    all_total_successes: dict[str, list[int]] = {m: [] for m in method_names}
    all_final_regret: dict[str, list[float]] = {m: [] for m in method_names}
    all_avg_cost: dict[str, list[float]] = {m: [] for m in method_names}
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
        cum_regret: dict[str, np.ndarray] = {m: np.zeros(num_episodes) for m in method_names}
        cost_curves: dict[str, np.ndarray] = {m: np.zeros(num_episodes) for m in method_names}
        episode_successes: dict[str, list[bool]] = {m: [] for m in method_names}

        for ep in range(num_episodes):
            ft = fts[ep]
            optimal_r = _OPTIMAL_RATE[ft]

            for mname, sel in selectors.items():
                retry_count = outcome_rng.randint(0, 3)
                success, cost = _simulate_episode(
                    sel, ft, outcome_rng, ep + 1,
                    method_name=mname, retry_count=retry_count,
                )
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

                # Running average cost
                if ep == 0:
                    cost_curves[mname][ep] = cost
                else:
                    cost_curves[mname][ep] = (
                        cost_curves[mname][ep - 1] * ep + cost
                    ) / (ep + 1)

        for mname in method_names:
            c = curves[mname]
            cr = cum_regret[mname]
            cc = cost_curves[mname]
            late_successes = episode_successes[mname][late_stage_start_idx:]
            all_cumulative_rates[mname].append(float(c[-1]))
            all_late_stage_rates[mname].append(float(np.mean(late_successes)))
            all_total_successes[mname].append(int(round(c[-1] * num_episodes)))
            all_final_regret[mname].append(float(cr[-1]))
            all_avg_cost[mname].append(float(cc[-1]))

            for ep_idx in range(num_episodes):
                raw_rows.append({
                    "trial": trial_idx,
                    "episode": ep_idx + 1,
                    "method": mname,
                    "cumulative_success_rate": float(c[ep_idx]),
                    "cumulative_regret": float(cr[ep_idx]),
                    "average_cost": float(cc[ep_idx]),
                })

    # --- summary ---
    summary_rows = []
    for mname in method_names:
        cum_arr = np.array(all_cumulative_rates[mname])
        late_arr = np.array(all_late_stage_rates[mname])
        cost_arr = np.array(all_avg_cost[mname])
        n = len(cum_arr)

        cum_sem = float(np.std(cum_arr, ddof=1) / np.sqrt(n))
        late_sem = float(np.std(late_arr, ddof=1) / np.sqrt(n))
        cost_sem = float(np.std(cost_arr, ddof=1) / np.sqrt(n))

        # Bootstrap 95% CIs (10,000 resamples) — see MC_VARIATION_NOTE.
        boot_late_lo, boot_late_hi = bootstrap_ci(late_arr)
        boot_regret_lo, boot_regret_hi = bootstrap_ci(all_final_regret[mname])

        summary_rows.append({
            "method": mname,
            "mean_cumulative_success_rate": float(np.mean(cum_arr)),
            "std_cumulative_success_rate": float(np.std(cum_arr)),
            "ci95_lower_cumulative": float(np.mean(cum_arr) - 1.96 * cum_sem),
            "ci95_upper_cumulative": float(np.mean(cum_arr) + 1.96 * cum_sem),
            "mean_late_stage_success_rate": float(np.mean(late_arr)),
            "std_late_stage_success_rate": float(np.std(late_arr)),
            "ci95_lower_late_stage": float(np.mean(late_arr) - 1.96 * late_sem),
            "ci95_upper_late_stage": float(np.mean(late_arr) + 1.96 * late_sem),
            "bootstrap_ci95_lower_late_stage": boot_late_lo,
            "bootstrap_ci95_upper_late_stage": boot_late_hi,
            "mean_total_successes": float(np.mean(all_total_successes[mname])),
            "mean_final_cumulative_regret": float(np.mean(all_final_regret[mname])),
            "std_final_cumulative_regret": float(np.std(all_final_regret[mname])),
            "bootstrap_ci95_lower_regret": boot_regret_lo,
            "bootstrap_ci95_upper_regret": boot_regret_hi,
            "mean_average_cost": float(np.mean(cost_arr)),
            "std_average_cost": float(np.std(cost_arr)),
            "ci95_lower_cost": float(np.mean(cost_arr) - 1.96 * cost_sem),
            "ci95_upper_cost": float(np.mean(cost_arr) + 1.96 * cost_sem),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("method")
    summary_df = _add_normalized_scores(summary_df)
    summary_df.attrs["late_stage_significance"] = _build_significance_table(all_late_stage_rates)
    summary_df.attrs["thompson_effect_sizes"] = _build_effect_size_table(all_late_stage_rates)
    summary_df.attrs["cost_effect_sizes"] = _build_cost_effect_size_table(all_avg_cost)

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
    """Plot mean cumulative regret with shaded std band per method.

    Annotates the Thompson Sampling line with its convergence slope
    (regret growth rate per episode) over the final 100 episodes.
    A slope near zero indicates the algorithm has converged.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)
    thompson_mean_curve = None

    for mname, style in _METHOD_STYLES.items():
        sub = raw_df[raw_df["method"] == mname]
        pivot = sub.pivot_table(index="episode", columns="trial", values="cumulative_regret")
        mean_curve = pivot.mean(axis=1).values
        std_curve = pivot.std(axis=1).values

        ax.plot(episodes, mean_curve, label=mname, color=style["color"],
                linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                        color=style["color"], alpha=0.10)

        if mname == "Adaptive (Thompson)":
            thompson_mean_curve = mean_curve

    # --- convergence slope annotation for Thompson Sampling ---
    if thompson_mean_curve is not None and num_episodes > 100:
        tail = 100  # analyse the final 100 episodes
        tail_episodes = episodes[-tail:]
        tail_regret = thompson_mean_curve[-tail:]
        # Linear fit: regret ≈ slope * episode + intercept
        slope, _ = np.polyfit(tail_episodes, tail_regret, 1)
        annotation_x = num_episodes - tail // 2
        annotation_y = thompson_mean_curve[annotation_x - 1]
        converged = "converged ✓" if slope < 0.05 else "still exploring"
        ax.annotate(
            f"Thompson slope={slope:.3f}/ep ({converged})",
            xy=(annotation_x, annotation_y),
            xytext=(annotation_x - num_episodes * 0.25, annotation_y + 5),
            fontsize=8, color=_METHOD_STYLES["Adaptive (Thompson)"]["color"],
            arrowprops=dict(
                arrowstyle="->",
                color=_METHOD_STYLES["Adaptive (Thompson)"]["color"],
                lw=1.2,
            ),
        )

    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1, alpha=0.5, label="Oracle (zero regret)")
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


def plot_cost(
    raw_df: pd.DataFrame,
    num_episodes: int,
    save_dir: Path,
) -> None:
    """Plot mean average per-episode cost with shaded std band per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)

    for mname, style in _METHOD_STYLES.items():
        sub = raw_df[raw_df["method"] == mname]
        pivot = sub.pivot_table(index="episode", columns="trial", values="average_cost")
        mean_curve = pivot.mean(axis=1).values
        std_curve = pivot.std(axis=1).values

        ax.plot(episodes, mean_curve, label=mname, color=style["color"],
                linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                        color=style["color"], alpha=0.10)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average per-episode Cost", fontsize=12)
    ax.set_title("Average Recovery Cost - Method Comparison", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / "cost_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Cost plot saved to {save_dir / 'cost_curve.png'}")


def run_and_report(
    num_episodes: int = 500,
    num_trials: int = 50,
    data_dir: Optional[Path] = None,
    show_late_stage_marker: bool = True,
    results_dir: Optional[Path] = None,
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
    plot_cost(raw_df, num_episodes, data_dir)

    # Print summary
    print(f"\n  {'='*120}")
    print(f"  Summary (averaged over {num_trials} trials x {num_episodes} episodes)")
    print(f"  {'='*120}")
    print(summary_df.to_string(float_format="{:.4f}".format))
    print()

    # 95% CI summary — success rates (compact view)
    ci_cols = ["mean_cumulative_success_rate", "ci95_lower_cumulative", "ci95_upper_cumulative",
               "mean_late_stage_success_rate", "ci95_lower_late_stage", "ci95_upper_late_stage"]
    print(f"  {'='*100}")
    print("  95% Confidence Intervals — Success Rate (mean +/- 1.96*SEM)")
    print(f"  {'='*100}")
    print(summary_df[ci_cols].to_string(float_format="{:.4f}".format))
    print()

    # Oracle/UniformRandom-normalized scores (0 = floor, 1 = ceiling)
    norm_cols = ["mean_cumulative_success_rate", "normalized_cumulative_score",
                 "mean_late_stage_success_rate", "normalized_late_stage_score"]
    print(f"  {'='*100}")
    print("  Normalized Score — (method_rate - UniformRandom_rate) / (Oracle_rate - UniformRandom_rate)")
    print(f"  {'='*100}")
    print(summary_df[norm_cols].to_string(float_format="{:.4f}".format))
    print()

    # 95% CI summary — cost (compact view)
    cost_ci_cols = ["mean_average_cost", "std_average_cost", "ci95_lower_cost", "ci95_upper_cost"]
    print(f"  {'='*100}")
    print("  95% Confidence Intervals — Average Recovery Cost (mean +/- 1.96*SEM)")
    print(f"  {'='*100}")
    print(summary_df[cost_ci_cols].to_string(float_format="{:.4f}".format))
    print()

    significance_df = summary_df.attrs.get("late_stage_significance")
    if significance_df is not None:
        print(f"  {'='*80}")
        print("  Late-stage significance vs Adaptive (UCB1)")
        print(f"  {'='*80}")
        print(significance_df.to_string(float_format="{:.6f}".format))
        print()

    effect_df = summary_df.attrs.get("thompson_effect_sizes")
    if effect_df is not None:
        print(f"  {'='*100}")
        print("  Cohen's d effect sizes — Thompson Sampling vs each method (late-stage success)")
        print(f"  {'='*100}")
        print(effect_df.to_string(float_format="{:.4f}".format))
        print()

    cost_effect_df = summary_df.attrs.get("cost_effect_sizes")
    if cost_effect_df is not None:
        print(f"  {'='*100}")
        print("  Cohen's d effect sizes — Contextual Cost-Aware vs each method (average cost)")
        print(f"  {'='*100}")
        print(cost_effect_df.to_string(float_format="{:.4f}".format))
        print()

    # Save cost-efficiency summary CSV (separate from raw results)
    cost_summary_df = summary_df[["mean_average_cost", "std_average_cost",
                                   "ci95_lower_cost", "ci95_upper_cost",
                                   "mean_late_stage_success_rate",
                                   "ci95_lower_late_stage", "ci95_upper_late_stage",
                                   "normalized_late_stage_score"]].copy()
    if cost_effect_df is not None:
        cost_summary_df = cost_summary_df.join(
            cost_effect_df[["cohens_d", "effect_magnitude", "p_value", "significant"]],
            how="left",
        )
    cost_csv_path = data_dir / "cost_efficiency_summary.csv"
    cost_summary_df.to_csv(str(cost_csv_path))
    print(f"  Cost-efficiency summary saved to {cost_csv_path}")

    # Save the updated comparison table (with normalized scores)
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase1"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = results_dir / "comparison_summary.csv"
    summary_df.to_csv(str(summary_csv_path))
    print(f"  Comparison summary (with normalized scores) saved to {summary_csv_path}")
    if significance_df is not None:
        significance_df.to_csv(str(results_dir / "late_stage_significance.csv"))
    if effect_df is not None:
        effect_df.to_csv(str(results_dir / "thompson_effect_sizes.csv"))
    if cost_effect_df is not None:
        cost_effect_df.to_csv(str(results_dir / "cost_effect_sizes.csv"))

    print(f"\n  NOTE: {MC_VARIATION_NOTE}")

    return summary_df


# ======================================================================
# Lambda-cost sweep (discussion-section analysis)
# ======================================================================


def run_lambda_sweep(
    lambda_values: list[float] | None = None,
    num_episodes: int = 500,
    num_trials: int = 50,
    base_seed: int = 123,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Sweep lambda_cost for Contextual Cost-Aware and compare vs Thompson.

    For each lambda value, runs both Thompson Sampling and
    Contextual Cost-Aware over the same episode sequences, then reports:

    - Mean late-stage success rate and average cost (with 95 % CIs)
    - Cohen's d vs Thompson on both metrics

    This is a **separate analysis** for the discussion section.  It does
    NOT change the default lambda (0.3) used in the main comparison.

    Returns:
        A DataFrame with one row per lambda value.
    """
    if lambda_values is None:
        lambda_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    late_stage_start_idx = min(max(int(num_episodes * 0.8), 0), num_episodes - 1)
    sweep_rows: list[dict] = []

    print(f"\n  Lambda sweep: {lambda_values}")
    print(f"  {num_episodes} episodes x {num_trials} trials per lambda value\n")

    for lam in lambda_values:
        # Per-trial late-stage success & cost for each method
        ts_late_rates: list[float] = []
        ts_costs: list[float] = []
        cc_late_rates: list[float] = []
        cc_costs: list[float] = []

        for trial_idx in range(num_trials):
            trial_seed = base_seed + trial_idx * 1000

            # Shared failure-type sequence
            type_rng = random.Random(trial_seed)
            fts = [type_rng.choice(list(FailureType)) for _ in range(num_episodes)]
            outcome_rng = random.Random(trial_seed + 7)

            thompson = SelectorFactory.create("thompson_sampling", seed=trial_seed + 500)
            contextual = ContextualCostAwareSelector(
                lambda_cost=lam, alpha=0.25, seed=trial_seed + 1000,
            )

            methods = {"Thompson": thompson, "Contextual": contextual}
            successes: dict[str, list[bool]] = {m: [] for m in methods}
            total_cost: dict[str, float] = {m: 0.0 for m in methods}

            for ep in range(num_episodes):
                ft = fts[ep]
                for mname, sel in methods.items():
                    retry_count = outcome_rng.randint(0, 3)
                    success, cost = _simulate_episode(
                        sel, ft, outcome_rng, ep + 1,
                        method_name=mname, retry_count=retry_count,
                    )
                    successes[mname].append(success)
                    total_cost[mname] += cost

            for mname, target_rates, target_costs in [
                ("Thompson", ts_late_rates, ts_costs),
                ("Contextual", cc_late_rates, cc_costs),
            ]:
                late = successes[mname][late_stage_start_idx:]
                target_rates.append(float(np.mean(late)))
                target_costs.append(total_cost[mname] / num_episodes)

        # --- Compute stats for this lambda ---
        ts_rate_arr = np.array(ts_late_rates)
        cc_rate_arr = np.array(cc_late_rates)
        ts_cost_arr = np.array(ts_costs)
        cc_cost_arr = np.array(cc_costs)
        n = num_trials

        # Success-rate Cohen's d (positive = Thompson better)
        rate_pooled = np.sqrt(
            ((n - 1) * np.var(ts_rate_arr, ddof=1) + (n - 1) * np.var(cc_rate_arr, ddof=1))
            / (2 * n - 2)
        )
        rate_d = float((np.mean(ts_rate_arr) - np.mean(cc_rate_arr)) / rate_pooled) if rate_pooled > 0 else 0.0
        _, rate_p = ttest_rel(ts_rate_arr, cc_rate_arr)

        # Cost Cohen's d (positive = Cost-Aware cheaper)
        cost_pooled = np.sqrt(
            ((n - 1) * np.var(ts_cost_arr, ddof=1) + (n - 1) * np.var(cc_cost_arr, ddof=1))
            / (2 * n - 2)
        )
        cost_d = float((np.mean(ts_cost_arr) - np.mean(cc_cost_arr)) / cost_pooled) if cost_pooled > 0 else 0.0
        _, cost_p = ttest_rel(ts_cost_arr, cc_cost_arr)

        # CIs
        cc_rate_sem = float(np.std(cc_rate_arr, ddof=1) / np.sqrt(n))
        cc_cost_sem = float(np.std(cc_cost_arr, ddof=1) / np.sqrt(n))
        ts_rate_sem = float(np.std(ts_rate_arr, ddof=1) / np.sqrt(n))
        ts_cost_sem = float(np.std(ts_cost_arr, ddof=1) / np.sqrt(n))

        sweep_rows.append({
            "lambda_cost": lam,
            "cc_success_rate": float(np.mean(cc_rate_arr)),
            "cc_success_ci": f"[{np.mean(cc_rate_arr) - 1.96 * cc_rate_sem:.4f}, {np.mean(cc_rate_arr) + 1.96 * cc_rate_sem:.4f}]",
            "cc_avg_cost": float(np.mean(cc_cost_arr)),
            "cc_cost_ci": f"[{np.mean(cc_cost_arr) - 1.96 * cc_cost_sem:.4f}, {np.mean(cc_cost_arr) + 1.96 * cc_cost_sem:.4f}]",
            "ts_success_rate": float(np.mean(ts_rate_arr)),
            "ts_success_ci": f"[{np.mean(ts_rate_arr) - 1.96 * ts_rate_sem:.4f}, {np.mean(ts_rate_arr) + 1.96 * ts_rate_sem:.4f}]",
            "ts_avg_cost": float(np.mean(ts_cost_arr)),
            "ts_cost_ci": f"[{np.mean(ts_cost_arr) - 1.96 * ts_cost_sem:.4f}, {np.mean(ts_cost_arr) + 1.96 * ts_cost_sem:.4f}]",
            "success_cohens_d": rate_d,
            "success_p_value": float(rate_p),
            "cost_cohens_d": cost_d,
            "cost_p_value": float(cost_p),
        })

        print(f"  lambda={lam:.1f}  CC: success={np.mean(cc_rate_arr):.4f}  cost={np.mean(cc_cost_arr):.4f}  |  "
              f"TS: success={np.mean(ts_rate_arr):.4f}  cost={np.mean(ts_cost_arr):.4f}  |  "
              f"d(success)={rate_d:+.3f}  d(cost)={cost_d:+.3f}")

    sweep_df = pd.DataFrame(sweep_rows)

    # Print summary table
    print(f"\n  {'='*120}")
    print(f"  Lambda Sweep: Contextual Cost-Aware vs Thompson Sampling ({num_trials} trials x {num_episodes} episodes)")
    print(f"  {'='*120}")
    print(sweep_df.to_string(index=False, float_format="{:.4f}".format))
    print()
    print("  Cohen's d sign convention:")
    print("    success_cohens_d > 0  =>  Thompson has HIGHER success rate")
    print("    cost_cohens_d > 0     =>  Cost-Aware is CHEAPER (lower cost)")
    print()

    # Save CSV
    csv_path = data_dir / "lambda_sweep_results.csv"
    sweep_df.to_csv(str(csv_path), index=False)
    print(f"  Sweep results saved to {csv_path}\n")

    return sweep_df


# ======================================================================
# Phase 2: classifier-in-the-loop context-noise sweep
# ======================================================================

_CONTEXT_SWEEP_METHODS: dict[str, dict] = {
    "Adaptive (Thompson)": {"color": "#16a34a", "marker": "P"},
    "Fixed Rule":          {"color": "#f59e0b", "marker": "s"},
    "Naive Retry":         {"color": "#9ca3af", "marker": "x"},
}


def _make_context_sweep_selectors(trial_seed: int) -> dict[str, BaseSelector]:
    return {
        "Adaptive (Thompson)": SelectorFactory.create("thompson_sampling", seed=trial_seed + 500),
        "Fixed Rule":          FixedRuleBaseline(),
        "Naive Retry":         NaiveRetryBaseline(),
    }


def _load_phase1_floor_ceiling(results_dir: Optional[Path] = None) -> tuple[float, float]:
    """Read Phase 1's UniformRandom/Oracle late-stage rates as the fixed
    normalization floor/ceiling (see harness._add_normalized_scores).

    These two methods are, by construction, unaffected by classifier
    noise — Oracle always has privileged access to the true failure type
    and UniformRandom ignores the failure type entirely — so it is valid
    to hold them fixed across the whole context-noise sweep.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase1"
    phase1_csv = results_dir / "comparison_summary.csv"
    if not phase1_csv.exists():
        raise FileNotFoundError(
            f"Phase 1 summary not found at {phase1_csv}. Run run_and_report() first "
            f"so the context sweep has a fixed Oracle/UniformRandom reference frame."
        )
    phase1_df = pd.read_csv(phase1_csv, index_col="method")
    oracle_rate = float(phase1_df.loc["Oracle", "mean_late_stage_success_rate"])
    random_rate = float(phase1_df.loc["UniformRandom", "mean_late_stage_success_rate"])
    return random_rate, oracle_rate


def run_context_mode_sweep(
    beta_values: Optional[list[float]] = None,
    num_episodes: int = 500,
    num_trials: int = 50,
    base_seed: int = 123,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Sweep synthetic classifier noise (beta) and compare Thompson Sampling,
    Fixed Rule, and Naive Retry when they only see a *noisy* failure-type
    observation instead of ground truth.

    For each beta, C(beta) = beta * C_emp + (1 - beta) * Uniform interpolates
    between the real classifier's confusion matrix (beta=1) and pure random
    guessing among the 7 types (beta=0). Each episode's true failure type is
    still used to sample the actual recovery outcome (SUCCESS_MATRIX);
    only the selector's *belief* about the failure type is drawn from
    C(beta)'s row for that true type — this isolates the effect of
    classifier noise from every other source of randomness in the harness.

    Returns:
        A long-format DataFrame with one row per (beta, method): beta,
        effective_accuracy, late_stage_success_rate, and
        normalized_late_stage_score (using Phase 1's fixed Oracle/
        UniformRandom floor & ceiling).
    """
    if beta_values is None:
        beta_values = [round(1.0 - 0.1 * i, 1) for i in range(11)]  # 1.0 .. 0.0
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    random_rate, oracle_rate = _load_phase1_floor_ceiling()
    denom = oracle_rate - random_rate

    print("\n  Training/loading classifier and computing empirical confusion matrix...")
    clf = load_or_train_classifier(data_dir)
    c_emp = compute_empirical_confusion_matrix(clf)

    late_stage_start_idx = min(max(int(num_episodes * 0.8), 0), num_episodes - 1)
    method_names = list(_CONTEXT_SWEEP_METHODS.keys())
    sweep_rows: list[dict] = []

    print(f"  Context-noise sweep: beta={beta_values}")
    print(f"  {num_episodes} episodes x {num_trials} trials per beta\n")

    for beta in beta_values:
        c_beta = interpolated_confusion_matrix(c_emp, beta)
        eff_acc = effective_accuracy(c_beta)
        late_rates: dict[str, list[float]] = {m: [] for m in method_names}

        for trial_idx in range(num_trials):
            trial_seed = base_seed + trial_idx * 1000
            type_rng = random.Random(trial_seed)
            fts = [type_rng.choice(list(FailureType)) for _ in range(num_episodes)]
            outcome_rng = random.Random(trial_seed + 7)
            context_rng = random.Random(trial_seed + 9001)

            selectors = _make_context_sweep_selectors(trial_seed)
            episode_successes: dict[str, list[bool]] = {m: [] for m in method_names}

            for ep in range(num_episodes):
                true_ft = fts[ep]
                observed_ft = sample_synthetic_context(true_ft, c_beta, context_rng)

                for mname, sel in selectors.items():
                    retry_count = outcome_rng.randint(0, 3)
                    success, _cost = _simulate_episode(
                        sel, true_ft, outcome_rng, ep + 1,
                        method_name=mname, retry_count=retry_count,
                        observed_failure_type=observed_ft,
                    )
                    episode_successes[mname].append(success)

            for mname in method_names:
                late = episode_successes[mname][late_stage_start_idx:]
                late_rates[mname].append(float(np.mean(late)))

        row_summary = []
        for mname in method_names:
            mean_rate = float(np.mean(late_rates[mname]))
            normalized = (mean_rate - random_rate) / denom if denom != 0 else float("nan")
            # Bootstrap 95% CI on the rate, then map through the same fixed
            # affine normalization (denom > 0 checked above) to get a CI on
            # the normalized score too.
            rate_ci_lo, rate_ci_hi = bootstrap_ci(late_rates[mname])
            norm_ci_lo = (rate_ci_lo - random_rate) / denom if denom != 0 else float("nan")
            norm_ci_hi = (rate_ci_hi - random_rate) / denom if denom != 0 else float("nan")
            sweep_rows.append({
                "beta": beta,
                "effective_accuracy": eff_acc,
                "method": mname,
                "late_stage_success_rate": mean_rate,
                "bootstrap_ci95_lower_rate": rate_ci_lo,
                "bootstrap_ci95_upper_rate": rate_ci_hi,
                "normalized_late_stage_score": normalized,
                "bootstrap_ci95_lower_normalized": norm_ci_lo,
                "bootstrap_ci95_upper_normalized": norm_ci_hi,
            })
            row_summary.append(f"{mname}={mean_rate:.3f} (norm={normalized:+.3f})")

        print(f"  beta={beta:.1f}  eff_acc={eff_acc:.3f}  |  " + "  ".join(row_summary))

    return pd.DataFrame(sweep_rows)


def plot_context_sweep(sweep_df: pd.DataFrame, save_dir: Path) -> Optional[float]:
    """Plot normalized late-stage score vs. effective classifier accuracy.

    Marks the effective-accuracy point at which Adaptive (Thompson)'s
    line crosses below Fixed Rule's line, if any.

    Returns:
        The crossover effective_accuracy (linearly interpolated between
        the two nearest beta points), or None if the lines never cross.
    """
    pivot = sweep_df.pivot_table(
        index="effective_accuracy", columns="method", values="normalized_late_stage_score",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    for mname, style in _CONTEXT_SWEEP_METHODS.items():
        if mname not in pivot.columns:
            continue
        ax.plot(
            pivot.index, pivot[mname], label=mname,
            color=style["color"], marker=style["marker"], linewidth=2, markersize=6,
        )

    crossover_x = None
    if "Adaptive (Thompson)" in pivot.columns and "Fixed Rule" in pivot.columns:
        xs = pivot.index.values
        diff = (pivot["Adaptive (Thompson)"] - pivot["Fixed Rule"]).values
        for i in range(len(xs) - 1):
            d0, d1 = diff[i], diff[i + 1]
            if d0 == 0:
                crossover_x = float(xs[i])
                break
            if (d0 > 0) != (d1 > 0):
                frac = d0 / (d0 - d1)
                crossover_x = float(xs[i] + frac * (xs[i + 1] - xs[i]))
                break

        if crossover_x is not None:
            ax.axvline(crossover_x, color="#dc2626", linestyle="--", linewidth=1.3, alpha=0.8)
            y_top = ax.get_ylim()[1]
            ax.annotate(
                f"Thompson crosses below\nFixed Rule at acc={crossover_x:.3f}",
                xy=(crossover_x, y_top * 0.5),
                xytext=(crossover_x + 0.05, y_top * 0.75),
                fontsize=9, color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
            )

    ax.axhline(0, color="#6b7280", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(1, color="#6b7280", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Effective classification accuracy", fontsize=12)
    ax.set_ylabel("Late-stage % of Oracle (normalized score)", fontsize=12)
    ax.set_title("Classifier-in-the-Loop: Context-Noise Sweep", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "context_sweep.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Context-noise sweep figure saved to {fig_path}")

    return crossover_x


def run_phase2_context_sweep(
    beta_values: Optional[list[float]] = None,
    num_episodes: int = 500,
    num_trials: int = 50,
    base_seed: int = 123,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full Phase 2 sweep, plot the figure, and save data to results/phase2/.

    Returns the long-format sweep DataFrame (see ``run_context_mode_sweep``).
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase2"
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = run_context_mode_sweep(
        beta_values=beta_values, num_episodes=num_episodes,
        num_trials=num_trials, base_seed=base_seed, data_dir=data_dir,
    )

    csv_path = results_dir / "context_sweep_results.csv"
    sweep_df.to_csv(str(csv_path), index=False)
    print(f"\n  Context-noise sweep data saved to {csv_path}")

    crossover_x = plot_context_sweep(sweep_df, results_dir)
    if crossover_x is not None:
        print(f"  Crossover: the Adaptive (Thompson) and Fixed Rule lines cross at "
              f"effective accuracy ~= {crossover_x:.3f}")
    else:
        # No sign change in (Thompson - Fixed Rule) across the sweep — report
        # which method is on top rather than assuming a fixed narrative.
        pivot = sweep_df.pivot_table(
            index="effective_accuracy", columns="method",
            values="normalized_late_stage_score",
        )
        if "Adaptive (Thompson)" in pivot and "Fixed Rule" in pivot:
            mean_gap = float((pivot["Adaptive (Thompson)"] - pivot["Fixed Rule"]).mean())
            leader = "Adaptive (Thompson)" if mean_gap > 0 else "Fixed Rule"
            print(f"  No crossover found — {leader} stayed above the other across the "
                  f"whole accuracy range swept (mean normalized-score gap "
                  f"Thompson-minus-FixedRule = {mean_gap:+.3f}).")
        else:
            print("  No crossover found between Thompson and Fixed Rule.")

    print(f"\n  NOTE: {MC_VARIATION_NOTE}")

    return sweep_df
