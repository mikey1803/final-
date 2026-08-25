"""Phase 3B: non-stationarity mini-experiment.

At episode 250 of 500, the reward distribution for 2 of the 7 failure
types is shifted (their SUCCESS_MATRIX row is permuted across strategies
— simulating a tool/API degradation where the previously-best recovery
strategy for that failure type is no longer reliable). Compares stock
Thompson Sampling against a discounted variant
(``DiscountedThompsonSamplingSelector``, gamma=0.99) to see whether
discounting the Beta posterior lets the bandit re-adapt faster after the
shift.

Outcomes are sampled directly against a local matrix (pre- or
post-shift) rather than through ``strategies.execute_strategy`` (which
is hardwired to the module-level SUCCESS_MATRIX), so the shift can be
applied without mutating global state.
"""

import random
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.baselines import FixedRuleBaseline
from src.evaluation.stats_utils import MC_VARIATION_NOTE, bootstrap_ci
from src.failures.taxonomy import FailureType
from src.recovery.selector import SelectorFactory
from src.recovery.strategies import RecoveryStrategy, SUCCESS_MATRIX, get_action_cost

_TYPES: list[FailureType] = list(FailureType)
_STRATS: list[RecoveryStrategy] = list(RecoveryStrategy)

_METHOD_STYLES: dict[str, dict] = {
    "thompson_sampling":            {"label": "Stock TS",       "color": "#2563eb"},
    "discounted_thompson_sampling": {"label": "Discounted TS (gamma=0.99)", "color": "#dc2626"},
}

# Extended method set including the (blind) static Fixed Rule baseline, used
# by the Phase 4b re-run. Fixed Rule has no ``update()`` — it conditions on
# the true failure type and plays a fixed strategy, so by construction it
# cannot re-adapt after the mid-run reward shift.
_METHOD_STYLES_WITH_FIXED: dict[str, dict] = {
    "thompson_sampling":            {"label": "Stock TS",       "color": "#2563eb"},
    "discounted_thompson_sampling": {"label": "Discounted TS (gamma=0.99)", "color": "#dc2626"},
    "fixed_rule":                   {"label": "Fixed Rule (blind)", "color": "#f59e0b", "linestyle": "--"},
}


def _make_nonstat_selector(name: str, trial_seed: int):
    """Construct one selector by name for the non-stationarity experiment.

    ``fixed_rule`` -> the blind static ``FixedRuleBaseline``; anything else
    is delegated to ``SelectorFactory`` (bandit algorithms).
    """
    if name == "fixed_rule":
        return FixedRuleBaseline()
    return SelectorFactory.create(name, seed=trial_seed + 500)


def pick_shift_types(rng: random.Random, n: int = 2) -> list[FailureType]:
    """Deterministically (given rng) choose which failure types degrade."""
    return rng.sample(_TYPES, n)


def build_shifted_matrix(
    shift_types: list[FailureType],
    rng: random.Random,
) -> dict[FailureType, dict[RecoveryStrategy, float]]:
    """Permute the strategy->probability mapping for *shift_types* only.

    The set of probability values available for that failure type is
    unchanged (so the row's difficulty/gap is preserved) — only which
    strategy achieves which probability is scrambled, so whatever was
    previously the best recovery strategy is no longer reliably best.
    """
    shifted = {ft: dict(row) for ft, row in SUCCESS_MATRIX.items()}
    for ft in shift_types:
        values = [SUCCESS_MATRIX[ft][rs] for rs in _STRATS]
        shuffled = values[:]
        rng.shuffle(shuffled)
        shifted[ft] = {rs: v for rs, v in zip(_STRATS, shuffled)}
    return shifted


def _sample_outcome(
    matrix: dict[FailureType, dict[RecoveryStrategy, float]],
    ft: FailureType,
    strategy: RecoveryStrategy,
    rng: random.Random,
) -> bool:
    return rng.random() < matrix[ft][strategy]


def run_nonstationarity_experiment(
    num_episodes: int = 500,
    shift_episode: int = 250,
    num_trials: int = 50,
    base_seed: int = 777,
    algorithms: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run stock vs discounted Thompson Sampling through a mid-run reward
    shift.

    Returns:
        (per_episode_df, shift_type_names) where per_episode_df has one
        row per (trial, episode, method) with a rolling-window success
        indicator, ready for curve plotting.
    """
    if algorithms is None:
        algorithms = list(_METHOD_STYLES.keys())

    shift_picker_rng = random.Random(base_seed)
    shift_types = pick_shift_types(shift_picker_rng)
    permute_rng = random.Random(base_seed + 1)
    shifted_matrix = build_shifted_matrix(shift_types, permute_rng)

    print(f"  Shift at episode {shift_episode}: permuting rows for "
          f"{[ft.value for ft in shift_types]}")

    rows: list[dict] = []
    for trial_idx in range(num_trials):
        trial_seed = base_seed + trial_idx * 1000
        type_rng = random.Random(trial_seed)
        fts = [type_rng.choice(_TYPES) for _ in range(num_episodes)]
        outcome_rng = random.Random(trial_seed + 7)

        selectors = {algo: _make_nonstat_selector(algo, trial_seed) for algo in algorithms}

        for ep in range(num_episodes):
            ft = fts[ep]
            matrix = shifted_matrix if (ep >= shift_episode and ft in shift_types) else SUCCESS_MATRIX
            for algo in algorithms:
                sel = selectors[algo]
                strategy = sel.select_strategy(ft)
                success = _sample_outcome(matrix, ft, strategy, outcome_rng)
                cost = get_action_cost(strategy)
                if hasattr(sel, "update"):  # static baselines (Fixed Rule) have none
                    sel.update(ft, strategy, success=success, cost=cost, episode_number=ep + 1)
                rows.append({
                    "trial": trial_idx, "episode": ep + 1, "method": algo, "success": success,
                    "failure_type": ft.value, "is_shifted_type": ft in shift_types,
                })

    per_episode_df = pd.DataFrame(rows)
    return per_episode_df, [ft.value for ft in shift_types]


def _rolling_success_rate(df: pd.DataFrame, method: str, window: int, num_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std (across trials) of a rolling-window success rate per episode."""
    sub = df[df["method"] == method]
    pivot = sub.pivot_table(index="episode", columns="trial", values="success", aggfunc="mean")
    pivot = pivot.reindex(range(1, num_episodes + 1))
    rolling = pivot.rolling(window=window, min_periods=1).mean()
    return rolling.mean(axis=1).values, rolling.std(axis=1).values


def plot_nonstationarity_shifted_type_only(
    per_episode_df: pd.DataFrame,
    shift_episode: int,
    save_dir: Path,
    window: int = 15,
    method_styles: Optional[dict] = None,
    filename: str = "nonstationarity_shifted_type_only.png",
) -> None:
    """Plot recovery curves restricted to episodes of the *shifted* failure
    types only, x-axis = occurrence index (1st, 2nd, ... exposure to a
    shifted type within the trial). The aggregate all-types curve dilutes
    the shift's effect since only 2 of 7 types are actually affected; this
    isolates the signal.
    """
    if method_styles is None:
        method_styles = _METHOD_STYLES
    fig, ax = plt.subplots(figsize=(10, 6))
    shift_occurrences = []

    for algo, style in method_styles.items():
        sub = per_episode_df[
            (per_episode_df["method"] == algo) & (per_episode_df["is_shifted_type"])
        ].sort_values(["trial", "episode"]).copy()
        sub["occurrence_idx"] = sub.groupby("trial").cumcount() + 1

        pivot = sub.pivot_table(index="occurrence_idx", columns="trial", values="success")
        rolling = pivot.rolling(window=window, min_periods=1).mean()
        mean_curve = rolling.mean(axis=1)
        std_curve = rolling.std(axis=1)

        ax.plot(mean_curve.index, mean_curve.values, label=style["label"],
                 color=style["color"], linewidth=2)
        ax.fill_between(mean_curve.index, mean_curve.values - std_curve.values,
                         mean_curve.values + std_curve.values, color=style["color"], alpha=0.15)

        shift_occ = sub[sub["episode"] >= shift_episode].groupby("trial")["occurrence_idx"].min().mean()
        if not np.isnan(shift_occ):
            shift_occurrences.append(shift_occ)

    if shift_occurrences:
        mean_shift_occ = float(np.mean(shift_occurrences))
        ax.axvline(mean_shift_occ, color="#111827", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(mean_shift_occ, 0.03, "  Shift", rotation=90, va="bottom", ha="left",
                fontsize=9, color="#111827")

    ax.set_xlabel("Occurrence index (exposures to a shifted failure type)", fontsize=12)
    ax.set_ylabel(f"Rolling {window}-exposure success rate", fontsize=12)
    ax.set_title("Non-Stationarity: Shifted-Type-Only Recovery Curve", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / filename
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Shifted-type-only figure saved to {fig_path}")


def plot_nonstationarity(
    per_episode_df: pd.DataFrame,
    num_episodes: int,
    shift_episode: int,
    shift_types: list[str],
    save_dir: Path,
    window: int = 30,
    method_styles: Optional[dict] = None,
    filename: str = "nonstationarity_recovery_curve.png",
    title: str = "Non-Stationarity: Stock vs Discounted Thompson Sampling",
) -> None:
    """Plot rolling-window recovery curves for each method,
    with a marker at the shift episode."""
    if method_styles is None:
        method_styles = _METHOD_STYLES
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)

    for algo, style in method_styles.items():
        mean_curve, std_curve = _rolling_success_rate(per_episode_df, algo, window, num_episodes)
        linestyle = style.get("linestyle", "-")
        ax.plot(episodes, mean_curve, label=style["label"], color=style["color"],
                linewidth=2, linestyle=linestyle)
        ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                         color=style["color"], alpha=0.15)

    ax.axvline(shift_episode, color="#111827", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(shift_episode, ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0,
            f"  Shift ({', '.join(shift_types)})",
            rotation=0, va="top", ha="left", fontsize=9, color="#111827")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(f"Rolling {window}-episode success rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / filename
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Non-stationarity figure saved to {fig_path}")


def run_phase3b(
    num_episodes: int = 500,
    shift_episode: int = 250,
    num_trials: int = 50,
    base_seed: int = 777,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full Phase 3B experiment, plot the figure, and save data/notes
    to results/phase3b/."""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase3b"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_episode_df, shift_types = run_nonstationarity_experiment(
        num_episodes=num_episodes, shift_episode=shift_episode,
        num_trials=num_trials, base_seed=base_seed,
    )

    csv_path = results_dir / "nonstationarity_raw.csv"
    per_episode_df.to_csv(str(csv_path), index=False)
    print(f"  Raw per-episode data saved to {csv_path}")

    plot_nonstationarity(per_episode_df, num_episodes, shift_episode, shift_types, results_dir)
    plot_nonstationarity_shifted_type_only(per_episode_df, shift_episode, results_dir)

    # Pre-shift vs post-shift late-window summary with bootstrap CIs
    pre_window = (max(shift_episode - 50, 1), shift_episode)
    post_window = (shift_episode + 1, min(shift_episode + 50, num_episodes))
    summary_rows = []
    for algo, style in _METHOD_STYLES.items():
        sub = per_episode_df[per_episode_df["method"] == algo]
        for label, (lo_ep, hi_ep) in [("pre_shift_50ep", pre_window), ("post_shift_50ep", post_window)]:
            window_df = sub[(sub["episode"] >= lo_ep) & (sub["episode"] <= hi_ep)]
            per_trial_rate = window_df.groupby("trial")["success"].mean().values
            ci_lo, ci_hi = bootstrap_ci(per_trial_rate)
            summary_rows.append({
                "method": style["label"], "window": label,
                "mean_success_rate": float(np.mean(per_trial_rate)),
                "bootstrap_ci95_lower": ci_lo, "bootstrap_ci95_upper": ci_hi,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "nonstationarity_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"  Pre/post-shift summary (with bootstrap CIs) saved to {summary_csv}")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))

    notes_path = results_dir / "NOTES.txt"
    notes_path.write_text(
        f"Phase 3B: non-stationarity mini-experiment. At episode {shift_episode} of "
        f"{num_episodes}, SUCCESS_MATRIX rows for failure types {shift_types} were "
        "permuted across strategies (simulating a tool/API degradation that "
        "invalidates the previously-best recovery strategy for those types). "
        f"Compared stock Thompson Sampling vs a discounted variant (gamma=0.99) "
        f"over {num_trials} trials.\n\n"
        f"{MC_VARIATION_NOTE}\n",
        encoding="utf-8",
    )
    print(f"  Notes saved to {notes_path}")

    return per_episode_df


def _shifted_type_window_rates(
    per_episode_df: pd.DataFrame,
    method: str,
    lo_ep: int,
    hi_ep: int,
) -> np.ndarray:
    """Per-trial success rate on *shifted-type* episodes within [lo_ep, hi_ep]."""
    sub = per_episode_df[
        (per_episode_df["method"] == method)
        & (per_episode_df["is_shifted_type"])
        & (per_episode_df["episode"] >= lo_ep)
        & (per_episode_df["episode"] <= hi_ep)
    ]
    return sub.groupby("trial")["success"].mean().values


def run_phase4b_nonstationarity(
    num_episodes: int = 500,
    shift_episode: int = 250,
    num_trials: int = 50,
    base_seed: int = 777,
    results_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 4b re-run: stock TS vs discounted TS vs blind Fixed Rule through
    the mid-run shift.

    Quantifies each method's *adaptation* on shifted-type episodes across
    three windows (pre-shift / early-post-shift / late-post-shift). The
    recovery delta (late-post minus early-post) is the headline: a static
    Fixed Rule cannot re-adapt, so its recovery delta should be ~0, while
    the bandits' should be positive.

    Returns:
        (per_episode_df, adaptation_df)
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase4b"
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = list(_METHOD_STYLES_WITH_FIXED.keys())
    per_episode_df, shift_types = run_nonstationarity_experiment(
        num_episodes=num_episodes, shift_episode=shift_episode,
        num_trials=num_trials, base_seed=base_seed, algorithms=methods,
    )

    csv_path = results_dir / "nonstationarity_raw.csv"
    per_episode_df.to_csv(str(csv_path), index=False)
    print(f"  Raw per-episode data saved to {csv_path}")

    plot_nonstationarity(
        per_episode_df, num_episodes, shift_episode, shift_types, results_dir,
        method_styles=_METHOD_STYLES_WITH_FIXED,
        filename="nonstationarity_recovery_curve.png",
        title="Non-Stationarity: Stock TS vs Discounted TS vs Fixed Rule (blind)",
    )
    plot_nonstationarity_shifted_type_only(
        per_episode_df, shift_episode, results_dir,
        method_styles=_METHOD_STYLES_WITH_FIXED,
        filename="nonstationarity_shifted_type_only.png",
    )

    # --- Adaptation quantification on shifted-type episodes ---
    win = min(100, shift_episode, num_episodes - shift_episode)
    windows = {
        "pre_shift":        (shift_episode - win, shift_episode - 1),
        "early_post_shift": (shift_episode + 1, shift_episode + win),
        "late_post_shift":  (num_episodes - win + 1, num_episodes),
    }
    adapt_rows = []
    for algo in methods:
        style = _METHOD_STYLES_WITH_FIXED[algo]
        rates: dict[str, np.ndarray] = {}
        row = {"method": style["label"]}
        for wname, (lo, hi) in windows.items():
            r = _shifted_type_window_rates(per_episode_df, algo, lo, hi)
            rates[wname] = r
            ci_lo, ci_hi = bootstrap_ci(r)
            row[f"{wname}_rate"] = float(np.mean(r))
            row[f"{wname}_ci_lo"] = ci_lo
            row[f"{wname}_ci_hi"] = ci_hi
        # Recovery delta = late-post minus early-post (paired across trials)
        recovery = rates["late_post_shift"] - rates["early_post_shift"]
        rec_lo, rec_hi = bootstrap_ci(recovery)
        row["recovery_delta"] = float(np.mean(recovery))
        row["recovery_delta_ci_lo"] = rec_lo
        row["recovery_delta_ci_hi"] = rec_hi
        # Immediate drop = early-post minus pre-shift
        row["shift_drop"] = float(np.mean(rates["early_post_shift"]) - np.mean(rates["pre_shift"]))
        adapt_rows.append(row)

    adaptation_df = pd.DataFrame(adapt_rows)
    adapt_csv = results_dir / "nonstationarity_adaptation.csv"
    adaptation_df.to_csv(str(adapt_csv), index=False)
    print(f"\n  Adaptation summary (shifted-type episodes) saved to {adapt_csv}")
    show_cols = ["method", "pre_shift_rate", "early_post_shift_rate",
                 "late_post_shift_rate", "shift_drop", "recovery_delta",
                 "recovery_delta_ci_lo", "recovery_delta_ci_hi"]
    print(adaptation_df[show_cols].to_string(index=False, float_format="{:.4f}".format))

    notes_path = results_dir / "NONSTATIONARITY_NOTES.md"
    notes_path.write_text(
        "# Phase 4b: Non-stationarity with the blind Fixed Rule\n\n"
        f"At episode {shift_episode} of {num_episodes}, the SUCCESS_MATRIX rows "
        f"for failure types {shift_types} were permuted across strategies "
        "(tool/API degradation invalidating the previously-best recovery "
        "strategy). Compared stock Thompson Sampling, discounted Thompson "
        f"(gamma=0.99), and the blind static Fixed Rule over {num_trials} trials.\n\n"
        "Metrics below are on **shifted-type episodes only** (the aggregate "
        "all-types curve dilutes the effect since only 2/7 types shift), in "
        f"{win}-episode windows:\n\n"
        "- **shift_drop** = early-post-shift rate minus pre-shift rate (how far "
        "each method falls right after the shift).\n"
        "- **recovery_delta** = late-post-shift rate minus early-post-shift rate. "
        "This is the adaptation signal: a static Fixed Rule cannot re-adapt by "
        "construction (it plays the same strategy for a given failure type "
        "regardless of outcomes), so its recovery_delta is ~0 within noise; the "
        "bandits' recovery_delta is positive if they re-learn the new best arm.\n\n"
        f"{MC_VARIATION_NOTE}\n",
        encoding="utf-8",
    )
    print(f"  Notes saved to {notes_path}")

    return per_episode_df, adaptation_df
