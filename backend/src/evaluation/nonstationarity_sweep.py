"""Phase 6: shift frequency x magnitude sweep.

Generalizes Phase 3B/4B's single mid-run reward shift
(``nonstationarity.py``) into a *periodic* shift applied every
``frequency`` episodes, affecting ``magnitude`` of the 7 failure types
each time. Sweeps frequency in {500, 250, 100, 50} x magnitude in
{1, 2, 4, 7} and compares the blind static Fixed Rule (6/7 correct
against SUCCESS_MATRIX) against Thompson Sampling, Discounted Thompson
Sampling, and LinUCB.

Reuses ``matrix_family.py``'s randomized reward matrix (``sample_matrix``)
and analytic Oracle/UniformRandom normalization (``analytic_oracle_rate``
/ ``analytic_random_rate`` — both invariant to the shifts applied here,
since a shift only permutes which strategy achieves which probability
within a row, never changing the row's own max or mean). The per-episode
select/update loop mirrors ``matrix_family.simulate_selector_on_matrix``
and ``nonstationarity.run_nonstationarity_experiment`` (same RNG/seed
pattern), generalized to a mutating matrix — that generalization is the
one genuinely new piece of simulation logic this phase needs, since
neither existing pipeline supports a time-varying matrix.

Recovery delta reuses the exact definition from
``nonstationarity.run_phase4b_nonstationarity`` (late-post-shift rate
minus early-post-shift rate), applied per shift event instead of once.
"""

import random
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.baselines import FixedRuleBaseline, LinUCBBaseline
from src.evaluation.matrix_family import (
    _STRATS,
    _TYPES,
    analytic_oracle_rate,
    analytic_random_rate,
    sample_matrix,
)
from src.evaluation.stats_utils import MC_VARIATION_NOTE, bootstrap_ci
from src.recovery.selector import SelectorFactory

_METHOD_LABELS: dict[str, str] = {
    "fixed_rule":                   "Blind Static Rule (6/7)",
    "thompson_sampling":            "Thompson Sampling",
    "discounted_thompson_sampling": "Discounted Thompson Sampling",
    "linucb":                       "LinUCB",
}
_METHOD_COLORS: dict[str, str] = {
    "fixed_rule":                   "#f59e0b",
    "thompson_sampling":            "#16a34a",
    "discounted_thompson_sampling": "#7c3aed",
    "linucb":                       "#0d9488",
}


def _make_selector(method: str, seed: int):
    """Construct one selector by name, matching the seed-offset convention
    used throughout the harness/nonstationarity modules (+500 Thompson-family,
    +1500 LinUCB)."""
    if method == "fixed_rule":
        return FixedRuleBaseline()
    if method == "linucb":
        return LinUCBBaseline(alpha=1.0, seed=seed + 1500)
    return SelectorFactory.create(method, seed=seed + 500)


def build_shift_schedule(
    base_matrix: np.ndarray,
    magnitude: int,
    frequency: int,
    num_episodes: int,
    rng: random.Random,
) -> list[tuple[int, dict[int, np.ndarray]]]:
    """Precompute a periodic shift schedule for one (frequency, magnitude)
    environment: at episodes ``frequency, 2*frequency, ...``, permute the
    strategy->probability assignment (shuffle values within the row, same
    mechanic as ``nonstationarity.build_shifted_matrix``) for *magnitude*
    randomly chosen failure-type rows. Shuffles compound over successive
    events on a running working copy, so later shifts are relative to the
    most recently shifted state.

    Shared across every method/seed evaluated on this environment, so all
    methods face an identical sequence of shifts (only their own learning
    and the outcome-sampling noise differ per seed).
    """
    n_types, n_strats = base_matrix.shape
    working = base_matrix.copy()
    schedule: list[tuple[int, dict[int, np.ndarray]]] = []
    for episode in range(frequency, num_episodes, frequency):
        type_idxs = rng.sample(range(n_types), magnitude)
        new_rows: dict[int, np.ndarray] = {}
        for t in type_idxs:
            perm = list(range(n_strats))
            rng.shuffle(perm)
            working[t] = working[t][perm]
            new_rows[t] = working[t].copy()
        schedule.append((episode, new_rows))
    return schedule


def simulate_selector_with_shifts(
    base_matrix: np.ndarray,
    shift_schedule: list[tuple[int, dict[int, np.ndarray]]],
    selector,
    num_episodes: int,
    seed: int,
) -> np.ndarray:
    """Run one trial of *selector* against *base_matrix*, applying
    *shift_schedule*'s row permutations as their episodes are reached.

    Same type_rng/outcome_rng seeding and select/update loop as
    ``matrix_family.simulate_selector_on_matrix``, generalized to a
    matrix that mutates over time.
    """
    type_rng = random.Random(seed + 13)
    outcome_rng = random.Random(seed + 29)
    current_matrix = base_matrix.copy()
    schedule_idx = 0
    successes = np.zeros(num_episodes, dtype=bool)

    for ep in range(num_episodes):
        episode_num = ep + 1
        while schedule_idx < len(shift_schedule) and shift_schedule[schedule_idx][0] == episode_num:
            _, new_rows = shift_schedule[schedule_idx]
            for t, row in new_rows.items():
                current_matrix[t] = row
            schedule_idx += 1

        f_idx = type_rng.randrange(current_matrix.shape[0])
        ft = _TYPES[f_idx]
        strategy = selector.select_strategy(ft)
        s_idx = _STRATS.index(strategy)
        success = outcome_rng.random() < current_matrix[f_idx, s_idx]
        if hasattr(selector, "update"):  # Fixed Rule is static, no update
            selector.update(ft, strategy, success=success, cost=1.0, episode_number=episode_num)
        successes[ep] = success

    return successes


def _window_rate(successes: np.ndarray, ep_lo: int, ep_hi: int, num_episodes: int) -> float:
    """Mean success rate over 1-indexed inclusive episode range [ep_lo, ep_hi]."""
    ep_lo, ep_hi = max(ep_lo, 1), min(ep_hi, num_episodes)
    if ep_hi < ep_lo:
        return float("nan")
    return float(np.mean(successes[ep_lo - 1:ep_hi]))


def compute_event_metrics(
    successes: np.ndarray,
    shift_schedule: list[tuple[int, dict[int, np.ndarray]]],
    num_episodes: int,
    frequency: int,
) -> list[dict]:
    """Per shift-event pre/early-post/late-post windowed rates, recovery
    delta (late-post minus early-post — same definition as
    ``nonstationarity.run_phase4b_nonstationarity``), and adaptation
    half-life (episodes after the shift to recover 50% of the rate lost
    immediately after it; ``None`` if never reached before the next
    shift or run end).
    """
    window = max(3, frequency // 3)
    rw = max(3, window // 3)  # smoothing window for the half-life scan
    events = [e for e, _ in shift_schedule]
    results = []

    for idx, e in enumerate(events):
        cycle_end = events[idx + 1] - 1 if idx + 1 < len(events) else num_episodes

        pre_rate = _window_rate(successes, e - window, e - 1, num_episodes)
        early_rate = _window_rate(successes, e, e + window - 1, num_episodes)
        late_rate = _window_rate(successes, cycle_end - window + 1, cycle_end, num_episodes)

        shift_drop = early_rate - pre_rate
        recovery_delta = late_rate - early_rate

        lost = pre_rate - early_rate
        if not np.isfinite(lost) or lost <= 1e-9:
            half_life: Optional[int] = 0
        else:
            target = early_rate + 0.5 * lost
            half_life = None
            for ep in range(e + window, cycle_end + 1):
                r = _window_rate(successes, ep - rw + 1, ep, num_episodes)
                if np.isfinite(r) and r >= target:
                    half_life = ep - e
                    break

        results.append({
            "event_episode": e,
            "pre_rate": pre_rate,
            "early_rate": early_rate,
            "late_rate": late_rate,
            "shift_drop": shift_drop,
            "recovery_delta": recovery_delta,
            "half_life": half_life,
        })

    return results


def run_shift_frequency_magnitude_sweep(
    frequencies: Optional[list[int]] = None,
    magnitudes: Optional[list[int]] = None,
    num_episodes: int = 1000,
    n_seeds: int = 20,
    gap: float = 0.2,
    base_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep (frequency, magnitude) combinations; for each, build one
    shared randomized-matrix environment (``sample_matrix`` + periodic
    shift schedule) and replay every method over ``n_seeds`` independent
    outcome sequences.

    Returns:
        (results_df, event_metrics_df) — per-(frequency, magnitude,
        method, seed) overall normalized score, and per-(frequency,
        magnitude, method, seed, shift event) windowed recovery metrics.
    """
    if frequencies is None:
        frequencies = [500, 250, 100, 50]
    if magnitudes is None:
        magnitudes = [1, 2, 4, 7]

    result_rows: list[dict] = []
    event_rows: list[dict] = []

    print(f"  Shift frequency x magnitude sweep: frequencies={frequencies}  magnitudes={magnitudes}")
    print(f"  num_episodes={num_episodes}  n_seeds={n_seeds}  methods={list(_METHOD_LABELS.values())}\n")

    for frequency, magnitude in product(frequencies, magnitudes):
        combo_seed = base_seed + frequency * 13 + magnitude * 97

        matrix_rng = np.random.default_rng(combo_seed)
        base_matrix = sample_matrix(gap=gap, rng=matrix_rng)
        oracle_rate = analytic_oracle_rate(base_matrix)
        random_rate = analytic_random_rate(base_matrix)
        denom = oracle_rate - random_rate

        schedule_rng = random.Random(combo_seed + 1)
        shift_schedule = build_shift_schedule(base_matrix, magnitude, frequency, num_episodes, schedule_rng)

        for method in _METHOD_LABELS:
            for seed_idx in range(n_seeds):
                sim_seed = combo_seed + 1000 * (seed_idx + 1)
                selector = _make_selector(method, sim_seed)
                successes = simulate_selector_with_shifts(
                    base_matrix, shift_schedule, selector, num_episodes, sim_seed,
                )

                overall_rate = float(np.mean(successes))
                normalized = (overall_rate - random_rate) / denom if denom != 0 else float("nan")
                result_rows.append({
                    "frequency": frequency,
                    "magnitude": magnitude,
                    "method": method,
                    "method_label": _METHOD_LABELS[method],
                    "seed": seed_idx,
                    "overall_success_rate": overall_rate,
                    "normalized_score": normalized,
                })

                for event in compute_event_metrics(successes, shift_schedule, num_episodes, frequency):
                    event_rows.append({
                        "frequency": frequency,
                        "magnitude": magnitude,
                        "method": method,
                        "method_label": _METHOD_LABELS[method],
                        "seed": seed_idx,
                        **event,
                    })

        print(f"  frequency={frequency:>4}  magnitude={magnitude}  "
              f"n_shift_events={len(shift_schedule)}  oracle={oracle_rate:.3f}  random={random_rate:.3f}")

    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)


def summarize_results(results_df: pd.DataFrame, event_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean normalized score + 95% bootstrap CI, and mean recovery delta +
    95% bootstrap CI, per (frequency, magnitude, method)."""
    rows = []
    for (frequency, magnitude, method), sub in results_df.groupby(["frequency", "magnitude", "method"]):
        vals = sub["normalized_score"].values
        ci_lo, ci_hi = bootstrap_ci(vals)

        recov = event_metrics_df[
            (event_metrics_df["frequency"] == frequency)
            & (event_metrics_df["magnitude"] == magnitude)
            & (event_metrics_df["method"] == method)
        ]["recovery_delta"].dropna().values
        rec_ci_lo, rec_ci_hi = bootstrap_ci(recov) if len(recov) else (float("nan"), float("nan"))

        rows.append({
            "frequency": frequency,
            "magnitude": magnitude,
            "method": method,
            "method_label": _METHOD_LABELS[method],
            "n": len(vals),
            "mean_normalized_score": float(np.mean(vals)),
            "bootstrap_ci95_lower": ci_lo,
            "bootstrap_ci95_upper": ci_hi,
            "mean_recovery_delta": float(np.mean(recov)) if len(recov) else float("nan"),
            "recovery_delta_ci95_lower": rec_ci_lo,
            "recovery_delta_ci95_upper": rec_ci_hi,
        })

    return pd.DataFrame(rows).sort_values(["frequency", "magnitude", "method"]).reset_index(drop=True)


def summarize_half_life(event_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Adaptation half-life per (frequency, magnitude, method): mean
    episodes-to-50%-recovery over shift events/seeds where recovery was
    reached, the fraction of events that reached it, and "Undefined"
    when none ever did.
    """
    rows = []
    for (frequency, magnitude, method), sub in event_metrics_df.groupby(["frequency", "magnitude", "method"]):
        half_lives = sub["half_life"]
        n_total = len(half_lives)
        defined = half_lives.dropna()
        n_defined = len(defined)

        if n_defined == 0:
            mean_half_life = "Undefined"
        else:
            mean_half_life = float(defined.mean())

        rows.append({
            "frequency": frequency,
            "magnitude": magnitude,
            "method": method,
            "method_label": _METHOD_LABELS[method],
            "n_events": n_total,
            "n_recovered": n_defined,
            "fraction_recovered": n_defined / n_total if n_total else float("nan"),
            "mean_half_life_episodes": mean_half_life,
        })

    return pd.DataFrame(rows).sort_values(["frequency", "magnitude", "method"]).reset_index(drop=True)


def plot_heatmaps(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """One normalized-score heatmap per method: x=frequency, y=magnitude."""
    methods = list(_METHOD_LABELS.keys())
    frequencies = sorted(summary_df["frequency"].unique())
    magnitudes = sorted(summary_df["magnitude"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, method in zip(axes, methods):
        pivot = summary_df[summary_df["method"] == method].pivot(
            index="magnitude", columns="frequency", values="mean_normalized_score",
        ).reindex(index=magnitudes, columns=frequencies)

        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(frequencies)))
        ax.set_xticklabels([f"every {f}" for f in frequencies])
        ax.set_yticks(range(len(magnitudes)))
        ax.set_yticklabels([f"{m}/7" for m in magnitudes])
        ax.set_xlabel("Shift frequency", fontsize=10)
        ax.set_ylabel("Shift magnitude", fontsize=10)
        ax.set_title(_METHOD_LABELS[method], fontsize=11, color=_METHOD_COLORS[method])

        for i in range(len(magnitudes)):
            for j in range(len(frequencies)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="#111827")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Normalized Score by Shift Frequency x Magnitude", fontsize=13)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "shift_frequency_magnitude_heatmap.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Heatmap figure saved to {fig_path}")


def plot_vs_frequency(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """x=shift frequency, y=normalized score (averaged over magnitude),
    one line per method."""
    fig, ax = plt.subplots(figsize=(9, 6))
    frequencies = sorted(summary_df["frequency"].unique())

    for method in _METHOD_LABELS:
        sub = summary_df[summary_df["method"] == method]
        means = [sub[sub["frequency"] == f]["mean_normalized_score"].mean() for f in frequencies]
        ax.plot(frequencies, means, marker="o", linewidth=2, markersize=7,
                 color=_METHOD_COLORS[method], label=_METHOD_LABELS[method])

    ax.set_xlabel("Shift frequency (episodes between shifts)", fontsize=12)
    ax.set_ylabel("Normalized score (mean over magnitudes)", fontsize=12)
    ax.set_title("Normalized Score vs. Shift Frequency", fontsize=13)
    ax.invert_xaxis()
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "normalized_score_vs_frequency.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Frequency line plot saved to {fig_path}")


def plot_vs_magnitude(summary_df: pd.DataFrame, save_dir: Path) -> None:
    """x=shift magnitude, y=normalized score (averaged over frequency),
    one line per method."""
    fig, ax = plt.subplots(figsize=(9, 6))
    magnitudes = sorted(summary_df["magnitude"].unique())

    for method in _METHOD_LABELS:
        sub = summary_df[summary_df["method"] == method]
        means = [sub[sub["magnitude"] == m]["mean_normalized_score"].mean() for m in magnitudes]
        ax.plot(magnitudes, means, marker="o", linewidth=2, markersize=7,
                 color=_METHOD_COLORS[method], label=_METHOD_LABELS[method])

    ax.set_xlabel("Shift magnitude (failure types affected out of 7)", fontsize=12)
    ax.set_ylabel("Normalized score (mean over frequencies)", fontsize=12)
    ax.set_title("Normalized Score vs. Shift Magnitude", fontsize=13)
    ax.set_xticks(magnitudes)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "normalized_score_vs_magnitude.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Magnitude line plot saved to {fig_path}")


def run_phase6(
    frequencies: Optional[list[int]] = None,
    magnitudes: Optional[list[int]] = None,
    num_episodes: int = 1000,
    n_seeds: int = 20,
    gap: float = 0.2,
    base_seed: int = 2026,
    results_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full Phase 6 shift frequency x magnitude sweep, plot the
    three required figures, and save data to results/phase6/."""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase6"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df, event_metrics_df = run_shift_frequency_magnitude_sweep(
        frequencies=frequencies, magnitudes=magnitudes, num_episodes=num_episodes,
        n_seeds=n_seeds, gap=gap, base_seed=base_seed,
    )
    results_csv = results_dir / "shift_frequency_magnitude_results.csv"
    results_df.to_csv(str(results_csv), index=False)
    print(f"\n  Raw results saved to {results_csv}")

    summary_df = summarize_results(results_df, event_metrics_df)
    summary_csv = results_dir / "shift_frequency_magnitude_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"  Summary (normalized score + recovery delta, 95% CI) saved to {summary_csv}")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))

    half_life_df = summarize_half_life(event_metrics_df)
    half_life_csv = results_dir / "adaptation_half_life.csv"
    half_life_df.to_csv(str(half_life_csv), index=False)
    print(f"\n  Adaptation half-life saved to {half_life_csv}")
    print(half_life_df.to_string(index=False, float_format="{:.4f}".format))

    plot_heatmaps(summary_df, results_dir)
    plot_vs_frequency(summary_df, results_dir)
    plot_vs_magnitude(summary_df, results_dir)

    print(f"\n  NOTE: {MC_VARIATION_NOTE}")

    return results_df, summary_df, half_life_df
