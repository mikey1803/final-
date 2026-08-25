"""Phase 3A: robustness of the algorithm comparison across a *family* of
randomized reward matrices, not just the single hand-authored SUCCESS_MATRIX.

Reuses FailureType/RecoveryStrategy purely as a convenient 7-slot /
6-slot index set for BanditSelector's generic per-(type, strategy) table
— the bandit classes don't care what the labels mean, only that the
table shape matches. Outcomes are sampled directly from a supplied
matrix rather than through ``strategies.execute_strategy`` (which is
hardwired to the module-level SUCCESS_MATRIX), keeping this experiment
fully additive and decoupled from the taxonomy-specific baselines.
"""

import random
import zlib
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.stats_utils import MC_VARIATION_NOTE, bootstrap_ci
from src.failures.taxonomy import FailureType
from src.recovery.selector import BanditSelector, SelectorFactory
from src.recovery.strategies import RecoveryStrategy, SUCCESS_MATRIX

_TYPES: list[FailureType] = list(FailureType)
_STRATS: list[RecoveryStrategy] = list(RecoveryStrategy)
N_TYPES = len(_TYPES)   # 7
N_STRATS = len(_STRATS)  # 6

_ALGO_STYLES: dict[str, dict] = {
    "thompson_sampling": {"label": "Adaptive (Thompson)", "color": "#16a34a", "marker": "P"},
    "ucb1":               {"label": "Adaptive (UCB1)",     "color": "#2563eb", "marker": "o"},
}


def sample_matrix(
    n_types: int = N_TYPES,
    n_strats: int = N_STRATS,
    gap: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a random (n_types, n_strats) success-probability matrix with
    a controlled optimal-arm gap.

    Every row gets a uniform-random baseline in [0.15, 0.55]; one
    randomly chosen strategy per row is then boosted to
    ``row.max() + gap`` (capped at 0.95), guaranteeing a unique best arm
    separated from the row's prior best by approximately ``gap``.
    """
    if rng is None:
        rng = np.random.default_rng()
    base = rng.uniform(0.15, 0.55, size=(n_types, n_strats))
    for f in range(n_types):
        best = rng.integers(n_strats)
        base[f, best] = min(0.95, base[f].max() + gap)
    return base


def hand_authored_as_matrix() -> np.ndarray:
    """Render the project's hand-authored SUCCESS_MATRIX in the same
    (n_types, n_strats) layout, so it scores as one instance of the
    family rather than a privileged special case."""
    return np.array([[SUCCESS_MATRIX[ft][rs] for rs in _STRATS] for ft in _TYPES])


def analytic_oracle_rate(matrix: np.ndarray) -> float:
    """Exact expected success rate of an oracle always playing
    argmax_s matrix[f, s], under uniform failure-type sampling."""
    return float(matrix.max(axis=1).mean())


def analytic_random_rate(matrix: np.ndarray) -> float:
    """Exact expected success rate of uniform-random strategy selection,
    under uniform failure-type sampling."""
    return float(matrix.mean())


def simulate_selector_on_matrix(
    matrix: np.ndarray,
    selector,
    num_episodes: int,
    seed: int,
    type_labels: Optional[list] = None,
    strategy_labels: Optional[list] = None,
) -> float:
    """Run one trial of an already-constructed selector against *matrix*;
    return its late-stage (last 20% of episodes) success rate.

    Generic over any selector exposing ``select_strategy(failure_type)``
    and ``update(failure_type, strategy, success=..., cost=..., episode_number=...)``
    — bandit selectors, ``LinUCBBaseline``, or a fixed/static policy that
    ignores the update call entirely.

    ``type_labels``/``strategy_labels`` optionally override the row/column
    index labels (default: the real ``FailureType``/``RecoveryStrategy``
    lists, requiring ``matrix.shape == (7, 6)``) — used by the
    taxonomy-size stress test to run this same loop against matrices with
    more than 7 rows.
    """
    types = type_labels if type_labels is not None else _TYPES
    strats = strategy_labels if strategy_labels is not None else _STRATS
    type_rng = random.Random(seed + 13)
    outcome_rng = random.Random(seed + 29)
    late_start = int(num_episodes * 0.8)

    successes: list[bool] = []
    for ep in range(num_episodes):
        f_idx = type_rng.randrange(matrix.shape[0])
        ft = types[f_idx]
        strategy = selector.select_strategy(ft)
        s_idx = strats.index(strategy)
        success = outcome_rng.random() < matrix[f_idx, s_idx]
        selector.update(ft, strategy, success=success, cost=1.0, episode_number=ep + 1)
        successes.append(success)

    late = successes[late_start:]
    return float(np.mean(late)) if late else float("nan")


def _simulate_late_stage_rate(
    matrix: np.ndarray,
    algorithm: str,
    num_episodes: int,
    seed: int,
) -> float:
    """Run one bandit trial against *matrix*; return its late-stage
    (last 20% of episodes) success rate."""
    selector: BanditSelector = SelectorFactory.create(algorithm, seed=seed)
    return simulate_selector_on_matrix(matrix, selector, num_episodes, seed)


def run_matrix_family_sweep(
    gaps: Optional[list[float]] = None,
    n_matrices: int = 100,
    n_seeds: int = 20,
    algorithms: Optional[list[str]] = None,
    num_episodes: int = 150,
    base_seed: int = 2026,
) -> pd.DataFrame:
    """Run the algorithm comparison over many sampled reward matrices per
    gap value, plus the hand-authored SUCCESS_MATRIX as one extra
    (labelled) instance for reference — not the headline result.

    Returns a long-format DataFrame: one row per
    (gap, matrix_idx, seed, algorithm) with the late-stage success rate
    and its Oracle/UniformRandom-normalized score (both computed
    analytically and exactly per matrix, since Oracle/UniformRandom's
    expected rates have closed forms — see ``analytic_oracle_rate`` /
    ``analytic_random_rate``).
    """
    if gaps is None:
        gaps = [0.05, 0.10, 0.20, 0.30]
    if algorithms is None:
        algorithms = list(_ALGO_STYLES.keys())

    rows: list[dict] = []

    # --- the hand-authored matrix, scored once as a labelled reference point ---
    hand_matrix = hand_authored_as_matrix()
    hand_oracle = analytic_oracle_rate(hand_matrix)
    hand_random = analytic_random_rate(hand_matrix)
    hand_denom = hand_oracle - hand_random
    for algo in algorithms:
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            rate = _simulate_late_stage_rate(hand_matrix, algo, num_episodes, seed)
            normalized = (rate - hand_random) / hand_denom if hand_denom != 0 else float("nan")
            rows.append({
                "gap": float("nan"), "matrix_idx": -1, "seed": seed_idx, "algorithm": algo,
                "late_stage_success_rate": rate, "normalized_score": normalized,
                "source": "hand_authored",
            })

    print(f"  Hand-authored SUCCESS_MATRIX: oracle_rate={hand_oracle:.4f}  random_rate={hand_random:.4f}")
    print(f"  Sweeping gaps={gaps}  n_matrices={n_matrices}  n_seeds={n_seeds}  algorithms={algorithms}")
    print(f"  {num_episodes} episodes per (matrix, seed, algorithm) run\n")

    matrix_rng = np.random.default_rng(base_seed)

    for gap in gaps:
        gap_rates: dict[str, list[float]] = {a: [] for a in algorithms}
        for matrix_idx in range(n_matrices):
            matrix = sample_matrix(gap=gap, rng=matrix_rng)
            oracle_rate = analytic_oracle_rate(matrix)
            random_rate = analytic_random_rate(matrix)
            denom = oracle_rate - random_rate

            for algo in algorithms:
                for seed_idx in range(n_seeds):
                    # Task 13 fix: Python's built-in hash() is randomized
                    # per-process (PYTHONHASHSEED) for str inputs, so the
                    # old `hash(algo) % 997` term silently made every
                    # re-run of this sweep non-deterministic across
                    # process invocations, despite an identical seed/RNG
                    # discipline everywhere else -- confirmed by direct
                    # reproduction (two fresh processes, same n_seeds,
                    # same base_seed, different results). zlib.crc32 is a
                    # deterministic, well-distributed replacement with the
                    # same purpose (a stable per-algorithm-name offset).
                    seed = base_seed + 100_000 * matrix_idx + 1_000 * seed_idx + zlib.crc32(algo.encode()) % 997
                    rate = _simulate_late_stage_rate(matrix, algo, num_episodes, seed)
                    normalized = (rate - random_rate) / denom if denom != 0 else float("nan")
                    gap_rates[algo].append(normalized)
                    rows.append({
                        "gap": gap, "matrix_idx": matrix_idx, "seed": seed_idx, "algorithm": algo,
                        "late_stage_success_rate": rate, "normalized_score": normalized,
                        "source": "sampled",
                    })

        summary = "  ".join(
            f"{_ALGO_STYLES.get(a, {}).get('label', a)}={np.mean(gap_rates[a]):+.3f}" for a in algorithms
        )
        print(f"  gap={gap:.2f}  n={n_matrices * n_seeds} per algorithm  |  {summary}")

    return pd.DataFrame(rows)


def plot_matrix_family_sweep(sweep_df: pd.DataFrame, save_dir: Path) -> None:
    """Plot normalized score vs. gap with bootstrap 95% CIs, one line per
    algorithm. The hand-authored matrix's per-algorithm normalized score
    is overlaid as a horizontal reference (not part of the sweep lines)."""
    sampled = sweep_df[sweep_df["source"] == "sampled"]
    hand = sweep_df[sweep_df["source"] == "hand_authored"]

    fig, ax = plt.subplots(figsize=(9, 6))
    gaps = sorted(sampled["gap"].unique())

    for algo, style in _ALGO_STYLES.items():
        sub = sampled[sampled["algorithm"] == algo]
        if sub.empty:
            continue
        means, lo, hi = [], [], []
        for gap in gaps:
            vals = sub[sub["gap"] == gap]["normalized_score"].values
            means.append(float(np.mean(vals)))
            ci_lo, ci_hi = bootstrap_ci(vals)
            lo.append(ci_lo)
            hi.append(ci_hi)
        means, lo, hi = np.array(means), np.array(lo), np.array(hi)

        ax.plot(gaps, means, label=style["label"], color=style["color"],
                marker=style["marker"], linewidth=2, markersize=7)
        ax.fill_between(gaps, lo, hi, color=style["color"], alpha=0.18)

        # hand-authored reference point for this algorithm
        hand_sub = hand[hand["algorithm"] == algo]["normalized_score"]
        if not hand_sub.empty:
            ax.scatter(
                [gaps[-1] * 1.0], [float(hand_sub.mean())],
                color=style["color"], marker="*", s=180, edgecolor="black", zorder=5,
                label=f"{style['label']} (hand-authored matrix)",
            )

    ax.axhline(0, color="#6b7280", linestyle=":", linewidth=1, alpha=0.6, label="UniformRandom floor")
    ax.axhline(1, color="#6b7280", linestyle="--", linewidth=1, alpha=0.6, label="Oracle ceiling")
    ax.set_xlabel("Optimal-arm gap", fontsize=12)
    ax.set_ylabel("Normalized score (bootstrap 95% CI)", fontsize=12)
    ax.set_title("Algorithm Comparison Across Randomized Reward-Matrix Family", fontsize=13)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "matrix_family_sweep.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Matrix-family sweep figure saved to {fig_path}")


def run_phase3a(
    gaps: Optional[list[float]] = None,
    n_matrices: int = 100,
    n_seeds: int = 20,
    algorithms: Optional[list[str]] = None,
    num_episodes: int = 150,
    base_seed: int = 2026,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full Phase 3A sweep, plot the figure, and save data/notes
    to results/phase3a/."""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase3a"
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = run_matrix_family_sweep(
        gaps=gaps, n_matrices=n_matrices, n_seeds=n_seeds,
        algorithms=algorithms, num_episodes=num_episodes, base_seed=base_seed,
    )

    csv_path = results_dir / "matrix_family_results.csv"
    sweep_df.to_csv(str(csv_path), index=False)
    print(f"\n  Matrix-family sweep data saved to {csv_path}")

    plot_matrix_family_sweep(sweep_df, results_dir)

    # Aggregate summary table with bootstrap CIs (what the figure plots, in table form)
    sampled = sweep_df[sweep_df["source"] == "sampled"]
    summary_rows = []
    for gap in sorted(sampled["gap"].unique()):
        for algo, style in _ALGO_STYLES.items():
            vals = sampled[(sampled["gap"] == gap) & (sampled["algorithm"] == algo)]["normalized_score"].values
            if len(vals) == 0:
                continue
            ci_lo, ci_hi = bootstrap_ci(vals)
            summary_rows.append({
                "gap": gap, "algorithm": style["label"],
                "n": len(vals), "mean_normalized_score": float(np.mean(vals)),
                "bootstrap_ci95_lower": ci_lo, "bootstrap_ci95_upper": ci_hi,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "matrix_family_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"  Matrix-family summary (with bootstrap CIs) saved to {summary_csv}")

    notes_path = results_dir / "NOTES.txt"
    notes_path.write_text(
        "Phase 3A: algorithm comparison across a family of randomized reward "
        f"matrices (n_matrices={n_matrices}, n_seeds={n_seeds}, gaps={gaps or [0.05, 0.10, 0.20, 0.30]}), "
        "in addition to the single hand-authored SUCCESS_MATRIX (plotted as a "
        "reference point, not the headline result).\n\n"
        f"{MC_VARIATION_NOTE}\n",
        encoding="utf-8",
    )
    print(f"  Notes saved to {notes_path}")

    return sweep_df
