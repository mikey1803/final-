"""Simulation test for the AdaptiveSelector.

Runs 500 episodes for both UCB1 and epsilon-greedy (same failure-type
sequence for fair comparison), snapshotting the learned-best strategy
every 25 episodes to plot a convergence curve.

Outputs:
  - data/convergence_curve.png  (line chart, 150 dpi)
  - Console: final checkpoint table + per-algorithm detail tables
"""

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.failures.taxonomy import FailureEvent, FailureType
from src.recovery.strategies import (
    SUCCESS_MATRIX,
    RecoveryStrategy,
    execute_strategy,
)
from src.recovery.selector import AdaptiveSelector


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

TRUE_OPTIMAL: dict[FailureType, RecoveryStrategy] = {
    ft: max(rates, key=rates.get)  # type: ignore[arg-type]
    for ft, rates in SUCCESS_MATRIX.items()
}


def _count_matches(selector: AdaptiveSelector) -> int:
    """Count how many failure types have their best predicted strategy match the true optimal."""
    best = selector.get_best_strategies()
    return sum(1 for ft in FailureType if best[ft] == TRUE_OPTIMAL[ft])


def _run_with_tracking(
    n_episodes: int,
    algorithm: str,
    failure_types: list[FailureType],
    snapshot_interval: int = 25,
    base_seed: int = 42,
) -> tuple[AdaptiveSelector, list[int], list[int]]:
    """Run a simulation, recording correct-match counts at regular checkpoints.

    Returns:
        (final_selector, checkpoint_episodes, checkpoint_matches)
    """
    selector = AdaptiveSelector(algorithm=algorithm, seed=base_seed)
    # Use a deterministic per-algorithm offset — hash() is randomised in
    # Python 3 so we must not rely on it for reproducibility.
    algo_offset = 0 if algorithm == "ucb1" else 1000
    rng = random.Random(base_seed + algo_offset)

    episodes: list[int] = []
    matches: list[int] = []

    for i, ft in enumerate(failure_types, start=1):
        strategy = selector.select_strategy(ft)
        event = FailureEvent(failure_type=ft, raw_error="simulated")
        ctx = {"tool_name": "web_search", "step_index": i, "goal": "test"}
        outcome = execute_strategy(strategy, event, ctx, rng=rng)
        selector.update(ft, strategy, success=outcome["success"])

        if i % snapshot_interval == 0 or i == n_episodes:
            episodes.append(i)
            matches.append(_count_matches(selector))

    return selector, episodes, matches


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    n_episodes = 500
    snapshot_interval = 25
    base_seed = 42

    # Generate the shared failure-type sequence once so both algorithms
    # face identical problem instances.
    type_rng = random.Random(base_seed)
    failure_types = [type_rng.choice(list(FailureType)) for _ in range(n_episodes)]

    # --- run both algorithms ---
    ucb1_sel, ucb1_eps, ucb1_matches = _run_with_tracking(
        n_episodes, "ucb1", failure_types, snapshot_interval, base_seed,
    )
    eps_sel, eps_eps, eps_matches = _run_with_tracking(
        n_episodes, "epsilon_greedy", failure_types, snapshot_interval, base_seed,
    )

    # --- convergence curve plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ucb1_eps, ucb1_matches, marker="o", markersize=4, linewidth=2,
            label="UCB1", color="#2563eb")
    ax.plot(eps_eps, eps_matches, marker="s", markersize=4, linewidth=2,
            label="Epsilon-greedy (eps=0.15)", color="#dc2626", linestyle="--")
    ax.axhline(y=7, color="gray", linestyle=":", alpha=0.5, label="Perfect (7/7)")
    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel("Correct matches (out of 7)", fontsize=12)
    ax.set_title("Convergence: Recovery-Strategy Selector", fontsize=13)
    ax.set_ylim(-0.3, 7.5)
    ax.set_yticks(range(8))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / "convergence_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Convergence plot saved to {out_dir / 'convergence_curve.png'}\n")

    # --- final checkpoint table ---
    print(f"  {'='*60}")
    print(f"  Final checkpoint (episode {n_episodes})")
    print(f"  {'='*60}")
    print(f"  {'Algorithm':24s}  {'Correct':>8s}  {'Accuracy':>10s}")
    print(f"  {'-'*48}")
    print(f"  {'UCB1':24s}  {ucb1_matches[-1]:>8d}  {ucb1_matches[-1]/7:>10.2%}")
    print(f"  {'Epsilon-greedy (eps=0.15)':24s}  {eps_matches[-1]:>8d}  {eps_matches[-1]/7:>10.2%}")
    print()

    # --- full checkpoint history ---
    print(f"  {'Episode':>8s}  {'UCB1':>6s}  {'Eps-greedy':>10s}")
    print(f"  {'-'*30}")
    for ep, m1, m2 in zip(ucb1_eps, ucb1_matches, eps_matches):
        print(f"  {ep:>8d}  {m1:>6d}  {m2:>10d}")
    print()

    # --- per-algorithm detail tables ---
    for label, sel in [("UCB1", ucb1_sel), ("EPSILON_GREEDY", eps_sel)]:
        print(f"\n{'='*70}")
        print(f"  {label} — Final learned success rates")
        print(f"{'='*70}")
        stats = sel.get_stats_table()
        print(stats.to_string(float_format="{:.2f}".format))

        best = sel.get_best_strategies()
        print(f"\n  {'FailureType':30s}  {'Learned':20s}  {'Optimal':20s}  Match")
        print(f"  {'-'*80}")
        for ft in FailureType:
            learned = best[ft]
            optimal = TRUE_OPTIMAL[ft]
            print(
                f"  {ft.value:30s}  {learned.value:20s}  {optimal.value:20s}  "
                f"{'YES' if learned == optimal else 'no'}"
            )
        print()


if __name__ == "__main__":
    main()
