"""Phase 5: static-rule quality sweep.

How does a hand-coded, non-learning rule table perform as a function of
how many of its 7 type->strategy mappings happen to be correct? Each
static policy is a fixed ``FailureType -> RecoveryStrategy`` table with
no learning (``update`` is a no-op) — the static-rule analogue of
``FixedRuleBaseline``, but with a *controlled* number of correct
mappings instead of one fixed, hand-authored table.

Policies are blind by construction: a policy is built by picking, for a
freshly sampled reward matrix, which ``num_correct`` of its 7 rows get
the true best strategy (uniformly at random) and assigning a uniformly
random *incorrect* strategy to the rest — the reward magnitudes are
never consulted, only which column is the argmax. This mirrors how
``FixedRuleBaseline``'s rule table was designed without reading
SUCCESS_MATRIX and only scored for correctness afterward (see
``baselines.py``), generalized so every quality level 3/7-7/7 has many
independent instances instead of one fixed 6/7 table.

Because each policy is tied to its own freshly sampled matrix (reusing
``sample_matrix`` from ``matrix_family.py``), a 7/7-correct policy is
not unique across draws — different matrices have different best-arm
structure — so at every quality level many distinct policies exist.

Reuses ``matrix_family.py``'s matrix generation (``sample_matrix``),
per-episode simulation (``simulate_selector_on_matrix``), analytic
normalization (``analytic_oracle_rate`` / ``analytic_random_rate``),
and gaps/seeds/episode-count conventions, plus ``run_matrix_family_sweep``
itself for the Thompson / Discounted Thompson / LinUCB reference lines.
No evaluation logic is duplicated.
"""

import random
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.baselines import LinUCBBaseline
from src.evaluation.matrix_family import (
    _STRATS,
    _TYPES,
    analytic_oracle_rate,
    analytic_random_rate,
    run_matrix_family_sweep,
    sample_matrix,
    simulate_selector_on_matrix,
)
from src.evaluation.stats_utils import MC_VARIATION_NOTE, bootstrap_ci
from src.failures.taxonomy import FailureType
from src.recovery.selector import SelectorFactory
from src.recovery.strategies import RecoveryStrategy

N_TYPES = len(_TYPES)  # 7

# LinUCBBaseline isn't in SelectorFactory's default registry (it's a
# baseline class, not a BanditSelector) — register it so it can be
# selected by name through the same factory the rest of the pipeline uses.
SelectorFactory.register("linucb", LinUCBBaseline)

_REFERENCE_ALGORITHMS: dict[str, dict] = {
    "thompson_sampling":            {"label": "Thompson Sampling",            "color": "#16a34a", "linestyle": "--"},
    "discounted_thompson_sampling": {"label": "Discounted Thompson Sampling", "color": "#7c3aed", "linestyle": "-."},
    "linucb":                       {"label": "LinUCB",                       "color": "#0d9488", "linestyle": ":"},
}


class StaticPolicySelector:
    """A fixed, non-learning ``FailureType -> RecoveryStrategy`` table.

    ``update`` is a no-op: the policy never changes regardless of
    observed outcomes, matching ``FixedRuleBaseline``'s behaviour.
    """

    def __init__(self, policy: dict[FailureType, RecoveryStrategy]) -> None:
        self._policy = policy

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        return self._policy[failure_type]

    def update(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool = True,
        cost: float = 1.0,
        context: Optional[dict] = None,
        episode_number: int = 0,
    ) -> None:
        pass


def generate_static_policy(
    matrix: np.ndarray,
    num_correct: int,
    rng: random.Random,
) -> dict[FailureType, RecoveryStrategy]:
    """Build a blind static policy with exactly *num_correct* of the
    matrix's rows mapped to their true best (argmax) strategy.

    Which rows are correct is chosen uniformly at random; every
    incorrect row is assigned a uniformly random *non*-best strategy.
    Reward magnitudes are never consulted beyond locating the argmax.
    """
    n_types, n_strats = matrix.shape
    best_idx = matrix.argmax(axis=1)
    correct_rows = set(rng.sample(range(n_types), num_correct))

    policy: dict[FailureType, RecoveryStrategy] = {}
    for t_idx, ft in enumerate(_TYPES):
        best_s = int(best_idx[t_idx])
        if t_idx in correct_rows:
            policy[ft] = _STRATS[best_s]
        else:
            wrong_choices = [s for s in range(n_strats) if s != best_s]
            policy[ft] = _STRATS[rng.choice(wrong_choices)]
    return policy


def _policy_signature(policy: dict[FailureType, RecoveryStrategy]) -> tuple:
    """Hashable fingerprint of a policy, for uniqueness checks."""
    return tuple(policy[ft].value for ft in _TYPES)


def run_static_rule_quality_sweep(
    quality_levels: Optional[list[int]] = None,
    n_policies: int = 25,
    gaps: Optional[list[float]] = None,
    n_seeds: int = 20,
    num_episodes: int = 150,
    base_seed: int = 2026,
) -> pd.DataFrame:
    """Sweep static-rule quality (3/7 .. 7/7 correct) over many blind,
    independently sampled policies, evaluated via the existing
    randomized matrix-family pipeline.

    For each quality level, ``n_policies`` (>= 20, enforced below) blind
    policies are generated, each tied to its own freshly sampled reward
    matrix (cycling through *gaps*, mirroring ``run_matrix_family_sweep``).
    Every policy is then replayed over ``n_seeds`` independent outcome
    sequences of ``num_episodes`` episodes (``simulate_selector_on_matrix``),
    and its late-stage success rate is normalized against that same
    matrix's own analytic Oracle/UniformRandom rates.

    Returns a long-format DataFrame: one row per
    (quality, policy_idx, gap, seed) with the late-stage success rate
    and its normalized score.
    """
    if quality_levels is None:
        quality_levels = [7, 6, 5, 4, 3]
    if gaps is None:
        gaps = [0.05, 0.10, 0.20, 0.30]
    if n_policies < 20:
        raise ValueError("n_policies must be >= 20 (need >=20 unique policies per quality level)")

    rows: list[dict] = []

    print(f"  Static-rule quality sweep: quality_levels={quality_levels}  n_policies={n_policies}")
    print(f"  gaps={gaps}  n_seeds={n_seeds}  num_episodes={num_episodes}\n")

    for q in quality_levels:
        signatures: set[tuple] = set()

        for policy_idx in range(n_policies):
            gap = gaps[policy_idx % len(gaps)]
            matrix_rng = np.random.default_rng(base_seed + q * 100_000 + policy_idx * 97)
            matrix = sample_matrix(gap=gap, rng=matrix_rng)

            policy_rng = random.Random(base_seed + q * 500_000 + policy_idx * 31)
            policy = generate_static_policy(matrix, num_correct=q, rng=policy_rng)
            signatures.add(_policy_signature(policy))

            oracle_rate = analytic_oracle_rate(matrix)
            random_rate = analytic_random_rate(matrix)
            denom = oracle_rate - random_rate

            for seed_idx in range(n_seeds):
                sim_seed = base_seed + q * 1_000_000 + policy_idx * 1_000 + seed_idx
                selector = StaticPolicySelector(policy)
                rate = simulate_selector_on_matrix(matrix, selector, num_episodes, sim_seed)
                normalized = (rate - random_rate) / denom if denom != 0 else float("nan")
                rows.append({
                    "quality": q,
                    "policy_idx": policy_idx,
                    "gap": gap,
                    "seed": seed_idx,
                    "late_stage_success_rate": rate,
                    "normalized_score": normalized,
                })

        assert len(signatures) >= 20, (
            f"quality={q}/7: only {len(signatures)} unique policies generated, need >= 20"
        )
        print(f"  quality={q}/7  unique_policies={len(signatures)}/{n_policies}")

    return pd.DataFrame(rows)


def summarize_static_rule_quality_sweep(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Mean normalized score and 95% CI (bootstrap) per quality level."""
    rows = []
    for q in sorted(sweep_df["quality"].unique(), reverse=True):
        vals = sweep_df[sweep_df["quality"] == q]["normalized_score"].values
        ci_lo, ci_hi = bootstrap_ci(vals)
        rows.append({
            "quality": q,
            "n": len(vals),
            "mean_normalized_score": float(np.mean(vals)),
            "bootstrap_ci95_lower": ci_lo,
            "bootstrap_ci95_upper": ci_hi,
        })
    return pd.DataFrame(rows)


def run_reference_baselines(
    gaps: Optional[list[float]] = None,
    n_matrices: int = 100,
    n_seeds: int = 20,
    num_episodes: int = 150,
    base_seed: int = 2026,
) -> pd.DataFrame:
    """Mean normalized score for Thompson Sampling, Discounted Thompson
    Sampling, and LinUCB across the same randomized matrix-family sweep
    used for the static-rule policies (via ``run_matrix_family_sweep`` —
    no evaluation logic is reimplemented here).
    """
    print("\n  Reference baselines (Thompson / Discounted Thompson / LinUCB) "
          "over the randomized matrix family...")
    sweep_df = run_matrix_family_sweep(
        gaps=gaps,
        n_matrices=n_matrices,
        n_seeds=n_seeds,
        algorithms=list(_REFERENCE_ALGORITHMS.keys()),
        num_episodes=num_episodes,
        base_seed=base_seed,
    )
    sampled = sweep_df[sweep_df["source"] == "sampled"]

    rows = []
    for algo, style in _REFERENCE_ALGORITHMS.items():
        vals = sampled[sampled["algorithm"] == algo]["normalized_score"].values
        ci_lo, ci_hi = bootstrap_ci(vals)
        rows.append({
            "algorithm": algo,
            "label": style["label"],
            "n": len(vals),
            "mean_normalized_score": float(np.mean(vals)),
            "bootstrap_ci95_lower": ci_lo,
            "bootstrap_ci95_upper": ci_hi,
        })
    return pd.DataFrame(rows)


def plot_static_rule_quality(
    summary_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    save_dir: Path,
) -> None:
    """x-axis = static rule quality (3/7 .. 7/7), y-axis = normalized
    score, with horizontal reference lines for Thompson, Discounted
    Thompson, and LinUCB."""
    ordered = summary_df.sort_values("quality")
    qualities = ordered["quality"].values
    means = ordered["mean_normalized_score"].values
    lo = ordered["bootstrap_ci95_lower"].values
    hi = ordered["bootstrap_ci95_upper"].values
    yerr = np.vstack([means - lo, hi - means])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        qualities, means, yerr=yerr, fmt="o-", color="#f59e0b",
        linewidth=2, markersize=7, capsize=4, label="Static Rule (blind)",
    )

    for _, row in reference_df.iterrows():
        style = _REFERENCE_ALGORITHMS[row["algorithm"]]
        ax.axhline(
            row["mean_normalized_score"], color=style["color"],
            linestyle=style["linestyle"], linewidth=1.8, label=style["label"],
        )

    ax.set_xlabel("Static rule quality (correct mappings out of 7)", fontsize=12)
    ax.set_ylabel("Normalized score", fontsize=12)
    ax.set_title("Static-Rule Quality vs. Adaptive Baselines", fontsize=13)
    ax.set_xticks(sorted(qualities))
    ax.set_xticklabels([f"{int(q)}/7" for q in sorted(qualities)])
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "static_rule_quality_sweep.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Static-rule quality sweep figure saved to {fig_path}")


def run_phase5(
    quality_levels: Optional[list[int]] = None,
    n_policies: int = 25,
    gaps: Optional[list[float]] = None,
    n_seeds: int = 20,
    num_episodes: int = 150,
    base_seed: int = 2026,
    n_matrices_reference: int = 40,
    results_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Phase 5 static-rule quality sweep, plot the figure,
    and save data/notes to results/phase5/."""
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "phase5"
    results_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = run_static_rule_quality_sweep(
        quality_levels=quality_levels, n_policies=n_policies, gaps=gaps,
        n_seeds=n_seeds, num_episodes=num_episodes, base_seed=base_seed,
    )
    raw_csv = results_dir / "static_rule_quality_results.csv"
    sweep_df.to_csv(str(raw_csv), index=False)
    print(f"\n  Raw static-rule quality results saved to {raw_csv}")

    summary_df = summarize_static_rule_quality_sweep(sweep_df)
    summary_csv = results_dir / "static_rule_quality_summary.csv"
    summary_df.to_csv(str(summary_csv), index=False)
    print(f"  Static-rule quality summary (mean, 95% CI) saved to {summary_csv}")
    print(f"\n  {'='*80}")
    print("  Static-rule quality: mean normalized score (95% bootstrap CI)")
    print(f"  {'='*80}")
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))

    reference_df = run_reference_baselines(
        gaps=gaps, n_matrices=n_matrices_reference, n_seeds=n_seeds,
        num_episodes=num_episodes, base_seed=base_seed,
    )
    reference_csv = results_dir / "reference_baselines_summary.csv"
    reference_df.to_csv(str(reference_csv), index=False)
    print(f"\n  Reference baseline summary saved to {reference_csv}")
    print(f"\n  {'='*80}")
    print("  Reference baselines: mean normalized score (95% bootstrap CI)")
    print(f"  {'='*80}")
    print(reference_df.to_string(index=False, float_format="{:.4f}".format))

    plot_static_rule_quality(summary_df, reference_df, results_dir)

    notes_path = results_dir / "NOTES.txt"
    notes_path.write_text(
        "Phase 5: static-rule quality sweep. Blind static policies "
        "(FailureType -> RecoveryStrategy tables, no learning) generated with "
        f"exactly N/7 correct mappings for N in {quality_levels or [7, 6, 5, 4, 3]}, "
        f"{n_policies} unique policies per quality level, each tied to its own "
        "freshly sampled reward matrix from the same randomized matrix-family "
        "pipeline used in Phase 3A (matrix_family.py: sample_matrix, "
        "simulate_selector_on_matrix, analytic Oracle/UniformRandom normalization). "
        "Compared against Thompson Sampling, Discounted Thompson Sampling, and "
        "LinUCB evaluated via run_matrix_family_sweep over the same matrix family.\n\n"
        f"{MC_VARIATION_NOTE}\n",
        encoding="utf-8",
    )
    print(f"\n  Notes saved to {notes_path}")

    return sweep_df, summary_df
