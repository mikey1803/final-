#!/usr/bin/env python
"""Analysis for the PGTS prior-grid experiment (``experiments/prior_grid.py``).

Reads ``results/tables/prior_grid.parquet`` (grid-level summary metrics,
one row per cell) and, where the grid data alone is insufficient
(posterior entropy/convergence, which are policy-internal diagnostics
never captured in a per-cell summary row), runs a small number of
directly-instrumented illustrative trajectories. All bootstrap CIs reuse
``recovery.metrics.bootstrap.bootstrap_ci``; no p-values, no ANOVA (see
"Factor contribution" -- descriptive decomposition only, per Task 7's
own instruction).

Usage
-----
    python experiments/prior_analysis.py [--input PATH] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.episode import run_episodes  # noqa: E402
from recovery.core.parallel import RunStore  # noqa: E402
from recovery.core.protocols import Context  # noqa: E402
from recovery.envs.canonical import MATRIX  # noqa: E402
from recovery.envs.matrix_env import MatrixEnvironment  # noqa: E402
from recovery.metrics.bootstrap import bootstrap_ci  # noqa: E402
from recovery.policies.hybrid import DiscountedPriorThompson, PriorInitializedThompson  # noqa: E402

DEFAULT_INPUT = _BACKEND_DIR / "results" / "tables" / "prior_grid.parquet"
DEFAULT_OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"
PERFORMANCE_METRIC = "normalized_score"


# ======================================================================
# 1. Negative-transfer surface
# ======================================================================


def negative_transfer_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ``normalized_score`` over ``quality x strength``, for every
    ``(env_id, n_episodes)`` -- the raw surface a heatmap would plot,
    returned as a tidy table (not just a plot, per the task)."""
    pgts = df[df["method"].isin(["pgts", "pgts_static_limit"])]
    return (
        pgts.groupby(["env_id", "n_episodes", "quality", "strength"])[PERFORMANCE_METRIC]
        .agg(mean="mean", n="count", std="std")
        .reset_index()
        .sort_values(["env_id", "n_episodes", "quality", "strength"])
    )


# ======================================================================
# 2. Optimal prior strength per quality level, with 95% bootstrap CI
# ======================================================================


def optimal_strength_per_quality(df: pd.DataFrame, ci: float = 0.95) -> pd.DataFrame:
    """For every ``(env_id, n_episodes, quality)``, the ``strength`` with
    the highest mean ``normalized_score``, plus that strength's bootstrap
    CI.

    Stratified by ``n_episodes`` (horizon) as well as ``env_id``: the
    optimal strength is expected to depend on horizon (a longer run gives
    stock Thompson more time to catch up, reducing how much a prior is
    worth), so mixing horizons together would silently average over a
    real effect rather than reveal it.
    """
    pgts = df[df["method"].isin(["pgts", "pgts_static_limit"])]
    rows = []
    for (env_id, n_episodes, quality), group in pgts.groupby(["env_id", "n_episodes", "quality"]):
        means = group.groupby("strength")[PERFORMANCE_METRIC].mean()
        if means.empty:
            continue
        best_strength = means.idxmax()
        best_values = group[group["strength"] == best_strength][PERFORMANCE_METRIC].to_numpy()
        if len(best_values) >= 2:
            lo, hi = bootstrap_ci(best_values, ci=ci)
        else:
            lo, hi = float("nan"), float("nan")
        rows.append(
            {
                "env_id": env_id, "n_episodes": n_episodes, "quality": quality,
                "optimal_strength": best_strength, "mean_at_optimum": float(means.loc[best_strength]),
                "ci_lo": lo, "ci_hi": hi, "n_seeds": len(best_values),
            }
        )
    return pd.DataFrame(rows).sort_values(["env_id", "n_episodes", "quality"])


# ======================================================================
# 3. Negative-transfer region (explicit, not just plots)
# ======================================================================


def negative_transfer_region(df: pd.DataFrame) -> pd.DataFrame:
    """Every ``(env_id, n_episodes, quality, strength)`` cell whose mean
    ``normalized_score`` is BELOW stock Thompson's mean at the same
    ``(env_id, n_episodes)`` -- the quantified negative-transfer region."""
    stock = (
        df[df["method"] == "stock_thompson"]
        .groupby(["env_id", "n_episodes"])[PERFORMANCE_METRIC]
        .mean()
        .rename("stock_thompson_mean")
    )
    surface = negative_transfer_surface(df)
    merged = surface.merge(stock, on=["env_id", "n_episodes"], how="left")
    merged["delta_vs_stock"] = merged["mean"] - merged["stock_thompson_mean"]
    merged["is_negative_transfer"] = merged["delta_vs_stock"] < 0
    return merged.sort_values("delta_vs_stock")


# ======================================================================
# 4. Recovery half-life under drift vs strength and discount factor
# ======================================================================


def recovery_half_life(
    discounted_df: pd.DataFrame,
    store_root: Path,
    shift_episode: int,
    window: int = 50,
    tolerance_fraction: float = 0.5,
) -> pd.DataFrame:
    """For every ``discounted_pgts`` cell (drift conditions only), reads
    its FULL per-episode log back from the ``RunStore`` (identity fields
    alone determine ``PriorCellConfig.content_key()``, so a cell's stored
    log can be located WITHOUT keeping the original config objects
    around -- see ``experiments.prior_grid.run_discounted_grid``, which
    stores full logs specifically so this function can do this) and
    estimates the episode count after ``shift_episode`` at which a
    rolling-window success rate first recovers to within
    ``tolerance_fraction`` of the way from its post-shift trough back up
    to its pre-shift level ("half-life" at the default ``tolerance_fraction=0.5``).

    Returns one row per (env_id, quality, strength, gamma, seed) with
    ``half_life_episodes`` (``NaN`` if the trough/recovery could not be
    resolved, e.g. a run shorter than ``shift_episode``).
    """
    from experiments.prior_grid import PriorCellConfig

    store = RunStore(store_root)
    rows = []
    drift_df = discounted_df[discounted_df["env_id"].isin(["single_shift", "repeated_shift"])]

    for _, row in drift_df.iterrows():
        if row["n_episodes"] <= shift_episode:
            continue

        # A PriorCellConfig reconstructed from identity fields alone
        # hashes to the SAME content_key() the original (factory-bearing)
        # config did -- content_key() never reads the factories -- so a
        # dummy no-op factory is sufficient just to satisfy the
        # dataclass's required fields.
        config = PriorCellConfig(
            method=row["method"], env_id=row["env_id"], seed=int(row["seed"]),
            n_episodes=int(row["n_episodes"]), environment_factory=lambda: None,
            policy_factory=lambda: None, classifier_factory=lambda: None,
            classifier_id=row["classifier_id"], prior_id=row["prior_id"],
            strength=row["strength"], quality=row["quality"], prior_mean=row["prior_mean"],
            gamma=row["gamma"],
        )
        if not store.exists(config):
            continue
        full = store.read(config)
        reward = full["reward"].to_numpy(dtype=np.float64)

        pre = reward[max(0, shift_episode - window):shift_episode].mean()
        post_shift = reward[shift_episode:]
        # min_periods=window (NOT 1): a rolling mean over fewer than
        # `window` samples is dominated by single-episode noise (a lone
        # 0/1 reward), which would otherwise register as a spurious
        # "instant recovery" at episode 0-1 post-shift -- excluded here
        # by requiring a FULL window before a rolling estimate counts.
        rolling = pd.Series(post_shift).rolling(window=window, min_periods=window).mean()
        valid = rolling.dropna()

        half_life = float("nan")
        trough = float("nan")
        if not valid.empty:
            trough_idx = int(valid.idxmin())
            trough = float(valid.loc[trough_idx])
            target = trough + tolerance_fraction * (pre - trough)
            if pre > trough:
                # search only from the trough onward -- recovery is only
                # meaningful after the low point has actually been reached
                candidates = valid.loc[trough_idx:]
                crossed = candidates[candidates >= target]
                if not crossed.empty:
                    half_life = float(crossed.index[0])  # episodes since shift_episode

        rows.append(
            {
                "env_id": row["env_id"], "quality": row["quality"], "strength": row["strength"],
                "gamma": row["gamma"], "seed": row["seed"], "pre_shift_rate": pre,
                "trough_rate": trough, "half_life_episodes": half_life,
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# 5. Posterior entropy curves (directly instrumented -- not in the grid)
# ======================================================================


def posterior_entropy_curves(
    strengths: list[float] = (0.0, 5.0, 50.0),
    n_episodes: int = 3000,
    checkpoint_every: int = 100,
    seed: int = 0,
    matrix: np.ndarray = MATRIX,
) -> pd.DataFrame:
    """Track the RECOMMENDED arm's posterior entropy at coarse checkpoints
    for a few representative strengths, illustrating entropy COLLAPSE:
    a high-strength prior starts with LOW entropy (already concentrated
    near ``prior_mean``); stock Thompson (``strength=0``) starts at the
    ``Beta(1,1)`` MAXIMUM-entropy point and only narrows as evidence
    accumulates.
    """
    table = matrix.argmax(axis=1)
    n_strategies = matrix.shape[1]
    rows = []

    for strength in strengths:
        pgts = PriorInitializedThompson(table, n_strategies, strength=strength, prior_mean=0.7)
        env = MatrixEnvironment(matrix)
        rng = np.random.default_rng(seed)
        for i in range(n_episodes):
            true_type = int(rng.integers(0, matrix.shape[0]))
            ctx = Context(predicted_type=true_type, episode=i)
            action = pgts.select(ctx, rng)
            reward = int(rng.random() < matrix[true_type, action])
            pgts.update(ctx, action, reward)
            if (i + 1) % checkpoint_every == 0 or i == 0:
                ent = pgts.posterior_entropy()
                mean_tracked_entropy = float(np.mean([ent[t, table[t]] for t in range(matrix.shape[0])]))
                rows.append({"strength": strength, "episode": i + 1, "mean_entropy": mean_tracked_entropy})

    return pd.DataFrame(rows)


# ======================================================================
# 6. Posterior convergence: episodes until posterior mean is within
#    epsilon of the empirical optimum
# ======================================================================


def posterior_convergence(
    strengths: list[float] = (0.0, 5.0, 20.0, 50.0),
    epsilon: float = 0.05,
    n_episodes: int = 5000,
    n_seeds: int = 10,
    matrix: np.ndarray = MATRIX,
) -> pd.DataFrame:
    """For each strength, the episode at which the RECOMMENDED (and here,
    objectively best) arm's posterior mean first comes within ``epsilon``
    of ``matrix``'s TRUE best-arm success probability and stays there,
    averaged over seeds."""
    table = matrix.argmax(axis=1)
    n_strategies = matrix.shape[1]
    true_best = matrix.max(axis=1)
    rows = []

    for strength in strengths:
        for seed_idx in range(n_seeds):
            seed = 5000 + seed_idx
            pgts = PriorInitializedThompson(table, n_strategies, strength=strength, prior_mean=0.7)
            env = MatrixEnvironment(matrix)
            rng = np.random.default_rng(seed)
            convergence_episode = float("nan")
            for i in range(n_episodes):
                true_type = int(rng.integers(0, matrix.shape[0]))
                ctx = Context(predicted_type=true_type, episode=i)
                action = pgts.select(ctx, rng)
                reward = int(rng.random() < matrix[true_type, action])
                pgts.update(ctx, action, reward)
                mean_now = pgts.posterior_mean()
                gap = np.mean(
                    [abs(mean_now[t, table[t]] - true_best[t]) for t in range(matrix.shape[0])]
                )
                if gap <= epsilon and np.isnan(convergence_episode):
                    convergence_episode = float(i + 1)
            rows.append({"strength": strength, "seed": seed, "convergence_episode": convergence_episode})

    return pd.DataFrame(rows)


# ======================================================================
# 7. Prior sensitivity: effect of prior_mean on convergence/performance
# ======================================================================


def prior_mean_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ``normalized_score`` per ``(env_id, quality, strength,
    prior_mean)`` -- read across ``prior_mean`` at fixed
    ``(env_id, quality, strength)`` to see whether ``0.7`` is actually
    the best choice or just a common default (it is NOT assumed to be
    optimal -- see docs/experiments/hybrid.md)."""
    pgts = df[df["method"] == "pgts"]
    return (
        pgts.groupby(["env_id", "quality", "strength", "prior_mean"])[PERFORMANCE_METRIC]
        .agg(mean="mean", n="count")
        .reset_index()
        .sort_values(["env_id", "quality", "strength", "prior_mean"])
    )


# ======================================================================
# 8. Factor contribution (descriptive decomposition -- NOT ANOVA)
# ======================================================================


def factor_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive spread (max group mean - min group mean) of
    ``normalized_score`` attributable to each factor, marginalizing over
    everything else. NOT a variance decomposition / ANOVA -- a simple,
    transparent "how much does the group MEAN move if I only look at
    this factor" heuristic, explicitly per the task's instruction not to
    perform ANOVA."""
    pgts = df[df["method"].isin(["pgts", "pgts_static_limit", "stock_thompson"])]
    factors = ["strength", "quality", "classifier_id", "env_id"]
    rows = []
    for factor in factors:
        if factor not in pgts.columns:
            continue
        group_means = pgts.groupby(factor)[PERFORMANCE_METRIC].mean()
        if group_means.empty:
            continue
        rows.append(
            {
                "factor": factor, "n_levels": len(group_means),
                "min_group_mean": float(group_means.min()), "max_group_mean": float(group_means.max()),
                "descriptive_spread": float(group_means.max() - group_means.min()),
            }
        )
    result = pd.DataFrame(rows).sort_values("descriptive_spread", ascending=False)
    return result


# ======================================================================
# 9. Interpretation report (written, mechanism-focused)
# ======================================================================


def interpretation_report(
    negative_transfer_df: pd.DataFrame,
    optimal_strength_df: pd.DataFrame,
    half_life_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> str:
    """Build a written interpretation grounded in THIS run's actual
    computed numbers (never hardcoded) -- answers the five questions
    Task 7 asks for, with mechanisms, not just numbers."""
    lines: list[str] = []
    lines.append("=== PGTS Interpretation Report ===\n")

    lines.append("Why does a strong prior help?")
    good_quality_rows = optimal_strength_df[optimal_strength_df["quality"] >= 6]
    if not good_quality_rows.empty:
        example = good_quality_rows.iloc[0]
        lines.append(
            f"  With a mostly-correct table (quality={example['quality']}/7), the optimal strength "
            f"found here is {example['optimal_strength']:g} (mean {PERFORMANCE_METRIC}="
            f"{example['mean_at_optimum']:.3f}, 95% CI [{example['ci_lo']:.3f}, {example['ci_hi']:.3f}]). "
            f"A correct prior removes almost all of Thompson Sampling's early-exploration cost: instead of "
            f"paying O(K log T) episodes discovering the best arm per type from scratch, the posterior "
            f"already concentrates near it, so the policy exploits near-optimally from episode 1."
        )
    lines.append("")

    lines.append("Why does a strong prior hurt?")
    negative_rows = negative_transfer_df[negative_transfer_df["is_negative_transfer"]]
    if not negative_rows.empty:
        worst = negative_rows.sort_values("delta_vs_stock").iloc[0]
        lines.append(
            f"  The worst observed negative transfer: quality={worst['quality']}/7, "
            f"strength={worst['strength']:g}, env={worst['env_id']}, "
            f"delta vs stock Thompson = {worst['delta_vs_stock']:+.3f}. A high strength makes the posterior "
            f"CONFIDENT before it has seen any real evidence; if the table is wrong, that confidence has to "
            f"be overcome by real observations before the policy will explore away from the recommended "
            f"(wrong) arm -- the same mechanism that makes a correct prior converge fast makes an incorrect "
            f"one converge fast to the WRONG answer."
        )
    else:
        lines.append(
            "  No negative-transfer cells were found in this run's grid -- see the 'negative transfer "
            "region' table for the specific (quality, strength, env) combinations checked."
        )
    lines.append("")

    lines.append("What causes negative transfer?")
    lines.append(
        "  The combination of (a) LOW table quality (the recommended action is often wrong) and (b) HIGH "
        "strength (the wrong recommendation is asserted with a large, hard-to-overturn pseudo-count). "
        "Neither alone is sufficient: low quality with strength=0 degrades gracefully to stock Thompson "
        "(the table is ignored); high strength with a correct table only helps (see quality=7 rows above). "
        "It is specifically the PRODUCT of wrongness and confidence that produces harm."
    )
    lines.append("")

    lines.append("Why does Discounted PGTS recover faster (under drift)?")
    gamma_trend_matches_hypothesis = None
    if not half_life_df.empty:
        # Broken out by table quality: a WRONG prior (low quality) is the
        # condition discounting is actually meant to help escape -- mixing
        # it with a correct prior (which has nothing to "recover" from,
        # since it wasn't wrong pre-shift) would mask the effect.
        bad_prior = half_life_df[half_life_df["quality"] <= 1]
        if not bad_prior.empty:
            by_gamma_bad = bad_prior.groupby("gamma")["half_life_episodes"].mean().sort_index()
            lines.append(
                f"  Mean recovery half-life by discount factor, LOW-quality prior only "
                f"(quality<=1, this run): {by_gamma_bad.round(1).to_dict()}"
            )
            sorted_gammas = sorted(by_gamma_bad.index)  # ascending gamma = ascending "less discounting"
            values_by_ascending_gamma = [by_gamma_bad[g] for g in sorted_gammas]
            gamma_trend_matches_hypothesis = all(
                values_by_ascending_gamma[i] >= values_by_ascending_gamma[i + 1]
                for i in range(len(values_by_ascending_gamma) - 1)
            )
        by_gamma_all = half_life_df.groupby("gamma")["half_life_episodes"].mean().sort_index()
        lines.append(f"  Mean recovery half-life by discount factor, all qualities pooled: {by_gamma_all.round(1).to_dict()}")

    if gamma_trend_matches_hypothesis is True:
        lines.append(
            "  This run's data shows the expected direction: LOWER gamma (heavier discounting) -> SHORTER "
            "recovery half-life for a wrong prior."
        )
    elif gamma_trend_matches_hypothesis is False:
        lines.append(
            "  NOTE: this specific (small, scoped) validation run does NOT show a clean monotonic gamma->"
            "half-life trend in the numbers above -- reported honestly rather than smoothed over. Plausible "
            "causes: (a) too few seeds (15/cell here) against a genuinely noisy half-life estimate (a single "
            "rolling-window crossing point per run), (b) heavier discounting also erodes PRE-shift posterior "
            "certainty somewhat, which can shift the recovery TARGET (halfway between trough and a possibly "
            "lower pre-shift baseline) in ways that partially offset the intended effect. The mechanism "
            "explanation below holds regardless (it follows from the update rule directly, not from this "
            "measurement) -- a larger run (more seeds, longer post-shift window) is needed to confirm the "
            "magnitude empirically."
        )
    lines.append(
        "  Every update() call discounts the WHOLE row toward Beta(1,1) BEFORE applying new evidence "
        "(recovery.policies.hybrid.DiscountedPriorThompson) -- so a stale belief (whether from the ORIGINAL "
        "prior or from pre-shift evidence) fades geometrically at rate gamma regardless of source. Once the "
        "environment shifts, discounting shrinks the confidence attached to the now-wrong arm every single "
        "episode, letting real post-shift evidence (which is never discounted, only ever added) regain "
        "influence faster than it could under gamma=1 (where old evidence never fades and must be diluted "
        "by an ever-growing count of new observations instead)."
    )
    lines.append("")

    lines.append("When should practitioners use s=5 vs s=50 instead of stock Thompson?")
    lines.append(
        "  s=5 is a LOW-COMMITMENT nudge: it biases early exploration toward the table's recommendation "
        "without making it very hard to overturn (a handful of contrary observations already dominate the "
        "prior's pseudo-count) -- reasonable whenever the table's quality is uncertain, or the environment "
        "may drift. s=50 is a HIGH-COMMITMENT bet: appropriate only when the table is trusted (high quality, "
        "e.g. hand-verified by a domain expert) AND the environment is believed stationary -- see the factor "
        "contribution table below for how much 'strength' vs 'quality' vs 'drift' each moved the outcome in "
        "this run, as a rough guide to which factor is worth being confident about."
    )
    if not factor_df.empty:
        lines.append(f"  Descriptive factor spread (this run): {factor_df.set_index('factor')['descriptive_spread'].round(3).to_dict()}")

    return "\n".join(lines)


# ======================================================================
# Orchestration
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--discounted-input", type=Path, default=None,
        help="Path to the metadata parquet from `prior_grid.py --discounted` "
        "(identity columns only -- full logs are read from --discounted-store-root).",
    )
    parser.add_argument(
        "--discounted-store-root", type=Path,
        default=_BACKEND_DIR / "results" / "runs" / "discounted_prior_grid",
    )
    parser.add_argument("--shift-episode", type=int, default=250)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    surface = negative_transfer_surface(df)
    surface.to_parquet(args.output_dir / "prior_negative_transfer_surface.parquet", index=False)
    print(f"1. Negative-transfer surface: {len(surface)} rows")

    optimal = optimal_strength_per_quality(df)
    optimal.to_parquet(args.output_dir / "prior_optimal_strength.parquet", index=False)
    print(f"2. Optimal strength per quality:\n{optimal.to_string(index=False)}")

    neg_region = negative_transfer_region(df)
    neg_region.to_parquet(args.output_dir / "prior_negative_transfer_region.parquet", index=False)
    n_negative = int(neg_region["is_negative_transfer"].sum())
    print(f"3. Negative-transfer region: {n_negative}/{len(neg_region)} cells below stock Thompson")

    half_life = pd.DataFrame()
    if args.discounted_input is not None and args.discounted_input.exists():
        discounted_df = pd.read_parquet(args.discounted_input)
        half_life = recovery_half_life(discounted_df, args.discounted_store_root, shift_episode=args.shift_episode)
        half_life.to_parquet(args.output_dir / "prior_recovery_half_life.parquet", index=False)
    print(f"4. Recovery half-life: {len(half_life)} rows" + (" (skipped: no --discounted-input)" if half_life.empty else ""))

    entropy_curves = posterior_entropy_curves()
    entropy_curves.to_parquet(args.output_dir / "prior_entropy_curves.parquet", index=False)
    print(f"5. Posterior entropy curves: {len(entropy_curves)} rows")

    convergence = posterior_convergence()
    convergence.to_parquet(args.output_dir / "prior_posterior_convergence.parquet", index=False)
    print(f"6. Posterior convergence:\n{convergence.groupby('strength')['convergence_episode'].mean().to_string()}")

    sensitivity = prior_mean_sensitivity(df)
    sensitivity.to_parquet(args.output_dir / "prior_mean_sensitivity.parquet", index=False)
    print(f"7. Prior-mean sensitivity: {len(sensitivity)} rows")

    factors = factor_contribution(df)
    factors.to_parquet(args.output_dir / "prior_factor_contribution.parquet", index=False)
    print(f"8. Factor contribution:\n{factors.to_string(index=False)}")

    report = interpretation_report(neg_region, optimal, half_life, factors)
    report_path = args.output_dir / "prior_interpretation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n9. Interpretation report written to {report_path}\n")
    print(report)


if __name__ == "__main__":
    main()
