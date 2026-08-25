#!/usr/bin/env python
"""Task 12: SUCCESS_MATRIX sensitivity and robustness analysis.

Every result in this project's paper is conditional on ONE hand-authored
success-probability matrix (``recovery.envs.canonical.MATRIX``, see that
module's docstring: it "mirrors" ``src/recovery/strategies.py``'s
``SUCCESS_MATRIX``, which was authored by hand for this project -- not
derived from literature, not measured from real agent telemetry). This
script empirically asks whether the paper's headline stationary-world
claim ("a well-designed blind static rule beats every tested adaptive
method") describes the FRAMEWORK, or merely describes that one matrix.

Three perturbation schemes (``recovery.envs.matrix_perturbation``):

1. Independent per-entry truncated-Gaussian noise (``gaussian_perturb``),
   swept over ``sigma in {0.05, 0.10, 0.15, 0.20, 0.30}``, >=200 sampled
   matrices per sigma level.
2. Fractional row-entry permutation (``permute_row_entries``), swept over
   ``fraction in {0.1, 0.25, 0.5, 1.0}``, reported separately for
   rank-preserving (argmax value never moves, by construction) and
   rank-breaking (argmax value IS eligible to move) modes.
3. Three hand-authored structured matrices: uniform-difficulty,
   single-dominant-strategy, and a matrix where ``malformed_output`` is
   deliberately NOT the ceiling-hardest type.

For every perturbed world this script computes, per (matrix):

- ``Oracle``, ``BlindRule`` (the canonical ``BLIND_RULE``, 6/7 correct),
  and ``UniformRandom`` success rates -- all EXACT/closed-form via
  ``recovery.theory.analytic`` (``oracle_rate``/``policy_rate``/
  ``random_rate``), never simulated: these are properties of a FIXED
  policy under a known matrix, so simulating them would only add noise
  with zero scientific benefit.
- ``Thompson`` and ``LinUCB`` late-stage (final 20% of 500 episodes)
  success rates -- REAL Monte Carlo simulation via the existing kernel
  (``recovery.core.episode.run_episodes``), averaged over
  ``--n-seeds`` independent seeds, because an adaptive policy's behavior
  genuinely depends on its learning trajectory and has no exact closed
  form in this codebase.
- The exact policy ranking (all 5 methods, ordered by rate), and whether
  the paper's headline relation (``BlindRule > Thompson`` AND
  ``BlindRule > LinUCB``) is preserved.
- The crossover budget ``T*`` (``recovery.theory.crossover.predict_crossover``,
  reusing Task 10's ALREADY-CALIBRATED constant ``c`` -- never refit
  here), including whether it is infinite (a perfect-table
  ``BLIND_RULE`` would report this).
- The ceiling-hardest failure type (``argmin_t max_k P[t, k]``) -- see
  ``recovery.envs.matrix_perturbation``'s module docstring for why this
  is a DIFFERENT construct from Task 11's blind-designer-accuracy
  finding, not a re-test of it.

Reuse discipline (see docs/experiments/matrix_robustness.md for the full
account): the simulation KERNEL (``run_episodes``), the ANALYTIC layer
(``recovery.theory.analytic``, ``recovery.theory.crossover``), the
METRICS layer (``recovery.metrics.bootstrap``), and the GUARDS layer
(``recovery.guards.invariants``) are all reused unmodified. ``RunStore``
(``recovery.core.parallel``) is deliberately NOT used here -- this
script's own docstring justifies the deviation: this sweep's output is a
single aggregated summary table (~2,200 worlds x a handful of scalar
fields), not a resumable large-artifact grid, and per-episode-log
persistence for ~13 million simulated episodes would be pure I/O
overhead with no reuse benefit within one sweep execution.
``experiments/stationary_comparison.py`` already establishes the
precedent of calling ``run_episodes`` directly in a loop rather than
through ``RunStore``/``run_grid`` for exactly this reason.

Usage
-----
    python experiments/matrix_robustness.py --smoke     # ~seconds, correctness check
    python experiments/matrix_robustness.py             # full run, ~10-20 minutes
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.episode import run_episodes  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX, TYPE_NAMES  # noqa: E402
from recovery.envs.matrix_env import MatrixEnvironment  # noqa: E402
from recovery.envs.matrix_perturbation import (  # noqa: E402
    gaussian_perturb,
    hardest_type_by_ceiling,
    permute_row_entries,
    structured_malformed_not_hardest,
    structured_single_dominant_strategy,
    structured_uniform_difficulty,
)
from recovery.guards.invariants import InvariantViolation, run_all_guards  # noqa: E402
from recovery.metrics.bootstrap import bootstrap_ci  # noqa: E402
from recovery.policies.adapters import LinUCBPolicy, ThompsonPolicy  # noqa: E402
from recovery.theory.analytic import oracle_rate, policy_rate, random_rate  # noqa: E402
from recovery.theory.crossover import predict_crossover  # noqa: E402

from experiments.study_analysis import load_calibrated_c  # noqa: E402

OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"
FIGURES_DIR = _BACKEND_DIR / "results" / "figures"

N_EPISODES = 500
LATE_STAGE_START_FRAC = 0.8
LATE_STAGE_N_EPISODES = int(N_EPISODES * (1 - LATE_STAGE_START_FRAC))

SIGMA_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
PERMUTE_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
PERMUTE_MODES = ["rank_preserving", "rank_breaking"]

N_MATRICES_SCHEME1_FULL = 200
N_MATRICES_SCHEME2_FULL = 150
N_SEEDS_FULL = 30  # Task 13: raised from 10 -- Task 12's noise-floor calibration
# (docs/experiments/matrix_robustness.md) showed n_seeds=10 leaves a non-trivial
# ~2.5% spurious-breakdown rate from simulation noise alone; Task 13's audit
# requires every reported CI to rest on >=30 seeds. Re-run, do not just relabel.

N_MATRICES_SCHEME1_SMOKE = 5
N_MATRICES_SCHEME2_SMOKE = 5
N_SEEDS_SMOKE = 2

N_NOISE_FLOOR_REPS_FULL = 200
N_NOISE_FLOOR_REPS_SMOKE = 10

BASE_SEED = 120_026  # arbitrary fixed constant; every world's seed derives from this deterministically


def evaluate_world(
    P: np.ndarray,
    scheme: str,
    magnitude: str,
    world_idx: int,
    seed: int,
    n_seeds: int,
    calibrated_c: float,
) -> dict:
    """Compute the complete per-world record for one (possibly perturbed) matrix.

    Parameters
    ----------
    P : ndarray, shape (7, 6)
        The world's success-probability matrix.
    scheme : str
        ``"canonical"``, ``"gaussian"``, ``"permute"``, or ``"structured"``.
    magnitude : str
        A scheme-specific label (e.g. ``"sigma=0.10"``,
        ``"fraction=0.25,rank_breaking"``, ``"uniform_difficulty"``).
    world_idx : int
        Index of this world within its (scheme, magnitude) cell -- for
        bookkeeping/reproducibility only, not used in any computation.
    seed : int
        Base seed for this world's Thompson/LinUCB simulations. Fully
        determines this world's simulated results (deterministic RNG).
    n_seeds : int
        Number of independent simulation seeds per method.
    calibrated_c : float
        Task 10's already-fitted crossover-prediction constant -- passed
        through to ``predict_crossover``, never refit here.

    Returns
    -------
    dict
        One row's worth of fields for the sweep's summary table.
    """
    o = oracle_rate(P)
    r_rand = random_rate(P)
    r_blind = policy_rate(P, BLIND_RULE)
    T_star, truncated = predict_crossover(P, BLIND_RULE, form="kl", c=calibrated_c)
    hardest_idx = hardest_type_by_ceiling(P)

    env = MatrixEnvironment(P)
    method_rates: dict[str, float] = {}
    for method_name, policy_cls in (("Thompson", ThompsonPolicy), ("LinUCB", LinUCBPolicy)):
        seed_rates = []
        for s in range(n_seeds):
            trial_seed = seed + s
            policy = policy_cls(seed=trial_seed)
            log = run_episodes(env, policy, PerfectClassifier(), n_episodes=N_EPISODES, seed=trial_seed)
            rate = log.success_rate(LATE_STAGE_START_FRAC, 1.0)
            run_all_guards(rate, P, n_episodes=LATE_STAGE_N_EPISODES)
            seed_rates.append(rate)
        method_rates[method_name] = float(np.mean(seed_rates))

    all_rates = {
        "Oracle": o,
        "BlindRule": r_blind,
        "LinUCB": method_rates["LinUCB"],
        "Thompson": method_rates["Thompson"],
        "UniformRandom": r_rand,
    }
    order = sorted(all_rates.keys(), key=lambda k: -all_rates[k])
    canonical_order = ["Oracle", "BlindRule", "LinUCB", "Thompson", "UniformRandom"]
    headline_preserved = r_blind > method_rates["Thompson"] and r_blind > method_rates["LinUCB"]

    return {
        "scheme": scheme,
        "magnitude": magnitude,
        "world_idx": world_idx,
        "seed": seed,
        "oracle_rate": o,
        "blind_rule_rate": r_blind,
        "uniform_random_rate": r_rand,
        "thompson_rate": method_rates["Thompson"],
        "linucb_rate": method_rates["LinUCB"],
        "T_star": T_star,
        "T_star_truncated": truncated,
        "T_star_infinite": bool(np.isinf(T_star)),
        "hardest_type_idx": hardest_idx,
        "hardest_type_name": TYPE_NAMES[hardest_idx],
        "ranking_order": " > ".join(order),
        "full_order_preserved": order == canonical_order,
        "headline_preserved": headline_preserved,
    }


def build_world_specs(
    n_matrices_scheme1: int, n_matrices_scheme2: int, n_seeds: int
) -> list[dict]:
    """Build the full list of (matrix, scheme, magnitude, seed) specs to evaluate.

    Deterministic: identical inputs always produce an identical list of
    matrices, in an identical order, via a fixed derivation from
    ``BASE_SEED`` -- no world's matrix or simulation seed depends on
    wall-clock time, process ID, or execution order.
    """
    specs: list[dict] = []

    # Reference point: the unperturbed canonical matrix, under the SAME
    # simulation settings used for every perturbed world -- this is what
    # "ranking preserved" is measured AGAINST, not Phase 1's old
    # 500ep/50trial/11-method numbers (different settings would confound
    # "did perturbation break it" with "did the settings change").
    specs.append({
        "matrix": MATRIX.copy(), "scheme": "canonical", "magnitude": "canonical",
        "world_idx": 0, "seed": BASE_SEED,
    })

    # Scheme 1: Gaussian noise.
    matrix_rng = np.random.default_rng(BASE_SEED + 1)
    for sigma_idx, sigma in enumerate(SIGMA_LEVELS):
        for i in range(n_matrices_scheme1):
            P = gaussian_perturb(MATRIX, sigma=sigma, rng=matrix_rng)
            seed = BASE_SEED + 1_000_000 + sigma_idx * 100_000 + i * 10
            specs.append({
                "matrix": P, "scheme": "gaussian", "magnitude": f"sigma={sigma:.2f}",
                "world_idx": i, "seed": seed,
            })

    # Scheme 2: fractional row-entry permutation, both modes.
    matrix_rng = np.random.default_rng(BASE_SEED + 2)
    for frac_idx, fraction in enumerate(PERMUTE_FRACTIONS):
        for mode_idx, mode in enumerate(PERMUTE_MODES):
            for i in range(n_matrices_scheme2):
                P = permute_row_entries(MATRIX, fraction=fraction, mode=mode, rng=matrix_rng)
                seed = BASE_SEED + 2_000_000 + frac_idx * 200_000 + mode_idx * 100_000 + i * 10
                specs.append({
                    "matrix": P, "scheme": "permute",
                    "magnitude": f"fraction={fraction:.2f},{mode}",
                    "world_idx": i, "seed": seed,
                })

    # Scheme 3: structured matrices.
    structured = {
        "uniform_difficulty": structured_uniform_difficulty(),
        "single_dominant_strategy": structured_single_dominant_strategy(),
        "malformed_not_hardest": structured_malformed_not_hardest(),
    }
    for i, (label, P) in enumerate(structured.items()):
        specs.append({
            "matrix": P, "scheme": "structured", "magnitude": label,
            "world_idx": 0, "seed": BASE_SEED + 3_000_000 + i * 100_000,
        })

    return specs


def build_noise_floor_specs(n_reps: int, n_seeds: int) -> list[dict]:
    """Replicate the UNPERTURBED canonical matrix ``n_reps`` times, each
    with an independent, disjoint block of ``n_seeds`` simulation seeds.

    Purpose: ``headline_preserved`` compares a REAL, noisy Monte Carlo
    estimate (Thompson/LinUCB's simulated late-stage rate, averaged over
    only ``n_seeds`` seeds) against an EXACT value (``BlindRule``'s
    closed-form rate). ``results/tables/table_v.parquet`` (Task-1-era,
    50 seeds/method) shows per-seed late-stage-rate standard deviation of
    ~0.04-0.06 for Thompson/LinUCB -- large enough that, at only
    ``n_seeds`` per world, SOME worlds will show a spuriously-broken
    headline ranking from simulation noise ALONE, even at ZERO matrix
    perturbation. This calibration run measures exactly how often that
    happens, so the main sweep's per-magnitude preservation fractions can
    be interpreted relative to this noise floor, not against a naive
    (and wrong) 100% expectation. See
    ``docs/experiments/matrix_robustness.md`` for the measured value and
    its role in this report's headline conclusion.
    """
    specs = []
    for i in range(n_reps):
        seed = BASE_SEED + 9_000_000 + i * (n_seeds + 1)  # disjoint seed blocks, no overlap across reps
        specs.append({
            "matrix": MATRIX.copy(), "scheme": "noise_floor", "magnitude": "canonical_replicate",
            "world_idx": i, "seed": seed,
        })
    return specs


def run_sweep(specs: list[dict], n_seeds: int, calibrated_c: float, n_jobs: int = -1) -> pd.DataFrame:
    """Evaluate every world spec (in parallel across matrices) and return
    the full tidy results DataFrame."""
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_world)(
            spec["matrix"], spec["scheme"], spec["magnitude"], spec["world_idx"], spec["seed"],
            n_seeds, calibrated_c,
        )
        for spec in specs
    )
    return pd.DataFrame(results)


# ======================================================================
# Aggregation
# ======================================================================


def ranking_preservation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of worlds preserving the headline ranking, per
    (scheme, magnitude), with bootstrap 95% CIs (each world's 0/1
    ``headline_preserved`` flag is the sample; ``bootstrap_ci`` with the
    default ``statistic=np.mean`` gives exactly the CI on a proportion)."""
    rows = []
    for (scheme, magnitude), sub in df.groupby(["scheme", "magnitude"], sort=False):
        flags = sub["headline_preserved"].to_numpy(dtype=np.float64)
        frac = float(flags.mean())
        ci = bootstrap_ci(flags) if len(flags) >= 2 else None
        full_frac = float(sub["full_order_preserved"].mean())
        rows.append({
            "scheme": scheme, "magnitude": magnitude, "n": len(sub),
            "fraction_headline_preserved": frac,
            "headline_ci95_lower": ci[0] if ci else None,
            "headline_ci95_upper": ci[1] if ci else None,
            "fraction_full_order_preserved": full_frac,
        })
    return pd.DataFrame(rows)


def t_star_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of T* per (scheme, magnitude): fraction infinite, and
    a bootstrap 95% CI on the mean over the FINITE subset only (matching
    Task 11's ``q_prior_crossover_distribution`` precedent for handling
    infinite crossover predictions)."""
    rows = []
    for (scheme, magnitude), sub in df.groupby(["scheme", "magnitude"], sort=False):
        t_stars = sub["T_star"].to_numpy(dtype=np.float64)
        finite = t_stars[np.isfinite(t_stars)]
        ci = bootstrap_ci(finite) if len(finite) >= 2 else None
        rows.append({
            "scheme": scheme, "magnitude": magnitude, "n": len(sub),
            "n_infinite": int(np.sum(~np.isfinite(t_stars))),
            "fraction_infinite": float(np.mean(~np.isfinite(t_stars))),
            "T_star_finite_mean": float(finite.mean()) if len(finite) else float("nan"),
            "T_star_finite_median": float(np.median(finite)) if len(finite) else float("nan"),
            "T_star_ci95_lower": ci[0] if ci else None,
            "T_star_ci95_upper": ci[1] if ci else None,
        })
    return pd.DataFrame(rows)


def hardest_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the ceiling-hardest failure type per (scheme, magnitude)."""
    rows = []
    for (scheme, magnitude), sub in df.groupby(["scheme", "magnitude"], sort=False):
        counts = sub["hardest_type_name"].value_counts(normalize=True).to_dict()
        rows.append({"scheme": scheme, "magnitude": magnitude, "n": len(sub), **counts})
    return pd.DataFrame(rows)


def breakdown_threshold(ranking_df: pd.DataFrame, scheme: str, magnitude_order: list[str]) -> "str | None":
    """The smallest tested magnitude (in ``magnitude_order``) at which
    ``fraction_headline_preserved`` first drops below 0.5 for the given
    scheme. Returns ``None`` (never a fabricated interpolated value) if
    no tested magnitude drops below 0.5."""
    sub = ranking_df[ranking_df["scheme"] == scheme].set_index("magnitude")
    for m in magnitude_order:
        if m in sub.index and sub.loc[m, "fraction_headline_preserved"] < 0.5:
            return m
    return None


# ======================================================================
# Figures
# ======================================================================


def _save_fig(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / {stem}.pdf")


def plot_ranking_preservation(ranking_df: pd.DataFrame) -> None:
    """Ranking-preservation vs. perturbation magnitude, one line per scheme."""
    fig, ax = plt.subplots(figsize=(9, 6))

    gaussian = ranking_df[ranking_df["scheme"] == "gaussian"].copy()
    if not gaussian.empty:
        gaussian["x"] = gaussian["magnitude"].str.replace("sigma=", "").astype(float)
        gaussian = gaussian.sort_values("x")
        yerr = [
            gaussian["fraction_headline_preserved"] - gaussian["headline_ci95_lower"],
            gaussian["headline_ci95_upper"] - gaussian["fraction_headline_preserved"],
        ]
        ax.errorbar(gaussian["x"], gaussian["fraction_headline_preserved"], yerr=yerr,
                    label="Scheme 1: Gaussian noise (sigma)", color="#2563eb", marker="o", capsize=4)

    for mode, color, marker in (("rank_preserving", "#16a34a", "s"), ("rank_breaking", "#dc2626", "^")):
        sub = ranking_df[ranking_df["scheme"] == "permute"].copy()
        sub = sub[sub["magnitude"].str.endswith(mode)]
        if sub.empty:
            continue
        sub["x"] = sub["magnitude"].str.extract(r"fraction=([\d.]+),")[0].astype(float)
        sub = sub.sort_values("x")
        yerr = [
            sub["fraction_headline_preserved"] - sub["headline_ci95_lower"],
            sub["headline_ci95_upper"] - sub["fraction_headline_preserved"],
        ]
        ax.errorbar(sub["x"], sub["fraction_headline_preserved"], yerr=yerr,
                    label=f"Scheme 2: permutation ({mode})", color=color, marker=marker, capsize=4)

    ax.axhline(0.5, color="#6b7280", linestyle=":", linewidth=1.5, label="50% breakdown threshold")
    ax.set_xlabel("Perturbation magnitude (sigma or permuted fraction)")
    ax.set_ylabel("Fraction of worlds preserving\nBlindRule > Thompson AND BlindRule > LinUCB")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Headline ranking preservation vs. perturbation magnitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "robustness_ranking_preservation")


def plot_t_star_distribution(df: pd.DataFrame) -> None:
    """Distribution of finite T* values, one panel per scheme, pooled
    across all magnitudes within that scheme (the fraction infinite is
    annotated per scheme, since a log-scale histogram cannot show mass at
    infinity)."""
    schemes = ["gaussian", "permute", "structured"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(15, 5), sharey=True)
    for ax, scheme in zip(axes, schemes):
        sub = df[df["scheme"] == scheme]
        finite = sub.loc[np.isfinite(sub["T_star"]), "T_star"].to_numpy()
        frac_inf = float((~np.isfinite(sub["T_star"])).mean()) if len(sub) else float("nan")
        if len(finite) > 0:
            ax.hist(np.log10(np.clip(finite, 1, None)), bins=30, color="#7c3aed", edgecolor="black", alpha=0.85)
        ax.set_title(f"{scheme}\n(n={len(sub)}, {frac_inf:.1%} infinite)")
        ax.set_xlabel("log10(T*) [finite draws only]")
    axes[0].set_ylabel("Count")
    fig.suptitle("Distribution of the crossover budget T* by perturbation scheme")
    fig.tight_layout()
    _save_fig(fig, "robustness_tstar_distribution")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Small, fast correctness-check run.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    n_matrices_scheme1 = N_MATRICES_SCHEME1_SMOKE if args.smoke else N_MATRICES_SCHEME1_FULL
    n_matrices_scheme2 = N_MATRICES_SCHEME2_SMOKE if args.smoke else N_MATRICES_SCHEME2_FULL
    n_seeds = N_SEEDS_SMOKE if args.smoke else N_SEEDS_FULL
    n_noise_floor_reps = N_NOISE_FLOOR_REPS_SMOKE if args.smoke else N_NOISE_FLOOR_REPS_FULL

    calibrated_c = load_calibrated_c()
    print(f"Loaded Task 10's calibrated crossover constant c={calibrated_c:.4f} (not refit)")

    # --- Noise-floor calibration: how often does the headline ranking
    # spuriously appear "broken" at ZERO matrix perturbation, purely from
    # Monte Carlo noise at n_seeds seeds/method? Run FIRST so the main
    # sweep's results can be reported against this baseline. ---
    noise_specs = build_noise_floor_specs(n_noise_floor_reps, n_seeds)
    print(f"\nRunning noise-floor calibration: {len(noise_specs)} independent replicates of the "
          f"UNPERTURBED canonical matrix, n_seeds={n_seeds} each...")
    t0 = time.time()
    noise_df = run_sweep(noise_specs, n_seeds, calibrated_c, n_jobs=args.n_jobs)
    print(f"  done in {time.time() - t0:.1f}s")
    noise_floor_rate = float(noise_df["headline_preserved"].mean())
    noise_floor_ci = bootstrap_ci(noise_df["headline_preserved"].to_numpy(dtype=np.float64))
    print(f"  Noise floor: fraction_headline_preserved={noise_floor_rate:.3f} "
          f"(95% CI [{noise_floor_ci[0]:.3f}, {noise_floor_ci[1]:.3f}]) at ZERO perturbation, "
          f"n_seeds={n_seeds} -- interpret every sweep result below RELATIVE TO this baseline, "
          f"not against a naive 100% expectation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    noise_path = args.output_dir / ("matrix_robustness_noise_floor_smoke.parquet" if args.smoke else "matrix_robustness_noise_floor.parquet")
    noise_df.to_parquet(noise_path, index=False)

    specs = build_world_specs(n_matrices_scheme1, n_matrices_scheme2, n_seeds)
    print(f"\nMode: {'SMOKE' if args.smoke else 'FULL'} -- {len(specs)} worlds, n_seeds={n_seeds}, "
          f"{N_EPISODES} episodes/seed, n_jobs={args.n_jobs}")

    t0 = time.time()
    df = run_sweep(specs, n_seeds, calibrated_c, n_jobs=args.n_jobs)
    elapsed = time.time() - t0
    print(f"Sweep complete in {elapsed:.1f}s ({len(df) / elapsed:.2f} worlds/sec)")
    raw_path = args.output_dir / ("matrix_robustness_raw_smoke.parquet" if args.smoke else "matrix_robustness_raw.parquet")
    df.to_parquet(raw_path, index=False)
    print(f"Wrote raw per-world results to {raw_path}")

    ranking_df = ranking_preservation_summary(df)
    ranking_path = args.output_dir / ("matrix_robustness_ranking_smoke.parquet" if args.smoke else "matrix_robustness_ranking.parquet")
    ranking_df.to_parquet(ranking_path, index=False)
    print(f"\n=== Ranking preservation summary ===")
    print(ranking_df.to_string(index=False))

    tstar_df = t_star_summary(df)
    tstar_path = args.output_dir / ("matrix_robustness_tstar_smoke.parquet" if args.smoke else "matrix_robustness_tstar.parquet")
    tstar_df.to_parquet(tstar_path, index=False)
    print(f"\n=== T* summary ===")
    print(tstar_df.to_string(index=False))

    hardest_df = hardest_type_summary(df)
    hardest_path = args.output_dir / ("matrix_robustness_hardest_type_smoke.parquet" if args.smoke else "matrix_robustness_hardest_type.parquet")
    hardest_df.to_parquet(hardest_path, index=False)
    print(f"\n=== Ceiling-hardest-type distribution ===")
    print(hardest_df.to_string(index=False))

    print("\n=== Breakdown thresholds (first tested magnitude with fraction_headline_preserved < 0.5) ===")
    gaussian_mags = [f"sigma={s:.2f}" for s in SIGMA_LEVELS]
    print(f"  Scheme 1 (gaussian): {breakdown_threshold(ranking_df, 'gaussian', gaussian_mags)}")
    for mode in PERMUTE_MODES:
        permute_mags = [f"fraction={f:.2f},{mode}" for f in PERMUTE_FRACTIONS]
        print(f"  Scheme 2 ({mode}): {breakdown_threshold(ranking_df, 'permute', permute_mags)}")

    print("\n=== Scheme 3: structured matrices ===")
    struct_rows = df[df["scheme"] == "structured"]
    print(struct_rows[["magnitude", "blind_rule_rate", "thompson_rate", "linucb_rate", "headline_preserved",
                        "hardest_type_name", "ranking_order"]].to_string(index=False))

    if not args.smoke:
        print("\n=== Figures ===")
        plot_ranking_preservation(ranking_df)
        plot_t_star_distribution(df)

    canonical_row = df[df["scheme"] == "canonical"].iloc[0]
    print(f"\n=== Canonical (unperturbed) matrix reference, same settings as the sweep ===")
    print(f"ranking: {canonical_row['ranking_order']}")
    print(f"hardest_type (ceiling sense): {canonical_row['hardest_type_name']}")
    print(f"T*: {canonical_row['T_star']}  (truncated={canonical_row['T_star_truncated']})")


if __name__ == "__main__":
    main()
