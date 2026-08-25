#!/usr/bin/env python
"""Task 11: combine the human and LLM-designer arms of the blind-design study.

Reads real (never fabricated) data from two places:

- ``study/data/submissions.jsonl`` (via ``study.storage.SubmissionStore``)
  -- the human arm. As of this writing this file does not exist / is
  empty: no study has been run against real participants yet, because
  doing so requires the institutional approval described in
  ``study/README.md``, which has not been obtained. Every section below
  that depends on human data reports this explicitly (``n=0``, "no data
  collected") rather than silently skipping or fabricating a distribution.
- ``results/tables/llm_designers.parquet`` -- the LLM arm, produced by
  ``experiments/llm_designers.py``. As of this writing this contains a
  small REAL validation run against the one model family this
  environment has API credentials for (see that script's module
  docstring); it is not the full 3-family x 21-sample run Task 11
  specifies, and is never presented as such (every group-by-family
  table states its own ``n``).

The headline crossover-budget analysis reuses Task 10's ALREADY-CALIBRATED
constant (``results/tables/crossover_calibration.parquet``) -- it is never
refit here.

Usage
-----
    python experiments/study_analysis.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from recovery.analysis.human_study import (  # noqa: E402
    compare_human_llm_q,
    entropy_per_type,
    fleiss_kappa,
    pairwise_agreement,
    q_prior_crossover_distribution,
    score_submission,
    study_summary,
)
from recovery.analysis.significance import n_distinct_mappings  # noqa: E402
from recovery.envs.canonical import MATRIX, TYPE_NAMES  # noqa: E402
from recovery.theory.crossover import predict_crossover  # noqa: E402
from study.storage import SubmissionStore  # noqa: E402

DEFAULT_HUMAN_DATA = _REPO_ROOT / "study" / "data" / "submissions.jsonl"
DEFAULT_LLM_DATA = _BACKEND_DIR / "results" / "tables" / "llm_designers.parquet"
DEFAULT_CALIBRATION = _BACKEND_DIR / "results" / "tables" / "crossover_calibration.parquet"
OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"
FIGURES_DIR = _BACKEND_DIR / "results" / "figures"
RATIONALE_DUMP_PATH = OUTPUT_DIR / "human_study_rationales.csv"

DEPLOYMENT_BUDGETS = [100, 1_000, 10_000, 100_000]


def _save_fig(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / {stem}.pdf")


def load_human_submissions(path: Path = DEFAULT_HUMAN_DATA) -> list[dict]:
    return SubmissionStore(path).read_all()


def load_llm_results(path: Path = DEFAULT_LLM_DATA) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_calibrated_c(path: Path = DEFAULT_CALIBRATION) -> float:
    """Read Task 10's already-fitted constant -- NEVER refit here."""
    calib = pd.read_parquet(path)
    calibrated = calib[calib["calibrated"]]
    return float(calibrated["best_fit_c"].iloc[0])


def llm_q_by_family(llm_df: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """Score every successfully-parsed LLM sample against ``P``, grouped
    by family. Non-``"ok"`` samples are EXCLUDED from the q distribution
    (they have no mapping to score) but their counts are reported
    alongside, per family, never silently dropped from the record --
    ``parse_status="parse_failure"`` (the model answered but not
    parseably), ``parse_status="network_timeout"`` (a read timeout on
    every retry attempt), and ``parse_status="rate_limited"`` (every
    retry attempt hit an HTTP 429 the provider's own reset window never
    cleared within the attempt budget) are counted SEPARATELY, since
    these are different failure modes with different implications -- a
    timeout or rate limit says nothing about the model's design quality;
    a parse failure might."""
    if llm_df.empty:
        return pd.DataFrame()
    rows = []
    for family, sub in llm_df.groupby("family"):
        ok = sub[sub["parse_status"] == "ok"]
        n_parse_failed = int((sub["parse_status"] == "parse_failure").sum())
        n_timed_out = int((sub["parse_status"] == "network_timeout").sum())
        n_rate_limited = int((sub["parse_status"] == "rate_limited").sum())
        qs = [score_submission(m, P)["q_count"] for m in ok["parsed_mapping"]]
        n_distinct = n_distinct_mappings(ok["parsed_mapping"]) if len(ok) else 0
        rows.append(
            {
                "family": family,
                "n_parsed": len(qs),
                "n_parse_failures": n_parse_failed,
                "n_network_timeouts": n_timed_out,
                "n_rate_limited": n_rate_limited,
                "n_distinct_mappings": n_distinct,
                # Task 13: a bootstrap CI resampled from <=3 distinct point
                # masses cannot represent genuine sampling uncertainty --
                # google-gemini's real n_distinct=2 is exactly this case.
                "bootstrap_ci_degenerate": n_distinct <= 3,
                "q_mean": float(np.mean(qs)) if qs else float("nan"),
                "q_median": float(np.median(qs)) if qs else float("nan"),
                "q_values": qs,
            }
        )
    return pd.DataFrame(rows)


def llm_per_type_accuracy_by_family(llm_df: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """Per-failure-type accuracy (fraction of parsed samples that chose
    that row's optimal strategy), one row per (family, failure_type) --
    the LLM-arm analogue of the human arm's ``study_summary()["per_type_accuracy"]``,
    computable purely from the LLM data (no human submissions needed),
    so it can be reported even while the human arm remains at ``n=0``."""
    if llm_df.empty:
        return pd.DataFrame()
    rows = []
    for family, sub in llm_df.groupby("family"):
        ok = sub[sub["parse_status"] == "ok"]
        if ok.empty:
            continue
        per_row = np.array([score_submission(m, P)["per_row_correct"] for m in ok["parsed_mapping"]])
        n_distinct = n_distinct_mappings(ok["parsed_mapping"])
        for t_idx, type_name in enumerate(TYPE_NAMES):
            rows.append({
                "family": family, "failure_type": type_name,
                "accuracy": float(per_row[:, t_idx].mean()), "n": len(ok),
                "n_distinct_mappings": n_distinct,
                "bootstrap_ci_degenerate": n_distinct <= 3,
            })
    return pd.DataFrame(rows)


def crossover_budget_analysis(
    human_submissions: list[dict], P: np.ndarray, c: float, form: str = "kl", budgets: list = DEPLOYMENT_BUDGETS
) -> pd.DataFrame:
    """THE HEADLINE OUTPUT: for each deployment budget T, the fraction of
    REAL human designers' tables whose predicted crossover T* (Task 10's
    calibrated predictor, applied to THIS submission's actual mapping --
    not a quality-level abstraction) exceeds T, i.e. the static table is
    still ahead of an adaptive bandit at that budget.

    Returns an EMPTY DataFrame (never a fabricated one) if
    ``human_submissions`` is empty -- see this module's docstring for why
    that is the current, honest state.
    """
    if not human_submissions:
        return pd.DataFrame()

    per_submission = []
    for sub in human_submissions:
        T_star, truncated = predict_crossover(P, np.asarray(sub["mapping"]), form=form, c=c)
        per_submission.append({"token": sub["token"], "T_star": T_star, "prediction_truncated": truncated})
    per_df = pd.DataFrame(per_submission)

    rows = []
    for budget in budgets:
        beats = np.mean(per_df["T_star"].to_numpy() > budget)
        rows.append({"budget_episodes": budget, "fraction_beats_bandit": float(beats), "n": len(per_df)})
    return pd.DataFrame(rows)


def llm_crossover_budget_analysis(
    llm_df: pd.DataFrame, P: np.ndarray, c: float, form: str = "kl", budgets: list = DEPLOYMENT_BUDGETS
) -> pd.DataFrame:
    """The LLM-arm analogue of ``crossover_budget_analysis``: for each
    deployment budget T and each model family, the fraction of that
    family's REAL, successfully-parsed designs whose predicted crossover
    T* (Task 10's calibrated predictor, applied to each sample's actual
    mapping) exceeds T. Grouped by family (not pooled) because the
    families have different, non-comparable q distributions -- pooling
    would average over that difference rather than expose it. Unlike
    ``crossover_budget_analysis`` (still empty at n=0 pending the human
    arm), this is computable NOW because the LLM arm has real, complete
    data -- but it answers a different question ("what does an LLM's
    blind design imply") from the paper's actual headline claim (what a
    HUMAN blind design implies), and must not be substituted for it."""
    if llm_df.empty:
        return pd.DataFrame()
    rows = []
    for family, sub in llm_df.groupby("family"):
        ok = sub[sub["parse_status"] == "ok"]
        if ok.empty:
            continue
        t_stars = np.array([predict_crossover(P, np.asarray(m), form=form, c=c)[0] for m in ok["parsed_mapping"]])
        for budget in budgets:
            beats = np.mean(t_stars > budget)
            rows.append({"family": family, "budget_episodes": budget, "fraction_beats_bandit": float(beats), "n": len(ok)})
    return pd.DataFrame(rows)


def dump_rationales(human_submissions: list[dict], scored: list[dict], output_path: Path = RATIONALE_DUMP_PATH) -> pd.DataFrame:
    """Every rationale, grouped by failure type and correct/incorrect, for
    manual thematic (qualitative) coding -- never analysed automatically
    beyond this grouping."""
    rows = []
    for sub, sc in zip(human_submissions, scored):
        for t_idx, type_name in enumerate(TYPE_NAMES):
            rationale = (sub.get("rationale") or [""] * len(TYPE_NAMES))[t_idx]
            if not rationale:
                continue
            rows.append(
                {
                    "token": sub["token"],
                    "failure_type": type_name,
                    "correct": sc["per_row_correct"][t_idx],
                    "rationale": rationale,
                }
            )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


# ======================================================================
# Publication figures
# ======================================================================


def plot_q_distribution(scored: list[dict], N: int) -> None:
    if not scored:
        print("  [skipped] q distribution: no human submissions")
        return
    qs = [s["q_count"] for s in scored]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(qs, bins=np.arange(-0.5, N + 1.5, 1), color="#2563eb", edgecolor="black", alpha=0.85)
    ax.set_xlabel("q (correct rows out of N)")
    ax.set_ylabel("Number of participants")
    ax.set_title(f"Distribution of blind-design quality q (n={len(qs)})")
    ax.set_xticks(range(N + 1))
    fig.tight_layout()
    _save_fig(fig, "human_study_q_distribution")


def plot_confidence_vs_correctness(human_submissions: list[dict], scored: list[dict]) -> None:
    all_conf, all_correct = [], []
    for sub, sc in zip(human_submissions, scored):
        for c, correct in zip(sub.get("confidence") or [], sc["per_row_correct"]):
            all_conf.append(c)
            all_correct.append(correct)
    if not all_conf:
        print("  [skipped] confidence vs correctness: no human submissions")
        return
    df = pd.DataFrame({"confidence": all_conf, "correct": all_correct})
    rates = df.groupby("confidence")["correct"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(rates.index.astype(str), rates.values, color="#16a34a", edgecolor="black")
    ax.set_xlabel("Self-reported confidence (1-5)")
    ax.set_ylabel("Fraction of rows correct")
    ax.set_ylim(0, 1)
    ax.set_title("Confidence calibration")
    fig.tight_layout()
    _save_fig(fig, "human_study_confidence_vs_correctness")


def plot_human_vs_llm_q(human_scored: list[dict], llm_by_family: pd.DataFrame) -> None:
    if not human_scored or llm_by_family.empty:
        print("  [skipped] human vs LLM q: needs both a non-empty human arm and LLM data")
        return
    human_qs = [s["q_count"] for s in human_scored]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["human"] + list(llm_by_family["family"])
    means = [np.mean(human_qs)] + list(llm_by_family["q_mean"])
    ax.bar(labels, means, color=["#2563eb"] + ["#f59e0b"] * len(llm_by_family), edgecolor="black")
    ax.set_ylabel("Mean q")
    ax.set_title("Human vs. LLM designer quality")
    fig.tight_layout()
    _save_fig(fig, "human_study_human_vs_llm_q")


def plot_per_type_accuracy(summary: dict) -> None:
    acc = summary.get("per_type_accuracy") or {}
    if not acc or summary["n"] == 0:
        print("  [skipped] per-failure-type accuracy: no human submissions")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(acc.keys())
    values = list(acc.values())
    ax.bar(names, values, color="#7c3aed", edgecolor="black")
    ax.set_ylabel("Fraction correct")
    ax.set_ylim(0, 1)
    ax.set_title("Per-failure-type accuracy (which types are hardest for humans)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save_fig(fig, "human_study_per_type_accuracy")


def plot_llm_q_by_family(llm_by_family: pd.DataFrame) -> None:
    """LLM-arm-only figure (no human data required): one strip of q
    values per model family, so the LLM arm has a real published figure
    even while the human arm remains at n=0 pending institutional
    approval."""
    if llm_by_family.empty:
        print("  [skipped] LLM q by family: no LLM run found")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    families = list(llm_by_family["family"])
    q_lists = list(llm_by_family["q_values"])
    ax.boxplot(q_lists, tick_labels=families, showmeans=True)
    for i, qs in enumerate(q_lists, start=1):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(qs))
        ax.scatter(np.full(len(qs), i) + jitter, qs, color="#f59e0b", edgecolor="black", zorder=3, s=25, alpha=0.7)
    ax.set_ylabel("q (correct rows out of N)")
    ax.set_title("LLM designer quality by model family")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    _save_fig(fig, "llm_study_q_by_family")


def plot_llm_per_type_accuracy(llm_type_acc: pd.DataFrame) -> None:
    """Grouped bar chart, one group per failure type, one bar per LLM
    family -- the LLM-arm analogue of ``plot_per_type_accuracy``,
    computable without any human data."""
    if llm_type_acc.empty:
        print("  [skipped] LLM per-failure-type accuracy: no LLM run found")
        return
    families = list(llm_type_acc["family"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(families)
    x = np.arange(len(TYPE_NAMES))
    colors = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626"]
    for i, family in enumerate(families):
        sub = llm_type_acc[llm_type_acc["family"] == family].set_index("failure_type").reindex(TYPE_NAMES)
        ax.bar(x + i * width, sub["accuracy"].to_numpy(), width=width, label=family, color=colors[i % len(colors)], edgecolor="black")
    ax.set_xticks(x + width * (len(families) - 1) / 2)
    ax.set_xticklabels(TYPE_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Fraction correct")
    ax.set_ylim(0, 1)
    ax.set_title("Per-failure-type accuracy by LLM model family")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "llm_study_per_type_accuracy_by_family")


def plot_time_vs_q(human_submissions: list[dict], scored: list[dict]) -> None:
    if not human_submissions:
        print("  [skipped] completion time vs q: no human submissions")
        return
    times = [s.get("seconds_on_task") for s in human_submissions]
    qs = [s["q_count"] for s in scored]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(times, qs, color="#dc2626", edgecolor="black", s=60)
    ax.set_xlabel("Seconds on task")
    ax.set_ylabel("q")
    ax.set_title("Completion time vs. quality")
    fig.tight_layout()
    _save_fig(fig, "human_study_time_vs_q")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-data", type=Path, default=DEFAULT_HUMAN_DATA)
    parser.add_argument("--llm-data", type=Path, default=DEFAULT_LLM_DATA)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()

    human_submissions = load_human_submissions(args.human_data)
    llm_df = load_llm_results(args.llm_data)
    print(f"Human submissions: n={len(human_submissions)}" + (" (NO DATA -- pending institutional approval, see study/README.md)" if not human_submissions else ""))
    print(f"LLM samples: n={len(llm_df)}" + (f" ({(llm_df['parse_status'] == 'ok').sum()} parsed OK)" if not llm_df.empty else " (no LLM run found)"))

    summary = study_summary(human_submissions, MATRIX) if human_submissions else {
        "n": 0, "per_type_accuracy": {}, "scored_submissions": [],
    }
    scored = summary["scored_submissions"]

    llm_by_family = llm_q_by_family(llm_df, MATRIX)

    print("\n=== Human q distribution ===")
    if summary["n"] > 0:
        print(f"n={summary['n']} mean={summary['q_mean']:.3f} median={summary['q_median']:.3f} "
              f"CI={summary['q_mean_ci']} histogram={summary['q_histogram']}")
    else:
        print("No human submissions collected yet.")

    print("\n=== LLM q distribution by family ===")
    print(llm_by_family.to_string(index=False) if not llm_by_family.empty else "No LLM data.")

    print("\n=== LLM per-failure-type accuracy by family ===")
    llm_type_acc = llm_per_type_accuracy_by_family(llm_df, MATRIX)
    if not llm_type_acc.empty:
        print(llm_type_acc.to_string(index=False))
        llm_type_acc.to_parquet(OUTPUT_DIR / "llm_study_per_type_accuracy_by_family.parquet", index=False)
    else:
        print("No LLM data.")

    print("\n=== Agreement analysis (human arm) ===")
    if human_submissions:
        N = MATRIX.shape[0]
        K = MATRIX.shape[1]
        kappa = fleiss_kappa(human_submissions, N, K)
        pairwise = pairwise_agreement(human_submissions, N)
        entropy = entropy_per_type(human_submissions, N, K)
        print(f"Fleiss' kappa: {kappa}")
        print(f"Pairwise agreement: {pairwise}")
        print(f"Entropy per failure type (bits, max={np.log2(K):.3f}): {entropy}")
    else:
        print("Cannot compute: fewer than 2 human submissions.")

    print("\n=== Human vs. LLM comparison ===")
    if summary["n"] > 0 and not llm_by_family.empty:
        all_llm_qs = [q for qs in llm_by_family["q_values"] for q in qs]
        human_qs = [s["q_count"] for s in scored]
        comparison = compare_human_llm_q(human_qs, all_llm_qs)
        print(comparison)
    else:
        comparison = None
        print("Cannot compare: need both a non-empty human arm and LLM data.")

    c = load_calibrated_c(args.calibration)

    print("\n=== Crossover-budget analysis (HEADLINE -- human arm) ===")
    if summary["n"] > 0:
        budget_df = crossover_budget_analysis(human_submissions, MATRIX, c)
        print(budget_df.to_string(index=False))
        budget_df.to_parquet(OUTPUT_DIR / "human_study_crossover_budgets.parquet", index=False)
    else:
        budget_df = pd.DataFrame()
        print("Cannot compute: no human submissions.")

    print("\n=== Crossover-budget analysis (LLM arm -- a different question, not a substitute for the human headline) ===")
    llm_budget_df = llm_crossover_budget_analysis(llm_df, MATRIX, c)
    if not llm_budget_df.empty:
        print(llm_budget_df.to_string(index=False))
        llm_budget_df.to_parquet(OUTPUT_DIR / "llm_study_crossover_budgets.parquet", index=False)
    else:
        print("No LLM data.")

    print("\n=== T* distribution from the empirical q PRIOR (Task 10 consuming q_prior) ===")
    if summary["n"] > 0:
        q_dist = q_prior_crossover_distribution(scored, MATRIX, c)
        print(
            f"n_input_submissions={q_dist['n_input_submissions']} n_draws={q_dist['n_draws']} "
            f"T*_median={q_dist['T_star_median']:.1f} T*_CI={q_dist['T_star_ci']} "
            f"(finite draws: {q_dist['n_finite']}/{q_dist['n_draws']})"
        )
        pd.DataFrame({"T_star": q_dist["T_star_samples"]}).to_parquet(
            OUTPUT_DIR / "human_study_q_prior_crossover_distribution.parquet", index=False
        )
    else:
        print("Cannot compute: q_prior requires at least one scored submission.")

    if human_submissions:
        rationale_df = dump_rationales(human_submissions, scored)
        print(f"\nDumped {len(rationale_df)} rationales to {RATIONALE_DUMP_PATH}")

    print("\n=== Figures ===")
    plot_q_distribution(scored, MATRIX.shape[0])
    plot_confidence_vs_correctness(human_submissions, scored)
    plot_human_vs_llm_q(scored, llm_by_family)
    plot_per_type_accuracy(summary)
    plot_time_vs_q(human_submissions, scored)
    plot_llm_q_by_family(llm_by_family)
    plot_llm_per_type_accuracy(llm_type_acc)

    if not llm_by_family.empty:
        llm_by_family.drop(columns=["q_values"]).to_parquet(OUTPUT_DIR / "human_study_llm_by_family.parquet", index=False)


if __name__ == "__main__":
    main()
