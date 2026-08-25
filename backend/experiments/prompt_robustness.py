#!/usr/bin/env python
"""Task 15: prompt-robustness supplementary experiment.

This is a SUPPLEMENTARY analysis of the Task 11 LLM blind-design study. It
does NOT replace or modify the primary protocol or its results
(``results/tables/llm_designers.parquet``, read here READ-ONLY) -- the
primary run remains the paper's headline. Every output this script writes
goes to a separate, clearly-``prompt_robustness_``-prefixed file, both in
``results/tables/`` and ``results/figures/`` -- never pooled with, and
never able to overwrite, the primary run (``experiments/llm_designers.py``
itself structurally refuses to write a non-primary ``--prompt-variant`` to
its default output path -- see that script's ``main()``).

Part A -- paraphrase stability
-------------------------------
Two paraphrases of the Task 11 instrument (``recovery.envs.paraphrases``),
preserving the task, the 7 failure types, the 6 strategies (names, labels,
costs), and the response-format requirement EXACTLY -- only the framing
and description PROSE differs. Each is audited against the same
blindness check the study backend's HTTP responses are audited against
(``recovery.analysis.prompt_blindness_audit``) before any API quota is
spent. Each paraphrase is then run, at reduced ``n``, across all 3
configured LLM families, by invoking ``experiments/llm_designers.py`` as a
subprocess (100% reuse of its request/retry/429/parsing logic -- this
script does not reimplement any of it) with an explicit non-default
``--output``. Per-family ``q`` (with a bootstrap CI) is reported for each
paraphrase and compared against the primary run's ``q`` via a real
hypothesis test (``recovery.analysis.significance.variant_q_comparison_test``,
justified for this real-API-sample data by the same policy documented in
``docs/statistics_registry.md`` for Task 13's per-type tests), Holm-corrected
within each family across its 3 pairwise variant comparisons.

Part B -- presentation-order randomization (google-gemini only)
------------------------------------------------------------------
Re-runs google-gemini with the PRIMARY wording, but each sample gets a
per-sample-index-seeded randomized display order for the failure-type and
strategy list (``recovery.envs.display_order`` -- the EXACT SAME
deterministic randomiser the human study instrument uses, reused here with
an LLM sample index in place of a participant token). Tests whether the
Task 13 finding of only 2 distinct mappings across 21 identically-worded
repeated calls is an artefact of the identical-prompt structure, or
genuine low-diversity model behaviour that persists even when the prompt's
surface structure varies per call. Calls ``experiments.llm_designers.run_one_sample``
directly (the same tested per-call logic Part A's subprocess also runs) in
a thin per-sample loop, since randomizing per-sample needs a prompt that
varies per call -- something the CLI (one prompt built once per run) does
not support and this task does not require changing.

Usage
-----
    python experiments/prompt_robustness.py --dry-run
    python experiments/prompt_robustness.py                        # Part A + Part B, full
    python experiments/prompt_robustness.py --skip-randomization   # Part A only
    python experiments/prompt_robustness.py --skip-paraphrase      # Part B only
    python experiments/prompt_robustness.py --families groq-llama,google-gemini
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from experiments.llm_designers import (  # noqa: E402
    DEFAULT_CONFIG,
    RESPONSE_FORMAT_INSTRUCTIONS,
    build_task_instructions,
    prompt_hash,
    run_one_sample,
)
from recovery.analysis.human_study import score_submission  # noqa: E402
from recovery.analysis.prompt_blindness_audit import audit_prompt_text  # noqa: E402
from recovery.analysis.significance import n_distinct_mappings, variant_q_comparison_test  # noqa: E402
from recovery.envs.canonical import MATRIX  # noqa: E402
from recovery.envs.display_order import display_order, seed_from_token  # noqa: E402
from recovery.envs.paraphrases import PARAPHRASE_BUILDERS  # noqa: E402
from recovery.metrics.bootstrap import bootstrap_ci  # noqa: E402
from src.evaluation.stats_utils import holm_bonferroni  # noqa: E402

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

_env_path = _BACKEND_DIR / ".env"
load_dotenv(_env_path if _env_path.exists() else _BACKEND_DIR / ".env.example")

TABLES_DIR = _BACKEND_DIR / "results" / "tables"
FIGURES_DIR = _BACKEND_DIR / "results" / "figures"
PRIMARY_LLM_DATA = TABLES_DIR / "llm_designers.parquet"

PARAPHRASE_VARIANTS = list(PARAPHRASE_BUILDERS)  # ["paraphrase_a", "paraphrase_b"]
DEFAULT_N_TEMP1 = 8
DEFAULT_N_TEMP0 = 1
DEFAULT_GEMINI_RANDOMIZED_N_TEMP1 = 20  # matches the primary google-gemini run's n=21, for a fair distinct-mapping comparison
DEFAULT_GEMINI_RANDOMIZED_N_TEMP0 = 1
GEMINI_FAMILY_CFG_LOOKUP = "google-gemini"


def _save_fig(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / {stem}.pdf")


# ======================================================================
# Blindness audit -- run BEFORE spending any API quota
# ======================================================================


def audit_all_variants() -> bool:
    """Audits primary + both paraphrases' FULL prompt text (instructions +
    response-format block) against the same check the study backend's HTTP
    responses are audited against. Returns True iff every variant passes.
    Never spends API quota if any variant fails."""
    print("=== Blindness audit (prompt text) ===")
    all_ok = True
    variants = {"primary": build_task_instructions, **PARAPHRASE_BUILDERS}
    for name, builder in variants.items():
        prompt = builder() + RESPONSE_FORMAT_INSTRUCTIONS
        violations = audit_prompt_text(prompt)
        status = "PASS" if not violations else "FAIL"
        print(f"  [{status}] {name}: {len(prompt)} chars, hash={prompt_hash(prompt)[:12]}")
        if violations:
            all_ok = False
            for v in violations:
                print(f"    VIOLATION: {v}")
    return all_ok


# ======================================================================
# Part A -- paraphrase runs (subprocess, 100% reuse of llm_designers.py)
# ======================================================================


def run_paraphrase_variant(
    variant: str, families: list[str] | None, n_temp1: int, n_temp0: int, dry_run: bool
) -> Path:
    output_path = TABLES_DIR / f"prompt_robustness_{variant}.parquet"
    cmd = [
        sys.executable, str(_BACKEND_DIR / "experiments" / "llm_designers.py"),
        "--config", str(DEFAULT_CONFIG),
        "--output", str(output_path),
        "--prompt-variant", variant,
        "--n-temp1", str(n_temp1),
        "--n-temp0", str(n_temp0),
    ]
    if families:
        cmd += ["--families", ",".join(families)]
    if dry_run:
        cmd += ["--dry-run"]
        # --dry-run refuses the default output path only; a non-default
        # path (ours, always) is fine, but --dry-run also makes zero API
        # calls and writes nothing -- print the command and stop here.
        print(f"  [DRY RUN] would invoke: {' '.join(cmd)}")
        return output_path

    print(f"\n--- Running paraphrase variant {variant!r} (n_temp1={n_temp1}, n_temp0={n_temp0}) ---")
    result = subprocess.run(cmd, cwd=str(_BACKEND_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"experiments/llm_designers.py --prompt-variant {variant} exited {result.returncode}")
    return output_path


# ======================================================================
# Part B -- google-gemini presentation-order randomization
# ======================================================================


def run_gemini_randomized_order(
    config: dict, n_temp1: int, n_temp0: int, dry_run: bool, overwrite: bool
) -> Path:
    output_path = TABLES_DIR / "prompt_robustness_gemini_randomized_order.parquet"
    if output_path.exists() and not overwrite and not dry_run:
        raise RuntimeError(
            f"{output_path} already exists; pass --overwrite-randomized to intentionally replace it "
            "(this script never silently overwrites a prior supplementary run)"
        )

    cfg = next((m for m in config["models"] if m["family"] == GEMINI_FAMILY_CFG_LOOKUP), None)
    if cfg is None:
        raise ValueError(f"{GEMINI_FAMILY_CFG_LOOKUP!r} not found in {DEFAULT_CONFIG}")
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        print(f"  [SKIPPED] {GEMINI_FAMILY_CFG_LOOKUP}: environment variable {cfg['api_key_env']!r} is not set")
        return output_path

    print(f"\n--- Running google-gemini under per-sample presentation-order randomization "
          f"(n_temp1={n_temp1}, n_temp0={n_temp0}) ---")
    if dry_run:
        for i in range(min(3, n_temp1)):
            t_order, s_order = display_order(f"llm-sample-{i:04d}", len(MATRIX), MATRIX.shape[1])
            prompt = build_task_instructions(t_order, s_order) + RESPONSE_FORMAT_INSTRUCTIONS
            print(f"  [DRY RUN] sample {i}: type_order={t_order} strategy_order={s_order} "
                  f"prompt_hash={prompt_hash(prompt)[:12]}")
        return output_path

    max_retries = config.get("max_parse_retries", 3)
    timeout = config.get("request_timeout_seconds", 60)
    rows = []
    t0 = time.time()
    for temperature, n in [(1.0, n_temp1), (0.0, n_temp0)]:
        for i in range(n):
            token = f"llm-sample-{i:04d}-temp{temperature}"
            type_order, strategy_order = display_order(token, len(MATRIX), MATRIX.shape[1])
            prompt = build_task_instructions(type_order, strategy_order) + RESPONSE_FORMAT_INSTRUCTIONS
            row = run_one_sample(
                family=GEMINI_FAMILY_CFG_LOOKUP, provider_type=cfg["provider_type"], base_url=cfg["base_url"],
                api_key=api_key, model=cfg["model"], prompt=prompt, temperature=temperature,
                max_tokens=cfg.get("max_tokens", 512), seed=i, timeout=timeout, max_retries=max_retries,
                sample_index=i, supports_seed=cfg.get("supports_seed", True),
            )
            row["prompt_variant"] = "primary_randomized_order"
            row["type_order"] = list(type_order)
            row["strategy_order"] = list(strategy_order)
            row["display_order_seed_token"] = token
            rows.append(row)
    elapsed = time.time() - t0

    df = pd.DataFrame(rows)
    df["parsed_mapping"] = df["parsed_mapping"].apply(lambda m: list(m) if m is not None else None)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_ok = int((df["parse_status"] == "ok").sum())
    print(f"  wrote {len(df)} rows to {output_path} in {elapsed:.1f}s ({n_ok}/{len(df)} parsed OK)")
    return output_path


# ======================================================================
# Analysis
# ======================================================================


def _q_values_for(df: pd.DataFrame, family: str) -> tuple[list[int], list, int]:
    """Returns (q_counts, parsed_mappings, n_distinct) for one family's
    successfully-parsed rows."""
    sub = df[(df["family"] == family) & (df["parse_status"] == "ok")]
    mappings = list(sub["parsed_mapping"])
    qs = [score_submission(m, MATRIX)["q_count"] for m in mappings]
    return qs, mappings, n_distinct_mappings(mappings) if mappings else 0


def paraphrase_q_report(primary_df: pd.DataFrame, paraphrase_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    N = MATRIX.shape[0]
    rows = []
    all_families = sorted(primary_df["family"].unique()) if not primary_df.empty else []
    sources = {"primary": primary_df, **paraphrase_dfs}
    for variant, df in sources.items():
        if df.empty:
            continue
        for family in sorted(df["family"].unique()):
            qs, mappings, n_distinct = _q_values_for(df, family)
            if not qs:
                rows.append({
                    "prompt_variant": variant, "family": family, "n_parsed": 0, "n_distinct_mappings": 0,
                    "bootstrap_ci_degenerate": True, "q_mean": float("nan"), "q_ratio_ci95_lower": float("nan"),
                    "q_ratio_ci95_upper": float("nan"),
                })
                continue
            degenerate = n_distinct <= 3 or len(qs) < 2
            ci = bootstrap_ci(np.asarray(qs, dtype=float) / N) if len(qs) >= 2 else (float("nan"), float("nan"))
            rows.append({
                "prompt_variant": variant, "family": family, "n_parsed": len(qs),
                "n_distinct_mappings": n_distinct, "bootstrap_ci_degenerate": degenerate,
                "q_mean": float(np.mean(qs)), "q_ratio_ci95_lower": ci[0], "q_ratio_ci95_upper": ci[1],
            })
    df_out = pd.DataFrame(rows)
    if not df_out.empty and all_families:
        df_out = df_out[df_out["family"].isin(all_families) | (df_out["prompt_variant"] != "primary")]
    return df_out


def variant_stability_tests(primary_df: pd.DataFrame, paraphrase_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per family: 3 pairwise Mann-Whitney tests on q (primary-vs-A,
    primary-vs-B, A-vs-B), Holm-corrected within that family's 3 tests."""
    sources = {"primary": primary_df, **paraphrase_dfs}
    variant_names = list(sources)
    families = sorted(set().union(*[set(df["family"].unique()) for df in sources.values() if not df.empty]))

    all_rows = []
    for family in families:
        q_by_variant = {}
        for variant in variant_names:
            df = sources[variant]
            if df.empty or family not in df["family"].unique():
                continue
            qs, _, _ = _q_values_for(df, family)
            if qs:
                q_by_variant[variant] = qs

        present = list(q_by_variant)
        pair_rows = []
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                v_a, v_b = present[i], present[j]
                result = variant_q_comparison_test(q_by_variant[v_a], q_by_variant[v_b])
                pair_rows.append({"family": family, "variant_a": v_a, "variant_b": v_b, **result})
        if not pair_rows:
            continue
        raw_p = [r["p_value"] for r in pair_rows]
        corrected = holm_bonferroni([p if not np.isnan(p) else 1.0 for p in raw_p])
        for r, p_holm in zip(pair_rows, corrected):
            r["p_value_holm"] = p_holm if not np.isnan(r["p_value"]) else float("nan")
            r["correction_method"] = "Holm-Bonferroni, within-family across its pairwise variant comparisons"
            r["significant_after_correction"] = bool(p_holm < 0.05) if not np.isnan(r["p_value"]) else None
        all_rows.extend(pair_rows)
    return pd.DataFrame(all_rows)


def gemini_randomization_report(original_df: pd.DataFrame, randomized_df: pd.DataFrame) -> dict:
    family = GEMINI_FAMILY_CFG_LOOKUP
    orig_qs, orig_mappings, orig_distinct = _q_values_for(original_df, family) if not original_df.empty else ([], [], 0)
    rand_qs, rand_mappings, rand_distinct = _q_values_for(randomized_df, family) if not randomized_df.empty else ([], [], 0)
    test = variant_q_comparison_test(orig_qs, rand_qs) if orig_qs and rand_qs else None
    return {
        "family": family,
        "n_original": len(orig_qs), "n_distinct_mappings_original": orig_distinct,
        "q_mean_original": float(np.mean(orig_qs)) if orig_qs else float("nan"),
        "n_randomized": len(rand_qs), "n_distinct_mappings_randomized": rand_distinct,
        "q_mean_randomized": float(np.mean(rand_qs)) if rand_qs else float("nan"),
        "mann_whitney_test": test,
    }


# ======================================================================
# Figures (SUPPLEMENTARY -- distinct filenames, distinct section)
# ======================================================================


def plot_paraphrase_q_by_family(report: pd.DataFrame) -> None:
    if report.empty:
        print("  [skipped] paraphrase q by family: no data")
        return
    families = sorted(report["family"].unique())
    variants = ["primary"] + PARAPHRASE_VARIANTS
    colors = {"primary": "#2563eb", "paraphrase_a": "#f59e0b", "paraphrase_b": "#16a34a"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / len(variants)
    x = np.arange(len(families))
    for i, variant in enumerate(variants):
        sub = report[report["prompt_variant"] == variant].set_index("family").reindex(families)
        means = sub["q_mean"].to_numpy()
        lo = means - sub["q_ratio_ci95_lower"].to_numpy() * MATRIX.shape[0]
        hi = sub["q_ratio_ci95_upper"].to_numpy() * MATRIX.shape[0] - means
        ax.bar(x + i * width, means, width=width, label=variant, color=colors.get(variant, "#888"), edgecolor="black")
        ax.errorbar(x + i * width, means, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], fmt="none",
                    ecolor="black", capsize=3, linewidth=1)
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(families, rotation=10)
    ax.set_ylabel("Mean q (correct rows out of 7)")
    ax.set_title("SUPPLEMENTARY (Task 15): q by prompt wording -- primary vs. 2 paraphrases\n"
                  "(NOT the paper's headline instrument comparison; see docs/experiments/prompt_robustness.md)")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "prompt_robustness_paraphrase_q_by_family")


def plot_gemini_randomization_comparison(report: dict) -> None:
    if report["n_original"] == 0 or report["n_randomized"] == 0:
        print("  [skipped] gemini randomization comparison: missing original or randomized data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(
        ["original\n(identical prompt)", "randomized order\n(per-sample)"],
        [report["n_distinct_mappings_original"], report["n_distinct_mappings_randomized"]],
        color=["#dc2626", "#2563eb"], edgecolor="black",
    )
    axes[0].set_ylabel("Number of distinct mappings")
    axes[0].set_title(f"Distinct mappings (n_original={report['n_original']}, n_randomized={report['n_randomized']})")

    axes[1].bar(
        ["original\n(identical prompt)", "randomized order\n(per-sample)"],
        [report["q_mean_original"], report["q_mean_randomized"]],
        color=["#dc2626", "#2563eb"], edgecolor="black",
    )
    axes[1].set_ylabel("Mean q")
    axes[1].set_title("Mean design quality")
    fig.suptitle("SUPPLEMENTARY (Task 15): google-gemini under presentation-order randomization")
    fig.tight_layout()
    _save_fig(fig, "prompt_robustness_gemini_randomized_order")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--families", type=str, default=None, help="Comma-separated subset (default: all configured).")
    parser.add_argument("--n-temp1", type=int, default=DEFAULT_N_TEMP1)
    parser.add_argument("--n-temp0", type=int, default=DEFAULT_N_TEMP0)
    parser.add_argument("--gemini-n-temp1", type=int, default=DEFAULT_GEMINI_RANDOMIZED_N_TEMP1)
    parser.add_argument("--gemini-n-temp0", type=int, default=DEFAULT_GEMINI_RANDOMIZED_N_TEMP0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-paraphrase", action="store_true")
    parser.add_argument("--skip-randomization", action="store_true")
    parser.add_argument("--overwrite-randomized", action="store_true",
                         help="Allow re-writing an existing prompt_robustness_gemini_randomized_order.parquet.")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",")] if args.families else None

    if not audit_all_variants():
        print("\nABORTING: at least one prompt variant failed the blindness audit. No API quota spent.")
        sys.exit(1)

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paraphrase_paths: dict[str, Path] = {}
    if not args.skip_paraphrase:
        for variant in PARAPHRASE_VARIANTS:
            paraphrase_paths[variant] = run_paraphrase_variant(
                variant, families, args.n_temp1, args.n_temp0, args.dry_run
            )

    randomized_path = None
    if not args.skip_randomization:
        randomized_path = run_gemini_randomized_order(
            config, args.gemini_n_temp1, args.gemini_n_temp0, args.dry_run, args.overwrite_randomized
        )

    if args.dry_run:
        print("\nDry run complete -- no API calls made, no analysis run.")
        return

    print("\n" + "=" * 70)
    print("SUPPLEMENTARY ANALYSIS (Task 15) -- never pooled with primary results")
    print("=" * 70)

    primary_df = pd.read_parquet(PRIMARY_LLM_DATA) if PRIMARY_LLM_DATA.exists() else pd.DataFrame()
    paraphrase_dfs = {
        v: (pd.read_parquet(p) if p.exists() else pd.DataFrame()) for v, p in paraphrase_paths.items()
    }

    if paraphrase_dfs:
        report = paraphrase_q_report(primary_df, paraphrase_dfs)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        report.to_parquet(TABLES_DIR / "prompt_robustness_q_report.parquet", index=False)
        print("\n=== q by (family, prompt_variant), with bootstrap 95% CI ===")
        print(report.to_string(index=False))

        tests = variant_stability_tests(primary_df, paraphrase_dfs)
        if not tests.empty:
            tests.to_parquet(TABLES_DIR / "prompt_robustness_variant_tests.parquet", index=False)
            print("\n=== Does q differ across prompt variants? (Mann-Whitney U, Holm-corrected within family) ===")
            print(tests.to_string(index=False))
            any_sig = bool(tests["significant_after_correction"].fillna(False).any())
            print(f"\nAny significant difference after correction: {any_sig}")
            print("Stability strengthens the paper's LLM-arm claims; instability is a genuine, reportable "
                  "limitation of LLM-as-designer methodology (either way, reported honestly here).")

        plot_paraphrase_q_by_family(report)

    if randomized_path is not None and randomized_path.exists():
        randomized_df = pd.read_parquet(randomized_path)
        rep = gemini_randomization_report(primary_df, randomized_df)
        pd.DataFrame([{
            k: v for k, v in rep.items() if k != "mann_whitney_test"
        } | (rep["mann_whitney_test"] or {})]).to_parquet(
            TABLES_DIR / "prompt_robustness_gemini_randomization_report.parquet", index=False
        )
        print("\n=== google-gemini: original (identical prompt) vs. randomized presentation order ===")
        print(f"  n_distinct_mappings: original={rep['n_distinct_mappings_original']} "
              f"(n={rep['n_original']}) vs. randomized={rep['n_distinct_mappings_randomized']} "
              f"(n={rep['n_randomized']})")
        print(f"  q_mean: original={rep['q_mean_original']:.3f} vs. randomized={rep['q_mean_randomized']:.3f}")
        if rep["mann_whitney_test"]:
            print(f"  Mann-Whitney U={rep['mann_whitney_test']['u_statistic']:.1f} "
                  f"p={rep['mann_whitney_test']['p_value']:.4f}")
        plot_gemini_randomization_comparison(rep)


if __name__ == "__main__":
    main()
