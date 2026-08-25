#!/usr/bin/env python
"""Task 13: statistical review and hypothesis registry.

Two distinct jobs, kept structurally separate because they answer
different kinds of questions (see ``recovery/analysis/significance.py``
and ``docs/statistics.md`` for the full reasoning):

1. **New real-sample inferential tests** on the Task 11 LLM blind-design
   study (``results/tables/llm_designers.parquet``) -- real API responses,
   not simulation seeds, so a p-value here is not degenerate. Specifically:
   per LLM family, an omnibus Cochran's Q test of "is per-type accuracy
   uniform across all 7 failure types" (matched/paired design -- each
   sample answers all 7 types), plus a post-hoc per-type-vs-rest binomial
   test (the direct test of the "malformed_output is hardest" claim),
   Holm-corrected within each family across its 7 types.

2. **A full registry** of every hypothesis/comparison "tested" anywhere in
   this project (this script's own new tests, plus every existing
   simulation-based effect-size/CI result already on disk, plus the
   legacy pre-Task-5 harness.py p-values) -- assembled by READING already-
   written result files, never by re-deriving or re-simulating anything.
   Written as one machine-readable table (``hypothesis_registry.parquet``)
   plus a rendered Markdown table
   (``docs/statistics_registry.md``, written by hand alongside this
   script's output, not auto-generated prose).

Usage
-----
    python experiments/hypothesis_registry.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.human_study import score_submission  # noqa: E402
from recovery.analysis.significance import (  # noqa: E402
    cochrans_q_test,
    n_distinct_mappings,
    per_type_vs_rest_binomial_tests,
)
from recovery.envs.canonical import MATRIX, TYPE_NAMES  # noqa: E402
from recovery.metrics.bootstrap import bootstrap_ci  # noqa: E402
from src.evaluation.stats_utils import holm_bonferroni  # noqa: E402

OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"
LLM_DATA = OUTPUT_DIR / "llm_designers.parquet"

# A family's bootstrap CI is flagged degenerate when its samples collapse
# to this few (or fewer) distinct mappings -- a percentile bootstrap of a
# near-point-mass distribution cannot represent genuine sampling
# uncertainty. google-gemini's real n_distinct=2 is exactly the case this
# threshold is calibrated to catch; groq-llama (15/19) and
# nvidia-mistral-nemotron (6/21) are not degenerate by this rule.
DEGENERATE_DISTINCT_THRESHOLD = 3


def per_family_correctness_matrix(llm_df: pd.DataFrame, family: str, P: np.ndarray) -> tuple[np.ndarray, list]:
    """Re-derive the (n_ok_samples, 7) boolean correctness matrix for one
    family from ``llm_designers.parquet``'s raw ``parsed_mapping`` column
    via ``score_submission`` -- this per-sample binary matrix is not
    itself persisted anywhere (only the aggregated per-type ``accuracy``
    is, in ``llm_study_per_type_accuracy_by_family.parquet``), so it must
    be regenerated here, not re-derived from scratch (the scoring logic
    itself is 100% reused from Task 11's ``score_submission``)."""
    sub = llm_df[llm_df["family"] == family]
    ok = sub[sub["parse_status"] == "ok"]
    mappings = list(ok["parsed_mapping"])
    matrix = np.array([score_submission(m, P)["per_row_correct"] for m in mappings], dtype=bool)
    return matrix, mappings


def run_family_tests(llm_df: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    families = sorted(llm_df["family"].unique())
    all_rows = []
    for family in families:
        n_api_calls = int((llm_df["family"] == family).sum())
        matrix, mappings = per_family_correctness_matrix(llm_df, family, P)
        n_ok = matrix.shape[0]
        n_distinct = n_distinct_mappings(mappings)
        degenerate = n_distinct <= DEGENERATE_DISTINCT_THRESHOLD

        # --- Omnibus test: is per-type accuracy uniform within this family? ---
        omnibus = cochrans_q_test(matrix)

        # --- Post-hoc: per-type-vs-rest, Holm-corrected across this family's 7 types ---
        per_type = per_type_vs_rest_binomial_tests(matrix, TYPE_NAMES)
        raw_p = [row["p_value"] for row in per_type]
        corrected_p = holm_bonferroni(raw_p)

        for row, p_holm in zip(per_type, corrected_p):
            ci_lo, ci_hi = bootstrap_ci(matrix[:, TYPE_NAMES.index(row["type_name"])].astype(float))
            all_rows.append({
                "family": family,
                "failure_type": row["type_name"],
                "n_api_calls": n_api_calls,
                "n_parsed_ok": n_ok,
                "n_distinct_mappings": n_distinct,
                "bootstrap_ci_degenerate": degenerate,
                "accuracy": row["accuracy"],
                "accuracy_ci95_lower": ci_lo,
                "accuracy_ci95_upper": ci_hi,
                "pooled_rest_rate": row["pooled_rest_rate"],
                "test": "binomial (vs. pooled rest, two-sided)",
                "test_statistic_k_successes": row["k_successes"],
                "p_value_raw": row["p_value"],
                "p_value_holm": p_holm,
                "correction_method": "Holm-Bonferroni, within-family across 7 failure types",
                "effect_size_cohens_h": row["cohens_h"],
                "significant_after_correction": bool(p_holm < 0.05),
                "omnibus_cochrans_Q": omnibus["Q"],
                "omnibus_df": omnibus["df"],
                "omnibus_p_value": omnibus["p_value"],
                "omnibus_effect_size_v": omnibus["effect_size_v"],
                "omnibus_significant": bool(omnibus["p_value"] < 0.05) if not np.isnan(omnibus["p_value"]) else None,
            })
    return pd.DataFrame(all_rows)


def main() -> None:
    llm_df = pd.read_parquet(LLM_DATA)
    print(f"Loaded {len(llm_df)} LLM samples across {llm_df['family'].nunique()} families from {LLM_DATA}")

    results = run_family_tests(llm_df, MATRIX)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "hypothesis_registry_llm_per_type.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n=== Omnibus Cochran's Q per family (is per-type accuracy uniform?) ===")
    omnibus_view = results.drop_duplicates("family")[
        ["family", "n_api_calls", "n_parsed_ok", "n_distinct_mappings", "bootstrap_ci_degenerate",
         "omnibus_cochrans_Q", "omnibus_df", "omnibus_p_value", "omnibus_effect_size_v", "omnibus_significant"]
    ]
    print(omnibus_view.to_string(index=False))

    print("\n=== Per-type vs. rest, Holm-corrected within family (malformed_output row highlighted) ===")
    view_cols = ["family", "failure_type", "accuracy", "accuracy_ci95_lower", "accuracy_ci95_upper",
                 "pooled_rest_rate", "effect_size_cohens_h", "p_value_raw", "p_value_holm",
                 "significant_after_correction"]
    print(results[view_cols].to_string(index=False))

    print("\n=== malformed_output specifically, across all 3 families ===")
    mo = results[results["failure_type"] == "malformed_output"][
        ["family", "n_api_calls", "n_parsed_ok", "n_distinct_mappings", "bootstrap_ci_degenerate",
         "accuracy", "accuracy_ci95_lower", "accuracy_ci95_upper", "p_value_raw", "p_value_holm",
         "significant_after_correction"]
    ]
    print(mo.to_string(index=False))


if __name__ == "__main__":
    main()
