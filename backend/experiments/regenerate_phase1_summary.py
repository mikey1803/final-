#!/usr/bin/env python
"""Regenerate results/phase1/ (Phase 1 baseline comparison, 500 episodes x 50
trials) -- Task 20 step 6.

Why this needs care
-------------------
results/phase1/*.csv were silently overwritten with n=2-trial smoke numbers
by an old test (fixed in 4d0f730). Re-running the legacy harness
(src/evaluation/harness.py, Layer 1, read-only) naively would NOT reproduce
the paper's rows, for two reasons found on 2026-10-21:

1. The One-Shot LLM baseline is unusable today: llama-3.1-8b-instant is gone
   from Groq and the Layer 1 client's stopgap (openai/gpt-oss-20b at
   max_tokens=20) returns empty replies 7/7, which _parse_strategy silently
   turns into RETRY.
2. All 11 methods share ONE outcome RNG per trial (harness.run_comparison:
   outcome_rng), and SWITCH_TOOL consumes an extra draw (r.choice of an
   alternative tool). So One-Shot's choices shift the random sequence seen by
   every later method and episode: with any other One-Shot choices, 0/11
   methods reproduce.

Fix, without touching Layer 1: pin One-Shot to its ORIGINAL 2026-07-28 choices
by pre-filling OneShotLLMBaseline's class-level response cache (the same
mechanism the baseline uses itself). The choices were recovered from the
original raw traces (data/phase4/evaluation_results.csv): per-type episode
costs are constant (retry 1.0 / verify 1.5 / switch_tool-or-clarify 2.0), and
the 2^3 switch_tool/clarify ambiguity was resolved by bitwise replay of trial
0 -- exactly one combination reproduces all 11 methods. No API call is made
(GROQ_API_KEY is forced empty; the cache answers every lookup).

Per the Task 20 decision, the One-Shot LLM row itself is reported with its
original 2026-07-28 numbers (provenance: results/phase4/comparison_summary.csv,
the same run), carrying the llama-3.1-8b-instant deprecation note. A
provenance sidecar JSON is written next to the CSV.

    python experiments/regenerate_phase1_summary.py            # full: 500 x 50 -> results/phase1/
    python experiments/regenerate_phase1_summary.py --smoke    # 50 x 2 -> results/phase1_smoke/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["GROQ_API_KEY"] = ""  # before any Layer 1 import loads .env: no live LLM calls, ever

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import harness  # noqa: E402
from src.evaluation.baselines import OneShotLLMBaseline  # noqa: E402
from src.failures.taxonomy import FailureType  # noqa: E402
from src.recovery.strategies import RecoveryStrategy  # noqa: E402

RESULTS_DIR = _BACKEND_DIR / "results"
ORIGINAL_SUMMARY = RESULTS_DIR / "phase4" / "comparison_summary.csv"   # the 2026-07-28 run
ONE_SHOT = "One-Shot LLM"

# Recovered 2026-10-21 from data/phase4/evaluation_results.csv (see module docstring).
ORIGINAL_ONE_SHOT_CHOICES: dict[FailureType, RecoveryStrategy] = {
    FailureType.TOOL_TIMEOUT: RecoveryStrategy.RETRY,
    FailureType.TOOL_ERROR: RecoveryStrategy.SWITCH_TOOL,
    FailureType.WRONG_TOOL_SELECTED: RecoveryStrategy.SWITCH_TOOL,
    FailureType.HALLUCINATED_OUTPUT: RecoveryStrategy.VERIFY,
    FailureType.MALFORMED_OUTPUT: RecoveryStrategy.VERIFY,
    FailureType.CONTEXT_LOSS: RecoveryStrategy.VERIFY,
    FailureType.AMBIGUOUS_INSTRUCTION: RecoveryStrategy.SWITCH_TOOL,
}


def pin_one_shot_choices() -> None:
    OneShotLLMBaseline._REPRESENTATIVE_ERRORS.clear()
    OneShotLLMBaseline._ensure_representative_errors()
    OneShotLLMBaseline._RESPONSE_CACHE.clear()
    for ft, strategy in ORIGINAL_ONE_SHOT_CHOICES.items():
        OneShotLLMBaseline._RESPONSE_CACHE[OneShotLLMBaseline._REPRESENTATIVE_ERRORS[ft]] = strategy
    OneShotLLMBaseline.used_real_llm = False


def compare(regen: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Per method: max absolute difference across all numeric columns."""
    rows = []
    for method in original.index:
        a, b = regen.loc[method], original.loc[method]
        cols = [c for c in original.columns if c in regen.columns]
        diff = np.nanmax(np.abs(a[cols].astype(float).to_numpy() - b[cols].astype(float).to_numpy()))
        rows.append({"method": method, "max_abs_diff": float(diff)})
    return pd.DataFrame(rows).set_index("method")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="50 episodes x 2 trials -> results/phase1_smoke/")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    n_ep, n_tr = (50, 2) if args.smoke else (500, 50)
    out_dir = args.output_dir or RESULTS_DIR / ("phase1_smoke" if args.smoke else "phase1")
    out_dir.mkdir(parents=True, exist_ok=True)

    pin_one_shot_choices()
    with tempfile.TemporaryDirectory() as scratch:  # harness data_dir (raw CSV, plots): never backend/data
        harness.run_and_report(num_episodes=n_ep, num_trials=n_tr, data_dir=Path(scratch),
                               results_dir=out_dir)
    if OneShotLLMBaseline.used_real_llm:
        raise SystemExit("FATAL: a live LLM call happened; the pinned cache should have answered every lookup")

    summary_path = out_dir / "comparison_summary.csv"
    regen = pd.read_csv(summary_path, index_col=0)
    provenance = {
        "regenerated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "backend/experiments/regenerate_phase1_summary.py",
        "num_episodes": n_ep, "num_trials": n_tr, "base_seed": 123,
        "one_shot_choices_pinned": {ft.value: s.value for ft, s in ORIGINAL_ONE_SHOT_CHOICES.items()},
        "one_shot_choices_source": "recovered from data/phase4/evaluation_results.csv (2026-07-28 raw traces)",
        "live_llm_calls": 0,
    }

    if not args.smoke:
        original = pd.read_csv(ORIGINAL_SUMMARY, index_col=0)
        diffs = compare(regen, original)
        print("\nRegenerated vs original 2026-07-28 run (results/phase4/comparison_summary.csv):")
        print(diffs.to_string(float_format=lambda v: f"{v:.3e}"))
        provenance["max_abs_diff_vs_original_by_method"] = diffs["max_abs_diff"].to_dict()
        # Task 20 decision: the One-Shot LLM row keeps its original numbers.
        regen.loc[ONE_SHOT] = original.loc[ONE_SHOT, regen.columns]
        regen.to_csv(summary_path)
        shutil.copy2(ORIGINAL_SUMMARY, out_dir / "comparison_summary_original_2026-07-28.csv")
        provenance["one_shot_row"] = {
            "values": "original 2026-07-28 numbers, copied from results/phase4/comparison_summary.csv",
            "model": "llama-3.1-8b-instant (Groq), used_real_llm=True at collection time",
            "deprecation_note": "llama-3.1-8b-instant was removed from Groq (found 2026-08-20); the row "
                                "cannot be regenerated with a live call. See docs/PAPER_PREP.md sec. 6.",
        }

    (out_dir / "comparison_summary.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                                 encoding="utf-8")
    print(f"\nWrote {summary_path} and comparison_summary.provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
