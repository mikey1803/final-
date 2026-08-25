#!/usr/bin/env python
"""Task 14 Part 2: token and monetary-cost accounting for the LLM
blind-designer study (Task 11), backfilled from ALREADY-LOGGED API
responses -- makes zero new API calls, per this task's explicit rule
("do NOT re-run paid calls purely to collect token counts").

Every row of ``results/tables/llm_designers.parquet`` already stores the
complete raw provider JSON response in its ``raw_response_json`` column
(``experiments/llm_designers.py::run_one_sample`` serializes the whole
response body, not just the parsed mapping) -- and every one of the 3
configured providers (Groq, Google, NVIDIA NIM) happens to return an
OpenAI-compatible ``usage: {prompt_tokens, completion_tokens,
total_tokens}`` block, confirmed by inspection of real logged responses
(not assumed from documentation). This script re-reads that field; it
does not call any provider API.

Known, documented backfill limitation (reported, not silently ignored):
for the 2 rows with ``n_attempts > 1`` (both ``groq-llama``,
``parse_status="parse_failure"``), ``run_one_sample`` only preserves the
LAST attempt's raw response -- every prior attempt was a real API call
that consumed real tokens, but its usage was never recorded. For these 2
rows, this script reports the recorded (last-attempt) token counts as an
explicit LOWER BOUND on the true total, flagged in the
``tokens_are_lower_bound_only`` column -- never silently presented as
the complete total.

Monetary cost: computed only where this script found and can cite a
real, current, official per-token price (see ``PRICING`` below and its
inline source citations, captured 2026-08-04 via WebFetch/WebSearch
against the vendors' own pricing pages, since this repo has no earlier
price capture and prices are not in the logged data). Where no
confirmed official price could be found (NVIDIA NIM, for the specific
model used), monetary cost is left unestimated and explicitly marked
``no_confirmed_price`` -- per this task's rule, not guessed at.

**All three providers' calls almost certainly consumed $0 of real
money**: this project's own `.env`/`conf/llm_designers.yaml` and prior
session notes (Groq's observed 6,000-tokens/minute rate limit, NVIDIA
NIM's documented "unlimited prototyping" free developer tier, Google's
free-tier Gemini API key usage) are all consistent with free/development
tiers, not metered billing. The pay-as-you-go cost figures below are
therefore a HYPOTHETICAL "what this would cost at published list rates,"
not a claim that money was actually spent -- stated explicitly in the
output, not left implicit.
"""

import json
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

DATA_PATH = _BACKEND_DIR / "results" / "tables" / "llm_designers.parquet"
OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"

# Per-million-token USD pricing, captured 2026-08-04 from the vendors' own
# pricing pages (see docs/experiments/instrumentation.md Part 2 for full
# citations and exact quotes). None of these were re-derived or guessed.
# "model_pinned" records whether the price is for the EXACT model string
# used, or an assumption about which dated model an alias resolved to.
PRICING = {
    "groq-llama": {
        "model_priced": "llama-3.1-8b-instant",
        "input_usd_per_1m": 0.05,
        "output_usd_per_1m": 0.08,
        "model_pinned": True,
        "source": "https://groq.com/pricing (fetched 2026-08-04)",
    },
    "google-gemini": {
        # The actual model string used is the ALIAS "gemini-flash-lite-latest";
        # Google's own API response echoes the alias back verbatim in
        # `model_version` (confirmed: every row's model_version ==
        # "gemini-flash-lite-latest"), so the underlying dated model
        # Google resolved it to is NOT disclosed by the data. Priced here
        # using Gemini 2.5 Flash-Lite's standard (non-batch) rate -- the
        # Flash-Lite tier Google's pricing page listed as current on the
        # collection date (2026-08-04); Gemini 3.5 Flash-Lite ($0.30/$2.50)
        # also existed at a higher price. This is a documented ASSUMPTION,
        # not a confirmed fact.
        "model_priced": "gemini-2.5-flash-lite (ASSUMED resolution of the 'latest' alias -- not confirmed)",
        "input_usd_per_1m": 0.10,
        "output_usd_per_1m": 0.40,
        "model_pinned": False,
        "source": "https://ai.google.dev/gemini-api/docs/pricing (fetched 2026-08-04, page states 'Last updated 2026-07-30 UTC')",
    },
    "nvidia-mistral-nemotron": {
        "model_priced": None,
        "input_usd_per_1m": None,
        "output_usd_per_1m": None,
        "model_pinned": False,
        "source": (
            "NO CONFIRMED OFFICIAL PRICE FOUND for 'mistralai/mistral-nemotron' on NVIDIA NIM "
            "as of 2026-08-04 (build.nvidia.com's per-model pricing page did not load; general "
            "NIM marketing pages describe a free 'unlimited prototyping' developer tier with no "
            "disclosed per-token rate). Per this task's rule ('where backfill is impossible, say "
            "so rather than estimating'), monetary cost for this family is left unestimated."
        ),
    },
}


def extract_usage(raw_response_json: str) -> dict:
    """Pull the OpenAI-compatible ``usage`` block out of one logged raw
    response. Returns a dict of NaNs (never zeros -- zero would silently
    imply "measured zero cost") if the field is missing."""
    if raw_response_json is None:
        return {"prompt_tokens": float("nan"), "completion_tokens": float("nan"), "total_tokens": float("nan")}
    body = json.loads(raw_response_json)
    usage = body.get("usage")
    if usage is None:
        return {"prompt_tokens": float("nan"), "completion_tokens": float("nan"), "total_tokens": float("nan")}
    return {
        "prompt_tokens": usage.get("prompt_tokens", float("nan")),
        "completion_tokens": usage.get("completion_tokens", float("nan")),
        "total_tokens": usage.get("total_tokens", float("nan")),
    }


def estimate_cost_usd(family: str, prompt_tokens: float, completion_tokens: float) -> float:
    price = PRICING.get(family, {})
    if price.get("input_usd_per_1m") is None or price.get("output_usd_per_1m") is None:
        return float("nan")
    return (prompt_tokens / 1e6) * price["input_usd_per_1m"] + (completion_tokens / 1e6) * price["output_usd_per_1m"]


def backfill(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        usage = extract_usage(r["raw_response_json"])
        cost = estimate_cost_usd(r["family"], usage["prompt_tokens"], usage["completion_tokens"])
        rows.append({
            "family": r["family"],
            "sample_index": r["sample_index"],
            "temperature": r["temperature"],
            "parse_status": r["parse_status"],
            "n_attempts": r["n_attempts"],
            "tokens_are_lower_bound_only": bool(r["n_attempts"] > 1),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd_pay_as_you_go": cost,
            "price_source": PRICING.get(r["family"], {}).get("source"),
        })
    return pd.DataFrame(rows)


def summarize(per_sample: pd.DataFrame) -> pd.DataFrame:
    agg = per_sample.groupby("family").agg(
        n_samples=("sample_index", "count"),
        n_lower_bound_only=("tokens_are_lower_bound_only", "sum"),
        total_prompt_tokens=("prompt_tokens", "sum"),
        total_completion_tokens=("completion_tokens", "sum"),
        total_tokens=("total_tokens", "sum"),
        # min_count=1 is required so a family with an ALL-NaN cost column
        # (nvidia-mistral-nemotron, no confirmed price) sums to NaN, not
        # pandas' default empty-sum-is-0.0 -- 0.0 would misleadingly read
        # as "confirmed zero cost" rather than "price unknown."
        total_estimated_cost_usd=("estimated_cost_usd_pay_as_you_go", lambda s: s.sum(min_count=1)),
    ).reset_index()
    return agg


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} logged LLM samples from {DATA_PATH}")

    per_sample = backfill(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_sample_path = OUTPUT_DIR / "llm_cost_accounting_per_sample.parquet"
    per_sample.to_parquet(per_sample_path, index=False)
    print(f"Wrote {per_sample_path}")

    summary = summarize(per_sample)
    summary_path = OUTPUT_DIR / "llm_cost_accounting_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    print(f"Wrote {summary_path}\n")

    print("=== Per-family token/cost summary (backfilled from logged responses, zero new API calls) ===")
    print(summary.to_string(index=False))

    grand_total_tokens = summary["total_tokens"].sum()
    grand_total_cost = summary["total_estimated_cost_usd"].sum(skipna=True)
    n_priced_families = summary["family"].isin(
        [f for f, p in PRICING.items() if p.get("input_usd_per_1m") is not None]
    ).sum()
    print(f"\nGrand total tokens (all 3 families, 63 API calls): {grand_total_tokens:,.0f}")
    print(f"Grand total estimated pay-as-you-go cost ({n_priced_families}/3 families priced; "
          f"nvidia-mistral-nemotron excluded, no confirmed price): ${grand_total_cost:.4f}")
    print("\nNOTE: this is a HYPOTHETICAL list-price figure. The actual API keys used for this "
          "study were free/developer-tier credentials (see module docstring) -- real money spent "
          "was almost certainly $0. Do not present the figure above as actual incurred cost.")

    n_lower_bound = int(per_sample["tokens_are_lower_bound_only"].sum())
    if n_lower_bound:
        print(f"\nWARNING: {n_lower_bound} sample(s) have n_attempts > 1 -- their token counts are "
              "the LAST attempt only, a LOWER BOUND on true total tokens spent for that sample "
              "(earlier failed attempts consumed real tokens that were never logged). See "
              "'tokens_are_lower_bound_only' column.")


if __name__ == "__main__":
    main()
