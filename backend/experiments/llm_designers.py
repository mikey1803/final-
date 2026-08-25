#!/usr/bin/env python
"""Task 11: run the IDENTICAL blind-design protocol against LLM designers.

Models are configured entirely via ``conf/llm_designers.yaml`` -- no
model name is hardcoded anywhere in this file. For each configured model
this script draws ``n_samples_temperature_1`` independent samples at
``temperature=1.0`` and ``n_samples_temperature_0`` samples at
``temperature=0.0`` (defaults: 20 and 1, matching Task 11's
specification; overridable per model in the YAML, and reducible via
``--validation`` for a cheap smoke run -- see "Validation vs. full runs"
below).

The prompt is constructed from ``recovery.envs.descriptions.build_task_instructions()``
-- the EXACT SAME function the study web app's FastAPI service serves to
human participants -- plus a short, separate response-format instruction
appended after it. ``tests/test_study.py`` asserts the shared
instructions appear in the constructed prompt VERBATIM (string equality
against the shared module, not a paraphrase), which is what makes the
two arms' task presentation comparable.

Parse failures are NEVER silently replaced with a default/heuristic
mapping -- this project's own paper already documents a silent-fallback
hazard in its One-Shot LLM baseline (``src/evaluation/*``) and this
script must not repeat it. A parse failure after
``max_parse_retries`` attempts is recorded as ``parse_status="parse_failure"``,
``parsed_mapping=None``, and the run continues to the next sample.

A read TIMEOUT on an individual API call is retried within the same
``max_parse_retries`` attempt budget (observed on NVIDIA NIM's free
tier: real responses completing in ~2s interleaved with calls that never
return within 60s -- consistent with serverless cold-start queueing) --
unlike other network/HTTP errors (auth failures, deprecated models,
quota exhaustion), which are deterministic and still propagate
immediately rather than wasting retries on a guaranteed-to-fail call. If
every attempt times out, the sample is recorded as
``parse_status="network_timeout"`` -- distinct from ``"parse_failure"``,
because the model never got a chance to answer at all, which is a
different (and less informative) failure mode than answering
unparseably.

Validation vs. full runs
--------------------------
This script can only QUERY a model family whose API key
(``models[].api_key_env`` in the YAML) is present in the environment.
Every other configured family is SKIPPED, explicitly logged as skipped,
and recorded as such in the run summary -- never silently omitted, never
backfilled with fabricated rows. See ``docs/experiments/human_study.md``
for exactly which families were queryable, and with how many samples,
in the run whose results that document reports. ``--validation`` caps
every model's sample counts at a small number (see ``VALIDATION_N_TEMP1``/
``VALIDATION_N_TEMP0`` below) to keep a smoke-test run cheap and fast
while still making REAL API calls -- it is not a dry run, and its output
must never be presented as the full-paper run. ``--dry-run`` makes zero
API calls at all (for testing this script's own plumbing) and refuses to
write to the default output path, to make it structurally difficult to
mistake dry-run output for real data.

Usage
-----
    python experiments/llm_designers.py                         # full run, all keyed families
    python experiments/llm_designers.py --validation             # small real run (few samples/model)
    python experiments/llm_designers.py --dry-run --output /tmp/x.parquet   # no API calls, plumbing only
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.envs.descriptions import STRATEGY_NAMES, TYPE_NAMES, build_task_instructions  # noqa: E402
from recovery.envs.paraphrases import PARAPHRASE_BUILDERS  # noqa: E402 -- Task 15

# Load API keys from backend/.env, matching src.agent.llm_client's
# existing convention (the only precedent for env-based config in this
# repo -- see study/protocol.md's "no YAML config convention existed"
# note, which is why model SELECTION is YAML-driven but key LOADING
# follows this project's established .env pattern instead).
from dotenv import load_dotenv  # noqa: E402

_env_path = _BACKEND_DIR / ".env"
load_dotenv(_env_path if _env_path.exists() else _BACKEND_DIR / ".env.example")

DEFAULT_CONFIG = _BACKEND_DIR / "conf" / "llm_designers.yaml"
DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "llm_designers.parquet"

VALIDATION_N_TEMP1 = 3
VALIDATION_N_TEMP0 = 1

RESPONSE_FORMAT_INSTRUCTIONS = (
    "\n\nRespond with ONLY a JSON object mapping each failure type's exact name to your "
    "chosen strategy's exact name, and nothing else -- no explanation, no markdown code "
    "fence, no extra text. Example shape (illustrative names only):\n"
    '{"tool_timeout": "retry", "tool_error": "switch_tool", ...}'
)


PROMPT_VARIANTS = {"primary": build_task_instructions, **PARAPHRASE_BUILDERS}


def build_prompt(variant: str = "primary") -> str:
    """The exact prompt sent to every LLM designer: the shared
    instructions block (byte-identical to what the study web app serves
    to human participants, for ``variant="primary"``) plus a
    response-format instruction appended AFTER it -- never interleaved, so
    the shared block remains a verbatim, independently-checkable substring
    of the full prompt.

    ``variant`` : one of ``PROMPT_VARIANTS`` (default ``"primary"``, the
    Task 11 headline instrument -- unchanged, byte-identical behaviour to
    before this parameter existed). ``"paraphrase_a"``/``"paraphrase_b"``
    are Task 15's supplementary, differently-worded instruments
    (``recovery.envs.paraphrases``) -- SAME task, SAME response-format
    requirement, different prose only.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[variant]() + RESPONSE_FORMAT_INSTRUCTIONS


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_mapping_response(text: str) -> Optional[list[int]]:
    """Parse an LLM's raw text response into a canonical-order mapping
    (list of ``len(TYPE_NAMES)`` strategy indices). Returns ``None`` on
    ANY failure (bad JSON, missing type, unknown strategy name, wrong
    value type) -- callers must never substitute a default on ``None``.
    """
    if text is None:
        return None
    candidate = text.strip()
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    mapping = [None] * len(TYPE_NAMES)
    for t_idx, type_name in enumerate(TYPE_NAMES):
        if type_name not in obj:
            return None
        strategy_name = obj[type_name]
        if not isinstance(strategy_name, str) or strategy_name not in STRATEGY_NAMES:
            return None
        mapping[t_idx] = STRATEGY_NAMES.index(strategy_name)
    return mapping


# ======================================================================
# Provider dispatch -- two generic HTTP shapes, no per-model-name branching.
# ======================================================================


def _call_openai_compatible(base_url, api_key, model, prompt, temperature, max_tokens, seed, timeout) -> dict:
    """Covers Groq, OpenAI, Together, Fireworks, OpenRouter, DeepSeek,
    Mistral, and any other provider implementing the standard
    chat-completions request/response shape."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    model_version = body.get("model", model)
    return {"text": text, "model_version": model_version, "raw_json": body}


def _call_anthropic(base_url, api_key, model, prompt, temperature, max_tokens, timeout) -> dict:
    """Claude's native API -- a distinct request/response shape (no
    ``seed`` parameter is supported by this API, so ``seed`` is always
    ``None`` for this provider_type, recorded honestly as such)."""
    resp = requests.post(
        f"{base_url}/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    text = "".join(block.get("text", "") for block in body.get("content", []))
    model_version = body.get("model", model)
    return {"text": text, "model_version": model_version, "raw_json": body}


def call_model(provider_type: str, base_url: str, api_key: str, model: str, prompt: str, temperature: float,
                max_tokens: int, seed: Optional[int], timeout: float) -> dict:
    if provider_type == "openai_compatible":
        return _call_openai_compatible(base_url, api_key, model, prompt, temperature, max_tokens, seed, timeout)
    if provider_type == "anthropic":
        return _call_anthropic(base_url, api_key, model, prompt, temperature, max_tokens, timeout)
    raise ValueError(f"unknown provider_type {provider_type!r}")


_RESET_DURATION_RE = re.compile(r"(?:(?P<min>\d+)m)?(?P<sec>\d+(?:\.\d+)?)(?P<unit>ms|s)?$")


def _parse_rate_limit_wait_seconds(response, default: float = 10.0, cap: float = 65.0) -> float:
    """Best-effort wait time before retrying a 429, from whichever header
    the provider actually sent -- standard ``Retry-After`` (seconds) if
    present, else Groq's ``x-ratelimit-reset-tokens``/``-requests``
    (format ``"6s"``/``"1m42s"``/``"570ms"``, observed directly against
    Groq's real API), else ``default``. The wait is always capped at
    ``cap`` seconds -- a malformed or absurdly large header value must
    never hang the run indefinitely."""
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(float(retry_after), cap)
        except ValueError:
            pass
    for key in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if not raw:
            continue
        m = _RESET_DURATION_RE.match(raw.strip())
        if not m:
            continue
        minutes = float(m.group("min") or 0)
        seconds = float(m.group("sec"))
        if m.group("unit") == "ms":
            seconds /= 1000.0
        return min(minutes * 60 + seconds, cap)
    return min(default, cap)


# ======================================================================
# Run loop
# ======================================================================


def run_one_sample(
    *, family: str, provider_type: str, base_url: str, api_key: str, model: str, prompt: str, temperature: float,
    max_tokens: int, seed: Optional[int], timeout: float, max_retries: int, sample_index: int,
    supports_seed: bool = True,
) -> dict:
    """Run one (model, temperature, sample_index) draw, retrying on parse
    failure up to ``max_retries`` times. Returns one result row -- never
    raises for a parse failure (network/API errors DO propagate, since
    those are infrastructure failures, not "the model answered badly").

    ``supports_seed`` : bool, default True
        Some ``openai_compatible`` providers reject an unrecognised
        ``seed`` field outright (e.g. Google's Gemini OpenAI-compatible
        endpoint returns HTTP 400 "Unknown name \"seed\"") rather than
        silently ignoring it, unlike Groq/OpenAI. When ``False``, the
        actual API call omits ``seed`` entirely, and the recorded
        ``seed`` field in the output row is ``None`` -- the same
        "record what was actually used, never a fabricated intended
        value" rule already applied to the Anthropic provider type.
    """
    effective_seed = seed if supports_seed else None
    last_raw_text = None
    last_model_version = None
    last_raw_json = None
    parsed = None
    attempts_used = 0
    got_any_response = False  # distinguishes "model answered but didn't parse" from "never got a response"
    last_no_response_reason = None  # "timeout" or "rate_limited" -- which kind of no-response, if any

    for attempt in range(1, max_retries + 1):
        attempts_used = attempt
        try:
            result = call_model(
                provider_type, base_url, api_key, model, prompt, temperature, max_tokens, effective_seed, timeout
            )
        except requests.exceptions.Timeout:
            # A read timeout on a free/shared inference endpoint (observed
            # repeatedly on NVIDIA NIM's free tier: real request/response
            # pairs completing in ~2s, interleaved with requests that never
            # return within 60s -- consistent with serverless cold-start
            # queueing, not a persistent outage) is treated as TRANSIENT and
            # retried within this same attempt budget, unlike other
            # RequestException subclasses (HTTP errors, connection refused,
            # DNS failure), which are deterministic and still propagate
            # immediately -- retrying those would only waste the attempt
            # budget on a guaranteed-to-fail call.
            last_no_response_reason = "timeout"
            continue
        except requests.exceptions.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise  # a genuine deterministic error (401/404/410/5xx/...) -- never silently retried
            # A 429 with real, low per-minute token budgets (observed
            # directly against Groq: 6000 tokens/min, and this study's
            # ~500-token prompt makes that budget bind well before 20
            # sequential samples complete) is transient and load-dependent
            # -- wait out the provider's OWN reported reset window (never a
            # guessed/fixed sleep) and retry within the same attempt budget.
            wait_s = _parse_rate_limit_wait_seconds(exc.response)
            print(f"    [RATE LIMIT] family={family!r} attempt={attempt}: 429, waiting {wait_s:.1f}s before retry")
            time.sleep(wait_s)
            last_no_response_reason = "rate_limited"
            continue
        got_any_response = True
        last_raw_text = result["text"]
        last_model_version = result["model_version"]
        last_raw_json = result["raw_json"]
        parsed = parse_mapping_response(last_raw_text)
        if parsed is not None:
            break

    if parsed is not None:
        status = "ok"
    elif not got_any_response:
        # every attempt failed before the model ever got a chance to answer
        status = "rate_limited" if last_no_response_reason == "rate_limited" else "network_timeout"
    else:
        status = "parse_failure"  # the model answered at least once; none of the answers parsed

    return {
        "family": family,
        "model": model,
        "model_version": last_model_version,
        "provider_type": provider_type,
        "temperature": temperature,
        "seed": effective_seed,
        "max_tokens": max_tokens,
        "sample_index": sample_index,
        "prompt_hash": prompt_hash(prompt),
        "raw_response": last_raw_text,
        "raw_response_json": json.dumps(last_raw_json) if last_raw_json is not None else None,
        "parsed_mapping": parsed,
        "parse_status": status,
        "n_attempts": attempts_used,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_model_config(
    cfg: dict, prompt: str, max_retries: int, timeout: float, validation: bool,
    n_temp1_override: Optional[int] = None, n_temp0_override: Optional[int] = None,
) -> list[dict]:
    family = cfg["family"]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        print(f"  [SKIPPED] family={family!r}: environment variable {cfg['api_key_env']!r} is not set")
        return []

    if n_temp1_override is not None:
        n_temp1 = n_temp1_override
    else:
        n_temp1 = VALIDATION_N_TEMP1 if validation else cfg.get("n_samples_temperature_1", 20)
    if n_temp0_override is not None:
        n_temp0 = n_temp0_override
    else:
        n_temp0 = VALIDATION_N_TEMP0 if validation else cfg.get("n_samples_temperature_0", 1)
    supports_seed = cfg.get("supports_seed", True)

    rows = []
    print(f"  [RUNNING] family={family!r} model={cfg['model']!r} n_temp1={n_temp1} n_temp0={n_temp0}")
    for i in range(n_temp1):
        rows.append(
            run_one_sample(
                family=family, provider_type=cfg["provider_type"], base_url=cfg["base_url"], api_key=api_key,
                model=cfg["model"], prompt=prompt, temperature=1.0, max_tokens=cfg.get("max_tokens", 512),
                seed=i, timeout=timeout, max_retries=max_retries, sample_index=i, supports_seed=supports_seed,
            )
        )
    for i in range(n_temp0):
        rows.append(
            run_one_sample(
                family=family, provider_type=cfg["provider_type"], base_url=cfg["base_url"], api_key=api_key,
                model=cfg["model"], prompt=prompt, temperature=0.0, max_tokens=cfg.get("max_tokens", 512),
                seed=0, timeout=timeout, max_retries=max_retries, sample_index=i, supports_seed=supports_seed,
            )
        )
    n_ok = sum(1 for r in rows if r["parse_status"] == "ok")
    print(f"  [DONE] family={family!r}: {n_ok}/{len(rows)} parsed successfully")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation", action="store_true", help="Small REAL run (few samples/model), not a dry run.")
    parser.add_argument("--dry-run", action="store_true", help="No API calls at all; refuses the default --output path.")
    parser.add_argument(
        "--families", type=str, default=None,
        help="Comma-separated subset of configured family names to run (default: all). Rows for families NOT "
             "in this subset are preserved from any existing --output file rather than dropped -- lets a "
             "transient provider-side failure (e.g. one family's rate limit) be retried in isolation without "
             "re-querying (and re-spending quota on) families that already succeeded.",
    )
    parser.add_argument(
        "--prompt-variant", type=str, default="primary", choices=sorted(PROMPT_VARIANTS),
        help="Task 15: which instrument wording to send. 'primary' (default) is the unchanged Task 11 "
             "headline instrument. 'paraphrase_a'/'paraphrase_b' are supplementary, differently-worded "
             "instruments used ONLY for the prompt-robustness experiment -- never write these to the "
             "default --output path (results/tables/llm_designers.parquet); use a separate --output file.",
    )
    parser.add_argument(
        "--n-temp1", type=int, default=None,
        help="Task 15: override n_samples_temperature_1 for every configured family this run (default: "
             "each model's conf/*.yaml value, or --validation's small constant).",
    )
    parser.add_argument(
        "--n-temp0", type=int, default=None,
        help="Task 15: override n_samples_temperature_0 for every configured family this run.",
    )
    args = parser.parse_args()

    if args.prompt_variant != "primary" and args.output == DEFAULT_OUTPUT:
        parser.error(
            f"--prompt-variant={args.prompt_variant!r} refuses to write to the default --output path "
            f"({DEFAULT_OUTPUT}); pass an explicit --output so the primary Task 11 results are never "
            "overwritten by a supplementary paraphrase run"
        )

    if args.dry_run and args.output == DEFAULT_OUTPUT:
        parser.error("--dry-run refuses to write to the default --output path; pass an explicit path for dry-run output")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.families is not None:
        requested = {f.strip() for f in args.families.split(",") if f.strip()}
        configured = {m["family"] for m in config["models"]}
        unknown = requested - configured
        if unknown:
            parser.error(f"--families names not present in {args.config}: {sorted(unknown)}")
        config["models"] = [m for m in config["models"] if m["family"] in requested]

    prompt = build_prompt(args.prompt_variant)
    if args.prompt_variant == "primary":
        assert build_task_instructions() in prompt, "shared instructions block must be a verbatim substring of the prompt"
    print(f"Prompt variant: {args.prompt_variant!r}")
    print(f"Prompt hash: {prompt_hash(prompt)}")
    print(f"Mode: {'DRY RUN (no API calls)' if args.dry_run else ('VALIDATION' if args.validation else 'FULL')}")

    all_rows: list[dict] = []
    skipped_families: list[str] = []  # no API key set -- see run_model_config
    errored_families: list[tuple[str, str]] = []  # infrastructure failure mid-run (family, error message)
    t0 = time.time()

    for model_cfg in config["models"]:
        if args.dry_run:
            api_key_present = bool(os.environ.get(model_cfg["api_key_env"]))
            print(f"  [DRY RUN] family={model_cfg['family']!r}: would run (api_key_present={api_key_present})")
            continue
        try:
            rows = run_model_config(
                model_cfg, prompt, config.get("max_parse_retries", 3), config.get("request_timeout_seconds", 60),
                args.validation, n_temp1_override=args.n_temp1, n_temp0_override=args.n_temp0,
            )
        except requests.exceptions.RequestException as exc:
            # An infrastructure failure (network error, rate limit, auth
            # error, etc.) for ONE family must not silently end the run
            # for every OTHER configured family -- but it is also never
            # swallowed: it is loudly printed and recorded, distinctly
            # from "skipped, no API key", so the run summary can never be
            # misread as "this family was queried and had zero results."
            family = model_cfg["family"]
            print(f"  [ERROR] family={family!r}: infrastructure failure, aborting this family only: {exc}")
            errored_families.append((family, str(exc)))
            continue
        if not rows:
            skipped_families.append(model_cfg["family"])
        all_rows.extend(rows)

    elapsed = time.time() - t0

    if args.dry_run:
        print("Dry run complete -- no output written (no data to write).")
        return

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["parsed_mapping"] = df["parsed_mapping"].apply(lambda m: list(m) if m is not None else None)
        # Task 15: self-describing even out of context. Existing rows already
        # on disk (written before this column existed) are left as-is by the
        # merge below -- their prompt_variant reads NaN, not a fabricated
        # "primary" backfill of a fact this script cannot actually confirm
        # about data it didn't just produce.
        df["prompt_variant"] = args.prompt_variant

    # Merge with any existing output rather than overwrite wholesale: only
    # replace rows for families that produced NEW rows THIS run. A family
    # that errored or was filtered out via --families keeps whatever real
    # data it already had on disk -- never silently deleted by an unrelated
    # partial re-run.
    families_this_run = set(df["family"].unique()) if not df.empty else set()
    if args.output.exists():
        existing = pd.read_parquet(args.output)
        preserved = existing[~existing["family"].isin(families_this_run)]
        df = pd.concat([preserved, df], ignore_index=True) if not preserved.empty else df

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df)} rows to {args.output} in {elapsed:.1f}s ({len(all_rows)} new this run)")
    print(f"Families skipped (no API key): {skipped_families or 'none'}")
    print(f"Families errored (infrastructure failure, NOT run to completion): {errored_families or 'none'}")
    if not df.empty:
        print(df.groupby("family")["parse_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
