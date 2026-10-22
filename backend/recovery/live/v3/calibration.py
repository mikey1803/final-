"""Offline false-alarm calibration for VERIFY (pre-registration §6, decision 2026-10-21).

After collection, a seeded sample of delivered bodies of NON-degraded calls
from ``degraded_output`` blocks -- bodies the agent already received during
collection -- goes through VERIFY's v2 pipeline once each (schema check, then
the LLM judge, no retries). That gives each field's false-alarm rate for the
schema check alone and for the full pipeline. Detection is claimed for a
field x transformation family only when its detection rate exceeds that
false-alarm rate by the pre-set margin (``M_DET``, configuration choice).

Eligible items: D blocks, ``degraded`` false, ``success`` true, not a resend
probe, delivered body stored, session not quarantined. Sample: 50 per field,
without replacement, ``FA_SAMPLE_SEED = 23002``, one
``SeedSequence(23002).spawn`` child per field in ``FIELD_ORDER``; items are
first put in a fixed order (session seed, block, call id) so the draw is
reproducible. Fewer than 50 eligible: all are used and n is reported.

Settings are those used in collection and are all recorded: requested model
id, reasoning effort, temperature, judge max tokens, the judge input (the v3
answer-field-first input of D-4, as in collection), the judge prompt
template's sha256, and the provider-reported model id per item.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Optional

import numpy as np

from recovery.live.executors.base import LLMError
from recovery.live.executors.prompts import parse_judgement
from recovery.live.executors.schemas import check_output_schema
from recovery.live.executors.verify import VerifyExecutor
from recovery.live.tools import ToolCallResult
from recovery.live.v3 import verify_v3
from recovery.live.v3.verify_v3 import JUDGE_INPUT, judge_prompt_v3

__all__ = ["FA_SAMPLE_SEED", "FA_PER_FIELD", "FIELD_ORDER", "M_DET", "eligible_items", "draw_sample",
           "judge_settings", "judge_sample"]

FA_SAMPLE_SEED = 23002
FA_PER_FIELD = 50
M_DET = 0.15  # configuration choice (pre-reg §6)
FIELD_ORDER = ("countriesnow", "wikipedia", "open_meteo", "wttr", "jsonplaceholder", "dummyjson")


def _quarantined(root: Path) -> set:
    d = root / "v3_quarantine"
    return {int(json.loads(p.read_text(encoding="utf-8"))["session_seed"]) for p in d.glob("*.json")} if d.exists() else set()


def eligible_items(root: Path) -> dict[str, list[dict]]:
    root = Path(root)
    q = _quarantined(root)
    out: dict[str, list[dict]] = {f: [] for f in FIELD_ORDER}
    d = root / "v3_injection_logs"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if (e.get("mode") == "degraded_output" and not e.get("degraded") and e.get("success")
                    and not e.get("probe") and e.get("actor") != "resend_probe"
                    and e.get("delivered_text") and e.get("tool") in out
                    and int(e["session_seed"]) not in q):
                out[e["tool"]].append(e)
    for f in out:
        out[f].sort(key=lambda e: (int(e["session_seed"]), int(e["block_index"]), int(e["call_id"])))
    return out


def draw_sample(items: dict[str, list[dict]], per_field: int = FA_PER_FIELD) -> dict[str, list[dict]]:
    children = np.random.SeedSequence(FA_SAMPLE_SEED).spawn(len(FIELD_ORDER))
    sample = {}
    for field, child in zip(FIELD_ORDER, children):
        pool = items.get(field, [])
        n = min(per_field, len(pool))
        idx = sorted(np.random.default_rng(child).choice(len(pool), size=n, replace=False).tolist()) if n else []
        sample[field] = [pool[i] for i in idx]
    return sample


def judge_settings(llm, judge_max_tokens: Optional[int] = None) -> dict:
    template = "".join(inspect.getsource(f) for f in (verify_v3.answer_field_label, verify_v3.answer_field_text,
                                                       judge_prompt_v3))
    return {
        "requested_model_id": getattr(llm, "model", None), "reasoning_effort": getattr(llm, "reasoning_effort", None),
        "temperature": 0.0, "judge_max_tokens": judge_max_tokens or VerifyExecutor().judge_max_tokens,
        "judge_prompt_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "schema_check": True, "llm_judge": True, "judged_once": True, "judge_input": JUDGE_INPUT,
    }


def judge_sample(sample: dict[str, list[dict]], llm) -> dict:
    """Each item judged once: schema check, then (if it passes) the LLM judge.
    An item whose judge call fails is excluded and counted (``llm_error``)."""
    settings = judge_settings(llm)
    records, summary = [], {}
    for field in FIELD_ORDER:
        n = schema_rej = pipe_rej = llm_err = 0
        for e in sample.get(field, []):
            text = e["delivered_text"]
            res = ToolCallResult(True, e.get("http_status"), text, None, 0.0, full_text=text)
            ok, why = check_output_schema(field, res)
            rec = {"field": field, "call_id": e["call_id"], "session_seed": e["session_seed"],
                   "block_index": e["block_index"], "episode_id": e.get("episode_id"),
                   "schema_passed": ok, "schema_detail": why, "judge": None, "model_id": None}
            if ok:
                try:
                    reply = llm.complete(judge_prompt_v3(e.get("step_goal") or "", field, e.get("http_status"), text, None),
                                         max_tokens=settings["judge_max_tokens"], timeout_s=20.0)
                except LLMError as exc:
                    rec["judge"] = {"llm_error": str(exc)[:300]}
                    llm_err += 1
                    records.append(rec)
                    continue
                verdict = parse_judgement(reply.text)
                rec["model_id"] = reply.model_id
                rec["judge"] = verdict if verdict is not None else {"unparseable": reply.text[:200]}
            n += 1
            schema_rej += int(not ok)
            rejected = (not ok) or (rec["judge"] is not None and rec["judge"].get("satisfies") is False)
            pipe_rej += int(rejected)
            records.append(rec)
        summary[field] = {"n": n, "llm_error_excluded": llm_err,
                          "false_alarm_schema": None if n == 0 else schema_rej / n,
                          "false_alarm_pipeline": None if n == 0 else pipe_rej / n}
    return {"settings": settings, "fa_sample_seed": FA_SAMPLE_SEED, "per_field": summary, "items": records}
