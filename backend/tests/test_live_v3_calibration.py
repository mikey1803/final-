"""A11: the false-alarm calibration samples only non-degraded, non-probe calls
from degraded_output blocks with FA_SAMPLE_SEED, judges each item once, and
records the settings."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.v3 import calibration as cal  # noqa: E402
from tests._v3_fakes import RecordingLLM, body_for  # noqa: E402


def _entry(i, tool="countriesnow", mode="degraded_output", degraded=False, success=True, probe=False, seed=231001):
    return {"call_id": i, "mode": mode, "degraded": degraded, "success": success, "probe": probe,
            "actor": "resend_probe" if probe else "agent", "tool": tool, "session_seed": seed, "block_index": 2,
            "http_status": 200, "step_goal": "Find the capital city of India.", "episode_id": f"e{i}",
            "delivered_text": body_for(tool, {"country": "India", "title": "India", "id": 1, "location": "x"})}


def _write_logs(root, entries):
    d = root / "v3_injection_logs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "s231001_b2.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_eligibility_only_non_degraded_d_block_non_probe(tmp_path):
    entries = [_entry(1), _entry(2, degraded=True), _entry(3, probe=True), _entry(4, mode="transient"),
               _entry(5, success=False), _entry(6, tool="wttr")]
    _write_logs(tmp_path, entries)
    items = cal.eligible_items(tmp_path)
    assert [e["call_id"] for e in items["countriesnow"]] == [1]
    assert [e["call_id"] for e in items["wttr"]] == [6]


def test_sample_is_seeded_capped_and_reproducible(tmp_path):
    _write_logs(tmp_path, [_entry(i) for i in range(1, 121)])
    items = cal.eligible_items(tmp_path)
    s1, s2 = cal.draw_sample(items), cal.draw_sample(items)
    assert [e["call_id"] for e in s1["countriesnow"]] == [e["call_id"] for e in s2["countriesnow"]]
    assert len(s1["countriesnow"]) == cal.FA_PER_FIELD == 50 and cal.FA_SAMPLE_SEED == 23002
    assert s1["wikipedia"] == []


def test_each_item_judged_once_and_settings_recorded(tmp_path):
    _write_logs(tmp_path, [_entry(i) for i in range(1, 8)])
    llm = RecordingLLM(verdict=False)
    out = cal.judge_sample(cal.draw_sample(cal.eligible_items(tmp_path)), llm)
    assert len(llm.prompts) == 7                         # once per item, no retries
    s = out["settings"]
    assert s["requested_model_id"] == "fake/model-1" and s["reasoning_effort"] == "low"
    assert s["temperature"] == 0.0 and s["judge_max_tokens"] == 200 and len(s["judge_prompt_template_sha256"]) == 64
    pf = out["per_field"]["countriesnow"]
    assert pf["n"] == 7 and pf["false_alarm_schema"] == 0.0 and pf["false_alarm_pipeline"] == 1.0
    assert all(r["model_id"] == "fake/model-1" for r in out["items"])
