"""D-8 (outage index k, H1 exemption at N - k = 0, k = 1 primary analysis and
stopping) and D-9 (HTTP 400 tool_use_failed is an agent-side malformed_output)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.executors.base import LLMError, LLMReply  # noqa: E402
from recovery.live.v3 import report, stopping  # noqa: E402
from recovery.live.v3.selection import validate_selection  # noqa: E402
from recovery.live.v3.selector import ToolUseFailed, V3Selector  # noqa: E402
from tests._v3_fakes import RecordingLLM, ScriptedSelector  # noqa: E402
from tests.test_live_v3_loop import COUNTRY, PICK_COUNTRY, FakeRng, Rig  # noqa: E402

TUF_BODY = json.dumps({"error": {"message": "Tool choice is none, but model called a tool",
                                 "type": "invalid_request_error", "code": "tool_use_failed",
                                 "failed_generation": '{"name": "countriesnow", "arguments": {"country": "India"}}'}})


# ---------------------------------------------------------------- D-8

def _o2_two_episodes(tmp_path, arm="switch_tool"):
    rig = Rig(tmp_path, "tool_outage", n=2, arm=arm)
    rig.wrapper._rng["outage"] = FakeRng([0.0])          # the first fresh draw starts an outage on countriesnow
    rig.recovery.arms = [arm]
    rig.run(COUNTRY)                                       # agent call = k 1; switch_tool's wikipedia draws clean
    rig.run(COUNTRY)                                       # next episode's agent call = k 2 = N: the last call
    return rig


def test_outage_k_and_remaining_are_recorded(tmp_path):
    rig = _o2_two_episodes(tmp_path)
    evs = sorted(rig.events.iter_events(), key=lambda e: e.v3["episode_index"])
    assert [(e.v3["outage_k"], e.v3["outage_remaining"]) for e in evs] == [(1, 1), (2, 0)]
    a = {x.v3["episode_id"]: x for x in rig.attempts()}
    assert sorted(x.v3["outage_k"] for x in a.values()) == [1, 2]


def test_h1_exempts_the_outage_last_call_and_tags_it(tmp_path):
    rig = _o2_two_episodes(tmp_path)                       # would have raised HaltH1 before D-8
    last = [e for e in rig.events.iter_events() if e.v3["outage_k"] == 2][0]
    row = json.loads((tmp_path / "v3_resends" / f"{last.event_id}.json").read_text())
    assert row["outcome"] is True and row["h1_exempt_outage_end"] is True
    a = [x for x in rig.attempts() if x.v3["outage_k"] == 2][0]
    assert a.v3["h1_exempt_outage_end"] is True
    mc = report.manipulation_checks(tmp_path)
    assert mc["h1_exempt_outage_end"] == 1 and mc["resend_on_injected"]["O2"]["successes"] == 0


def test_stopping_counts_only_k1_in_outage_cells(tmp_path):
    _o2_two_episodes(tmp_path)
    st = stopping.stopping_status(tmp_path)
    assert st["cells"]["O2/switch_tool"] == 1                # 2 failures, only the k = 1 one counts


# ---------------------------------------------------------------- D-9

class TUFLLM:
    model, reasoning_effort = "fake/model-1", "low"

    def __init__(self, body=TUF_BODY, status=400):
        self.body, self.status = body, status

    def complete(self, prompt, max_tokens, timeout_s):
        raise LLMError(f"HTTP {self.status}: {self.body[:300]}", status=self.status, body=self.body)


def test_tool_use_failed_is_an_unusable_selection_with_failed_generation():
    s = V3Selector(TUFLLM())
    with pytest.raises(ToolUseFailed) as ei:
        s("Find the capital city of India using the countriesnow tool.")
    assert '"countriesnow"' in ei.value.failed_generation
    assert s.log[-1]["outcome"] == "tool_use_failed" and "countriesnow" in s.log[-1]["failed_generation"]
    sel = validate_selection(s, "goal")
    assert sel.kind == "malformed_output" and sel.detail["tool_use_failed"] is True
    assert "arguments" in sel.detail["failed_generation"]


def test_other_provider_errors_are_still_provider_errors():
    from recovery.live.v3.selection import SelectionProviderError
    s = V3Selector(TUFLLM(body=json.dumps({"error": {"code": "rate_limit_exceeded"}}), status=429))
    with pytest.raises(SelectionProviderError):
        validate_selection(s, "goal")
    assert s.log[-1]["outcome"] == "provider_error"


class OnceTUFSelector:
    """First selection: tool_use_failed; afterwards a normal valid call."""

    def __init__(self):
        self.n, self.context, self.log = 0, {}, []

    def __call__(self, goal):
        self.n += 1
        if self.n == 1:
            raise ToolUseFailed('{"name": "countriesnow", "arguments": {"country": "India"}}')
        return dict(PICK_COUNTRY)


def test_tool_use_failed_becomes_an_agent_side_event_and_strategies_apply(tmp_path):
    rig = Rig(tmp_path, "transient", arm="retry", selector=OnceTUFSelector(), rate=0.0)
    ep = rig.run(COUNTRY)
    e = rig.event()
    assert e.v3["fault_origin"] == "agent_side" and e.v3["agent_side_kind"] == "malformed_output"
    assert e.v3["selection_detail"]["tool_use_failed"] is True
    assert "arguments" in e.v3["selection_detail"]["failed_generation"]
    (a,) = rig.attempts()
    assert a.v3["agent_side_detail"]["tool_use_failed"] is True
    assert a.result["success"] and ep["task_completed"] and not ep["aborted"]


def test_selector_by_task_reports_tool_use_failed_rate(tmp_path):
    d = tmp_path / "v3_selector_log"
    d.mkdir()
    rows = [{"task_id": "ambiguous_01", "outcome": "tool_use_failed"}, {"task_id": "ambiguous_01", "outcome": "parsed"},
            {"task_id": "wiki_basic_01", "outcome": "parsed"}]
    (d / "s1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t = report.selector_by_task(tmp_path)
    assert t.loc["ambiguous_01", "tool_use_failed_rate"] == 0.5 and t.loc["wiki_basic_01", "tool_use_failed"] == 0
