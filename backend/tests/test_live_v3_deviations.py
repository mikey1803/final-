"""Deviations D-2, D-4, D-5, D-6 (v3 pre-registration §11). No network."""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.agent.llm_client as layer1  # noqa: E402
from recovery.live.event_schema import AgentState, FailureEvent, Task  # noqa: E402
from recovery.live.event_store import EventStore  # noqa: E402
from recovery.live.executors.base import LLMReply  # noqa: E402
from recovery.live.executors.verify import VerifyExecutor  # noqa: E402
from recovery.live.executors.prompts import judge_prompt  # noqa: E402
from recovery.live.recovery_attempt import RecoveryStore  # noqa: E402
from recovery.live.tools import TOOL_DESCRIPTIONS, ToolCallResult  # noqa: E402
from recovery.live.v3.flaky import OUTAGE_TIMEOUT_S, FlakyWrapperV3  # noqa: E402
from recovery.live.v3.loop import BlockSpec, V3Dirs, run_episode_v3  # noqa: E402
from recovery.live.v3.pacing import Pacer  # noqa: E402
from recovery.live.v3.recovery import InlineRecoveryV3  # noqa: E402
from recovery.live.v3.selector import (  # noqa: E402
    LAYER1_SOURCE_SHA256, SELECTOR_MAX_TOKENS, SELECTOR_REASONING_EFFORT, V3Selector, selector_prompt,
)
from recovery.live.v3.verify_v3 import VerifyExecutorV3, answer_field_label, judge_prompt_v3  # noqa: E402
from tests._v3_fakes import FakeServices, RecordingLLM, ScriptedSelector, tiny_libraries  # noqa: E402

GOALS = ["Find the capital city of India using the countriesnow tool.", "Find out about the weather situation there.",
         "Book a flight from Kolkata to London for tomorrow morning.", "Fetch todo item number 5 using the jsonplaceholder tool."]


# ---------------------------------------------------------------- D-2 selector

class _Captured:
    def __init__(self, content):
        self.prompts, self.content = [], content

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("R", (), {"content": self.content})()


class _TextLLM:
    model, reasoning_effort = "fake/model-1", "low"

    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, prompt, max_tokens, timeout_s):
        self.calls.append((prompt, max_tokens))
        return LLMReply(self.text, "fake/model-1", 10, 5, 1.0)


def test_layer1_source_is_pinned():
    src = inspect.getsource(layer1.parse_tool_request)
    assert hashlib.sha256(src.encode()).hexdigest() == LAYER1_SOURCE_SHA256, \
        "Layer 1 parse_tool_request changed: re-check the v3 selector copy (D-2)"


@pytest.mark.parametrize("goal", GOALS)
def test_selector_prompt_is_byte_identical_to_layer1(monkeypatch, goal):
    fake = _Captured('{"tool": "wikipedia", "args": {"title": "x"}}')
    monkeypatch.setattr(layer1, "_get_agent_chat_model", lambda: fake)
    layer1.parse_tool_request(goal, TOOL_DESCRIPTIONS)
    assert fake.prompts[0] == selector_prompt(goal, TOOL_DESCRIPTIONS)
    llm = _TextLLM('{"tool": "wikipedia", "args": {"title": "x"}}')
    V3Selector(llm)(goal)
    assert llm.calls[0] == (fake.prompts[0], SELECTOR_MAX_TOKENS)


@pytest.mark.parametrize("raw", ["", "   ", "not json", "```json\n{\"tool\": \"wttr\", \"args\": {}}\n```",
                                 '{"tool": "wttr"}', '{"tool": "wttr", "args": {"location": "x"}}', '["tool"]'])
def test_selector_parsing_matches_layer1(monkeypatch, raw):
    monkeypatch.setattr(layer1, "_get_agent_chat_model", lambda: _Captured(raw))

    def run(f):
        try:
            return ("ok", f())
        except ValueError as e:
            return ("ValueError", str(e))
    expected = run(lambda: layer1.parse_tool_request(GOALS[0], TOOL_DESCRIPTIONS))
    got = run(lambda: V3Selector(_TextLLM(raw.strip()))(GOALS[0]))
    assert got == expected


def test_selector_settings_and_log():
    assert SELECTOR_MAX_TOKENS == 300 and SELECTOR_REASONING_EFFORT == "low"
    s = V3Selector(_TextLLM(""))
    with pytest.raises(ValueError):
        s(GOALS[1])
    assert s.log[-1]["outcome"] == "empty" and s.log[-1]["empty"] is True
    assert s.settings()["max_tokens"] == 300


# ---------------------------------------------------------------- D-4 judge input

def test_v3_judge_sees_the_answer_field_even_past_500_chars():
    filler = {"k%d" % i: "x" * 40 for i in range(20)}
    body = json.dumps({**filler, "title": "Kolkata", "extract": "UNIQUE-EXTRACT-TEXT about Kolkata."})
    v2 = judge_prompt("Look up Kolkata", "wikipedia", 200, body, None)
    v3 = judge_prompt_v3("Look up Kolkata", "wikipedia", 200, body, None)
    assert "UNIQUE-EXTRACT-TEXT" not in v2 and "UNIQUE-EXTRACT-TEXT" in v3
    assert answer_field_label("wttr") == "current_condition[0].temp_C"
    assert "(missing)" in judge_prompt_v3("s", "wikipedia", 200, json.dumps({"title": "x"}), None)
    long = json.dumps({"extract": "y" * 900})
    assert judge_prompt_v3("s", "wikipedia", 200, long, None).count("y") <= 1000


def test_v2_verify_judge_input_is_unchanged():
    body = '{"title": "India", "extract": "India is a country."}'
    res = ToolCallResult(True, 200, body, None, 1.0, full_text=body)
    ctx = type("C", (), {"step_goal": "g", "tool_name": "wikipedia"})()
    assert VerifyExecutor()._judge_prompt(ctx, res) == judge_prompt("g", "wikipedia", 200, body, None)
    assert VerifyExecutorV3()._judge_prompt(ctx, res) == judge_prompt_v3("g", "wikipedia", 200, body, None)
    assert VerifyExecutorV3().config()["judge_input"] == "v3_answer_field_then_body_excerpt"
    rec = InlineRecoveryV3(store=RecoveryStore(Path("unused")), session_seed=1)
    assert isinstance(rec.executors["verify"], VerifyExecutorV3)


# ---------------------------------------------------------------- D-5 outage timeouts

def test_outage_uses_a_read_timeout():
    assert OUTAGE_TIMEOUT_S == (10.0, 0.001)
    svc = FakeServices()
    w = FlakyWrapperV3(mode="tool_outage", session_seed=1, block_index=0, rate=1.0, outage_N=2, tools=svc.tools())
    w.start_episode("e")
    w.call("wttr", {"location": "x"})
    assert svc.calls[-1][3] == (10.0, 0.001)


# ---------------------------------------------------------------- D-6 pacing

class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class _Client:
    def __init__(self, info):
        self.last_rate_limit = info


def test_pacer_waits_only_below_the_reserve_and_logs():
    clk = _Clock()
    p = Pacer(_Client({"remaining_tokens": 7000, "reset_tokens_s": 3.0, "remaining_requests": 900,
                       "reset_requests_s": 7000.0, "retry_after_s": None, "at": clk.t}), sleep=clk.sleep, clock=clk)
    assert p.before("strategy") == 0.0 and p.log == []
    p.client.last_rate_limit = {"remaining_tokens": 1000, "reset_tokens_s": 3.0, "remaining_requests": 900,
                                "reset_requests_s": 7000.0, "retry_after_s": None, "at": clk.t}
    w = p.before("strategy")
    assert w == pytest.approx(3.5) and p.log[-1]["reasons"] == ["tokens_per_minute"]
    p.client.last_rate_limit = {"remaining_tokens": 8000, "reset_tokens_s": 1.0, "remaining_requests": 0,
                                "reset_requests_s": 60.0, "retry_after_s": None, "at": clk.t}
    assert p.before("selection") == pytest.approx(60.5) and "requests_per_day" in p.log[-1]["reasons"]


def test_pacing_wait_is_outside_the_strategy_budget_clock(tmp_path):
    """A 30 s pacing wait before the strategy must not appear in its wall time."""
    clk = _Clock()
    svc = FakeServices()
    w = FlakyWrapperV3(mode="sticky_args", session_seed=231000, block_index=0, rate=1.0,
                       libraries=tiny_libraries(), tools=svc.tools())
    llm = RecordingLLM()
    llm.last_rate_limit = {"remaining_tokens": 100, "reset_tokens_s": 30.0, "remaining_requests": 900,
                           "reset_requests_s": 7000.0, "retry_after_s": None, "at": clk.t}
    rec = InlineRecoveryV3(store=RecoveryStore(tmp_path), session_seed=231000, llm=llm, sleep=clk.sleep, clock=clk)
    rec.arms = ["switch_tool"]
    pacer = Pacer(llm, sleep=clk.sleep, clock=clk)
    pacer.reserve_tokens["selection"] = 0          # make the wait land right before the strategy
    task = Task(task_id="country_basic_01", goal="capital of India", category="tool_call_basic")
    ep = run_episode_v3(task, wrapper=w, events=EventStore(tmp_path), recoveries=RecoveryStore(tmp_path),
                        recovery=rec, select=ScriptedSelector([{"tool": "countriesnow", "args": {"country": "India"}}]),
                        block=BlockSpec(0, "sticky_args", None, 1), session={"session_seed": 231000, "session_index": 0},
                        episode_index=0, tool_health=None, dirs=V3Dirs(tmp_path), pacer=pacer)
    (a,) = list(RecoveryStore(tmp_path).iter_attempts())
    assert ep["pacing_wait_s"] == pytest.approx(30.5) and pacer.log[-1]["checkpoint"] == "strategy"
    assert a.v3["pacing_wait_before_strategy_s"] == pytest.approx(30.5)
    assert a.result["latency_ms"] < 1000.0     # the strategy's own clock never saw the wait
