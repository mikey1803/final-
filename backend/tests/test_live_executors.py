"""Recovery executors against fake flaky tools / a fake LLM (Task 21).

No network. A fake clock makes wall-time budgets deterministic: tool calls
advance it by their latency, sleeps by their duration, LLM calls by theirs.
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.event_schema import AgentState, FailureEvent, Task  # noqa: E402
from recovery.live.executors import (  # noqa: E402
    Budget, ClarifyExecutor, NullControlExecutor, ReGroundExecutor, ReplanExecutor, RetryExecutor,
    SwitchToolExecutor, VerifyExecutor, build_executors,
)
from recovery.live.executors.base import (  # noqa: E402
    REASONS, TOOL_CALL_RESERVE_MS, FailureContext, LLMError, LLMReply,
)
from recovery.live.executors.responders import CallbackResponder  # noqa: E402
from recovery.live.tools import ToolCallResult  # noqa: E402

GOOD = {
    "countriesnow": '{"error": false, "msg": "ok", "data": {"name": "India", "capital": "New Delhi"}}',
    "wikipedia": '{"title": "India", "extract": "India is a country in South Asia."}',
    "open_meteo": '{"current_weather": {"temperature": 30.5, "time": "2026-10-21T12:00"}}',
    "wttr": '{"current_condition": [{"temp_C": "30"}], "weather": []}',
    "jsonplaceholder": '{"userId": 1, "id": 1, "title": "t", "completed": false}',
    "dummyjson": '{"id": 1, "todo": "t", "completed": false, "userId": 1}',
}


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleep_calls = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleep_calls.append(s)
        self.t += s


def ok(tool, latency=100.0):
    return ToolCallResult(True, 200, GOOD[tool], None, latency, full_text=GOOD[tool])


def fail(latency=100.0, status=500, text='{"error": true}'):
    return ToolCallResult(False, status, text, f"HTTP {status}: {text}", latency, full_text=text)


def timeout(latency=200.0):
    return ToolCallResult(False, None, None, "TimeoutError: read timed out", latency, injected=True,
                          injected_kind="timeout")


class FakeTools:
    """Returns scripted results in order (per call); advances the clock."""

    def __init__(self, clock, results):
        self.clock, self.results, self.calls = clock, list(results), []

    def __call__(self, tool, args):
        self.calls.append((tool, dict(args)))
        r = self.results.pop(0) if self.results else fail()
        self.clock.t += r.latency_ms / 1000.0
        return r


class FakeLLM:
    def __init__(self, clock, replies, prompt_tokens=None, completion_tokens=40, latency_ms=150.0, error=False):
        self.clock, self.replies, self.prompts = clock, list(replies), []
        self.pt, self.ct, self.lat, self.error = prompt_tokens, completion_tokens, latency_ms, error

    def complete(self, prompt, max_tokens, timeout_s):
        self.prompts.append(prompt)
        if self.error:
            raise LLMError("HTTP 503: unavailable")
        lat = min(self.lat, timeout_s * 1000.0)
        self.clock.t += lat / 1000.0
        pt = self.pt if self.pt is not None else math.ceil(len(prompt) / 4)
        return LLMReply(self.replies.pop(0) if self.replies else "garbage", "fake/model-1", pt,
                        min(self.ct, max_tokens), lat)


def make_ctx(tool="countriesnow", args=None, failed=None, results=(), replies=(), task=None,
             history=None, responder=None, llm_error=False, seed=0):
    clock = FakeClock()
    args ={"country": "India"} if args is None else args
    task = task or Task(task_id="t1", goal="Find the capital city of India.", category="tool_call_basic",
                        clarify_canned_answer="I mean India, the country.")
    event = FailureEvent(session_id="s1", step_index=0, tool_name=tool, tool_args=args, raw_error="x",
                         agent_state=AgentState(goal=task.goal), captured_at="2026-10-21T00:00:00+00:00",
                         task_id=task.task_id)
    tools = FakeTools(clock, results)
    llm = FakeLLM(clock, replies, error=llm_error)
    ctx = FailureContext(event=event, task=task, step_goal=task.goal, tool_name=tool, tool_args=args,
                         failed_result=failed or fail(), history=history or [], call_tool=tools,
                         rng=random.Random(seed), llm=llm, responder=responder, sleep=clock.sleep, clock=clock)
    return ctx, tools, llm, clock


def call_json(tool, args):
    return json.dumps({"tool": tool, "args": args})


def check_result_contract(r, budget=Budget()):
    assert isinstance(r.success, bool) and isinstance(r.cost_tokens, int) and isinstance(r.attempts, int)
    assert r.evidence["reason"] in REASONS
    assert {"calls", "llm_calls", "answered_by", "reason"} <= set(r.evidence)
    assert r.attempts == len(r.evidence["calls"])
    assert r.attempts <= budget.max_attempts and r.cost_tokens <= budget.max_tokens
    assert r.latency_ms <= budget.max_wall_ms


# ---------------- RETRY ----------------

def test_retry_v2_success_after_backoff():
    ctx, tools, _, clock = make_ctx(results=[fail(), ok("countriesnow")])
    r = RetryExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert r.success and r.attempts == 2 and r.evidence["answered_by"] == "countriesnow"
    d = r.evidence["backoff_delays_s"]
    assert 0 <= d[0] <= 0.5 and 0 <= d[1] <= 1.0
    assert tools.calls == [("countriesnow", {"country": "India"})] * 2


def test_retry_v2_failure_exhausts_attempts():
    ctx, tools, _, _ = make_ctx(results=[fail(), fail(), fail()])
    r = RetryExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert not r.success and r.attempts == 3 and r.evidence["reason"] == "exhausted_attempts"


def test_retry_v1_is_one_immediate_reissue():
    ctx, tools, _, clock = make_ctx(results=[fail(), ok("countriesnow")])
    r = RetryExecutor.v1().execute(ctx, Budget())
    assert r.attempts == 1 and not r.success and r.evidence["backoff_delays_s"] == [0.0]
    assert clock.sleep_calls == []


def test_retry_stops_when_wall_budget_cannot_fit_another_call():
    ctx, *_ = make_ctx(results=[fail(latency=20_000.0), fail()])
    r = RetryExecutor().execute(ctx, Budget(max_wall_ms=30_000))
    assert r.attempts == 1 and r.evidence["reason"] == "budget_exhausted"


# ---------------- SWITCH_TOOL ----------------

def test_switch_tool_success_uses_peer():
    ctx, tools, _, _ = make_ctx(results=[ok("wikipedia")])
    r = SwitchToolExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert r.success and tools.calls == [("wikipedia", {"title": "India"})]
    assert r.evidence["answered_by"] == "wikipedia"


def test_switch_tool_tried_and_failed():
    ctx, *_ = make_ctx(results=[fail()])
    r = SwitchToolExecutor().execute(ctx, Budget())
    assert not r.success and r.attempts == 1 and r.evidence["reason"] == "exhausted_attempts"


def test_switch_tool_untranslatable_makes_no_call():
    ctx, tools, _, _ = make_ctx(tool="wikipedia", args={"title": "Kolkata"})
    r = SwitchToolExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert not r.success and r.attempts == 0 and tools.calls == []
    assert r.evidence["reason"] == "untranslatable_args"
    assert r.evidence["translation"]["ok"] is False


# ---------------- VERIFY ----------------

def test_verify_v2_reissue_passes_schema_and_judge():
    ctx, tools, llm, _ = make_ctx(results=[ok("countriesnow")],
                                  replies=['{"satisfies": true, "reason": "capital found"}'])
    r = VerifyExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert r.success and r.attempts == 1 and r.evidence["no_action_taken"] is False
    stages = [(c["stage"], c["check"], c["passed"]) for c in r.evidence["checks"]]
    # original failed body rejected by schema (judge skipped), re-issue passes both
    assert stages == [("original", "schema", False), ("reissue", "schema", True), ("reissue", "llm_judge", True)]


def test_verify_v2_failure_when_judge_rejects():
    ctx, *_ = make_ctx(results=[ok("countriesnow")], replies=['{"satisfies": false, "reason": "made up"}'])
    r = VerifyExecutor().execute(ctx, Budget())
    assert not r.success and r.evidence["reason"] == "checks_rejected" and r.evidence["answered_by"] is None


def test_verify_original_accepted_sets_no_action_taken():
    ctx, tools, _, _ = make_ctx(failed=ok("countriesnow"), replies=['{"satisfies": true, "reason": "fine"}'])
    r = VerifyExecutor().execute(ctx, Budget())
    assert r.success and r.attempts == 0 and tools.calls == []
    assert r.evidence["reason"] == "original_result_accepted" and r.evidence["no_action_taken"] is True


def test_verify_no_result_reissues_once_then_checks():
    ctx, tools, _, _ = make_ctx(failed=timeout(), results=[ok("countriesnow")],
                                replies=['{"satisfies": true, "reason": "ok"}'])
    r = VerifyExecutor().execute(ctx, Budget())
    assert r.success and r.attempts == 1 and r.evidence["original_had_result"] is False
    assert [c["stage"] for c in r.evidence["checks"]] == ["reissue", "reissue"]


def test_verify_schema_check_uses_untruncated_full_text():
    big = json.dumps({"title": "India", "extract": "y" * 5000})
    truncated = ToolCallResult(True, 200, big[:2000] + "...[truncated]", None, 50.0, full_text=big)
    ctx, *_ = make_ctx(tool="wikipedia", args={"title": "India"}, failed=truncated,
                       replies=['{"satisfies": true, "reason": "ok"}'])
    r = VerifyExecutor().execute(ctx, Budget())
    assert r.evidence["checks"][0] == {"stage": "original", "check": "schema", "passed": True, "detail": None}


def test_verify_v1_is_llm_judge_only():
    v1 = VerifyExecutor.v1()
    assert v1.schema_check is False and v1.llm_judge is True and v1.definition_version == "v1"
    ctx, *_ = make_ctx(results=[ok("countriesnow")],
                       replies=['{"satisfies": false, "reason": "error"}', '{"satisfies": true, "reason": "ok"}'])
    r = v1.execute(ctx, Budget())
    assert r.success and all(c["check"] == "llm_judge" for c in r.evidence["checks"])


def test_verify_unparseable_judge_and_llm_error():
    ctx, *_ = make_ctx(results=[ok("countriesnow")], replies=["sure, looks fine"])
    assert VerifyExecutor().execute(ctx, Budget()).evidence["reason"] == "unparseable_llm_output"
    ctx, *_ = make_ctx(results=[ok("countriesnow")], llm_error=True)
    r = VerifyExecutor().execute(ctx, Budget())
    assert r.evidence["reason"] == "llm_error" and not r.success


def test_verify_requires_a_check():
    with pytest.raises(ValueError):
        VerifyExecutor(schema_check=False, llm_judge=False)


# ---------------- RE_GROUND ----------------

def test_re_ground_v2_model_reselects_without_seeing_failure():
    hist = [{"role": "tool", "tool": "wikipedia", "args": {"title": "India"}, "content": "India is..."}]
    ctx, tools, llm, _ = make_ctx(results=[ok("countriesnow")], history=hist,
                                  replies=[call_json("countriesnow", {"country": "India"})])
    ctx.failed_result = fail(text='{"error": true, "msg": "UNIQUE-FAILURE-TEXT"}')
    r = ReGroundExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert r.success and r.evidence["reground_chosen_tool"] == "countriesnow" and r.evidence["same_call"] is True
    assert "UNIQUE-FAILURE-TEXT" not in llm.prompts[0] and "India is..." in llm.prompts[0]


def test_re_ground_v2_failure_and_unparseable():
    ctx, *_ = make_ctx(results=[fail()], replies=[call_json("wikipedia", {"title": "India"})])
    r = ReGroundExecutor().execute(ctx, Budget())
    assert not r.success and r.evidence["same_call"] is False and r.evidence["reason"] == "exhausted_attempts"
    ctx, tools, _, _ = make_ctx(replies=["not json"])
    r = ReGroundExecutor().execute(ctx, Budget())
    assert r.evidence["reason"] == "unparseable_llm_output" and tools.calls == []


def test_re_ground_v1_reissues_same_call_without_llm():
    ctx, tools, llm, _ = make_ctx(results=[ok("countriesnow")])
    r = ReGroundExecutor.v1().execute(ctx, Budget())
    assert r.success and llm.prompts == [] and tools.calls == [("countriesnow", {"country": "India"})]


# ---------------- REPLAN ----------------

def test_replan_success_logs_chosen_tool():
    ctx, tools, llm, _ = make_ctx(results=[ok("wikipedia")], replies=[call_json("wikipedia", {"title": "India"})])
    r = ReplanExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert r.success and r.evidence["replan_chosen_tool"] == "wikipedia" and r.evidence["same_call"] is False
    assert "Error:" in llm.prompts[0]


def test_replan_failure_path():
    ctx, *_ = make_ctx(results=[fail()], replies=[call_json("countriesnow", {"country": "India"})])
    r = ReplanExecutor().execute(ctx, Budget())
    assert not r.success and r.evidence["same_call"] is True and r.evidence["reason"] == "exhausted_attempts"
    assert r.evidence["llm_calls"][0]["model_id"] == "fake/model-1"


# ---------------- CLARIFY ----------------

def test_clarify_success_with_canned_answer():
    ctx, tools, llm, _ = make_ctx(results=[ok("countriesnow")],
                                  replies=["Which country do you mean?", call_json("countriesnow", {"country": "India"})])
    r = ClarifyExecutor().execute(ctx, Budget())
    check_result_contract(r)
    assert r.success and r.evidence["answer"] == "I mean India, the country." and r.evidence["responder"] == "canned"
    assert "I mean India, the country." in llm.prompts[1]


def test_clarify_question_unmatched_without_canned_answer():
    task = Task(task_id="t2", goal="Find the capital city of India.", category="tool_call_basic")
    ctx, tools, _, _ = make_ctx(task=task, replies=["Which country?"])
    r = ClarifyExecutor().execute(ctx, Budget())
    assert not r.success and r.evidence["reason"] == "question_unmatched" and r.evidence["aborted"] is True
    assert tools.calls == [] and r.attempts == 0


def test_clarify_human_responder_timeout_is_unmatched():
    import time as _time
    slow = CallbackResponder(lambda q, t: (_time.sleep(0.5), "late")[1], timeout_s=0.05)
    ctx, *_ = make_ctx(responder=slow, replies=["Which country?"])
    r = ClarifyExecutor().execute(ctx, Budget())
    assert r.evidence["reason"] == "question_unmatched" and r.evidence["responder"] == "human_callback"


def test_clarify_tool_failure_path():
    ctx, *_ = make_ctx(results=[fail()], replies=["Which country?", call_json("countriesnow", {"country": "India"})])
    r = ClarifyExecutor().execute(ctx, Budget())
    assert not r.success and r.evidence["reason"] == "exhausted_attempts"


# ---------------- NULL_CONTROL ----------------

def test_null_control_requires_wait_and_waits_then_reissues():
    with pytest.raises(TypeError):
        NullControlExecutor()  # no default wait_s: [RESULT REQUIRED]
    ctx, tools, _, clock = make_ctx(results=[ok("countriesnow")])
    r = NullControlExecutor(wait_s=1.5).execute(ctx, Budget())
    assert r.success and clock.sleep_calls == [1.5] and r.attempts == 1


def test_build_executors_versions_and_null_control_exclusion():
    v2 = build_executors("v2")
    assert list(v2) == ["retry", "replan", "switch_tool", "re_ground", "verify", "clarify"]
    assert all(e.definition_version == "v2" for e in v2.values())
    assert all(e.definition_version == "v1" for e in build_executors("v1").values())
    assert "null_control" in build_executors("v2", null_control_wait_s=2.0)
    with pytest.raises(ValueError):
        build_executors("v3")


# ---------------- property test: no executor ever exceeds its budget ----------------

N_SCENARIOS = 400


def _random_reply(rng):
    pick = rng.integers(5)
    if pick == 0:
        return "not json at all"
    if pick == 1:
        return '{"satisfies": %s, "reason": "r"}' % ("true" if rng.random() < 0.5 else "false")
    if pick == 2:
        return "Which one do you mean?"
    tool = ["countriesnow", "wikipedia", "open_meteo", "wttr", "jsonplaceholder", "dummyjson"][int(rng.integers(6))]
    return call_json(tool, {"country": "India"})


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_property_no_executor_exceeds_budget(version):
    # Seeds from SeedSequence (CLAUDE.md rule 6), one child per scenario.
    children = np.random.SeedSequence(21_000 + (version == "v2")).spawn(N_SCENARIOS)
    executors = build_executors(version, null_control_wait_s=0.7)
    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        budget = Budget(max_attempts=int(rng.integers(1, 5)), max_wall_ms=float(rng.uniform(5_000, 60_000)),
                        max_tokens=int(rng.integers(100, 2_500)))
        for name, executor in executors.items():
            results = []
            for _ in range(6):
                lat = float(rng.uniform(5, TOOL_CALL_RESERVE_MS))  # a call never outlasts the assumed worst case
                kind = rng.integers(3)
                tool = ["countriesnow", "wikipedia", "wttr"][int(rng.integers(3))]
                results.append(ok(tool, lat) if kind == 0 else fail(lat) if kind == 1 else timeout(lat))
            replies = [_random_reply(rng) for _ in range(6)]
            failed = [fail(), timeout(), ok("countriesnow")][int(rng.integers(3))]
            ctx, tools, llm, clock = make_ctx(failed=failed, results=results, replies=replies, seed=int(rng.integers(1 << 30)))
            llm.lat = float(rng.uniform(10, 25_000))
            llm.ct = int(rng.integers(1, 400))
            llm.pt = None  # ceil(len/4) <= ceil(len/2): within the reservation rule's assumption
            r = executor.execute(ctx, budget)
            where = f"scenario {i} executor {name} budget {budget}"
            assert r.attempts <= budget.max_attempts, where
            assert r.cost_tokens <= budget.max_tokens, where
            assert r.evidence["budget_exceeded"] is False, where       # unrounded wall/attempt/token check
            assert r.latency_ms <= budget.max_wall_ms + 0.05, where    # stored value is rounded to 0.1 ms
            assert r.evidence["reason"] in REASONS, where
