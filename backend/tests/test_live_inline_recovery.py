"""Inline recovery wired into the live agent loop (Task 21). No network:
tool selection and the FlakyWrapper are replaced by fakes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recovery.live.agent_loop as loop  # noqa: E402
from recovery.live.event_schema import Task  # noqa: E402
from recovery.live.event_store import EventStore  # noqa: E402
from recovery.live.executors import Budget  # noqa: E402
from recovery.live.inline_recovery import NOT_APPLICABLE_SELECTION_FAILURE, InlineRecovery  # noqa: E402
from recovery.live.recovery_attempt import RecoveryAttempt, RecoveryStore  # noqa: E402
from recovery.live.tools import ToolCallResult  # noqa: E402

TASK = Task(task_id="t1", goal="Find the capital city of India using the countriesnow tool.",
            category="tool_call_basic", clarify_canned_answer="India, the country.")
GOOD = '{"error": false, "msg": "ok", "data": {"name": "India", "capital": "New Delhi"}}'

REQUIRED_FIELDS = {
    "event_id", "strategy", "definition_version", "selection_propensity", "seed", "model_id",
    "budget_limits", "budget_used", "result", "started_at", "finished_at",
}


class FakeWrapper:
    """First call fails (as if injected), every later call succeeds."""

    def __init__(self):
        self.n = 0

    def call(self, tool, args):
        self.n += 1
        if self.n == 1:
            return ToolCallResult(False, None, None, "TimeoutError: injected", 200.0, injected=True,
                                  injected_kind="timeout")
        return ToolCallResult(True, 200, GOOD, None, 50.0, full_text=GOOD)


class FakeLLM:
    def complete(self, prompt, max_tokens, timeout_s):
        from recovery.live.executors.base import LLMReply
        if "satisfies" in prompt:
            text = '{"satisfies": true, "reason": "ok"}'
        elif "clarifying question" in prompt:
            text = "Which country?"
        else:
            text = json.dumps({"tool": "countriesnow", "args": {"country": "India"}})
        return LLMReply(text, "fake/model-1", 100, 30, 10.0)


def _pick(monkeypatch, value):
    def fake_select(step_goal):
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(loop, "_select_tool", fake_select)


def _recovery(tmp_path, seed=7):
    return InlineRecovery(store=RecoveryStore(tmp_path), session_seed=seed, llm=FakeLLM(),
                          sleep=lambda s: None)


def test_default_loop_applies_no_strategy(monkeypatch, tmp_path):
    _pick(monkeypatch, {"tool": "countriesnow", "args": {"country": "India"}})
    out = loop.run_task(TASK, EventStore(tmp_path), FakeWrapper(), session_id="s")
    assert not out.completed and out.failure_event is not None and out.recovery_attempt is None
    assert not (tmp_path / "recoveries").exists()


def test_inline_recovery_logs_full_attempt(monkeypatch, tmp_path):
    _pick(monkeypatch, {"tool": "countriesnow", "args": {"country": "India"}})
    rec = _recovery(tmp_path)
    out = loop.run_task(TASK, EventStore(tmp_path), FakeWrapper(), session_id="s", recovery=rec)
    a = out.recovery_attempt
    assert a is not None and a.event_id == out.failure_event.event_id
    stored = list(RecoveryStore(tmp_path).iter_attempts())
    assert len(stored) == 1
    d = stored[0].to_dict()
    assert REQUIRED_FIELDS <= set(d)
    assert d["selection_propensity"] == 1 / 6 and d["selection_policy"] == "uniform"
    assert d["definition_version"] == "v2" and d["context"] == "inline"
    assert d["budget_limits"] == Budget().as_dict()
    assert set(d["result"]) == {"success", "cost_tokens", "latency_ms", "attempts", "evidence"}
    assert d["result"]["evidence"]["reason"]
    assert d["seed"]["session_seed"] == 7 and d["seed"]["failure_index"] == 0
    # event store still loads alongside recoveries/
    assert len(list(EventStore(tmp_path).iter_events())) == 1


def test_selection_is_reproducible_and_covers_all_six(tmp_path):
    def draws(seed):
        rec = InlineRecovery(store=RecoveryStore(tmp_path / str(seed)), session_seed=seed)
        import numpy as np
        out = []
        for i in range(60):
            seq = np.random.SeedSequence(seed, spawn_key=(i,))
            derived = int(seq.generate_state(1, dtype=np.uint64)[0])
            out.append(rec.policy.choose(rec.arms, np.random.default_rng(derived))[0])
        return out
    assert draws(11) == draws(11)
    assert set(draws(11)) == {"retry", "replan", "switch_tool", "re_ground", "verify", "clarify"}


def test_null_control_excluded_until_wait_is_set(tmp_path):
    assert "null_control" not in InlineRecovery(store=RecoveryStore(tmp_path), session_seed=1).arms
    with_wait = InlineRecovery(store=RecoveryStore(tmp_path), session_seed=1, null_control_wait_s=2.0)
    assert "null_control" in with_wait.arms and len(with_wait.arms) == 7


def test_selection_failure_is_skipped_and_counted(monkeypatch, tmp_path):
    _pick(monkeypatch, ValueError("LLM did not return valid JSON"))
    rec = _recovery(tmp_path)
    out = loop.run_task(TASK, EventStore(tmp_path), FakeWrapper(), session_id="s", recovery=rec)
    assert out.recovery_attempt is None and out.recovery_skipped_reason == NOT_APPLICABLE_SELECTION_FAILURE
    assert rec.skipped == {NOT_APPLICABLE_SELECTION_FAILURE: 1}
    assert list(RecoveryStore(tmp_path).iter_attempts()) == []


def test_executor_crash_is_recorded_not_raised(monkeypatch, tmp_path):
    _pick(monkeypatch, {"tool": "countriesnow", "args": {"country": "India"}})
    rec = _recovery(tmp_path)

    class Boom:
        name, definition_version = "retry", "v2"

        def config(self):
            return {}

        def execute(self, ctx, budget):
            raise RuntimeError("bug")

    rec.executors = {k: Boom() for k in rec.executors}
    out = loop.run_task(TASK, EventStore(tmp_path), FakeWrapper(), session_id="s", recovery=rec)
    assert out.recovery_attempt.result["evidence"]["reason"] == "executor_error"


def test_recovery_attempt_roundtrip_and_validation():
    a = RecoveryAttempt(
        event_id="e", session_id="s", task_id="t", strategy="retry", definition_version="v2",
        selection_policy="uniform", selection_propensity=1 / 6, seed={"session_seed": 1}, model_id=None,
        executor_config={}, budget_limits={}, budget_used={}, budget_exceeded=False,
        result={"success": True}, started_at="2026-10-21T00:00:00+00:00", finished_at="2026-10-21T00:00:01+00:00",
    )
    assert RecoveryAttempt.from_dict(a.to_dict()).attempt_id == a.attempt_id
    import pytest
    for bad in ({"strategy": "pray"}, {"definition_version": "v9"}, {"selection_propensity": 0.0}):
        with pytest.raises(ValueError):
            RecoveryAttempt(**{**{k: v for k, v in a.to_dict().items() if k != "attempt_id"}, **bad})
