"""Task 22: runtime guardrails (recovery.guards.runtime) as units."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.guards.runtime import (  # noqa: E402
    BreakerConfig, BreakerState, CircuitBreaker, MeanTable, TaskBudget, choose_arm, escalate, exploit_choice,
)
from recovery.live.executors.base import TOOL_CALL_RESERVE_MS, Budget  # noqa: E402
from recovery.live.executors.responders import CallbackResponder  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_at_threshold_probes_after_cooldown_and_recovers():
    c = Clock()
    b = CircuitBreaker("api", BreakerConfig(failure_threshold=3, cooldown_s=10.0), c)
    for _ in range(2):
        assert b.allow()
        b.record(False)
    assert b.state is BreakerState.CLOSED
    b.record(False)
    assert b.state is BreakerState.OPEN and not b.allow()
    c.t = 9.99
    assert not b.allow()
    c.t = 10.0
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow() and not b.allow()            # exactly one probe in flight
    b.record(False)                                # failed probe: reopen, cooldown restarts
    assert b.state is BreakerState.OPEN
    c.t = 19.0
    assert not b.allow()
    c.t = 20.0
    assert b.allow()
    b.record(True)
    assert b.state is BreakerState.CLOSED and b.consecutive_failures == 0
    assert [(x["from"], x["to"]) for x in b.transitions] == [
        ("closed", "open"), ("open", "half_open"), ("half_open", "open"), ("open", "half_open"),
        ("half_open", "closed")]


def test_a_success_resets_the_consecutive_count():
    b = CircuitBreaker("api", BreakerConfig(failure_threshold=3), Clock())
    for ok in (False, False, True, False, False):
        b.record(ok)
    assert b.state is BreakerState.CLOSED


def test_breaker_config_rejects_nonsense():
    with pytest.raises(ValueError):
        BreakerConfig(failure_threshold=0)


def test_task_budget_admission_and_remaining():
    tb = TaskBudget(Budget(max_attempts=3, max_wall_ms=30_000, max_tokens=500))
    assert tb.admits_strategy()
    tb.charge(2, 400, 17_000)
    r = tb.remaining()
    assert (r.max_attempts, r.max_tokens, r.max_wall_ms) == (1, 100, 13_000)
    assert tb.admits_strategy()                    # 13 s >= the 12 s worst-case call
    tb.charge(0, 0, 13_000 - TOOL_CALL_RESERVE_MS + 1)
    assert not tb.admits_strategy() and not tb.exceeded()


def test_escalate_with_and_without_a_responder():
    kw = dict(tool="api", args={"q": 1}, error="HTTP 500: x", predicted_type="tool_error", confidence=1.0,
              attempts_made=2, budget_used={"attempts": 3}, attempt_ids=["a", "b"], task=None)
    e = escalate("budget_exhausted", responder=CallbackResponder(lambda q, t: "use yesterday's value"), **kw)
    assert e.handled and e.answer == "use yesterday's value" and e.responder == "human_callback"
    assert "budget exhausted" in e.question and "use yesterday's value" in e.to_message()
    e2 = escalate("low_confidence", **kw)
    assert not e2.handled and e2.answer is None and e2.responder is None

    class Broken:
        kind = "broken"

        def answer(self, q, t, s):
            raise RuntimeError("ui down")

    e3 = escalate("low_confidence", responder=Broken(), **kw)
    assert not e3.handled and "ui down" in e3.responder_error
    with pytest.raises(ValueError):
        escalate("bored", **kw)


def test_exploit_choice_is_highest_mean_with_lowest_index_ties():
    assert exploit_choice([0.1, 0.9, 0.9, 0.2], [0, 1, 2, 3]) == 1
    assert exploit_choice([0.1, 0.9, 0.9, 0.2], [0, 2, 3]) == 2


def test_choose_arm_never_calls_select_for_side_effect_tools():
    class Spy:
        name, calls = "spy", 0

        def select(self, ctx, rng):
            self.calls += 1
            return 0

    class Ctx:
        predicted_type = 0

    mt = MeanTable(1, 3)
    mt.update(0, 2, True)
    spy = Spy()
    arm, info = choose_arm(spy, Ctx(), np.random.default_rng(0), side_effect=True, allowed=[0, 1, 2], fallback=mt)
    assert (arm, spy.calls, info["mode"], info["propensity"]) == (2, 0, "exploit_only", 1.0)
    arm, info = choose_arm(spy, Ctx(), np.random.default_rng(0), side_effect=False, allowed=[1, 2], fallback=mt)
    assert spy.calls == 1 and info["mode"] == "masked" and info["masked_from"] == 0 and arm == 2
