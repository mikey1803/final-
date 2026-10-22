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
    assert (arm, spy.calls, info["selection_mode"], info["propensity"]) == (2, 0, "exploit_only", 1.0)
    arm, info = choose_arm(spy, Ctx(), np.random.default_rng(0), side_effect=False, allowed=[1, 2], fallback=mt)
    assert spy.calls == 1 and info["selection_mode"] == "masked" and info["masked_from"] == 0 and arm == 2
    assert info["propensity"] == 1.0 and info["propensity_known"] and info["masked_out_arms"] == [0]


# ---------------------------------------------------------------- Thompson propensity

def _p_b_beats_a(a_a, b_a, a_b, b_b):
    """Closed form P(theta_B > theta_A), theta ~ Beta, integer a_b (Evan Miller's formula)."""
    from math import exp, lgamma

    def lbeta(x, y):
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    return sum(exp(lbeta(a_a + i, b_a + b_b) - lbeta(1 + i, b_b) - lbeta(a_a, b_a)) / (b_b + i)
               for i in range(int(a_b)))


def _two_arm_pgts():
    from recovery.policies.hybrid import PriorInitializedThompson
    pol = PriorInitializedThompson(np.array([0]), 2, strength=0.0, name="pgts-2")

    class Ctx:
        predicted_type = 0

    for r in (1,) * 7 + (0,) * 3:                 # arm 0: Beta(8, 4)
        pol.update(Ctx(), 0, r)
    for r in (1,) * 4 + (0,) * 6:                 # arm 1: Beta(5, 7)
        pol.update(Ctx(), 1, r)
    return pol, Ctx()


def test_thompson_propensity_matches_the_closed_form_on_two_arms():
    from recovery.guards.runtime import thompson_posterior, thompson_propensity
    pol, _ = _two_arm_pgts()
    a, b = thompson_posterior(pol, 0)
    assert a.tolist() == [8.0, 5.0] and b.tolist() == [4.0, 7.0]
    p1 = _p_b_beats_a(8, 4, 5, 7)                 # P(arm 1 picked)
    exact = {0: 1.0 - p1, 1: p1}
    for arm in (0, 1):
        est = thompson_propensity(a, b, arm, [0, 1], 0, 200_000, np.random.default_rng(arm))
        assert abs(est - exact[arm]) < 0.005, (arm, est, exact[arm])


def test_choose_arm_records_a_seeded_monte_carlo_propensity_within_tolerance():
    pol, ctx = _two_arm_pgts()
    p1 = _p_b_beats_a(8, 4, 5, 7)
    exact = {0: 1.0 - p1, 1: p1}
    seen = set()
    for seed in range(40):
        for m in (1000, 20_000):
            arm, info = choose_arm(pol, ctx, np.random.default_rng(seed), side_effect=False, allowed=[0, 1],
                                   fallback=MeanTable(1, 2), propensity_rng=np.random.default_rng(10_000 + seed),
                                   propensity_draws=m)
            p = exact[arm]
            tol = 4 * np.sqrt(p * (1 - p) / m)
            assert info["selection_mode"] == "policy" and info["propensity_method"] == "monte_carlo_posterior"
            assert info["propensity_known"] and info["propensity_draws"] == m
            assert abs(info["propensity"] - p) < tol, (seed, m, arm, info["propensity"], p)
            seen.add(arm)
        again = choose_arm(pol, ctx, np.random.default_rng(seed), side_effect=False, allowed=[0, 1],
                           fallback=MeanTable(1, 2), propensity_rng=np.random.default_rng(10_000 + seed),
                           propensity_draws=20_000)
        assert again == (arm, info)              # same seeds -> same pick and same estimate
    assert seen == {0, 1}


def test_propensity_accounts_for_masking_when_the_pick_is_runnable():
    from recovery.guards.runtime import thompson_propensity
    a, b = np.array([8.0, 5.0, 1.0]), np.array([4.0, 7.0, 50.0])
    # arm 2 masked, fallback arm 0: arm 0's propensity includes draws that landed on arm 2
    plain = thompson_propensity(a, b, 0, [0, 1, 2], 0, 50_000, np.random.default_rng(1))
    masked = thompson_propensity(a, b, 0, [0, 1], 0, 50_000, np.random.default_rng(1))
    assert masked >= plain
