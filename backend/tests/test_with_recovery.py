"""Task 22: the with_recovery wrapper (recovery.integrations.langgraph).

Fake tools, fake clock, fake LLM; nothing touches the network, and the only
store writes go under tmp_path."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recovery.integrations.langgraph as wr  # noqa: E402
from recovery.guards.runtime import BreakerConfig, BreakerState, EscalationResult, exploit_choice  # noqa: E402
from recovery.integrations import RecoveryRuntime, default_policy, with_recovery  # noqa: E402
from recovery.integrations.classify import Classification  # noqa: E402
from recovery.live.event_schema import STRATEGY_NAMES  # noqa: E402
from recovery.live.executors.base import TOOL_CALL_RESERVE_MS, Budget, LLMReply  # noqa: E402
from recovery.live.executors.responders import CallbackResponder  # noqa: E402

RETRY, SWITCH = STRATEGY_NAMES.index("retry"), STRATEGY_NAMES.index("switch_tool")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Outage:
    """A primary that times out while ``down`` and a backup that always works."""

    def __init__(self, clock=None, latency_s=0.0):
        self.down, self.primary_calls, self.backup_calls = True, 0, 0
        self.clock, self.latency_s = clock, latency_s

    def primary(self):
        def primary_api(q: str):
            self.primary_calls += 1
            if self.clock:
                self.clock.t += self.latency_s
            if self.down:
                raise TimeoutError("read timed out")
            return {"src": "primary", "q": q}
        return primary_api

    def backup(self):
        def backup_api(q: str):
            self.backup_calls += 1
            return {"src": "backup", "q": q}
        return backup_api


def _rt(clock, **kw):
    kw.setdefault("breaker", BreakerConfig(failure_threshold=3, cooldown_s=10.0))
    kw.setdefault("seed", 7)
    return RecoveryRuntime(clock=clock, sleep=clock.sleep, **kw)


# ---------------------------------------------------------------- breaker in a simulated outage

def test_breaker_opens_routes_to_backup_and_recovers_through_half_open():
    c, o = Clock(), Outage()
    rt = _rt(c)
    api = rt.wrap(o.primary(), backup=o.backup())
    br = rt.breakers.get("primary_api")

    while br.state is BreakerState.CLOSED:            # agent + recovery calls fail until the threshold
        api("x")
    assert o.primary_calls == 3 and br.state is BreakerState.OPEN

    before = o.primary_calls
    outs = [api("x") for _ in range(5)]             # open: straight to the backup, primary never called
    assert o.primary_calls == before and all(x == {"src": "backup", "q": "x"} for x in outs)
    assert {r["purpose"] for r in list(rt.log.calls)[-5:]} == {"routed_from:primary_api"}

    c.t += 10.5                                      # cooldown over, outage still on: one probe, it fails
    out = api("x")
    assert o.primary_calls == before + 1 and br.state is BreakerState.OPEN
    assert out == {"src": "backup", "q": "x"}        # recovered by switch_tool (retry masked: breaker open)

    o.down = False
    c.t += 10.5                                      # outage over: the probe succeeds and closes the breaker
    assert api("x") == {"src": "primary", "q": "x"} and br.state is BreakerState.CLOSED
    assert api("y") == {"src": "primary", "q": "y"}
    assert [(t["from"], t["to"]) for t in rt.breakers.transitions if t["tool"] == "primary_api"] == [
        ("closed", "open"), ("open", "half_open"), ("half_open", "open"), ("open", "half_open"),
        ("half_open", "closed")]


# ---------------------------------------------------------------- side-effect tools

class SpyPolicy:
    """PGTS with a counter on select()."""

    def __init__(self):
        self.inner, self.name, self.selects, self.updates = default_policy(), "spy-pgts", 0, 0

    def select(self, ctx, rng):
        self.selects += 1
        return self.inner.select(ctx, rng)

    def update(self, ctx, a, r):
        self.updates += 1
        self.inner.update(ctx, a, r)

    def reset(self):
        self.inner.reset()

    def posterior_mean(self):
        return self.inner.posterior_mean()


def test_side_effect_tool_is_never_explored_over_10000_seeded_failures(monkeypatch):
    rng = random.Random(22)
    choices = []
    real = wr.choose_arm

    def recording(policy, ctx, g, *, side_effect, allowed, fallback):
        expected = exploit_choice(policy.posterior_mean()[ctx.predicted_type], allowed)
        arm, info = real(policy, ctx, g, side_effect=side_effect, allowed=allowed, fallback=fallback)
        choices.append((arm, expected, side_effect, info["mode"]))
        return arm, info

    monkeypatch.setattr(wr, "choose_arm", recording)
    spy, c = SpyPolicy(), Clock()

    def book_slot(n: int):
        if rng.random() < 0.7:
            raise RuntimeError("HTTP 503 upstream busy")
        return {"booked": n}

    def book_slot_mirror(n: int):
        if rng.random() < 0.5:
            raise RuntimeError("HTTP 500 mirror error")
        return {"booked": n, "mirror": True}

    rt = _rt(c, policy=spy, breaker=BreakerConfig(failure_threshold=10**9))
    book = rt.wrap(book_slot, backup=book_slot_mirror, side_effect=True,
                   translate=lambda a: dict(a))
    i = 0
    while len(choices) < 10_000:
        book(i)
        i += 1
    assert spy.selects == 0 and spy.updates == len(choices)
    assert all(side and mode == "exploit_only" and arm == exp for arm, exp, side, mode in choices)
    assert {a.executor_config["selection"]["mode"] for a in rt.log.attempts} == {"exploit_only"}
    assert all(a.selection_propensity == 1.0 for a in rt.log.attempts)


def test_ordinary_tools_do_use_policy_select():
    spy, c, o = SpyPolicy(), Clock(), Outage()
    rt = _rt(c, policy=spy)
    rt.wrap(o.primary(), backup=o.backup())("x")
    assert spy.selects >= 1


# ---------------------------------------------------------------- budget never exceeded

class RandomLLM:
    """Respects the v2 budget assumptions: prompt tokens <= ceil(len/2),
    completion <= max_tokens, latency <= timeout."""

    model = "fake/llm"

    def __init__(self, rng, clock, names):
        self.rng, self.clock, self.names = rng, clock, names

    def complete(self, prompt, max_tokens, timeout_s):
        self.clock.t += self.rng.uniform(0, timeout_s)
        roll = self.rng.random()
        if roll < 0.4:
            text = json.dumps({"tool": self.rng.choice(self.names), "args": {"q": "x"}})
        elif roll < 0.7:
            text = json.dumps({"satisfies": self.rng.random() < 0.5, "reason": "r"})
        elif roll < 0.85:
            text = "Which city do you mean?"
        else:
            text = "garbage"
        return LLMReply(text, self.model, self.rng.randint(1, (len(prompt) + 1) // 2),
                        self.rng.randint(1, max_tokens), 1.0)


def test_budget_is_never_exceeded_seeded_randomised():
    rng = random.Random(2022)
    c = Clock()
    names = ["svc_a", "svc_b", "svc_c"]
    p_fail = {}

    def make(name):
        def fn(q: str):
            c.t += rng.uniform(0, TOOL_CALL_RESERVE_MS / 1000.0)   # within the declared worst case
            if rng.random() < p_fail[name]:
                raise rng.choice([TimeoutError("timed out"), RuntimeError("HTTP 502 bad gateway"),
                                  ConnectionError("reset")])
            return f"{name}:{q}"
        fn.__name__ = name
        return fn

    checked = 0
    for task_i in range(500):
        for n in names:
            p_fail[n] = rng.random()
        budget = Budget(max_attempts=rng.randint(1, 5), max_wall_ms=rng.uniform(12_000, 60_000),
                        max_tokens=rng.randint(200, 3000))
        rt = _rt(c, budget=budget, llm=RandomLLM(rng, c, names), seed=task_i,
                 responder=CallbackResponder(lambda q, t: "Kolkata"),
                 breaker=BreakerConfig(failure_threshold=rng.randint(1, 5), cooldown_s=rng.uniform(1, 30)))
        tools = {n: rt.wrap(make(n), backup=names[(i + 1) % 3]) for i, n in enumerate(names)}
        with rt.task(f"t{task_i}", goal="look something up") as scope:
            for _ in range(rng.randint(1, 4)):
                tools[rng.choice(names)]("x")
            b, used = scope.budget.budget, scope.budget
            assert used.attempts <= b.max_attempts and used.tokens <= b.max_tokens
            assert used.wall_ms <= b.max_wall_ms + 1e-6 and not used.exceeded()
            assert not any(a.budget_exceeded for a in rt.log.attempts)
            checked += len(rt.log.attempts)
    assert checked >= 400    # the test actually exercised recovery (495 attempts with this seed)


# ---------------------------------------------------------------- escalation

class LowConfidence:
    def classify(self, error_text, tool_name, http_status=None, exc=None):
        return Classification("hallucinated_output", 0.2, "trained")


def test_escalation_on_low_confidence_runs_no_strategy(tmp_path):
    c, o = Clock(), Outage()
    rt = _rt(c, classifier=LowConfidence(), responder=CallbackResponder(lambda q, t: "skip this step"),
             store=tmp_path / "log")
    out = rt.wrap(o.primary(), backup=o.backup())("x")
    assert isinstance(out, EscalationResult) and out.reason == "low_confidence"
    assert out.confidence == 0.2 and out.attempts_made == 0 and not rt.log.attempts
    assert out.handled and out.answer == "skip this step" and o.backup_calls == 0
    assert len(list((tmp_path / "log" / "escalations").glob("*.json"))) == 1


def test_escalation_on_budget_exhaustion():
    c = Clock()

    def always_down(q: str):
        raise TimeoutError("timed out")

    rt = _rt(c, budget=Budget(max_attempts=1))
    out = rt.wrap(always_down)("x")
    assert isinstance(out, EscalationResult) and out.reason == "budget_exhausted"
    assert out.attempts_made == 1 and out.budget_used["attempts"] == 1 and not out.handled
    assert rt.log.escalations[0]["reason"] == "budget_exhausted"


def test_escalation_on_langchain_tool_returns_a_message():
    from langchain_core.tools import tool

    @tool
    def flaky(q: str) -> str:
        """Always times out."""
        raise TimeoutError("timed out")

    c = Clock()
    w = with_recovery(flaky, budget=Budget(max_attempts=1), clock=c, sleep=c.sleep)
    msg = w.invoke({"q": "x"})
    assert isinstance(msg, str) and msg.startswith("[escalated: budget_exhausted]")


# ---------------------------------------------------------------- a never-failing tool

def test_never_failing_callable_adds_no_calls_and_changes_no_output(tmp_path):
    calls = []

    def lookup(q: str, n: int = 2):
        calls.append((q, n))
        return {"q": q, "items": list(range(n))}

    spy = SpyPolicy()
    w = with_recovery(lookup, policy=spy, store=tmp_path / "log")
    rng = random.Random(5)
    for _ in range(200):
        q, n = str(rng.random()), rng.randint(0, 5)
        expected = lookup(q, n=n)
        got = w(q, n=n)
        assert got == expected
    assert len(calls) == 400 and calls[::2] == calls[1::2]
    rt = w.__recovery_runtime__
    assert not rt.log.attempts and not rt.log.escalations and spy.selects == spy.updates == 0
    assert not (tmp_path / "log").exists() or not any((tmp_path / "log").rglob("*"))
    assert w.__name__ == "lookup" and w.__doc__ == lookup.__doc__


def test_never_failing_langchain_tool_changes_no_output():
    from langchain_core.tools import tool

    n = {"calls": 0}

    @tool
    def echo(x: int, label: str = "v") -> str:
        """Echo x."""
        n["calls"] += 1
        return f"{label}={x * 2}"

    w = with_recovery(echo)
    for x in range(50):
        assert w.invoke({"x": x, "label": "k"}) == echo.invoke({"x": x, "label": "k"})
    assert n["calls"] == 100 and w.name == "echo" and w.args == echo.args


def test_langgraph_toolnode_recovers_through_the_backup():
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @tool
    def primary(q: str) -> str:
        """Primary lookup (down)."""
        raise TimeoutError("timed out")

    @tool
    def secondary(q: str) -> str:
        """Backup lookup."""
        return f"secondary:{q}"

    c = Clock()
    rt = _rt(c, policy=_FixedPolicy(SWITCH))
    g = StateGraph(MessagesState)
    g.add_node("tools", ToolNode([rt.wrap(primary, backup=secondary)]))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    msg = AIMessage(content="", tool_calls=[{"name": "primary", "args": {"q": "Kolkata"}, "id": "c1",
                                             "type": "tool_call"}])
    out = g.compile().invoke({"messages": [msg]})
    assert out["messages"][-1].content == "secondary:Kolkata"
    assert rt.log.attempts[0].strategy == "switch_tool" and rt.log.attempts[0].context == "with_recovery"


class _FixedPolicy:
    def __init__(self, arm):
        self.arm, self.name = arm, f"fixed-{arm}"

    def select(self, ctx, rng):
        return self.arm

    def update(self, ctx, a, r):
        pass

    def reset(self):
        pass


# ---------------------------------------------------------------- store, executors, masking

def test_store_refuses_live_pilot_and_v3_paths(tmp_path):
    (tmp_path / "v3store" / "v3_sessions").mkdir(parents=True)
    for bad in (tmp_path / "v3store", tmp_path / "v3store" / "sub", tmp_path / "results" / "live_pilot_v3",
                tmp_path / "faarf-v3" / "store"):
        with pytest.raises(ValueError, match="refusing store"):
            RecoveryRuntime(store=bad)
    RecoveryRuntime(store=tmp_path / "ok")


def test_live_tool_names_use_the_unchanged_v2_executors_and_others_the_generic_ones():
    c = Clock()

    def wttr(location: str):
        raise TimeoutError("timed out")

    def open_meteo(latitude: float, longitude: float):
        return {"current_weather": {"temperature": 30.0}}

    def my_api(q: str):
        raise TimeoutError("timed out")

    rt = _rt(c, policy=_FixedPolicy(SWITCH))
    w = rt.wrap(wttr)
    rt.wrap(open_meteo)
    out = w(location="22.57,88.36")
    a = rt.log.attempts[0]
    assert a.executor_config["executors"] == "live_v2" and a.strategy == "switch_tool"
    assert a.result["evidence"]["translation"]["translated_args"] == {"latitude": 22.57, "longitude": 88.36}
    assert out == {"current_weather": {"temperature": 30.0}}
    assert rt._executor("wttr", "retry") is rt._live_execs["retry"]
    rt.wrap(my_api)
    assert rt._executor("my_api", "replan") is rt._generic_execs["replan"]


def test_llm_arms_are_masked_without_an_llm():
    c = Clock()
    rt = _rt(c)

    def solo(q: str):
        return q

    def wttr(location: str):
        return location

    rt.wrap(solo)
    rt.wrap(wttr)
    names = lambda arms: [STRATEGY_NAMES[i] for i in arms]   # noqa: E731
    assert names(rt._runnable_arms("solo")) == ["retry"]
    assert names(rt._runnable_arms("wttr")) == ["retry", "switch_tool", "verify"]   # schema-only verify
    rt2 = _rt(c, llm=RandomLLM(random.Random(0), c, ["solo"]))
    rt2.wrap(solo)
    assert names(rt2._runnable_arms("solo")) == ["retry", "replan", "re_ground", "verify", "clarify"]


def test_policy_is_updated_with_the_outcome():
    c, o = Clock(), Outage()
    o.down = True
    pol = _FixedPolicy(SWITCH)
    seen = []
    pol.update = lambda ctx, a, r: seen.append((a, r))
    rt = _rt(c, policy=pol)
    assert rt.wrap(o.primary(), backup=o.backup())("x") == {"src": "backup", "q": "x"}
    assert seen == [(SWITCH, 1)]


# ---------------------------------------------------------------- classifier

def test_classifier_rules_then_trained_fallback(tmp_path):
    import requests

    from recovery.integrations.classify import RuleThenTrainedClassifier
    clf = RuleThenTrainedClassifier()
    cases = [
        ("TimeoutError: HTTPSConnectionPool(host='wttr.in', port=443): Read timed out.", None, None, "tool_timeout"),
        ("ReadTimeout: whatever", None, requests.exceptions.ReadTimeout("x"), "tool_timeout"),
        ("HTTP 503: Service Unavailable", None, None, "tool_error"),
        ("RuntimeError: HTTP 404 not found", None, None, "tool_error"),
        ("HTTPError: 500 Server Error", 500, None, "tool_error"),
        ("ConnectionError: reset by peer", None, None, "tool_error"),
        ("CircuitOpen: wttr breaker is open; call not issued", None, None, "tool_error"),
        ("ToolFailure: ConnectionError: reset by peer", None, None, "tool_error"),
        ("ToolFailure: HTTP 502: bad gateway", None, None, "tool_error"),
        ("ToolFailure: TimeoutError: read", None, None, "tool_timeout"),
    ]
    for text, status, exc, want in cases:
        c = clf.classify(text, "wttr", status, exc)
        assert (c.failure_type, c.confidence) == (want, 1.0) and c.source.startswith("rule:"), text
    c = clf.classify("The answer mentions a city that does not exist", "wikipedia")
    assert c.source == "trained" and 0.0 < c.confidence <= 1.0 and abs(sum(c.proba.values()) - 1) < 1e-3   # proba stored rounded to 4 dp
    missing = RuleThenTrainedClassifier(model_path=tmp_path / "nope.joblib")
    assert missing.classify("odd output", "x") == Classification("unknown", 0.0, "none")
