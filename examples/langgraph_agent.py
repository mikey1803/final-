#!/usr/bin/env python
"""A minimal LangGraph agent with two live weather tools, each wrapped with
``with_recovery`` in one line (Task 22).

    python examples/langgraph_agent.py                                # scripted agent, no outage
    python examples/langgraph_agent.py --outage wttr:12 --cooldown 4  # simulated wttr outage for 12 s

The agent node is scripted -- it asks for the weather at 22.57,88.36 with the
``wttr`` tool, once per query -- so the demo makes NO LLM calls (decided
2026-10-22: the Groq account is shared with a running collection). The tool
calls are live HTTP. Without an LLM the runnable recovery arms are retry,
switch_tool and a schema-only verify.

``--outage TOOL:SECONDS`` makes that tool's calls fail for SECONDS from start
with a genuine ``requests`` ReadTimeout (normal connect timeout, 1 ms read
timeout -- the v3 D-5 technique), not a synthetic error.

``--llm groq`` builds a ReAct agent on Groq instead of the scripted node, and
gives the recovery runtime a ``GroqChatClient`` so the LLM arms (replan,
re_ground, clarify, verify's judge) can run. Reads GROQ_API_KEY from the
environment or backend/.env. Tested offline with mocked Groq responses
(``backend/tests/test_langgraph_example_groq.py``); not run live while the v3
collection shares the Groq account.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402

from recovery.guards.runtime import BreakerConfig  # noqa: E402
from recovery.integrations import RecoveryRuntime  # noqa: E402
from recovery.live.executors.base import Budget  # noqa: E402
from recovery.live.tools import DEFAULT_TIMEOUT_S, TOOLS  # noqa: E402

T0 = time.monotonic()
OUTAGE = {"tool": None, "until": 0.0}


class ToolFailure(Exception):
    """A live tool call that did not succeed; the message is the live error text."""


def _live(name: str, args: dict) -> str:
    timeout = DEFAULT_TIMEOUT_S
    if OUTAGE["tool"] == name and time.monotonic() - T0 < OUTAGE["until"]:
        timeout = (DEFAULT_TIMEOUT_S, 0.001)          # genuine ReadTimeout (D-5)
    r = TOOLS[name](args, False, timeout)
    if not r.success:
        raise ToolFailure(r.error_text)
    return r.full_text


@tool
def wttr(location: str) -> str:
    """Get current weather for a named location or a 'lat,lon' pair from wttr.in."""
    return _live("wttr", {"location": location})


@tool
def open_meteo(latitude: float, longitude: float) -> str:
    """Get current weather for a latitude/longitude from open-meteo."""
    return _live("open_meteo", {"latitude": latitude, "longitude": longitude})


def build_runtime(args, llm=None, policy=None) -> RecoveryRuntime:
    if llm is None and args.llm == "groq":
        from recovery.live.executors.llm import GroqChatClient
        llm = GroqChatClient()                       # recovery-side LLM for the LLM arms
    return RecoveryRuntime(policy=policy, budget=Budget(), llm=llm, store=args.store, seed=args.seed,
                           breaker=BreakerConfig(failure_threshold=3, cooldown_s=args.cooldown))


def scripted_graph(tools):
    """agent -> tools -> agent -> end, with a fixed tool call."""
    def agent(state: MessagesState):
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content=f"Weather result received ({len(last.content)} chars).")]}
        call = {"name": "wttr", "args": {"location": "22.57,88.36"}, "id": f"call-{time.monotonic_ns()}",
                "type": "tool_call"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def groq_graph(tools, http_client=None):
    """A ReAct agent on Groq. ``http_client`` (an ``httpx.Client``) is for tests."""
    import os

    from dotenv import load_dotenv
    from langchain_groq import ChatGroq
    from langchain.agents import create_agent
    if not os.environ.get("GROQ_API_KEY"):
        load_dotenv(_BACKEND / ".env")
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_retries=0, http_client=http_client)
    return create_agent(model, tools)


def _source(content: str) -> str:
    if content.startswith("[escalated"):
        return content.splitlines()[0]
    try:
        doc = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content[:80]
    if "current_condition" in doc:
        return f"wttr.in: {doc['current_condition'][0].get('temp_C')} C"
    if "current_weather" in doc:
        return f"open-meteo: {doc['current_weather'].get('temperature')} C"
    return content[:80]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", type=int, default=14)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between queries")
    ap.add_argument("--outage", default=None, help="TOOL:SECONDS, e.g. wttr:12")
    ap.add_argument("--cooldown", type=float, default=30.0, help="breaker cooldown (s); default 30")
    ap.add_argument("--store", default=None, help="directory for attempt/escalation logs (default: memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", choices=["scripted", "groq"], default="scripted")
    args = ap.parse_args()
    if args.outage:
        name, secs = args.outage.split(":")
        OUTAGE.update(tool=name, until=float(secs))

    rt = build_runtime(args)
    weather = rt.wrap(wttr)          # one line per tool
    meteo = rt.wrap(open_meteo)      # one line per tool
    graph = scripted_graph([weather, meteo]) if args.llm == "scripted" else groq_graph([weather, meteo])

    print(f"outage: {args.outage or 'none'}; breaker: 3 failures, cooldown {args.cooldown} s; agent: {args.llm}\n")
    seen_calls, seen_tr = 0, 0
    for q in range(1, args.queries + 1):
        with rt.task(f"q{q}", goal="Get the current weather at latitude 22.57, longitude 88.36."):
            out = graph.invoke({"messages": [HumanMessage("What's the weather at 22.57,88.36 right now?")]})
        tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        calls = list(rt.log.calls)[seen_calls:]
        seen_calls = len(rt.log.calls)
        print(f"[t={time.monotonic() - T0:5.1f}s] query {q:2d} -> {_source(tool_msgs[-1].content) if tool_msgs else '-'}")
        for c in calls:
            mark = "ok  " if c["success"] else ("BLOCKED" if not c["issued"] else "FAIL")
            err = f"  {c['error'][:70]}" if c["error"] and c["issued"] else ""
            print(f"      {c['tool']:10s} {c['purpose']:22s} {mark:7s} breaker={c['breaker_after']}{err}")
        for t in rt.breakers.transitions[seen_tr:]:
            print(f"      ** breaker {t['tool']}: {t['from']} -> {t['to']} ({t['why']})")
        seen_tr = len(rt.breakers.transitions)
        time.sleep(args.interval)

    print("\nrecovery attempts:")
    for a in rt.log.attempts:
        print(f"  {a.task_id:4s} {a.strategy:12s} {a.result['evidence']['reason']:20s} "
              f"selection={a.executor_config['selection_mode']} type={a.executor_config['classification']['type']}")
    print(f"escalations: {len(rt.log.escalations)}; final breakers: {rt.breakers.states()}")
    if args.store:
        print(f"logs: {rt.log.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
