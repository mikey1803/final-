"""Task 22: the example's ``--llm groq`` path, fully offline.

Both Groq clients are mocked: the agent's ChatGroq through an
``httpx.MockTransport``, the recovery runtime's ``GroqChatClient`` through
``requests.post``. Sockets are blocked, the API key is a fake set in the
environment (so backend/.env is never read), and the live tools are replaced
by fakes. One full wrapper recovery runs: the agent's wttr call times out,
REPLAN's first LLM call gets HTTP 429 (recorded as ``llm_error``), the second
REPLAN chooses open_meteo, and the agent finishes on that result.
"""

import importlib.util
import json
import socket
import sys
from pathlib import Path

import httpx
import pytest

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

import recovery.live.executors.llm as llm_mod  # noqa: E402
from recovery.live.event_schema import STRATEGY_NAMES  # noqa: E402
from recovery.live.tools import ToolCallResult  # noqa: E402

FAKE_KEY = "test-key-not-real"
METEO_BODY = json.dumps({"current_weather": {"temperature": 30.8, "windspeed": 5.0}})


def _load_example():
    path = _BACKEND.parent / "examples" / "langgraph_agent.py"
    spec = importlib.util.spec_from_file_location("langgraph_agent_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def offline(monkeypatch):
    def refuse(*a, **k):
        raise OSError("network disabled in this test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)


class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self.ok = status, 200 <= status < 300
        self._body, self.text = body, json.dumps(body)
        self.headers = headers or {}

    def json(self):
        return self._body


class _FixedPolicy:
    name = "fixed-replan"

    def select(self, ctx, rng):
        return STRATEGY_NAMES.index("replan")

    def update(self, ctx, a, r):
        pass

    def reset(self):
        pass


def _completion(message: dict, finish: str) -> dict:
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}}


def test_groq_path_recovers_through_a_429_then_replan(offline, monkeypatch):
    ex = _load_example()

    # live tools -> fakes (no HTTP)
    def fake_wttr(args, malformed, timeout):
        return ToolCallResult(False, None, None, "TimeoutError: HTTPSConnectionPool(host='wttr.in', port=443): "
                                                 "Read timed out. (read timeout=0.001)", 1.0)

    def fake_meteo(args, malformed, timeout):
        assert args == {"latitude": 22.57, "longitude": 88.36}
        return ToolCallResult(True, 200, METEO_BODY, None, 1.0, full_text=METEO_BODY)

    monkeypatch.setattr(ex, "TOOLS", {"wttr": fake_wttr, "open_meteo": fake_meteo})

    # recovery-side Groq (GroqChatClient -> requests.post): 429 first, then a tool choice
    posts = []
    recovery_replies = [
        _Resp(429, {"error": {"message": "Rate limit reached for model openai/gpt-oss-20b", "type": "tokens",
                              "code": "rate_limit_exceeded"}}, {"retry-after": "2"}),
        _Resp(200, {"model": "openai/gpt-oss-20b",
                    "choices": [{"message": {"content": json.dumps(
                        {"tool": "open_meteo", "args": {"latitude": 22.57, "longitude": 88.36}})}}],
                    "usage": {"prompt_tokens": 300, "completion_tokens": 25}}),
    ]

    def fake_post(url, json=None, timeout=None, headers=None):
        posts.append({"url": url, "auth": headers["Authorization"], "model": json["model"]})
        return recovery_replies[len(posts) - 1]

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)

    # agent-side Groq (ChatGroq via httpx): a wttr tool call, then a final answer
    agent_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.groq.com"
        assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
        agent_requests.append(json.loads(request.content))
        if len(agent_requests) == 1:
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "wttr", "arguments": json.dumps({"location": "22.57,88.36"})}}]}
            return httpx.Response(200, json=_completion(msg, "tool_calls"))
        return httpx.Response(200, json=_completion({"role": "assistant", "content": "It is 30.8 C."}, "stop"))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    class Args:
        llm, cooldown, store, seed = "groq", 30.0, None, 0

    rt = ex.build_runtime(Args(), policy=_FixedPolicy())
    assert isinstance(rt.llm, llm_mod.GroqChatClient)
    graph = ex.groq_graph([rt.wrap(ex.wttr), rt.wrap(ex.open_meteo)], http_client=client)
    with rt.task("groq-q1", goal="Get the current weather at latitude 22.57, longitude 88.36."):
        out = graph.invoke({"messages": [HumanMessage("What's the weather at 22.57,88.36?")]})

    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_msgs] == [METEO_BODY]
    assert isinstance(out["messages"][-1], AIMessage) and out["messages"][-1].content == "It is 30.8 C."

    first, second = rt.log.attempts
    assert (first.strategy, first.result["evidence"]["reason"]) == ("replan", "llm_error")
    assert "HTTP 429" in first.result["evidence"]["llm_error"] and first.result["attempts"] == 0
    assert (second.strategy, second.result["evidence"]["reason"]) == ("replan", "success")
    assert second.model_id == "openai/gpt-oss-20b"
    assert second.result["evidence"]["replan_chosen_tool"] == "open_meteo"
    assert second.result["cost_tokens"] == 325 and not rt.log.escalations
    assert first.executor_config["executors"] == "live_v2"

    assert len(posts) == 2 and {p["auth"] for p in posts} == {f"Bearer {FAKE_KEY}"}
    assert all(p["url"].startswith("https://api.groq.com/") for p in posts)
    assert len(agent_requests) == 2 and agent_requests[0]["model"] == "openai/gpt-oss-20b"
