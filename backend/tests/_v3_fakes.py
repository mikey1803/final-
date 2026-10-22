"""Shared fakes for the live-pilot v3 tests: fake services (no network), a
fake LLM that records every prompt, a scripted selector, and tiny frozen-shape
libraries. Genuine answers are unique marker strings so a test can search for
them anywhere they must not appear (acceptance test A4)."""

from __future__ import annotations

import json
from typing import Callable, Optional

from recovery.live.executors.base import LLMReply
from recovery.live.tools import ToolCallResult
from recovery.live.v3.libraries import build_libraries

CAPITAL = {"india": "Genuinopolis", "france": "Parisgenuine", "japan": "Tokyogenuine"}


def _ok(text: str, latency: float = 50.0) -> ToolCallResult:
    return ToolCallResult(True, 200, text, None, latency, full_text=text)


def _err(status: int, text: str) -> ToolCallResult:
    return ToolCallResult(False, status, text, f"HTTP {status}: {text}", 50.0, full_text=text)


def _timeout() -> ToolCallResult:
    return ToolCallResult(False, None, None, "TimeoutError: fake timed out", 1.0)


def body_for(tool: str, args: dict) -> str:
    if tool == "countriesnow":
        c = str(args.get("country", "")).strip().casefold()
        return json.dumps({"error": False, "msg": "ok", "data": {"name": c, "capital": CAPITAL.get(c, "Nowhere")}})
    if tool == "wikipedia":
        t = str(args.get("title"))
        return json.dumps({"type": "standard", "title": t, "coordinates": {"lat": 1.0, "lon": 2.0},
                           "extract": f"GENUINE-EXTRACT about {t}."})
    if tool == "open_meteo":
        return json.dumps({"current_weather": {"temperature": 30.5, "time": "2026-10-22T00:00"}},
                          separators=(",", ":"))
    if tool == "wttr":
        return json.dumps({"current_condition": [{"temp_C": "30"}]}, indent=4)
    if tool == "jsonplaceholder":
        i = int(args.get("id", 1))
        return json.dumps({"userId": 1, "id": i, "title": f"genuine jp title {i}", "completed": False})
    if tool == "dummyjson":
        i = int(args.get("id", 1))
        return json.dumps({"id": i, "todo": f"genuine dj todo {i}", "completed": False, "userId": 1})
    raise KeyError(tool)


class FakeServices:
    """tools dict for FlakyWrapperV3: (args, malformed, timeout) -> ToolCallResult.

    ``natural_fail``: tools whose next un-injected calls fail naturally (count per tool).
    ``accept_malformed``: simulate an instrument fault (a malformed request that succeeds).
    """

    def __init__(self, natural_fail: Optional[dict] = None, accept_malformed: bool = False):
        self.calls: list[tuple] = []
        self.natural_fail = dict(natural_fail or {})
        self.accept_malformed = accept_malformed

    def tools(self) -> dict:
        return {t: self._make(t) for t in ("wikipedia", "countriesnow", "open_meteo", "wttr",
                                          "jsonplaceholder", "dummyjson")}

    def _make(self, tool: str) -> Callable:
        def fn(args, malformed, timeout):
            self.calls.append((tool, dict(args), malformed, timeout))
            read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
            if read_timeout <= 0.2:
                return _timeout()
            if malformed and not self.accept_malformed:
                return _err(404, '{"error": true}')
            if self.natural_fail.get(tool, 0) > 0:
                self.natural_fail[tool] -= 1
                return _err(503, "The service is overloaded")
            return _ok(body_for(tool, args))
        return fn


class RecordingLLM:
    """Returns a scripted call for selection-style prompts, a pass verdict for
    judge prompts, a question for clarify; records every prompt."""

    model = "fake/model-1"
    reasoning_effort = "low"

    def __init__(self, call: Optional[dict] = None, verdict: bool = True):
        self.prompts: list[str] = []
        self.call = call
        self.verdict = verdict

    def complete(self, prompt, max_tokens, timeout_s):
        self.prompts.append(prompt)
        if "satisfies" in prompt:
            text = json.dumps({"satisfies": self.verdict, "reason": "fake"})
        elif "clarifying question" in prompt:
            text = "Which one?"
        else:
            text = json.dumps(self.call or {"tool": "countriesnow", "args": {"country": "India"}})
        return LLMReply(text, "fake/model-1", 50, 20, 5.0)


class ScriptedSelector:
    """``select(step_goal)``: returns/raises the scripted items in order, then repeats the last."""

    def __init__(self, items: list):
        self.items = list(items)
        self.goals: list[str] = []

    def __call__(self, step_goal: str):
        self.goals.append(step_goal)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return json.loads(json.dumps(item))


def tiny_libraries():
    capitals = [{"country": "India", "capital": "Genuinopolis"}, {"country": "France", "capital": "Parisgenuine"},
                {"country": "Japan", "capital": "Tokyogenuine"}, {"country": "Peru", "capital": "Limawrong"}]
    extracts = [{"kind": "city", "key": "Genuinopolis", "title": "Genuinopolis", "canonical_title": "Genuinopolis",
                 "extract": "WRONG-CITY-EXTRACT one."},
                {"kind": "city", "key": "Limawrong", "title": "Limawrong", "canonical_title": "Limawrong",
                 "extract": "WRONG-CITY-EXTRACT two."},
                {"kind": "country", "key": "India", "title": "India", "canonical_title": "India",
                 "extract": "WRONG-COUNTRY-EXTRACT India."},
                {"kind": "country", "key": "Peru", "title": "Peru", "canonical_title": "Peru",
                 "extract": "WRONG-COUNTRY-EXTRACT Peru."},
                {"kind": "other", "key": "Python_(programming_language)", "title": "Python_(programming_language)",
                 "canonical_title": "Python (programming language)", "extract": "WRONG-OTHER-EXTRACT Python."},
                {"kind": "other", "key": "Perl", "title": "Perl", "canonical_title": "Perl",
                 "extract": "WRONG-OTHER-EXTRACT Perl."}]
    jp = [{"id": i, "text": f"library jp title {i}"} for i in range(1, 11)]
    dj = [{"id": i, "text": f"library dj todo {i}"} for i in range(1, 11)]
    return build_libraries(capitals, extracts, jp, dj, {"test": "tiny"})
