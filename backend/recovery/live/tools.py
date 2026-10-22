"""The six live tools plus the fault-injection wrapper.

Every endpoint below was verified live by direct HTTP request before being
wired in here (2026-08-20) -- see ``docs/live_pilot/strategy_definitions.md``
for the verification notes, including two real access-policy changes caught
in the process (REST Countries and reqres.in both moved to key-gated access
since originally proposed) and one broken-redirect bug found in
countriesnow.space's own documented POST endpoint (worked around by calling
its GET form directly, never following the documented POST route).

Fault injection is a wrapper around normal calls, not a 7th tool, applied
uniformly and SYMMETRICALLY across both tools in a peer pair (see
``TOOL_ALTERNATIVES`` and the frozen doc's rationale -- asymmetric injection
would let ``switch_tool`` trivially escape an injected fault by construction).
Both injection kinds produce genuine errors, never synthesized text:

- timeout injection: a real call issued with an artificially short
  client-side timeout, so it's a real ``requests`` timeout exception.
- malformed-request injection: a deliberately bad real request, so the
  error text captured is the real server's own reply.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

__all__ = [
    "ToolCallResult", "TOOLS", "TOOL_ALTERNATIVES", "FlakyWrapper",
    "health_check_all_tools",
]

DEFAULT_TIMEOUT_S = 10.0
INJECTED_TIMEOUT_S = 0.2  # short enough to abort any of these real calls

# Wikipedia's REST API 403s on requests' default User-Agent ("please set a
# user-agent and respect our robot policy" -- caught live, not assumed).
# Applied to every tool, not just Wikipedia: harmless for the others, and
# good practice generally rather than a Wikipedia-specific special case.
_USER_AGENT = "FAARF-live-pilot/1 (student research project; contact via repo)"
_REQUEST_HEADERS = {"User-Agent": _USER_AGENT}


@dataclass
class ToolCallResult:
    """Uniform result shape every tool wrapper returns, real or injected."""

    success: bool
    http_status: Optional[int]
    body: Optional[str]       # raw response text, truncated to _MAX_BODY_CHARS
    error_text: Optional[str]
    latency_ms: float
    injected: bool = False
    injected_kind: Optional[str] = None  # "timeout" | "malformed_request" | None
    # Untruncated response text (Task 21). Never stored in a FailureEvent --
    # events keep the truncated ``body`` exactly as before; only recovery
    # executors that must parse the whole response (VERIFY's v2 schema
    # check) read this. None when no response was received.
    full_text: Optional[str] = None
    # v3 call metadata set only by ``recovery.live.v3.flaky.FlakyWrapperV3``
    # (injection mode, key, outage index, degradation flags). Never holds a
    # genuine value -- those live in the wrapper's injection log only. None
    # for every v1/v2 call, so v1/v2 behaviour and records are unchanged.
    v3: Optional[dict] = None


_MAX_BODY_CHARS = 2000


def _truncate(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= _MAX_BODY_CHARS else text[:_MAX_BODY_CHARS] + "...[truncated]"


def _do_request(
    method: str, url: str, *, params: Optional[dict] = None, timeout: float
) -> ToolCallResult:
    start = time.monotonic()
    try:
        resp = requests.request(method, url, params=params, timeout=timeout, headers=_REQUEST_HEADERS)
        latency_ms = (time.monotonic() - start) * 1000.0
        if resp.ok:
            return ToolCallResult(
                success=True, http_status=resp.status_code,
                body=_truncate(resp.text), error_text=None, latency_ms=latency_ms,
                full_text=resp.text,
            )
        return ToolCallResult(
            success=False, http_status=resp.status_code,
            body=_truncate(resp.text),
            error_text=f"HTTP {resp.status_code}: {_truncate(resp.text)}",
            latency_ms=latency_ms,
            full_text=resp.text,
        )
    except requests.exceptions.Timeout as exc:
        latency_ms = (time.monotonic() - start) * 1000.0
        return ToolCallResult(
            success=False, http_status=None, body=None,
            error_text=f"TimeoutError: {exc}", latency_ms=latency_ms,
        )
    except requests.exceptions.RequestException as exc:
        latency_ms = (time.monotonic() - start) * 1000.0
        return ToolCallResult(
            success=False, http_status=None, body=None,
            error_text=f"{type(exc).__name__}: {exc}", latency_ms=latency_ms,
        )


# ======================================================================
# Six tool wrappers. Each takes (args, malformed, timeout) and constructs
# a REAL request either way -- malformed=True never fabricates error text,
# it only makes a request the real server will genuinely reject.
# ======================================================================

def _wikipedia(args: dict, malformed: bool, timeout: float) -> ToolCallResult:
    title = "ZzNonexistentPage9999XyZQQQ" if malformed else args["title"]
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    return _do_request("GET", url, timeout=timeout)


def _countriesnow(args: dict, malformed: bool, timeout: float) -> ToolCallResult:
    country = "ZzNonexistentCountryland9999" if malformed else args["country"]
    # Deliberately NOT the documented POST endpoint -- it redirect-loops
    # (verified live, see module docstring). The GET form works cleanly.
    url = "https://countriesnow.space/api/v0.1/countries/capital/q"
    return _do_request("GET", url, params={"country": country}, timeout=timeout)


def _open_meteo(args: dict, malformed: bool, timeout: float) -> ToolCallResult:
    lat = 9999.0 if malformed else args["latitude"]
    lon = args.get("longitude", 0.0) if malformed else args["longitude"]
    url = "https://api.open-meteo.com/v1/forecast"
    return _do_request(
        "GET", url,
        params={"latitude": lat, "longitude": lon, "current_weather": "true"},
        timeout=timeout,
    )


def _wttr(args: dict, malformed: bool, timeout: float) -> ToolCallResult:
    location = "@@invalid@@location@@" if malformed else args["location"]
    url = f"https://wttr.in/{location}"
    return _do_request("GET", url, params={"format": "j1"}, timeout=timeout)


def _jsonplaceholder(args: dict, malformed: bool, timeout: float) -> ToolCallResult:
    todo_id = 999999 if malformed else args["id"]
    url = f"https://jsonplaceholder.typicode.com/todos/{todo_id}"
    return _do_request("GET", url, timeout=timeout)


def _dummyjson(args: dict, malformed: bool, timeout: float) -> ToolCallResult:
    todo_id = 999999 if malformed else args["id"]
    url = f"https://dummyjson.com/todos/{todo_id}"
    return _do_request("GET", url, timeout=timeout)


TOOLS: dict[str, Callable[[dict, bool, float], ToolCallResult]] = {
    "wikipedia": _wikipedia,
    "countriesnow": _countriesnow,
    "open_meteo": _open_meteo,
    "wttr": _wttr,
    "jsonplaceholder": _jsonplaceholder,
    "dummyjson": _dummyjson,
}

TOOL_DESCRIPTIONS: list[dict] = [
    {"name": "wikipedia", "description": "Look up a Wikipedia page summary by title.", "args": {"title": "str"}},
    {"name": "countriesnow", "description": "Look up a country's capital city.", "args": {"country": "str"}},
    {"name": "open_meteo", "description": "Get current weather for a latitude/longitude.", "args": {"latitude": "float", "longitude": "float"}},
    {"name": "wttr", "description": "Get current weather for a named location.", "args": {"location": "str"}},
    {"name": "jsonplaceholder", "description": "Fetch a placeholder todo item by numeric id.", "args": {"id": "int"}},
    {"name": "dummyjson", "description": "Fetch a placeholder todo item by numeric id (mirror service).", "args": {"id": "int"}},
]

# Frozen per docs/live_pilot/strategy_definitions.md -- bidirectional,
# same-category only, no cross-category substitution.
TOOL_ALTERNATIVES: dict[str, str] = {
    "wikipedia": "countriesnow",
    "countriesnow": "wikipedia",
    "open_meteo": "wttr",
    "wttr": "open_meteo",
    "jsonplaceholder": "dummyjson",
    "dummyjson": "jsonplaceholder",
}


# One known-good args set per tool, used only for health checks -- a real,
# unmalformed, uninjected call, the same shape a normal task would make.
_HEALTH_CHECK_ARGS: dict[str, dict] = {
    "wikipedia": {"title": "Python_(programming_language)"},
    "countriesnow": {"country": "India"},
    "open_meteo": {"latitude": 22.57, "longitude": 88.36},
    "wttr": {"location": "Kolkata"},
    "jsonplaceholder": {"id": 1},
    "dummyjson": {"id": 1},
}


def health_check_all_tools(timeout: float = DEFAULT_TIMEOUT_S) -> dict[str, dict]:
    """Real, uninjected, unmalformed call to each of the 6 tools, so a
    peer-tool outage is recorded as an outage at collection time rather
    than surfacing later as an unexplained switch_tool result. Meant to be
    run once per collection session and the result attached to every
    FailureEvent captured that session -- re-deriving this at replay time
    would not reflect the world state at collection time, since these are
    third-party services that can flake or recover independently.
    """
    results: dict[str, dict] = {}
    for tool_name, fn in TOOLS.items():
        args = _HEALTH_CHECK_ARGS[tool_name]
        result = fn(args, False, timeout)
        results[tool_name] = {
            "healthy": result.success,
            "http_status": result.http_status,
            "latency_ms": result.latency_ms,
            "error_text": result.error_text,
        }
    return results


class FlakyWrapper:
    """Applies the frozen background fault-injection rate to any tool call.

    One instance, one fixed ``injection_rate``, shared across every tool
    call for the whole collection run -- deliberately not per-tool, so both
    members of a peer pair face identical injection risk (the frozen doc's
    symmetry requirement for ``switch_tool``).
    """

    def __init__(self, injection_rate: float = 0.20, rng: Optional[random.Random] = None) -> None:
        if not (0.0 <= injection_rate <= 1.0):
            raise ValueError(f"injection_rate must be in [0, 1], got {injection_rate!r}")
        self.injection_rate = injection_rate
        self._rng = rng or random.Random()

    def call(self, tool_name: str, args: dict) -> ToolCallResult:
        if tool_name not in TOOLS:
            raise ValueError(f"unknown tool {tool_name!r}, must be one of {list(TOOLS)}")
        fn = TOOLS[tool_name]

        if self._rng.random() >= self.injection_rate:
            result = fn(args, False, DEFAULT_TIMEOUT_S)
            return result

        # Injected trial: coin flip between the two injection kinds.
        if self._rng.random() < 0.5:
            result = fn(args, False, INJECTED_TIMEOUT_S)
            result.injected = True
            result.injected_kind = "timeout"
            return result
        else:
            result = fn(args, True, DEFAULT_TIMEOUT_S)
            result.injected = True
            result.injected_kind = "malformed_request"
            return result
