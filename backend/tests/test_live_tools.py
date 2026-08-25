"""Tests for recovery.live.tools: request-result shaping and the
FlakyWrapper's injection logic. No real network calls -- requests.request
is monkeypatched throughout, per the project's record-and-replay/mock
convention for external services in the standard test suite (see
docs/live_pilot/strategy_definitions.md's Phase-1 test plan)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

import recovery.live.tools as tools_mod
from recovery.live.tools import (
    TOOLS, TOOL_ALTERNATIVES, FlakyWrapper, _do_request, health_check_all_tools,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.ok = 200 <= status_code < 300


def test_do_request_success(monkeypatch):
    monkeypatch.setattr(tools_mod.requests, "request", lambda *a, **k: _FakeResponse(200, '{"ok":true}'))
    result = _do_request("GET", "https://example.invalid/x", timeout=5.0)
    assert result.success is True
    assert result.http_status == 200
    assert result.error_text is None


def test_do_request_http_error(monkeypatch):
    monkeypatch.setattr(tools_mod.requests, "request", lambda *a, **k: _FakeResponse(404, "not found"))
    result = _do_request("GET", "https://example.invalid/x", timeout=5.0)
    assert result.success is False
    assert result.http_status == 404
    assert "404" in result.error_text


def test_do_request_timeout(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.Timeout("timed out")
    monkeypatch.setattr(tools_mod.requests, "request", _raise)
    result = _do_request("GET", "https://example.invalid/x", timeout=0.2)
    assert result.success is False
    assert result.http_status is None
    assert "Timeout" in result.error_text


def test_do_request_other_exception(monkeypatch):
    def _raise(*a, **k):
        raise requests.exceptions.ConnectionError("dns failure")
    monkeypatch.setattr(tools_mod.requests, "request", _raise)
    result = _do_request("GET", "https://example.invalid/x", timeout=5.0)
    assert result.success is False
    assert "ConnectionError" in result.error_text


def test_tool_alternatives_covers_every_tool():
    assert set(TOOL_ALTERNATIVES.keys()) == set(TOOLS.keys())


def test_tool_alternatives_are_bidirectional_and_no_self_pairs():
    for tool, peer in TOOL_ALTERNATIVES.items():
        assert peer != tool, f"{tool} must not be its own alternative"
        assert TOOL_ALTERNATIVES[peer] == tool, f"{tool}<->{peer} must be symmetric"


def test_flaky_wrapper_rejects_out_of_range_rate():
    with pytest.raises(ValueError):
        FlakyWrapper(injection_rate=1.5)
    with pytest.raises(ValueError):
        FlakyWrapper(injection_rate=-0.1)


def test_flaky_wrapper_unknown_tool_raises():
    wrapper = FlakyWrapper(injection_rate=0.0)
    with pytest.raises(ValueError):
        wrapper.call("not_a_real_tool", {})


def test_flaky_wrapper_never_injects_at_rate_zero(monkeypatch):
    import random

    def fake_tool(args, malformed, timeout):
        assert malformed is False
        return tools_mod.ToolCallResult(success=True, http_status=200, body="ok", error_text=None, latency_ms=1.0)

    monkeypatch.setitem(TOOLS, "fake", fake_tool)
    wrapper = FlakyWrapper(injection_rate=0.0, rng=random.Random(0))
    for _ in range(50):
        result = wrapper.call("fake", {})
        assert result.injected is False


def test_flaky_wrapper_always_injects_at_rate_one(monkeypatch):
    import random

    def fake_tool(args, malformed, timeout):
        return tools_mod.ToolCallResult(success=True, http_status=200, body="ok", error_text=None, latency_ms=1.0)

    monkeypatch.setitem(TOOLS, "fake", fake_tool)
    wrapper = FlakyWrapper(injection_rate=1.0, rng=random.Random(0))
    kinds_seen = set()
    for _ in range(50):
        result = wrapper.call("fake", {})
        assert result.injected is True
        kinds_seen.add(result.injected_kind)
    assert kinds_seen <= {"timeout", "malformed_request"}


def test_health_check_all_tools_covers_every_tool(monkeypatch):
    monkeypatch.setattr(tools_mod.requests, "request", lambda *a, **k: _FakeResponse(200, "ok"))
    results = health_check_all_tools()
    assert set(results.keys()) == set(TOOLS.keys())
    assert all(r["healthy"] is True for r in results.values())
    assert all(r["http_status"] == 200 for r in results.values())


def test_health_check_all_tools_records_a_down_tool(monkeypatch):
    calls = {"n": 0}

    def flaky_request(*a, **k):
        calls["n"] += 1
        # wikipedia is checked first (dict insertion order); make it fail.
        if calls["n"] == 1:
            return _FakeResponse(503, "service unavailable")
        return _FakeResponse(200, "ok")

    monkeypatch.setattr(tools_mod.requests, "request", flaky_request)
    results = health_check_all_tools()
    assert results["wikipedia"]["healthy"] is False
    assert results["wikipedia"]["http_status"] == 503
    assert results["countriesnow"]["healthy"] is True


def test_flaky_wrapper_injection_distribution_near_rate(monkeypatch):
    """Sanity-check distribution at a moderate rate, same eyeball-style
    check as test_injector.py's test_injector_distribution."""
    import random

    def fake_tool(args, malformed, timeout):
        return tools_mod.ToolCallResult(success=True, http_status=200, body="ok", error_text=None, latency_ms=1.0)

    monkeypatch.setitem(TOOLS, "fake", fake_tool)
    wrapper = FlakyWrapper(injection_rate=0.3, rng=random.Random(42))

    n = 500
    n_injected = sum(1 for _ in range(n) if wrapper.call("fake", {}).injected)
    rate = n_injected / n
    print(f"\ninjection_rate=0.3 target, observed={rate:.3f} over n={n}")
    assert 0.2 < rate < 0.4, "observed injection rate should land near the configured rate"
