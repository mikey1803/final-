"""``FlakyWrapperV3``: the v3 injection persistence modes and the read-only
resend probe (v3 §3, §4). The v1 ``recovery.live.tools.FlakyWrapper`` is not
changed.

One wrapper instance = one session block = one mode (and one outage level).

| mode | persistence | produces |
|---|---|---|
| ``transient`` | each call drawn independently (as v1): timeout or malformed request, by a coin flip | tool_timeout, tool_error |
| ``sticky_args`` | drawn once per ``(tool, normalised args)`` key per episode; a corrupted key is sent as the same malformed request every time; never a timeout | tool_error |
| ``tool_outage`` | per tool; a drawn call starts an outage of N calls to that tool (it is call 1); call N + 1 is served normally without a draw; other tools' calls don't count | tool_timeout |
| ``degraded_output`` | drawn once per key per episode; the real response's answer field is modified, HTTP 200 | malformed_output, hallucinated_output |

Error text is genuine in every mode except ``degraded_output``, which is
declared on every record (``synthetic_content``).

**Timeouts.** ``transient`` keeps v1's 0.2 s client timeout unchanged
(v3 §3.1). Note that ``requests`` applies a timeout per socket operation, not
per call, so a 0.2 s timeout does not always fire (found 2026-10-22; v1/v2
threats addendum T-A1). Outage calls must fail, so ``tool_outage`` connects
with the normal 10 s timeout and gives the read 1 ms (``OUTAGE_TIMEOUT_S``,
v3 pre-registration D-5): a genuine ``requests`` ``ReadTimeout`` (``Read timed
out. (read timeout=0.001)``), the same exception type a real slow-server
timeout raises. It failed 30/30 across all six tools on 2026-10-22. If one
ever succeeds anyway, the genuine success is returned and flagged
``outage_leak``.

**Genuine values** (v3 §3.6) are captured only in ``degraded_output`` blocks,
on every successful call to an answer-bearing tool, and stored only in
``self.log`` (the injection log) -- never in the returned result.

**Resend probe** (v3 §4): ``resend_probe`` sees the injection state as it is,
but never creates or advances it, and its own transient draws come from the
separate ``resend`` stream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from recovery.live.tools import DEFAULT_TIMEOUT_S, INJECTED_TIMEOUT_S, TOOLS, ToolCallResult, _truncate
from recovery.live.v3.answers import ANSWER_FIELDS, read_answer
from recovery.live.v3.degrade import DegradeSpec, apply_degradation, draw_spec
from recovery.live.v3.libraries import Libraries
from recovery.live.v3.seeds import stream_rng

__all__ = [
    "MODES", "OUTAGE_LEVELS", "FlakyWrapperV3", "normalise_args", "injection_key",
    "TRANSIENT_TIMEOUT_S", "OUTAGE_TIMEOUT_S",
]

MODES = ("transient", "sticky_args", "tool_outage", "degraded_output")
OUTAGE_LEVELS = (2, 5, 20)               # configuration choices (v3 §3.3)
TRANSIENT_TIMEOUT_S = INJECTED_TIMEOUT_S  # 0.2 s, unchanged from v1
# D-5: connect normally, give the read 1 ms -> a genuine requests ReadTimeout.
OUTAGE_TIMEOUT_S = (DEFAULT_TIMEOUT_S, 0.001)
_LOG_ERROR_CLIP = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_value(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return v.strip().casefold()
    if isinstance(v, dict):
        return {str(k): _norm_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_norm_value(x) for x in v]
    return str(v)


def normalise_args(args: dict) -> dict:
    """v3 §3.2: canonical JSON (sorted keys), strings stripped and case-folded,
    numbers as floats -- so {"country": "india "} and {"country": "India"} are one key."""
    return {str(k): _norm_value(v) for k, v in sorted((args or {}).items())}


def injection_key(tool: str, args: dict) -> str:
    payload = json.dumps({"tool": tool, "args": normalise_args(args)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sha(text: Optional[str]) -> Optional[str]:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class _Outage:
    active: bool = False
    index: int = 0
    outage_id: Optional[int] = None
    serve_next: bool = False


@dataclass
class FlakyWrapperV3:
    mode: str
    session_seed: int
    block_index: int
    rate: float = 0.20
    outage_N: Optional[int] = None
    libraries: Optional[Libraries] = None
    tools: dict = field(default_factory=lambda: dict(TOOLS))
    clock: Callable[[], str] = _now

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate!r}")
        if self.mode == "tool_outage":
            if not isinstance(self.outage_N, int) or self.outage_N < 2:
                raise ValueError("tool_outage needs an integer outage_N >= 2")
        elif self.outage_N is not None:
            raise ValueError("outage_N is only valid for tool_outage")
        if self.mode == "degraded_output" and self.libraries is None:
            raise ValueError("degraded_output needs the frozen plausible-wrong libraries")
        self._rng = {}
        self.seed_records = {}
        for name in ("injection", "resend", "outage", "degradation"):
            self._rng[name], self.seed_records[name] = stream_rng(self.session_seed, name, self.block_index)
        self.log: list[dict] = []
        self._by_id: dict[int, dict] = {}
        self._results: dict[int, ToolCallResult] = {}  # in memory only, for continue_after_recovery
        self._call_seq = 0
        self._keys: dict[str, Any] = {}           # per-episode key state
        self._outages = {t: _Outage() for t in self.tools}
        self._outage_counter = 0
        self.context: dict = {}

    # ---------------------------------------------------------------- params
    def params(self) -> dict:
        return {"mode": self.mode, "rate": self.rate, "outage_N": self.outage_N, "block_index": self.block_index,
                "transient_timeout_s": TRANSIENT_TIMEOUT_S, "outage_timeout_s": OUTAGE_TIMEOUT_S,
                "seeds": self.seed_records,
                "libraries": None if self.libraries is None else self.libraries.hashes}

    def start_episode(self, episode_id: str) -> None:
        """Sticky and degraded keys persist for one episode (v3 §3.2, §3.4); outages span the block."""
        self._keys = {}
        self._results = {}
        self.context = {"episode_id": episode_id}

    def set_context(self, **kw) -> None:
        self.context.update(kw)

    # ---------------------------------------------------------------- lookups (instrument only)
    def entry(self, call_id: int) -> Optional[dict]:
        return self._by_id.get(call_id)

    def genuine(self, call_id: int) -> Optional[dict]:
        e = self._by_id.get(call_id)
        if e is None or "genuine_value" not in e:
            return None
        return {"tool": e["tool"], "genuine_raw": e["genuine_raw"], "genuine_value": e["genuine_value"],
                "genuine_sha256": e["genuine_sha256"]}

    def result_for(self, call_id: int) -> Optional[ToolCallResult]:
        return self._results.get(call_id)

    def key_state(self, tool: str, args: dict) -> Any:
        return self._keys.get(injection_key(tool, args))

    def in_outage(self, tool: str) -> bool:
        return self._outages[tool].active

    # ---------------------------------------------------------------- calls
    def call(self, tool: str, args: dict) -> ToolCallResult:
        return self._call(tool, args, probe=False)

    def resend_probe(self, tool: str, args: dict) -> ToolCallResult:
        return self._call(tool, args, probe=True)

    def _fn(self, tool: str):
        if tool not in self.tools:
            raise ValueError(f"unknown tool {tool!r}, must be one of {list(self.tools)}")
        return self.tools[tool]

    def _call(self, tool: str, args: dict, probe: bool) -> ToolCallResult:
        fn = self._fn(tool)
        self._call_seq += 1
        cid = self._call_seq
        key = injection_key(tool, args)
        meta: dict = {"call_id": cid, "mode": self.mode, "block_index": self.block_index, "probe": probe,
                      "injection_key": key, "injected": False, "injected_kind": None, "fresh_draw": False,
                      "synthetic_content": False, "degraded": False}
        handler = {"transient": self._transient, "sticky_args": self._sticky,
                   "tool_outage": self._outage, "degraded_output": self._degraded}[self.mode]
        result, extra = handler(fn, tool, args, key, probe, meta)
        result.injected = meta["injected"]
        result.injected_kind = meta["injected_kind"]
        meta["natural_failure"] = (not result.success) and not meta["injected"]
        result.v3 = meta
        self._record(cid, tool, args, result, meta, extra)
        self._results[cid] = result
        return result

    def _transient(self, fn, tool, args, key, probe, meta):
        rng = self._rng["resend" if probe else "injection"]
        u, u_kind = float(rng.random()), float(rng.random())   # always two draws per call
        meta["fresh_draw"], meta["draw_injected"] = True, u < self.rate
        if u >= self.rate:
            return fn(args, False, DEFAULT_TIMEOUT_S), {}
        meta["injected"] = True
        if u_kind < 0.5:
            meta["injected_kind"] = "timeout"
            r = fn(args, False, TRANSIENT_TIMEOUT_S)
            meta["injected_call_succeeded"] = r.success   # v1 behaviour: a 0.2 s timeout can fail to fire
            return r, {}
        meta["injected_kind"] = "malformed_request"
        return fn(args, True, DEFAULT_TIMEOUT_S), {}

    def _sticky(self, fn, tool, args, key, probe, meta):
        state = self._keys.get(key)
        if state is None and not probe:
            state = {"corrupted": float(self._rng["injection"].random()) < self.rate}
            self._keys[key] = state
            meta["fresh_draw"], meta["draw_injected"] = True, state["corrupted"]
            meta["sticky_hit"] = False
        else:
            meta["sticky_hit"] = bool(state and state["corrupted"])
        if state and state["corrupted"]:
            meta["injected"], meta["injected_kind"] = True, "malformed_request"
            return fn(args, True, DEFAULT_TIMEOUT_S), {}
        return fn(args, False, DEFAULT_TIMEOUT_S), {}

    def _outage_fail(self, fn, args, meta):
        meta["injected"], meta["injected_kind"] = True, "timeout"
        r = fn(args, False, OUTAGE_TIMEOUT_S)
        meta["outage_leak"] = r.success
        return r

    def _outage(self, fn, tool, args, key, probe, meta):
        o = self._outages[tool]
        meta.update(outage_N=self.outage_N, outage_id=o.outage_id if o.active else None,
                    outage_call_index=None, in_outage=o.active)
        if probe:  # read-only: never counts as one of the N calls, never consumes serve_next
            return (self._outage_fail(fn, args, meta) if o.active else fn(args, False, DEFAULT_TIMEOUT_S)), {}
        if o.active:
            o.index += 1
            meta.update(outage_call_index=o.index)
            r = self._outage_fail(fn, args, meta)
            if o.index >= self.outage_N:
                o.active, o.serve_next = False, True
            return r, {}
        if o.serve_next:
            o.serve_next = False
            meta["served_after_outage"] = True
            return fn(args, False, DEFAULT_TIMEOUT_S), {}
        started = float(self._rng["outage"].random()) < self.rate
        meta["fresh_draw"], meta["draw_injected"] = True, started
        if started:
            self._outage_counter += 1
            o.active, o.index, o.outage_id = True, 1, self._outage_counter
            meta.update(in_outage=True, outage_id=o.outage_id, outage_call_index=1)
            return self._outage_fail(fn, args, meta), {}
        return fn(args, False, DEFAULT_TIMEOUT_S), {}

    def _degraded(self, fn, tool, args, key, probe, meta):
        spec: Optional[DegradeSpec] = self._keys.get(key)
        if spec is None and not probe:
            spec = draw_spec(self._rng["degradation"], self.rate)
            self._keys[key] = spec
            meta["fresh_draw"], meta["draw_injected"] = True, spec.degrade
        r = fn(args, False, DEFAULT_TIMEOUT_S)
        extra: dict = {}
        if not r.success or tool not in ANSWER_FIELDS:
            return r, extra
        genuine_text = r.full_text if r.full_text is not None else r.body
        ans = read_answer(tool, genuine_text)
        extra.update(genuine_raw=ans.raw, genuine_value=ans.value,
                     genuine_sha256=_sha(json.dumps(ans.raw, ensure_ascii=False, sort_keys=True)),
                     original_sha256=_sha(genuine_text))
        if spec is not None and spec.degrade:
            meta.update(family=spec.family, transformation=spec.transformation)
            if not (ans.present and ans.parses) and spec.transformation != "truncate":
                meta["degrade_error"] = "answer field missing or unparseable in the genuine response"
            else:
                try:
                    delivered, detail = apply_degradation(tool, args, genuine_text, ans.raw, spec, self.libraries)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    meta["degrade_error"] = f"{type(exc).__name__}: {exc}"[:200]
                else:
                    r.full_text, r.body = delivered, _truncate(delivered)
                    meta.update(degraded=True, synthetic_content=True,
                                original_sha256=extra["original_sha256"], delivered_sha256=_sha(delivered))
                    extra["degrade_detail"] = detail
                    extra["degrade_spec"] = spec.as_dict()
        extra["delivered_text"] = r.full_text if r.full_text is not None else r.body
        return r, extra

    def _record(self, cid: int, tool: str, args: dict, r: ToolCallResult, meta: dict, extra: dict) -> None:
        entry = {
            "call_id": cid, "ts_utc": self.clock(), "episode_id": self.context.get("episode_id"),
            "step_index": self.context.get("step_index"), "step_goal": self.context.get("step_goal"),
            "actor": "resend_probe" if meta["probe"] else self.context.get("actor", "agent"),
            "tool": tool, "args": args, **meta,
            "success": r.success, "http_status": r.http_status, "latency_ms": round(r.latency_ms, 1),
            "error_text": None if r.error_text is None else r.error_text[:_LOG_ERROR_CLIP],
            **extra,
        }
        self.log.append(entry)
        self._by_id[cid] = entry
