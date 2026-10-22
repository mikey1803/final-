"""v3 inline recovery: one uniformly drawn strategy per captured failure, plus
the instrument-side outcomes the v3 analysis needs (v3 §3.4, §4, §6, §7;
pre-registration §3-§4).

Arms: the six v2-configured strategies (retry v2, replan, switch_tool,
re_ground v2, verify v2, clarify), uniform, propensity 1/6. Retry v1 is not
drawn (v3 §4); NULL_CONTROL is not drawn (its ``wait_s`` is still
[RESULT REQUIRED], v2). Randomness: the ``selection`` and ``jitter`` streams,
keyed by the session-wide ``failure_index`` (``recovery.live.v3.seeds``).

Outcomes recorded on ``attempt.v3`` (instrument side, never shown to the agent):

- ``applicable``: the strategy sent at least one tool call (pre-reg §4.1).
- ``final_call_id``: the strategy's answer -- its last successful call when it
  succeeded (or, for a VERIFY that accepted the original result, the failing
  call itself).
- ``corrected`` (degraded_output only): the final answer matches the genuine
  value -- reference = the final call's own captured genuine value
  (``same_call``), else the failing call's (``degraded_call``).
- ``detected`` (degraded_output only): a check run by the strategy rejected
  the degraded original (VERIFY's schema check or judge).
- ``primary_outcome``: ``corrected`` in degraded_output, else ``success``.
- ``fresh`` (pre-reg §4.3): the strategy's final call has a different
  injection key (sticky_args, degraded_output) or a different tool
  (tool_outage) from the failing call; None when it sent no call.
- ``instrument_fault``: ``evidence.reason`` in {llm_error, executor_error};
  such attempts are excluded from rates and counted (v3 §6).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np

from recovery.live.event_schema import FailureEvent, Task
from recovery.live.executors import build_executors
from recovery.live.executors.base import Budget, FailureContext, LLMClient, Responder
from recovery.live.recovery_attempt import RecoveryAttempt, RecoveryStore
from recovery.live.tools import ToolCallResult
from recovery.live.v3 import DEFINITION_VERSION
from recovery.live.v3.answers import compare, read_answer
from recovery.live.v3.flaky import FlakyWrapperV3
from recovery.live.v3.seeds import stream_seed
from recovery.live.v3.selection import (
    Selection, SelectionFailureContext, build_selection_executors, failed_selection_result,
)

__all__ = ["InlineRecoveryV3", "INSTRUMENT_FAULTS", "answer_outcome"]

INSTRUMENT_FAULTS = frozenset({"llm_error", "executor_error"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def answer_outcome(wrapper: FlakyWrapperV3, final_call_id: Optional[int], failing_call_id: Optional[int]) -> dict:
    """``corrected`` for one answer call (degraded_output only), instrument side."""
    out = {"corrected": False, "corrected_reference": None, "corrected_tolerance": None,
           "corrected_abs_diff": None}
    if final_call_id is None:
        return out
    entry = wrapper.entry(final_call_id)
    if entry is None or not entry.get("success"):
        return out
    ref = wrapper.genuine(final_call_id)
    out["corrected_reference"] = "same_call"
    if ref is None and failing_call_id is not None:
        ref = wrapper.genuine(failing_call_id)
        out["corrected_reference"] = "degraded_call"
    if ref is None:
        out["corrected_reference"] = None
        return out
    cmp = compare(entry["tool"], read_answer(entry["tool"], entry.get("delivered_text")),
                  ref["tool"], ref["genuine_value"])
    out.update(corrected=cmp.match, corrected_tolerance=cmp.tolerance, corrected_abs_diff=cmp.abs_diff)
    return out


@dataclass
class InlineRecoveryV3:
    store: RecoveryStore
    session_seed: int
    budget: Budget = field(default_factory=Budget)
    llm: Optional[LLMClient] = None
    responder: Optional[Responder] = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    failure_index: int = 0

    def __post_init__(self) -> None:
        self.executors = build_executors("v2")              # v2 configurations carry over (v3 preamble)
        self.selection_executors = build_selection_executors()
        self.arms = list(self.executors)
        if "null_control" in self.arms or list(self.selection_executors) != self.arms:
            raise RuntimeError(f"v3 arms must be the six strategies, got {self.arms}")

    def _draw(self) -> tuple[int, str, float, dict, int, dict]:
        index = self.failure_index
        self.failure_index += 1
        sel_seed, sel_rec = stream_seed(self.session_seed, "selection", index)
        jit_seed, jit_rec = stream_seed(self.session_seed, "jitter", index)
        rng = np.random.default_rng(sel_seed)
        strategy = self.arms[int(rng.integers(len(self.arms)))]
        return index, strategy, 1.0 / len(self.arms), sel_rec, jit_seed, jit_rec

    def recover(self, *, event: FailureEvent, task: Task, step_goal: str, history: list[dict],
                wrapper: FlakyWrapperV3, failed_result: Optional[ToolCallResult], select: Callable[[str], dict],
                bad_selection: Optional[Selection], resend_baseline: dict, v3_base: dict) -> RecoveryAttempt:
        index, strategy, propensity, sel_rec, jit_seed, jit_rec = self._draw()
        agent_side = bad_selection is not None
        common = dict(event=event, task=task, step_goal=step_goal, history=list(history),
                      call_tool=wrapper.call, rng=random.Random(jit_seed), llm=self.llm,
                      responder=self.responder, sleep=self.sleep, clock=self.clock)
        if agent_side:
            executor = self.selection_executors[strategy]
            ctx: FailureContext = SelectionFailureContext(
                tool_name=event.tool_name, tool_args={}, failed_result=failed_selection_result(bad_selection),
                select=select, agent_side_kind=bad_selection.kind, bad_selection=bad_selection, **common)
        else:
            executor = self.executors[strategy]
            ctx = FailureContext(tool_name=event.tool_name, tool_args=dict(event.tool_args),
                                 failed_result=failed_result, **common)

        started, t0 = _now_iso(), self.clock()
        try:
            result_d = executor.execute(ctx, self.budget).to_dict()
        except Exception as exc:  # a bug in an executor must not crash the live run
            result_d = {"success": False, "cost_tokens": 0, "latency_ms": round((self.clock() - t0) * 1000, 1),
                        "attempts": 0, "evidence": {"calls": [], "llm_calls": [], "answered_by": None,
                                                    "reason": "executor_error",
                                                    "executor_error": f"{type(exc).__name__}: {exc}"[:300]}}
        ev = result_d["evidence"]
        v3 = dict(v3_base)
        v3.update(self._outcomes(wrapper, v3_base, ev, result_d["success"], strategy, agent_side))
        v3.update(resend_baseline=resend_baseline,
                  seeds={"selection": sel_rec, "jitter": jit_rec, "failure_index": index},
                  task_completed=None, task_completed_applicable=None)
        used = {"attempts": result_d["attempts"], "wall_ms": result_d["latency_ms"], "tokens": result_d["cost_tokens"]}
        b = self.budget
        attempt = RecoveryAttempt(
            event_id=event.event_id, session_id=event.session_id, task_id=event.task_id,
            strategy=strategy, definition_version=DEFINITION_VERSION,
            selection_policy="uniform", selection_propensity=propensity,
            seed={"session_seed": self.session_seed, "failure_index": index,
                  "derived_seed": sel_rec["derived_seed"], "method": sel_rec["method"]},
            model_id=next((c["model_id"] for c in ev["llm_calls"]), None),
            executor_config={"name": executor.name, **executor.config(), "responder": ev.get("responder"),
                             "arms": self.arms, "on": "selection_failure" if agent_side else "tool_failure"},
            budget_limits=b.as_dict(), budget_used=used,
            budget_exceeded=bool(ev.get("budget_exceeded", used["attempts"] > b.max_attempts
                                        or used["tokens"] > b.max_tokens)),
            result=result_d, started_at=started, finished_at=_now_iso(),
            aborted=bool(ev.get("aborted", False)), no_action_taken=bool(ev.get("no_action_taken", False)),
            v3=v3,
        )
        self.store.write(attempt)
        return attempt

    @staticmethod
    def _outcomes(wrapper: FlakyWrapperV3, base: dict, ev: dict, success: bool, strategy: str,
                  agent_side: bool) -> dict:
        calls = ev.get("calls", [])
        mode = base["injection_mode"]
        failing_call_id = base.get("call_id")
        final_call_id: Optional[int] = None
        if success:
            ok = [c for c in calls if c.get("success") and c.get("v3")]
            if ok:
                final_call_id = ok[-1]["v3"]["call_id"]
            elif ev.get("no_action_taken") and not agent_side:
                final_call_id = failing_call_id  # VERIFY accepted the (degraded) original as the answer
        out: dict[str, Any] = {
            "applicable": len(calls) >= 1, "n_tool_calls": len(calls), "final_call_id": final_call_id,
            "instrument_fault": ev.get("reason") in INSTRUMENT_FAULTS,
        }
        last = calls[-1] if calls else None
        if agent_side or last is None or mode == "transient":
            out["fresh"] = None
        elif mode == "tool_outage":
            out["fresh"] = last["tool"] != base.get("tool")
        else:
            failing_key = (wrapper.entry(failing_call_id) or {}).get("injection_key")
            out["fresh"] = (last.get("v3") or {}).get("injection_key") != failing_key
        if mode == "degraded_output":
            out.update(answer_outcome(wrapper, final_call_id, failing_call_id))
            rejected = [c for c in ev.get("checks", []) if c.get("stage") == "original" and c.get("passed") is False]
            out["detected"] = (strategy == "verify" and bool(rejected)) if not agent_side else None
            out["primary_outcome"] = bool(out["corrected"])
        else:
            out.update(detected=None, corrected=None)
            out["primary_outcome"] = bool(success)
        return out
