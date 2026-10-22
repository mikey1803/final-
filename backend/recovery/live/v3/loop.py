"""The v3 episode loop (v3 §2-§7; pre-registration §2-§3).

One episode = one run of one task inside one block (one ``FlakyWrapperV3``).
Per step: the agent selects a call (Layer 1 selector); an unusable selection
is an agent-side failure (v3 §5); otherwise the call goes through the
wrapper. A failed call -- or, in ``degraded_output``, a degraded one, surfaced
by the oracle (``detected_by="oracle"``) -- is a captured failure:

1. a ``FailureEvent`` with its v3 fields is written;
2. the **resend baseline** runs immediately (read-only probe, v3 §4) and is
   stored as its own row;
3. **halt H1**: on an ``injected`` failure outside ``transient``, a resend
   success (``corrected`` in degraded_output) is impossible by construction,
   so collection stops (``HaltH1``). A ``natural`` failure is exempt: it is
   tagged ``h1_exempt_natural`` and the run continues (pre-reg §2);
4. one drawn strategy runs (``InlineRecoveryV3``);
5. ``continue_after_recovery`` (v3 §7, default True): if the strategy's
   primary outcome holds (``corrected`` in degraded_output, else success) the
   recovered result becomes the step's result and the task moves on;
   otherwise the task stops.

``task_completed`` is computed once when the task ends (v3 §7 per-tool table:
the last step's answer field present and parsing; in degraded_output blocks
also matching the genuine value) and written onto every attempt of the
episode. ``out_of_scope`` tasks are recorded with
``task_completed_applicable=False`` (pre-reg §7.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from recovery.live.event_schema import AgentState, FailureEvent, Task
from recovery.live.event_store import EventStore
from recovery.live.recovery_attempt import RecoveryStore
from recovery.live.v3 import DEFINITION_VERSION
from recovery.live.v3.answers import task_completed as _task_completed
from recovery.live.v3.flaky import FlakyWrapperV3
from recovery.live.v3.recovery import InlineRecoveryV3, answer_outcome
from recovery.live.v3.selection import SELECTION_FAILURE_TOOL, Selection, SelectionProviderError, validate_selection

__all__ = ["BlockSpec", "BLOCK_TYPES", "HaltH1", "run_episode_v3", "V3Dirs", "ORACLE_ERROR_TEXT"]

# The message a strategy sees for an oracle-surfaced degraded result: it says a
# check flagged the result, never what was changed (synthetic, declared).
ORACLE_ERROR_TEXT = "The result was flagged as incorrect by an external check."

BLOCK_TYPES: tuple[tuple[str, Optional[int]], ...] = (
    ("transient", None), ("sticky_args", None), ("tool_outage", 2), ("tool_outage", 5),
    ("tool_outage", 20), ("degraded_output", None),
)
_LABELS = {("transient", None): "T", ("sticky_args", None): "S", ("tool_outage", 2): "O2",
           ("tool_outage", 5): "O5", ("tool_outage", 20): "O20", ("degraded_output", None): "D"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BlockSpec:
    block_index: int
    mode: str
    outage_N: Optional[int]
    passes: int

    @property
    def label(self) -> str:
        return _LABELS[(self.mode, self.outage_N)]


class HaltH1(RuntimeError):
    """An impossible-by-construction resend success on an injected failure (pre-reg §2)."""


@dataclass
class V3Dirs:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def resends(self) -> Path:
        return self.root / "v3_resends"

    @property
    def episodes(self) -> Path:
        return self.root / "v3_episodes"

    def write_json(self, sub: Path, name: str, obj: dict) -> Path:
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path


@dataclass
class _Answer:
    tool: str
    call_id: int
    text: Optional[str]


def _resend(wrapper: FlakyWrapperV3, select: Callable[[str], dict], step_goal: str,
            tool: Optional[str], args: Optional[dict], failing_call_id: Optional[int]) -> dict:
    """The resend baseline (v3 §4): one immediate identical re-send through the
    read-only probe; for a selection failure, the identical selection prompt once."""
    rec: dict = {"kind": "selection_rerun" if tool is None else "identical_call"}
    if tool is None:
        try:
            sel = validate_selection(select, step_goal)
        except SelectionProviderError as exc:
            rec.update(success=False, corrected=None, instrument_fault=True, error_text=str(exc), call_id=None)
            return rec
        rec["selection_kind"] = sel.kind
        if not sel.usable:
            rec.update(success=False, corrected=False if wrapper.mode == "degraded_output" else None,
                       instrument_fault=False, call_id=None, error_text=sel.reply_text[:300])
            return rec
        tool, args = sel.tool, sel.args
    wrapper.set_context(actor="resend_probe")
    r = wrapper.resend_probe(tool, args)
    m = r.v3
    rec.update(call_id=m["call_id"], tool=tool, success=bool(r.success), http_status=r.http_status,
               injected=m["injected"], injected_kind=m["injected_kind"], sticky_hit=m.get("sticky_hit"),
               in_outage=m.get("in_outage"), degraded=m.get("degraded"), latency_ms=round(r.latency_ms, 1),
               error_text=None if r.error_text is None else r.error_text[:300], instrument_fault=False)
    if wrapper.mode == "degraded_output":
        rec.update(answer_outcome(wrapper, m["call_id"] if r.success else None, failing_call_id))
    else:
        rec["corrected"] = None
    return rec


def _resend_outcome(mode: str, rec: dict) -> bool:
    return bool(rec.get("corrected")) if mode == "degraded_output" else bool(rec.get("success"))


def run_episode_v3(task: Task, *, wrapper: FlakyWrapperV3, events: EventStore, recoveries: RecoveryStore,
                   recovery: InlineRecoveryV3, select: Callable[[str], dict], block: BlockSpec,
                   session: dict, episode_index: int, tool_health: Optional[dict], dirs: V3Dirs,
                   continue_after_recovery: bool = True) -> dict:
    episode_id = f"v3-s{session['session_seed']}-b{block.block_index}-e{episode_index:04d}"
    wrapper.start_episode(episode_id)
    steps = task.steps or [task.goal]
    history: list[dict] = []
    attempts: list = []
    failures: list[dict] = []
    last: Optional[_Answer] = None
    finished_all = False
    aborted: Optional[str] = None
    degraded_block = block.mode == "degraded_output"

    for step_index, step_goal in enumerate(steps):
        history.append({"role": "user", "content": step_goal})
        wrapper.set_context(step_index=step_index, step_goal=step_goal, actor="agent")
        try:
            sel = validate_selection(select, step_goal)
        except SelectionProviderError as exc:
            aborted = f"selection_provider_error: {exc}"[:300]
            break

        result = None
        if sel.usable:
            wrapper.set_context(actor="agent")
            result = wrapper.call(sel.tool, sel.args)
            meta = result.v3
            if result.success and not meta["degraded"]:
                history.append({"role": "tool", "tool": sel.tool, "args": sel.args, "content": result.body})
                last = _Answer(sel.tool, meta["call_id"], result.full_text if result.full_text is not None else result.body)
                if step_index == len(steps) - 1:
                    finished_all = True
                continue

        # ---------------------------------------------------------- a captured failure
        ordinal = len(failures) + 1
        if not sel.usable:
            origin, tool_name, tool_args = "agent_side", SELECTION_FAILURE_TOOL, {}
            raw_error = f"unusable tool selection ({sel.kind}): {sel.reply_text}"
            detected_by, call_id, meta = None, None, None
        else:
            meta = result.v3
            tool_name, tool_args = sel.tool, dict(sel.args)
            if meta["degraded"]:
                origin, detected_by = "injected", "oracle"
                raw_error = "[oracle] degraded_output detected by the instrument (synthetic content, declared)"
            else:
                origin, detected_by = ("injected" if meta["injected"] else "natural"), None
                raw_error = result.error_text or ""
            call_id = meta["call_id"]
        v3_event = {
            "definition_version": DEFINITION_VERSION, "fault_origin": origin, "injection_mode": block.mode,
            "block_index": block.block_index, "block_label": block.label, "outage_N": block.outage_N,
            "injection_params": wrapper.params(), "agent_side_kind": sel.kind, "detected_by": detected_by,
            "episode_id": episode_id, "episode_index": episode_index, "failure_ordinal": ordinal,
            "call_id": call_id, "call_meta": meta, "session_seed": session["session_seed"],
            "session_index": session["session_index"],
        }
        event = FailureEvent(
            session_id=episode_id, task_id=task.task_id, step_index=step_index, tool_name=tool_name,
            tool_args=tool_args, raw_error=raw_error,
            http_status=None if result is None else result.http_status,
            raw_response_body=None if result is None else result.body,
            latency_ms=None if result is None else result.latency_ms,
            injected=origin == "injected", injected_kind=None if meta is None else meta["injected_kind"],
            agent_state=AgentState(goal=task.goal, conversation_history=list(history), tool_call_index=step_index),
            captured_at=_now(), tool_health=tool_health, v3=v3_event,
        )
        events.write(event)

        resend = _resend(wrapper, select, step_goal, None if not sel.usable else tool_name,
                         None if not sel.usable else tool_args, call_id)
        b = _resend_outcome(block.mode, resend)
        resend["outcome"] = b
        resend["h1_exempt_natural"] = bool(origin == "natural" and b and block.mode != "transient")
        dirs.write_json(dirs.resends, event.event_id, {
            "event_id": event.event_id, "episode_id": episode_id, "fault_origin": origin,
            "injection_mode": block.mode, "block_label": block.label, "block_index": block.block_index,
            "session_seed": session["session_seed"], "strategy_drawn": None, **resend})
        if origin == "injected" and block.mode != "transient" and b:
            raise HaltH1(f"resend {'corrected' if degraded_block else 'success'} on an injected "
                         f"{block.label} failure (event {event.event_id}): impossible by construction")

        base = {k: v3_event[k] for k in ("definition_version", "fault_origin", "injection_mode", "block_index",
                                         "block_label", "outage_N", "agent_side_kind", "detected_by",
                                         "episode_id", "failure_ordinal", "call_id", "session_seed",
                                         "session_index")}
        base.update(tool=tool_name, continue_after_recovery=continue_after_recovery,
                    prior_attempt_ids=[a.attempt_id for a in attempts], h1_exempt_natural=resend["h1_exempt_natural"])
        failed_result = result
        if result is not None and meta["degraded"]:
            failed_result = replace(result, error_text=ORACLE_ERROR_TEXT)
        wrapper.set_context(actor="strategy")
        attempt = recovery.recover(event=event, task=task, step_goal=step_goal, history=history, wrapper=wrapper,
                                   failed_result=failed_result, select=select,
                                   bad_selection=None if sel.usable else sel, resend_baseline=resend, v3_base=base)
        dirs.write_json(dirs.resends, event.event_id, {
            "event_id": event.event_id, "episode_id": episode_id, "fault_origin": origin,
            "injection_mode": block.mode, "block_label": block.label, "block_index": block.block_index,
            "session_seed": session["session_seed"], "strategy_drawn": attempt.strategy, **resend})
        attempts.append(attempt)
        failures.append({"event_id": event.event_id, "attempt_id": attempt.attempt_id, "strategy": attempt.strategy,
                         "fault_origin": origin, "primary_outcome": attempt.v3["primary_outcome"]})

        if not (continue_after_recovery and attempt.v3["primary_outcome"]):
            break
        final_id = attempt.v3["final_call_id"]
        entry = wrapper.entry(final_id) if final_id is not None else None
        res = wrapper.result_for(final_id) if final_id is not None else None
        if entry is None or res is None:
            break
        history.append({"role": "tool", "tool": entry["tool"], "args": entry["args"], "content": res.body,
                        "recovered_by": attempt.strategy})
        last = _Answer(entry["tool"], final_id, res.full_text if res.full_text is not None else res.body)
        if step_index == len(steps) - 1:
            finished_all = True

    completed = False
    if finished_all and last is not None:
        g = wrapper.genuine(last.call_id) if degraded_block else None
        completed = _task_completed(last.tool, last.text, degraded_block,
                                    None if g is None else g["tool"], None if g is None else g["genuine_value"])
    applicable = task.category != "out_of_scope"
    for a in attempts:
        a.v3["task_completed"] = completed
        a.v3["task_completed_applicable"] = applicable
        recoveries.write(a)
    record = {
        "episode_id": episode_id, "task_id": task.task_id, "task_category": task.category,
        "definition_version": DEFINITION_VERSION, "session_seed": session["session_seed"],
        "session_index": session["session_index"], "block_index": block.block_index, "block_label": block.label,
        "injection_mode": block.mode, "outage_N": block.outage_N, "episode_index": episode_index,
        "n_steps": len(steps), "finished_all_steps": finished_all, "task_completed": completed,
        "task_completed_applicable": applicable, "aborted": aborted, "failures": failures,
        "attempt_ids": [a.attempt_id for a in attempts], "last_answer_call_id": None if last is None else last.call_id,
        "continue_after_recovery": continue_after_recovery,
    }
    dirs.write_json(dirs.episodes, episode_id, record)
    return record
