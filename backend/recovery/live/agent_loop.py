"""Minimal live agent loop: run one Task, capture a FailureEvent on the
first failure (tool-level or the model's own tool-selection failure).

Deliberately single-pass per step, no retry/recovery here -- Phase 1 is
collection only (see ``docs/live_pilot/strategy_definitions.md``). A task
stops at its first failure rather than continuing past it, since later
steps' context would otherwise be built on a state the agent never
actually reached cleanly.

Reuses ``src.agent.llm_client.parse_tool_request`` (the existing Groq
wrapper, already reading ``GROQ_API_KEY`` from ``.env`` via
``_ensure_env_loaded``) rather than adding a second LLM client.

Task 21: ``run_task(..., recovery=InlineRecovery(...))`` additionally runs one
inline recovery strategy on a captured tool-call failure and stores a
``RecoveryAttempt`` next to the event (``recovery.live.inline_recovery``,
``docs/live_pilot/strategy_definitions_v2.md``). The task still stops at that
first failure either way. With ``recovery=None`` (the default) behaviour is
exactly the Phase 1 collection loop above -- no strategy is applied.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from recovery.live.event_schema import AgentState, FailureEvent, Task
from recovery.live.event_store import EventStore
from recovery.live.inline_recovery import NOT_APPLICABLE_SELECTION_FAILURE
from recovery.live.tools import TOOL_DESCRIPTIONS, FlakyWrapper

if TYPE_CHECKING:
    from recovery.live.inline_recovery import InlineRecovery
    from recovery.live.recovery_attempt import RecoveryAttempt

__all__ = ["TaskOutcome", "run_task"]


@dataclass
class TaskOutcome:
    task_id: str
    session_id: str
    completed: bool
    failure_event: Optional[FailureEvent] = None
    steps_completed: int = 0
    recovery_attempt: Optional["RecoveryAttempt"] = None
    recovery_skipped_reason: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_tool(step_goal: str) -> dict:
    """Ask the model to pick a tool + args for one step. Raises ValueError
    on unparseable output -- caller turns that into a FailureEvent rather
    than crashing the run, since a model producing unusable tool-selection
    output is itself a real, capturable agent-level failure."""
    from src.agent.llm_client import parse_tool_request

    return parse_tool_request(step_goal, TOOL_DESCRIPTIONS)


def run_task(
    task: Task,
    store: EventStore,
    wrapper: FlakyWrapper,
    session_id: Optional[str] = None,
    tool_health: Optional[dict] = None,
    recovery: Optional["InlineRecovery"] = None,
) -> TaskOutcome:
    """Run one task's steps in sequence. Stops and captures a FailureEvent
    at the first failure; returns ``completed=True`` only if every step
    succeeded.

    ``tool_health`` should be the result of one
    ``recovery.live.tools.health_check_all_tools()`` call made once at the
    start of the collection session (not per task) and threaded through
    unchanged, so every event captured in that session records the same
    session-start health snapshot -- see ``FailureEvent.tool_health``.
    """
    session_id = session_id or uuid.uuid4().hex[:16]
    steps = task.steps or [task.goal]
    history: list[dict] = []

    for step_index, step_goal in enumerate(steps):
        history.append({"role": "user", "content": step_goal})

        try:
            picked = _select_tool(step_goal)
        except ValueError as exc:
            event = FailureEvent(
                session_id=session_id,
                task_id=task.task_id,
                step_index=step_index,
                tool_name="<llm_tool_selection>",
                tool_args={},
                raw_error=str(exc),
                agent_state=AgentState(
                    goal=task.goal, conversation_history=list(history), tool_call_index=step_index,
                ),
                captured_at=_now_iso(),
                tool_health=tool_health,
            )
            store.write(event)
            outcome = TaskOutcome(task.task_id, session_id, completed=False, failure_event=event, steps_completed=step_index)
            if recovery is not None:
                # v2 defines strategies for tool calls only: counted, not recovered.
                recovery.recover(event, task, step_goal, None, history, wrapper.call)
                outcome.recovery_skipped_reason = NOT_APPLICABLE_SELECTION_FAILURE
            return outcome

        tool_name, tool_args = picked["tool"], picked["args"]
        result = wrapper.call(tool_name, tool_args)

        if not result.success:
            event = FailureEvent(
                session_id=session_id,
                task_id=task.task_id,
                step_index=step_index,
                tool_name=tool_name,
                tool_args=tool_args,
                raw_error=result.error_text or "",
                http_status=result.http_status,
                raw_response_body=result.body,
                latency_ms=result.latency_ms,
                injected=result.injected,
                injected_kind=result.injected_kind,
                agent_state=AgentState(
                    goal=task.goal, conversation_history=list(history), tool_call_index=step_index,
                ),
                captured_at=_now_iso(),
                tool_health=tool_health,
            )
            store.write(event)
            outcome = TaskOutcome(task.task_id, session_id, completed=False, failure_event=event, steps_completed=step_index)
            if recovery is not None:
                outcome.recovery_attempt = recovery.recover(event, task, step_goal, result, history, wrapper.call)
            return outcome

        history.append({
            "role": "tool", "tool": tool_name, "args": tool_args,
            "content": result.body,
        })

    return TaskOutcome(task.task_id, session_id, completed=True, steps_completed=len(steps))
