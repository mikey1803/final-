"""Shared types for the real recovery executors (Task 21).

Definitions: ``docs/live_pilot/strategy_definitions_v2.md`` (FROZEN v2), whose
§4 also gives the v1 configuration of each strategy. Every executor implements

    class RecoveryExecutor(Protocol):
        name: str
        def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult

``FailureContext`` wraps the existing ``FailureEvent``/``Task`` rather than
copying their fields; there was no pre-existing context or budget type in
``recovery.live`` or ``recovery.core`` to reuse (``core.protocols.Context`` is
the simulator's predicted-type context, unrelated).

Budget rule (v2 §3): an action is started only if its worst case fits in the
remaining budget -- a tool call needs ``TOOL_CALL_RESERVE_MS`` of wall time
and a free attempt; an LLM call needs ``ceil(len(prompt)/2) + max_completion``
tokens. Anything that still overruns is reported as ``budget_exceeded``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Protocol

from recovery.live.event_schema import FailureEvent, Task
from recovery.live.tools import ToolCallResult

__all__ = [
    "Budget", "BudgetTracker", "FailureContext", "RecoveryResult", "RecoveryExecutor",
    "LLMClient", "LLMReply", "LLMError", "Responder", "Evidence",
    "TOOL_CALL_RESERVE_MS", "LLM_TIMEOUT_CAP_S", "REASONS",
]

TOOL_CALL_RESERVE_MS = 12_000.0   # 10 s tool timeout + 2 s connection allowance (v2 §3)
LLM_TIMEOUT_CAP_S = 20.0
ERROR_CLIP = 500

# evidence.reason values defined by v2 §5, plus two fault codes ("llm_error",
# "executor_error") for an LLM provider failure or a bug in an executor --
# instrument faults, not strategy outcomes, reported separately.
REASONS = frozenset({
    "success", "exhausted_attempts", "budget_exhausted", "untranslatable_args",
    "question_unmatched", "unparseable_llm_output", "checks_rejected",
    "original_result_accepted", "llm_error", "executor_error",
})


@dataclass(frozen=True)
class Budget:
    """v2 §3 defaults -- configuration choices, not measured values."""

    max_attempts: int = 3
    max_wall_ms: float = 40_000.0
    max_tokens: int = 1_500

    def as_dict(self) -> dict:
        return asdict(self)


class BudgetTracker:
    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self.attempts = 0
        self.tokens = 0

    @property
    def elapsed_ms(self) -> float:
        return (self._clock() - self._start) * 1000.0

    @property
    def remaining_ms(self) -> float:
        return self.budget.max_wall_ms - self.elapsed_ms

    def can_call_tool(self, after_sleep_s: float = 0.0) -> bool:
        return (self.attempts < self.budget.max_attempts
                and self.remaining_ms - after_sleep_s * 1000.0 >= TOOL_CALL_RESERVE_MS)

    def llm_reservation(self, prompt: str, max_completion: int) -> int:
        return math.ceil(len(prompt) / 2) + max_completion

    def can_call_llm(self, prompt: str, max_completion: int) -> bool:
        return (self.tokens + self.llm_reservation(prompt, max_completion) <= self.budget.max_tokens
                and self.remaining_ms > 0)

    def used(self) -> dict:
        return {"attempts": self.attempts, "wall_ms": round(self.elapsed_ms, 1), "tokens": self.tokens}

    def exceeded(self) -> bool:
        # 1e-6 ms absorbs float error when an LLM call uses exactly its timeout
        # (= the remaining budget); a real overrun is many orders larger.
        return (self.attempts > self.budget.max_attempts or self.tokens > self.budget.max_tokens
                or self.elapsed_ms > self.budget.max_wall_ms + 1e-6)


@dataclass
class RecoveryResult:
    success: bool
    cost_tokens: int
    latency_ms: float
    attempts: int
    evidence: dict

    def to_dict(self) -> dict:
        return asdict(self)


class LLMError(RuntimeError):
    """The LLM provider call failed (HTTP error, timeout, malformed envelope)."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int, timeout_s: float) -> LLMReply: ...


class Responder(Protocol):
    """Answers CLARIFY's question; returns None when it cannot."""

    kind: str

    def answer(self, question: str, task: Task, timeout_s: float) -> Optional[str]: ...


@dataclass
class FailureContext:
    """Everything an executor needs about one captured failure."""

    event: FailureEvent
    task: Task
    step_goal: str
    tool_name: str
    tool_args: dict
    failed_result: ToolCallResult
    history: list[dict]
    call_tool: Callable[[str, dict], ToolCallResult]   # FlakyWrapper.call: symmetric injection
    rng: random.Random                                  # backoff jitter only
    llm: Optional[LLMClient] = None
    responder: Optional[Responder] = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def last_good_state(self) -> Optional[dict]:
        """Most recent successful tool result in this task's history."""
        for item in reversed(self.history):
            if item.get("role") == "tool":
                return item
        return None


class RecoveryExecutor(Protocol):
    name: str
    definition_version: str

    def config(self) -> dict: ...

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult: ...


# ----------------------------------------------------------------------
# evidence + guarded actions, shared by every executor
# ----------------------------------------------------------------------

def _clip(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= ERROR_CLIP else text[:ERROR_CLIP] + "...[truncated]"


@dataclass
class Evidence:
    calls: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    answered_by: Optional[str] = None
    reason: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        if self.reason not in REASONS:
            raise ValueError(f"evidence.reason {self.reason!r} is not one of {sorted(REASONS)}")
        return {"calls": self.calls, "llm_calls": self.llm_calls, "answered_by": self.answered_by,
                "reason": self.reason, **self.extra}


def guarded_tool_call(ctx: FailureContext, tracker: BudgetTracker, ev: Evidence,
                      tool: str, args: dict) -> Optional[ToolCallResult]:
    """Issue one tool call if the budget allows; None (and nothing issued) if not."""
    if not tracker.can_call_tool():
        return None
    tracker.attempts += 1
    result = ctx.call_tool(tool, args)
    ev.calls.append({
        "tool": tool, "args": args, "http_status": result.http_status, "success": result.success,
        "injected": result.injected, "injected_kind": result.injected_kind,
        "error_text": _clip(result.error_text), "latency_ms": round(result.latency_ms, 1),
    })
    if result.success and ev.answered_by is None:
        ev.answered_by = tool
    return result


def guarded_llm_call(ctx: FailureContext, tracker: BudgetTracker, ev: Evidence,
                     purpose: str, prompt: str, max_completion: int) -> Optional[LLMReply]:
    """One LLM call if the token/wall budget allows; None if not. Raises
    LLMError on provider failure (the executor records it as llm_error)."""
    if ctx.llm is None:
        raise RuntimeError(f"{purpose}: this strategy needs an LLM client and none was configured")
    if not tracker.can_call_llm(prompt, max_completion):
        return None
    timeout_s = max(0.001, min(LLM_TIMEOUT_CAP_S, tracker.remaining_ms / 1000.0))
    reply = ctx.llm.complete(prompt, max_tokens=max_completion, timeout_s=timeout_s)
    tracker.tokens += reply.prompt_tokens + reply.completion_tokens
    ev.llm_calls.append({
        "purpose": purpose, "model_id": reply.model_id, "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens, "latency_ms": round(reply.latency_ms, 1),
    })
    return reply


def finish(tracker: BudgetTracker, ev: Evidence, success: bool) -> RecoveryResult:
    # Decided on the unrounded tracker values: latency_ms below is rounded to
    # 0.1 ms for storage and must not be what flags an overrun.
    ev.extra["budget_exceeded"] = tracker.exceeded()
    return RecoveryResult(
        success=success, cost_tokens=tracker.tokens, latency_ms=round(tracker.elapsed_ms, 1),
        attempts=tracker.attempts, evidence=ev.to_dict(),
    )
