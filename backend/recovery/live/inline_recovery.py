"""Inline recovery: on a captured tool-call failure, draw one strategy, run
its executor under a budget, and store a RecoveryAttempt (Task 21).

Selection (strategy_definitions_v2.md §2): uniform over the six real
strategies, propensity 1/6 recorded on every attempt; NULL_CONTROL joins the
draw only once its ``wait_s`` is set. Each failure's randomness comes from
``numpy.random.SeedSequence(session_seed, spawn_key=(failure_index,))`` --
never ``hash()`` -- and the derived seed is recorded, so the draw and the
backoff jitter are reproducible.

Failures of the model's own tool selection (``tool_name="<llm_tool_selection>"``)
are NOT recovered: v2 defines strategies for tool calls only. They are counted
as ``recovery_not_applicable_selection_failure`` and left for a v3 decision.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from recovery.live.event_schema import FailureEvent, Task
from recovery.live.executors import build_executors
from recovery.live.executors.base import Budget, FailureContext, LLMClient, Responder
from recovery.live.recovery_attempt import RecoveryAttempt, RecoveryStore
from recovery.live.tools import ToolCallResult

__all__ = ["UniformPolicy", "InlineRecovery", "SELECTION_FAILURE_TOOL", "NOT_APPLICABLE_SELECTION_FAILURE"]

SELECTION_FAILURE_TOOL = "<llm_tool_selection>"
NOT_APPLICABLE_SELECTION_FAILURE = "recovery_not_applicable_selection_failure"


class UniformPolicy:
    name = "uniform"

    def choose(self, arms: list[str], rng: np.random.Generator) -> tuple[str, float]:
        return arms[int(rng.integers(len(arms)))], 1.0 / len(arms)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InlineRecovery:
    store: RecoveryStore
    session_seed: int
    definition_version: str = "v2"
    budget: Budget = field(default_factory=Budget)
    llm: Optional[LLMClient] = None
    responder: Optional[Responder] = None
    policy: UniformPolicy = field(default_factory=UniformPolicy)
    null_control_wait_s: Optional[float] = None  # [RESULT REQUIRED] until the timing pre-study
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    failure_index: int = 0
    skipped: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.executors = build_executors(self.definition_version, self.null_control_wait_s)
        self.arms = list(self.executors)  # null_control included only when wait_s is set

    def recover(self, event: FailureEvent, task: Task, step_goal: str, failed_result: Optional[ToolCallResult],
                history: list[dict], call_tool: Callable[[str, dict], ToolCallResult]) -> Optional[RecoveryAttempt]:
        if event.tool_name == SELECTION_FAILURE_TOOL or failed_result is None:
            self.skipped[NOT_APPLICABLE_SELECTION_FAILURE] = self.skipped.get(NOT_APPLICABLE_SELECTION_FAILURE, 0) + 1
            return None

        index = self.failure_index
        self.failure_index += 1
        seq = np.random.SeedSequence(self.session_seed, spawn_key=(index,))
        derived = int(seq.generate_state(1, dtype=np.uint64)[0])
        strategy, propensity = self.policy.choose(self.arms, np.random.default_rng(derived))
        executor = self.executors[strategy]

        ctx = FailureContext(
            event=event, task=task, step_goal=step_goal, tool_name=event.tool_name,
            tool_args=dict(event.tool_args), failed_result=failed_result, history=list(history),
            call_tool=call_tool, rng=random.Random(derived), llm=self.llm, responder=self.responder,
            sleep=self.sleep, clock=self.clock,
        )
        started = _now_iso()
        t0 = self.clock()
        try:
            result = executor.execute(ctx, self.budget)
            result_d = result.to_dict()
        except Exception as exc:  # a bug in an executor must not crash the live run
            result_d = {"success": False, "cost_tokens": 0, "latency_ms": round((self.clock() - t0) * 1000, 1),
                        "attempts": 0, "evidence": {"calls": [], "llm_calls": [], "answered_by": None,
                                                    "reason": "executor_error",
                                                    "executor_error": f"{type(exc).__name__}: {exc}"[:300]}}
        ev = result_d["evidence"]
        used = {"attempts": result_d["attempts"], "wall_ms": result_d["latency_ms"], "tokens": result_d["cost_tokens"]}
        b = self.budget
        attempt = RecoveryAttempt(
            event_id=event.event_id, session_id=event.session_id, task_id=event.task_id,
            strategy=strategy, definition_version=self.definition_version,
            selection_policy=self.policy.name, selection_propensity=propensity,
            seed={"session_seed": self.session_seed, "failure_index": index, "derived_seed": derived,
                  "method": "numpy.SeedSequence(session_seed, spawn_key=(failure_index,))"},
            model_id=next((c["model_id"] for c in ev["llm_calls"]), None),
            executor_config={"name": executor.name, **executor.config(),
                             "responder": ev.get("responder"), "arms": self.arms},
            budget_limits=b.as_dict(), budget_used=used,
            # executor's own unrounded check; for an executor_error, judged from what was recorded
            budget_exceeded=bool(ev.get("budget_exceeded", used["attempts"] > b.max_attempts
                                        or used["tokens"] > b.max_tokens)),
            result=result_d, started_at=started, finished_at=_now_iso(),
            aborted=bool(ev.get("aborted", False)), no_action_taken=bool(ev.get("no_action_taken", False)),
        )
        self.store.write(attempt)
        return attempt
