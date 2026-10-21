"""NULL_CONTROL (v1 = v2): a fixed wait of ``wait_s``, then one identical re-issue.

``wait_s`` has no default: it is frozen by the Phase 2 timing pre-study and is
[RESULT REQUIRED] until then. NULL_CONTROL is excluded from inline selection
until it is set (strategy_definitions_v2.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass

from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, RecoveryResult, finish, guarded_tool_call,
)


@dataclass
class NullControlExecutor:
    wait_s: float
    definition_version: str = "v2"
    name: str = "null_control"

    def __post_init__(self) -> None:
        if not isinstance(self.wait_s, (int, float)) or self.wait_s < 0:
            raise ValueError("wait_s must be a non-negative number (frozen by the timing pre-study)")

    def config(self) -> dict:
        return {"wait_s": self.wait_s}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        if not tracker.can_call_tool(after_sleep_s=self.wait_s):
            ev.reason = "budget_exhausted"
            return finish(tracker, ev, False)
        ctx.sleep(self.wait_s)
        result = guarded_tool_call(ctx, tracker, ev, ctx.tool_name, ctx.tool_args)
        if result is None:
            ev.reason = "budget_exhausted"
        else:
            ev.reason = "success" if result.success else "exhausted_attempts"
        return finish(tracker, ev, ev.reason == "success")
