"""RETRY (strategy_definitions_v2.md §4).

v2: re-issue the identical call up to ``max_attempts`` times, waiting
``U(0, min(cap, base * 2**k))`` before re-issue k (full jitter); stop at the
first success, at ``max_attempts``, or when the next attempt would not fit the
budget. v1 (``RetryExecutor.v1()``): one immediate re-issue, zero backoff --
exactly the frozen v1 definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, RecoveryResult, finish, guarded_tool_call,
)


@dataclass
class RetryExecutor:
    definition_version: str = "v2"
    max_attempts: Optional[int] = None  # None: use the budget's max_attempts
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 4.0
    name: str = "retry"

    @classmethod
    def v1(cls) -> "RetryExecutor":
        return cls(definition_version="v1", max_attempts=1, backoff_base_s=0.0, backoff_cap_s=0.0)

    def config(self) -> dict:
        return {"max_attempts": self.max_attempts, "backoff": "full_jitter",
                "backoff_base_s": self.backoff_base_s, "backoff_cap_s": self.backoff_cap_s}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        limit = min(self.max_attempts or budget.max_attempts, budget.max_attempts)
        delays = []
        ev.reason = "exhausted_attempts"
        for k in range(limit):
            delay = ctx.rng.uniform(0.0, min(self.backoff_cap_s, self.backoff_base_s * 2 ** k)) \
                if self.backoff_base_s > 0 else 0.0
            if not tracker.can_call_tool(after_sleep_s=delay):
                ev.reason = "budget_exhausted"
                break
            if delay:
                ctx.sleep(delay)
            delays.append(round(delay, 4))
            result = guarded_tool_call(ctx, tracker, ev, ctx.tool_name, ctx.tool_args)
            if result is None:
                ev.reason = "budget_exhausted"
                break
            if result.success:
                ev.reason = "success"
                break
        ev.extra["backoff_delays_s"] = delays
        return finish(tracker, ev, ev.reason == "success")
