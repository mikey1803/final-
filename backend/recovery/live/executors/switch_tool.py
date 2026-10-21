"""SWITCH_TOOL (v1 = v2 action, v2 translation rules).

Deterministic substitution by ``recovery.live.tool_registry``, issued once
through the same FlakyWrapper (so the peer faces the same injection risk).
Untranslatable args: no call, ``success=False``,
``evidence.reason="untranslatable_args"`` -- "could not apply", kept distinct
from "tried and failed".
"""

from __future__ import annotations

from dataclasses import dataclass

from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, RecoveryResult, finish, guarded_tool_call,
)
from recovery.live.tool_registry import translate_args


@dataclass
class SwitchToolExecutor:
    definition_version: str = "v2"
    name: str = "switch_tool"

    @classmethod
    def v1(cls) -> "SwitchToolExecutor":
        return cls(definition_version="v1")

    def config(self) -> dict:
        return {"translation": "tool_registry v2 rules"}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        tr = translate_args(ctx.tool_name, ctx.tool_args)
        ev.extra["translation"] = {"source": tr.source, "target": tr.target, "ok": tr.ok,
                                   "translated_args": tr.args, "why_untranslatable": tr.reason}
        if not tr.ok:
            ev.reason = "untranslatable_args"
            return finish(tracker, ev, False)
        result = guarded_tool_call(ctx, tracker, ev, tr.target, tr.args)
        if result is None:
            ev.reason = "budget_exhausted"
        else:
            ev.reason = "success" if result.success else "exhausted_attempts"
        return finish(tracker, ev, ev.reason == "success")
