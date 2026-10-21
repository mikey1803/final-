"""RE_GROUND (strategy_definitions_v2.md §4).

v2: the model gets the original goal, the step goal and the last good state
(most recent successful tool result) -- NOT the failure -- and re-selects the
call once; that call is issued once. Distinct from RETRY (the call can change)
and from REPLAN (no failure shown, no new approach requested).

v1 (``ReGroundExecutor.v1()``): re-inject context, re-issue the same call
unchanged, as frozen. A tool call is deterministic given its args, so v1
RE_GROUND behaves like RETRY for tool-level failures; recorded, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from recovery.live.executors._single_call import model_call_then_issue
from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, LLMError, RecoveryResult, finish, guarded_tool_call,
)
from recovery.live.executors.prompts import reground_prompt


@dataclass
class ReGroundExecutor:
    definition_version: str = "v2"
    max_completion_tokens: int = 300
    name: str = "re_ground"

    @classmethod
    def v1(cls) -> "ReGroundExecutor":
        return cls(definition_version="v1")

    def config(self) -> dict:
        mode = "model_reselects_call" if self.definition_version == "v2" else "same_call_unchanged"
        return {"mode": mode, "max_completion_tokens": self.max_completion_tokens}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        last_good = ctx.last_good_state()
        ev.extra["last_good_state"] = None if last_good is None else {
            "tool": last_good.get("tool"), "args": last_good.get("args")}
        try:
            if self.definition_version == "v1":
                ev.extra.update(reground_chosen_tool=ctx.tool_name, same_call=True, context_reinjected=True)
                result = guarded_tool_call(ctx, tracker, ev, ctx.tool_name, ctx.tool_args)
                ev.reason = "budget_exhausted" if result is None else (
                    "success" if result.success else "exhausted_attempts")
            else:
                prompt = reground_prompt(ctx.task.goal, ctx.step_goal, last_good)
                model_call_then_issue(ctx, tracker, ev, "re_ground_reselect", prompt,
                                      self.max_completion_tokens, "reground_chosen_tool")
        except LLMError as exc:
            ev.reason = "llm_error"
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")
