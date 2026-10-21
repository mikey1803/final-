"""REPLAN (v1 = v2, strategy_definitions_v2.md §4).

The model sees the goal, the step, the failed call and its error, and returns
exactly one call for a DIFFERENT approach (any tool, any args); it is
executed once. ``replan_chosen_tool`` and ``same_call`` are logged so REPLAN's
overlap with SWITCH_TOOL / RETRY is measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

from recovery.live.executors._single_call import model_call_then_issue
from recovery.live.executors.base import Budget, BudgetTracker, Evidence, FailureContext, LLMError, RecoveryResult, finish
from recovery.live.executors.prompts import replan_prompt


@dataclass
class ReplanExecutor:
    definition_version: str = "v2"
    max_completion_tokens: int = 300
    name: str = "replan"

    @classmethod
    def v1(cls) -> "ReplanExecutor":
        return cls(definition_version="v1")

    def config(self) -> dict:
        return {"max_completion_tokens": self.max_completion_tokens}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        prompt = replan_prompt(ctx.task.goal, ctx.step_goal, ctx.tool_name, ctx.tool_args,
                               ctx.failed_result.error_text)
        try:
            model_call_then_issue(ctx, tracker, ev, "replan", prompt, self.max_completion_tokens,
                                  "replan_chosen_tool")
        except LLMError as exc:
            ev.reason = "llm_error"
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")
