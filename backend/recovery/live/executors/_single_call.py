"""The 'ask the model for one call, issue it once' pattern shared by
RE_GROUND (v2), REPLAN and CLARIFY."""

from __future__ import annotations

from recovery.live.executors.base import BudgetTracker, Evidence, FailureContext, guarded_llm_call, guarded_tool_call
from recovery.live.executors.prompts import parse_tool_call


def model_call_then_issue(ctx: FailureContext, tracker: BudgetTracker, ev: Evidence, purpose: str,
                          prompt: str, max_completion: int, chosen_tool_field: str) -> None:
    """Sets ev.reason; raises LLMError on provider failure."""
    reply = guarded_llm_call(ctx, tracker, ev, purpose, prompt, max_completion)
    if reply is None:
        ev.reason = "budget_exhausted"
        return
    call = parse_tool_call(reply.text)
    if call is None:
        ev.reason = "unparseable_llm_output"
        ev.extra["llm_output_excerpt"] = reply.text[:200]
        return
    ev.extra[chosen_tool_field] = call["tool"]
    ev.extra["chosen_args"] = call["args"]
    ev.extra["same_call"] = call["tool"] == ctx.tool_name and call["args"] == ctx.tool_args
    result = guarded_tool_call(ctx, tracker, ev, call["tool"], call["args"])
    if result is None:
        ev.reason = "budget_exhausted"
    else:
        ev.reason = "success" if result.success else "exhausted_attempts"
