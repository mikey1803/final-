"""CLARIFY (v1 = v2, strategy_definitions_v2.md §4).

The model asks ONE clarifying question; the responder answers (canned by
default); the model re-selects the call given goal + question + answer; the
call is issued once. If the responder returns None or times out: stop, no
second question, ``success=False``, ``reason="question_unmatched"``,
``aborted`` -- excluded from success tallies, reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from recovery.live.executors._single_call import model_call_then_issue
from recovery.live.executors.base import (
    TOOL_CALL_RESERVE_MS, Budget, BudgetTracker, Evidence, FailureContext, LLMError, RecoveryResult,
    finish, guarded_llm_call,
)
from recovery.live.executors.prompts import clarify_action_prompt, clarify_question_prompt
from recovery.live.executors.responders import CannedAnswerResponder


@dataclass
class ClarifyExecutor:
    definition_version: str = "v2"
    question_max_tokens: int = 200
    action_max_tokens: int = 300
    responder_timeout_s: float = 60.0
    name: str = "clarify"

    @classmethod
    def v1(cls) -> "ClarifyExecutor":
        return cls(definition_version="v1")

    def config(self) -> dict:
        return {"question_max_tokens": self.question_max_tokens, "action_max_tokens": self.action_max_tokens,
                "responder_timeout_s": self.responder_timeout_s}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        responder = ctx.responder or CannedAnswerResponder()
        ev.extra.update(responder=responder.kind, aborted=False)
        try:
            reply = guarded_llm_call(ctx, tracker, ev, "clarify_question",
                                     clarify_question_prompt(ctx.task.goal, ctx.step_goal), self.question_max_tokens)
            if reply is None:
                ev.reason = "budget_exhausted"
                return finish(tracker, ev, False)
            question = reply.text.strip()
            ev.extra["question"] = question[:300]
            timeout_s = max(0.0, min(self.responder_timeout_s, (tracker.remaining_ms - TOOL_CALL_RESERVE_MS) / 1000.0))
            answer = responder.answer(question, ctx.task, timeout_s) if question else None
            ev.extra["answer"] = answer
            if answer is None:
                ev.reason = "question_unmatched"
                ev.extra["aborted"] = True
                return finish(tracker, ev, False)
            model_call_then_issue(ctx, tracker, ev, "clarify_action",
                                  clarify_action_prompt(ctx.task.goal, ctx.step_goal, question, answer),
                                  self.action_max_tokens, "clarify_chosen_tool")
        except LLMError as exc:
            ev.reason = "llm_error"
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")
