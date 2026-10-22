"""VERIFY (strategy_definitions_v2.md §4).

v2: on the last tool result run (1) the per-tool schema check, then (2) the
LLM judge -- both ON by default, each switchable off. A check that rejects
stops the remaining checks for that result. If the original result is
rejected, or there is no result at all (the call raised / timed out), one
corrective re-issue of the same call is made and its result goes through the
same enabled checks. If the original result already passes, VERIFY did
nothing: ``success=True, attempts=0, reason="original_result_accepted",
no_action_taken=True`` -- analysis reports VERIFY both with and without these.

v1 (``VerifyExecutor.v1()``): LLM judge only, as frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, LLMError, RecoveryResult, finish,
    guarded_llm_call, guarded_tool_call,
)
from recovery.live.executors.prompts import judge_prompt, parse_judgement
from recovery.live.executors.schemas import check_output_schema
from recovery.live.tools import ToolCallResult

_VERDICT_REASON = {"budget": "budget_exhausted", "unparseable": "unparseable_llm_output"}


@dataclass
class VerifyExecutor:
    definition_version: str = "v2"
    schema_check: bool = True
    llm_judge: bool = True
    judge_max_tokens: int = 200
    name: str = "verify"

    def __post_init__(self) -> None:
        if not (self.schema_check or self.llm_judge):
            raise ValueError("VERIFY needs at least one enabled check")

    @classmethod
    def v1(cls) -> "VerifyExecutor":
        return cls(definition_version="v1", schema_check=False, llm_judge=True)

    def config(self) -> dict:
        return {"schema_check": self.schema_check, "llm_judge": self.llm_judge,
                "judge_max_tokens": self.judge_max_tokens}

    def _judge_prompt(self, ctx, result: ToolCallResult) -> str:
        """v2 judge input: the first 500 characters of the body (``prompts.judge_prompt``).
        Overridden by the v3 executor (v3 pre-registration D-4)."""
        body = result.full_text if result.full_text is not None else result.body
        return judge_prompt(ctx.step_goal, ctx.tool_name, result.http_status, body, result.error_text)

    def _run_checks(self, ctx, tracker, ev, result: ToolCallResult, stage: str, log: list) -> str:
        """'pass' | 'reject' | 'budget' | 'unparseable'. LLMError propagates."""
        if self.schema_check:
            ok, why = check_output_schema(ctx.tool_name, result)
            log.append({"stage": stage, "check": "schema", "passed": ok, "detail": why})
            if not ok:
                return "reject"
        if self.llm_judge:
            prompt = self._judge_prompt(ctx, result)
            reply = guarded_llm_call(ctx, tracker, ev, "verify_judge", prompt, self.judge_max_tokens)
            if reply is None:
                return "budget"
            verdict = parse_judgement(reply.text)
            if verdict is None:
                log.append({"stage": stage, "check": "llm_judge", "passed": None, "detail": "unparseable"})
                return "unparseable"
            log.append({"stage": stage, "check": "llm_judge", "passed": verdict["satisfies"],
                        "detail": verdict["reason"]})
            if not verdict["satisfies"]:
                return "reject"
        return "pass"

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        log: list[dict] = []
        ev.extra.update(checks=log, no_action_taken=False)
        try:
            original = ctx.failed_result
            has_result = original.http_status is not None
            ev.extra["original_had_result"] = has_result
            if has_result:
                verdict = self._run_checks(ctx, tracker, ev, original, "original", log)
                if verdict == "pass":
                    ev.reason = "original_result_accepted"
                    ev.extra["no_action_taken"] = True
                    return finish(tracker, ev, True)
                if verdict in _VERDICT_REASON:
                    ev.reason = _VERDICT_REASON[verdict]
                    return finish(tracker, ev, False)
            reissued: Optional[ToolCallResult] = guarded_tool_call(ctx, tracker, ev, ctx.tool_name, ctx.tool_args)
            if reissued is None:
                ev.reason = "budget_exhausted"
                return finish(tracker, ev, False)
            verdict = self._run_checks(ctx, tracker, ev, reissued, "reissue", log)
            ev.reason = {"pass": "success", "reject": "checks_rejected"}.get(verdict) or _VERDICT_REASON[verdict]
            if ev.reason != "success":
                ev.answered_by = None  # a re-issue that fails a check answered nothing
        except LLMError as exc:
            ev.reason, ev.answered_by = "llm_error", None
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")
