"""Generic recovery executors for tools other than the six live ones (Task 22).

The live executors (``recovery.live.executors``, frozen v2 definitions) are
tied to the six live tools: their prompts list those tools, ``parse_tool_call``
accepts only their names, VERIFY's schemas cover only them and SWITCH_TOOL
translates only between them. They are used unchanged for those six names.

For any other tool the wrapper uses these Layer-2 variants. They reuse the
existing ``BudgetTracker``, ``guarded_tool_call``, ``guarded_llm_call`` and
``finish`` (so the v2 budget rule is identical) and the existing prompts: each
prompt is the live prompt with only its tool list replaced by the runtime's
tools. Nothing in ``recovery.live`` is modified.

- retry: the existing ``RetryExecutor`` (already tool-agnostic).
- switch_tool: the backup and argument translation come from the wrapper's
  registry (``backup``, ``translate``); no translator = the same args.
- replan / re_ground / clarify: model-chosen call among the runtime's tools.
- verify: the existing ``VerifyExecutor`` with the LLM judge only (no
  schema exists for an arbitrary tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, LLMError, RecoveryResult, TOOL_CALL_RESERVE_MS, finish,
    guarded_llm_call, guarded_tool_call,
)
from recovery.live.executors import prompts as live_prompts
from recovery.live.executors.responders import CannedAnswerResponder
from recovery.live.executors.retry import RetryExecutor
from recovery.live.executors.verify import VerifyExecutor

__all__ = ["ToolCatalog", "GenericSwitchToolExecutor", "GenericReplanExecutor", "GenericReGroundExecutor",
           "GenericClarifyExecutor", "build_generic_executors", "LLM_ARMS"]

LLM_ARMS = frozenset({"replan", "re_ground", "clarify"})   # plus verify when its judge is on


@dataclass
class ToolCatalog:
    """The runtime's tools as the LLM executors see them."""

    describe: Callable[[], list[dict]]                        # [{"name", "description", "args"}]
    backup_for: Callable[[str, dict], Optional[tuple[str, Optional[dict]]]]   # (backup, args|None)

    def names(self) -> set[str]:
        return {t["name"] for t in self.describe()}

    def block(self) -> str:
        return "\n".join(f"- {t['name']}: {t['description']} (args: {t['args']})" for t in self.describe())


def _with_tools(live_prompt: str, catalog: ToolCatalog) -> str:
    """The live prompt with its six-tool list swapped for the runtime's tools."""
    live_block = live_prompts._tools_block()
    if live_block not in live_prompt:
        raise RuntimeError("live prompt no longer contains the live tool list; generic prompt cannot be built")
    return live_prompt.replace(live_block, catalog.block())


def _parse_call(text: str, names: set[str]) -> Optional[dict]:
    obj = live_prompts._json_object(text)
    if not obj or obj.get("tool") not in names or not isinstance(obj.get("args"), dict):
        return None
    return {"tool": obj["tool"], "args": obj["args"]}


def _model_call_then_issue(ctx, tracker, ev, purpose, prompt, max_completion, field_name, catalog) -> None:
    """``executors._single_call.model_call_then_issue`` with the runtime's tool names."""
    reply = guarded_llm_call(ctx, tracker, ev, purpose, prompt, max_completion)
    if reply is None:
        ev.reason = "budget_exhausted"
        return
    call = _parse_call(reply.text, catalog.names())
    if call is None:
        ev.reason = "unparseable_llm_output"
        ev.extra["llm_output_excerpt"] = reply.text[:200]
        return
    ev.extra[field_name] = call["tool"]
    ev.extra["chosen_args"] = call["args"]
    ev.extra["same_call"] = call["tool"] == ctx.tool_name and call["args"] == ctx.tool_args
    result = guarded_tool_call(ctx, tracker, ev, call["tool"], call["args"])
    ev.reason = "budget_exhausted" if result is None else ("success" if result.success else "exhausted_attempts")


@dataclass
class GenericSwitchToolExecutor:
    catalog: ToolCatalog
    definition_version: str = "v2"
    name: str = "switch_tool"

    def config(self) -> dict:
        return {"translation": "with_recovery registry", "variant": "generic"}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        found = self.catalog.backup_for(ctx.tool_name, ctx.tool_args)
        target, args = (found if found else (None, None))
        ev.extra["translation"] = {"source": ctx.tool_name, "target": target, "ok": args is not None,
                                   "translated_args": args}
        if args is None:
            ev.reason = "untranslatable_args"
            return finish(tracker, ev, False)
        result = guarded_tool_call(ctx, tracker, ev, target, args)
        ev.reason = "budget_exhausted" if result is None else ("success" if result.success else "exhausted_attempts")
        return finish(tracker, ev, ev.reason == "success")


@dataclass
class GenericReplanExecutor:
    catalog: ToolCatalog
    max_completion_tokens: int = 300
    definition_version: str = "v2"
    name: str = "replan"

    def config(self) -> dict:
        return {"max_completion_tokens": self.max_completion_tokens, "variant": "generic"}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        prompt = _with_tools(live_prompts.replan_prompt(ctx.task.goal, ctx.step_goal, ctx.tool_name, ctx.tool_args,
                                                        ctx.failed_result.error_text), self.catalog)
        try:
            _model_call_then_issue(ctx, tracker, ev, "replan", prompt, self.max_completion_tokens,
                                   "replan_chosen_tool", self.catalog)
        except LLMError as exc:
            ev.reason = "llm_error"
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")


@dataclass
class GenericReGroundExecutor:
    catalog: ToolCatalog
    max_completion_tokens: int = 300
    definition_version: str = "v2"
    name: str = "re_ground"

    def config(self) -> dict:
        return {"mode": "model_reselects_call", "max_completion_tokens": self.max_completion_tokens,
                "variant": "generic"}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        last_good = ctx.last_good_state()
        ev.extra["last_good_state"] = None if last_good is None else {
            "tool": last_good.get("tool"), "args": last_good.get("args")}
        try:
            prompt = _with_tools(live_prompts.reground_prompt(ctx.task.goal, ctx.step_goal, last_good), self.catalog)
            _model_call_then_issue(ctx, tracker, ev, "re_ground_reselect", prompt, self.max_completion_tokens,
                                   "reground_chosen_tool", self.catalog)
        except LLMError as exc:
            ev.reason = "llm_error"
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")


@dataclass
class GenericClarifyExecutor:
    catalog: ToolCatalog
    question_max_tokens: int = 200
    action_max_tokens: int = 300
    responder_timeout_s: float = 60.0
    definition_version: str = "v2"
    name: str = "clarify"

    def config(self) -> dict:
        return {"question_max_tokens": self.question_max_tokens, "action_max_tokens": self.action_max_tokens,
                "responder_timeout_s": self.responder_timeout_s, "variant": "generic"}

    def execute(self, ctx: FailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        responder = ctx.responder or CannedAnswerResponder()
        ev.extra.update(responder=responder.kind, aborted=False)
        try:
            reply = guarded_llm_call(ctx, tracker, ev, "clarify_question",
                                     live_prompts.clarify_question_prompt(ctx.task.goal, ctx.step_goal),
                                     self.question_max_tokens)
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
            prompt = _with_tools(live_prompts.clarify_action_prompt(ctx.task.goal, ctx.step_goal, question, answer),
                                 self.catalog)
            _model_call_then_issue(ctx, tracker, ev, "clarify_action", prompt, self.action_max_tokens,
                                   "clarify_chosen_tool", self.catalog)
        except LLMError as exc:
            ev.reason = "llm_error"
            ev.extra["llm_error"] = str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")


def build_generic_executors(catalog: ToolCatalog) -> dict:
    """Same arm names and order as ``event_schema.STRATEGY_NAMES``."""
    return {
        "retry": RetryExecutor(),
        "replan": GenericReplanExecutor(catalog),
        "switch_tool": GenericSwitchToolExecutor(catalog),
        "re_ground": GenericReGroundExecutor(catalog),
        "verify": VerifyExecutor(schema_check=False, llm_judge=True),
        "clarify": GenericClarifyExecutor(catalog),
    }
