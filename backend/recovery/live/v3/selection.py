"""Agent-side failures: an unusable tool selection (v3 §5).

v2 skipped these. In v3 they are ordinary ``FailureEvent``s with
``fault_origin="agent_side"``, and every drawn strategy applies:

| strategy | on a selection failure |
|---|---|
| retry | re-run the selection with the v2 backoff, up to the attempt budget |
| replan | selection prompt + the failure text ("your previous answer was unusable: ..."), one call |
| re_ground | the v2 RE_GROUND executor unchanged: goal + step + last good state, no failure shown |
| verify | judge the selection reply; if rejected, re-run the selection once and check that call's result as in v2 |
| clarify | the v2 CLARIFY executor unchanged: one question, the canned answer, then selection |
| switch_tool | ``wrong_tool_selected`` naming a known tool: the registry peer with translated args; otherwise ``untranslatable_args`` |

Success (every strategy): a valid call is produced AND it succeeds by its
tool criterion (in ``degraded_output`` blocks the primary outcome is
``corrected``, computed by the instrument afterwards).

The provisional tag ``agent_side_kind`` is the instrument's rule, set at
capture; it is never copied into the human ``true_type`` label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from recovery.live.executors import ClarifyExecutor, ReGroundExecutor, SwitchToolExecutor
from recovery.live.executors.base import (
    Budget, BudgetTracker, Evidence, FailureContext, LLMError, RecoveryResult, _clip, finish, guarded_llm_call,
)
from recovery.live.executors._single_call import model_call_then_issue
from recovery.live.executors.prompts import _CALL_FORMAT, _tools_block, excerpt, parse_judgement
from recovery.live.tools import TOOL_DESCRIPTIONS, TOOLS, ToolCallResult

__all__ = [
    "SELECTION_FAILURE_TOOL", "EXPECTED_ARGS", "Selection", "validate_selection", "SelectionFailureContext",
    "build_selection_executors", "SelectionProviderError",
]

SELECTION_FAILURE_TOOL = "<llm_tool_selection>"
_NO_FG = object()
EXPECTED_ARGS: dict[str, frozenset] = {t["name"]: frozenset(t["args"]) for t in TOOL_DESCRIPTIONS}


class SelectionProviderError(RuntimeError):
    """The selection LLM call itself failed (network / provider), not the model's output."""


@dataclass(frozen=True)
class Selection:
    tool: Optional[str]
    args: Optional[dict]
    kind: Optional[str]        # None (usable) | "malformed_output" | "wrong_tool_selected"
    reply_text: str            # the model's reply (or the parser's error text quoting it)
    detail: Optional[dict] = None   # D-9: {"tool_use_failed": True, "failed_generation": ...}

    @property
    def usable(self) -> bool:
        return self.kind is None


def validate_selection(select: Callable[[str], dict], step_goal: str) -> Selection:
    """Run the agent's selection once and classify it (v3 §5 provisional tags).

    ``malformed_output``: the reply is not valid JSON or lacks ``tool`` / ``args``.
    ``wrong_tool_selected``: valid JSON, but ``tool`` is not one of the six, or
    the args don't match that tool's argument names.
    """
    try:
        picked = select(step_goal)
    except ValueError as exc:
        fg = getattr(exc, "failed_generation", _NO_FG)
        detail = None if fg is _NO_FG else {"tool_use_failed": True, "failed_generation": fg}
        return Selection(None, None, "malformed_output", str(exc), detail)
    except Exception as exc:  # provider / network failure: not the model's output
        raise SelectionProviderError(f"{type(exc).__name__}: {exc}"[:300]) from None
    if not isinstance(picked, dict) or "tool" not in picked or not isinstance(picked.get("args"), dict):
        return Selection(None, None, "malformed_output", json.dumps(picked, default=str)[:1000])
    tool, args = picked["tool"], picked["args"]
    text = json.dumps({"tool": tool, "args": args}, default=str)[:1000]
    if tool not in TOOLS or set(args) != EXPECTED_ARGS[tool]:
        return Selection(tool if isinstance(tool, str) else None, args, "wrong_tool_selected", text)
    return Selection(tool, args, None, text)


@dataclass
class SelectionFailureContext(FailureContext):
    select: Optional[Callable[[str], dict]] = None
    agent_side_kind: Optional[str] = None
    bad_selection: Optional[Selection] = None


def failed_selection_result(sel: Selection) -> ToolCallResult:
    return ToolCallResult(False, None, None, f"unusable tool selection ({sel.kind}): {sel.reply_text}", 0.0)


def _issue(ctx: SelectionFailureContext, ev: Evidence, tool: str, args: dict) -> ToolCallResult:
    """Issue the selected call; recorded exactly like ``guarded_tool_call``.
    The caller has already counted the attempt (a selection re-run is the attempt)."""
    result = ctx.call_tool(tool, args)
    ev.calls.append({
        "tool": tool, "args": args, "http_status": result.http_status, "success": result.success,
        "injected": result.injected, "injected_kind": result.injected_kind,
        "error_text": _clip(result.error_text), "latency_ms": round(result.latency_ms, 1),
        **({"v3": result.v3} if result.v3 is not None else {}),
    })
    if result.success and ev.answered_by is None:
        ev.answered_by = tool
    return result


def _rerun(ctx: SelectionFailureContext, ev: Evidence) -> Selection:
    sel = validate_selection(ctx.select, ctx.step_goal)
    ev.extra.setdefault("selection_reruns", []).append(
        {"kind": sel.kind, "tool": sel.tool, "args": sel.args, "reply_excerpt": sel.reply_text[:200],
         "detail": sel.detail})
    return sel


# ---------------------------------------------------------------- strategies

@dataclass
class SelRetry:
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 4.0
    name: str = "retry"
    definition_version: str = "v3"

    def config(self) -> dict:
        return {"on": "selection_failure", "backoff": "full_jitter", "backoff_base_s": self.backoff_base_s,
                "backoff_cap_s": self.backoff_cap_s}

    def execute(self, ctx: SelectionFailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        ev.reason, delays = "exhausted_attempts", []
        try:
            for k in range(budget.max_attempts):
                delay = ctx.rng.uniform(0.0, min(self.backoff_cap_s, self.backoff_base_s * 2 ** k))
                if not tracker.can_call_tool(after_sleep_s=delay):
                    ev.reason = "budget_exhausted"
                    break
                ctx.sleep(delay)
                delays.append(round(delay, 4))
                tracker.attempts += 1
                sel = _rerun(ctx, ev)
                if not sel.usable:
                    continue
                if _issue(ctx, ev, sel.tool, sel.args).success:
                    ev.reason = "success"
                    break
        except SelectionProviderError as exc:
            ev.reason, ev.extra["llm_error"] = "llm_error", str(exc)
        ev.extra["backoff_delays_s"] = delays
        return finish(tracker, ev, ev.reason == "success")


def selection_replan_prompt(goal: str, step_goal: str, reply_text: str) -> str:
    return (
        "You are an AI agent deciding which ONE of the listed services to use for the current step.\n\n"
        f"Overall goal: {goal}\nCurrent step: {step_goal}\n"
        f"Your previous answer was unusable: {excerpt(reply_text, 300)}\n\n"
        f"Available tools:\n{_tools_block()}\n\n{_CALL_FORMAT}"
    )


def selection_judge_prompt(step_goal: str, reply_text: str) -> str:
    return (
        "You check an AI agent's tool selection. Is this reply a usable choice of ONE of the listed "
        "services, with the right argument names, for the step?\n\n"
        f"Step: {step_goal}\nReply: {excerpt(reply_text, 300)}\n\n"
        f"Available tools:\n{_tools_block()}\n\n"
        'Respond with ONLY a JSON object {"satisfies": true|false, "reason": "<one line>"}.'
    )


@dataclass
class SelReplan:
    max_completion_tokens: int = 300
    name: str = "replan"
    definition_version: str = "v3"

    def config(self) -> dict:
        return {"on": "selection_failure", "max_completion_tokens": self.max_completion_tokens}

    def execute(self, ctx: SelectionFailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        prompt = selection_replan_prompt(ctx.task.goal, ctx.step_goal, ctx.bad_selection.reply_text)
        try:
            model_call_then_issue(ctx, tracker, ev, "replan_selection", prompt, self.max_completion_tokens,
                                  "replan_chosen_tool")
        except LLMError as exc:
            ev.reason, ev.extra["llm_error"] = "llm_error", str(exc)[:300]
        return finish(tracker, ev, ev.reason == "success")


@dataclass
class SelVerify:
    judge_max_tokens: int = 200
    name: str = "verify"
    definition_version: str = "v3"

    def config(self) -> dict:
        return {"on": "selection_failure", "judge_max_tokens": self.judge_max_tokens,
                "then": "v2 schema check + LLM judge (v3 judge input, D-4) on the re-selected call's result"}

    def execute(self, ctx: SelectionFailureContext, budget: Budget) -> RecoveryResult:
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        log: list[dict] = []
        ev.extra.update(checks=log, no_action_taken=False)
        try:
            reply = guarded_llm_call(ctx, tracker, ev, "verify_selection_judge",
                                     selection_judge_prompt(ctx.step_goal, ctx.bad_selection.reply_text),
                                     self.judge_max_tokens)
            if reply is None:
                ev.reason = "budget_exhausted"
                return finish(tracker, ev, False)
            verdict = parse_judgement(reply.text)
            if verdict is None:
                log.append({"stage": "selection", "check": "llm_judge", "passed": None, "detail": "unparseable"})
                ev.reason = "unparseable_llm_output"
                return finish(tracker, ev, False)
            log.append({"stage": "selection", "check": "llm_judge", "passed": verdict["satisfies"],
                        "detail": verdict["reason"]})
            if verdict["satisfies"]:
                # the reply is unusable by construction: accepting it recovers nothing
                ev.reason, ev.extra["no_action_taken"] = "original_result_accepted", True
                return finish(tracker, ev, False)
            if not tracker.can_call_tool():
                ev.reason = "budget_exhausted"
                return finish(tracker, ev, False)
            tracker.attempts += 1
            sel = _rerun(ctx, ev)
            if not sel.usable:
                ev.reason = "unparseable_llm_output"
                return finish(tracker, ev, False)
            result = _issue(ctx, ev, sel.tool, sel.args)
            from recovery.live.v3.verify_v3 import VerifyExecutorV3
            checker = VerifyExecutorV3()
            verdict2 = checker._run_checks(replace(ctx, tool_name=sel.tool, tool_args=sel.args), tracker, ev,
                                           result, "reissue", log)
            ev.reason = {"pass": "success", "reject": "checks_rejected", "budget": "budget_exhausted",
                         "unparseable": "unparseable_llm_output"}[verdict2]
            if ev.reason != "success":
                ev.answered_by = None
        except LLMError as exc:
            ev.reason, ev.answered_by = "llm_error", None
            ev.extra["llm_error"] = str(exc)[:300]
        except SelectionProviderError as exc:
            ev.reason, ev.answered_by = "llm_error", None
            ev.extra["llm_error"] = str(exc)
        return finish(tracker, ev, ev.reason == "success")


@dataclass
class SelSwitchTool:
    name: str = "switch_tool"
    definition_version: str = "v3"

    def config(self) -> dict:
        return {"on": "selection_failure", "translation": "tool_registry v2 rules, wrong_tool_selected only"}

    def execute(self, ctx: SelectionFailureContext, budget: Budget) -> RecoveryResult:
        sel = ctx.bad_selection
        if ctx.agent_side_kind == "wrong_tool_selected" and sel.tool in TOOLS:
            try:
                return SwitchToolExecutor().execute(replace(ctx, tool_name=sel.tool, tool_args=dict(sel.args)), budget)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                why = f"translation failed: {type(exc).__name__}: {exc}"[:200]
        else:
            why = f"{ctx.agent_side_kind}: no known tool to substitute"
        tracker, ev = BudgetTracker(budget, ctx.clock), Evidence()
        ev.reason = "untranslatable_args"
        ev.extra["translation"] = {"source": sel.tool, "target": None, "ok": False, "translated_args": None,
                                   "why_untranslatable": why}
        return finish(tracker, ev, False)


def build_selection_executors() -> dict:
    """Selection-failure executors, in the canonical STRATEGY_NAMES order."""
    return {
        "retry": SelRetry(), "replan": SelReplan(), "switch_tool": SelSwitchTool(),
        "re_ground": ReGroundExecutor(), "verify": SelVerify(), "clarify": ClarifyExecutor(),
    }
