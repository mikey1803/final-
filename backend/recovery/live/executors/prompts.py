"""Prompts and output parsing shared by the LLM-using executors.

Every prompt asks for strict JSON; ``parse_tool_call`` / ``parse_judgement``
return None on anything unusable, which the executor records as
``unparseable_llm_output`` -- never repaired or guessed.
"""

from __future__ import annotations

import json
from typing import Optional

from recovery.live.tools import TOOL_DESCRIPTIONS, TOOLS

EXCERPT_CHARS = 500


def excerpt(text: Optional[str], n: int = EXCERPT_CHARS) -> str:
    if not text:
        return "(none)"
    return text if len(text) <= n else text[:n] + "...[truncated]"


def _tools_block() -> str:
    return "\n".join(f"- {t['name']}: {t['description']} (args: {t['args']})" for t in TOOL_DESCRIPTIONS)


# Phrased as "write a JSON description", not "call a tool": with tool-like
# wording gpt-oss on Groq attempts a native function call, which the API
# rejects (HTTP 400 tool_use_failed) because no tools are declared --
# observed live 2026-10-21.
_CALL_FORMAT = ("Do not call any function yourself. Write your decision as text: ONLY a JSON object "
                '{"tool": "<name from the list>", "args": {...}}. No other text, no markdown.')


def reground_prompt(goal: str, step_goal: str, last_good: Optional[dict]) -> str:
    state = "none" if last_good is None else (
        f"tool {last_good.get('tool')} with args {json.dumps(last_good.get('args'))} returned: "
        f"{excerpt(last_good.get('content'))}")
    return (
        "You are an AI agent deciding which ONE of the listed services to use for the current step.\n\n"
        f"Overall goal: {goal}\nCurrent step: {step_goal}\nLast good state: {state}\n\n"
        f"Available tools:\n{_tools_block()}\n\n{_CALL_FORMAT}"
    )


def replan_prompt(goal: str, step_goal: str, tool: str, args: dict, error_text: Optional[str]) -> str:
    return (
        "You are an AI agent. Your last tool call failed. Propose a DIFFERENT approach to the "
        "same step: exactly one tool call (any tool, any args).\n\n"
        f"Overall goal: {goal}\nCurrent step: {step_goal}\n"
        f"Failed call: tool {tool} with args {json.dumps(args)}\nError: {excerpt(error_text, 300)}\n\n"
        f"Available tools:\n{_tools_block()}\n\n{_CALL_FORMAT}"
    )


def clarify_question_prompt(goal: str, step_goal: str) -> str:
    return (
        "You are an AI agent about to act on a request that may be unclear. Ask the user ONE "
        "short clarifying question about the goal.\n\n"
        f"Overall goal: {goal}\nCurrent step: {step_goal}\n\n"
        "Respond with ONLY the question."
    )


def clarify_action_prompt(goal: str, step_goal: str, question: str, answer: str) -> str:
    return (
        "You are an AI agent deciding which ONE of the listed services to use for the current step.\n\n"
        f"Overall goal: {goal}\nCurrent step: {step_goal}\n"
        f"You asked: {question}\nThe user answered: {answer}\n\n"
        f"Available tools:\n{_tools_block()}\n\n{_CALL_FORMAT}"
    )


def judge_prompt(step_goal: str, tool: str, http_status: Optional[int], body: Optional[str],
                 error_text: Optional[str]) -> str:
    return (
        "You check an AI agent's tool result. Does the result actually satisfy the step?\n\n"
        f"Step: {step_goal}\nTool: {tool}\nHTTP status: {http_status}\n"
        f"Result: {excerpt(body)}\nError: {excerpt(error_text, 200)}\n\n"
        'Respond with ONLY a JSON object {"satisfies": true|false, "reason": "<one line>"}.'
    )


def _json_object(text: str) -> Optional[dict]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_tool_call(text: str) -> Optional[dict]:
    obj = _json_object(text)
    if not obj or obj.get("tool") not in TOOLS or not isinstance(obj.get("args"), dict):
        return None
    return {"tool": obj["tool"], "args": obj["args"]}


def parse_judgement(text: str) -> Optional[dict]:
    obj = _json_object(text)
    if not obj or not isinstance(obj.get("satisfies"), bool):
        return None
    return {"satisfies": obj["satisfies"], "reason": str(obj.get("reason", ""))[:200]}
