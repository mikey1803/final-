"""The v3 agent tool selector (v3 pre-registration D-2).

Layer 2 copy of the Layer 1 selector (``src.agent.llm_client.parse_tool_request``,
read-only, unchanged): the prompt is **byte-identical** and the parsing rules
and error messages are the same, so a selection failure means exactly what it
means in Layer 1. Only the call settings differ -- the approved OneShotLLM-v2
settings: ``openai/gpt-oss-20b``, ``max_tokens=300``, ``reasoning_effort="low"``,
temperature 0 (Layer 1: ``max_tokens=200``, no reasoning effort, which left
most hard prompts with an EMPTY reply in the s = 0 smoke run: reasoning tokens
used up the cap).

``LAYER1_SOURCE_SHA256`` pins the Layer 1 function this copies; a test fails
if Layer 1 ever changes, and another checks the prompts are byte-identical.
Every call is logged (empty reply or not) for the empty-reply rate.

D-9: an HTTP 400 ``tool_use_failed`` (the model emitted a native tool call --
the Layer 1 wording "by calling exactly one tool" invites it) is raised as
``ToolUseFailed`` (a ``ValueError``), so it is an agent-side
``malformed_output`` like any other unusable reply, with the model's
``failed_generation`` kept.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from recovery.live.executors.base import LLMClient, LLMError
from recovery.live.tools import TOOL_DESCRIPTIONS

__all__ = ["SELECTOR_MAX_TOKENS", "SELECTOR_REASONING_EFFORT", "LAYER1_SOURCE_SHA256", "selector_prompt",
           "V3Selector", "ToolUseFailed", "tool_use_failed_generation"]

SELECTOR_MAX_TOKENS = 300
SELECTOR_REASONING_EFFORT = "low"
SELECTOR_TIMEOUT_S = 20.0
LAYER1_SOURCE_SHA256 = "e969b75094f111114303455dca9087228891e2979ae1da7ebdde14bb2bf016ef"


def selector_prompt(user_request: str, tool_descriptions: list[dict]) -> str:
    """Byte-identical to the prompt built inside Layer 1 ``parse_tool_request``."""
    tools_desc = "\n".join(
        f"- {t['name']}: {t['description']} (args: {t['args']})" for t in tool_descriptions
    )
    return (
        "You are an AI agent that fulfils a user's request by calling exactly one tool.\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        f"User request: {user_request}\n\n"
        'Respond with ONLY a JSON object of the form {"tool": "<tool_name>", "args": {...}}, '
        "choosing the single best tool and filling in its arguments from the request. "
        "No other text, no markdown formatting."
    )


class ToolUseFailed(ValueError):
    """D-9: the provider rejected the reply because the model emitted a native
    tool call instead of the requested JSON text (HTTP 400 ``tool_use_failed``).
    An unusable selection -- agent-side ``malformed_output`` -- not a provider fault.
    ``failed_generation`` is the model's attempted call, from the error body."""

    def __init__(self, failed_generation: Optional[str]) -> None:
        super().__init__("LLM returned a native tool call instead of the requested JSON text "
                         f"(tool_use_failed): {failed_generation!r}")
        self.failed_generation = failed_generation


def tool_use_failed_generation(exc: Exception) -> Optional[tuple]:
    """``(failed_generation,)`` if ``exc`` is a Groq HTTP 400 ``tool_use_failed``, else None."""
    if not isinstance(exc, LLMError) or getattr(exc, "status", None) != 400 or not getattr(exc, "body", None):
        return None
    try:
        err = json.loads(exc.body).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return None
    if err.get("code") != "tool_use_failed":
        return None
    return (err.get("failed_generation"),)


def _parse_like_layer1(raw: str) -> dict:
    """Layer 1's parsing rules and error messages, unchanged."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text!r}") from exc
    if "tool" not in parsed or "args" not in parsed:
        raise ValueError(f"LLM JSON missing 'tool'/'args' keys: {parsed!r}")
    return parsed


class V3Selector:
    """``select(step_goal) -> {"tool", "args"}``; raises ValueError on an unusable
    reply (Layer 1 semantics) and lets provider errors (``LLMError``) propagate."""

    def __init__(self, llm: LLMClient, tool_descriptions: Optional[list[dict]] = None) -> None:
        self.llm = llm
        self.tool_descriptions = tool_descriptions or TOOL_DESCRIPTIONS
        self.log: list[dict] = []
        self.context: dict = {}

    def settings(self) -> dict:
        return {"model": getattr(self.llm, "model", None), "max_tokens": SELECTOR_MAX_TOKENS,
                "reasoning_effort": getattr(self.llm, "reasoning_effort", None), "temperature": 0.0,
                "prompt": "byte-identical to src.agent.llm_client.parse_tool_request",
                "layer1_source_sha256": LAYER1_SOURCE_SHA256}

    def __call__(self, step_goal: str) -> dict:
        prompt = selector_prompt(step_goal, self.tool_descriptions)
        rec = {"ts_utc": datetime.now(timezone.utc).isoformat(), **self.context}
        try:
            reply = self.llm.complete(prompt, max_tokens=SELECTOR_MAX_TOKENS, timeout_s=SELECTOR_TIMEOUT_S)
        except Exception as exc:
            tuf = tool_use_failed_generation(exc)
            if tuf is not None:                      # D-9: an unusable selection, not a provider fault
                rec.update(outcome="tool_use_failed", failed_generation=tuf[0])
                self.log.append(rec)
                raise ToolUseFailed(tuf[0]) from None
            rec.update(outcome="provider_error", error=f"{type(exc).__name__}: {exc}"[:300])
            self.log.append(rec)
            raise
        rec.update(model_id=reply.model_id, prompt_tokens=reply.prompt_tokens,
                   completion_tokens=reply.completion_tokens, empty=not reply.text.strip())
        try:
            parsed = _parse_like_layer1(reply.text)
        except ValueError:
            rec["outcome"] = "empty" if rec["empty"] else "unparseable"
            self.log.append(rec)
            raise
        rec["outcome"] = "parsed"
        self.log.append(rec)
        return parsed
