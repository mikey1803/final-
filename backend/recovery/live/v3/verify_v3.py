"""VERIFY under v3: the judge sees the task's extracted answer field first
(v3 pre-registration D-4).

Judge input: the answer field (``recovery.live.v3.answers.ANSWER_FIELDS``) as
delivered, truncated to 500 characters if longer, followed by the first 500
characters of the raw body. Under the v2 input (body excerpt only) the
500-character window never reaches Wikipedia's ``extract`` field, so
detection there would measure truncation, not judging. The v2 executor
(``recovery.live.executors.verify.VerifyExecutor``) is unchanged and stays
runnable for replay. The offline false-alarm calibration uses this same
prompt (pre-reg §6).

The answer shown is the DELIVERED value (what the agent received), never the
genuine value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from recovery.live.executors.prompts import excerpt
from recovery.live.executors.verify import VerifyExecutor
from recovery.live.tools import ToolCallResult
from recovery.live.v3.answers import ANSWER_FIELDS, read_answer

__all__ = ["VerifyExecutorV3", "judge_prompt_v3", "answer_field_label", "answer_field_text", "JUDGE_INPUT"]

JUDGE_INPUT = "v3_answer_field_then_body_excerpt"
ANSWER_CHARS = 500
BODY_CHARS = 500


def answer_field_label(tool: str) -> str:
    path = ANSWER_FIELDS.get(tool)
    if path is None:
        return "(no answer field for this tool)"
    out = ""
    for step in path:
        out += f"[{step}]" if isinstance(step, int) else (f".{step}" if out else step)
    return out


def answer_field_text(tool: str, body: Optional[str]) -> str:
    ans = read_answer(tool, body)
    if not ans.present:
        return "(missing)"
    raw = ans.raw if isinstance(ans.raw, str) else json.dumps(ans.raw, ensure_ascii=False)
    return excerpt(raw, ANSWER_CHARS)


def judge_prompt_v3(step_goal: str, tool: str, http_status: Optional[int], body: Optional[str],
                    error_text: Optional[str]) -> str:
    return (
        "You check an AI agent's tool result. Does the result actually satisfy the step?\n\n"
        f"Step: {step_goal}\nTool: {tool}\nHTTP status: {http_status}\n"
        f"Answer field ({answer_field_label(tool)}): {answer_field_text(tool, body)}\n"
        f"Raw result (first {BODY_CHARS} characters): {excerpt(body, BODY_CHARS)}\n"
        f"Error: {excerpt(error_text, 200)}\n\n"
        'Respond with ONLY a JSON object {"satisfies": true|false, "reason": "<one line>"}.'
    )


@dataclass
class VerifyExecutorV3(VerifyExecutor):
    definition_version: str = "v3"

    def config(self) -> dict:
        return {**super().config(), "judge_input": JUDGE_INPUT}

    def _judge_prompt(self, ctx, result: ToolCallResult) -> str:
        body = result.full_text if result.full_text is not None else result.body
        return judge_prompt_v3(ctx.step_goal, ctx.tool_name, result.http_status, body, result.error_text)
