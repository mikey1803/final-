"""Task 15: two semantically-identical paraphrases of
``recovery.envs.descriptions.build_task_instructions()``, used ONLY by the
LLM-designer prompt-robustness supplementary experiment
(``experiments/prompt_robustness.py``). Never imported by
``study/app/backend/`` and never shown to human participants -- these
exist purely to test whether the LLM arm's headline ``q`` is sensitive to
the instrument's surface wording, not to replace the primary instrument.

Hard rule, inherited unchanged from ``recovery/envs/descriptions.py``:
this module imports ONLY ``TYPE_NAMES``/``STRATEGY_NAMES``/``STRATEGY_COSTS``
(structural identifiers and publicly-shown costs) from
``recovery.envs.descriptions`` -- never
``recovery.envs.canonical.MATRIX``/``BLIND_RULE``. The text built here is
shown to blind LLM designers over the exact same channel as the primary
prompt, so it is bound by the identical blindness discipline, even though
(unlike ``descriptions.py``) it is never reachable from the study web
backend's request-serving path.

What is preserved EXACTLY across primary / paraphrase A / paraphrase B
--------------------------------------------------------------------------
- The 7 failure-type ``name``/``label`` identifiers and 6 strategy
  ``name``/``label`` identifiers (required for response parsing --
  ``experiments/llm_designers.py::parse_mapping_response`` matches on the
  literal ``name`` strings).
- The 6 strategy cost values (``STRATEGY_COSTS``, imported not
  retyped -- retyping them would risk a silent transcription drift).
- The task itself (choose exactly one strategy per failure type,
  maximizing recovery chance; tie-break toward lower cost) and the
  epistemic constraint (no performance feedback given).
- The response-format instructions (``experiments.llm_designers.RESPONSE_FORMAT_INSTRUCTIONS``,
  appended identically by ``build_prompt()`` regardless of variant --
  never paraphrased).

What DIFFERS
------------
Only prose: the framing sentences and the per-type/per-strategy
description wording. Paraphrase A uses a more casual, narrative register;
paraphrase B uses a more formal, enumerated register -- deliberately
distinct from each other as well as from the primary wording, so a
stability finding isn't an artifact of two paraphrases being suspiciously
similar to one another.
"""

from recovery.envs.descriptions import STRATEGY_COSTS, STRATEGY_NAMES, TYPE_NAMES

__all__ = [
    "build_task_instructions_paraphrase_a",
    "build_task_instructions_paraphrase_b",
    "PARAPHRASE_BUILDERS",
]

# ======================================================================
# Paraphrase A -- casual / narrative register
# ======================================================================

_TYPE_DESCRIPTIONS_A: dict[str, str] = {
    "tool_timeout": (
        "The agent invokes a tool and gets nothing back -- it just sits there waiting "
        "indefinitely for a reply that isn't coming."
    ),
    "tool_error": (
        "The tool does respond, but with a failure message rather than a real result "
        "-- the call itself got turned down."
    ),
    "wrong_tool_selected": (
        "The agent picks a completely inappropriate tool for the job -- the equivalent "
        "of using a stopwatch to measure distance."
    ),
    "hallucinated_output": (
        "The agent states something false with total confidence, giving no sign that it "
        "might be wrong."
    ),
    "malformed_output": (
        "What the agent returns doesn't match the format anything downstream expects, "
        "so it can't be processed."
    ),
    "context_loss": (
        "The agent loses track of something important that was mentioned earlier in "
        "the exchange."
    ),
    "ambiguous_instruction": (
        "The instruction is genuinely open to two different readings, and the agent "
        "picks the wrong one."
    ),
}

_TYPE_LABELS_A: dict[str, str] = {
    "tool_timeout": "Timeout",
    "tool_error": "Tool Error",
    "wrong_tool_selected": "Wrong Tool",
    "hallucinated_output": "Hallucination",
    "malformed_output": "Malformed Output",
    "context_loss": "Context Loss",
    "ambiguous_instruction": "Ambiguous Instruction",
}

_STRATEGY_DESCRIPTIONS_A: dict[str, str] = {
    "retry": "Attempt the identical action a second time.",
    "replan": "Throw out the current plan and build a new one from scratch.",
    "switch_tool": "Swap in a different tool to accomplish the same task.",
    "re_ground": "Pull fresh context and re-anchor the agent's state to it.",
    "verify": "Check the previous output over before moving on.",
    "clarify": "Ask a follow-up question to resolve the uncertainty.",
}

_STRATEGY_LABELS_A: dict[str, str] = {
    "retry": "Retry",
    "replan": "Replan",
    "switch_tool": "Switch Tool",
    "re_ground": "Re-ground",
    "verify": "Verify",
    "clarify": "Clarify",
}


def build_task_instructions_paraphrase_a() -> str:
    lines = [
        "Your task: build a static decision table that tells an autonomous agent which "
        "recovery action to take after each kind of failure.",
        "",
        "The agent can run into 7 distinct failure situations, and has 6 possible recovery "
        "actions available to it. For every one of the 7 failure situations, pick exactly "
        "one recovery action -- whichever one you judge most likely to let the agent "
        "recover successfully.",
        "",
        "No performance feedback will be given to you at any point. Base every choice "
        "purely on the descriptions provided and your own reasoning.",
        "",
        "The 7 failure situations:",
    ]
    for name in TYPE_NAMES:
        lines.append(f"  - {_TYPE_LABELS_A[name]} ({name}): {_TYPE_DESCRIPTIONS_A[name]}")
    lines.append("")
    lines.append("The 6 recovery actions, each with a listed cost:")
    for name, cost in zip(STRATEGY_NAMES, STRATEGY_COSTS):
        lines.append(f"  - {_STRATEGY_LABELS_A[name]} ({name}, cost={cost}): {_STRATEGY_DESCRIPTIONS_A[name]}")
    lines.append("")
    lines.append(
        "Assign one action to each of the 7 failure situations. When two actions seem "
        "equally likely to help, prefer whichever one costs less."
    )
    return "\n".join(lines)


# ======================================================================
# Paraphrase B -- formal / enumerated register
# ======================================================================

_TYPE_DESCRIPTIONS_B: dict[str, str] = {
    "tool_timeout": (
        "A tool invocation is issued and no response is ever received within the "
        "expected window; the call remains pending indefinitely."
    ),
    "tool_error": (
        "The invoked tool executes and returns, but the response consists of an error "
        "condition rather than a usable result."
    ),
    "wrong_tool_selected": (
        "An unsuitable tool is selected for the task at hand, one that is functionally "
        "mismatched to the requirement."
    ),
    "hallucinated_output": (
        "The system asserts a factually incorrect statement while presenting it with "
        "unwarranted certainty."
    ),
    "malformed_output": (
        "The produced output does not conform to the required structural format, "
        "rendering it unparseable by downstream components."
    ),
    "context_loss": (
        "Previously established information from earlier in the interaction is not "
        "retained and is effectively discarded."
    ),
    "ambiguous_instruction": (
        "The provided instruction admits multiple valid interpretations, and an "
        "incorrect interpretation is selected."
    ),
}

_TYPE_LABELS_B: dict[str, str] = {
    "tool_timeout": "Timeout",
    "tool_error": "Tool Error",
    "wrong_tool_selected": "Wrong Tool",
    "hallucinated_output": "Hallucination",
    "malformed_output": "Malformed Output",
    "context_loss": "Context Loss",
    "ambiguous_instruction": "Ambiguous Instruction",
}

_STRATEGY_DESCRIPTIONS_B: dict[str, str] = {
    "retry": "Re-execute the same action without modification.",
    "replan": "Discard the existing plan and generate a new one.",
    "switch_tool": "Substitute an alternative tool for the same subtask.",
    "re_ground": "Refresh the available context and re-establish the agent's state relative to it.",
    "verify": "Perform a validation pass on the most recent output.",
    "clarify": "Issue a clarifying question to the source of the instruction.",
}

_STRATEGY_LABELS_B: dict[str, str] = {
    "retry": "Retry",
    "replan": "Replan",
    "switch_tool": "Switch Tool",
    "re_ground": "Re-ground",
    "verify": "Verify",
    "clarify": "Clarify",
}


def build_task_instructions_paraphrase_b() -> str:
    lines = [
        "Task specification: construct a static mapping from failure category to "
        "recovery action for an autonomous agent's error-handling module.",
        "",
        "Two enumerations are defined below: seven (7) failure categories, and six (6) "
        "available recovery actions. Your assignment is to select, for every failure "
        "category, precisely one recovery action -- the one you assess as maximizing the "
        "probability of successful recovery.",
        "",
        "Note: no outcome feedback or performance data will be supplied. Base your "
        "assignment solely on the category/action descriptions given and your own "
        "judgement.",
        "",
        "Failure categories (7):",
    ]
    for name in TYPE_NAMES:
        lines.append(f"  - {_TYPE_LABELS_B[name]} ({name}): {_TYPE_DESCRIPTIONS_B[name]}")
    lines.append("")
    lines.append("Recovery actions (6), with associated operational cost:")
    for name, cost in zip(STRATEGY_NAMES, STRATEGY_COSTS):
        lines.append(f"  - {_STRATEGY_LABELS_B[name]} ({name}, cost={cost}): {_STRATEGY_DESCRIPTIONS_B[name]}")
    lines.append("")
    lines.append(
        "Provide exactly one recovery action per failure category. In cases of a tie in "
        "expected effectiveness, select the lower-cost action."
    )
    return "\n".join(lines)


PARAPHRASE_BUILDERS = {
    "paraphrase_a": build_task_instructions_paraphrase_a,
    "paraphrase_b": build_task_instructions_paraphrase_b,
}
