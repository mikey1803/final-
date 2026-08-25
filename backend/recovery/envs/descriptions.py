"""Single source of truth for the plain-English failure-type and
recovery-strategy descriptions shown to blind-design study participants
(Task 11).

Every surface that shows a human or an LLM designer what a "failure
type" or "recovery strategy" IS -- the study web app (via the FastAPI
study service), the LLM-designer script (`experiments/llm_designers.py`),
and any future paper table generator -- imports the descriptions from
THIS module and nowhere else. Three independent copies of this text is
exactly how a blind protocol silently stops being blind (a wording drift
between the web app and the LLM prompt would make the two arms
non-comparable, and any copy that quietly grew a hint about the reward
structure would break blindness without anyone noticing). See
`study/protocol.md` and `study/blindness_audit.md`.

Hard rule: this module imports ONLY the two ORDERED NAME LISTS from
`recovery.envs.canonical` (`TYPE_NAMES`, `STRATEGY_NAMES`) -- never
`MATRIX`, never `BLIND_RULE`. Importing the name lists guarantees this
module's ordering can never silently drift from the matrix's row/column
order (the ordering is exactly what makes a participant's ``mapping``
list scorable against ``MATRIX`` later); importing anything else would
put reward information one accidental re-export away from the
request-serving path this module is designed to sit in.
"""

from typing import NamedTuple, Optional, Sequence

from recovery.envs.canonical import STRATEGY_NAMES, TYPE_NAMES

__all__ = [
    "TYPE_NAMES",
    "STRATEGY_NAMES",
    "TYPE_DESCRIPTIONS",
    "STRATEGY_DESCRIPTIONS",
    "STRATEGY_COSTS",
    "FailureTypeEntry",
    "StrategyEntry",
    "type_table",
    "strategy_table",
    "build_task_instructions",
]


class FailureTypeEntry(NamedTuple):
    name: str
    label: str
    description: str


class StrategyEntry(NamedTuple):
    name: str
    label: str
    description: str
    cost: float


# Ordered EXACTLY as recovery.envs.canonical.TYPE_NAMES (index-aligned with
# MATRIX's rows). Wording matches the plain-English descriptions already
# shown in the project's public demo (frontend/dashboard-react/src/data/
# failureTypes.js), reproduced here as the canonical Python source rather
# than copy-pasted a second time -- the demo dashboard is a separate,
# non-study surface and is left untouched (see study/protocol.md,
# "Deviations from a literal reading of the task").
_TYPE_DESCRIPTIONS_BY_NAME: dict[str, str] = {
    "tool_timeout": (
        "The agent calls a tool and just... never hears back. It hangs, "
        "waiting for a response that never comes."
    ),
    "tool_error": (
        "The tool runs, but comes back with an error instead of a result "
        "-- the request itself was rejected."
    ),
    "wrong_tool_selected": (
        "The agent reaches for the wrong tool entirely -- like asking a "
        "calculator to check the weather."
    ),
    "hallucinated_output": (
        "The agent confidently states something that simply isn't true, "
        "with no indication it's unsure."
    ),
    "malformed_output": (
        "The agent's response breaks the expected format, so nothing "
        "downstream can actually parse it."
    ),
    "context_loss": (
        "The agent forgets something important it was told earlier in "
        "the conversation."
    ),
    "ambiguous_instruction": (
        "The instruction could reasonably mean two different things, and "
        "the agent guesses -- wrong."
    ),
}

_TYPE_LABELS_BY_NAME: dict[str, str] = {
    "tool_timeout": "Timeout",
    "tool_error": "Tool Error",
    "wrong_tool_selected": "Wrong Tool",
    "hallucinated_output": "Hallucination",
    "malformed_output": "Malformed Output",
    "context_loss": "Context Loss",
    "ambiguous_instruction": "Ambiguous Instruction",
}

# Ordered EXACTLY as recovery.envs.canonical.STRATEGY_NAMES (index-aligned
# with MATRIX's columns). Costs mirror backend/src/recovery/strategies.py's
# per-action `cost` class attributes exactly (RetryAction=1.0,
# ReplanAction=3.0, SwitchToolAction=2.0, ReGroundAction=2.5,
# VerifyAction=1.5, ClarifyAction=2.0) -- reused as literal numbers rather
# than imported, since `src.recovery.strategies` pulls in the full legacy
# action-execution machinery (ABCs, RNG-driven `_simulate_outcome`, etc.)
# that this pure-description module has no reason to depend on; the values
# are covered by tests/test_study.py's cross-check.
_STRATEGY_DESCRIPTIONS_BY_NAME: dict[str, str] = {
    "retry": "Try the exact same action again.",
    "replan": "Scrap the plan, start fresh.",
    "switch_tool": "Use a different tool for the job.",
    "re_ground": "Re-fetch context, re-anchor state.",
    "verify": "Double-check the last output.",
    "clarify": "Ask a clarifying question.",
}

_STRATEGY_LABELS_BY_NAME: dict[str, str] = {
    "retry": "Retry",
    "replan": "Replan",
    "switch_tool": "Switch Tool",
    "re_ground": "Re-ground",
    "verify": "Verify",
    "clarify": "Clarify",
}

_STRATEGY_COSTS_BY_NAME: dict[str, float] = {
    "retry": 1.0,
    "replan": 3.0,
    "switch_tool": 2.0,
    "re_ground": 2.5,
    "verify": 1.5,
    "clarify": 2.0,
}

TYPE_DESCRIPTIONS: tuple[FailureTypeEntry, ...] = tuple(
    FailureTypeEntry(name=n, label=_TYPE_LABELS_BY_NAME[n], description=_TYPE_DESCRIPTIONS_BY_NAME[n])
    for n in TYPE_NAMES
)

STRATEGY_DESCRIPTIONS: tuple[StrategyEntry, ...] = tuple(
    StrategyEntry(
        name=n,
        label=_STRATEGY_LABELS_BY_NAME[n],
        description=_STRATEGY_DESCRIPTIONS_BY_NAME[n],
        cost=_STRATEGY_COSTS_BY_NAME[n],
    )
    for n in STRATEGY_NAMES
)

STRATEGY_COSTS: tuple[float, ...] = tuple(e.cost for e in STRATEGY_DESCRIPTIONS)


def type_table() -> list[dict]:
    """JSON-serializable Table I: one dict per failure type, in
    ``TYPE_NAMES`` order. Contains ONLY name/label/description -- no
    reward information exists to leak here."""
    return [e._asdict() for e in TYPE_DESCRIPTIONS]


def strategy_table() -> list[dict]:
    """JSON-serializable Table II: one dict per strategy, in
    ``STRATEGY_NAMES`` order, including its (publicly-known, non-secret)
    cost. Costs are shown to participants deliberately -- the study
    measures whether a designer picks a good ACTION given costs and
    descriptions, not whether they can guess costs are hidden too."""
    return [e._asdict() for e in STRATEGY_DESCRIPTIONS]


def build_task_instructions(
    type_order: Optional[Sequence[int]] = None,
    strategy_order: Optional[Sequence[int]] = None,
) -> str:
    """The exact plain-English task text shown to BOTH a human participant
    (rendered by the study frontend's Task screen) and an LLM designer
    (embedded verbatim in ``experiments/llm_designers.py``'s prompt).

    Returned as one canonical string so both surfaces are constructed
    from -- not merely inspired by -- the same source
    (``tests/test_study.py`` asserts the LLM prompt contains this string
    verbatim, not a paraphrase).

    Parameters
    ----------
    type_order, strategy_order : sequence of int, optional
        A permutation of ``range(len(TYPE_DESCRIPTIONS))`` /
        ``range(len(STRATEGY_DESCRIPTIONS))`` controlling the order the
        failure types / strategies are LISTED in (content is identical
        either way -- only presentation order changes). Defaults (``None``)
        to natural (``TYPE_NAMES``/``STRATEGY_NAMES``) order, which makes
        this 100% backward compatible with every existing caller
        (``build_task_instructions()`` with no arguments is byte-identical
        to before this parameter was added).

        Added for Task 15's presentation-order-randomisation supplementary
        experiment (``experiments/prompt_robustness.py``), which reuses
        ``recovery.envs.display_order`` -- the SAME deterministic
        randomiser the human study instrument uses -- seeded off LLM
        sample index instead of a participant token.
    """
    ordered_types = TYPE_DESCRIPTIONS if type_order is None else [TYPE_DESCRIPTIONS[i] for i in type_order]
    ordered_strategies = (
        STRATEGY_DESCRIPTIONS if strategy_order is None else [STRATEGY_DESCRIPTIONS[i] for i in strategy_order]
    )
    lines = [
        "You are designing a fixed lookup table for an AI agent's failure-recovery system.",
        "",
        "There are 7 kinds of failure the agent can encounter, and 6 recovery strategies "
        "it can try. For EACH failure type, choose EXACTLY ONE recovery strategy that you "
        "believe gives the agent the best chance of recovering successfully.",
        "",
        "You will not be told how well any strategy actually performs. Choose based only "
        "on the descriptions below and your own judgement.",
        "",
        "Failure types:",
    ]
    for e in ordered_types:
        lines.append(f"  - {e.label} ({e.name}): {e.description}")
    lines.append("")
    lines.append("Recovery strategies (with their operational cost):")
    for e in ordered_strategies:
        lines.append(f"  - {e.label} ({e.name}, cost={e.cost}): {e.description}")
    lines.append("")
    lines.append(
        "For each of the 7 failure types, choose exactly one strategy. Ties in expected "
        "benefit should be broken in favor of lower cost."
    )
    return "\n".join(lines)
