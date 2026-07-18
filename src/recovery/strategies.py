"""Recovery strategies for the failure-aware agent.

Each strategy is a callable that takes a ``FailureEvent`` and a
``step_context`` dict, simulates attempting the recovery action, and
returns a dict indicating what was done and whether it succeeded.

The success probability is **not** uniform — it depends on the
combination of ``FailureType`` and ``RecoveryStrategy``.  A hidden
ground-truth success-rate matrix (``SUCCESS_MATRIX``) encodes these
dependencies.  The whole purpose of the adaptive selector (see
``selector.py``) is to learn this matrix through experience without
being told its values directly.
"""

import random
from enum import Enum
from typing import Optional

from src.failures.taxonomy import FailureEvent, FailureType


# ======================================================================
# Recovery strategy enumeration
# ======================================================================

class RecoveryStrategy(Enum):
    """The six recovery actions the agent can take after a failure."""

    RETRY = "retry"
    REPLAN = "replan"
    SWITCH_TOOL = "switch_tool"
    RE_GROUND = "re_ground"
    VERIFY = "verify"
    CLARIFY = "clarify"


# ======================================================================
# Hidden ground-truth success-rate matrix
#
# SUCCESS_MATRIX[failure_type][strategy] = probability of success (0.0–1.0)
#
# These values are the "reality" that the adaptive selector will learn
# about purely through trial-and-error.  They are *not* exposed to the
# selector at any point.
# ======================================================================

SUCCESS_MATRIX: dict[FailureType, dict[RecoveryStrategy, float]] = {
    FailureType.TOOL_TIMEOUT: {
        RecoveryStrategy.RETRY:       0.70,
        RecoveryStrategy.REPLAN:      0.50,
        RecoveryStrategy.SWITCH_TOOL: 0.60,
        RecoveryStrategy.RE_GROUND:   0.15,
        RecoveryStrategy.VERIFY:      0.30,
        RecoveryStrategy.CLARIFY:     0.10,
    },
    FailureType.TOOL_ERROR: {
        RecoveryStrategy.RETRY:       0.65,
        RecoveryStrategy.REPLAN:      0.55,
        RecoveryStrategy.SWITCH_TOOL: 0.75,
        RecoveryStrategy.RE_GROUND:   0.20,
        RecoveryStrategy.VERIFY:      0.35,
        RecoveryStrategy.CLARIFY:     0.25,
    },
    FailureType.WRONG_TOOL_SELECTED: {
        RecoveryStrategy.RETRY:       0.15,
        RecoveryStrategy.REPLAN:      0.60,
        RecoveryStrategy.SWITCH_TOOL: 0.75,
        RecoveryStrategy.RE_GROUND:   0.45,
        RecoveryStrategy.VERIFY:      0.40,
        RecoveryStrategy.CLARIFY:     0.55,
    },
    FailureType.HALLUCINATED_OUTPUT: {
        RecoveryStrategy.RETRY:       0.10,
        RecoveryStrategy.REPLAN:      0.45,
        RecoveryStrategy.SWITCH_TOOL: 0.30,
        RecoveryStrategy.RE_GROUND:   0.55,
        RecoveryStrategy.VERIFY:      0.75,
        RecoveryStrategy.CLARIFY:     0.50,
    },
    FailureType.MALFORMED_OUTPUT: {
        RecoveryStrategy.RETRY:       0.40,
        RecoveryStrategy.REPLAN:      0.50,
        RecoveryStrategy.SWITCH_TOOL: 0.60,
        RecoveryStrategy.RE_GROUND:   0.35,
        RecoveryStrategy.VERIFY:      0.75,
        RecoveryStrategy.CLARIFY:     0.30,
    },
    FailureType.CONTEXT_LOSS: {
        RecoveryStrategy.RETRY:       0.10,
        RecoveryStrategy.REPLAN:      0.55,
        RecoveryStrategy.SWITCH_TOOL: 0.25,
        RecoveryStrategy.RE_GROUND:   0.80,
        RecoveryStrategy.VERIFY:      0.40,
        RecoveryStrategy.CLARIFY:     0.50,
    },
    FailureType.AMBIGUOUS_INSTRUCTION: {
        RecoveryStrategy.RETRY:       0.10,
        RecoveryStrategy.REPLAN:      0.45,
        RecoveryStrategy.SWITCH_TOOL: 0.20,
        RecoveryStrategy.RE_GROUND:   0.35,
        RecoveryStrategy.VERIFY:      0.30,
        RecoveryStrategy.CLARIFY:     0.80,
    },
}


# ======================================================================
# Per-strategy execution functions
#
# Each function simulates what happens when the agent applies a
# particular recovery strategy.  The outcome is drawn from the
# ground-truth matrix above — the function does not *know* the matrix;
# it merely samples from it.
# ======================================================================

_rng = random.Random()  # fallback only; prefer passing an explicit rng


def _simulate_outcome(
    failure_type: FailureType,
    strategy: RecoveryStrategy,
    cost: float = 1.0,
    rng: Optional[random.Random] = None,
) -> dict:
    """Sample a success/failure outcome from the hidden ground truth."""
    r = rng or _rng
    p = SUCCESS_MATRIX[failure_type][strategy]
    success = r.random() < p
    return {"success": success, "cost": cost}


def execute_retry(event: FailureEvent, step_context: dict, rng: Optional[random.Random] = None) -> dict:
    """Re-attempt the exact same action that just failed.

    Simple but ineffective when the failure is systematic (e.g. a
    hallucinated output will simply be hallucinated again).
    """
    r = rng or _rng
    tool = step_context.get("tool_name", "unknown")
    outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.RETRY, rng=r)
    return {
        "strategy": RecoveryStrategy.RETRY.value,
        "action": f"Re-executing '{tool}' with identical parameters.",
        **outcome,
    }


def execute_replan(event: FailureEvent, step_context: dict, rng: Optional[random.Random] = None) -> dict:
    """Discard the current plan and generate a fresh approach from scratch."""
    r = rng or _rng
    outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.REPLAN, rng=r)
    return {
        "strategy": RecoveryStrategy.REPLAN.value,
        "action": "Abandoned previous plan. Generating new plan from the original goal.",
        **outcome,
    }


def execute_switch_tool(event: FailureEvent, step_context: dict, rng: Optional[random.Random] = None) -> dict:
    """Pick a different tool to accomplish the same sub-task."""
    r = rng or _rng
    current = step_context.get("tool_name", "tool_a")
    alternatives = [t for t in ["web_search", "calculator", "file_reader", "api_caller", "database_query", "code_executor"] if t != current]
    new_tool = r.choice(alternatives) if alternatives else "fallback_tool"
    outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.SWITCH_TOOL, rng=r)
    return {
        "strategy": RecoveryStrategy.SWITCH_TOOL.value,
        "action": f"Switched from '{current}' to '{new_tool}' for the same sub-task.",
        **outcome,
    }


def execute_re_ground(event: FailureEvent, step_context: dict, rng: Optional[random.Random] = None) -> dict:
    """Re-fetch context, re-read the original task, and re-anchor state."""
    r = rng or _rng
    outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.RE_GROUND, rng=r)
    return {
        "strategy": RecoveryStrategy.RE_GROUND.value,
        "action": "Re-fetched original goal and current state. Re-anchored agent context.",
        **outcome,
    }


def execute_verify(event: FailureEvent, step_context: dict, rng: Optional[random.Random] = None) -> dict:
    """Run a self-check / verification step on the most recent output."""
    r = rng or _rng
    outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.VERIFY, rng=r)
    return {
        "strategy": RecoveryStrategy.VERIFY.value,
        "action": "Ran verification pass on last output. Checking for factual consistency and format validity.",
        **outcome,
    }


def execute_clarify(event: FailureEvent, step_context: dict, rng: Optional[random.Random] = None) -> dict:
    """Request a clarifying question or more specific instruction."""
    r = rng or _rng
    outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.CLARIFY, rng=r)
    return {
        "strategy": RecoveryStrategy.CLARIFY.value,
        "action": "Issued clarification request to the user regarding ambiguous or incomplete input.",
        **outcome,
    }


# ======================================================================
# Strategy dispatch table
# ======================================================================

STRATEGY_EXECUTORS: dict[RecoveryStrategy, callable] = {
    RecoveryStrategy.RETRY:       execute_retry,
    RecoveryStrategy.REPLAN:      execute_replan,
    RecoveryStrategy.SWITCH_TOOL: execute_switch_tool,
    RecoveryStrategy.RE_GROUND:   execute_re_ground,
    RecoveryStrategy.VERIFY:      execute_verify,
    RecoveryStrategy.CLARIFY:     execute_clarify,
}


def execute_strategy(
    strategy: RecoveryStrategy,
    event: FailureEvent,
    step_context: dict,
    rng: Optional[random.Random] = None,
) -> dict:
    """Execute a recovery strategy and return the outcome dict.

    This is the main entry point — callers select a strategy (via the
    adaptive selector) and then call this function to simulate the
    result.
    """
    executor = STRATEGY_EXECUTORS[strategy]
    return executor(event, step_context, rng=rng)
