"""Recovery strategies for the failure-aware agent.

Each strategy is a concrete ``RecoveryAction`` subclass that takes a
``FailureEvent`` and a ``step_context`` dict, simulates attempting the
recovery action, and returns a dict indicating what was done and whether
it succeeded.

The success probability is **not** uniform — it depends on the
combination of ``FailureType`` and ``RecoveryStrategy``.  A hidden
ground-truth success-rate matrix (``SUCCESS_MATRIX``) encodes these
dependencies.  The whole purpose of the adaptive selector (see
``selector.py``) is to learn this matrix through experience without
being told its values directly.
"""

import random
from abc import ABC, abstractmethod
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
# Abstract base class
# ======================================================================

class RecoveryAction(ABC):
    """Abstract base for all recovery actions.

    Each subclass implements ``execute()`` and carries a ``cost``
    attribute representing the relative expense of that recovery action.
    """

    cost: float = 1.0

    @abstractmethod
    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        """Execute the recovery action and return outcome dict.

        Args:
            event: The failure event to recover from.
            step_context: The agent's state when the failure happened.
            rng: Optional RNG for reproducibility.

        Returns:
            A dict with keys ``strategy``, ``action``, ``success``, ``cost``.
        """


# ======================================================================
# Per-strategy concrete classes
# ======================================================================

_rng = random.Random()


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


class RetryAction(RecoveryAction):
    """Re-attempt the exact same action that just failed.

    Simple but ineffective when the failure is systematic (e.g. a
    hallucinated output will simply be hallucinated again).
    """

    cost = 1.0

    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        r = rng or _rng
        tool = step_context.get("tool_name", "unknown")
        outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.RETRY, cost=self.cost, rng=r)
        return {
            "strategy": RecoveryStrategy.RETRY.value,
            "action": f"Re-executing '{tool}' with identical parameters.",
            **outcome,
        }


class ReplanAction(RecoveryAction):
    """Discard the current plan and generate a fresh approach from scratch."""

    cost = 3.0

    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        r = rng or _rng
        outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.REPLAN, cost=self.cost, rng=r)
        return {
            "strategy": RecoveryStrategy.REPLAN.value,
            "action": "Abandoned previous plan. Generating new plan from the original goal.",
            **outcome,
        }


class SwitchToolAction(RecoveryAction):
    """Pick a different tool to accomplish the same sub-task."""

    cost = 2.0

    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        r = rng or _rng
        current = step_context.get("tool_name", "tool_a")
        alternatives = [
            t for t in [
                "web_search", "calculator", "file_reader",
                "api_caller", "database_query", "code_executor",
            ] if t != current
        ]
        new_tool = r.choice(alternatives) if alternatives else "fallback_tool"
        outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.SWITCH_TOOL, cost=self.cost, rng=r)
        return {
            "strategy": RecoveryStrategy.SWITCH_TOOL.value,
            "action": f"Switched from '{current}' to '{new_tool}' for the same sub-task.",
            **outcome,
        }


class ReGroundAction(RecoveryAction):
    """Re-fetch context, re-read the original task, and re-anchor state."""

    cost = 2.5

    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        r = rng or _rng
        outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.RE_GROUND, cost=self.cost, rng=r)
        return {
            "strategy": RecoveryStrategy.RE_GROUND.value,
            "action": "Re-fetched original goal and current state. Re-anchored agent context.",
            **outcome,
        }


class VerifyAction(RecoveryAction):
    """Run a self-check / verification step on the most recent output."""

    cost = 1.5

    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        r = rng or _rng
        outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.VERIFY, cost=self.cost, rng=r)
        return {
            "strategy": RecoveryStrategy.VERIFY.value,
            "action": "Ran verification pass on last output. Checking for factual consistency and format validity.",
            **outcome,
        }


class ClarifyAction(RecoveryAction):
    """Request a clarifying question or more specific instruction."""

    cost = 2.0

    def execute(
        self,
        event: FailureEvent,
        step_context: dict,
        rng: Optional[random.Random] = None,
    ) -> dict:
        r = rng or _rng
        outcome = _simulate_outcome(event.failure_type, RecoveryStrategy.CLARIFY, cost=self.cost, rng=r)
        return {
            "strategy": RecoveryStrategy.CLARIFY.value,
            "action": "Issued clarification request to the user regarding ambiguous or incomplete input.",
            **outcome,
        }


# ======================================================================
# Strategy dispatch table (maps enum -> action instance)
# ======================================================================

_ACTION_INSTANCES: dict[RecoveryStrategy, RecoveryAction] = {
    RecoveryStrategy.RETRY:       RetryAction(),
    RecoveryStrategy.REPLAN:      ReplanAction(),
    RecoveryStrategy.SWITCH_TOOL: SwitchToolAction(),
    RecoveryStrategy.RE_GROUND:   ReGroundAction(),
    RecoveryStrategy.VERIFY:      VerifyAction(),
    RecoveryStrategy.CLARIFY:     ClarifyAction(),
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
    action = _ACTION_INSTANCES[strategy]
    return action.execute(event, step_context, rng=rng)


def get_action_cost(strategy: RecoveryStrategy) -> float:
    """Return the relative cost of the given recovery strategy."""
    return _ACTION_INSTANCES[strategy].cost


# Backward-compatible aliases for code that imports the functions directly.
execute_retry = RetryAction().execute
execute_replan = ReplanAction().execute
execute_switch_tool = SwitchToolAction().execute
execute_re_ground = ReGroundAction().execute
execute_verify = VerifyAction().execute
execute_clarify = ClarifyAction().execute

STRATEGY_EXECUTORS: dict[RecoveryStrategy, callable] = {
    RecoveryStrategy.RETRY:       execute_retry,
    RecoveryStrategy.REPLAN:      execute_replan,
    RecoveryStrategy.SWITCH_TOOL: execute_switch_tool,
    RecoveryStrategy.RE_GROUND:   execute_re_ground,
    RecoveryStrategy.VERIFY:      execute_verify,
    RecoveryStrategy.CLARIFY:     execute_clarify,
}
