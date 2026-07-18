"""Failure type definitions and event recording for the failure-aware agent.

Provides a FailureType enum that enumerates every recognisable failure mode
the agent can encounter, and a FailureEvent dataclass that captures the full
context of a single failure occurrence for downstream classification and
recovery.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class FailureType(Enum):
    """Enumeration of recognised agent failure modes.

    Each value corresponds to a distinct category of failure that the agent's
    pipeline can encounter, from tool-level errors through to reasoning-level
    mistakes such as hallucination or context loss.
    """

    TOOL_TIMEOUT = "tool_timeout"
    TOOL_ERROR = "tool_error"
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    HALLUCINATED_OUTPUT = "hallucinated_output"
    MALFORMED_OUTPUT = "malformed_output"
    CONTEXT_LOSS = "context_loss"
    AMBIGUOUS_INSTRUCTION = "ambiguous_instruction"


@dataclass
class FailureEvent:
    """Records every detail of a single failure occurrence.

    Attributes:
        failure_type: The category of failure from the FailureType taxonomy.
        timestamp: UTC timestamp of when the failure was recorded.
        step_context: A dictionary describing the agent's state when the
            failure happened — typically includes fields like ``step_index``,
            ``goal``, ``tool_name``, ``tool_args``, and any other relevant
            runtime metadata.
        raw_error: The actual error message, malformed output, or fabricated
            text that constitutes the failure.
    """

    failure_type: FailureType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_context: dict = field(default_factory=dict)
    raw_error: str = ""

    def to_dict(self) -> dict:
        """Serialise the event to a plain dictionary for storage or logging."""
        return {
            "failure_type": self.failure_type.value,
            "timestamp": self.timestamp.isoformat(),
            "step_context": self.step_context,
            "raw_error": self.raw_error,
        }
