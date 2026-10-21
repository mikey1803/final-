"""Real recovery executors for the live loop (Task 21).

Definitions: ``docs/live_pilot/strategy_definitions_v2.md`` (FROZEN v2) and,
for each strategy's v1 configuration, ``docs/live_pilot/strategy_definitions.md``
(FROZEN v1). ``build_executors(version)`` returns one executor per real
strategy, configured for that definition version.

Inline RecoveryAttempt records produced with these executors are never pooled
with Phase 2 ReplayResult data (v2 §1) -- see ``recovery.live.recovery_attempt``.
"""

from __future__ import annotations

from typing import Optional

from recovery.live.event_schema import STRATEGY_NAMES
from recovery.live.executors.base import Budget, FailureContext, RecoveryExecutor, RecoveryResult
from recovery.live.executors.clarify import ClarifyExecutor
from recovery.live.executors.null_control import NullControlExecutor
from recovery.live.executors.re_ground import ReGroundExecutor
from recovery.live.executors.replan import ReplanExecutor
from recovery.live.executors.retry import RetryExecutor
from recovery.live.executors.switch_tool import SwitchToolExecutor
from recovery.live.executors.verify import VerifyExecutor

__all__ = [
    "Budget", "FailureContext", "RecoveryExecutor", "RecoveryResult", "build_executors",
    "RetryExecutor", "SwitchToolExecutor", "VerifyExecutor", "ReGroundExecutor", "ReplanExecutor",
    "ClarifyExecutor", "NullControlExecutor",
]

_CLASSES = {
    "retry": RetryExecutor, "replan": ReplanExecutor, "switch_tool": SwitchToolExecutor,
    "re_ground": ReGroundExecutor, "verify": VerifyExecutor, "clarify": ClarifyExecutor,
}


def build_executors(version: str = "v2", null_control_wait_s: Optional[float] = None) -> dict[str, RecoveryExecutor]:
    if version not in ("v1", "v2"):
        raise ValueError(f"definition version must be 'v1' or 'v2', got {version!r}")
    execs = {name: (cls.v1() if version == "v1" else cls()) for name, cls in _CLASSES.items()}
    if list(execs) != STRATEGY_NAMES:  # same canonical order as event_schema.STRATEGY_NAMES
        raise RuntimeError(f"executor set {list(execs)} does not match STRATEGY_NAMES {STRATEGY_NAMES}")
    if null_control_wait_s is not None:
        execs["null_control"] = NullControlExecutor(wait_s=null_control_wait_s, definition_version=version)
    return execs
