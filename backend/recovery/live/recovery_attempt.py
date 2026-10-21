"""Inline recovery records (Task 21; strategy_definitions_v2.md §1, §5).

A ``RecoveryAttempt`` is one strategy applied INLINE, during a live run,
immediately after a captured failure and before any label exists.

**Never pooled with Phase 2 data.** ``ReplayResult`` (event_schema.py, frozen)
is a replay against an already-labelled event under randomised arm order;
RecoveryAttempts come from a different selection process, against the live
world at failure time, and are stored separately (``<store>/recoveries/``).
They must never be counted toward Phase 2 ``(type, strategy)`` cells.

Stores written before Task 21 have no ``recoveries/`` directory and load
exactly as before; ``FailureEvent`` itself is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from recovery.live.event_schema import ALL_ARMS

__all__ = ["RecoveryAttempt", "RecoveryStore", "RECOVERY_SCHEMA_VERSION"]

RECOVERY_SCHEMA_VERSION = "1"


@dataclass
class RecoveryAttempt:
    event_id: str
    session_id: str
    task_id: Optional[str]
    strategy: str
    definition_version: str
    selection_policy: str
    selection_propensity: float
    seed: dict
    model_id: Optional[str]
    executor_config: dict
    budget_limits: dict
    budget_used: dict
    budget_exceeded: bool
    result: dict           # RecoveryResult.to_dict(): success, cost_tokens, latency_ms, attempts, evidence
    started_at: str
    finished_at: str
    aborted: bool = False
    no_action_taken: bool = False
    context: str = "inline"
    schema_version: str = RECOVERY_SCHEMA_VERSION
    attempt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.strategy not in ALL_ARMS:
            raise ValueError(f"strategy must be one of {ALL_ARMS}, got {self.strategy!r}")
        if self.definition_version not in ("v1", "v2"):
            raise ValueError(f"definition_version must be 'v1' or 'v2', got {self.definition_version!r}")
        if not 0.0 < self.selection_propensity <= 1.0:
            raise ValueError(f"selection_propensity must be in (0, 1], got {self.selection_propensity!r}")
        self.attempt_id = self._compute_id()

    def _compute_id(self) -> str:
        payload = {"event_id": self.event_id, "strategy": self.strategy,
                   "definition_version": self.definition_version, "started_at": self.started_at}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RecoveryAttempt":
        d = dict(d)
        stored_id = d.pop("attempt_id", None)
        obj = cls(**d)
        if stored_id is not None and stored_id != obj.attempt_id:
            raise ValueError(f"stored attempt_id {stored_id!r} != recomputed {obj.attempt_id!r}")
        return obj


class RecoveryStore:
    """One JSON file per attempt under ``root/recoveries/`` (same root as the
    EventStore), atomic tmp-then-rename like ``EventStore.write``."""

    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / "recoveries"

    def write(self, attempt: RecoveryAttempt) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{attempt.attempt_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(attempt.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def iter_attempts(self) -> Iterator[RecoveryAttempt]:
        if not self.dir.exists():
            return
        for p in sorted(self.dir.glob("*.json")):
            yield RecoveryAttempt.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def for_event(self, event_id: str) -> list[RecoveryAttempt]:
        return [a for a in self.iter_attempts() if a.event_id == event_id]
