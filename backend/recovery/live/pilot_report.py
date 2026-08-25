"""Pilot-batch tooling: build an empty task-manifest skeleton (task text is
authored by hand, never generated here) and, after a pilot run, report
per-type yield rate and how many more attempts would be needed to reach
the confusion-matrix sample floor.

Yield rate is defined per intended_failure_type as
    n_observed / n_attempts
where, for the five agent-side types, n_attempts counts only attempts of
tasks explicitly targeting that type (Task.targeted=True), and for
tool_timeout/tool_error (never task-targetable -- see
recovery.live.event_schema.NOT_TASK_TARGETABLE_TYPES) n_attempts counts
attempts of ALL untargeted tasks combined, since those two types emerge
from FlakyWrapper injection on any ordinary task, not from any one task's
design.
"""

from __future__ import annotations

import math
from typing import Optional

from recovery.live.event_schema import (
    AGENT_SIDE_TARGETABLE_TYPES, NOT_TASK_TARGETABLE_TYPES, Task,
)
from recovery.live.event_store import EventStore

__all__ = ["build_pilot_skeleton", "compute_yield_report"]


def build_pilot_skeleton(per_type: int = 4, n_natural: int = 8) -> list[Task]:
    """Empty pilot-manifest skeleton: ``per_type`` placeholder tasks for
    each of the five agent-side types (targeted=True) plus ``n_natural``
    ordinary tasks (targeted=False) that exercise real tool calls under
    FlakyWrapper injection, exactly as the pilot design specifies. ``goal``
    is a TODO placeholder -- task text is authored by hand, never
    generated here.
    """
    tasks: list[Task] = []
    for ftype in sorted(AGENT_SIDE_TARGETABLE_TYPES):
        for i in range(1, per_type + 1):
            tasks.append(
                Task(
                    task_id=f"pilot_{ftype}_{i:02d}",
                    goal=f"TODO: write task text designed to elicit {ftype!r}",
                    category=ftype,
                    intended_failure_type=ftype,
                    targeted=True,
                )
            )
    for i in range(1, n_natural + 1):
        tasks.append(
            Task(
                task_id=f"pilot_natural_{i:02d}",
                goal="TODO: write an ordinary task (real tool call, no steering toward any failure type)",
                category="natural",
                intended_failure_type=None,
                targeted=False,
            )
        )
    return tasks


def _yield_row(n_attempts: int, n_observed: int, target_low: int, target_high: int) -> dict:
    if n_attempts == 0:
        return {
            "n_attempts": 0, "n_observed": n_observed, "yield_rate": None,
            "attempts_needed_for_low": None, "attempts_needed_for_high": None,
        }
    yield_rate = n_observed / n_attempts
    if yield_rate == 0.0:
        needed_low = needed_high = None  # cannot project attempts needed from zero observed yield
    else:
        needed_low = math.ceil(target_low / yield_rate)
        needed_high = math.ceil(target_high / yield_rate)
    return {
        "n_attempts": n_attempts, "n_observed": n_observed, "yield_rate": yield_rate,
        "attempts_needed_for_low": needed_low, "attempts_needed_for_high": needed_high,
    }


def compute_yield_report(
    tasks: list[Task],
    store: EventStore,
    attempts_by_task_id: dict[str, int],
    target_n_low: int = 30,
    target_n_high: int = 45,
) -> dict[str, dict]:
    """Per FailureType: attempts made, labelled events observed of that
    exact type, resulting yield rate, and how many total attempts (at the
    same per-attempt yield) would be needed to reach ``target_n_low``/
    ``target_n_high`` labelled events -- None where not yet computable
    (zero attempts, or zero observed yield) rather than a fabricated
    number. Only labelled events count as observed: an unlabelled
    failure's type isn't known yet.
    """
    targeted_task_ids_by_type: dict[str, set[str]] = {t: set() for t in AGENT_SIDE_TARGETABLE_TYPES}
    natural_task_ids: set[str] = set()
    for t in tasks:
        if t.targeted:
            targeted_task_ids_by_type[t.intended_failure_type].add(t.task_id)
        else:
            natural_task_ids.add(t.task_id)

    events_by_task_and_type: dict[tuple, int] = {}
    for event in store.iter_events():
        if event.true_type is None or event.task_id is None:
            continue
        key = (event.task_id, event.true_type)
        events_by_task_and_type[key] = events_by_task_and_type.get(key, 0) + 1

    report: dict[str, dict] = {}
    for ftype, task_ids in targeted_task_ids_by_type.items():
        n_attempts = sum(attempts_by_task_id.get(tid, 0) for tid in task_ids)
        n_observed = sum(
            count for (tid, ttype), count in events_by_task_and_type.items()
            if tid in task_ids and ttype == ftype
        )
        report[ftype] = _yield_row(n_attempts, n_observed, target_n_low, target_n_high)

    n_natural_attempts = sum(attempts_by_task_id.get(tid, 0) for tid in natural_task_ids)
    for ftype in NOT_TASK_TARGETABLE_TYPES:
        n_observed = sum(
            count for (tid, ttype), count in events_by_task_and_type.items()
            if tid in natural_task_ids and ttype == ftype
        )
        report[ftype] = _yield_row(n_natural_attempts, n_observed, target_n_low, target_n_high)

    return report
