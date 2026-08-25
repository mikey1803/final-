"""Select which live-pilot events may feed a confusion-matrix computation
-- Campaign A (targeted) events ONLY, the mirror image of
``recovery.live.deployment_distribution``'s Campaign-B-only rule for p.
Together the two modules are the "never pooled" guard: each refuses the
other campaign's events by construction, not by caller discipline.

OPEN QUESTION, flagged rather than silently resolved: ``tool_timeout``/
``tool_error`` can never occur on a targeted task -- ``Task.__post_init__``
forbids targeting either, since they only occur via ``FlakyWrapper``
injection on untargeted/Campaign-B tasks. A confusion matrix sourced from
Campaign A alone therefore has zero possible samples for those two rows.
Whether Campaign B's tool_timeout/tool_error events are a legitimate
confusion-matrix input for just those two rows, or those two types need
their own resolution, is a real design fork this module does not decide --
it implements the literal rule as specified (Campaign A only) and leaves
the gap visible rather than papering over it.
"""

from __future__ import annotations

from recovery.live.event_schema import FailureEvent, Task
from recovery.live.event_store import EventStore

__all__ = ["ConfusionMatrixSourceError", "collect_confusion_matrix_events"]


class ConfusionMatrixSourceError(ValueError):
    """Raised when there's nothing legitimate to build a confusion matrix from."""


def collect_confusion_matrix_events(tasks: list[Task], store: EventStore) -> list[FailureEvent]:
    """Every labelled event produced by a targeted=True (Campaign A) task.
    Raises ConfusionMatrixSourceError if given no targeted tasks, or if
    zero labelled targeted-task events exist yet -- never silently returns
    an empty/fabricated matrix input.
    """
    targeted_task_ids = {t.task_id for t in tasks if t.targeted}
    if not targeted_task_ids:
        raise ConfusionMatrixSourceError(
            "no targeted=True tasks in the given task list -- confusion-matrix events "
            "must come from Campaign A (targeted, balanced) tasks only, never Campaign B"
        )

    events = [
        ev for ev in store.iter_events()
        if ev.true_type is not None and ev.task_id in targeted_task_ids
    ]
    if not events:
        raise ConfusionMatrixSourceError(
            "no labelled events from targeted tasks yet -- nothing to build a confusion matrix from"
        )
    return events
