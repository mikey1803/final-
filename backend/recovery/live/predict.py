"""Batch classifier-prediction pass over already-labelled events.

Structurally separate from labelling: this only ever reads events that
already have ``true_type`` set (``EventStore.labelled()``), and
``FailureEvent.validate_label_prediction_order`` (enforced by
``EventStore.write``) refuses to persist a predicted-but-unlabelled event
regardless of how it got constructed. There is no code path in this module
that can compute a prediction before a human label exists for that event.

Not wired into any collection/labelling command -- run explicitly, and only
after labelling is done for the events being predicted.
"""

from __future__ import annotations

from typing import Callable, Optional

from recovery.live.event_schema import FailureEvent
from recovery.live.event_store import EventStore

__all__ = ["predict_labelled_events"]

# (predicted_type: str, predicted_proba: dict[str, float])
ClassifyFn = Callable[[str], tuple[str, dict]]


def predict_labelled_events(
    store: EventStore, classify_fn: ClassifyFn, overwrite: bool = False
) -> int:
    """Run ``classify_fn(raw_error)`` over every labelled event and store
    the result. Skips events that already have a prediction unless
    ``overwrite=True``. Returns the number of events predicted.

    Only ever iterates ``store.labelled()`` -- an unlabelled event is never
    even visited, let alone predicted.
    """
    n_predicted = 0
    for event in store.labelled():
        if event.predicted_type is not None and not overwrite:
            continue
        predicted_type, predicted_proba = classify_fn(event.raw_error)
        event.predicted_type = predicted_type
        event.predicted_proba = predicted_proba
        store.write(event)
        n_predicted += 1
    return n_predicted
