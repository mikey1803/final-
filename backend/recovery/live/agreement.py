"""Label-agreement statistics for the live-pilot event store.

Currently one check: Cohen's kappa between a labeller's original ``true_type``
and a later, blind second pass (``relabel_type``) over the same events, via
the labelling CLI's ``--relabel`` mode. Both labels come from the same
human by design (single-labeller pilot); this measures self-consistency,
not inter-rater agreement between two different people.
"""

from __future__ import annotations

from typing import Optional

from recovery.live.event_store import EventStore

__all__ = ["compute_relabel_kappa"]


def compute_relabel_kappa(store: EventStore) -> dict:
    """Cohen's kappa over every event with both ``true_type`` and
    ``relabel_type`` set. Returns a dict with ``n``, ``kappa`` (None if
    ``n < 2``, since kappa is undefined/meaningless below that), and
    ``agreement_rate`` (raw fraction of exact matches, always computable).
    """
    originals: list[str] = []
    relabels: list[str] = []
    for event in store.iter_events():
        if event.true_type is not None and event.relabel_type is not None:
            originals.append(event.true_type)
            relabels.append(event.relabel_type)

    n = len(originals)
    if n == 0:
        return {"n": 0, "kappa": None, "agreement_rate": None}

    agreement_rate = sum(a == b for a, b in zip(originals, relabels)) / n

    kappa: Optional[float] = None
    if n >= 2:
        from sklearn.metrics import cohen_kappa_score

        kappa = float(cohen_kappa_score(originals, relabels))

    return {"n": n, "kappa": kappa, "agreement_rate": agreement_rate}
