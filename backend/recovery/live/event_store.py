"""Content-addressed on-disk store for live failure events.

Mirrors ``recovery.core.parallel.RunStore``'s pattern (one file per record,
named by content key, atomic tmp-then-rename write, file-presence-is-resume)
adapted to JSON records instead of parquet, because ``FailureEvent`` carries
a nested ``conversation_history`` that doesn't flatten into tabular rows.

A second, derived flat parquet index holds only the scalar columns
(``FailureEvent.index_row()``), rebuilt from the JSON files on demand, so
labelling progress and basic queries don't require reading every JSON file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import json

import pandas as pd

from recovery.live.event_schema import FailureEvent

__all__ = ["EventStore"]


class EventStore:
    """One JSON file per event under ``root/events/``, named by
    ``event_id``. ``root/index.parquet`` is a derived, rebuildable cache of
    scalar columns -- never the source of truth, always safe to delete."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.parquet"

    def path_for(self, event_id: str) -> Path:
        return self.events_dir / f"{event_id}.json"

    def exists(self, event_id: str) -> bool:
        return self.path_for(event_id).exists()

    def write(self, event: FailureEvent) -> Path:
        """Write (or overwrite -- e.g. after labelling) one event.

        Atomic tmp-then-rename, same reasoning as ``RunStore.write``: a
        process killed mid-write must never leave a partial/corrupt JSON
        file at the final path for a later run to mistake for a complete
        record.

        Validates label/prediction ordering before writing -- see
        ``FailureEvent.validate_label_prediction_order``. This makes the
        store itself refuse to persist a predicted-before-labelled event,
        not just a test that catches it after the fact.
        """
        event.validate_label_prediction_order()
        path = self.path_for(event.event_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(event.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)
        return path

    def read(self, event_id: str) -> FailureEvent:
        data = json.loads(self.path_for(event_id).read_text(encoding="utf-8"))
        return FailureEvent.from_dict(data)

    def iter_events(self) -> Iterator[FailureEvent]:
        for p in sorted(self.events_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            yield FailureEvent.from_dict(data)

    def unlabelled(self) -> Iterator[FailureEvent]:
        for ev in self.iter_events():
            if not ev.is_labelled():
                yield ev

    def labelled(self) -> Iterator[FailureEvent]:
        for ev in self.iter_events():
            if ev.is_labelled():
                yield ev

    def rebuild_index(self) -> pd.DataFrame:
        """Recompute ``index.parquet`` from the JSON files on disk. Safe to
        call any time; the JSON files are always the source of truth."""
        rows = [ev.index_row() for ev in self.iter_events()]
        df = pd.DataFrame(rows)
        tmp_path = self.index_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(self.index_path)
        return df

    def read_index(self, rebuild: bool = False) -> pd.DataFrame:
        if rebuild or not self.index_path.exists():
            return self.rebuild_index()
        return pd.read_parquet(self.index_path)

    def labelling_progress(self) -> dict:
        """Overall labelling counts, mixing targeted and untargeted tasks.

        ``by_type`` here is NOT the deployment distribution p and must never
        be fed into ``recovery.theory.analytic``'s ``p`` parameter -- it
        includes targeted tasks' events, which are deliberately balanced by
        design, not representative of real deployment frequency. Use
        ``recovery.live.deployment_distribution.estimate_deployment_distribution``
        for that, which filters to untargeted tasks only.
        """
        df = self.read_index(rebuild=True)
        if df.empty:
            return {
                "total": 0, "labelled": 0, "unlabelled": 0, "by_type": {},
                "unsure_fraction": None,
            }
        labelled_mask = df["true_type"].notna()
        n_labelled = int(labelled_mask.sum())
        if n_labelled > 0 and "label_confidence" in df.columns:
            unsure_fraction = float(
                (df.loc[labelled_mask, "label_confidence"] == "unsure").sum() / n_labelled
            )
        else:
            unsure_fraction = None
        return {
            "total": len(df),
            "labelled": n_labelled,
            "unlabelled": int((~labelled_mask).sum()),
            "by_type": df.loc[labelled_mask, "true_type"].value_counts().to_dict(),
            "unsure_fraction": unsure_fraction,
        }
