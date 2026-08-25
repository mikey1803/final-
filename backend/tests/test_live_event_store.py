"""Tests for recovery.live.event_store.EventStore: content-addressed write/
read, index rebuilding, and labelling-progress accounting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.event_schema import AgentState, FailureEvent
from recovery.live.event_store import EventStore
from src.failures.taxonomy import FailureType


def _make_event(step_index: int = 0, raw_error: str = "HTTP 404") -> FailureEvent:
    return FailureEvent(
        session_id="sess1",
        step_index=step_index,
        tool_name="wikipedia",
        tool_args={"title": "Kolkata"},
        raw_error=raw_error,
        agent_state=AgentState(goal="look up kolkata", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )


def test_write_then_read_roundtrip(tmp_path):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)
    assert store.exists(event.event_id)
    restored = store.read(event.event_id)
    assert restored.event_id == event.event_id
    assert restored.tool_name == "wikipedia"


def test_write_is_atomic_no_tmp_file_left_behind(tmp_path):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)
    tmp_files = list(store.events_dir.glob("*.json.tmp"))
    assert tmp_files == []


def test_rewrite_after_labelling_overwrites_same_file(tmp_path):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)
    event.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    store.write(event)  # same event_id, must overwrite not duplicate

    files = list(store.events_dir.glob("*.json"))
    assert len(files) == 1
    restored = store.read(event.event_id)
    assert restored.true_type == "tool_error"


def test_write_rejects_prediction_before_label(tmp_path):
    """Structural labelling-integrity guard: the store must refuse to
    persist an event with predicted_type set but true_type still None --
    a prediction pass must never run ahead of a human label."""
    store = EventStore(tmp_path)
    event = _make_event()
    event.predicted_type = "tool_error"  # no label() call first
    with pytest.raises(ValueError):
        store.write(event)
    assert not store.exists(event.event_id), "the invalid event must never reach disk"


def test_write_allows_prediction_after_label(tmp_path):
    store = EventStore(tmp_path)
    event = _make_event()
    event.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    event.predicted_type = "tool_error"
    store.write(event)  # must not raise
    assert store.read(event.event_id).predicted_type == "tool_error"


def test_labelled_iterator_returns_only_labelled_events(tmp_path):
    store = EventStore(tmp_path)
    e1 = _make_event(step_index=0, raw_error="err A")
    e2 = _make_event(step_index=1, raw_error="err B")
    store.write(e1)
    e2.label(FailureType.TOOL_TIMEOUT, labelled_by="pio", labelled_at="t")
    store.write(e2)

    labelled = list(store.labelled())
    assert len(labelled) == 1
    assert labelled[0].event_id == e2.event_id


def test_iter_events_and_unlabelled(tmp_path):
    store = EventStore(tmp_path)
    e1 = _make_event(step_index=0, raw_error="err A")
    e2 = _make_event(step_index=1, raw_error="err B")
    store.write(e1)
    store.write(e2)
    e2.label(FailureType.TOOL_TIMEOUT, labelled_by="pio", labelled_at="t")
    store.write(e2)

    all_events = list(store.iter_events())
    assert len(all_events) == 2

    unlabelled = list(store.unlabelled())
    assert len(unlabelled) == 1
    assert unlabelled[0].event_id == e1.event_id


def test_rebuild_index_matches_event_count(tmp_path):
    store = EventStore(tmp_path)
    for i in range(5):
        store.write(_make_event(step_index=i, raw_error=f"err {i}"))
    df = store.rebuild_index()
    assert len(df) == 5
    assert "event_id" in df.columns
    assert "conversation_history" not in df.columns  # nested fields excluded


def test_labelling_progress_counts(tmp_path):
    store = EventStore(tmp_path)
    for i in range(3):
        store.write(_make_event(step_index=i, raw_error=f"err {i}"))
    events = list(store.iter_events())
    events[0].label(FailureType.TOOL_TIMEOUT, labelled_by="pio", labelled_at="t")
    events[1].label(FailureType.TOOL_TIMEOUT, labelled_by="pio", labelled_at="t")
    store.write(events[0])
    store.write(events[1])

    progress = store.labelling_progress()
    assert progress["total"] == 3
    assert progress["labelled"] == 2
    assert progress["unlabelled"] == 1
    assert progress["by_type"] == {"tool_timeout": 2}


def test_labelling_progress_empty_store(tmp_path):
    store = EventStore(tmp_path)
    progress = store.labelling_progress()
    assert progress == {
        "total": 0, "labelled": 0, "unlabelled": 0, "by_type": {}, "unsure_fraction": None,
    }


def test_labelling_progress_unsure_fraction(tmp_path):
    store = EventStore(tmp_path)
    e1 = _make_event(step_index=0, raw_error="err A")
    e2 = _make_event(step_index=1, raw_error="err B")
    e1.label(FailureType.TOOL_TIMEOUT, labelled_by="pio", labelled_at="t", confidence="unsure")
    e2.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t", confidence="confident")
    store.write(e1)
    store.write(e2)

    progress = store.labelling_progress()
    assert progress["unsure_fraction"] == 0.5


def test_read_index_rebuild_flag(tmp_path):
    store = EventStore(tmp_path)
    store.write(_make_event())
    df1 = store.read_index(rebuild=True)
    assert len(df1) == 1
    store.write(_make_event(step_index=1, raw_error="second"))
    df2 = store.read_index(rebuild=False)  # stale cache from before second write
    assert len(df2) == 1
    df3 = store.read_index(rebuild=True)
    assert len(df3) == 2
