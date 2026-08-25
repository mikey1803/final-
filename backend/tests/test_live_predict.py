"""Tests for recovery.live.predict: the classifier-prediction pass must
only ever touch already-labelled events, structurally enforced by
iterating EventStore.labelled() and by EventStore.write()'s label/
prediction-order guard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.event_schema import AgentState, FailureEvent
from recovery.live.event_store import EventStore
from recovery.live.predict import predict_labelled_events
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


def _fake_classifier(raw_error: str):
    return "tool_error", {"tool_error": 0.9, "tool_timeout": 0.1}


def test_predict_only_touches_labelled_events(tmp_path):
    store = EventStore(tmp_path)
    labelled = _make_event(step_index=0, raw_error="err A")
    labelled.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    unlabelled = _make_event(step_index=1, raw_error="err B")
    store.write(labelled)
    store.write(unlabelled)

    n = predict_labelled_events(store, _fake_classifier)

    assert n == 1
    assert store.read(labelled.event_id).predicted_type == "tool_error"
    assert store.read(unlabelled.event_id).predicted_type is None


def test_predict_skips_already_predicted_unless_overwrite(tmp_path):
    store = EventStore(tmp_path)
    event = _make_event()
    event.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    store.write(event)

    n1 = predict_labelled_events(store, _fake_classifier)
    assert n1 == 1

    def _different_classifier(raw_error: str):
        return "tool_timeout", {"tool_timeout": 1.0}

    n2 = predict_labelled_events(store, _different_classifier)
    assert n2 == 0, "already-predicted events must be skipped without overwrite=True"
    assert store.read(event.event_id).predicted_type == "tool_error"

    n3 = predict_labelled_events(store, _different_classifier, overwrite=True)
    assert n3 == 1
    assert store.read(event.event_id).predicted_type == "tool_timeout"


def test_predict_on_empty_store_returns_zero(tmp_path):
    store = EventStore(tmp_path)
    assert predict_labelled_events(store, _fake_classifier) == 0
