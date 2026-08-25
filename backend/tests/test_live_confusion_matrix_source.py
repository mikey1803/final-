"""Tests for recovery.live.confusion_matrix_source: confusion-matrix input
must come only from Campaign A (targeted) events, mirroring
deployment_distribution's Campaign-B-only rule for p."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.confusion_matrix_source import (
    ConfusionMatrixSourceError, collect_confusion_matrix_events,
)
from recovery.live.event_schema import AgentState, FailureEvent, Task
from recovery.live.event_store import EventStore
from src.failures.taxonomy import FailureType


def _make_labelled_event(task_id, step_index, raw_error, true_type) -> FailureEvent:
    ev = FailureEvent(
        session_id="sess1", task_id=task_id, step_index=step_index, tool_name="wikipedia",
        tool_args={}, raw_error=raw_error, agent_state=AgentState(goal="g", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )
    ev.label(true_type, labelled_by="pio", labelled_at="t")
    return ev


def test_collect_uses_only_targeted_events(tmp_path):
    tasks = [
        Task(task_id="targeted_1", goal="a", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
        Task(task_id="natural_1", goal="b", category="basic", targeted=False),
    ]
    store = EventStore(tmp_path)
    store.write(_make_labelled_event("targeted_1", 0, "unclear", FailureType.AMBIGUOUS_INSTRUCTION))
    # Campaign B event must never leak into confusion-matrix input, even though
    # it's a real labelled event.
    store.write(_make_labelled_event("natural_1", 0, "err", FailureType.TOOL_ERROR))

    events = collect_confusion_matrix_events(tasks, store)

    assert len(events) == 1
    assert events[0].task_id == "targeted_1"


def test_collect_raises_when_no_targeted_tasks_given():
    tasks = [Task(task_id="natural_1", goal="b", category="basic", targeted=False)]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store = EventStore(Path(d))
        with pytest.raises(ConfusionMatrixSourceError, match="Campaign A"):
            collect_confusion_matrix_events(tasks, store)


def test_collect_raises_when_no_labelled_targeted_events_yet(tmp_path):
    tasks = [Task(task_id="targeted_1", goal="a", category="ambiguous",
                  intended_failure_type="ambiguous_instruction", targeted=True)]
    store = EventStore(tmp_path)
    with pytest.raises(ConfusionMatrixSourceError, match="nothing to build"):
        collect_confusion_matrix_events(tasks, store)


def test_collect_ignores_unlabelled_targeted_events(tmp_path):
    tasks = [Task(task_id="targeted_1", goal="a", category="ambiguous",
                  intended_failure_type="ambiguous_instruction", targeted=True)]
    store = EventStore(tmp_path)
    unlabelled = FailureEvent(
        session_id="s", task_id="targeted_1", step_index=0, tool_name="wikipedia",
        tool_args={}, raw_error="err", agent_state=AgentState(goal="g", conversation_history=[]),
        captured_at="t",
    )
    store.write(unlabelled)
    with pytest.raises(ConfusionMatrixSourceError):
        collect_confusion_matrix_events(tasks, store)
