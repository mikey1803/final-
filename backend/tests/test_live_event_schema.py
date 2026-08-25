"""Tests for recovery.live.event_schema: content-addressing, label(),
serialization round-trip, and ReplayResult's frozen-strategy validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.event_schema import (
    ALL_ARMS, AgentState, FailureEvent, ReplayResult, Task,
)
from src.failures.taxonomy import FailureType


def _make_event(**overrides) -> FailureEvent:
    defaults = dict(
        session_id="sess1",
        step_index=0,
        tool_name="wikipedia",
        tool_args={"title": "Kolkata"},
        raw_error="HTTP 404: Not Found",
        agent_state=AgentState(goal="look up kolkata", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )
    defaults.update(overrides)
    return FailureEvent(**defaults)


def test_event_id_is_deterministic_over_identity_fields():
    e1 = _make_event()
    e2 = _make_event()
    assert e1.event_id == e2.event_id, "same identity fields must hash to the same event_id"


def test_event_id_changes_with_raw_error():
    e1 = _make_event(raw_error="HTTP 404")
    e2 = _make_event(raw_error="HTTP 500")
    assert e1.event_id != e2.event_id


def test_event_id_does_not_depend_on_non_identity_fields():
    e1 = _make_event(latency_ms=100.0)
    e2 = _make_event(latency_ms=999.0)
    assert e1.event_id == e2.event_id, "latency_ms must not affect identity"


def test_label_sets_all_three_fields():
    e = _make_event()
    assert not e.is_labelled()
    e.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="2026-08-20T11:00:00+00:00")
    assert e.is_labelled()
    assert e.true_type == "tool_error"
    assert e.labelled_by == "pio"
    assert e.labelled_at == "2026-08-20T11:00:00+00:00"


def test_label_accepts_second_choice_and_confidence():
    e = _make_event()
    e.label(
        FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t",
        second_choice_type=FailureType.MALFORMED_OUTPUT, confidence="unsure",
    )
    assert e.true_type == "tool_error"
    assert e.second_choice_type == "malformed_output"
    assert e.label_confidence == "unsure"


def test_label_rejects_invalid_confidence_value():
    e = _make_event()
    with pytest.raises(ValueError):
        e.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t", confidence="very_sure")


def test_relabel_sets_relabel_fields_not_true_type():
    e = _make_event()
    e.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    e.relabel(FailureType.TOOL_TIMEOUT, relabelled_by="pio", relabelled_at="t2")
    assert e.true_type == "tool_error", "relabel must never overwrite the original label"
    assert e.relabel_type == "tool_timeout"
    assert e.relabelled_by == "pio"


def test_validate_label_prediction_order_passes_when_unlabelled_and_unpredicted():
    e = _make_event()
    e.validate_label_prediction_order()  # no error


def test_validate_label_prediction_order_passes_when_labelled_then_predicted():
    e = _make_event()
    e.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    e.predicted_type = "tool_error"
    e.validate_label_prediction_order()  # no error


def test_validate_label_prediction_order_rejects_prediction_before_label():
    e = _make_event()
    e.predicted_type = "tool_error"  # set without labelling first
    with pytest.raises(ValueError):
        e.validate_label_prediction_order()


def test_to_dict_from_dict_roundtrip():
    e = _make_event()
    e.label(FailureType.MALFORMED_OUTPUT, labelled_by="pio", labelled_at="2026-08-20T11:00:00+00:00")
    d = e.to_dict()
    restored = FailureEvent.from_dict(d)
    assert restored.event_id == e.event_id
    assert restored.true_type == "malformed_output"
    assert restored.agent_state.goal == e.agent_state.goal


def test_from_dict_rejects_tampered_identity_fields():
    e = _make_event()
    d = e.to_dict()
    d["raw_error"] = "a completely different error"  # identity field, but stale event_id kept
    with pytest.raises(ValueError):
        FailureEvent.from_dict(d)


def test_index_row_has_no_nested_fields():
    e = _make_event()
    row = e.index_row()
    for value in row.values():
        assert not isinstance(value, (dict, list)), "index_row must be flat/scalar only"


def test_replay_result_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        ReplayResult(
            event_id="abc123", strategy="not_a_real_strategy", trial_index=0,
            replayed_at="2026-08-20T12:00:00+00:00", true_type="tool_error",
        )


def test_replay_result_accepts_all_frozen_arms():
    for i, strategy in enumerate(ALL_ARMS):
        r = ReplayResult(
            event_id="abc123", strategy=strategy, trial_index=i,
            replayed_at="2026-08-20T12:00:00+00:00", true_type="tool_error",
        )
        assert r.replay_id  # computed, non-empty


def test_replay_id_differs_by_trial_index():
    r1 = ReplayResult(event_id="abc", strategy="retry", trial_index=0,
                       replayed_at="t", true_type="tool_error")
    r2 = ReplayResult(event_id="abc", strategy="retry", trial_index=1,
                       replayed_at="t", true_type="tool_error")
    assert r1.replay_id != r2.replay_id, "distinct trials of the same strategy must not collide"


def test_task_defaults():
    t = Task(task_id="t1", goal="do a thing", category="basic")
    assert t.steps == []
    assert t.clarify_canned_answer is None
    assert t.targeted is False


def test_task_targeted_true_requires_agent_side_intended_type():
    t = Task(
        task_id="t1", goal="x", category="ambiguous",
        intended_failure_type="ambiguous_instruction", targeted=True,
    )
    assert t.targeted is True


def test_task_targeted_true_without_intended_type_rejected():
    with pytest.raises(ValueError):
        Task(task_id="t1", goal="x", category="basic", targeted=True)


def test_task_targeted_true_rejects_tool_timeout_or_tool_error():
    """tool_timeout/tool_error come from injection, not task design -- a
    task cannot legitimately target either."""
    with pytest.raises(ValueError):
        Task(task_id="t1", goal="x", category="basic",
             intended_failure_type="tool_timeout", targeted=True)
    with pytest.raises(ValueError):
        Task(task_id="t1", goal="x", category="basic",
             intended_failure_type="tool_error", targeted=True)


def test_task_targeted_false_with_intended_type_rejected():
    with pytest.raises(ValueError):
        Task(task_id="t1", goal="x", category="basic",
             intended_failure_type="tool_error", targeted=False)
