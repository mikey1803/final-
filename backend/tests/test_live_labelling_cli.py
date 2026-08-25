"""Tests for recovery.live.labelling_cli: single-keystroke labelling,
skip, invalid input handling, confidence/second-choice capture, blind
relabelling, and store write-back."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.agreement import compute_relabel_kappa
from recovery.live.event_schema import AgentState, FailureEvent
from recovery.live.event_store import EventStore
from recovery.live.labelling_cli import label_one, relabel_one, run, run_relabel


def _make_event(step_index: int = 0) -> FailureEvent:
    return FailureEvent(
        session_id="sess1",
        step_index=step_index,
        tool_name="wikipedia",
        tool_args={"title": "Kolkata"},
        raw_error="HTTP 404: Not Found",
        agent_state=AgentState(goal="look up kolkata", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )


def _feed_inputs(monkeypatch, answers: list[str]) -> None:
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_label_one_valid_key_labels_and_writes(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)

    # 2 = tool_error, then Enter/Enter for confidence+second-choice defaults
    _feed_inputs(monkeypatch, ["2", "", ""])
    outcome = label_one(store, event, labelled_by="pio")

    assert outcome == "labelled"
    restored = store.read(event.event_id)
    assert restored.true_type == "tool_error"
    assert restored.labelled_by == "pio"
    assert restored.label_confidence == "confident"
    assert restored.second_choice_type is None


def test_label_one_invalid_then_valid_key(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)

    # two bad keys, then 3 = wrong_tool_selected, then confidence/second-choice defaults
    _feed_inputs(monkeypatch, ["9", "abc", "3", "", ""])
    outcome = label_one(store, event, labelled_by="pio")

    assert outcome == "labelled"
    assert store.read(event.event_id).true_type == "wrong_tool_selected"


def test_label_one_captures_unsure_confidence_and_second_choice(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)

    # 2 = tool_error, u = unsure, 5 = malformed_output as second choice
    _feed_inputs(monkeypatch, ["2", "u", "5"])
    outcome = label_one(store, event, labelled_by="pio")

    assert outcome == "labelled"
    restored = store.read(event.event_id)
    assert restored.true_type == "tool_error"
    assert restored.label_confidence == "unsure"
    assert restored.second_choice_type == "malformed_output"


def test_label_one_skip_does_not_write_label(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)

    _feed_inputs(monkeypatch, ["s"])
    outcome = label_one(store, event, labelled_by="pio")

    assert outcome == "skipped"
    assert store.read(event.event_id).true_type is None


def test_label_one_quit(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    event = _make_event()
    store.write(event)

    _feed_inputs(monkeypatch, ["q"])
    outcome = label_one(store, event, labelled_by="pio")

    assert outcome == "quit"
    assert store.read(event.event_id).true_type is None


def test_run_labels_all_pending_and_updates_progress(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    for i in range(3):
        store.write(_make_event(step_index=i))

    _feed_inputs(monkeypatch, ["1", "", "", "2", "", "", "1", "", ""])
    run(tmp_path, labelled_by="pio")

    progress = store.labelling_progress()
    assert progress["labelled"] == 3
    assert progress["unlabelled"] == 0
    assert progress["by_type"] == {"tool_timeout": 2, "tool_error": 1}
    assert progress["unsure_fraction"] == 0.0


def test_run_stops_at_quit_leaving_rest_unlabelled(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    for i in range(3):
        store.write(_make_event(step_index=i))

    _feed_inputs(monkeypatch, ["1", "", "", "q"])
    run(tmp_path, labelled_by="pio")

    progress = store.labelling_progress()
    assert progress["labelled"] == 1
    assert progress["unlabelled"] == 2


def test_run_on_empty_store_does_not_crash(tmp_path, capsys):
    run(tmp_path, labelled_by="pio")
    out = capsys.readouterr().out
    assert "Nothing to label" in out


def test_relabel_one_writes_relabel_fields_not_true_type(tmp_path, monkeypatch):
    from src.failures.taxonomy import FailureType

    store = EventStore(tmp_path)
    event = _make_event()
    event.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="2026-08-20T10:00:00+00:00")
    store.write(event)

    _feed_inputs(monkeypatch, ["1"])  # different type on the blind second pass
    outcome = relabel_one(store, event, labelled_by="pio")

    assert outcome == "labelled"
    restored = store.read(event.event_id)
    assert restored.true_type == "tool_error"  # untouched
    assert restored.relabel_type == "tool_timeout"
    assert restored.relabelled_by == "pio"


def test_run_relabel_samples_and_writes_relabel_fields(tmp_path, monkeypatch):
    """Integration check: run_relabel samples the labelled pool and writes
    relabel_type on each sampled event. Sampling order isn't guaranteed
    (random.Random.sample), so this checks counts/fields, not agreement --
    see test_compute_relabel_kappa_* below for the agreement-math checks
    against hand-constructed, order-controlled events."""
    from src.failures.taxonomy import FailureType

    store = EventStore(tmp_path)
    types = [FailureType.TOOL_ERROR, FailureType.TOOL_TIMEOUT, FailureType.WRONG_TOOL_SELECTED]
    for i, ft in enumerate(types):
        ev = _make_event(step_index=i)
        ev.label(ft, labelled_by="pio", labelled_at="2026-08-20T10:00:00+00:00")
        store.write(ev)

    _feed_inputs(monkeypatch, ["2", "1", "3"])
    run_relabel(tmp_path, n=50, seed=0, labelled_by="pio")

    relabelled = [ev for ev in store.iter_events() if ev.relabel_type is not None]
    assert len(relabelled) == 3
    assert all(ev.relabelled_by == "pio" for ev in relabelled)
    assert all(ev.true_type is not None for ev in relabelled), "original labels must survive"

    result = compute_relabel_kappa(store)
    assert result["n"] == 3
    assert 0.0 <= result["agreement_rate"] <= 1.0


def test_compute_relabel_kappa_perfect_agreement(tmp_path):
    from src.failures.taxonomy import FailureType

    store = EventStore(tmp_path)
    types = [FailureType.TOOL_ERROR, FailureType.TOOL_TIMEOUT, FailureType.WRONG_TOOL_SELECTED]
    for i, ft in enumerate(types):
        ev = _make_event(step_index=i)
        ev.label(ft, labelled_by="pio", labelled_at="t")
        ev.relabel(ft, relabelled_by="pio", relabelled_at="t2")  # same type both passes
        store.write(ev)

    result = compute_relabel_kappa(store)
    assert result["n"] == 3
    assert result["agreement_rate"] == 1.0
    assert result["kappa"] == 1.0


def test_compute_relabel_kappa_partial_agreement(tmp_path):
    from src.failures.taxonomy import FailureType

    store = EventStore(tmp_path)
    pairs = [
        (FailureType.TOOL_ERROR, FailureType.TOOL_ERROR),          # agree
        (FailureType.TOOL_TIMEOUT, FailureType.TOOL_TIMEOUT),      # agree
        (FailureType.WRONG_TOOL_SELECTED, FailureType.TOOL_ERROR),  # disagree
    ]
    for i, (orig, second) in enumerate(pairs):
        ev = _make_event(step_index=i)
        ev.label(orig, labelled_by="pio", labelled_at="t")
        ev.relabel(second, relabelled_by="pio", relabelled_at="t2")
        store.write(ev)

    result = compute_relabel_kappa(store)
    assert result["n"] == 3
    assert abs(result["agreement_rate"] - (2 / 3)) < 1e-9
    assert result["kappa"] is not None and result["kappa"] < 1.0


def test_run_relabel_on_no_labelled_events_does_not_crash(tmp_path, capsys):
    run_relabel(tmp_path, n=50, seed=0, labelled_by="pio")
    out = capsys.readouterr().out
    assert "No labelled events" in out


def test_compute_relabel_kappa_empty_store_returns_none():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store = EventStore(Path(d))
        result = compute_relabel_kappa(store)
        assert result == {"n": 0, "kappa": None, "agreement_rate": None}
