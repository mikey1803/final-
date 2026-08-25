"""Tests for recovery.live.pilot_report: skeleton-manifest shape and
yield-rate computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.event_schema import AGENT_SIDE_TARGETABLE_TYPES, AgentState, FailureEvent
from recovery.live.event_store import EventStore
from recovery.live.pilot_report import build_pilot_skeleton, compute_yield_report
from src.failures.taxonomy import FailureType


def _make_labelled_event(task_id, step_index, raw_error, true_type) -> FailureEvent:
    ev = FailureEvent(
        session_id="sess1", task_id=task_id, step_index=step_index, tool_name="wikipedia",
        tool_args={}, raw_error=raw_error, agent_state=AgentState(goal="g", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )
    ev.label(true_type, labelled_by="pio", labelled_at="t")
    return ev


def test_build_pilot_skeleton_default_counts_and_flags():
    tasks = build_pilot_skeleton(per_type=4, n_natural=8)
    assert len(tasks) == 5 * 4 + 8  # 28, within the 25-30 pilot range

    targeted = [t for t in tasks if t.targeted]
    natural = [t for t in tasks if not t.targeted]
    assert len(targeted) == 20
    assert len(natural) == 8
    assert {t.intended_failure_type for t in targeted} == set(AGENT_SIDE_TARGETABLE_TYPES)
    assert all(t.intended_failure_type is None for t in natural)
    assert all("TODO" in t.goal for t in tasks), "task text must be a placeholder, never generated"

    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)), "task_ids must be unique"


def test_compute_yield_report_targeted_type(tmp_path):
    from recovery.live.event_schema import Task

    tasks = [
        Task(task_id="t_amb_01", goal="x", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
        Task(task_id="t_amb_02", goal="x", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
    ]
    store = EventStore(tmp_path)
    # t_amb_01 hit its intended type once; t_amb_02's failure was mislabelled
    # a different type (a real possibility -- design intent != guaranteed outcome).
    store.write(_make_labelled_event("t_amb_01", 0, "unclear", FailureType.AMBIGUOUS_INSTRUCTION))
    store.write(_make_labelled_event("t_amb_02", 0, "oops", FailureType.TOOL_ERROR))

    attempts = {"t_amb_01": 3, "t_amb_02": 3}  # each task run 3 times (n-runs=3)
    report = compute_yield_report(tasks, store, attempts, target_n_low=30, target_n_high=45)

    row = report["ambiguous_instruction"]
    assert row["n_attempts"] == 6
    assert row["n_observed"] == 1
    assert abs(row["yield_rate"] - (1 / 6)) < 1e-9
    assert row["attempts_needed_for_low"] == 180  # ceil(30 / (1/6))
    assert row["attempts_needed_for_high"] == 270  # ceil(45 / (1/6))


def test_compute_yield_report_natural_types_pool_across_all_natural_tasks(tmp_path):
    from recovery.live.event_schema import Task

    tasks = [
        Task(task_id="nat_01", goal="x", category="basic", targeted=False),
        Task(task_id="nat_02", goal="x", category="basic", targeted=False),
    ]
    store = EventStore(tmp_path)
    store.write(_make_labelled_event("nat_01", 0, "timeout", FailureType.TOOL_TIMEOUT))
    store.write(_make_labelled_event("nat_02", 0, "err", FailureType.TOOL_ERROR))

    attempts = {"nat_01": 10, "nat_02": 10}
    report = compute_yield_report(tasks, store, attempts)

    assert report["tool_timeout"]["n_attempts"] == 20
    assert report["tool_timeout"]["n_observed"] == 1
    assert report["tool_error"]["n_attempts"] == 20
    assert report["tool_error"]["n_observed"] == 1


def test_compute_yield_report_zero_attempts_returns_none_fields(tmp_path):
    from recovery.live.event_schema import Task

    tasks = [Task(task_id="t1", goal="x", category="ambiguous",
                  intended_failure_type="ambiguous_instruction", targeted=True)]
    store = EventStore(tmp_path)
    report = compute_yield_report(tasks, store, attempts_by_task_id={})

    row = report["ambiguous_instruction"]
    assert row["n_attempts"] == 0
    assert row["yield_rate"] is None
    assert row["attempts_needed_for_low"] is None


def test_compute_yield_report_zero_observed_yield_returns_none_projection(tmp_path):
    from recovery.live.event_schema import Task

    tasks = [Task(task_id="t1", goal="x", category="ambiguous",
                  intended_failure_type="ambiguous_instruction", targeted=True)]
    store = EventStore(tmp_path)
    report = compute_yield_report(tasks, store, attempts_by_task_id={"t1": 5})

    row = report["ambiguous_instruction"]
    assert row["n_attempts"] == 5
    assert row["n_observed"] == 0
    assert row["yield_rate"] == 0.0
    assert row["attempts_needed_for_low"] is None, "cannot project attempts needed from zero yield"
