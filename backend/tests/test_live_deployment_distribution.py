"""Tests for recovery.live.deployment_distribution: p must be estimated
only from targeted=False (untargeted/natural) task events, never from the
deliberately-balanced targeted set."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.envs.canonical import TYPE_NAMES
from recovery.live.deployment_distribution import (
    DeploymentDistributionError, estimate_deployment_distribution,
)
from recovery.live.event_schema import AgentState, FailureEvent, Task
from recovery.live.event_store import EventStore
from src.failures.taxonomy import FailureType


def _make_event(task_id, step_index, raw_error, true_type) -> FailureEvent:
    ev = FailureEvent(
        session_id="sess1",
        task_id=task_id,
        step_index=step_index,
        tool_name="wikipedia",
        tool_args={},
        raw_error=raw_error,
        agent_state=AgentState(goal="g", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )
    ev.label(true_type, labelled_by="pio", labelled_at="t")
    return ev


def test_estimate_uses_only_untargeted_events(tmp_path):
    tasks = [
        Task(task_id="natural_1", goal="a", category="basic", targeted=False),
        Task(task_id="targeted_1", goal="b", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
    ]
    store = EventStore(tmp_path)
    store.write(_make_event("natural_1", 0, "err A", FailureType.TOOL_ERROR))
    store.write(_make_event("natural_1", 1, "err B", FailureType.TOOL_ERROR))
    # Targeted task's event must NOT influence p, even though it's a real
    # labelled event -- it's balanced-by-design, not deployment-representative.
    store.write(_make_event("targeted_1", 0, "unclear", FailureType.AMBIGUOUS_INSTRUCTION))

    p = estimate_deployment_distribution(tasks, store)

    assert p.shape == (len(TYPE_NAMES),)
    assert np.isclose(p.sum(), 1.0)
    assert p[TYPE_NAMES.index("tool_error")] == 1.0
    assert p[TYPE_NAMES.index("ambiguous_instruction")] == 0.0, (
        "the targeted task's event must be excluded entirely"
    )


def test_estimate_raises_when_no_untargeted_tasks_given():
    tasks = [
        Task(task_id="targeted_1", goal="b", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store = EventStore(Path(d))
        with pytest.raises(DeploymentDistributionError, match="targeted=False"):
            estimate_deployment_distribution(tasks, store)


def test_estimate_raises_when_no_labelled_untargeted_events_yet(tmp_path):
    tasks = [Task(task_id="natural_1", goal="a", category="basic", targeted=False)]
    store = EventStore(tmp_path)
    with pytest.raises(DeploymentDistributionError, match="zero observations"):
        estimate_deployment_distribution(tasks, store)


def test_estimate_ignores_unlabelled_events(tmp_path):
    tasks = [Task(task_id="natural_1", goal="a", category="basic", targeted=False)]
    store = EventStore(tmp_path)
    unlabelled = FailureEvent(
        session_id="s", task_id="natural_1", step_index=0, tool_name="wikipedia",
        tool_args={}, raw_error="err", agent_state=AgentState(goal="g", conversation_history=[]),
        captured_at="t",
    )
    store.write(unlabelled)
    with pytest.raises(DeploymentDistributionError):
        estimate_deployment_distribution(tasks, store)
