"""Tests for recovery.live.task_manifest: save/load round-trip, the git
freeze check, and intended-vs-observed distribution reporting."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.event_schema import AgentState, FailureEvent, Task
from recovery.live.event_store import EventStore
from recovery.live.task_manifest import (
    assert_campaign, intended_vs_observed_distribution, is_manifest_frozen, load_manifest,
    require_frozen_manifest, save_manifest,
)
from src.failures.taxonomy import FailureType


def _sample_tasks() -> list[Task]:
    return [
        Task(task_id="t1", goal="look up X", category="basic",
             intended_failure_type=None, targeted=False),
        Task(task_id="t2", goal="ambiguous thing", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
    ]


def test_save_load_roundtrip(tmp_path):
    tasks = _sample_tasks()
    path = tmp_path / "manifest.json"
    save_manifest(tasks, path)
    loaded = load_manifest(path)

    assert len(loaded) == 2
    assert loaded[0].task_id == "t1"
    assert loaded[0].intended_failure_type is None
    assert loaded[1].intended_failure_type == "ambiguous_instruction"


def test_load_manifest_rejects_duplicate_task_ids(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '[{"task_id": "dup", "goal": "a", "category": "basic"}, '
        '{"task_id": "dup", "goal": "b", "category": "basic"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_manifest(path)


def test_task_rejects_invalid_intended_failure_type():
    with pytest.raises(ValueError):
        Task(task_id="t1", goal="x", category="basic", intended_failure_type="not_a_real_type")


def _init_git_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo_dir), check=True)


def test_is_manifest_frozen_false_when_untracked(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "manifest.json"
    save_manifest(_sample_tasks(), path)

    frozen, reason = is_manifest_frozen(path, repo_dir=tmp_path)
    assert frozen is False
    assert "not tracked" in reason


def test_is_manifest_frozen_false_when_uncommitted_changes(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "manifest.json"
    save_manifest(_sample_tasks(), path)
    subprocess.run(["git", "add", "manifest.json"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add manifest"], cwd=str(tmp_path), check=True)

    # Dirty it again after commit.
    save_manifest(_sample_tasks() + [Task(task_id="t3", goal="y", category="basic")], path)

    frozen, reason = is_manifest_frozen(path, repo_dir=tmp_path)
    assert frozen is False
    assert "uncommitted" in reason


def test_is_manifest_frozen_true_when_committed_and_clean(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "manifest.json"
    save_manifest(_sample_tasks(), path)
    subprocess.run(["git", "add", "manifest.json"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add manifest"], cwd=str(tmp_path), check=True)

    frozen, reason = is_manifest_frozen(path, repo_dir=tmp_path)
    assert frozen is True


def test_require_frozen_manifest_raises_when_not_frozen(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "manifest.json"
    save_manifest(_sample_tasks(), path)  # never committed

    with pytest.raises(RuntimeError):
        require_frozen_manifest(path, repo_dir=tmp_path)


def test_require_frozen_manifest_passes_when_frozen(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "manifest.json"
    save_manifest(_sample_tasks(), path)
    subprocess.run(["git", "add", "manifest.json"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add manifest"], cwd=str(tmp_path), check=True)

    require_frozen_manifest(path, repo_dir=tmp_path)  # must not raise


def _make_event(task_id, step_index, raw_error) -> FailureEvent:
    return FailureEvent(
        session_id="sess1",
        task_id=task_id,
        step_index=step_index,
        tool_name="wikipedia",
        tool_args={},
        raw_error=raw_error,
        agent_state=AgentState(goal="g", conversation_history=[]),
        captured_at="2026-08-20T10:00:00+00:00",
    )


def test_intended_vs_observed_distribution(tmp_path):
    tasks = [
        Task(task_id="t1", goal="a", category="basic", intended_failure_type=None, targeted=False),
        Task(task_id="t2", goal="b", category="ambiguous",
             intended_failure_type="ambiguous_instruction", targeted=True),
    ]
    store = EventStore(tmp_path)

    e1 = _make_event("t2", 0, "unclear request")
    e1.label(FailureType.AMBIGUOUS_INSTRUCTION, labelled_by="pio", labelled_at="t")
    store.write(e1)

    e2 = _make_event("t1", 0, "unexpected tool error")
    e2.label(FailureType.TOOL_ERROR, labelled_by="pio", labelled_at="t")
    store.write(e2)

    e3 = _make_event("t2", 1, "not yet labelled")  # must be excluded (unlabelled)
    store.write(e3)

    report = intended_vs_observed_distribution(tasks, store)
    assert report["intended"] == {None: 1, "ambiguous_instruction": 1}
    assert report["observed"] == {"ambiguous_instruction": 1, "tool_error": 1}
    assert report["per_task"]["t2"]["observed"] == ["ambiguous_instruction"]
    assert report["per_task"]["t1"]["observed"] == ["tool_error"]


def test_assert_campaign_a_rejects_untargeted_task():
    tasks = _sample_tasks()  # t1 is targeted=False
    with pytest.raises(ValueError, match="Campaign A"):
        assert_campaign(tasks, "A")


def test_assert_campaign_b_rejects_targeted_task():
    tasks = _sample_tasks()  # t2 is targeted=True
    with pytest.raises(ValueError, match="Campaign B"):
        assert_campaign(tasks, "B")


def test_assert_campaign_a_passes_all_targeted():
    tasks = [t for t in _sample_tasks() if t.targeted]
    assert_campaign(tasks, "A")  # must not raise


def test_assert_campaign_b_passes_all_untargeted():
    tasks = [t for t in _sample_tasks() if not t.targeted]
    assert_campaign(tasks, "B")  # must not raise


def test_assert_campaign_rejects_invalid_campaign_name():
    with pytest.raises(ValueError, match="'A' or 'B'"):
        assert_campaign(_sample_tasks(), "C")
