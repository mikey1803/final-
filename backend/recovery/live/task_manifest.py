"""Task manifest: a frozen, version-controlled list of Tasks with each
one's INTENDED failure type declared up front, so collection can later
report intended-vs-observed distribution rather than only what happened
to occur.

The freeze check (``require_frozen_manifest``) exists because a manifest
edited mid-collection invalidates the intended-vs-observed comparison the
same way a strategy definition changed mid-run would invalidate a P[t,k]
cell (see docs/live_pilot/strategy_definitions.md) -- both are measurement
instruments, not code, and must not move while data is being collected
against them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from recovery.live.event_schema import FailureEvent, Task
from recovery.live.event_store import EventStore

__all__ = [
    "load_manifest", "save_manifest", "is_manifest_frozen",
    "require_frozen_manifest", "intended_vs_observed_distribution",
    "assert_campaign",
]


def save_manifest(tasks: list[Task], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "task_id": t.task_id,
            "goal": t.goal,
            "category": t.category,
            "steps": t.steps,
            "clarify_canned_answer": t.clarify_canned_answer,
            "intended_failure_type": t.intended_failure_type,
            "targeted": t.targeted,
        }
        for t in tasks
    ]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> list[Task]:
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    task_ids = [r["task_id"] for r in records]
    if len(task_ids) != len(set(task_ids)):
        dupes = {t for t in task_ids if task_ids.count(t) > 1}
        raise ValueError(f"manifest {path} has duplicate task_id(s): {sorted(dupes)}")
    return [
        Task(
            task_id=r["task_id"],
            goal=r["goal"],
            category=r["category"],
            steps=r.get("steps", []),
            clarify_canned_answer=r.get("clarify_canned_answer"),
            intended_failure_type=r.get("intended_failure_type"),
            targeted=r.get("targeted", False),
        )
        for r in records
    ]


def is_manifest_frozen(path: Path, repo_dir: Optional[Path] = None) -> tuple[bool, str]:
    """A manifest is 'frozen' if git tracks it and it has no uncommitted
    changes relative to HEAD. Returns (frozen, reason) -- reason is a
    human-readable explanation either way, so a failed check can be
    reported directly without the caller re-deriving why.
    """
    path = Path(path).resolve()
    repo_dir = Path(repo_dir).resolve() if repo_dir else path.parent

    if not path.exists():
        return False, f"manifest file does not exist: {path}"

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        return False, f"manifest is not tracked by git: {path}. Commit it before collection."

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(path)],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    if dirty.returncode != 0:
        return False, f"git status failed on {path}: {dirty.stderr.strip()}"
    if dirty.stdout.strip():
        return False, f"manifest has uncommitted changes: {path}. Commit before collection."

    return True, f"manifest is committed and clean: {path}"


def require_frozen_manifest(path: Path, repo_dir: Optional[Path] = None) -> None:
    """Raise RuntimeError if the manifest is not frozen. Call this before
    any collection begins against a given manifest."""
    frozen, reason = is_manifest_frozen(path, repo_dir=repo_dir)
    if not frozen:
        raise RuntimeError(
            f"Refusing to start collection: {reason}\n"
            "The task manifest is part of the measurement instrument -- freeze it "
            "(commit, no uncommitted changes) before running collection against it."
        )


def assert_campaign(tasks: list[Task], campaign: str) -> None:
    """Post-pilot collection is two separate, never-pooled campaigns:
    Campaign 'A' (targeted, balanced across the 5 agent-side types --
    confusion-matrix input only) and Campaign 'B' (untargeted/natural, run
    at scale under FlakyWrapper injection -- deployment-distribution p
    input only). This is the collection-time half of that separation:
    it refuses to even START a labelled campaign run against a manifest
    that mixes the two, before any data is collected. The compute-time
    half is recovery.live.confusion_matrix_source (Campaign A only) and
    recovery.live.deployment_distribution (Campaign B only).

    Deliberately NOT applied to the 28-task pilot manifest, which mixes
    both by design (it exists to measure per-type yield, not to produce
    either final dataset) -- only call this when running a labelled,
    post-pilot campaign.
    """
    if campaign == "A":
        bad = [t.task_id for t in tasks if not t.targeted]
        if bad:
            raise ValueError(
                f"Campaign A (confusion-matrix, targeted-only) manifest contains "
                f"untargeted task(s): {bad}. Campaign A and B must never be pooled -- "
                f"move these to a Campaign B manifest instead."
            )
    elif campaign == "B":
        bad = [t.task_id for t in tasks if t.targeted]
        if bad:
            raise ValueError(
                f"Campaign B (deployment distribution, untargeted-only) manifest contains "
                f"targeted task(s): {bad}. Campaign A and B must never be pooled -- "
                f"move these to a Campaign A manifest instead."
            )
    else:
        raise ValueError(f"campaign must be 'A' or 'B', got {campaign!r}")


def intended_vs_observed_distribution(tasks: list[Task], store: EventStore) -> dict:
    """Compare each task's declared ``intended_failure_type`` against the
    ``true_type`` of events actually captured (and labelled) against that
    task_id. Only labelled events count as 'observed' -- an unlabelled
    event's true type isn't known yet.

    Returns {"intended": {type_or_None: count}, "observed": {type: count},
    "per_task": {task_id: {"intended": ..., "observed": [types]}}}.
    """
    intended_counts: dict = {}
    per_task: dict = {}
    for t in tasks:
        intended_counts[t.intended_failure_type] = intended_counts.get(t.intended_failure_type, 0) + 1
        per_task[t.task_id] = {"intended": t.intended_failure_type, "observed": []}

    observed_counts: dict = {}
    for event in store.iter_events():
        if event.true_type is None or event.task_id is None:
            continue
        observed_counts[event.true_type] = observed_counts.get(event.true_type, 0) + 1
        if event.task_id in per_task:
            per_task[event.task_id]["observed"].append(event.true_type)

    return {"intended": intended_counts, "observed": observed_counts, "per_task": per_task}
