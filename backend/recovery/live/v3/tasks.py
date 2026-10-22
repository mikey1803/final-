"""The frozen v3 task manifest (pre-registration D-10).

``data/v3_tasks.json`` holds the eight v1/v2 starter tasks with identical
goals, steps and categories, plus a pre-authored ``clarify_canned_answer`` for
every task (the seven new ones were written by the researcher on 2026-10-22,
before session 1; ``ambiguous_01`` keeps its v1/v2 answer). The v1/v2
``STARTER_TASKS`` list is unchanged.

A data session refuses to start unless git tracks the manifest and it has no
uncommitted changes (``task_manifest.require_frozen_manifest``).
"""

from __future__ import annotations

import re
from pathlib import Path

from recovery.live.event_schema import Task
from recovery.live.task_manifest import load_manifest, require_frozen_manifest

__all__ = ["TASKS_MANIFEST", "CANNED_MAX_CHARS", "load_v3_tasks", "canned_answer_violations"]

TASKS_MANIFEST = Path(__file__).resolve().parent / "data" / "v3_tasks.json"
CANNED_MAX_CHARS = 200

_TOOL_NAMES = ("wikipedia", "countriesnow", "open_meteo", "open-meteo", "wttr", "jsonplaceholder",
               "dummyjson", "restcountries")
_MENTIONS = ("error", "fail", "retry", "recover", "timeout")


def load_v3_tasks(require_frozen: bool = True) -> list[Task]:
    if require_frozen:
        require_frozen_manifest(TASKS_MANIFEST)
    return load_manifest(TASKS_MANIFEST)


def canned_answer_violations(task: Task) -> list[str]:
    """The mechanical subset of the D-10 canned-answer rules (format, length,
    tool names, failure mentions). Whether an answer states only the goal's
    intended reading, and does not contain the task's answer, was checked by
    hand when the answers were written."""
    a, out = task.clarify_canned_answer, []
    if not a:
        return ["missing"]
    if len(a) > CANNED_MAX_CHARS:
        out.append(f"longer than {CANNED_MAX_CHARS} characters")
    if "\n" in a or "\r" in a:
        out.append("not one line")
    if len(re.findall(r"[.!?](?:\s|$)", a)) > 2:
        out.append("more than two sentences")
    if re.search(r"[{}\[\]`*#=<>]", a):
        out.append("code / JSON / markdown characters")
    goal = (task.goal + " " + " ".join(task.steps)).lower()
    out += [f"names tool not in goal: {t}" for t in _TOOL_NAMES if t in a.lower() and t not in goal]
    out += [f"mentions {m}" for m in _MENTIONS if m in a.lower()]
    return out
