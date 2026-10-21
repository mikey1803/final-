"""CLARIFY responders.

Live collection always uses ``CannedAnswerResponder`` (v1 and v2): the task's
pre-authored ``clarify_canned_answer``, so batch runs stay unattended and
reproducible. ``CallbackResponder`` (a human via stdin, a UI, ...) is for the
later ``with_recovery`` wrapper and the dashboard, never collection runs.
Returning None, or timing out, means ``question_unmatched``.

Threat to validity (v2 §7): canned answers make CLARIFY's measured success an
optimistic upper bound on CLARIFY with a real human.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from recovery.live.event_schema import Task


class CannedAnswerResponder:
    kind = "canned"

    def answer(self, question: str, task: Task, timeout_s: float) -> Optional[str]:
        return task.clarify_canned_answer or None


class CallbackResponder:
    """Wraps any ``fn(question, task) -> str | None`` with a hard timeout."""

    def __init__(self, fn: Callable[[str, Task], Optional[str]], kind: str = "human_callback",
                 timeout_s: float = 60.0) -> None:
        self._fn, self.kind, self.timeout_s = fn, kind, timeout_s

    def answer(self, question: str, task: Task, timeout_s: float) -> Optional[str]:
        box: dict = {}
        worker = threading.Thread(target=lambda: box.setdefault("a", self._fn(question, task)), daemon=True)
        worker.start()
        worker.join(min(timeout_s, self.timeout_s))
        if worker.is_alive():
            return None  # timeout -> question_unmatched
        answer = box.get("a")
        return answer if isinstance(answer, str) and answer.strip() else None


def stdin_responder(timeout_s: float = 60.0) -> CallbackResponder:
    return CallbackResponder(lambda q, _task: input(f"\n[clarify] {q}\n> "), kind="human_stdin",
                             timeout_s=timeout_s)
