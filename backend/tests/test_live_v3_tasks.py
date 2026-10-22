"""D-10: the frozen v3 task manifest with a canned clarify answer per task."""

import dataclasses
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_live_collection import STARTER_TASKS  # noqa: E402
from recovery.live.executors.responders import CannedAnswerResponder  # noqa: E402
from recovery.live.v3 import tasks as v3_tasks  # noqa: E402
from recovery.live.v3.tasks import canned_answer_violations, load_v3_tasks  # noqa: E402

# The researcher's answers (2026-10-22), verbatim
D10 = {
    "wiki_basic_01": "I mean the Indian city of Kolkata, and a short general summary of it is enough.",
    "country_basic_01": "I mean the country India, and I only need the name of its capital city.",
    "weather_basic_01": "I mean the current conditions right now at exactly those coordinates.",
    "weather_basic_02": "I mean London in the United Kingdom, and I want the weather there right now.",
    "placeholder_basic_01": "I mean the single todo item numbered 5, and only that one item.",
    "out_of_scope_01": "I want one flight actually booked from Kolkata to London, departing tomorrow morning.",
    "long_context_01": "First I want India's capital city, then the current weather in that same city "
                       "without looking the capital up again.",
}


def _v3():
    return load_v3_tasks(require_frozen=False)


def test_manifest_is_the_starter_tasks_plus_canned_answers():
    v3 = _v3()
    assert [t.task_id for t in v3] == [t.task_id for t in STARTER_TASKS]
    for new, old in zip(v3, STARTER_TASKS):
        assert dataclasses.replace(new, clarify_canned_answer=None) == dataclasses.replace(
            old, clarify_canned_answer=None), new.task_id
        if old.clarify_canned_answer is not None:
            assert new.clarify_canned_answer == old.clarify_canned_answer          # ambiguous_01 kept
        else:
            assert new.clarify_canned_answer == D10[new.task_id]


def test_every_task_has_an_answer_and_none_breaks_the_mechanical_rules():
    for t in _v3():
        assert canned_answer_violations(t) == [], t.task_id
        assert CannedAnswerResponder().answer("Which one do you mean?", t, 1.0) == t.clarify_canned_answer


def test_v1_v2_starter_tasks_are_unchanged():
    assert sum(t.clarify_canned_answer is not None for t in STARTER_TASKS) == 1


@pytest.mark.parametrize("answer,expected", [
    ("x" * 201, "longer than 200"),
    ("one line\nand another", "not one line"),
    ("A. B. C.", "more than two sentences"),
    ('{"city": "Kolkata"}', "code / JSON"),
    ("Use the dummyjson tool.", "names tool not in goal: dummyjson"),
    ("If it fails, try again.", "mentions fail"),
])
def test_violation_checker_flags_bad_answers(answer, expected):
    t = dataclasses.replace(STARTER_TASKS[0], clarify_canned_answer=answer)
    assert any(expected in v for v in canned_answer_violations(t))


def test_data_sessions_refuse_an_unfrozen_manifest(tmp_path, monkeypatch):
    copy = tmp_path / "v3_tasks.json"
    shutil.copy(v3_tasks.TASKS_MANIFEST, copy)
    monkeypatch.setattr(v3_tasks, "TASKS_MANIFEST", copy)              # untracked by git
    with pytest.raises(RuntimeError, match="Refusing to start collection"):
        load_v3_tasks(require_frozen=True)
    assert len(load_v3_tasks(require_frozen=False)) == 8
