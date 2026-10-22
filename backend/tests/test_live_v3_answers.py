"""Answer fields, the instrument-side corrected check and task_completed (A5)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.v3.answers import (  # noqa: E402
    ANSWER_FIELDS, TEMPERATURE_TOLERANCE_C, compare, normalise_string, read_answer, task_completed,
)
from tests._v3_fakes import body_for  # noqa: E402

ARGS = {"wikipedia": {"title": "Kolkata"}, "countriesnow": {"country": "India"},
        "open_meteo": {"latitude": 1, "longitude": 2}, "wttr": {"location": "x"},
        "jsonplaceholder": {"id": 5}, "dummyjson": {"id": 5}}


def test_answer_field_table_matches_v3():
    assert ANSWER_FIELDS == {
        "countriesnow": ("data", "capital"), "wikipedia": ("extract",),
        "open_meteo": ("current_weather", "temperature"), "wttr": ("current_condition", 0, "temp_C"),
        "jsonplaceholder": ("title",), "dummyjson": ("todo",)}
    assert TEMPERATURE_TOLERANCE_C == 3.0


@pytest.mark.parametrize("tool", list(ANSWER_FIELDS))
def test_present_and_parses_on_real_shapes(tool):
    a = read_answer(tool, body_for(tool, ARGS[tool]))
    assert a.present and a.parses


def test_kind_parsing_rules():
    assert not read_answer("open_meteo", json.dumps({"current_weather": {"temperature": True}})).parses
    assert not read_answer("open_meteo", json.dumps({"current_weather": {"temperature": "30"}})).parses
    assert read_answer("wttr", json.dumps({"current_condition": [{"temp_C": "-4"}]})).value == -4.0
    assert not read_answer("wttr", json.dumps({"current_condition": [{"temp_C": "4.5"}]})).parses
    assert not read_answer("wttr", json.dumps({"current_condition": []})).present
    assert not read_answer("countriesnow", json.dumps({"data": {"capital": "  "}})).parses
    assert not read_answer("wikipedia", "not json").present
    assert not read_answer("wikipedia", None).present


def test_string_normalisation_and_match():
    assert normalise_string("  New   Delhi ") == normalise_string("new delhi")
    a = read_answer("countriesnow", json.dumps({"data": {"capital": " new  DELHI"}}))
    assert compare("countriesnow", a, "countriesnow", "New Delhi").match
    assert not compare("countriesnow", a, "countriesnow", "Mumbai").match


@pytest.mark.parametrize("final,ref,ok", [(30.0, 33.0, True), (30.0, 27.0, True), (30.0, 33.01, False),
                                          (30.0, 40.0, False)])
def test_temperature_tolerance_is_inclusive_3c(final, ref, ok):
    a = read_answer("open_meteo", json.dumps({"current_weather": {"temperature": final}}))
    c = compare("open_meteo", a, "open_meteo", ref)
    assert c.match is ok and c.tolerance == "abs<=3.0C"


def test_cross_provider_temperature_and_mixed_kinds():
    a = read_answer("wttr", json.dumps({"current_condition": [{"temp_C": "31"}]}))
    assert compare("wttr", a, "open_meteo", 30.5).match
    s = read_answer("countriesnow", json.dumps({"data": {"capital": "30"}}))
    assert not compare("countriesnow", s, "open_meteo", 30.0).match


@pytest.mark.parametrize("tool", list(ANSWER_FIELDS))
def test_task_completed_per_tool(tool):
    text = body_for(tool, ARGS[tool])
    genuine = read_answer(tool, text).value
    assert task_completed(tool, text, degraded_block=False)
    assert task_completed(tool, text, degraded_block=True, genuine_tool=tool, genuine_value=genuine)
    wrong = 99.0 if tool in ("open_meteo", "wttr") else "something else"
    assert task_completed(tool, text, degraded_block=False)                 # no genuine comparison outside D
    assert not task_completed(tool, text, degraded_block=True, genuine_tool=tool, genuine_value=wrong)
    assert not task_completed(tool, "{}", degraded_block=False)
    assert not task_completed(None, text, degraded_block=False)
