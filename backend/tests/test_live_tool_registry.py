"""Tool-equivalence registry and v2 argument translation (Task 21)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recovery.live.tools as tools_mod  # noqa: E402
from recovery.live import tool_registry as reg  # noqa: E402
from recovery.live.event_store import EventStore  # noqa: E402
from recovery.live.recovery_attempt import RecoveryStore  # noqa: E402
from recovery.live.tools import TOOL_ALTERNATIVES, TOOLS, ToolCallResult  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "live_event_pre_task21.json"


def test_registry_has_no_self_loops_and_every_backup_exists():
    assert reg.validate_registry() == []
    for name, spec in reg.REGISTRY.items():
        assert spec.backup != name
        assert spec.backup in reg.REGISTRY


def test_registry_covers_exactly_the_live_tools_and_matches_frozen_peer_map():
    assert set(reg.REGISTRY) == set(TOOLS)
    assert {n: s.backup for n, s in reg.REGISTRY.items()} == TOOL_ALTERNATIVES


def test_all_tools_flagged_side_effect_free():
    assert all(not reg.has_side_effect(t) for t in reg.REGISTRY)


@pytest.mark.parametrize("bad, problem", [
    ({"a": reg.ToolSpec("a", "x", "a")}, "self-loop"),
    ({"a": reg.ToolSpec("a", "x", "zzz")}, "not a registered tool"),
    ({"a": reg.ToolSpec("a", "x", "b"), "b": reg.ToolSpec("b", "x", "c"), "c": reg.ToolSpec("c", "x", "b")}, "asymmetric"),
    ({"a": reg.ToolSpec("a", "x", "b"), "b": reg.ToolSpec("b", "y", "a")}, "cross-category"),
])
def test_validate_registry_catches_bad_registries(bad, problem):
    assert any(problem in p for p in reg.validate_registry(bad))


def test_frozen_country_list_hash_and_size():
    names = reg.frozen_country_names()
    assert len(names) == 250
    assert "india" in names and "réunion" in names


def test_country_list_tampering_is_detected(tmp_path, monkeypatch):
    doc = json.loads(reg.COUNTRY_LIST_PATH.read_text(encoding="utf-8"))
    doc["names"].append("Atlantis")
    fake = tmp_path / "countries.json"
    fake.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(reg, "COUNTRY_LIST_PATH", fake)
    reg.frozen_country_names.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="content hash"):
            reg.frozen_country_names()
    finally:
        reg.frozen_country_names.cache_clear()


@pytest.mark.parametrize("tool, args, ok, expected", [
    ("wikipedia", {"title": "India"}, True, {"country": "India"}),
    ("wikipedia", {"title": "united_kingdom"}, True, {"country": "united kingdom"}),
    ("wikipedia", {"title": "  United   States "}, True, {"country": "United States"}),
    ("wikipedia", {"title": "Kolkata"}, False, None),
    ("wikipedia", {"title": "Python_(programming_language)"}, False, None),
    ("wikipedia", {"title": "Ind"}, False, None),                    # no partial matches
    ("countriesnow", {"country": "United States"}, True, {"title": "United_States"}),
    ("open_meteo", {"latitude": 22.57, "longitude": 88.36}, True, {"location": "22.57,88.36"}),
    ("open_meteo", {"latitude": "22.57"}, False, None),
    ("wttr", {"location": "22.57, 88.36"}, True, {"latitude": 22.57, "longitude": 88.36}),
    ("wttr", {"location": "Kolkata"}, False, None),
    ("jsonplaceholder", {"id": 200}, True, {"id": 200}),
    ("jsonplaceholder", {"id": "5"}, True, {"id": 5}),
    ("jsonplaceholder", {"id": 999999}, False, None),
    ("dummyjson", {"id": 200}, True, {"id": 200}),
    ("dummyjson", {"id": 201}, False, None),                         # outside jsonplaceholder todos 1-200
    ("dummyjson", {"id": 0}, False, None),
    ("dummyjson", {"id": True}, False, None),
])
def test_translate_args(tool, args, ok, expected):
    tr = reg.translate_args(tool, args)
    assert tr.ok is ok
    assert tr.target == reg.backup_of(tool)
    assert tr.args == expected
    if not ok:
        assert tr.reason


def test_placeholder_ranges_match_frozen_doc_table():
    r = reg.PLACEHOLDER_RANGES
    assert r["jsonplaceholder"]["todos"] == (1, 200) and r["dummyjson"]["todos"] == (1, 254)
    assert r["jsonplaceholder"]["comments"] == (1, 500) and r["dummyjson"]["comments"] == (1, 340)
    assert "albums" not in r["dummyjson"] and "products" not in r["jsonplaceholder"]


# ----------------------------------------------------------------------
# backward compatibility: old stores and old ToolCallResult construction
# ----------------------------------------------------------------------

def test_old_stored_event_still_deserializes(tmp_path):
    store = EventStore(tmp_path / "store")
    shutil.copy(FIXTURE, store.events_dir / FIXTURE.name)
    events = list(store.iter_events())
    assert len(events) == 1 and events[0].event_id == "c02bbcb99c1ad530cfce1cdf"
    assert list(RecoveryStore(tmp_path / "store").iter_attempts()) == []   # no recoveries/ dir: fine
    assert store.read_index(rebuild=True).shape[0] == 1


def test_tool_call_result_positional_construction_unchanged():
    r = ToolCallResult(True, 200, "body", None, 12.0)
    assert r.full_text is None and r.injected is False


def test_do_request_keeps_truncated_body_and_adds_full_text(monkeypatch):
    long_text = "x" * 5000

    class _Resp:
        status_code, ok, text = 200, True, long_text

    monkeypatch.setattr(tools_mod.requests, "request", lambda *a, **k: _Resp())
    r = tools_mod._do_request("GET", "https://example.invalid", timeout=1.0)
    assert r.body == "x" * 2000 + "...[truncated]"     # unchanged v1 behaviour
    assert r.full_text == long_text
