"""FlakyWrapperV3: per-mode persistence (A2), the read-only resend probe (A3),
plausible-wrong values and frozen libraries (A6), seeds (A8). No network."""

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.v3 import degrade as degrade_mod  # noqa: E402
from recovery.live.v3 import flaky as flaky_mod  # noqa: E402
from recovery.live.v3 import seeds as seeds_mod  # noqa: E402
from recovery.live.v3.answers import read_answer  # noqa: E402
from recovery.live.v3.degrade import DELTA_RANGE_C, DegradeSpec, apply_degradation, draw_spec  # noqa: E402
from recovery.live.v3.flaky import OUTAGE_TIMEOUT_S, FlakyWrapperV3, injection_key, normalise_args  # noqa: E402
from recovery.live.v3.libraries import (  # noqa: E402
    DATA_DIR, FROZEN_ITEMS_SHA256, build_libraries, items_sha256, load_libraries,
)
from recovery.live.v3.seeds import STREAMS, stream_seed  # noqa: E402
from tests._v3_fakes import FakeServices, body_for, tiny_libraries  # noqa: E402

TOOLS6 = ("wikipedia", "countriesnow", "open_meteo", "wttr", "jsonplaceholder", "dummyjson")
ARGS = {"wikipedia": {"title": "Kolkata"}, "countriesnow": {"country": "India"},
        "open_meteo": {"latitude": 22.57, "longitude": 88.36}, "wttr": {"location": "London"},
        "jsonplaceholder": {"id": 5}, "dummyjson": {"id": 5}}


def wrapper(mode, svc=None, rate=0.2, seed=231000, block=0, n=None, libs=True):
    svc = svc or FakeServices()
    return FlakyWrapperV3(mode=mode, session_seed=seed, block_index=block, rate=rate, outage_N=n,
                          libraries=tiny_libraries() if libs else None, tools=svc.tools()), svc


# ---------------------------------------------------------------- keys and seeds

def test_normalised_key_merges_case_and_whitespace():
    assert normalise_args({"country": "india "}) == normalise_args({"country": "India"})
    assert injection_key("countriesnow", {"country": "india "}) == injection_key("countriesnow", {"country": "India"})
    assert injection_key("countriesnow", {"country": "India"}) != injection_key("wikipedia", {"country": "India"})
    assert normalise_args({"id": 5}) == normalise_args({"id": 5.0})


def test_streams_are_seedsequence_derived_and_distinct():
    a = stream_seed(231001, "injection", 0)[0]
    assert a == stream_seed(231001, "injection", 0)[0]
    others = {stream_seed(231001, s, 0)[0] for s in STREAMS}
    assert len(others) == len(STREAMS)
    assert stream_seed(231001, "injection", 1)[0] != a
    with pytest.raises(ValueError):
        stream_seed(1, "nope")
    for mod in (seeds_mod, flaky_mod, degrade_mod):
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(mod)))
                 if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "hash"]
        assert calls == [], f"{mod.__name__} uses hash()"


def test_wrapper_validates_parameters():
    with pytest.raises(ValueError):
        FlakyWrapperV3(mode="sometimes", session_seed=1, block_index=0)
    with pytest.raises(ValueError):
        FlakyWrapperV3(mode="tool_outage", session_seed=1, block_index=0)          # needs N
    with pytest.raises(ValueError):
        FlakyWrapperV3(mode="transient", session_seed=1, block_index=0, outage_N=5)
    with pytest.raises(ValueError):
        FlakyWrapperV3(mode="degraded_output", session_seed=1, block_index=0)     # needs libraries


# ---------------------------------------------------------------- transient

def test_transient_is_independent_per_call_and_reproducible():
    def run():
        w, _ = wrapper("transient", rate=0.5)
        w.start_episode("e")
        return [(r.v3["injected"], r.v3["injected_kind"]) for r in (w.call("countriesnow", ARGS["countriesnow"])
                                                                     for _ in range(60))]
    seq = run()
    assert seq == run()
    flags = [i for i, _ in seq]
    assert 0 < sum(flags) < 60                      # same key, still independent draws
    assert {k for _, k in seq if k} == {"timeout", "malformed_request"}


def test_transient_rate_extremes():
    w, _ = wrapper("transient", rate=0.0)
    assert not any(w.call("dummyjson", {"id": 1}).injected for _ in range(30))
    w, _ = wrapper("transient", rate=1.0)
    assert all(w.call("dummyjson", {"id": 1}).injected for _ in range(30))


# ---------------------------------------------------------------- sticky_args

def test_sticky_same_key_same_corruption_new_key_fresh_draw_no_timeouts():
    w, svc = wrapper("sticky_args", rate=1.0)
    w.start_episode("e1")
    first = w.call("countriesnow", {"country": "India"})
    again = w.call("countriesnow", {"country": "india "})
    assert first.v3["injected"] and first.v3["injected_kind"] == "malformed_request" and not first.success
    assert again.v3["sticky_hit"] and not again.v3["fresh_draw"] and not again.success
    assert all(t == 10.0 for (_, _, _, t) in svc.calls)        # never a timeout
    w2, _ = wrapper("sticky_args", rate=0.0)
    w2.start_episode("e")
    assert w2.call("countriesnow", {"country": "India"}).success
    assert w2.call("countriesnow", {"country": "India"}).v3["sticky_hit"] is False


def test_sticky_keys_reset_per_episode_and_draw_once_per_key():
    w, _ = wrapper("sticky_args", rate=0.5, seed=231007)
    outcomes = {}
    for ep in range(30):
        w.start_episode(f"e{ep}")
        r1 = w.call("dummyjson", {"id": 1})
        r2 = w.call("dummyjson", {"id": 1})
        assert r1.success == r2.success                     # persists within the episode
        outcomes[ep] = r1.success
    assert len(set(outcomes.values())) == 2                  # but redrawn across episodes


# ---------------------------------------------------------------- tool_outage

@pytest.mark.parametrize("N", [2, 5, 20])
def test_outage_exactly_n_calls_then_served_and_other_tools_dont_count(N):
    w, svc = wrapper("tool_outage", rate=1.0, n=N)
    w.start_episode("e")
    results = []
    for i in range(N):
        results.append(w.call("open_meteo", ARGS["open_meteo"]))
        other = w.call("jsonplaceholder", {"id": 1})          # its own outage state; doesn't advance open_meteo
        assert other.v3["outage_id"] != results[0].v3["outage_id"] or other.v3["outage_id"] is None
    assert [r.v3["outage_call_index"] for r in results] == list(range(1, N + 1))
    assert all(not r.success and r.v3["injected_kind"] == "timeout" for r in results)
    served = w.call("open_meteo", ARGS["open_meteo"])
    assert served.success and served.v3.get("served_after_outage") and not served.v3["fresh_draw"]
    assert any(t == OUTAGE_TIMEOUT_S for (_, _, _, t) in svc.calls)


def test_outage_peers_are_independent():
    w, _ = wrapper("tool_outage", rate=0.5, n=5, seed=231003)
    w.start_episode("e")
    states = set()
    for _ in range(80):
        a = w.call("open_meteo", ARGS["open_meteo"])
        b = w.call("wttr", ARGS["wttr"])
        states.add((a.success, b.success))
    assert {(True, True), (False, True), (True, False)} <= states


# ---------------------------------------------------------------- resend probe is read-only (A3)

@pytest.mark.parametrize("mode,n", [("transient", None), ("sticky_args", None), ("tool_outage", 2),
                                    ("tool_outage", 5), ("tool_outage", 20), ("degraded_output", None)])
def test_resend_probe_never_changes_the_strategy_world(mode, n):
    def run(with_probes):
        w, _ = wrapper(mode, rate=0.4, n=n, seed=231011)
        out = []
        for ep in range(6):
            w.start_episode(f"e{ep}")
            for t in TOOLS6:
                for _ in range(3):
                    r = w.call(t, ARGS[t])
                    out.append((t, r.success, r.v3["injected"], r.v3.get("outage_call_index"),
                                r.v3.get("degraded"), r.v3.get("transformation"), r.full_text))
                    if with_probes and not r.success:
                        w.resend_probe(t, ARGS[t])
                    if with_probes:
                        w.resend_probe(t, {"id": 999} if t in ("jsonplaceholder", "dummyjson") else ARGS[t])
        return out
    assert run(False) == run(True)


def test_probe_sees_state_without_creating_or_advancing_it():
    w, _ = wrapper("sticky_args", rate=1.0)
    w.start_episode("e")
    p = w.resend_probe("countriesnow", {"country": "India"})          # unknown key: served, no key created
    assert p.success and w.key_state("countriesnow", {"country": "India"}) is None
    w.call("countriesnow", {"country": "India"})                       # creates the corrupted key
    assert not w.resend_probe("countriesnow", {"country": "India"}).success
    w, _ = wrapper("tool_outage", rate=1.0, n=2)
    w.start_episode("e")
    first = w.call("wttr", ARGS["wttr"])
    for _ in range(5):
        pr = w.resend_probe("wttr", ARGS["wttr"])
        assert not pr.success and pr.v3["outage_call_index"] is None
    assert w.call("wttr", ARGS["wttr"]).v3["outage_call_index"] == 2   # probes did not count
    assert first.v3["outage_call_index"] == 1


# ---------------------------------------------------------------- degraded_output

def test_degraded_is_sticky_per_key_one_transformation_and_declared():
    w, _ = wrapper("degraded_output", rate=1.0, seed=231005)
    w.start_episode("e")
    a = w.call("countriesnow", {"country": "India"})
    b = w.call("countriesnow", {"country": "india"})
    assert a.success and a.v3["degraded"] and a.v3["synthetic_content"]
    assert a.v3["transformation"] == b.v3["transformation"] and a.full_text == b.full_text
    assert a.v3["original_sha256"] != a.v3["delivered_sha256"]


def test_genuine_captured_on_every_d_call_and_never_in_the_result():
    w, _ = wrapper("degraded_output", rate=0.5, seed=231013)
    for ep in range(10):
        w.start_episode(f"e{ep}")
        for t in TOOLS6:
            r = w.call(t, ARGS[t])
            g = w.genuine(r.v3["call_id"])
            assert g is not None and g["genuine_sha256"]
            if r.v3["degraded"] and r.v3["family"] == "plausible_wrong":
                assert json.dumps(g["genuine_raw"])[1:-1] not in (r.full_text or "")
                assert "genuine" not in json.dumps(r.v3)
    assert any(e.get("degraded") for e in w.log) and any(not e.get("degraded") for e in w.log)


def test_transient_blocks_capture_no_genuine_values():
    w, _ = wrapper("transient", rate=0.0)
    r = w.call("countriesnow", ARGS["countriesnow"])
    assert w.genuine(r.v3["call_id"]) is None


def test_plausible_wrong_values_are_plausible_and_never_the_truth():
    import numpy as np
    libs = tiny_libraries()
    rng = np.random.default_rng(5)
    for _ in range(200):
        spec = draw_spec(rng, 1.0)
        if spec.family != "plausible_wrong":
            continue
        assert DELTA_RANGE_C[0] <= abs(spec.delta) <= DELTA_RANGE_C[1] and spec.k >= 1
        for t in TOOLS6:
            text = body_for(t, ARGS[t])
            genuine = read_answer(t, text)
            delivered, _ = apply_degradation(t, ARGS[t], text, genuine.raw, spec, libs)
            wrong = read_answer(t, delivered)
            assert wrong.present and wrong.parses and wrong.raw != genuine.raw
            if t in ("open_meteo", "wttr"):
                assert 8.0 - 0.5 <= abs(wrong.value - genuine.value) <= 15.0 + 0.5
    signs = {np.sign(draw_spec(np.random.default_rng(i), 1.0).delta or 0) for i in range(60)}
    assert {-1.0, 1.0} <= signs


def test_structural_transformations():
    libs = tiny_libraries()
    text = body_for("open_meteo", ARGS["open_meteo"])
    g = read_answer("open_meteo", text)
    removed, _ = apply_degradation("open_meteo", {}, text, g.raw, DegradeSpec(True, "structural", "remove_field"), libs)
    assert not read_answer("open_meteo", removed).present
    typed, d = apply_degradation("open_meteo", {}, text, g.raw, DegradeSpec(True, "structural", "change_type"), libs)
    assert read_answer("open_meteo", typed).raw == "30.5" and not read_answer("open_meteo", typed).parses
    cut, _ = apply_degradation("open_meteo", {}, text, g.raw, DegradeSpec(True, "structural", "truncate"), libs)
    assert len(cut.encode()) == len(text.encode()) // 2
    wt = body_for("wttr", ARGS["wttr"])
    typed, _ = apply_degradation("wttr", {}, wt, "30", DegradeSpec(True, "structural", "change_type"), libs)
    assert read_answer("wttr", typed).raw == 30


def test_wikipedia_kind_rule():
    libs = tiny_libraries()
    assert degrade_mod.wikipedia_kind("India", {}, libs) == "country"
    assert degrade_mod.wikipedia_kind("Kolkata", {"coordinates": {"lat": 1}}, libs) == "city"
    assert degrade_mod.wikipedia_kind("Python_(programming_language)", {}, libs) == "other"


# ---------------------------------------------------------------- libraries (A6)

def test_library_hash_check_refuses_an_edited_file(tmp_path):
    libs_src = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    doc = {"items": libs_src, "items_sha256": items_sha256(libs_src)}
    names = ("countriesnow_capitals_v3", "wikipedia_extracts_v3", "jsonplaceholder_todos_v3", "dummyjson_todos_v3")
    for n in names:
        items = [{"country": "X", "capital": "Y"}] if n.startswith("countries") else (
            [{"kind": "city", "key": "Y", "title": "Y", "canonical_title": "Y", "extract": "e"}]
            if n.startswith("wiki") else libs_src)
        (tmp_path / f"{n}.json").write_text(json.dumps({"items": items, "items_sha256": items_sha256(items)}))
    load_libraries(tmp_path, verify_frozen=False)
    edited = json.loads((tmp_path / "dummyjson_todos_v3.json").read_text())
    edited["items"][0]["text"] = "edited"
    (tmp_path / "dummyjson_todos_v3.json").write_text(json.dumps(edited))
    with pytest.raises(RuntimeError):
        load_libraries(tmp_path, verify_frozen=False)
    assert doc  # silence


def test_frozen_libraries_match_their_recorded_hashes():
    assert all(len(v) == 64 for v in FROZEN_ITEMS_SHA256.values()), "libraries not frozen yet"
    libs = load_libraries(DATA_DIR, verify_frozen=True)
    assert len(libs.jsonplaceholder_todos) == 200 and len(libs.dummyjson_todos) == 254
    assert len(libs.capitals) == 250
    assert libs.extracts["city"] and libs.extracts["country"] and libs.extracts["other"]
