"""Tests for the real-world drift probe (probe/, Task 25 C3).

No real network calls: requests.request is monkeypatched, as in
test_live_tools.py. All output goes to tmp_path (conftest write guard).
"""

import datetime as dt
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "probe"))

import requests  # noqa: E402

import run_probe  # noqa: E402
from schema_hash import merge, schema_hash, schema_of  # noqa: E402

TARGETS_PATH = REPO_ROOT / "probe" / "targets.yaml"
TOOLS_SOURCE = REPO_ROOT / "backend" / "recovery" / "live" / "tools.py"
FIXED_NOW = dt.datetime(2026, 10, 21, 18, 17, 3, 456000, tzinfo=dt.timezone.utc)
FAKE_KEY = "gsk_FAKEFAKEFAKEFAKE1234567890"


# ----------------------------------------------------------------------
# schema hashing
# ----------------------------------------------------------------------

BASE_DOC = {
    "id": 1, "title": "hello", "done": False, "score": 3.5, "note": None,
    "tags": ["a", "b"], "owner": {"name": "x", "age": 30},
    "items": [{"k": 1, "v": "p"}, {"k": 2, "v": "q"}],
}


def _h(doc, map_paths=()):
    return schema_hash(doc, map_paths)


@pytest.mark.parametrize("variant", [
    {**BASE_DOC, "title": "a completely different string"},
    {**BASE_DOC, "id": 999999, "score": -1e9},
    {**BASE_DOC, "done": True},
    {**BASE_DOC, "tags": ["only-one"]},
    {**BASE_DOC, "tags": ["a", "b", "c", "d", "e"]},
    {**BASE_DOC, "items": [{"k": 7, "v": "z"}] * 10},
    {**BASE_DOC, "owner": {"age": 1, "name": "reordered"}},
    dict(reversed(list(BASE_DOC.items()))),
])
def test_schema_hash_stable_under_value_changes(variant):
    assert _h(variant) == _h(BASE_DOC)


def test_int_and_float_are_both_number():
    assert _h({"p": 84612}) == _h({"p": 84612.5})


def test_bool_is_not_number():
    assert _h({"p": True}) != _h({"p": 1})


@pytest.mark.parametrize("variant", [
    {**BASE_DOC, "extra": 1},                                          # key added
    {k: v for k, v in BASE_DOC.items() if k != "note"},                # key removed
    {("heading" if k == "title" else k): v for k, v in BASE_DOC.items()},  # key renamed
    {**BASE_DOC, "title": 5},                                          # string -> number
    {**BASE_DOC, "owner": [{"name": "x", "age": 30}]},                 # object -> array
    {**BASE_DOC, "title": None},                                       # value -> null
    {**BASE_DOC, "owner": {"name": {"first": "x"}, "age": 30}},        # nesting change
    {**BASE_DOC, "items": [{"k": 1, "v": "p", "w": 0}]},               # key added inside array items
    {**BASE_DOC, "tags": []},                                          # non-empty -> empty array
])
def test_schema_hash_changes_under_key_or_type_changes(variant):
    assert _h(variant) != _h(BASE_DOC)


def test_map_paths_ignore_new_keys_but_not_type_changes():
    rates = {"base": "EUR", "rates": {"USD": 1.1, "GBP": 0.8}}
    more = {"base": "EUR", "rates": {"USD": 1.1, "GBP": 0.8, "INR": 95.2}}
    as_strings = {"base": "EUR", "rates": {"USD": "1.1", "GBP": "0.8"}}
    assert _h(rates, ["rates"]) == _h(more, ["rates"])
    assert _h(rates, ["rates"]) != _h(as_strings, ["rates"])
    assert _h(rates) != _h(more)  # without map_paths, a new key IS a schema change


def test_merge_is_order_and_multiplicity_independent():
    a, b = schema_of({"x": 1}), schema_of({"y": "s"})
    assert merge([a, b]) == merge([b, a, b, a])
    assert _h([1, "s", None]) == _h([None, None, "s", 1, 1])


# ----------------------------------------------------------------------
# targets.yaml
# ----------------------------------------------------------------------

def test_targets_yaml_is_valid_and_complete():
    cfg, sha = run_probe.load_targets(TARGETS_PATH)
    ids = [t["id"] for t in cfg["targets"]]
    assert len(ids) == 24 and len(set(ids)) == 24
    assert len(sha) == 64
    assert cfg["timeout_s"] == 10
    assert "piomicheal17@gmail.com" in cfg["user_agent"]
    assert {t["category"] for t in cfg["targets"]} == {
        "encyclopedic", "weather", "placeholder", "search", "geo", "finance", "llm_models"}


def test_live_pilot_targets_match_live_tools_source():
    cfg, _ = run_probe.load_targets(TARGETS_PATH)
    source = TOOLS_SOURCE.read_text(encoding="utf-8")
    pilot = [t for t in cfg["targets"] if t.get("live_pilot")]
    assert {t["id"] for t in pilot} == {
        "wikipedia", "countriesnow", "open_meteo", "wttr", "jsonplaceholder", "dummyjson"}
    for t in pilot:
        assert urlsplit(t["url"]).netloc in source, t["id"]


def test_invalid_targets_rejected():
    with pytest.raises(run_probe.TargetsError):
        run_probe.validate_targets({"user_agent": "x", "timeout_s": 1,
                                    "targets": [{"id": "a"}]})


# ----------------------------------------------------------------------
# run behaviour (mocked network)
# ----------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, body=None, text=None, headers=None, url="https://example.invalid/x"):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text if text is not None else json.dumps(body)
        self.content = self.text.encode("utf-8")
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.url = url
        self.history = []

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _cfg(*targets):
    return {"user_agent": "test-agent/1 (test@example.invalid)", "timeout_s": 10, "pause_s": 0,
            "contact_email": "test@example.invalid", "targets": list(targets)}


def _target(tid, url=None, **extra):
    return {"id": tid, "category": "test", "method": "GET", "url": url or f"https://{tid}.invalid/x",
            "auth": "none", "terms_url": "https://example.invalid/terms",
            "expect": [{"path": "ok", "type": "boolean", "equals": True}], **extra}


def _run(cfg, tmp_path, env=None):
    return run_probe.run(cfg, "0" * 64, tmp_path, env=env or {}, now=lambda: FIXED_NOW, sleep=lambda s: None)


def _lines(tmp_path):
    return [json.loads(l) for l in (tmp_path / "2026-10-21.jsonl").read_text(encoding="utf-8").splitlines()]


def test_failing_targets_are_recorded_not_raised(monkeypatch, tmp_path):
    def fake_request(method, url, **kw):
        host = urlsplit(url).netloc
        if host == "timeout.invalid":
            raise requests.exceptions.Timeout("read timed out")
        if host == "dns.invalid":
            raise requests.exceptions.ConnectionError("name resolution failed")
        if host == "boom.invalid":
            raise RuntimeError("unexpected bug")
        if host == "http500.invalid":
            return _FakeResponse(500, text="internal error")
        if host == "html.invalid":
            return _FakeResponse(200, text="<html>not json</html>", headers={"Content-Type": "text/html"})
        if host == "badpath.invalid":
            return _FakeResponse(200, {"something_else": 1})
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(run_probe.requests, "request", fake_request)
    cfg = _cfg(_target("timeout"), _target("dns"), _target("boom"), _target("http500"),
               _target("html"), _target("badpath"),
               _target("needskey", auth="bearer_env", auth_env="GROQ_API_KEY"),
               _target("good"))
    records = _run(cfg, tmp_path, env={})
    by_id = {r["target"]: r for r in records}

    assert [r["target"] for r in _lines(tmp_path)] == [t["id"] for t in cfg["targets"]]
    assert "Timeout" in by_id["timeout"]["error"] and by_id["timeout"]["http_status"] is None
    assert "ConnectionError" in by_id["dns"]["error"]
    assert "probe-internal RuntimeError" in by_id["boom"]["error"]
    assert by_id["http500"]["http_status"] == 500 and "HTTP 500" in by_id["http500"]["error"]
    assert by_id["http500"]["sanity"]["evaluated"] is False
    assert by_id["html"]["json_ok"] is False and by_id["html"]["schema_hash"] is None
    assert by_id["badpath"]["sanity"] == {"evaluated": True, "ok": False, "failures": ["ok: missing"]}
    assert by_id["needskey"]["error"] == "missing secret: GROQ_API_KEY not set"
    assert by_id["good"]["error"] is None and by_id["good"]["sanity"]["ok"] is True


def test_main_exits_zero_even_when_every_target_fails(monkeypatch, tmp_path):
    def always_timeout(*a, **k):
        raise requests.exceptions.Timeout("nope")
    monkeypatch.setattr(run_probe.requests, "request", always_timeout)
    monkeypatch.setattr(run_probe.time, "sleep", lambda s: None)
    assert run_probe.main(["--out-dir", str(tmp_path), "--only", "httpbin_json,wikipedia"]) == 0
    lines = list(tmp_path.glob("*.jsonl"))[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all("Timeout" in json.loads(l)["error"] for l in lines)


def test_main_exits_2_on_invalid_targets(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("targets: []\n", encoding="utf-8")
    assert run_probe.main(["--targets", str(bad), "--out-dir", str(tmp_path)]) == 2


def test_secrets_never_reach_output(monkeypatch, tmp_path):
    seen_headers = {}

    def fake_request(method, url, headers=None, **kw):
        seen_headers.update(headers)
        # a hostile server echoing the credential back in an error body
        return _FakeResponse(401, text=f"invalid key {FAKE_KEY}")

    monkeypatch.setattr(run_probe.requests, "request", fake_request)
    cfg = _cfg(_target("groq_models", auth="bearer_env", auth_env="GROQ_API_KEY",
                       model_ids={"path": "data", "id_field": "id"}))
    _run(cfg, tmp_path, env={"GROQ_API_KEY": FAKE_KEY})
    assert seen_headers["Authorization"] == f"Bearer {FAKE_KEY}"  # key was actually used
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert FAKE_KEY not in f.read_text(encoding="utf-8")
    assert run_probe.REDACTED in _lines(tmp_path)[0]["error"]


def test_gemini_key_goes_in_header_not_url(monkeypatch, tmp_path):
    calls = []

    def fake_request(method, url, params=None, headers=None, **kw):
        calls.append((url, dict(params or {}), dict(headers or {})))
        return _FakeResponse(200, {"models": [{"name": "models/b"}, {"name": "models/a"}]})

    monkeypatch.setattr(run_probe.requests, "request", fake_request)
    cfg = _cfg(_target("gemini_models", auth="goog_header_env", auth_env="GOOGLE_API_KEY",
                       expect=[{"path": "models", "type": "array", "nonempty": True}],
                       model_ids={"path": "models", "id_field": "name"}))
    rec = _run(cfg, tmp_path, env={"GOOGLE_API_KEY": FAKE_KEY})[0]
    url, params, headers = calls[0]
    assert FAKE_KEY not in url and FAKE_KEY not in json.dumps(params)
    assert headers["x-goog-api-key"] == FAKE_KEY
    assert rec["model_ids"] == ["models/a", "models/b"] and rec["model_count"] == 2


def test_model_id_pagination(monkeypatch, tmp_path):
    pages = {None: {"models": [{"name": "m1"}], "nextPageToken": "t2"},
             "t2": {"models": [{"name": "m2"}], "nextPageToken": "t3"},
             "t3": {"models": [{"name": "m0"}]}}

    def fake_request(method, url, params=None, **kw):
        return _FakeResponse(200, pages[(params or {}).get("pageToken")])

    monkeypatch.setattr(run_probe.requests, "request", fake_request)
    cfg = _cfg(_target("gem", expect=[{"path": "models", "type": "array"}],
                       model_ids={"path": "models", "id_field": "name",
                                  "next_page_token": "nextPageToken", "max_pages": 5}))
    rec = _run(cfg, tmp_path)[0]
    assert rec["model_ids"] == ["m0", "m1", "m2"] and rec["model_ids_complete"] is True


def test_output_file_appends_and_schema_store_written_once(monkeypatch, tmp_path):
    monkeypatch.setattr(run_probe.requests, "request", lambda *a, **k: _FakeResponse(200, {"ok": True}))
    cfg = _cfg(_target("good"))
    first = _run(cfg, tmp_path)[0]
    schema_file = tmp_path / "schemas" / f"{first['schema_hash']}.json"
    mtime = schema_file.stat().st_mtime_ns
    _run(cfg, tmp_path)
    assert len(_lines(tmp_path)) == 2                         # appended, not overwritten
    assert schema_file.stat().st_mtime_ns == mtime            # content-addressed: written once
    assert len(list((tmp_path / "schemas").iterdir())) == 1
    assert first["ts_utc"] == "2026-10-21T18:17:03.456Z" and first["run_id"] == "20261021T181703Z"


def test_lifecycle_headers_recorded(monkeypatch, tmp_path):
    hdrs = {"Deprecation": "@1779103800", "Link": '<https://x.invalid/v2>; rel="successor-version"',
            "Server": "irrelevant"}
    monkeypatch.setattr(run_probe.requests, "request", lambda *a, **k: _FakeResponse(200, {"ok": True}, headers=hdrs))
    rec = _run(_cfg(_target("dep")), tmp_path)[0]
    assert rec["lifecycle_headers"] == {"deprecation": "@1779103800",
                                        "link": '<https://x.invalid/v2>; rel="successor-version"'}


@pytest.mark.parametrize("path,doc,expected", [
    ("a.b", {"a": {"b": 1}}, 1),
    ("[0].base", [{"base": "EUR"}], "EUR"),
    ("places[0].place name", {"places": [{"place name": "Beverly Hills"}]}, "Beverly Hills"),
    ("", {"x": 1}, {"x": 1}),
])
def test_resolve(path, doc, expected):
    assert run_probe.resolve(doc, path) == expected


def test_resolve_missing_raises_lookup_error():
    with pytest.raises(LookupError):
        run_probe.resolve({"a": [1]}, "a[3]")
