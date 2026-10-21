"""FAARF real-world drift probe (Task 25, contribution C3).

Calls every target in targets.yaml once and appends one JSON line per target
to <out-dir>/YYYY-MM-DD.jsonl (UTC date of the run's start). Records status,
latency, error text, a structural schema hash (keys and types, not values), a
content sanity check, lifecycle headers (Deprecation/Sunset/successor Link)
and, for LLM model-list targets, the sorted model ids.

No response content is stored except model ids -- only structure and
metadata. No retries: a retry would hide the instability being measured.
A failing target never stops the run: every failure, including a bug in this
script's own per-target code, becomes a recorded line.

Usage:
    python probe/run_probe.py [--targets probe/targets.yaml] [--out-dir probe/data] [--only id1,id2]

Exit codes: 0 = run completed (whatever the targets did); 2 = targets.yaml
invalid; 3 = output could not be written.

API keys are read from the environment only (GROQ_API_KEY, GOOGLE_API_KEY)
and are scrubbed from every output line as a second line of defence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_hash import ALGORITHM_VERSION, canonical, hash_schema, schema_of  # noqa: E402

PROBE_VERSION = "1.0.0"
RECORD_VERSION = 1

PROBE_DIR = Path(__file__).resolve().parent
DEFAULT_TARGETS = PROBE_DIR / "targets.yaml"
DEFAULT_OUT_DIR = PROBE_DIR / "data"

AUTH_MODES = {"none", "bearer_env", "goog_header_env"}
JSON_TYPES = {"string", "number", "boolean", "object", "array", "null"}
REQUIRED_FIELDS = ("id", "category", "method", "url", "auth", "terms_url", "expect")
MAX_ERROR_CHARS = 500
LIFECYCLE_HEADERS = ("Deprecation", "Sunset")
REDACTED = "[REDACTED]"


class TargetsError(ValueError):
    """targets.yaml is malformed -- the instrument itself is broken."""


# ----------------------------------------------------------------------
# targets.yaml
# ----------------------------------------------------------------------

def load_targets(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    cfg = yaml.safe_load(raw)
    validate_targets(cfg)
    return cfg, hashlib.sha256(raw).hexdigest()


def validate_targets(cfg: Any) -> None:
    if not isinstance(cfg, dict) or not isinstance(cfg.get("targets"), list) or not cfg["targets"]:
        raise TargetsError("targets.yaml must be a mapping with a non-empty 'targets' list")
    for key in ("user_agent", "timeout_s"):
        if key not in cfg:
            raise TargetsError(f"missing top-level '{key}'")
    seen = set()
    for t in cfg["targets"]:
        missing = [f for f in REQUIRED_FIELDS if f not in t]
        if missing:
            raise TargetsError(f"target {t.get('id', '?')}: missing {missing}")
        if t["id"] in seen:
            raise TargetsError(f"duplicate target id {t['id']}")
        seen.add(t["id"])
        if t["auth"] not in AUTH_MODES:
            raise TargetsError(f"target {t['id']}: unknown auth {t['auth']!r}")
        if t["auth"] != "none" and not t.get("auth_env"):
            raise TargetsError(f"target {t['id']}: auth {t['auth']} needs auth_env")
        for e in t["expect"]:
            if "path" not in e or e.get("type") not in JSON_TYPES:
                raise TargetsError(f"target {t['id']}: bad expect entry {e!r}")


def secret_env_names(cfg: dict) -> list[str]:
    return sorted({t["auth_env"] for t in cfg["targets"] if t.get("auth_env")})


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

_SEGMENT = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def resolve(doc: Any, path: str) -> Any:
    """Dotted path with [n] indices; "" is the root. Raises LookupError if absent."""
    cur = doc
    for name, index in _SEGMENT.findall(path):
        if index:
            if not isinstance(cur, list) or int(index) >= len(cur):
                raise LookupError(path)
            cur = cur[int(index)]
        else:
            if not isinstance(cur, dict) or name not in cur:
                raise LookupError(path)
            cur = cur[name]
    return cur


def json_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if v is None:
        return "null"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return type(v).__name__


def check_expect(doc: Any, expects: list[dict]) -> dict:
    failures = []
    for e in expects:
        path = e["path"]
        try:
            value = resolve(doc, path)
        except LookupError:
            failures.append(f"{path}: missing")
            continue
        actual = json_type(value)
        if actual != e["type"]:
            failures.append(f"{path}: type {actual} != {e['type']}")
            continue
        if "equals" in e and value != e["equals"]:
            failures.append(f"{path}: value differs from expected")
        if "contains" in e and (not isinstance(value, str) or e["contains"] not in value):
            failures.append(f"{path}: does not contain expected substring")
        if e.get("nonempty") and len(value) == 0:
            failures.append(f"{path}: empty")
    return {"evaluated": True, "ok": not failures, "failures": failures}


def _not_evaluated(reason: str) -> dict:
    return {"evaluated": False, "ok": None, "failures": [reason]}


def _utc_iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS] + "...[truncated]"


def _lifecycle(headers: Mapping[str, str]) -> dict:
    out = {h.lower(): headers[h] for h in LIFECYCLE_HEADERS if headers.get(h)}
    link = headers.get("Link")
    if link and ("successor-version" in link or "deprecation" in link or "sunset" in link):
        out["link"] = link
    return out


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, REDACTED)
    return text


def _params(t: dict, cfg: dict) -> dict:
    out = {}
    for k, v in (t.get("params") or {}).items():
        out[k] = v.replace("{contact_email}", cfg.get("contact_email", "")) if isinstance(v, str) else v
    return out


def _headers(t: dict, cfg: dict, env: Mapping[str, str]) -> tuple[dict, Optional[str]]:
    headers = {"User-Agent": cfg["user_agent"], "Accept": "application/json"}
    if t["auth"] == "none":
        return headers, None
    key = env.get(t["auth_env"], "")
    if not key:
        return headers, f"missing secret: {t['auth_env']} not set"
    if t["auth"] == "bearer_env":
        headers["Authorization"] = f"Bearer {key}"
    elif t["auth"] == "goog_header_env":
        headers["x-goog-api-key"] = key  # header, never URL: cannot leak via URLs/errors
    return headers, None


# ----------------------------------------------------------------------
# one target
# ----------------------------------------------------------------------

def _base_record(t: dict, run_id: str, targets_sha: str, started: dt.datetime) -> dict:
    return {
        "record_version": RECORD_VERSION,
        "probe_version": PROBE_VERSION,
        "schema_algorithm": ALGORITHM_VERSION,
        "targets_sha256": targets_sha,
        "run_id": run_id,
        "ts_utc": _utc_iso(started),
        "target": t["id"],
        "category": t["category"],
        "method": t["method"],
        "url": t["url"],
        "http_status": None,
        "latency_ms": None,
        "error": None,
        "redirect_chain": [],
        "final_url": None,
        "content_type": None,
        "body_bytes": None,
        "lifecycle_headers": {},
        "json_ok": False,
        "schema_hash": None,
        "sanity": _not_evaluated("no response"),
        "model_ids": None,
        "model_count": None,
        "model_ids_complete": None,
    }


def probe_target(t: dict, cfg: dict, env: Mapping[str, str], run_id: str, targets_sha: str,
                 now: Callable[[], dt.datetime]) -> tuple[dict, Optional[Any]]:
    """Returns (record, schema). Raises nothing for network/HTTP problems."""
    rec = _base_record(t, run_id, targets_sha, now())
    headers, auth_problem = _headers(t, cfg, env)
    if auth_problem:
        rec["error"] = auth_problem
        rec["sanity"] = _not_evaluated("no request made")
        return rec, None

    params = _params(t, cfg)
    start = time.monotonic()
    try:
        resp = requests.request(t["method"], t["url"], params=params, headers=headers,
                                timeout=cfg["timeout_s"], allow_redirects=True)
    except requests.exceptions.RequestException as exc:
        rec["latency_ms"] = round((time.monotonic() - start) * 1000.0, 1)
        rec["error"] = _clip(f"{type(exc).__name__}: {exc}")
        return rec, None
    rec["latency_ms"] = round((time.monotonic() - start) * 1000.0, 1)

    rec["http_status"] = resp.status_code
    rec["redirect_chain"] = [[h.status_code, h.url] for h in resp.history]
    rec["final_url"] = resp.url
    rec["content_type"] = resp.headers.get("Content-Type")
    rec["body_bytes"] = len(resp.content)
    rec["lifecycle_headers"] = _lifecycle(resp.headers)
    if not resp.ok:
        rec["error"] = _clip(f"HTTP {resp.status_code}: {resp.text}")

    try:
        doc = resp.json()
        rec["json_ok"] = True
    except ValueError:
        if resp.ok:
            rec["error"] = "non-JSON body"
        rec["sanity"] = _not_evaluated("non-JSON body")
        return rec, None

    schema = schema_of(doc, t.get("map_paths") or ())
    rec["schema_hash"] = hash_schema(schema)
    rec["sanity"] = check_expect(doc, t["expect"]) if resp.ok else _not_evaluated(f"HTTP {resp.status_code}")

    if t.get("model_ids") and resp.ok:
        _collect_model_ids(rec, doc, t, params, headers, cfg)
    return rec, schema


def _collect_model_ids(rec: dict, first_doc: Any, t: dict, params: dict, headers: dict, cfg: dict) -> None:
    spec = t["model_ids"]
    ids: set[str] = set()
    complete = True
    doc, pages = first_doc, 1
    while True:
        try:
            for item in resolve(doc, spec["path"]):
                ids.add(str(item[spec["id_field"]]))
        except (LookupError, KeyError, TypeError):
            complete = False
            rec["error"] = rec["error"] or f"model ids not found at {spec['path']}[].{spec['id_field']}"
            break
        token_field = spec.get("next_page_token")
        token = doc.get(token_field) if token_field and isinstance(doc, dict) else None
        if not token:
            break
        if pages >= spec.get("max_pages", 1):
            complete = False
            break
        try:
            resp = requests.request(t["method"], t["url"], params={**params, "pageToken": token},
                                    headers=headers, timeout=cfg["timeout_s"], allow_redirects=True)
            resp.raise_for_status()
            doc = resp.json()
            pages += 1
        except (requests.exceptions.RequestException, ValueError) as exc:
            complete = False
            rec["error"] = _clip(f"page {pages + 1}: {type(exc).__name__}: {exc}")
            break
    rec["model_ids"] = sorted(ids)
    rec["model_count"] = len(ids)
    rec["model_ids_complete"] = complete


def probe_target_safe(t: dict, cfg: dict, env: Mapping[str, str], run_id: str, targets_sha: str,
                      now: Callable[[], dt.datetime]) -> tuple[dict, Optional[Any]]:
    try:
        return probe_target(t, cfg, env, run_id, targets_sha, now)
    except Exception as exc:  # a bug in per-target code must not stop the run
        rec = _base_record(t, run_id, targets_sha, now())
        rec["error"] = _clip(f"probe-internal {type(exc).__name__}: {exc}")
        rec["sanity"] = _not_evaluated("probe-internal error")
        return rec, None


# ----------------------------------------------------------------------
# a whole run
# ----------------------------------------------------------------------

def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run(cfg: dict, targets_sha: str, out_dir: Path, env: Mapping[str, str] = os.environ,
        only: Optional[set[str]] = None, now: Callable[[], dt.datetime] = _utcnow,
        sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    started = now()
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir)
    schema_dir = out_dir / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{started.strftime('%Y-%m-%d')}.jsonl"
    secrets = [env.get(name, "") for name in secret_env_names(cfg)]
    secrets = [s for s in secrets if len(s) >= 8]

    targets = [t for t in cfg["targets"] if not only or t["id"] in only]
    records = []
    for i, t in enumerate(targets):
        if i:
            sleep(cfg.get("pause_s", 0.0))
        rec, schema = probe_target_safe(t, cfg, env, run_id, targets_sha, now)
        line = redact(json.dumps(rec, ensure_ascii=False, sort_keys=True), secrets)
        with open(out_file, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        if schema is not None:
            path = schema_dir / f"{rec['schema_hash']}.json"
            if not path.exists():
                path.write_text(redact(canonical(schema), secrets) + "\n", encoding="utf-8")
        records.append(json.loads(line))
    return records


def summary(records: list[dict]) -> str:
    rows = [f"{'target':<20} {'status':>6} {'ms':>8}  sanity  schema                   error"]
    for r in records:
        sanity = {True: "ok", False: "FAIL", None: "-"}[r["sanity"]["ok"]]
        err = (r["error"] or "")[:60]
        rows.append(f"{r['target']:<20} {str(r['http_status']):>6} {str(r['latency_ms']):>8}  "
                    f"{sanity:<6}  {str(r['schema_hash']):<24} {err}")
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--only", help="comma-separated target ids")
    args = ap.parse_args(argv)

    try:
        cfg, sha = load_targets(args.targets)
    except (OSError, yaml.YAMLError, TargetsError) as exc:
        print(f"FATAL: cannot load targets: {exc}", file=sys.stderr)
        return 2
    only = set(args.only.split(",")) if args.only else None
    try:
        records = run(cfg, sha, args.out_dir, os.environ, only)
    except OSError as exc:
        print(f"FATAL: cannot write output: {exc}", file=sys.stderr)
        return 3
    print(summary(records))
    n_fail = sum(1 for r in records if r["error"] or r["sanity"]["ok"] is False)
    print(f"\n{len(records)} targets probed, {n_fail} with an error or sanity failure; targets_sha256={sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
