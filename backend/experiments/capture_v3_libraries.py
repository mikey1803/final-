#!/usr/bin/env python
"""Capture the v3 plausible-wrong libraries ONCE from the real services
(docs/live_pilot/strategy_definitions_v3.md §3.6) and write them, with their
content hashes, to backend/recovery/live/v3/data/.

These files are frozen with v3: once committed they are never re-fetched or
edited (rule 3). ``recovery.live.v3.libraries`` checks each file's content hash
on load against the constant recorded there, so a later re-capture can never
silently replace a frozen library.

    python experiments/capture_v3_libraries.py --out-dir recovery/live/v3/data
    python experiments/capture_v3_libraries.py --smoke --out-dir <tmp>   # 5 items per library

Sequential, one request at a time, 0.2 s apart, project User-Agent. This is a
one-off library capture, not a measurement, so a transient network error or
HTTP 429/5xx is retried up to 3 times (2, 4, 8 s backoff); Wikipedia lookups
are cached in ``<out-dir>/.wiki_cache.json`` so an interrupted capture resumes.
Plus the documented disambiguation fallback for capital titles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.live.tools import _USER_AGENT  # noqa: E402

HEADERS = {"User-Agent": _USER_AGENT}
PAUSE_S = 0.2
TIMEOUT_S = 20.0
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
CAPITALS_URL = "https://countriesnow.space/api/v0.1/countries/capital"
JP_TODOS_URL = "https://jsonplaceholder.typicode.com/todos"
DJ_TODOS_URL = "https://dummyjson.com/todos?limit=0"

# Non-geolocated ("other") titles: the one non-city task title
# (Python_(programming_language), the probe/health-check title) plus other
# articles of the same kind (programming languages).
OTHER_TITLES = [
    "Python_(programming_language)", "Java_(programming_language)", "C_(programming_language)",
    "C++", "JavaScript", "Rust_(programming_language)", "Go_(programming_language)",
    "Ruby_(programming_language)", "Haskell", "Perl",
]


def content_sha256(items) -> str:
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _get(url: str, params: Optional[dict] = None) -> requests.Response:
    for attempt in range(4):
        time.sleep(PAUSE_S)
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
        except requests.exceptions.RequestException:
            if attempt == 3:
                raise
        else:
            if r.status_code != 429 and r.status_code < 500:
                return r
            if attempt == 3:
                return r
        time.sleep(2.0 * 2 ** attempt)
    raise AssertionError("unreachable")


_WIKI_CACHE: dict = {}
_WIKI_CACHE_PATH: Optional[Path] = None


def wiki_summary(title: str) -> Optional[dict]:
    if title in _WIKI_CACHE:
        return _WIKI_CACHE[title]
    r = _get(WIKI_SUMMARY.format(quote(title, safe="(),_")))
    d = r.json() if r.ok else None
    out = d if d is not None and isinstance(d.get("extract"), str) and d["extract"].strip() else None
    if out is not None:
        out = {k: out.get(k) for k in ("type", "title", "extract")}
    _WIKI_CACHE[title] = out
    if _WIKI_CACHE_PATH is not None:
        _WIKI_CACHE_PATH.write_text(json.dumps(_WIKI_CACHE, ensure_ascii=False), encoding="utf-8")
    return out


def capture_capitals(limit: Optional[int]) -> list[dict]:
    data = _get(CAPITALS_URL).json()["data"]
    rows = [{"country": d["name"], "capital": d.get("capital") or ""} for d in data]
    return rows[:limit] if limit else rows


def capture_extracts(capitals: list[dict], limit: Optional[int]) -> list[dict]:
    """One row per article: kind (city | country | other), the title queried,
    the canonical title returned, and the extract."""
    rows: list[dict] = []
    cities = [c for c in capitals if c["capital"].strip()]
    for c in cities[:limit] if limit else cities:
        title = c["capital"].replace(" ", "_")
        s = wiki_summary(title)
        if s is not None and s.get("type") == "disambiguation":
            # documented fallback: "<capital>, <country>" disambiguates e.g. Kingston / Victoria
            title = f"{c['capital']},_{c['country']}".replace(" ", "_")
            s = wiki_summary(title)
        if s is not None and s.get("type") != "disambiguation":
            rows.append({"kind": "city", "key": c["capital"], "title": title,
                         "canonical_title": s.get("title"), "extract": s["extract"]})
    for c in capitals[:limit] if limit else capitals:
        title = c["country"].replace(" ", "_")
        s = wiki_summary(title)
        if s is not None and s.get("type") != "disambiguation":
            rows.append({"kind": "country", "key": c["country"], "title": title,
                         "canonical_title": s.get("title"), "extract": s["extract"]})
    for title in OTHER_TITLES[:limit] if limit else OTHER_TITLES:
        s = wiki_summary(title)
        if s is not None:
            rows.append({"kind": "other", "key": title, "title": title,
                         "canonical_title": s.get("title"), "extract": s["extract"]})
    return rows


def capture_todos(url: str, field: str, limit: Optional[int]) -> list[dict]:
    doc = _get(url).json()
    items = doc["todos"] if isinstance(doc, dict) else doc
    rows = [{"id": int(t["id"]), "text": t[field]} for t in items]
    rows.sort(key=lambda r: r["id"])
    return rows[:limit] if limit else rows


def write(out_dir: Path, name: str, source: str, items: list[dict]) -> str:
    digest = content_sha256(items)
    doc = {"library": name, "source": source, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
           "n_items": len(items), "items_sha256": digest, "items": items}
    (out_dir / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{name}: {len(items)} items, items_sha256={digest}")
    return digest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="5 items per library (never the frozen files)")
    args = ap.parse_args()
    limit = 5 if args.smoke else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global _WIKI_CACHE_PATH
    _WIKI_CACHE_PATH = args.out_dir / ".wiki_cache.json"
    if _WIKI_CACHE_PATH.exists():
        _WIKI_CACHE.update(json.loads(_WIKI_CACHE_PATH.read_text(encoding="utf-8")))

    capitals = capture_capitals(limit)
    write(args.out_dir, "countriesnow_capitals_v3", CAPITALS_URL, capitals)
    write(args.out_dir, "jsonplaceholder_todos_v3", JP_TODOS_URL, capture_todos(JP_TODOS_URL, "title", limit))
    write(args.out_dir, "dummyjson_todos_v3", DJ_TODOS_URL, capture_todos(DJ_TODOS_URL, "todo", limit))
    write(args.out_dir, "wikipedia_extracts_v3", WIKI_SUMMARY, capture_extracts(capitals, limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
