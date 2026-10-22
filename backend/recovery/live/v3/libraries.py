"""The frozen v3 plausible-wrong libraries (v3 §3.6).

Captured once from the real services by
``experiments/capture_v3_libraries.py`` and frozen with v3. Each file's
``items`` hash is checked on load against ``FROZEN_ITEMS_SHA256``, so a
re-captured or edited file is refused rather than silently used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

__all__ = ["Libraries", "load_libraries", "DATA_DIR", "FROZEN_ITEMS_SHA256", "items_sha256"]

DATA_DIR = Path(__file__).resolve().parent / "data"

# Captured 2026-10-21 (UTC) by experiments/capture_v3_libraries.py.
FROZEN_ITEMS_SHA256: dict[str, str] = {
    "countriesnow_capitals_v3": "5c95359d69ac1fb74caf6f7129122e336881886e27a684e4e6f355a23dc95b51",
    "wikipedia_extracts_v3": "a5665045519723d98d9efef2524e9b0170db066ebdaf43cbb7f3e7ffb85fd2c9",
    "jsonplaceholder_todos_v3": "5dee672ec564c6adc265c3a3a8987262632a53fff7e5a9dd35139901792758a9",
    "dummyjson_todos_v3": "09e4ed56f0e81718a437deb4d1be619246efd6592349cfcfc5510385caa93da6",
}


def items_sha256(items) -> str:
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Libraries:
    capitals: tuple                 # ({"country", "capital"}, ...) in the countriesnow order
    extracts: dict                  # kind ("city" | "country" | "other") -> tuple of rows
    jsonplaceholder_todos: dict     # id -> title
    dummyjson_todos: dict           # id -> todo text
    hashes: dict                    # library name -> items_sha256 (as verified)


def _load(path: Path, expected: Optional[str]) -> list:
    doc = json.loads(path.read_text(encoding="utf-8"))
    digest = items_sha256(doc["items"])
    if expected is not None and digest != expected:
        raise RuntimeError(f"{path.name}: items hash {digest} != frozen {expected}")
    if digest != doc.get("items_sha256"):
        raise RuntimeError(f"{path.name}: items hash {digest} != the hash recorded in the file")
    return doc["items"]


def build_libraries(capitals: list, extracts: list, jp: list, dj: list, hashes: Optional[dict] = None) -> Libraries:
    by_kind: dict[str, list] = {"city": [], "country": [], "other": []}
    for row in extracts:
        by_kind[row["kind"]].append(row)
    return Libraries(
        capitals=tuple(capitals),
        extracts={k: tuple(v) for k, v in by_kind.items()},
        jsonplaceholder_todos={int(r["id"]): r["text"] for r in jp},
        dummyjson_todos={int(r["id"]): r["text"] for r in dj},
        hashes=dict(hashes or {}),
    )


def load_libraries(data_dir: Path = DATA_DIR, verify_frozen: bool = True) -> Libraries:
    """Load and hash-check all four libraries. ``verify_frozen=False`` is for
    tests and smoke captures only; collection always verifies."""
    names = ("countriesnow_capitals_v3", "wikipedia_extracts_v3", "jsonplaceholder_todos_v3", "dummyjson_todos_v3")
    loaded, hashes = {}, {}
    for name in names:
        expected = FROZEN_ITEMS_SHA256[name] if verify_frozen else None
        loaded[name] = _load(Path(data_dir) / f"{name}.json", expected)
        hashes[name] = items_sha256(loaded[name])
    return build_libraries(loaded["countriesnow_capitals_v3"], loaded["wikipedia_extracts_v3"],
                           loaded["jsonplaceholder_todos_v3"], loaded["dummyjson_todos_v3"], hashes)


@lru_cache(maxsize=1)
def frozen_libraries() -> Libraries:
    return load_libraries(DATA_DIR, verify_frozen=True)
