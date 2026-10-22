"""``degraded_output`` transformations (v3 §3.4, §3.6).

The real response is received first; then the answer field is modified.

(a) structural: ``remove_field`` | ``change_type`` (string <-> number) | ``truncate`` (body cut at 50% of its bytes)
(b) ``plausible_wrong``: a wrong value of the same kind -- a library entry ``k``
    positions after the true one (wrapping, skipping entries equal to the
    true value), or a temperature shifted by ``delta`` with ``|delta|`` in
    [8, 15] °C and a random sign (configuration choices, v3 §3.6).

A key's ``DegradeSpec`` is drawn once, with a FIXED number of draws from the
``degradation`` stream whatever is chosen, so one key's outcome never shifts
the draws of the next key.

Family split: (a) and (b) are drawn with probability 1/2 each, then one of
the three structural transformations uniformly. The frozen v3 text does not
fix this split; it is an implementation choice, recorded on every record
(``family``) and reported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from recovery.live.v3.answers import ANSWER_FIELDS, normalise_string
from recovery.live.v3.libraries import Libraries

__all__ = ["DegradeSpec", "draw_spec", "apply_degradation", "wikipedia_kind", "STRUCTURAL", "DELTA_RANGE_C"]

STRUCTURAL = ("remove_field", "change_type", "truncate")
DELTA_RANGE_C = (8.0, 15.0)    # configuration choice (v3 §3.6)
P_PLAUSIBLE_FAMILY = 0.5       # implementation choice, see module docstring
_K_MAX = 1_000_000             # k >= 1; positions wrap modulo the library length


@dataclass(frozen=True)
class DegradeSpec:
    degrade: bool
    family: Optional[str] = None           # "structural" | "plausible_wrong"
    transformation: Optional[str] = None   # remove_field | change_type | truncate | plausible_wrong
    k: Optional[int] = None
    delta: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


def draw_spec(rng: np.random.Generator, rate: float) -> DegradeSpec:
    u_degrade = float(rng.random())
    u_family = float(rng.random())
    i_struct = int(rng.integers(len(STRUCTURAL)))
    k = int(rng.integers(1, _K_MAX))
    magnitude = float(rng.uniform(*DELTA_RANGE_C))
    sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    if u_degrade >= rate:
        return DegradeSpec(False)
    if u_family < P_PLAUSIBLE_FAMILY:
        return DegradeSpec(True, "plausible_wrong", "plausible_wrong", k=k, delta=round(sign * magnitude, 4))
    return DegradeSpec(True, "structural", STRUCTURAL[i_struct])


# ----------------------------------------------------------------------

def _dumps_like(original: str, doc: Any) -> str:
    if "\n" in original:
        return json.dumps(doc, ensure_ascii=False, indent=4)
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def _parent_and_key(doc: Any, path: tuple):
    cur = doc
    for step in path[:-1]:
        cur = cur[step]
    return cur, path[-1]


def _step_from(items: list, start: int, k: int, is_same) -> tuple[int, Any]:
    """Entry ``k`` positions after ``start`` (wrapping), skipping entries equal to the true value."""
    n = len(items)
    if n == 0:
        raise ValueError("empty library")
    idx = (start + k) % n
    for _ in range(n):
        if not is_same(items[idx]):
            return idx, items[idx]
        idx = (idx + 1) % n
    raise ValueError("every library entry equals the true value")


def wikipedia_kind(title: str, genuine_doc: dict, libs: Libraries) -> str:
    """country if the title is in the frozen capitals' country list; city if
    Wikipedia geolocates the article (``coordinates`` present); else other."""
    t = normalise_string(str(title).replace("_", " "))
    if any(normalise_string(c["country"]) == t for c in libs.capitals):
        return "country"
    return "city" if isinstance(genuine_doc.get("coordinates"), dict) else "other"


def _plausible(tool: str, args: dict, doc: dict, genuine_raw: Any, spec: DegradeSpec, libs: Libraries) -> tuple[Any, dict]:
    k, delta = spec.k, spec.delta
    if tool == "open_meteo":
        return round(float(genuine_raw) + delta, 1), {"delta": delta}
    if tool == "wttr":
        return str(round(int(genuine_raw) + delta)), {"delta": delta}
    if tool == "countriesnow":
        rows = [c for c in libs.capitals if str(c.get("capital", "")).strip()]
        true_country = normalise_string(str(args.get("country", "")))
        g = normalise_string(str(genuine_raw))
        start = next((i for i, c in enumerate(rows) if normalise_string(c["country"]) == true_country), None)
        if start is None:
            start = next((i for i, c in enumerate(rows) if normalise_string(c["capital"]) == g), -1)
        idx, row = _step_from(rows, start, k, lambda c: normalise_string(c["capital"]) == g)
        return row["capital"], {"k": k, "library_index": idx, "library_key": row["country"]}
    if tool == "wikipedia":
        kind = wikipedia_kind(args.get("title", ""), doc, libs)
        rows = list(libs.extracts[kind])
        g = normalise_string(str(genuine_raw))
        names = {normalise_string(str(args.get("title", "")).replace("_", " ")),
                 normalise_string(str(doc.get("title", "")))}
        start = next((i for i, r in enumerate(rows)
                      if normalise_string(str(r["key"]).replace("_", " ")) in names
                      or normalise_string(str(r.get("canonical_title") or "")) in names), -1)
        idx, row = _step_from(rows, start, k, lambda r: normalise_string(r["extract"]) == g)
        return row["extract"], {"k": k, "wikipedia_kind": kind, "library_index": idx, "library_key": row["key"]}
    if tool in ("jsonplaceholder", "dummyjson"):
        lib = libs.jsonplaceholder_todos if tool == "jsonplaceholder" else libs.dummyjson_todos
        ids = sorted(lib)
        true_id = doc.get("id") if isinstance(doc.get("id"), int) else None
        start = ids.index(true_id) if true_id in lib else -1
        g = normalise_string(str(genuine_raw))
        idx, todo_id = _step_from(ids, start, k, lambda i: normalise_string(lib[i]) == g)
        return lib[todo_id], {"k": k, "library_id": todo_id}
    raise ValueError(f"no plausible-wrong rule for {tool!r}")


def apply_degradation(tool: str, args: dict, genuine_text: str, genuine_raw: Any,
                      spec: DegradeSpec, libs: Libraries) -> tuple[str, dict]:
    """(delivered body, detail). Raises ValueError when the transformation
    cannot be applied to this body (recorded by the caller, body left genuine)."""
    if spec.transformation == "truncate":
        raw = genuine_text.encode("utf-8")
        return raw[: len(raw) // 2].decode("utf-8", errors="ignore"), {"truncated_to_bytes": len(raw) // 2}
    doc = json.loads(genuine_text)
    parent, key = _parent_and_key(doc, ANSWER_FIELDS[tool])
    detail: dict = {}
    if spec.transformation == "remove_field":
        del parent[key]
    elif spec.transformation == "change_type":
        value = parent[key]
        if isinstance(value, str):
            s = value.strip()
            try:
                parent[key] = int(s) if s.lstrip("-").isdigit() else float(s)
            except ValueError:
                parent[key] = 0
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parent[key] = str(value)
        else:
            raise ValueError(f"change_type: unsupported value type {type(value).__name__}")
        detail["new_type"] = type(parent[key]).__name__
    elif spec.transformation == "plausible_wrong":
        parent[key], detail = _plausible(tool, args, doc, genuine_raw, spec, libs)
    else:
        raise ValueError(f"unknown transformation {spec.transformation!r}")
    return _dumps_like(genuine_text, doc), detail
