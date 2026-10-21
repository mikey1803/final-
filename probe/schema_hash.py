"""Structural fingerprint of a JSON document: keys and types, never values.

Frozen as part of docs/experiments/drift_atlas_preregistration.md (v1). Any
change to this algorithm changes every hash, so it is a new instrument version,
not an edit.

Rules:
- object  -> {"object": {key: schema(value)}}           (keys sorted by canonical JSON)
- array   -> {"array": merge(schema(item) for item)}    (length ignored)
             {"array": "empty"} for []
- bool    -> "boolean"   (checked before numbers: bool is an int in Python)
- int/float -> "number"  (84612 -> 84612.5 is a value change, not drift)
- str     -> "string";  None -> "null"
- an object at a path listed in map_paths -> {"map": merge(schema(v) for v)}
  ({"map": "empty"} if it has no keys): its keys are data (e.g. currency
  codes), so adding a key is not schema drift.

merge() unions item schemas: objects merge key-wise (a key present in any
item is present in the result, its schemas merged recursively), arrays merge
their item schemas, maps merge their value schemas, and anything still
distinct becomes {"anyOf": [...]} sorted canonically. The result depends only
on the SET of item schemas, never on order or multiplicity.

Paths in map_paths are dotted from the root; "[]" stands for "any array
item", e.g. "data.rates" or "results[].rates".

hash = sha256(canonical JSON of the schema)[:24] -- the same
sort_keys + sha256[:24] pattern as recovery/core/parallel.py and
recovery/live/event_schema.py.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

ALGORITHM_VERSION = 1


def canonical(schema: Any) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _join(path: str, key: str) -> str:
    return key if not path else f"{path}.{key}"


def schema_of(value: Any, map_paths: Iterable[str] = (), _path: str = "") -> Any:
    map_paths = frozenset(map_paths)
    return _schema(value, map_paths, _path)


def _schema(value: Any, map_paths: frozenset, path: str) -> Any:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, dict):
        if path in map_paths:
            if not value:
                return {"map": "empty"}
            return {"map": merge(_schema(v, map_paths, f"{path}.*") for v in value.values())}
        return {"object": {k: _schema(v, map_paths, _join(path, k)) for k, v in sorted(value.items())}}
    if isinstance(value, list):
        if not value:
            return {"array": "empty"}
        return {"array": merge(_schema(v, map_paths, f"{path}[]") for v in value)}
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def _flatten(schemas: Iterable[Any]) -> list:
    out = []
    for s in schemas:
        if isinstance(s, dict) and "anyOf" in s:
            out.extend(_flatten(s["anyOf"]))
        else:
            out.append(s)
    return out


def merge(schemas: Iterable[Any]) -> Any:
    items = _flatten(schemas)
    objects = [s["object"] for s in items if isinstance(s, dict) and "object" in s]
    arrays = [s["array"] for s in items if isinstance(s, dict) and "array" in s]
    maps = [s["map"] for s in items if isinstance(s, dict) and "map" in s]
    scalars = {canonical(s): s for s in items if not isinstance(s, dict)}

    merged = list(scalars.values())
    if objects:
        keys = sorted(set().union(*objects))
        merged.append({"object": {k: merge(o[k] for o in objects if k in o) for k in keys}})
    if arrays:
        inner = [a for a in arrays if a != "empty"]
        merged.append({"array": merge(inner) if inner else "empty"})
    if maps:
        inner = [m for m in maps if m != "empty"]
        merged.append({"map": merge(inner) if inner else "empty"})

    if len(merged) == 1:
        return merged[0]
    return {"anyOf": sorted(merged, key=canonical)}


def schema_hash(value: Any, map_paths: Iterable[str] = ()) -> str:
    return hash_schema(schema_of(value, map_paths))


def hash_schema(schema: Any) -> str:
    return hashlib.sha256(canonical(schema).encode("utf-8")).hexdigest()[:24]
