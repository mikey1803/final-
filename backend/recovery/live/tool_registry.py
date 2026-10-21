"""Tool-equivalence registry: each live tool's backup (the SWITCH_TOOL peer),
its side-effect flag, and the frozen v2 argument-translation rules.

The peer map is the frozen v1 one (``docs/live_pilot/strategy_definitions.md``,
"Tool-alternatives map"), and ``recovery.live.tools.TOOL_ALTERNATIVES`` stays
the original definition; a test asserts this registry equals it, so the two
can never drift apart. Translation rules and the data behind them come from
``docs/live_pilot/strategy_definitions_v2.md`` §4 (SWITCH_TOOL):

- wikipedia -> countriesnow only for titles in the frozen countriesnow
  country list (``data/countriesnow_countries_v2.json``, never fetched at
  runtime; its content hash is checked on load);
- placeholder ids only for resources that exist on both APIs and ids inside
  the target's verified range;
- wttr -> open_meteo only for ``"lat,lon"`` locations.

Anything else is ``untranslatable_args``: the strategy could not apply, which
analysis keeps separate from "tried and failed".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

__all__ = [
    "ToolSpec", "REGISTRY", "backup_of", "has_side_effect", "validate_registry",
    "Translation", "translate_args", "frozen_country_names", "PLACEHOLDER_RANGES",
]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: str
    backup: str
    side_effect: bool = False  # all six live tools are read-only lookups


REGISTRY: dict[str, ToolSpec] = {
    "wikipedia": ToolSpec("wikipedia", "encyclopedic", "countriesnow"),
    "countriesnow": ToolSpec("countriesnow", "encyclopedic", "wikipedia"),
    "open_meteo": ToolSpec("open_meteo", "weather", "wttr"),
    "wttr": ToolSpec("wttr", "weather", "open_meteo"),
    "jsonplaceholder": ToolSpec("jsonplaceholder", "placeholder", "dummyjson"),
    "dummyjson": ToolSpec("dummyjson", "placeholder", "jsonplaceholder"),
}


def backup_of(tool: str) -> Optional[str]:
    spec = REGISTRY.get(tool)
    return spec.backup if spec else None


def has_side_effect(tool: str) -> bool:
    return REGISTRY[tool].side_effect


def validate_registry(registry: dict[str, ToolSpec] = REGISTRY) -> list[str]:
    """Structural problems with a registry (empty list = valid): self-loops,
    dangling backups, asymmetric pairs, cross-category substitution."""
    problems = []
    for name, spec in registry.items():
        if spec.name != name:
            problems.append(f"{name}: spec.name is {spec.name!r}")
        if spec.backup == name:
            problems.append(f"{name}: self-loop")
        elif spec.backup not in registry:
            problems.append(f"{name}: backup {spec.backup!r} is not a registered tool")
        else:
            peer = registry[spec.backup]
            if peer.backup != name:
                problems.append(f"{name}: asymmetric pair ({spec.backup} backs up to {peer.backup})")
            if peer.category != spec.category:
                problems.append(f"{name}: cross-category backup ({spec.category} -> {peer.category})")
    return problems


# ----------------------------------------------------------------------
# frozen v2 translation data
# ----------------------------------------------------------------------

COUNTRY_LIST_PATH = Path(__file__).resolve().parent / "data" / "countriesnow_countries_v2.json"
COUNTRY_NAMES_SHA256 = "dddab0439c7058d32b875a2cc051c4e6b3b757bc20a5bbf0fbba6bf352c54c15"

# Verified live 2026-10-21 (strategy_definitions_v2.md §4): inclusive id ranges.
PLACEHOLDER_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "jsonplaceholder": {
        "todos": (1, 200), "posts": (1, 100), "comments": (1, 500), "users": (1, 10),
        "albums": (1, 100), "photos": (1, 5000),
    },
    "dummyjson": {
        "todos": (1, 254), "posts": (1, 251), "comments": (1, 340), "users": (1, 208),
        "products": (1, 194), "carts": (1, 208), "quotes": (1, 1454), "recipes": (1, 50),
    },
}
# The resource each live placeholder tool calls (tools.py: /todos/{id}).
TOOL_RESOURCE = {"jsonplaceholder": "todos", "dummyjson": "todos"}

_LATLON = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


@lru_cache(maxsize=1)
def frozen_country_names() -> frozenset[str]:
    """Case-folded names from the frozen list. Raises if the file's content
    does not match the hash recorded at freeze time."""
    doc = json.loads(COUNTRY_LIST_PATH.read_text(encoding="utf-8"))
    names = doc["names"]
    digest = hashlib.sha256(
        json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != COUNTRY_NAMES_SHA256:
        raise RuntimeError(f"{COUNTRY_LIST_PATH.name}: content hash {digest} != frozen {COUNTRY_NAMES_SHA256}")
    return frozenset(_normalise(n).casefold() for n in names)


@dataclass(frozen=True)
class Translation:
    source: str
    target: Optional[str]
    args: Optional[dict]
    ok: bool
    reason: Optional[str] = None  # why untranslatable, human-readable


def _untranslatable(source: str, target: Optional[str], why: str) -> Translation:
    return Translation(source, target, None, False, why)


def _as_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def translate_args(tool: str, args: dict) -> Translation:
    """Translate one tool's args to its registered peer (v2 rules)."""
    target = backup_of(tool)
    if target is None:
        return _untranslatable(tool, None, f"{tool!r} has no registered backup")
    args = args or {}

    if tool == "wikipedia":
        title = args.get("title")
        if not isinstance(title, str) or not title.strip():
            return _untranslatable(tool, target, "missing title")
        country = _normalise(title)
        if country.casefold() not in frozen_country_names():
            return _untranslatable(tool, target, f"title {country!r} is not in the frozen country list")
        return Translation(tool, target, {"country": country}, True)

    if tool == "countriesnow":
        country = args.get("country")
        if not isinstance(country, str) or not country.strip():
            return _untranslatable(tool, target, "missing country")
        return Translation(tool, target, {"title": _normalise(country).replace(" ", "_")}, True)

    if tool == "open_meteo":
        lat, lon = args.get("latitude"), args.get("longitude")
        if isinstance(lat, bool) or isinstance(lon, bool) or not isinstance(lat, (int, float)) \
                or not isinstance(lon, (int, float)):
            return _untranslatable(tool, target, "latitude/longitude missing or not numeric")
        return Translation(tool, target, {"location": f"{lat},{lon}"}, True)

    if tool == "wttr":
        m = _LATLON.match(str(args.get("location", "")))
        if not m:
            return _untranslatable(tool, target, "location is not a 'lat,lon' pair (would need geocoding)")
        return Translation(tool, target, {"latitude": float(m.group(1)), "longitude": float(m.group(2))}, True)

    if tool in TOOL_RESOURCE:
        resource = TOOL_RESOURCE[tool]
        target_range = PLACEHOLDER_RANGES[target].get(resource)
        if target_range is None:
            return _untranslatable(tool, target, f"resource {resource!r} does not exist on {target}")
        item_id = _as_int(args.get("id"))
        if item_id is None:
            return _untranslatable(tool, target, "id missing or not an integer")
        lo, hi = target_range
        if not lo <= item_id <= hi:
            return _untranslatable(tool, target, f"id {item_id} outside {target} {resource} range {lo}-{hi}")
        return Translation(tool, target, {"id": item_id}, True)

    return _untranslatable(tool, target, f"no translation rule for {tool!r}")
