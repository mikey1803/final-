"""Per-tool answer fields, the instrument-side ``corrected`` check and
``task_completed`` (v3 §3.6, §7).

The comparison is done by the instrument against the genuine value captured
by ``FlakyWrapperV3`` -- never by the verify LLM judge.

- Strings: exact match after normalisation (Unicode NFC, whitespace collapsed
  and stripped, case-folded).
- Temperatures: ``|final - reference| <= 3 °C`` -- a configuration choice,
  not a measured value (v3 §3.6).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "ANSWER_FIELDS", "FIELD_KIND", "TEMPERATURE_TOLERANCE_C", "AnswerValue", "read_answer",
    "parse_kind", "normalise_string", "Comparison", "compare", "task_completed",
]

# tool -> path to the answer field in the tool's JSON response (v3 §3.6 table)
ANSWER_FIELDS: dict[str, tuple] = {
    "countriesnow": ("data", "capital"),
    "wikipedia": ("extract",),
    "open_meteo": ("current_weather", "temperature"),
    "wttr": ("current_condition", 0, "temp_C"),
    "jsonplaceholder": ("title",),
    "dummyjson": ("todo",),
}
# string | number (°C, JSON number) | int_string (°C, integer as a string)
FIELD_KIND: dict[str, str] = {
    "countriesnow": "string", "wikipedia": "string", "open_meteo": "number",
    "wttr": "int_string", "jsonplaceholder": "string", "dummyjson": "string",
}
TEMPERATURE_TOOLS = frozenset({"open_meteo", "wttr"})
TEMPERATURE_TOLERANCE_C = 3.0  # configuration choice (v3 §3.6)

_MISSING = object()


def _walk(doc: Any, path: tuple) -> Any:
    cur = doc
    for step in path:
        if isinstance(step, int):
            if not isinstance(cur, list) or len(cur) <= step:
                return _MISSING
            cur = cur[step]
        else:
            if not isinstance(cur, dict) or step not in cur:
                return _MISSING
            cur = cur[step]
    return cur


@dataclass(frozen=True)
class AnswerValue:
    present: bool          # the field exists in a JSON body
    parses: bool           # ...and parses as its kind
    raw: Any = None        # the field's JSON value as delivered
    value: Any = None      # parsed: normalised-comparable str, or float °C


def parse_kind(tool: str, raw: Any) -> Optional[Any]:
    """Parsed value, or None if ``raw`` does not parse as the field's kind."""
    kind = FIELD_KIND[tool]
    if kind == "string":
        return raw if isinstance(raw, str) and raw.strip() else None
    if kind == "number":
        return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    if kind == "int_string":
        if isinstance(raw, str) and re.fullmatch(r"\s*-?\d+\s*", raw):
            return float(int(raw))
        return None
    raise ValueError(f"unknown field kind {kind!r}")


def read_answer(tool: str, text: Optional[str]) -> AnswerValue:
    if tool not in ANSWER_FIELDS or not text:
        return AnswerValue(False, False)
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return AnswerValue(False, False)
    raw = _walk(doc, ANSWER_FIELDS[tool])
    if raw is _MISSING:
        return AnswerValue(False, False)
    parsed = parse_kind(tool, raw)
    return AnswerValue(True, parsed is not None, raw, parsed)


def normalise_string(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip().casefold()


@dataclass(frozen=True)
class Comparison:
    match: bool
    tolerance: str            # "exact_normalised" | "abs<=3.0C"
    abs_diff: Optional[float]  # temperatures only


def compare(final_tool: str, final: AnswerValue, reference_tool: str, reference_value: Any) -> Comparison:
    """``final`` (a delivered answer) against a captured genuine ``reference_value``
    (the parsed genuine value of the reference call)."""
    temp = final_tool in TEMPERATURE_TOOLS and reference_tool in TEMPERATURE_TOOLS
    tolerance = f"abs<={TEMPERATURE_TOLERANCE_C}C" if temp else "exact_normalised"
    if not final.parses or reference_value is None:
        return Comparison(False, tolerance, None)
    if temp:
        diff = abs(float(final.value) - float(reference_value))
        return Comparison(diff <= TEMPERATURE_TOLERANCE_C, tolerance, round(diff, 3))
    if final_tool in TEMPERATURE_TOOLS or reference_tool in TEMPERATURE_TOOLS:
        return Comparison(False, tolerance, None)  # a temperature never matches a string answer
    return Comparison(normalise_string(str(final.value)) == normalise_string(str(reference_value)), tolerance, None)


def task_completed(tool: Optional[str], delivered_text: Optional[str], degraded_block: bool,
                   genuine_tool: Optional[str] = None, genuine_value: Any = None) -> bool:
    """v3 §7 per-tool rule for the task's LAST step: the answer field is present
    and parses; in ``degraded_output`` blocks it must also match the genuine value."""
    if tool is None or tool not in ANSWER_FIELDS:
        return False
    ans = read_answer(tool, delivered_text)
    if not (ans.present and ans.parses):
        return False
    if not degraded_block:
        return True
    return compare(tool, ans, genuine_tool or tool, genuine_value).match
