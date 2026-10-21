"""Per-tool output schemas for VERIFY's v2 schema check (Pydantic, strict
types, extra fields allowed). Shapes match each tool's real response as
observed live on 2026-10-21; they check structure only -- whether the content
answers the step is the LLM judge's job."""

from __future__ import annotations

import json
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, ValidationError

from recovery.live.tools import ToolCallResult

Number = Union[StrictInt, StrictFloat]


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class WikipediaOut(_Model):
    title: StrictStr = Field(min_length=1)
    extract: StrictStr = Field(min_length=1)


class _CountriesNowData(_Model):
    capital: StrictStr = Field(min_length=1)


class CountriesNowOut(_Model):
    error: StrictBool
    data: _CountriesNowData


class _OpenMeteoCurrent(_Model):
    temperature: Number
    time: StrictStr


class OpenMeteoOut(_Model):
    current_weather: _OpenMeteoCurrent


class _WttrCurrent(_Model):
    temp_C: StrictStr


class WttrOut(_Model):
    current_condition: list[_WttrCurrent] = Field(min_length=1)


class JsonPlaceholderOut(_Model):
    id: StrictInt
    title: StrictStr
    completed: StrictBool


class DummyJsonOut(_Model):
    id: StrictInt
    todo: StrictStr
    completed: StrictBool


OUTPUT_SCHEMAS: dict[str, type[_Model]] = {
    "wikipedia": WikipediaOut, "countriesnow": CountriesNowOut, "open_meteo": OpenMeteoOut,
    "wttr": WttrOut, "jsonplaceholder": JsonPlaceholderOut, "dummyjson": DummyJsonOut,
}


def check_output_schema(tool: str, result: ToolCallResult) -> tuple[bool, Optional[str]]:
    """(passed, why-not). Uses the untruncated ``full_text`` when present."""
    text = result.full_text if result.full_text is not None else result.body
    if not text:
        return False, "no body"
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return False, "body is not valid JSON"
    model = OUTPUT_SCHEMAS.get(tool)
    if model is None:
        return False, f"no output schema for tool {tool!r}"
    try:
        model.model_validate(doc)
    except ValidationError as exc:
        first = exc.errors()[0]
        return False, f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return True, None
