"""Pydantic schemas for the blind-design study API.

Every RESPONSE model here is the technical enforcement of the blindness
guarantee, not a convention: FastAPI serializes a route's return value
through its declared ``response_model`` and drops anything not declared
on that model, so a field accidentally added to a handler's return dict
(e.g. a stray debug value) is silently filtered out rather than silently
leaked. ``tests/test_study.py``'s blindness test still independently
scans every raw response body byte-for-byte -- this module is the
enforcement mechanism the test is verifying, not a substitute for it.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from recovery.envs.descriptions import STRATEGY_NAMES, TYPE_NAMES

N_TYPES = len(TYPE_NAMES)
N_STRATEGIES = len(STRATEGY_NAMES)

YEARS_EXPERIENCE_LEVELS = ("0-2", "3-5", "6-10", "10+")
LLM_FAMILIARITY_LEVELS = ("none", "some", "expert")

# ======================================================================
# Response models -- the allowlist. Every field below is safe by
# construction: names/labels/descriptions/costs are PUBLIC study content
# (shown to every participant identically, modulo display order), never
# derived from recovery.envs.canonical.MATRIX or recovery.theory.*.
# ======================================================================


class FailureTypeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    label: str
    description: str


class StrategyOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    label: str
    description: str
    cost: float


class TaskContentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    types: list[FailureTypeOut]
    type_order: list[int]
    strategies: list[StrategyOut]
    strategy_order: list[int]
    protocol_version: str
    n_types: int
    n_strategies: int


class SubmitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    token: str
    protocol_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    protocol_version: str


# ======================================================================
# Request models -- field-level validation backing the HTTP 422 contract.
# ======================================================================


class BackgroundInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    years_experience: str
    llm_familiarity: str
    built_agent_systems: bool

    @field_validator("years_experience")
    @classmethod
    def _valid_years(cls, v: str) -> str:
        if v not in YEARS_EXPERIENCE_LEVELS:
            raise ValueError(f"years_experience must be one of {YEARS_EXPERIENCE_LEVELS}, got {v!r}")
        return v

    @field_validator("llm_familiarity")
    @classmethod
    def _valid_familiarity(cls, v: str) -> str:
        if v not in LLM_FAMILIARITY_LEVELS:
            raise ValueError(f"llm_familiarity must be one of {LLM_FAMILIARITY_LEVELS}, got {v!r}")
        return v


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=8, max_length=128)
    background: BackgroundInput
    type_order: list[int]
    strategy_order: list[int]
    mapping: list[int]
    confidence: list[int]
    rationale: list[str]
    seconds_on_task: float = Field(gt=0)

    @field_validator("type_order")
    @classmethod
    def _valid_type_order(cls, v: list[int]) -> list[int]:
        if sorted(v) != list(range(N_TYPES)):
            raise ValueError(f"type_order must be a permutation of 0..{N_TYPES - 1}, got {v}")
        return v

    @field_validator("strategy_order")
    @classmethod
    def _valid_strategy_order(cls, v: list[int]) -> list[int]:
        if sorted(v) != list(range(N_STRATEGIES)):
            raise ValueError(f"strategy_order must be a permutation of 0..{N_STRATEGIES - 1}, got {v}")
        return v

    @field_validator("mapping")
    @classmethod
    def _valid_mapping(cls, v: list[int]) -> list[int]:
        if len(v) != N_TYPES:
            raise ValueError(f"mapping must have exactly {N_TYPES} entries, got {len(v)}")
        bad = [x for x in v if not (0 <= x < N_STRATEGIES)]
        if bad:
            raise ValueError(f"mapping entries must be in [0, {N_STRATEGIES}), got out-of-range values {bad}")
        return v

    @field_validator("confidence")
    @classmethod
    def _valid_confidence(cls, v: list[int]) -> list[int]:
        if len(v) != N_TYPES:
            raise ValueError(f"confidence must have exactly {N_TYPES} entries, got {len(v)}")
        bad = [x for x in v if not (1 <= x <= 5)]
        if bad:
            raise ValueError(f"confidence entries must be in [1, 5], got out-of-range values {bad}")
        return v

    @field_validator("rationale")
    @classmethod
    def _valid_rationale(cls, v: list[str]) -> list[str]:
        if len(v) != N_TYPES:
            raise ValueError(f"rationale must have exactly {N_TYPES} entries, got {len(v)}")
        too_long = [i for i, r in enumerate(v) if len(r) > 500]
        if too_long:
            raise ValueError(f"rationale entries must be <= 500 characters (rows {too_long} exceed this)")
        return v
