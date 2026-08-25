"""Standalone FastAPI service for the blind-design elicitation study (Task 11).

BLINDNESS GUARANTEE -- read before touching this file
--------------------------------------------------------
This module must NEVER import ``recovery.envs.canonical.MATRIX``,
``recovery.envs.canonical.BLIND_RULE``, or anything from
``recovery.theory.*``. Scoring happens entirely OFFLINE, in
``recovery.analysis.human_study``, after a study is collected -- never
in this request-serving path. The only import this module takes from the
simulation package is ``recovery.envs.descriptions`` (plain-English
labels/descriptions/costs, not reward information).

Every response is additionally routed through an explicit Pydantic
``response_model`` (``study/app/backend/schemas.py``) that acts as a
field allowlist -- FastAPI drops any undeclared field before
serialization. ``tests/test_study.py``'s blindness test independently
scans every raw response body for any float appearing in ``MATRIX`` and
for reward/score/matrix-shaped key names, so this is a belt-and-braces
guarantee, not a single point of failure.

STANDALONE BY DESIGN
---------------------
This service depends on ``fastapi``, ``uvicorn``, ``pydantic``, and
``recovery.envs.descriptions`` (which itself depends on nothing beyond
``recovery.envs.canonical``'s two name lists) -- NOT on ``backend/src``
(scikit-learn classifier, Groq LLM client, SQLite outcome store) that
``backend/api/main.py`` (the project's OTHER, unrelated FastAPI service,
the live agent demo) requires. This is a deliberate choice, not a literal
"add routes to the existing app" reuse: this service needs to be
deployable to a public URL for real study participants without dragging
in the demo's heavier dependency surface. See ``study/protocol.md``,
"Deviations from a literal reading of the task", for the full rationale.

Run
---
    uvicorn study.app.backend.main:app --reload --port 8100

(from the REPO ROOT -- this module inserts both the repo root and
``backend/`` onto ``sys.path`` itself, mirroring the path-setup pattern
``backend/api/main.py`` already uses, so it does not matter which
directory the process is actually launched from.)
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_STUDY_DIR = Path(__file__).resolve().parent.parent.parent  # .../study/app/backend -> .../study
_REPO_ROOT = _STUDY_DIR.parent
_BACKEND_ROOT = _REPO_ROOT / "backend"
for _p in (_REPO_ROOT, _BACKEND_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from recovery.envs.descriptions import strategy_table, type_table  # noqa: E402
from recovery.envs.display_order import display_order as _shared_display_order  # noqa: E402
from study.app.backend.schemas import (  # noqa: E402
    HealthResponse,
    N_STRATEGIES,
    N_TYPES,
    SubmitRequest,
    SubmitResponse,
    TaskContentResponse,
)
from study.storage import (  # noqa: E402
    MalformedRecordError,
    StorageWriteError,
    SubmissionStore,
    TokenAlreadyExistsError,
)
from study.version import PROTOCOL_VERSION  # noqa: E402

DEFAULT_DATA_PATH = _STUDY_DIR / "data" / "submissions.jsonl"
DATA_PATH = Path(os.environ.get("STUDY_DATA_PATH", DEFAULT_DATA_PATH))

app = FastAPI(title="Blind-Design Elicitation Study API", version=PROTOCOL_VERSION)

# Study frontend's Vite dev server -- deliberately a DIFFERENT port than the
# unrelated demo dashboard (5173/5174, see backend/api/main.py) so the two
# apps can run simultaneously during development without a port clash.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5180",
        "http://127.0.0.1:5180",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = SubmissionStore(DATA_PATH)
_TYPES = type_table()
_STRATEGIES = strategy_table()


def _display_order(token: str) -> tuple[list[int], list[int]]:
    """So ``GET /task-content`` is idempotent and the client never needs
    server-side session state to redisplay a consistent order (e.g. after
    a page refresh, before submission). Thin wrapper around the shared
    randomiser (``recovery.envs.display_order``) -- also reused, with a
    non-token seed input, by Task 15's LLM presentation-order-randomisation
    experiment."""
    return _shared_display_order(token, N_TYPES, N_STRATEGIES)


@app.get("/api/study/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", protocol_version=PROTOCOL_VERSION)


@app.get("/api/study/task-content", response_model=TaskContentResponse)
def task_content(token: str) -> TaskContentResponse:
    """Serve Table I (failure types) and Table II (strategies + costs) in
    a per-token randomized DISPLAY order, with that order returned
    explicitly so the client can (a) show it and (b) echo it back on
    submit for the record (see ``study/protocol.md``, "Randomised
    display order")."""
    if not token or len(token) < 8:
        raise HTTPException(status_code=422, detail="token must be a non-empty opaque string of at least 8 characters")

    type_order, strategy_order = _display_order(token)
    return TaskContentResponse(
        types=[_TYPES[i] for i in type_order],
        type_order=type_order,
        strategies=[_STRATEGIES[i] for i in strategy_order],
        strategy_order=strategy_order,
        protocol_version=PROTOCOL_VERSION,
        n_types=N_TYPES,
        n_strategies=N_STRATEGIES,
    )


@app.post("/api/study/submit", response_model=SubmitResponse, status_code=201)
def submit(req: SubmitRequest) -> SubmitResponse:
    """Record one submission. Immutable once recorded: a second POST with
    an already-used ``token`` returns HTTP 409, never silently overwrites
    (see ``study/storage.py``'s ``TokenAlreadyExistsError``).

    Field-level validation (wrong-length mapping, out-of-range strategy
    index, etc.) is handled by ``SubmitRequest``'s Pydantic validators and
    surfaces automatically as HTTP 422 before this handler ever runs.
    """
    if _store.has_token(req.token):
        raise HTTPException(status_code=409, detail=f"token {req.token!r} has already submitted; responses are immutable")

    expected_type_order, expected_strategy_order = _display_order(req.token)
    if req.type_order != expected_type_order or req.strategy_order != expected_strategy_order:
        raise HTTPException(
            status_code=422,
            detail="type_order/strategy_order do not match the order this token was actually shown "
            "(GET /api/study/task-content for this token first, and submit exactly what it returned)",
        )

    record = {
        "token": req.token,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "background": req.background.model_dump(),
        "display_order": {"type_order": req.type_order, "strategy_order": req.strategy_order},
        "mapping": req.mapping,
        "confidence": req.confidence,
        "rationale": req.rationale,
        "seconds_on_task": req.seconds_on_task,
        "protocol_version": PROTOCOL_VERSION,
    }

    try:
        _store.append(record)
    except TokenAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MalformedRecordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except StorageWriteError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"submission was NOT recorded (storage write failed): {exc}",
        ) from exc

    return SubmitResponse(status="recorded", token=req.token, protocol_version=PROTOCOL_VERSION)
