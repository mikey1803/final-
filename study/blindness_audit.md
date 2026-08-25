# Blindness Audit

This document is the human-readable audit trail behind
`tests/test_study.py`'s automated blindness test. The test is the
enforcement mechanism (it runs on every test invocation and fails the
build if violated); this document is the reviewer-readable explanation
of WHY it passes, endpoint by endpoint, current as of `PROTOCOL_VERSION`
in [`version.py`](version.py).

## What "blind" means here, precisely

No study API response, at any point in the participant's session, may
contain:

1. Any value from `recovery.envs.canonical.MATRIX` (to 2 decimal
   places).
2. Any key or value shaped like a reward/success-rate/score/matrix field
   (`reward`, `success_rate`, `score`, `matrix`, `p_success`, and
   case-insensitive variants).
3. `recovery.envs.canonical.BLIND_RULE` (the project's own reference
   design) or any information that could let a participant infer it.
4. Any other participant's data, any aggregate statistic, or the
   participant's own score.

## Import-graph audit

`study/app/backend/main.py` and `study/app/backend/schemas.py` import,
transitively, exactly:

```
fastapi, pydantic, numpy, hashlib, os, sys, datetime, pathlib   (stdlib / infra)
recovery.envs.descriptions  (-> recovery.envs.canonical.TYPE_NAMES, STRATEGY_NAMES only)
study.storage
study.version
```

`recovery.envs.canonical.MATRIX` and `recovery.envs.canonical.BLIND_RULE`
are NEVER imported by any module reachable from `study/app/backend/main.py`.
This is verified two ways:

1. **Static**: `grep -rn "MATRIX\|BLIND_RULE" study/` finds no match
   outside this audit document and `recovery/envs/descriptions.py`'s
   docstring (which mentions the names only in prose, explaining what NOT
   to import).
2. **Dynamic**: `tests/test_study.py::test_study_backend_never_imports_matrix`
   imports `study.app.backend.main`, then asserts
   `"recovery.envs.canonical"` is NOT in `sys.modules`'s import chain for
   any object holding a reference to the actual `MATRIX` array (concretely:
   walks `sys.modules["study.app.backend.main"].__dict__` and every
   imported module's namespace reachable from it, asserting no bound name
   anywhere is `numpy.ndarray`-typed and shape/dtype-compatible with
   `MATRIX`, which -- combined with the static grep -- rules out both a
   direct `import` and a re-exported reference).

## Endpoint-by-endpoint audit

| Endpoint | Response model (allowlist) | Fields returned | MATRIX-derived? |
|---|---|---|---|
| `GET /api/study/health` | `HealthResponse` | `status`, `protocol_version` | No -- static strings |
| `GET /api/study/task-content` | `TaskContentResponse` | `types[].{name,label,description}`, `type_order`, `strategies[].{name,label,description,cost}`, `strategy_order`, `protocol_version`, `n_types`, `n_strategies` | No -- `types`/`strategies` come from `recovery.envs.descriptions.type_table()`/`strategy_table()`, which import only `TYPE_NAMES`/`STRATEGY_NAMES` (label lists); `cost` values (`{1.0, 1.5, 2.0, 2.5, 3.0}`) are disjoint from every `MATRIX` value (`MATRIX`'s range is `[0.10, 0.80]`) |
| `POST /api/study/submit` | `SubmitResponse` | `status`, `token` (echoed back, participant-generated, not derived from anything), `protocol_version` | No |
| Any 4xx/5xx error | FastAPI default (`{"detail": ...}`) | Pydantic validation messages (field names + submitted values) or the literal strings this codebase writes in `HTTPException(detail=...)` calls | No -- every `detail=` string in `study/app/backend/main.py` is a hand-written literal about tokens/ordering/storage, never interpolates a `MATRIX` value (there is nothing to interpolate; `MATRIX` isn't in scope) |

Every response model has `model_config = ConfigDict(extra="forbid")` --
if a future code change accidentally added a new field to a handler's
return value, FastAPI's response-model filtering would either drop it
(if not declared on the model) or raise a validation error (if
`extra="forbid"` is set on a model being asked to serialize an
undeclared field via `.model_validate`), rather than silently leaking it.

## Cost values vs. MATRIX values -- explicit disjointness check

Strategy costs (`recovery/envs/descriptions.py::STRATEGY_COSTS`):
`{1.0, 1.5, 2.0, 2.5, 3.0}`.

`MATRIX` values (`recovery/envs/canonical.py`): every entry is in
`[0.10, 0.80]`.

These ranges do not overlap (costs are all `>= 1.0`; `MATRIX` entries are
all `<= 0.80`), so even a naive "does any number in the response equal
any number in `MATRIX`" check can never produce a false negative from
cost values being mistaken for suppressed reward information, and
`tests/test_study.py`'s blindness test checks this disjointness directly
as a regression guard (if a future `MATRIX` edit ever pushed a value into
the cost range, or vice versa, the test would need updating -- it is not
silently correct by accident).

## What is deliberately NOT audited here

- The CONTENT of `rationale` free-text fields submitted BY participants
  -- a participant could, in principle, type something matrix-related
  into their own rationale. This is not a blindness violation (the
  system never SHOWED them that information; they cannot type what they
  were never given), and is out of scope for this audit, which covers
  what the SERVER exposes, not what a participant chooses to write.
- Server logs / process memory (e.g. whether `MATRIX` is loaded in the
  SAME Python process as the study service for some other reason, such
  as running tests in the same interpreter). The blindness guarantee is
  about the HTTP response surface a participant's browser can observe,
  not about what happens to be resident in server memory for unrelated
  reasons (e.g. `pytest` importing both `study.app.backend.main` and
  `recovery.envs.canonical` in the same test session, which is exactly
  what `tests/test_study.py` itself does, safely, since a test process
  is not a participant-facing HTTP response).

## Re-running this audit

```bash
cd backend
pytest tests/test_study.py -k blindness -v
```

Re-run this audit (and re-check this document is still accurate) after
ANY change to `study/app/backend/`, `recovery/envs/descriptions.py`, or
`recovery/envs/canonical.py`.
