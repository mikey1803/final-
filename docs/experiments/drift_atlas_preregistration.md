# Drift atlas — instrument pre-registration (FROZEN v1)

**Status: FROZEN before any collection run.** This document, together with
`probe/targets.yaml`, *is* the drift atlas measurement instrument (Task 25,
contribution C3), in the same sense that `docs/live_pilot/strategy_definitions.md`
is for the live pilot. It is committed in its own commit before the first
scheduled collection run, and that commit's hash is the freeze timestamp
(CLAUDE.md rule 4).

Changing anything below after that commit invalidates comparability with the
data already collected. If something turns out to be wrong, the fix is a new
frozen version (`drift_atlas_preregistration_v2.md` + `probe/targets_v2.yaml`),
never an edit to v1. v1 and v2 observations are never merged into one series.

**Scope of this document.** It freezes the instrument and the definitions of a
drift event. It contains no analysis plan. Estimators, uncertainty
quantification, and how measured rates map onto the crossover theory's drift
parameters will be pre-registered separately (Task 25 part 2) and committed
before any collected data is examined beyond the operational health checks in §6.

---

## 1. What is frozen

| Component | Frozen value |
|---|---|
| Target list, requests, sanity rules | `probe/targets.yaml`, sha256 `296c6967f50b1cb3199eb10f416116e01132188b499dbf511747916937faed40` (LF line endings, enforced by `.gitattributes`) |
| Schema fingerprint algorithm | `probe/schema_hash.py`, `ALGORITHM_VERSION = 1` (rules in its module docstring) |
| Record format | `probe/run_probe.py`, `RECORD_VERSION = 1`, `PROBE_VERSION = "1.0.0"` (fields in §3) |
| Cadence | every 6 hours, cron `17 */6 * * *` (UTC), GitHub Actions `.github/workflows/drift-probe.yml` |
| Request policy | one GET per target per run, sequential, 0.5 s pause, 10 s timeout, redirects followed (and recorded), **no retries** |
| User-Agent | `FAARF-drift-probe/1.0 (research; +https://github.com/mikey1803/final-; piomicheal17@gmail.com)` |

Each record carries `targets_sha256`, `record_version`, `schema_algorithm` and
`probe_version`, so every observation is bound to the exact instrument that
produced it.

## 2. Targets (24)

The six live-pilot tools, fifteen further keyless public APIs, and three LLM
model-list endpoints. The full definitions are in `probe/targets.yaml`.

| Category | Targets |
|---|---|
| encyclopedic | wikipedia, countriesnow |
| weather | open_meteo, wttr, nws, met_norway, 7timer |
| placeholder | jsonplaceholder, dummyjson, httpbin_json (**negative control**: a static document expected never to drift) |
| search | duckduckgo, openlibrary, crossref, wikidata |
| geo | zippopotam, usgs_quakes, postcodes_io |
| finance | frankfurter_v1 (**deprecated-endpoint watch**), frankfurter_v2, er_api, coinbase_rates |
| llm_models | groq_models (key), gemini_models (key), nvidia_nim_models (keyless) |

Excluded at design time:
- **Nominatim**: its usage policy says repeated identical queries may be classified as faulty and blocked, and it directs regular jobs to alternatives.
- **randomuser.me**: it generates person-like data.
- **IP-geolocation APIs**: they would record the runner's IP.

Liveness of every target was verified on 2026-10-21. All 24 returned 2xx and
passed their sanity checks in two local smoke runs. Those runs are not part of
the dataset (§4).

## 3. Record format (v1)

Each run appends one JSON line per target to `probe/data/YYYY-MM-DD.jsonl` on
the `data` branch. The date is the UTC date of the run's start.

**Run identity**

| Field | Meaning |
|---|---|
| `record_version`, `probe_version`, `schema_algorithm`, `targets_sha256` | Instrument identity |
| `run_id` | UTC start time of the run, `YYYYMMDDTHHMMSSZ` |
| `ts_utc` | UTC start time of this target's request (ms precision) |
| `target`, `category`, `method`, `url` | From `targets.yaml`. The URL never contains a secret: the Gemini key goes in a header. |

**Response**

| Field | Meaning |
|---|---|
| `http_status` | Final status after redirects; `null` if no response arrived |
| `latency_ms` | Wall-clock time to the response, or to the exception |
| `error` | `null`, or the text: `"HTTP <code>: <body…>"`, `"<ExceptionType>: <message>"`, `"non-JSON body"`, `"missing secret: <ENV> not set"`, or `"probe-internal <Type>: …"`. Clipped to 500 characters. Secrets are scrubbed to `[REDACTED]`. |
| `redirect_chain` | `[[status, url], …]` of intermediate responses |
| `final_url`, `content_type`, `body_bytes` | Response metadata |
| `lifecycle_headers` | Any `Deprecation` and `Sunset` headers, plus a `Link` header containing `successor-version`, `deprecation` or `sunset` |

**Content checks**

| Field | Meaning |
|---|---|
| `json_ok` | Whether the body parsed as JSON |
| `schema_hash` | `probe/schema_hash.py` fingerprint of the body (keys and types, never values); `null` if not JSON. The canonical schema is stored once at `probe/data/schemas/<hash>.json`. |
| `sanity` | `{evaluated, ok, failures[]}`. Evaluated only on 2xx JSON responses, against the target's `expect` rules. |
| `model_ids`, `model_count`, `model_ids_complete` | Model-list targets only: the sorted, de-duplicated ids, and whether every page was read |

No response content is stored except model ids.

## 4. The dataset

The dataset consists exactly of records that meet all three conditions:
1. They are on the `data` branch.
2. They were written by the GitHub Actions workflow.
3. Their `targets_sha256` equals the value in §1.

Local runs, including the 2026-10-21 smoke runs, write to temporary directories
and never enter the dataset. Collection starts at the first workflow run after
this document's commit.

## 5. Drift-event definitions (frozen; counted in part 2, not now)

For each target, its records are ordered by `ts_utc`. "Consecutive" means
adjacent in that recorded order, whatever the time gap between them. GitHub
can delay or skip scheduled runs, and gap lengths will be reported, not
imputed.

A **successful observation** has a 2xx `http_status` and `json_ok = true`.

| Class | An event occurs when… | Persistent if… |
|---|---|---|
| **E1 schema change** | the `schema_hash` of a successful observation differs from that of the previous successful observation | the new hash is seen in ≥ 2 consecutive successful observations |
| **E2 status-class change** | the status class differs from the previous observation's class | the new class holds for ≥ 2 consecutive observations |
| **E3 redirect / host change** | the redirect-host sequence differs from the previous observation's: the hosts of `redirect_chain` urls followed by the host of `final_url` | the new sequence holds for ≥ 2 consecutive observations |
| **E4 content-sanity failure** | an evaluated sanity check has `ok = false` | it fails in ≥ 2 consecutive evaluated observations |
| **E5 model id added / removed** | for a model-list target, an id is in the set but not in the previous complete observation's set (added), or the reverse (removed). Only observations with `model_ids_complete = true` count. Each id is one event. | the id stays added (or removed) in ≥ 2 consecutive complete observations |
| **L lifecycle signal** | `lifecycle_headers` differs from the previous observation | the new value holds for ≥ 2 consecutive observations |

Notes on the definitions:
- **E2 status classes** are: `2xx`, `3xx`, `4xx`, `5xx`, `timeout`, `connection_error` and `other_error`. The class is taken from `http_status` when present, otherwise from the exception type in `error`.
- **E3** compares hosts only. A change of path or query alone is not an E3 event.
- **Transient events.** An event that is not persistent is recorded as transient (flapping), and transient and persistent events are counted separately. The ≥ 2 rule means persistence is decided one observation later, at the latest about 6 h (one cadence step) after the change was first seen.

**Not drift (instrument faults).** These records are kept but excluded from all
drift counts, and reported separately as instrument availability:
- `missing secret`
- `probe-internal` errors
- workflow runs that failed before writing records

## 6. Allowed before the analysis pre-registration exists

Operational health checks only:
- whether the workflow ran
- whether the expected 24 lines were written per run
- whether a target is failing because of an instrument fault (for example a missing secret)

Event rates, counts or per-target time series must not be computed or inspected
until the part-2 analysis pre-registration is committed.

## 7. Known limitations, stated at freeze time

- **Account-scoped lists.** The Groq and Gemini model lists are account-scoped. A change may reflect the account or tier, not the provider's public catalog. NVIDIA's list is probed keyless and is not account-scoped.
- **Temporal resolution is 6 h.** A drift that appears and reverts between runs is invisible, and every event time is only known to within one cadence step. GitHub's scheduled runs can also be late or skipped.
- **Single vantage point.** All runs come from GitHub-hosted runners, so region- or IP-dependent behaviour (CDN routing, rate limits, geo-blocking) is that of those runners only.
- **Schema hash is strict.** Any structural change counts, including an optional field appearing or disappearing, or an array going between empty and non-empty. For search-style targets (duckduckgo, openlibrary, crossref, wikidata), content-driven optional fields may produce E1 events that are not API changes. The negative control (httpbin_json) bounds the instrument's own false-positive rate on a static document.
- **DuckDuckGo returns 202.** The Instant Answer API returned HTTP 202 (not 200) in both 2026-10-21 smoke runs, with a valid JSON body. The class is `2xx`, so this is not an event by itself.
- **Fixed queries.** Each target is probed with one fixed query. Drift affecting other endpoints or parameters of the same API is not measured.
- **Selection bias.** Targets were chosen in part for being free and keyless, and the six live-pilot tools because this project uses them. This is not a random sample of APIs that agents use.

## 8. Ethics (summary; full text in `docs/experiments/drift_atlas.md`)

- Public endpoints only.
- No personal data is requested or stored.
- Volume is low: 4 requests per host per day, 96 per day in total.
- An identifying User-Agent with contact details is sent.
- Each provider's published terms were checked on 2026-10-21.
- There are no retries and no load testing.
