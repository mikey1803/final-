# Seed drift events — observed before the probe started

Real-world API drift that this project ran into **before** the automated probe
(`probe/run_probe.py`) began collecting. These are anecdotal, not sampled: they
are listed because they motivated the drift atlas (Task 25, C3), and they must
be kept apart from probe-measured events in any analysis (they were found
because they broke something, which is a biased way to find drift).

Every row cites the repo document (or probe observation) that records it.
Dates nobody has confirmed are `[DATE REQUIRED]`; they are never estimated.

Event classes follow `docs/experiments/drift_atlas_preregistration.md` §5:
E1 schema change · E2 status-class change · E3 redirect/host change ·
E4 sanity failure on 2xx · E5 model id added/removed.

| # | Event | Provider / endpoint | Class | Evidence (where recorded) | First observed by this project | Provider change date |
|---|---|---|---|---|---|---|
| 1 | REST Countries v3.1 deprecated. v3.1 returns an explicit deprecation error and v5 needs an account and API key. Re-checked 2026-10-21: `/v3.1/name/india` → 301 → `files-03.restcountries.com/countries.00/legacy.json`, whose body is `{"success": false, … "This API version has been deprecated…"}`. | restcountries.com `/v3.1/*` | E2 + E3 + E4 | `docs/live_pilot/strategy_definitions.md` tool table (section headed "Verified live 2026-08-20"); probe planning liveness check 2026-10-21 | [DATE REQUIRED] | [DATE REQUIRED] |
| 2 | Groq removed `llama-3.1-8b-instant`. A live `client.models.list()` returned zero Llama chat models. The model was still live when Task 11 / Phase 1 / Phase 4 data was collected (2026-07-27/28). | api.groq.com `/openai/v1/models` | E5 | `docs/PAPER_PREP.md` §6 "2026-08-20 provenance audit"; `docs/experiments/human_study.md` "2026-08-20 provenance note" | [DATE REQUIRED] | [DATE REQUIRED] |

## Additional documented events (approved with the Task 25 plan)

| # | Event | Provider / endpoint | Class | Evidence (where recorded) | First observed by this project | Provider change date |
|---|---|---|---|---|---|---|
| 3 | reqres.in became key-gated: it now returns 401 and requires registration plus `x-api-key`. | reqres.in | E2 | `docs/live_pilot/strategy_definitions.md` tool table | [DATE REQUIRED] | [DATE REQUIRED] |
| 4 | NVIDIA NIM ended `mistralai/mixtral-8x7b-instruct-v0.1` on the hosted endpoint. A real call returned HTTP 410 Gone "with that exact date". | integrate.api.nvidia.com | E2 + E5 | `backend/conf/llm_designers.yaml` lines 65-69 (comment) | [DATE REQUIRED] | 2026-07-27, as stated in the 410 response (per the yaml comment; the raw response is not in the repo) |
| 5 | Frankfurter moved hosts and versions. `api.frankfurter.app/latest` → 301 → `api.frankfurter.dev/v1/latest`. v1 itself is deprecated: it sends `Deprecation: @1779103800` (2026-05-18T11:30:00Z) and `Link: <https://api.frankfurter.dev/v2/rates>; rel="successor-version"`. v1 is still served ("remains available indefinitely", frankfurter.dev). | api.frankfurter.app → api.frankfurter.dev | E3 + lifecycle header | Probe planning liveness check and smoke runs, 2026-10-21 | 2026-10-21 | Host move: [DATE REQUIRED]. v1 deprecation: 2026-05-18T11:30:00Z (from the header itself). |
| 6 | Groq catalog changed between two observations. On 2026-08-20 it listed `qwen/qwen3.6-27b` and `groq/compound` / `compound-mini`. On 2026-10-21 none of these are present and `qwen/qwen3.8-27b` is. | api.groq.com `/openai/v1/models` (account-scoped list) | E5 | 2026-08-20 list: `docs/PAPER_PREP.md` §6. 2026-10-21 list: probe smoke run (not part of the dataset). | 2026-10-21 | Between 2026-08-20 and 2026-10-21: [DATE REQUIRED] |

Row 6 directly affects Task 20 step 7. `qwen/qwen3.6-27b` is no longer
available on this Groq account, so it can't be the One-Shot LLM replacement.
