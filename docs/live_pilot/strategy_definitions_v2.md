# Live-pilot strategy definitions — v2

**Status: FROZEN v2. Approved 2026-10-21 and committed in its own commit
(the v2 freeze timestamp, CLAUDE.md rules 3–4), together with the static
country list it names in §4. Any change goes in a `v3` file.**

## Relationship to v1

`docs/live_pilot/strategy_definitions.md` (FROZEN v1) is unchanged and stays
authoritative for v1. This document adds a second, separately frozen set of
definitions:

- Every recovery record carries `definition_version` (`"v1"` or `"v2"`).
- **v1 and v2 data are never merged in one cell.**
- Both versions are implemented by the same executors, as different
  configurations, so either can be run at any time (§4 gives the v1 mapping
  for each strategy).

Unchanged from v1, and applying to both versions:
- the six tools and the peer map (`wikipedia ↔ countriesnow`, `open_meteo ↔ wttr`, `jsonplaceholder ↔ dummyjson`)
- background fault injection at `injection_rate = 0.20`, applied symmetrically to both members of a peer pair, so a recovery call faces the same injection risk as the original call
- the session-start tool-health snapshot
- prediction-after-labelling
- the Phase 2 cell floors (`n_replays ≥ 30` and `n_distinct_events ≥ 15`)
- clustering by `event_id` in any analysis
- the rule that cells at that floor are descriptive and not precise enough for the crossover predictor

---

## 1. Two execution contexts, never pooled

| | Phase 2 replay (v1 design) | Inline recovery (new in v2) |
|---|---|---|
| When | After collection, against an already-labelled `FailureEvent` | During a live run, immediately after the failure, before any label exists |
| Record type | `ReplayResult` (frozen schema, `event_schema.py`) | `RecoveryAttempt` (new, `recovery/live/executors/`) |
| Strategy choice | All arms, order randomised per event, `trial_index` | One strategy drawn per failure (§2) |
| Feeds | Phase 2 `(type, strategy)` cells | Its own analyses only (e.g. offline policy evaluation) |

**Inline `RecoveryAttempt` records are never pooled with Phase 2
`ReplayResult` data, and never counted toward Phase 2 cells.**
- They are produced before labelling, by a different selection process, against the live world state at that moment rather than a captured one.
- They are stored separately (`<store>/recoveries/`). Their schema has no `trial_index` and does not require `true_type`.

## 2. Inline strategy selection

- **Policy:** one strategy is drawn uniformly at random from the six real strategies for each captured failure.
- **Selection propensity:** recorded on every attempt. Under the uniform policy it is `1/6`, exactly. It is needed for offline policy evaluation, e.g. inverse-propensity weighting.
- **NULL_CONTROL is excluded from the draw** until its `wait_s` constant is frozen by the Phase 2 timing pre-study. Until then the constant is `[RESULT REQUIRED]`.
- **Seed:** the draw uses a per-attempt seed derived with `numpy.random.SeedSequence(session_seed).spawn` (or `zlib.crc32` for string keys; never `hash()`). That seed is recorded so the draw is reproducible.
- **Policy hook:** callers may supply another policy, but it must report its own propensity for the chosen strategy.

## 3. Budget (applies to every strategy)

| Limit | Default | Counts |
|---|---|---|
| `max_attempts` | 3 | tool calls issued by the executor |
| `max_wall_ms` | 40000 | wall-clock time from executor start |
| `max_tokens` | 1500 | LLM prompt + completion tokens, as reported by the provider |

**These defaults are configuration choices, not measured values.** Nothing
in this project's data was used to derive them.

`max_wall_ms` was raised from a first draft of 15000 to 40000. Under the
12 s per-tool-call reservation below, 15 s would have let a second tool call
start only if the first finished within about 3 s. RETRY would then have
stopped at 1–2 attempts, and the budget, not `max_attempts`, would have
decided how many attempts it got.

**Rule: an action is started only if its worst case fits in the remaining
budget.** Concretely:
- **Tool call:** started only if `remaining_wall_ms ≥ 12000`. That is the 10 s tool timeout plus a 2 s allowance for connection setup.
- **LLM call:** started only if `remaining_tokens ≥ ceil(len(prompt_chars)/2) + max_completion_tokens`, and it gets a timeout of `min(20 s, remaining wall)`.
- **Backoff sleeps** are capped by the remaining wall time.

When a needed action doesn't fit, the executor stops with
`success = False, evidence.reason = "budget_exhausted"`.

The limits **and** the amounts used are recorded on every attempt:
`budget_used = {attempts, wall_ms, tokens}`.

**Overruns.** The token rule assumes at most one token per two prompt
characters, and the wall rule assumes a tool call finishes within its timeout
plus 2 s. `requests`' timeout applies per connect/read phase, not to the
whole call, so a pathological server can exceed that. Any overrun is recorded
as `budget_exceeded = true` and reported, never hidden.

## 4. Strategy definitions (v2), with the v1 configuration of each

Success is always judged by the stated criterion, never by the executor's own
claim.

**"Tool criterion"** means `ToolCallResult.success`: an HTTP 2xx from the real
tool, through the `FlakyWrapper`. As in v1, this is tool-level success. Whether
the answer also satisfies the subgoal is judged only where a strategy
explicitly says so (VERIFY's LLM judge).

### RETRY

| | |
|---|---|
| v2 action | Re-issue the identical call (same tool, same args) up to `max_attempts` times. Wait before each re-issue with exponential backoff and full jitter: `delay_k ~ U(0, min(cap, base·2^k))`, defaults `base = 0.5 s`, `cap = 4 s`. Stop at the first success, at `max_attempts`, or when the next attempt doesn't fit the budget. |
| Success | Any re-issued call succeeds by the tool criterion. |
| v1 configuration | `max_attempts = 1`, `base = 0`: one immediate re-issue, zero backoff. This reproduces v1 exactly. |
| Distinct from NULL_CONTROL | NULL_CONTROL is one fixed, frozen wait followed by one re-issue. RETRY v2 uses randomised backoff over up to `max_attempts` re-issues. |

### SWITCH_TOOL

| | |
|---|---|
| Action (v1 = v2) | Deterministic substitution per the peer map, with the minimal argument translation in the table below, issued once through the `FlakyWrapper`. |
| Success | The substitute call succeeds by its own tool criterion. |
| Untranslatable | No call is made, and the attempt is recorded with `success = False`, `evidence.reason = "untranslatable_args"`, `attempts = 0`. Analysis must separate "the strategy could not apply" from "it was tried and failed". |

| From → to | Translation | Untranslatable when |
|---|---|---|
| wikipedia → countriesnow | `country = title` with `_` replaced by a space | the normalised title is **not** in the static country list below |
| countriesnow → wikipedia | `title = country` with spaces replaced by `_` | never |
| open_meteo → wttr | `location = "{latitude},{longitude}"` (wttr accepts `lat,lon`) | never |
| wttr → open_meteo | parse `location` as `"lat,lon"` | `location` is not a `lat,lon` pair (a place name would need geocoding, which is out of scope) |
| jsonplaceholder → dummyjson | same resource, same `id` | the resource does not exist on the target, or the `id` is outside the target's range (table below) |
| dummyjson → jsonplaceholder | same resource, same `id` | the resource does not exist on the target, or the `id` is outside the target's range (table below) |

**Static country list (wikipedia → countriesnow).**

| | |
|---|---|
| File | `backend/recovery/live/data/countriesnow_countries_v2.json`, committed in this freeze commit. **Never fetched at runtime.** |
| Source | countriesnow.space's own country list, `GET https://countriesnow.space/api/v0.1/countries/capital`, fetched once on 2026-10-21T17:13:03Z |
| Size and integrity | 250 names, no case-insensitive duplicates. Content hash `names_sha256 = dddab0439c7058d32b875a2cc051c4e6b3b757bc20a5bbf0fbba6bf352c54c15` (definition stored in the file). |
| Why this list | It is exactly the set of names the target tool knows, so "translatable" means "the target can in principle answer it". |
| Matching rule | Replace `_` with a space, collapse runs of whitespace to one space, strip. Then compare **case-insensitively and exactly** against the list: no fuzzy matching, no aliases, no partial matches. Example: `India` matches; `Kolkata` and `Python_(programming_language)` do not. |

**Placeholder resource mapping (jsonplaceholder ↔ dummyjson).** Verified live
on 2026-10-21 by direct requests:
- each list endpoint was fetched in full: JSONPlaceholder `/{resource}`, and DummyJSON `/{resource}?limit=0`, reading `total`
- ids were checked to be contiguous from 1
- the single-item boundaries the tools call were probed: `todos/200` → 200 and `todos/201` → 404 on JSONPlaceholder; `todos/254` → 200 and `todos/255` → 404 on DummyJSON; `todos/0` → 404 on both

| Resource | JSONPlaceholder ids | DummyJSON ids | jsonplaceholder → dummyjson | dummyjson → jsonplaceholder |
|---|---|---|---|---|
| todos | 1–200 | 1–254 | ids 1–200 | ids 1–200 (201–254 untranslatable) |
| posts | 1–100 | 1–251 | ids 1–100 | ids 1–100 |
| comments | 1–500 | 1–340 | ids 1–340 (341–500 untranslatable) | ids 1–340 |
| users | 1–10 | 1–208 | ids 1–10 | ids 1–10 (11–208 untranslatable) |
| albums, photos | exist | 404 | untranslatable | — |
| products, carts, quotes, recipes | 404 | exist | — | untranslatable |

The live tools currently call only `todos/{id}`. The other rows define the
rule for any future tool. An id outside the **source's** own range (for
example the malformed-request injection's `999999`) is also outside the
target's range, so it is untranslatable. These ranges are a snapshot of live
services; a change found later is drift, recorded as such, and the table is
not edited (CLAUDE.md rule 3).

### VERIFY

| | |
|---|---|
| v2 action | Two checks on the last tool result, in order. Each can be disabled by a config flag; **both are ON by default.** (1) **Schema check:** the result body is validated against that tool's output schema (one Pydantic model per tool). A missing or empty body fails. (2) **LLM judge:** the model is shown the step goal and the last result (present, absent or erroneous) and answers whether it satisfies the goal, with a one-line reason. If any enabled check rejects the result, **one** corrective re-issue of the same call is permitted, and its result goes through the same enabled checks. |
| No result to check | If the original call raised or timed out (no response body), VERIFY re-issues the call **once** and runs every enabled check on the new result. This is the one corrective re-issue. It is not followed by a second one. |
| Success | The final result passes every enabled check. |
| Original result already passes | Recorded as `success = True`, `attempts = 0`, `evidence.reason = "original_result_accepted"`, and **`no_action_taken = True`**. VERIFY did nothing in this case, so it must not get credit for fixing a failure. **Analysis reports VERIFY's success rate twice: including and excluding `no_action_taken` attempts.** |
| v1 configuration | `schema_check = off`, `llm_judge = on`: the frozen definition. |
| Why both are on in v2 | A schema check alone can never catch `hallucinated_output`. A well-formed but fabricated answer passes it. That would make VERIFY useless for that type and distort the measured success matrix. |

As in v1, for failures with nothing to check (e.g. `tool_timeout`), VERIFY
reduces to "notice nothing usable happened, then re-issue". Its resemblance to
RETRY for those types is a finding to report, not a defect.

### RE_GROUND

| | |
|---|---|
| v2 action | The model is given the original goal, the current step goal and the **last good state**, i.e. the most recent successful tool result in this task's history, or "none". It re-selects the call (tool + args) once, and that call is issued once. The model is **not** shown the failure and is **not** asked for a new approach. |
| Success | The re-selected call succeeds by its tool criterion. |
| Logged | `reground_chosen_tool` and whether the call equals the original (`same_call`), so overlap with RETRY and REPLAN is measurable. |
| v1 configuration | Re-inject the context, then re-issue the **same** call unchanged, as frozen. Because a tool call is deterministic given its args, v1 RE_GROUND behaves like RETRY for tool-level failures. That is recorded, not hidden. |
| Distinct from REPLAN | RE_GROUND restores context without the failure. REPLAN sees the failure and asks for a different approach. |

### REPLAN

| | |
|---|---|
| Action (v1 = v2) | The model is given the goal, the step goal, the failed call and its error text, and asked for a **different approach** to the same subgoal. It returns exactly one call (tool + args) and may choose any tool. The call is executed once. |
| Success | The chosen call succeeds by its tool criterion. |
| Logged | `replan_chosen_tool`, as required by v1, and `same_call`. |
| Unparseable output | No call is made, and the attempt is recorded with `success = False`, `evidence.reason = "unparseable_llm_output"`. |

### CLARIFY

| | |
|---|---|
| Action (v1 = v2) | The model generates **one** clarifying question about the goal. A **responder** answers it. The model then re-selects the call given the goal, the question and the answer, and the call is issued once. |
| Responder (live collection) | **CannedAnswerResponder in both v1 and v2.** It returns the task's pre-authored `clarify_canned_answer`, so batch runs stay unattended and reproducible. A human responder (a stdin or UI callback with a timeout) exists as an option, for the later `with_recovery` wrapper and the dashboard. It is never used in collection runs, and the responder type is recorded on every attempt. |
| question_unmatched | The responder returns `None` or the timeout fires. The attempt stops with no retry and no second question, and is recorded with `success = False`, `evidence.reason = "question_unmatched"`, `aborted = true`. It is **excluded from success/failure tallies** and reported separately, as in v1. |
| Success | The resulting call succeeds by its tool criterion. |

Known limitation (shared with v1): the canned responder returns the canned
answer whatever the question, so it cannot detect a question its answer does
not address. In practice `question_unmatched` therefore fires only for tasks
with no canned answer, or on a human-responder timeout. See also §7.

### NULL_CONTROL

| | |
|---|---|
| Action | Unchanged from v1: a fixed wait of `wait_s`, then re-issue the identical call once. |
| Status | The executor requires an explicit `wait_s` and has no default. The constant is `[RESULT REQUIRED]` until the Phase 2 timing pre-study freezes it. NULL_CONTROL is excluded from inline selection until then. |

## 5. RecoveryAttempt record (schema version 1)

**Identity and timing**

| Field | Meaning |
|---|---|
| `attempt_id` | sha256 of `(event_id, strategy, definition_version, started_at)`, first 24 characters |
| `event_id`, `session_id`, `task_id` | Link to the captured `FailureEvent` |
| `started_at`, `finished_at` | UTC ISO-8601 |
| `schema_version` | `"1"` |

**What ran, and why**

| Field | Meaning |
|---|---|
| `strategy` | One of `ALL_ARMS` |
| `definition_version` | `"v1"` or `"v2"` |
| `context` | `"inline"` |
| `selection_policy`, `selection_propensity` | e.g. `"uniform"`, `1/6` |
| `seed` | The per-attempt seed (§2) |
| `executor_config` | Every option in effect: backoff parameters, check flags, responder type |
| `model_id` | The exact model id returned by the provider, or `null` if no LLM was used |

**Budget**

| Field | Meaning |
|---|---|
| `budget_limits` | `{max_attempts, max_wall_ms, max_tokens}` |
| `budget_used` | `{attempts, wall_ms, tokens}` |
| `budget_exceeded` | Set if an overrun happened despite the rules in §3 |

**Outcome**

| Field | Meaning |
|---|---|
| `result.success` | `bool` |
| `result.cost_tokens` | `int` |
| `result.latency_ms` | `float` |
| `result.attempts` | `int`: tool calls issued |
| `result.evidence` | See below |
| `aborted` | `true` only for `question_unmatched` |
| `no_action_taken` | `true` only when VERIFY accepted the original result (§4) |

`result.evidence` always contains:
- `calls`: a list of `{tool, args, http_status, success, injected, injected_kind, error_text (clipped to 500 characters), latency_ms}`
- `answered_by`: the tool whose call succeeded, or `null`
- `llm_calls`: a list of `{purpose, model_id, prompt_tokens, completion_tokens, latency_ms}`
- `reason`: one of `success`, `exhausted_attempts`, `budget_exhausted`, `untranslatable_args`, `question_unmatched`, `unparseable_llm_output`, `checks_rejected`, `original_result_accepted`
- any strategy-specific fields: `replan_chosen_tool`, `reground_chosen_tool`, `same_call`, `question`, `answer`, `checks`

**Storage:** one JSON file per attempt at `<store>/recoveries/<attempt_id>.json`.
`FailureEvent`, `ReplayResult` and existing stores are untouched. A store with
no `recoveries/` directory loads exactly as before.

## 6. Freeze

This document and `backend/recovery/live/data/countriesnow_countries_v2.json`
are committed together, alone, in one commit: the v2 freeze timestamp. v2
attempts may be collected only from that commit on. Any later change goes in
a `v3` file. Smoke and demo runs write to scratch stores, never to a
collection store, and are not data.

## 7. Threats to validity specific to v2

- **CLARIFY is an optimistic upper bound.** With the canned responder, every question gets the pre-authored answer, which was written by the task author knowing what the task needs. CLARIFY's measured success rate is therefore an optimistic upper bound on CLARIFY with a real human. A real human may misread the question, answer only part of it, answer slowly, or not answer at all. Report it as such, never as CLARIFY's real-world efficacy.
- **VERIFY's no-action cases.** VERIFY's headline rate must be read next to the rate excluding `no_action_taken` attempts (§4).
- **The budget defaults decide some outcomes.** `max_attempts`, `max_wall_ms`, `max_tokens` and the backoff parameters are configuration choices (§3). Strategies that use more attempts or tokens (RETRY, VERIFY) are the most sensitive to them. Report the limits next to every rate.
- **SWITCH_TOOL applicability depends on frozen snapshots.** Whether it can be tried at all depends on the frozen country list and resource table (§4). These are point-in-time snapshots of live services, so report untranslatable cases separately.
- **Inline attempts face the live world.** They meet the world as it is at failure time, including peer-tool outages. Use the session's `tool_health` snapshot to separate an outage from a strategy failure.
