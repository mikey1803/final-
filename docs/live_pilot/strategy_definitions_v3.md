# Live-pilot strategy definitions — v3

**Status: FROZEN v3. Approved 2026-10-21 and committed in its own commit
(the v3 freeze timestamp, CLAUDE.md rules 3–4), before any v3 code or data.
Any change goes in a `v4` file.** v1 (`strategy_definitions.md`) and v2
(`strategy_definitions_v2.md`) stay frozen and unchanged. v3 data will never
be merged with v1 or v2 data. The hypotheses, metrics, tests and decision
rules are pre-registered separately in `v3_preregistration.md`.

Everything in v2 carries over unless this document changes it. The changes:
- §1: why v3 is needed
- §2: where a fault came from
- §3: injection persistence modes, plus the plausible-wrong library (§3.6)
- §4: the resend baseline
- §5: agent-side selection failures
- §6: fault codes
- §7: continuing after recovery
- §8: seeding and logging
- §9: decisions recorded and remaining items

---

## 1. Why v3: under transient injection the strategies can't be told apart

The 2026-10-21 Task 21 smoke session (scratch store, not data) captured 4
failures. All 4 were `malformed_request` injections. Three of the four
recovery attempts "succeeded" (re_ground, retry, verify) by re-sending a call.

This follows from how the injector is built; it was not measured.
`FlakyWrapper` draws the injection decision **independently for every call**,
at `rate = 0.20`. So any strategy that sends any call again, with the same or
different arguments to the same or a different tool, gets through with
probability `1 − 0.20 = 0.80` per call. That holds whatever the strategy
does. On injected faults, the measured success of every re-sending strategy
is dominated by that one number, so the recovery-strategy success matrix
can't separate the strategies.

v3 fixes this in two ways:
- **Injection modes (§3)** make injected faults *persist* in the ways real faults do.
- **The resend baseline (§4)** gives every strategy's success rate a paired reference, measured on the same failures.

## 2. Fault origin: never pooled

Every `FailureEvent` and every `RecoveryAttempt` carries `fault_origin`:

| `fault_origin` | Meaning | Example |
|---|---|---|
| `injected` | The wrapper's injection fired on the failing call. `injection_mode` (§3) says which mode. | the 4 failures in the smoke session |
| `natural` | No injection fired, and the real service failed by itself. | `open_meteo` HTTP 503 "The service is overloaded", seen at the 2026-10-21 health check |
| `agent_side` | The model's own output was unusable (§5). | an unparseable tool-selection reply |

**The three origins are never pooled.** Every rate is reported per origin,
and for `injected` also per mode (and per level, §3.3). Natural failures are
rare and not under experimental control. They are reported descriptively, as
evidence of what real faults look like, and not as a cell of the success
matrix.

## 3. Injection persistence modes

**One mode per session block.** Each block runs a single mode, and for
`tool_outage` a single level. The order of blocks is a seeded shuffle of the
session seed (`block_order` stream, §8). Every event and attempt records
`injection_mode` and `block_index`. Modes are never mixed call by call.

The injection *rate* (the share of calls whose first occurrence is
corrupted) stays `0.20` for every mode, so modes differ in persistence, not
frequency.

Shared principle, kept from v1: **error text is genuine.** A timeout is a real
client-side timeout, and a malformed request draws the real server's real
error reply. The one deliberate exception is `degraded_output`, which is
declared and flagged as such.

### 3.1 `transient` (current behaviour, kept for backward compatibility)

| | |
|---|---|
| Mechanism | Unchanged from v1/v2: each call independently draws inject/no-inject; an injected call is a short-timeout call or a malformed request, chosen by a coin flip. |
| Failure types produced | `tool_timeout`, `tool_error` |
| Recovered in theory by | Any strategy that sends any call, with probability ≈ `1 − rate`. The resend baseline gets the same ≈ `1 − rate`, so the expected **gain over the baseline is ≈ 0** for every strategy. Kept so v3 can reproduce v2 conditions and so the §1 confound is measured, not assumed. |
| Seeding and logging | Its own `injection` stream (§8). Each call logs `injected`, `injected_kind`, `injection_mode="transient"`. |

### 3.2 `sticky_args` (corrupted requests only)

| | |
|---|---|
| Mechanism | The first time a call `(tool, normalised args)` is drawn for injection, that key is marked corrupted for the rest of the **episode**. Every later call with the same tool and the same normalised arguments is sent as the same **malformed request**, so the real server returns its real error. A different argument value or a different tool gives a new key with a fresh draw. |
| No timeouts | `sticky_args` never produces timeouts. Timeouts belong to `transient` and `tool_outage`. |
| Normalised args | Canonical JSON (sorted keys). Strings are stripped and case-folded, numbers parsed as floats, so `{"country": "india "}` and `{"country": "India"}` are the same key. |
| Real-world analogue | Arguments the service can't resolve, e.g. a misspelled entity or an out-of-range id. Sending them again never helps. |
| Failure types produced | `tool_error` |
| Recovered in theory by | `switch_tool` (different tool and key), `replan` (shown the error, asked for a different approach). `re_ground` v2 and `clarify` only when the model actually changes tool or arguments; their `same_call` flag shows whether it did. |
| Not recovered by | `retry` (any backoff), `null_control`, `verify` (its re-issue is the same call), `re_ground` v1, the resend baseline |
| Seeding and logging | Its own `injection` stream; the decision is drawn once per key. Each call logs `injection_key` (sha256[:16] of the normalised key), `sticky_hit` (true when the corruption was inherited, not freshly drawn), and `injected_kind="malformed_request"`. |

### 3.3 `tool_outage` (length counted in calls, three fixed levels)

| | |
|---|---|
| Mechanism | Each tool has its own outage state. When a call to tool T is drawn for injection and T is not already in an outage, an outage of **N calls** starts. The triggering call is call 1 of N, so it and the next N − 1 calls **to T** fail as genuine short client-side timeouts; call N + 1 is served normally. Calls to other tools don't count toward T's N. The peer tool has an **independent** outage state at the same rate (symmetric, as in v1), so both members of a pair can be out at once. |
| Levels | **N ∈ {2, 5, 20}**, one level per block, each reported separately and never blended. These are **configuration choices, not measured values**; they are to be replaced by drift-atlas outage measurements in a later version (a new version, not an edit). |
| Why calls, not seconds | It keeps sessions fast and deterministic. Consequence, stated up front: **waiting cannot end an outage under this design.** Only further calls to T end it. `null_control` and retry's backoff delays therefore get no benefit from the wait itself. That follows from the design, and it is not evidence about waiting in the real world. This is accepted as a stated design property (decision 2026-10-21). A time-based outage mode may be added in a later version, once the drift atlas (C3) has measured real outage durations. |
| Real-world analogue | A service down for a period, like the open_meteo 503 above. |
| Failure types produced | `tool_timeout` |
| Recovered in theory by | `switch_tool`, unless the peer is also out. `replan`, `re_ground` v2 and `clarify` when they pick the other tool. `retry` v2 only if its attempts outlast the outage: with up to 3 attempts after the triggering call it recovers at N = 2 (its 2nd attempt is call 3), but not at N = 5 or 20. |
| Not recovered by | `verify` (one re-issue, which is call 2), `null_control` (one call after a wait the outage ignores), `re_ground` v1, the resend baseline |
| Logged on every record | `outage_N`, `outage_id`, `outage_call_index` (1…N), `in_outage` |
| Seeding | Its own `outage` stream, which decides when an outage starts. N is fixed by the block. |

### 3.4 `degraded_output` (sticky per key for the whole episode)

| | |
|---|---|
| Mechanism | The real call is made and its real response is received. **The wrapper then modifies the body** and returns HTTP 200. The first time a key `(tool, normalised args)` is drawn for degradation, one transformation is chosen for that key, and it applies to **every call with that key for the whole episode**. A different tool or different arguments gives a new key with a fresh draw. |
| Transformations | **(a) structural:** remove the answer field, change its JSON type, or truncate the body at 50% of its bytes. **(b) plausible-wrong:** replace the answer field with a wrong value of the same kind (§3.6). |
| Exception to the genuineness principle | This is the **only** mode with synthetic content, and it is declared on every record: `synthetic_content=true`, the transformation name, sha256 of the original body, and sha256 of the delivered body. |
| Failure types produced | (a) `malformed_output`; (b) `hallucinated_output`, a confident, well-formed, wrong answer |
| How the failure is surfaced | HTTP is 200, so the v2 loop would not detect it. The instrument (oracle) knows a degradation happened, and it hands that failure to the recovery layer like any other captured failure, with `detected_by="oracle"` on the **event**. Whether a *strategy* also detects it is recorded separately (next row). |
| Two-part outcome, recorded separately | **`detected`:** some check run *by the strategy* rejected the degraded result (verify's schema check or LLM judge, or a strategy's own check). **`corrected`:** the episode's final answer field matches the **genuine value** recorded by `FlakyWrapperV3` before degrading (§3.6). This is an instrument-side comparison, never the verify LLM judge, which would make the measure circular. |
| Credit rule | VERIFY can earn **detection** credit. **Correction** credit goes only to the strategy whose final result passes the content check, usually switch_tool or replan. The split is intentional: Task 23 will use verify's detection as diagnostic evidence. HTTP-level "tool criterion" success is not used in this mode. |
| Seeding and logging | Its own `degradation` stream decides whether the key degrades, which transformation, and which wrong value (§3.6). Logged as described above. |

### 3.5 Hypotheses (theory only; pre-registered separately in `v3_preregistration.md` before any v3 code)

✓ = expected to recover · ~ = only under the stated condition · ✗ = not expected

| Mode | retry (v2) | null_control | switch_tool | verify | re_ground v2 | replan | clarify | resend baseline |
|---|---|---|---|---|---|---|---|---|
| transient | ✓ ≈ 1 − r | ✓ ≈ 1 − r | ✓ ≈ 1 − r | ✓ ≈ 1 − r | ✓ ≈ 1 − r | ✓ ≈ 1 − r | ✓ ≈ 1 − r | ✓ ≈ 1 − r (gain ≈ 0 for all) |
| sticky_args | ✗ | ✗ | ✓ | ✗ | ~ if args change | ✓ | ~ if args change | ✗ |
| tool_outage N = 2 | ✓ (2nd attempt) | ✗ | ✓ unless peer out | ✗ | ~ if tool changes | ~ if tool changes | ~ if tool changes | ✗ |
| tool_outage N = 5 | ✗ | ✗ | ✓ unless peer out | ✗ | ~ if tool changes | ~ if tool changes | ~ if tool changes | ✗ |
| tool_outage N = 20 | ✗ | ✗ | ✓ unless peer out | ✗ | ~ if tool changes | ~ if tool changes | ~ if tool changes | ✗ |
| degraded_output: detected | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| degraded_output: corrected | ✗ | ✗ | ✓ unless peer key degraded | ✗ | ~ if call changes | ~ if call changes | ~ if call changes | ✗ |

Whether **measured** rates match this table is the v3 experiment. Nothing in
this table is a result.

### 3.6 Plausible-wrong library, genuine-value capture and the `corrected` check

**No existing content check.** The task manifest has **no** success check
that reads an answer field. `Task` has no such field, and
`experiments/build_pilot_manifest.py` contains hand-authored goals only. v3
therefore defines the answer field per tool (approved table below) and checks
it on the instrument side.

**Genuine-value capture.** On every degraded call, `FlakyWrapperV3` reads the
answer field from the **real** response *before* degrading it. It stores
`{genuine_value, genuine_sha256}` **only in the injection log**: never in the
body, the history, the evidence or anything else shown to the agent or to an
LLM.

Genuine values are captured on **every** call to an answer-bearing tool in a
`degraded_output` block, whether or not that call is degraded. That way a
strategy's final call usually has its own genuine reference.

**The `corrected` check (instrument-side; never the verify LLM judge).**
`corrected = true` when the **episode's final answer field** matches the
reference genuine value within the field's tolerance:
- **Reference.** If the final value came from a call that had its own genuine value captured, that is the reference. Otherwise, e.g. when a strategy switched to a tool that captured nothing, the reference is the degraded call's captured genuine value, compared within tolerance.
- **Strings:** exact match after normalisation, i.e. Unicode NFC, whitespace collapsed and stripped, case-folded.
- **Temperatures:** `|final − reference| ≤ 3 °C`. switch_tool reads a different provider (open_meteo ↔ wttr), so their genuine readings can legitimately differ. **The 3 °C tolerance is a configuration choice, not a measured value.**
- **Log.** The reference used (`same_call` / `degraded_call`), the tolerance and the absolute difference are logged on every attempt.

**Wrong values are plausible and deterministic.** For each degraded key, the
`degradation` stream draws:
- an integer offset `k ≥ 1` for library fields, and
- a temperature offset `δ` with `|δ| ∈ [8, 15] °C` and a random sign, for temperature fields. **The 8–15 °C range is a configuration choice, not a measured value.**

A library value is replaced by the entry `k` positions after the true value
(wrapping around, and skipping entries equal to the true value). Libraries are
captured once from the real services and frozen with v3, like the v2 country
list.

| Tool | Answer field | Kind | Plausible-wrong perturbation | Library to freeze |
|---|---|---|---|---|
| countriesnow | `data.capital` | capital city name | The capital of the country `k` places after the true country in the frozen list | countriesnow `/countries/capital`: 250 `(country, capital)` pairs, captured once |
| wikipedia | `extract` | article summary | The summary of a different real article of the same kind. For a city title: the extract of the capital city `k` places along the frozen capitals list. For any other title: the article `k` places along a fixed list of other titles of the same kind. | Real extracts for the frozen capitals list, plus a small fixed list for non-city task titles (currently `Python_(programming_language)`), captured once |
| open_meteo | `current_weather.temperature` | °C, number | `t + δ` (rounded to 1 decimal place, as the API reports) | None needed (seeded offset) |
| wttr | `current_condition[0].temp_C` | °C, integer as a string | `str(round(int(t) + δ))` | None needed |
| jsonplaceholder | `title` | todo title | The title of todo `((id − 1 + k) mod 200) + 1` | JSONPlaceholder `/todos`: 200 real titles, captured once |
| dummyjson | `todo` | todo text | The text of todo `((id − 1 + k) mod 254) + 1` | DummyJSON `/todos?limit=0`: 254 real texts, captured once |

Structural transformations (a) apply to the same answer field: remove it,
change its type (string ↔ number), or truncate the body. One of the three is
chosen per key by the `degradation` stream.

**What the verify judge can and cannot detect, in principle.** The judge
sees the step goal and the result, nothing else.
- **It can catch some plausible-wrong values from world knowledge**, e.g. a capital that belongs to another country, or an extract about a different city.
- **It cannot catch others in principle.** A temperature shifted by 8–15 °C is plausible, and nothing in the prompt reveals the true reading. A wrong todo title is arbitrary text.

Structural transformations are visible to the schema check whatever the field.
**Detection rates are therefore reported per field (and per transformation),
never pooled across fields.** A pooled verify detection rate would mostly
reflect the mix of fields in the sample.

**`task_completed`** uses the same answer fields, tolerances and reference
rule. Its per-tool definition is in §7, next to `continue_after_recovery`.

## 4. Resend baseline (recorded for every captured failure)

| | |
|---|---|
| What | One immediate, identical re-send of the failing call: same tool, same arguments, no wait, no strategy. It is a measurement probe, not one of the drawn arms: it is never selected, has no propensity, and is not a strategy. |
| When | Immediately after the failure is captured, **before** the drawn strategy runs. |
| A read-only probe of the injection state | The probe **never advances or creates** injection state. It does not count as one of an outage's N calls, it does not create a sticky or degraded key, and its own random draws (transient mode) come from a separate `resend` stream. |
| What it sees | The state as it is: inside an outage it fails; on a sticky or degraded key it gets the same corruption. |
| Common random numbers | The drawn strategy therefore faces exactly the same world with or without the baseline, in every mode. |
| Effect on the task | None. Its result is recorded only. Task progress, and `continue_after_recovery` (§7), depend only on the strategy's outcome. |
| Record | `resend_baseline: {success, corrected (degraded mode), http_status, injected, injected_kind, sticky_hit, in_outage, latency_ms, error_text}` on the RecoveryAttempt, plus a standalone row, so failures whose strategy was not drawn still have a baseline. |
| How it is read | For each strategy, a **paired** comparison on the same failures: `P(strategy succeeds) − P(resend succeeds)`, clustered by `event_id` (and by `episode_id`, §7). A strategy is credited only with its gain over the resend baseline. |
| Retry v1 is retired as a drawn arm | `retry` **v1** (one immediate identical re-issue) is identical to the resend baseline, so in v3 it is **no longer drawn**. The v3 draw is retry (v2 configuration), replan, switch_tool, re_ground (v2), verify (v2) and clarify, plus null_control once `wait_s` is frozen. The v1 configuration (`RetryExecutor.v1()`) stays runnable for Phase 2 replay and backward compatibility. |

## 5. Agent-side failures: unusable tool selection (decision 2026-10-21)

v2 skips these and counts them (`recovery_not_applicable_selection_failure`).
In v3 they become ordinary **agent-side `FailureEvent`s** with
`fault_origin="agent_side"`, and strategies apply to them like any other
failure.

| Provisional tag (rule, set at capture) | When |
|---|---|
| `malformed_output` | The selection reply is not valid JSON, or lacks `tool` / `args`. |
| `wrong_tool_selected` | Valid JSON, but `tool` is not one of the six tools, or the args don't match that tool's argument names. |

The provisional tag is the instrument's own rule and is stored as
`agent_side_kind`. **The human `true_type` label is still the ground truth**;
the tag is never copied into `true_type`.

What each strategy does when the failing "call" is the selection step:

| Strategy | v3 behaviour on a selection failure |
|---|---|
| retry | Run the same selection prompt again, with the v2 backoff. Success: a valid call is produced **and** it succeeds by its tool criterion. |
| null_control | Wait `wait_s`, then run the selection prompt again once. |
| replan | Selection prompt plus the failure text ("your previous answer was unusable: …"), asking for one call. |
| re_ground | Selection prompt plus goal and last good state, without the failure. |
| verify | Judge the selection reply. If rejected, run the selection again once, then check that call's tool result as in v2. |
| clarify | One question, the canned answer, then selection. |
| switch_tool | `wrong_tool_selected` with a known tool: substitute the registry peer with translated args. Otherwise `untranslatable_args`. |
| resend baseline | Run the identical selection prompt again once. |

## 6. Fault codes `llm_error` and `executor_error` (decision 2026-10-21)

v2 already records these two `evidence.reason` values, which are not in v2's
frozen list:
- `llm_error`: the language-model provider failed, i.e. an HTTP error, a timeout, or a malformed response envelope.
- `executor_error`: a bug inside an executor.

Both are faults in the instrument, not strategy outcomes.

**Rule:** attempts with either code are **excluded from strategy success
rates** and reported as a **separate count**, per strategy and in total.

**Bias note:** only LLM-using strategies (replan, re_ground v2, verify,
clarify) can produce `llm_error`. If provider errors cluster at hard moments,
excluding them could flatter those strategies. The `llm_error` count is
therefore always reported next to their rates.

## 7. `continue_after_recovery` (decision 2026-10-21)

| | |
|---|---|
| Flag | `continue_after_recovery: bool`. **v2 collection: `false`** (current behaviour; the task stops at its first failure). **v3 default: `true`.** |
| Behaviour when true | After a successful recovery, the recovered tool result is appended to the history as the step's result, and the loop moves on to the next step. A later failure in the same task gets its own `FailureEvent` and its own drawn strategy. Still one strategy per failure; multi-strategy sequences are Task 23's scope. |
| New per-attempt fields | `task_completed` (bool, written when the task ends), `episode_id` (one per task run), `failure_ordinal` (1, 2, … within the episode), `prior_attempt_ids` |
| Why | Needed for Task 23's multi-attempt episodes, and for the task-level outcome (did the *task* finish?), which per-step success hides. |
| Independence caveat | Attempts within one episode are not independent: later steps depend on earlier recoveries. Analysis must cluster by `episode_id` as well as `event_id`. |

**`task_completed` (decision 2026-10-21).** Written once, when the task
ends. It is `true` when the task's answer field is present, parses as its
kind, and, **in `degraded_output` blocks only**, matches the genuine value
under the §3.6 tolerance and reference rule. Outside `degraded_output` blocks
no genuine-value comparison is made. Per tool:

| Tool | Present and parses | In `degraded_output` blocks, also |
|---|---|---|
| countriesnow | `data.capital` is a non-empty string | matches genuine (string rule, §3.6) |
| wikipedia | `extract` is a non-empty string | matches genuine (string rule, §3.6) |
| open_meteo | `current_weather.temperature` is a number | within 3 °C of genuine |
| wttr | `current_condition[0].temp_C` parses as an integer | within 3 °C of genuine |
| jsonplaceholder | `title` is a non-empty string | matches genuine (string rule, §3.6) |
| dummyjson | `todo` is a non-empty string | matches genuine (string rule, §3.6) |

A multi-step task is completed when its **last** step's answer field
satisfies its row. A task that stops at an unrecovered failure has
`task_completed = false`.

## 8. Seeding and logging

**Seeds.** All per-session randomness comes from one recorded `session_seed`,
split with `numpy.random.SeedSequence(session_seed).spawn(...)` into
independent named streams:

| Stream | Drives |
|---|---|
| `block_order` | the seeded order of mode blocks (§3) |
| `selection` | which strategy is drawn (per failure, via `spawn_key=(failure_index,)` as in v2) |
| `jitter` | retry backoff |
| `injection` | per-call or per-key injection decisions (transient, sticky_args) |
| `resend` | the resend baseline's own transient draws (§4) |
| `outage` | outage starts |
| `degradation` | which keys degrade, which transformation, and the wrong-value offset `k` |

Never `hash()`. Every derived seed is recorded on the record that used it.

**Wrapper.** The v3 modes live in a new versioned wrapper, `FlakyWrapperV3`.
The v1 `FlakyWrapper` is not changed, so v1 and v2 runs reproduce exactly as
before.

**New fields:**

| On | Fields |
|---|---|
| Every event | `fault_origin`, `injection_mode`, `block_index`, `injection_params` (rate, mode parameters such as `outage_N`), `agent_side_kind` (§5), `detected_by` (degraded mode) |
| Every call record | Mode-specific fields from §3: `injection_key`, `sticky_hit`, `outage_N`, `outage_id`, `outage_call_index`, `in_outage`, `synthetic_content`, body hashes, transformation |
| Injection log only (never shown to the agent or an LLM) | `genuine_value`, `genuine_sha256`, `delta` / `k` drawn (degraded mode) |
| Every attempt | `resend_baseline` (§4); `continue_after_recovery`, `task_completed`, `episode_id`, `failure_ordinal`, `prior_attempt_ids` (§7); `detected` and `corrected`, plus the `corrected` reference used, tolerance and absolute difference (degraded mode) |

## 9. Decisions recorded, and what is still open

**Decided on 2026-10-21:**

| Question | Decision |
|---|---|
| degraded_output persistence | Sticky per `(tool, normalised args)` for the whole episode, with a two-part `detected` / `corrected` outcome. Verify can earn detection credit; correction credit goes only to the strategy whose final result passes the genuine-value check. |
| `corrected` check | Instrument-side genuine-value comparison, never the LLM judge. Strings exact after normalisation; temperatures within 3 °C (a configuration choice). |
| Wrong values | Plausible only: temperatures `±[8, 15] °C` (a configuration choice), library fields via the seeded offset `k`. The out-of-range +60 °C rule is dropped. |
| Answer-field table | Approved as proposed (§3.6). |
| sticky_args | Corrupted requests only; no sticky timeouts. |
| tool_outage | Counted in calls, at three fixed levels N ∈ {2, 5, 20}, reported separately. These are configuration choices; N is recorded on every record. |
| Mixing modes | One mode per session block, in a seeded block order. |
| Retry v1 | Retired as a drawn arm; still runnable for replay. |
| `task_completed` | Per-tool table in §7; the last step's field counts for multi-step tasks. |
| Waiting under call-counted outages | Waiting can never end an outage, so `null_control` gets no benefit by design. Accepted as a stated design property (§3.3); a time-based outage mode may follow once the drift atlas measures real outage durations. |

**Still open:** nothing.
