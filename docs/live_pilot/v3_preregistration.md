# Live pilot v3 — pre-registration (hypotheses, metrics, tests, decision rules)

**Status: PRE-REGISTERED. Committed in its own commit, after the v3 freeze
(`strategy_definitions_v3.md`, commit `c1705bf`) and BEFORE any v3 code is
written or any v3 data is collected (CLAUDE.md rules 3–4).** That commit's
hash is the timestamp. Nothing below was written after seeing v3 data; none
exists. Any change after that commit goes in `v3_preregistration_v2.md` and
in the deviations log (§11), never in an edit to this file.

**What this document covers.** It says what the v3 experiment expects, how
outcomes are measured, which intervals and verdicts are computed, and what
each verdict means. The instrument itself (injection modes, the resend
baseline, fault origins, `continue_after_recovery`, `task_completed`, the
`corrected` check and the plausible-wrong library) is defined in
`strategy_definitions_v3.md` and is not repeated here. Section numbers
written "v3 §n" refer to that document.

**What this document does not cover.** Task 23 (diagnosis-aware recovery)
and Task 24 (drift-gated recovery) will consume v3 results; their own
analyses are pre-registered separately. Natural and agent-side failures are
described here but are not hypothesis tests (§7.4).

---

## 1. Instrument and fixed parameters

| Item | Fixed value | Source |
|---|---|---|
| Strategy and instrument definitions | `docs/live_pilot/strategy_definitions_v3.md` | commit `c1705bf` |
| Executors | v2 executors (`build_executors("v2")`); retry v2 = up to `max_attempts = 3` re-sends with full-jitter backoff (base 0.5 s, cap 4 s) | `strategy_definitions_v2.md` |
| Budget per attempt | `max_attempts = 3`, `max_wall_ms = 40000`, `max_tokens = 1500` (configuration choices, v2 §budget) | v2 |
| Drawn arms | retry (v2), replan, switch_tool, re_ground (v2), verify (v2), clarify; uniform, propensity 1/6 | v3 §4 |
| `null_control` | **Not drawn in v3**: its `wait_s` is still `[RESULT REQUIRED]` (v2). Its v3 §3.5 predictions are recorded but are not tested by this pre-registration. | v2, v3 §4 |
| Injection rate `r` | 0.20 in every mode | v3 §3 |
| Outage levels | N ∈ {2, 5, 20} calls (configuration choices) | v3 §3.3 |
| Language model | Groq `openai/gpt-oss-20b`, `reasoning_effort="low"`, temperature 0 | `recovery/live/executors/llm.py` |
| Clarify responder | canned answer | v2 |
| `continue_after_recovery` | `true` | v3 §7 |
| Task list | the 8 `STARTER_TASKS` in `backend/experiments/run_live_collection.py` at commit `c1705bf`: `wiki_basic_01`, `country_basic_01`, `weather_basic_01`, `weather_basic_02`, `placeholder_basic_01`, `ambiguous_01`, `out_of_scope_01`, `long_context_01`. They are not edited during v3 collection. | code at `c1705bf` |

## 2. Collection design

**Blocks.** There are six block types:

| Block | Mode |
|---|---|
| T | `transient` |
| S | `sticky_args` |
| O2, O5, O20 | `tool_outage` with N = 2, 5, 20 |
| D | `degraded_output` |

**Sessions.**
- One **session** runs all six blocks once, in the seeded `block_order` (v3 §8).
- **Passes per block (a configuration choice):** the T block runs **R = 10 passes** over the task list; the S, O2, O5, O20 and D blocks each run **R = 5 passes**.
  - **Why T is larger:** it gives P1 (§9.2) enough attempts.
  - **The imbalance cannot bias any estimate:** every estimate and interval is computed **within one cell** (one mode × strategy, or one stratum), and no estimate pools attempts across blocks. P1 pools across arms only *within* the T block, where every arm shares the same passes. P2 aggregates cell *verdicts*, not attempts.
- Session `s` (s = 1, 2, …) uses `session_seed = 231000 + s`.
- Seeds are never chosen, skipped or re-drawn because of outcomes.

**Stopping rule.**
- After each completed session, a count-only script reports the number of included attempts (§3.3) in every verdict cell (§9.1). It reads no outcomes.
- Collection stops when every unstratified verdict cell has **at least n_min = 30** included attempts, or when **S_max = 60** sessions have been run, whichever comes first. Both numbers are configuration choices.
- A cell still below n_min at stop gets the verdict **insufficient data**. Its estimate and interval are still reported.
- There is no interim analysis of outcomes, and no stopping for success or futility.

**Not data.**
- The live smoke session that checks the v3 code (one block per mode, scratch store) is excluded from the dataset.
- So are any runs made before the first collection session is started with a recorded seed.
- The dataset is the v3 store written by the registered collection command (§12).

**Instrument halts.**
- **Halt H1.** Collection stops if, **on a failure with `fault_origin = injected`**, a session shows either of the following. Both are impossible by construction (v3 §4), so their occurrence means the instrument is faulty.
  - a resend-baseline success under S, O2, O5 or O20;
  - a resend-baseline `corrected = true` under D.
- **Natural failures inside these blocks do not trigger H1.** A `fault_origin = natural` failure, e.g. a real 503 that clears, can legitimately be fixed by a resend. It is tagged (`h1_exempt_natural = true`), counted, reported under §7.4, and collection continues.
- **Halt H2.** Collection also stops if any runtime invariant in `recovery/guards/invariants.py` fails.
- **After a halt:** that session's data is quarantined, reported and excluded from analysis. The fix goes into a new version under rule 3, and is recorded in §11.
- **Natural successes are not a halt.** A malformed request that a real server happens to accept is not an H1 event if no injection fired. It is recorded as `fault_origin = natural`.

## 3. Units, outcomes and inclusion

### 3.1 Units

| Unit | Definition |
|---|---|
| Attempt | One drawn strategy applied to one captured failure: a `RecoveryAttempt` with `definition_version = "v3"`. |
| Resend observation | The paired resend-baseline result recorded for the same failure (v3 §4). |
| Episode | One run of one task, identified by `episode_id`. It is the cluster for every interval (§8). |

### 3.2 Outcome per attempt

| Mode | Primary outcome `y` | Secondary |
|---|---|---|
| T, S, O2, O5, O20 | `success` (v2 tool criterion) | — |
| D | `corrected` (instrument-side genuine-value check, v3 §3.6) | `detected` (§6) |

The resend observation's outcome `b` uses the same definition. In D blocks
`b` is the resend's `corrected`, which by construction is false.

### 3.3 Inclusion and exclusion

| Rule | Effect |
|---|---|
| `fault_origin = injected` | Only these attempts enter hypothesis cells. `natural` and `agent_side` attempts are reported separately (§7.4) and never pooled. |
| `evidence.reason ∈ {llm_error, executor_error}` | Excluded from all rates. Counted separately per strategy and in total, and always shown next to the LLM-using strategies' rates (v3 §6). |
| `untranslatable_args` (switch_tool) | **Included** as a failure in ITT (§4.1). Counted. |
| `no_action_taken` (verify) | Included as a failure in ITT. Verify's rates are reported both including and excluding these attempts (v2 rule). |
| `budget_exceeded` | Included. Its outcome is the outcome as recorded. Counted. |
| Quarantined sessions (§2) | Excluded entirely. Reported. |

## 4. Expected recovery per mode

### 4.1 Two estimands

The v3 §3.5 predictions describe a *mechanism*: what happens when a strategy
actually sends a call. A strategy can also fail to send one. switch_tool can
hit `untranslatable_args`, for example `wiki_basic_01`'s title "Kolkata" is
not in the frozen country list. An LLM strategy can produce an unusable
call. So two quantities are reported for every cell:

- **Δ_ITT (intention to treat):** over all included attempts, `mean(y − b)`. This is the practical gain, the number a deployment would see.
- **Δ_M (mechanism):** over included attempts that issued at least one tool call (`len(evidence.calls) ≥ 1`), `mean(y − b)`. The §4.2 predictions are stated for Δ_M.

Δ_M conditions on something that happens after the strategy is drawn. It is
still a valid within-failure contrast, because the resend is paired on the
same failures. But Δ_M of different strategies covers different subsets of
failures, so cross-strategy comparisons use the randomized ITT rates
(§9.2, P3).

**Applicability rate.** Every strategy × mode cell reports the
**applicability rate**: the fraction of included attempts that actually sent
at least one tool call. It is reported with its interval. It is the
denominator ratio linking Δ_ITT and Δ_M.

### 4.2 Predicted Δ_M, per mode and strategy

Notation:
- `r = 0.20`.
- "fresh" means the strategy's call has a different injection key (S, D) or a different tool (O*) from the failing call, so it gets a fresh draw that succeeds with probability ≈ `1 − r = 0.80`.
- The resend baseline's expected success is ≈ `0.80` in T and `0` in every other mode, by construction (v3 §4).

Predicted direction and approximate size of Δ_M:

| Mode | retry (v2) | switch_tool | verify | re_ground v2 | replan | clarify |
|---|---|---|---|---|---|---|
| T | **+ (≈ +0.19)** | 0 | 0 | 0 | 0 | 0 |
| S | 0 | + (≈ +0.80) | 0 | ~ | + (≈ +0.80 if fresh) | ~ |
| O2 | + (≈ +1.0) | + (≈ +0.80, unless peer out) | 0 | ~ | ~ | ~ |
| O5 | 0 | + (≈ +0.80, unless peer out) | 0 | ~ | ~ | ~ |
| O20 | 0 | + (≈ +0.80, unless peer out) | 0 | ~ | ~ | ~ |
| D (`corrected`) | 0 | + (≈ +0.80, unless peer key degraded) | 0 | ~ | ~ | ~ |

`+` = positive gain expected · `0` = no gain expected · `~` = conditional; see §4.3.

**Discrepancy with the frozen v3 §3.5 table, recorded here openly.** v3 §3.5
gives retry (v2) under `transient` as "≈ 1 − r", i.e. no gain over the
resend baseline. That is wrong for retry v2. It sends up to 3 independent
re-sends in transient mode, so its expected success is ≈ `1 − r³ ≈ 0.99`, and
its expected gain is ≈ `+0.19`. This pre-registration uses the corrected
expectation. `strategy_definitions_v3.md` is frozen and is not edited (rule
3); the discrepancy is listed in §11 as D-1. The other strategies send one
call each in T mode (verify's single re-send, replan's, re_ground's and
clarify's chosen call, switch_tool's peer call), so "gain ≈ 0" stands for
them.

**Where the approximate values come from.** They follow from the injection
mechanics and nothing else:
- **Sources of shortfall.** Natural failures and budget limits lower every value. The `corrected` tolerance (v3 §3.6) can lower the D values.
- **O2 retry.** The triggering call is outage call 1, and retry's first re-send is call 2. The second re-send is call 3, which v3 §3.3 serves normally, so ≈ 1.0.
- **Sticky retry and verify.** Their re-sends hit the same key, so they get 0.

These values are expectations, not results.

### 4.3 Conditional (`~`) cells: stratified predictions

Each `~` cell is split by whether the condition holds. The condition is read
from the attempt's own call records:

| Mode | Condition "fresh" holds when |
|---|---|
| S, D | the strategy's final call has an `injection_key` different from the failing call's key (normalised arguments, v3 §3.2) |
| O2, O5, O20 | the strategy's final call goes to a different tool from the failing call |

Prediction:

| Stratum | Predicted Δ_M |
|---|---|
| fresh | + (≈ +0.80) |
| not fresh | 0 |

Exception: retry-like outcomes at O2 are not expected from the LLM strategies, because each sends a single call.

The share of attempts in the fresh stratum is reported descriptively. It
shows how often each LLM strategy actually changes its approach.

**Not tested.** `null_control` (not drawn, §1). The v3 §3.5 detection row is
replaced by the per-field predictions in §6.

## 5. The resend-baseline comparison

**Pairing.** For every included attempt `i`, the strategy's outcome `y_i` is
paired with the resend outcome `b_i` measured on the **same failure**,
immediately before the strategy ran. The resend uses its own `resend` stream
and never changes injection state (v3 §4). The strategy's world is therefore
the same as it would be without the baseline (common random numbers), and
`y_i − b_i` is a within-failure contrast.

**Estimator.** For each cell, `Δ = Σ_i (y_i − b_i) / n` over the cell's
included attempts. This is computed for both Δ_ITT and Δ_M.

**Interval.** A cluster bootstrap over episodes (§8). Clustering by episode
also covers the event level, since each event has exactly one attempt.

**Crediting rule.** A strategy is credited only with its gain Δ over the
resend baseline, never with its raw success rate. Raw rates are reported
next to Δ.

**Manipulation checks (reported, never verdicts on strategies).**

| Check | Expected | If not met |
|---|---|---|
| Resend success in T | 95% interval contains 0.80 | Reported. Investigated before any T verdict is interpreted. Not a halt. |
| Resend success in S, O2, O5, O20 (`injected` failures) | exactly 0 | Halt H1 (§2). Natural failures are exempt and tagged. |
| Resend `corrected` in D (`injected` failures) | exactly 0 | Halt H1 (§2). Natural failures are exempt and tagged. |
| Observed injection rate per mode (fresh draws ÷ eligible calls) | 95% interval contains 0.20 | Reported. Investigated. |

## 6. Per-field detection reporting (D blocks)

**Never pooled.** `detected` is reported per answer field and per
transformation family:
- **structural:** remove the field, change its type, truncate the body.
- **plausible-wrong.**

There is no pooled detection rate. Detection is attributed to verify. The
other strategies have no content check, and their `detected` rate is
reported as a check expected to be 0.

**Predictions for verify's detection:**

| Field | Structural | Plausible-wrong |
|---|---|---|
| countriesnow `data.capital` | ✓ high | ~ possible from world knowledge (descriptive only, no verdict) |
| wikipedia `extract` | ✓ high | ~ possible from world knowledge (descriptive only, no verdict) |
| open_meteo `current_weather.temperature` | ✓ high | ✗ not detectable in principle |
| wttr `current_condition[0].temp_C` | ✓ high | ✗ not detectable in principle |
| jsonplaceholder `title` | ✓ high | ✗ not detectable in principle |
| dummyjson `todo` | ✓ high | ✗ not detectable in principle |

**False-alarm baseline (offline calibration, decision 2026-10-21).** A
rejection rate means detection only when compared with how often verify
rejects *genuine* results. The false-alarm rate is measured offline, after
collection:

| | |
|---|---|
| Items | Delivered bodies of **non-degraded** calls to answer-bearing tools in **D blocks only**. `FlakyWrapperV3` stores them in the injection log during collection: tool, args, delivered body, its sha256, `episode_id`, and the goal of the step the call served. |
| Sample | **50 items per field**, drawn without replacement with `FA_SAMPLE_SEED = 23002`. The draw uses one `SeedSequence(23002).spawn` child per field, in the field order of the table above. If a field has fewer than 50 eligible items, all are used and n is reported. |
| Judging | Each sampled item goes through verify's v2 pipeline once: schema check, then LLM judge. There are no retries and no re-judging. An item whose judge call ends in `llm_error` is excluded and counted. |
| Settings | Exactly those used during collection: the same model id, judge prompt template (`recovery/live/executors/prompts.py` at the collection commit), temperature, `reasoning_effort`, max completion tokens and step goal. All are recorded in the calibration output, together with the model id the provider reports back. |
| Output | Per field: the false-alarm rate of the schema check alone (deterministic), and of the full verify pipeline. |
| Consistency with v3 §3.6 | The frozen rule bans showing the agent an answer key during an episode. This calibration judges, offline, bodies the agent already received during collection. It reveals nothing during any episode, so it is consistent with that rule. |

**Detection claim rule.** Detection is **claimed** for a field ×
transformation family only when the detection rate exceeds that field's
false-alarm rate by the margin: the 95% interval of `detected rate −
false-alarm rate` must have its lower bound above **m_det = 0.15**. m_det is
a configuration choice.

The interval comes from independent bootstraps of the two rates (§8), and
the difference is taken per resample:
- the detection rate is resampled by episode;
- the false-alarm items are resampled by episode as well.

**When the rule is not met,** the raw detection and false-alarm rates are
reported with **no detection claim**.

**Verdicts on the §6 predictions** (thresholds are configuration choices):
- **✓ high** is supported if detection is **claimed** and the detection rate's 95% interval has lower bound > 0.50. It is contradicted if the upper bound of `detected − false-alarm` is < m_det, or the detection rate's upper bound is < 0.50.
- **✗** is supported if the upper bound of `detected − false-alarm` is < m_det. It is contradicted if detection is claimed.
- Otherwise the cell is **inconclusive**.

The `~` fields (capital, extract) get no verdict. Detection is still claimed
or not claimed for them under the same rule.

**Correction remains separate.** Detection never earns correction credit. A
verify attempt that detects but whose final answer is not genuine has
`detected = true, corrected = false` (v3 §3.4).

## 7. Metrics

### 7.1 Primary
Δ_ITT and Δ_M per (mode, strategy) cell, and per stratum for `~` cells, each
with its 95% interval and n (§4, §5).

### 7.2 Secondary
- Raw success (or `corrected`) rate per cell, and the resend rate per cell, each with its interval.
- Applicability rate per strategy × mode (§4.1). For switch_tool, the `untranslatable_args` share. For verify, the `no_action_taken` share.
- Verify `detected` per field × transformation family (§6).
- **`task_completed`** rate per mode. Attribution:
  - **Single-failure episodes:** credited to that failure's strategy. The resend baseline has no task-level counterfactual, so `task_completed` is compared across strategies by raw rates only, which are randomized.
  - **Multi-failure episodes:** reported descriptively by strategy sequence.
  - **`out_of_scope_01`** has no answer-bearing tool, so it is excluded from `task_completed` rates. It still contributes step-level attempts.
- The `llm_error` and `executor_error` counts per strategy (v3 §6).
- Latency (ms) and tokens per attempt, as median and IQR per cell.

### 7.3 Descriptive
- The fresh-stratum share per `~` cell (§4.3).
- The `corrected` reference used (`same_call` / `degraded_call`), and the absolute difference for temperatures (v3 §3.6).
- Outage overlap: the share of O* switch_tool attempts whose peer was also out.

### 7.4 Separate origins (reported, no verdicts)
- `natural` failures: their count, tool, error text and strategy outcomes, reported as evidence of what real faults look like (v3 §2).
- `agent_side` failures: by `agent_side_kind`, with strategy outcomes per v3 §5. These are **exploratory**, with no pre-registered prediction.

## 8. Intervals

**Cluster bootstrap.**

| Parameter | Value |
|---|---|
| Resampling | Within each cell, **episodes** (`episode_id`) are resampled with replacement, keeping every included attempt of a resampled episode. The statistic is recomputed as a ratio, `Σ(y − b) / n` over the resampled attempts. |
| Resamples | 10,000 |
| Interval type | Percentile |
| Seed | `V3_BOOTSTRAP_SEED = 23001`, one fixed child stream per cell via `SeedSequence(23001).spawn` in the fixed cell order of §9.1. The seed is never redrawn. |

**Levels.**
- **Cell verdicts (§9.1):** per-cell 95% intervals, not family-wise corrected. With about 50 verdict cells, roughly 5% of cells whose true value sits on a boundary may be misclassified by chance. The verdict tables say so.
- **Primary claims (§9.2):** Bonferroni over the 3 claims, giving 98.33% intervals.

**No p-values.** There are no p-values and no significance stars, following
`docs/statistics.md`. Verdicts come from intervals against stated thresholds.
These are live calls against real services, but the injected faults come
from the instrument's own random process. The intervals describe uncertainty
over this set of episodes, not over a population of deployments.

**Small cells.**
- An unstratified cell with fewer than n_min = 30 included attempts gets **insufficient data**.
- A `~` stratum with fewer than 15 attempts gets **insufficient data**. Both 30 and 15 are configuration choices.
- Estimates and intervals are reported either way. When a cell has fewer than 2 episodes, no interval is computed.

## 9. Decision rules

### 9.1 Cell verdicts (for Δ_M, per cell or stratum)

The equivalence margin is **m = 0.15**, a configuration choice.

| Prediction | Supported | Contradicted | Otherwise |
|---|---|---|---|
| `+` (positive gain) | interval lower bound > 0 | interval upper bound ≤ 0 | inconclusive |
| `0` (no gain) | interval ⊂ [−m, +m] | interval lies entirely outside [−m, +m] | inconclusive |

**Fixed cell order** (for seeding and reporting): modes T, S, O2, O5, O20, D;
within each mode retry, switch_tool, verify, re_ground, replan, clarify;
`~` strata in the order fresh, not fresh.

Δ_ITT is reported for every cell next to Δ_M. It gets no verdict.

### 9.2 Primary claims

**P1. The confound exists (§1 of v3).**
- *Claim:* under T, the single-call strategies gain nothing over the resend baseline.
- *Test:* the pooled Δ_M over switch_tool, verify, re_ground, replan and clarify in T, with a 98.33% interval.
- *Supported:* interval ⊂ [−m, +m]. *Contradicted:* interval entirely outside [−m, +m]. *Otherwise:* inconclusive.
- *Note:* pooling over arms is allowed here, because the rule against pooling applies to fault origins and fields, not arms. The per-arm values are always shown.

**P2. Persistence separates the strategies.**
- *Claim:* in the persistent modes S, O2, O5, O20 and D, every `+` cell (switch_tool; replan in S; retry in O2; the fresh strata) is supported, and every `0` cell is supported.
- *Result:* the share of supported, contradicted and inconclusive cells is reported. P2 is **supported** if no cell in those modes is contradicted and at least 75% are supported. It is **contradicted** if at least 25% are contradicted. Otherwise it is **inconclusive**. These thresholds are configuration choices.
- *Note:* this claim aggregates cell verdicts, so it uses the 95% cell intervals, not the 98.33% level.

**P3. The best strategy depends on the mode.**
- *Claim:* retry beats switch_tool under transient faults, and loses under persistent ones.
- *Contrast:* the randomized ITT contrast `D_mode = rate(retry) − rate(switch_tool)` (raw success, or `corrected` in D), with a 98.33% interval for each mode. The two arms are drawn uniformly over the same failure stream, so this is an unbiased between-arm contrast. The resend baseline cancels in expectation, so raw rates are used to avoid adding its variance.
- *All attempts count (intention to treat).* The rates in `D_mode` include **all** included attempts (§3.3), among them attempts where the strategy could not act (`untranslatable_args`, no usable call, `no_action_taken`). Those count as failures. The question P3 answers is "which strategy should an agent use?", and a strategy that often cannot act must be charged for it. Δ_M is not used for P3.
- *Supported:* `D_T > 0`, `D_S < 0` and `D_O20 < 0`, each by its interval.
- *Contradicted:* the interval of `D_S` or of `D_O20` lies entirely above 0, or that of `D_T` lies entirely below 0.
- *Otherwise:* inconclusive. `D_O2`, `D_O5` and `D_D` are reported with no verdict.

### 9.3 What each outcome means

| Outcome | Consequence |
|---|---|
| P1 supported | The §1 confound is measured, not only argued. v2's transient-mode success rates on injected faults are reported as uninformative about strategy differences. |
| P1 contradicted or inconclusive | Reported as such. Before P2 and P3 are interpreted, the T-mode manipulation checks (§5) and issue rates are examined, and the explanation goes in the results. No re-analysis is substituted for the pre-registered one. |
| P2 supported | The v3 mode-specific success matrix can be offered to Task 23 / 24 as a measured, persistence-dependent P[t, k] for injected faults, stating that the persistence modes are designed, not observed. |
| P2 contradicted | Reported as a negative result: the designed persistence did not produce the predicted separation. Task 23 / 24 cannot use the v3 matrix as evidence for mode-dependent recovery. Which cells failed is reported. |
| P3 supported | Evidence that the best recovery depends on fault persistence, which motivates diagnosis-aware recovery (C1). |
| P3 contradicted | Reported as a negative result against that motivation, in the same place and with the same prominence as a positive result would have had. |
| Any halt (§2) | The affected sessions are quarantined. No verdicts are computed until the instrument is fixed under a new version and the stopping rule is met on clean sessions. |

## 10. Commitment to report negative and null results

- **Every cell, every claim.** Every pre-registered cell (§9.1), every primary claim (§9.2), every manipulation check (§5) and every count of exclusions (§3.3) is reported, whatever its verdict: supported, contradicted, inconclusive or insufficient data. No cell is dropped, merged or re-defined after data is seen.
- **Same prominence.** Contradicted and inconclusive results are reported in the same tables as supported ones, not moved to an appendix.
- **No relabelling.** Analyses not listed here are labelled **exploratory** wherever they appear, and are never described as confirming a hypothesis.
- **Stopping rule.** The stopping rule (§2) is followed even if early sessions look favourable or unfavourable. Collection is not extended beyond S_max to rescue a cell.
- **Deviations.** Any departure from this document is listed in §11 and in the results, with its reason and the date.
- **If v3 is abandoned.** If v3 collection is abandoned, the reason and all data collected up to that point are reported.

## 11. Deviations log

| ID | Date | Deviation | Reason |
|---|---|---|---|
| D-1 | 2026-10-21 | v3 §3.5 predicts retry (v2) ≈ 1 − r under `transient`. This document predicts ≈ 1 − r³ (gain ≈ +0.19, §4.2). | Arithmetic correction found while writing this document, before any v3 code or data. The frozen file is not edited (rule 3). |

## 12. Instrument acceptance tests (before the first collection session)

The v3 code must pass the following before session 1. The full suite must be
green, and `reproduce.py --mode validation` must pass (rule 8).

| # | Test |
|---|---|
| A1 | The v1 `FlakyWrapper` is byte-identical in behaviour: the existing v1/v2 tests pass unchanged, and a recorded v2 run replays to identical injection decisions. |
| A2 | Per-mode persistence, with the mode semantics of v3 §3:<br>• **transient:** independent per call.<br>• **sticky_args:** same normalised key → same corruption for the episode; a new key gets a fresh draw; no timeouts.<br>• **tool_outage:** exactly N failing calls to T, call N + 1 is served normally, other tools' calls don't count, and peers are independent.<br>• **degraded_output:** sticky per key, with one transformation per key. |
| A3 | The resend probe never creates or advances injection state in any mode. With the same seeds, the strategy's results are identical with and without the probe. |
| A4 | The genuine value never appears in anything passed to the agent or to an LLM. A test scans prompts and histories for it. |
| A5 | `corrected` and `task_completed` follow the v3 §3.6 and §7 per-tool rules, including the 3 °C tolerance and the reference rule. |
| A6 | Plausible-wrong values: temperature offsets are in [8, 15] °C with either sign; library replacements never equal the true value; and the libraries match their frozen sha256. |
| A7 | `llm_error` / `executor_error` attempts are excluded from rates by the analysis code and counted. |
| A8 | Seeds: every stream is derived from `session_seed` via `SeedSequence`, never `hash()`, and recorded. |
| A9 | The halt H1 check fires on a synthetic resend success on an `injected` failure under S, O* or D. It does **not** fire on a `natural` failure, which is tagged `h1_exempt_natural` instead. |
| A10 | The stopping-rule script cannot import or read outcome fields. It neither imports nor reads the outcome-bearing modules or columns (`success`, `corrected`, `detected`, `task_completed`, `resend_baseline`, `evidence`). It reads only the identity and inclusion columns it needs to count attempts per cell: `session_seed`, `block_index`, `injection_mode`, `outage_N`, `strategy`, `fault_origin` and `instrument_fault`. `instrument_fault` is a top-level boolean written with each attempt, true when `evidence.reason ∈ {llm_error, executor_error}`, so the exclusion rule can be applied without reading `evidence`. Checked both statically (imports and column names in its source) and at run time (it is given a store whose outcome columns raise on access). |
| A11 | The false-alarm calibration (§6) samples only non-degraded calls from D blocks, uses `FA_SAMPLE_SEED`, judges each item once, and records the model id, prompt template hash, temperature and settings. |

**Registration.** The v3 collection command and the v3 analysis script will
be registered in `docs/reproducibility.md` and in `reproduce.py`, with the
seeds above (rule 6). This happens when they are written, before session 1.
The analysis runs offline over the committed v3 store.
