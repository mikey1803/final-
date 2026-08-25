# Live-pilot measurement instrument — FROZEN v1

Status: **frozen before any collection begins** (Phase 1 or Phase 2). This document
*is* the measurement instrument, in the same sense `docs/theory/kappa_preregistration.md`
is for the crossover-kappa protocol elsewhere in this project. Changing any definition
below mid-run invalidates every cell collected before the change. If a definition turns
out to be wrong, the fix is a new frozen version (`v2`), not a silent edit to `v1`, and
existing `v1` data is not merged with `v2` data in any single cell.

Verified live 2026-08-20 (all six, direct HTTP checks, not assumed):

| Category | Primary | Peer | Note |
|---|---|---|---|
| Encyclopedic / entity facts | Wikipedia REST API | countriesnow.space | REST Countries (originally proposed peer) is **deprecated** — v3.1 now returns an explicit deprecation error, v5 requires account + API key. Substituted with countriesnow.space (verified live, no key). |
| Weather | Open-Meteo | wttr.in | Both verified live, no key. |
| Mock/placeholder data | JSONPlaceholder | dummyjson.com | reqres.in (originally proposed peer) now returns 401, requires registration + `x-api-key`. Substituted with dummyjson.com (verified live, no key, publishes `X-RateLimit-*` headers — 100/window observed). |

Two of the three originally-proposed peers had silently moved to key-gated access since
being named. Re-verify this table before Phase 2 collection if there is any gap in time
between Phase 1 and Phase 2 — API churn on free services is evidently not rare.

## Fault injection (background wrapper, not a 7th tool)

A single wrapper sits in front of all six tool calls, applied uniformly for the entire
collection run, at one fixed rate: **injection_rate = 0.20** (proposed default, adjust
before Phase 1 starts if you want a different rate — nothing about this number is derived
from data, it is a config choice balancing "enough tool_timeout/tool_error exposure to be
useful" against "collection isn't dominated by injected noise").

Mechanism, chosen specifically so `raw_error` is always genuine, never synthesized:

- **tool_timeout injection**: the real network call is issued normally; the wrapper sets
  an artificially short client-side timeout (e.g. 200ms) so the call is aborted before the
  real server responds. The resulting timeout exception is a real client-side timeout, not
  fabricated text.
- **tool_error injection**: the wrapper deliberately malforms the outgoing request (e.g. a
  nonexistent Wikipedia title, an out-of-range coordinate to Open-Meteo, a nonexistent
  JSONPlaceholder resource ID) so the **real server** returns a genuine error response.
  `raw_error`/`raw_response_body` capture that server's actual reply, verbatim.

**Critical property (your correction #1)**: the wrapper is applied to **both tools in a
peer pair, at the same rate**. `switch_tool` therefore faces the same background injection
risk on the peer as the original call faced — it is peer substitution only, never a way to
bypass the wrapper. Without this symmetry, `switch_tool` would show near-100% success
against `tool_timeout`/`tool_error` by construction, an artifact of the injector rather
than a measurement.

`wrong_tool_selected`, `hallucinated_output`, `context_loss`, `ambiguous_instruction` are
**not** produced by this wrapper — they're elicited through task design (genuinely
ambiguous instructions, tasks needing tools outside the available set, long-context
tasks), per your original scoping note.

## Tool-alternatives map (`switch_tool`)

```
wikipedia         <-> countriesnow.space
open_meteo        <-> wttr.in
jsonplaceholder   <-> dummyjson.com
```

Bidirectional within each pair. No cross-category substitution (e.g. `switch_tool` never
routes a weather failure to Wikipedia) — that would conflate "switch to a peer" with
"attempt an unrelated degraded fallback," a different, unrequested measurement.

## The seven strategy definitions (six real + one control)

Applies **identically to all 7 failure types**. No per-type restriction on any strategy.

### NULL_CONTROL (baseline arm, not one of the 6)

Wait, then re-issue the identical failing call unchanged.

**Wait duration — your correction #2: fixed, not adaptive.** Before Phase 2 collection
begins, run a **timing pre-study**: ~30 events, all 6 real strategies applied, measure
mean wall-clock across them, freeze **one single constant** from that pre-study result.
That constant is then used for every `null_control` trial for the rest of the run,
early and late alike, so the control's meaning doesn't drift across the collection
period. This pre-study is the first activity of Phase 2, not something decided here —
there is no data yet to derive it from, and inventing a number now would violate the
"no fabricated numbers" rule as much as any other cell would.

Success: the re-issued call now succeeds by the original tool's own criterion.
Exists to answer: would this have resolved on its own, with no strategy applied at all?
Distinguishes from `RETRY` only by the deliberate, frozen wait.

### RETRY

Re-issue the identical call (same `tool_name`, same `tool_args`) exactly once,
immediately, zero backoff.

Success: the re-issued call succeeds by the original tool's own criterion.
Does **not**: change args, change tool, re-inject context, or wait — zero backoff is
deliberate, so this never merges with `NULL_CONTROL`.

### REPLAN

Ask the model for a different approach to the same subgoal, with no tool prescribed —
the model may pick the same tool with different args, or a different tool, at its own
discretion.

Success: the resulting action completes the original subgoal.
Does **not**: fix the tool substitution (`SWITCH_TOOL`'s job) or merely restate context
without proposing a new approach (`RE_GROUND`'s job).

**Logging requirement (your addition)**: record `replan_chosen_tool` on every trial —
which tool the model's replanned action actually used. `REPLAN` can collapse into
`SWITCH_TOOL` behaviorally when the model happens to pick the peer tool; this field makes
that overlap measurable after the fact instead of hidden inside the cell.

### SWITCH_TOOL

Deterministic substitution per the tool-alternatives map above. Same subgoal framing,
minimal necessary argument translation. Not a model choice — the map decides, and the
peer is subject to the same background injection rate as the original tool (see above).

Success: the substitute tool's call succeeds by **its own** criterion, applied to the
original subgoal.

### RE_GROUND

Re-inject the original goal and relevant prior context, then re-issue the **same** call
unchanged. No new plan requested.

Success: the re-issued call succeeds by the original tool's own criterion.
Tests: was the failure caused by the agent losing track of context — distinct from
`RETRY` (no context re-injection) and `REPLAN` (no new approach requested).

### VERIFY

Before/instead of proceeding, ask the model to explicitly judge whether the last tool
result (present, absent, or erroneous) actually satisfies the goal. If it judges no, one
corrective re-issue is permitted.

Success: the goal is satisfied after this process completes.
Note: for failures with no result to check (e.g. `tool_timeout`), this reduces to "model
correctly notices nothing usable happened, then retries" — expected to resemble `RETRY`
for those types. That resemblance is itself a real finding, not a design flaw, and is not
to be papered over in reporting.

### CLARIFY

Model generates one clarifying question about the goal, answered by a **pre-authored
canned response**, fixed at task-design time, **before any failures are observed** —
frozen together with this document, not written reactively during collection.

Success: the resulting action completes the original subgoal.

**Rule (your addition)**: if the model's clarifying question is not addressed by the
canned answer, record the trial outcome as `question_unmatched` and **stop** — no retry,
no second question. This trial does **not** count toward the cell's success/failure
tally (it is not a fair trial of `CLARIFY`'s efficacy, it's an instrument-fit miss) but
**is** logged and reported separately, so the rate of `question_unmatched` itself is
visible rather than silently discarded. Allowing a second question would make `CLARIFY`
silently multi-round, contradicting its definition as a single clarifying exchange.

## Cell reporting requirement (your correction #3)

Every `(type, strategy)` cell reports **two counts**, not one:

- `n_replays` — total trial count (existing floor: **>= 30**)
- `n_distinct_events` — count of distinct underlying `event_id`s contributing those
  replays (new floor: **>= 15**)

Both floors must be met for a cell to be reported as usable; either one short means
`underpowered`, reported as such, never silently rounded up. Rows sharing `event_id` are
not independent draws (multiple strategies, or multiple trials of one strategy, replayed
against the same captured failure state) — 30 replays drawn from 5 events is materially
weaker evidence than 30 replays drawn from 30 events, and only the distinct-event count
makes that visible.

At the frozen n=30 floor, the 95% Wilson interval is ~0.32 wide against this project's
typical success-rate range (0.3-0.7). The crossover predictor's own design gap is 0.20.
**These cells are descriptive — do they resemble the synthetic matrix's shape — not
precise enough to feed the crossover predictor.** n~100 would be needed for that. No
code comment, log line, or summary output in this pipeline may imply otherwise.

## Cross-cutting requirements (all 6 strategies + control)

- Strategy order is randomized independently per event before any trials run against it.
- Every replay row carries its own timestamp (`replayed_at`) and a 0-based `trial_index`
  recording actual execution order — separate from any nominal list position, so the
  randomization is auditable after the fact.
- Rows sharing `event_id` are structurally non-independent. Any analysis must cluster by
  `event_id` (environment-level bootstrap, the same pattern already used for the kappa
  pre-registration protocol elsewhere in this project) rather than treat replay rows as
  independent Bernoulli draws.

## Phase scope (restated for this document's own record)

**Phase 1** (current): collection + labelling CLI only. No replays, no strategies applied,
no fault-injection wrapper exercised beyond what's needed to naturally produce
`tool_timeout`/`tool_error` events for labelling. Target 200-300 labelled events. Output:
the measured confusion matrix from the existing TF-IDF+LinearSVC classifier against real
error text and human labels. This needs none of the strategy machinery above.

**Phase 2** (separate, after Phase 1 lands): timing pre-study -> freeze `NULL_CONTROL`
wait constant -> replay collection on the 3 most frequently observed types x all 6
strategies + control = 21 cells, at the `n_replays>=30` AND `n_distinct_events>=15` floor.
Full 7x6 grid explicitly dropped as unreachable at this floor within stated time/budget.

## 2026-08-20 addendum: collection-integrity infrastructure

Process/infrastructure additions made while building out Phase 1 collection, not changes
to the 6 frozen strategy definitions or the injection design above:

- **Prediction-after-labelling is now structurally enforced**, not just a convention:
  `FailureEvent.validate_label_prediction_order` (called by every `EventStore.write`)
  refuses to persist an event with `predicted_type` set while `true_type` is still `None`.
  `recovery.live.predict.predict_labelled_events` only ever iterates already-labelled
  events. A classifier pass run against this store cannot get ahead of labelling by
  construction.
- **Relabelling/agreement mode**: `python -m recovery.live.labelling_cli <store> --relabel
  [--n 50] [--seed ...]` samples already-labelled events and re-presents them blind (no
  prior label shown), storing the second pass in `relabel_type`/`relabelled_by` --
  `true_type` is never touched. `recovery.live.agreement.compute_relabel_kappa` reports
  Cohen's kappa + raw agreement over the resulting pairs.
- **Confidence + second-choice**: the normal labelling flow now also asks confidence
  (`confident`/`unsure`, default confident) and an optional second-choice type after the
  primary label. `EventStore.labelling_progress()["unsure_fraction"]` reports the fraction
  of labelled events marked unsure -- an upper bound on achievable classifier accuracy.
- **Tool health snapshot**: `recovery.live.tools.health_check_all_tools()` makes one real,
  uninjected call to all 6 tools at collection-session start; the result is attached
  unchanged to every `FailureEvent.tool_health` captured that session, so a peer-tool
  outage at collection time is distinguishable after the fact from a genuine `switch_tool`
  result -- it is not re-derivable at analysis time since these are third-party services
  that flake and recover independently of when the data is read.
- **Task manifest + freeze check**: `recovery.live.task_manifest` adds a `Task.
  intended_failure_type` field, `save_manifest`/`load_manifest` (JSON), and
  `require_frozen_manifest` -- collection via `run_live_collection.py --manifest <path>`
  refuses to start unless that manifest file is git-tracked with no uncommitted changes,
  the same "freeze before collection" requirement already applied to this instrument
  document. `intended_vs_observed_distribution` compares the manifest's declared intent
  against labelled outcomes per task after a run. `FailureEvent.task_id` was added to
  make this traceable back to the manifest task that produced each event.

## 2026-08-20 addendum #2: pilot batch, two campaigns, and the yield-rate design

Also built while designing the pilot: `Task.targeted` (bool, structurally validated in
`__post_init__` -- `targeted=True` requires an agent-side `intended_failure_type`;
`tool_timeout`/`tool_error` can never be targeted, since they only occur via
`FlakyWrapper` injection on ordinary tasks, never by task design).

- **28-task pilot manifest** (`experiments/build_pilot_manifest.py`): 4 tasks per agent-side
  type (targeted=True) + 8 natural tasks (targeted=False), task TEXT authored by hand, never
  generated. `recovery.live.pilot_report.compute_yield_report` reports, per type,
  `n_observed / n_attempts` and the attempts needed to reach n=30-45, deliberately mixing
  targeted and natural tasks in one manifest since its purpose is measuring yield, not
  producing either final dataset -- this is the one place the two-campaign separation below
  does not apply.
- **Two campaigns, post-pilot, never pooled**: Campaign A (targeted, balanced across the 5
  agent-side types -- confusion-matrix input only) and Campaign B (untargeted/natural, run at
  scale under injection -- deployment-distribution p input only). Three independent layers
  enforce the separation: (1) `recovery.live.task_manifest.assert_campaign` refuses to start
  a `run_live_collection.py --manifest ... --campaign {A,B}` run if the manifest mixes
  targeted and untargeted tasks; (2) `recovery.live.deployment_distribution
  .estimate_deployment_distribution` only ever reads `targeted=False` events, raising rather
  than falling back to a fabricated uniform p if given none; (3)
  `recovery.live.confusion_matrix_source.collect_confusion_matrix_events` is the mirror
  image, only ever reading `targeted=True` events. `EventStore.labelling_progress()`'s
  `by_type` counts mix both campaigns and are explicitly documented as unsafe to feed into
  either.
- **Open question, not resolved**: `tool_timeout`/`tool_error` can never appear in Campaign A
  by construction (never task-targetable), so a confusion matrix sourced from Campaign A
  alone has zero possible samples for those two rows. Whether Campaign B's naturally-occurring
  `tool_timeout`/`tool_error` events are a legitimate confusion-matrix input for just those two
  rows, or those two types need their own resolution, is flagged in
  `recovery/live/confusion_matrix_source.py` and left for a decision before Campaign A/B
  collection begins at scale -- not decided here.
