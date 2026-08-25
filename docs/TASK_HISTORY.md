# Task History — What Was Built, Task by Task, and Why It Mattered

This document walks through all 16 implementation tasks behind this project, in order, explaining
**what each one built, the specific problem or gap it closed, and how it made the project's results more
correct, more rigorous, or more reproducible.** It is the "how we got here" companion to
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) (what the project currently is) and
[`PAPER_PREP.md`](PAPER_PREP.md) (the exact experimental numbers). Nothing here is final paper prose — it's
a plain-language build log.

Two numbering systems appear side by side in this codebase and are **not the same thing**:

- **Tasks (1–16)** — the order the project was actually built in: infrastructure, algorithms, audits,
  studies, and tooling, in the sequence described below.
- **Phases (1–9)** — the experimental *results* reported in `PAPER_PREP.md` §3. Several tasks (11, 12)
  map directly onto a phase; most of the early tasks (1–10) are the infrastructure and algorithms that
  the phases' experiment scripts are built on top of, not phases themselves.

---

## Summary

| Task | What it built | Why it mattered |
| --- | --- | --- |
| 1 | Closed-form analytic reference layer | A ground truth to check the simulator against, instead of trusting simulated numbers blindly |
| 2 | Shared deterministic simulation kernel | One correct, reusable episode loop instead of one-off scripts each with their own bug surface |
| 3 | Runtime invariant guards | Impossible results now crash loudly instead of silently entering a table |
| 4 / 4A | Table anomaly investigation + independent regeneration | Found a discrepancy honestly reported as unresolved, rather than hidden or hand-waved |
| 5 | Statistics / effect-size metrics layer | Replaced a statistically misleading use of p-values with bootstrap CIs + effect sizes |
| 6 | The horizon sweep | Tested whether "static beats adaptive" was really just a side-effect of a too-short budget |
| 7 | Prior-Guided Thompson Sampling (PGTS) | A new algorithm combining a static table's head start with online learning |
| 8 / 8A | Drift taxonomy (5 physically-motivated drift models) | Non-stationary results no longer rest on one worst-case drift mechanism |
| 9 | Prior-sensitivity study (quality × strength grid) | Quantified exactly when a prior helps vs. actively hurts (negative transfer) |
| 10 / 10B | Analytic crossover predictor + calibration/figures | A calibrated formula for *when* adaptive learning pays for itself, not just an empirical curve |
| 11 | Blind-Design Elicitation Study (human instrument + real LLM arm) | Turned the "one blind rule" assumption into a real, tested distribution |
| 12 | SUCCESS_MATRIX sensitivity & robustness analysis | Quantified how much the headline result depends on the one hand-authored matrix |
| 13 | Statistical rigor audit | Found and fixed a real non-determinism bug; weakened an overclaimed finding to what the data actually supports |
| 14 | Runtime, cost, and provenance instrumentation | Real hardware/latency numbers, real (backfilled) API cost accounting, exact environment capture |
| 15 | Prompt-robustness supplementary study | Showed a striking "low-diversity" LLM finding was partly a prompt-repetition artifact, not a model trait |
| 16 | One-command reproducibility pipeline | Every offline figure/table regenerable from raw data by a stranger, with zero fabrication risk |

---

## Task 1 — Closed-form analytic reference layer

**Built:** `recovery/theory/analytic.py` + `recovery/envs/canonical.py` — a numpy-only, closed-form
computation of oracle rate, random-baseline rate, and any fixed policy's rate for the canonical 7×6
environment, derived directly from the success matrix with no simulation involved.

**Why:** Every other number in this project comes from a stochastic simulator. Before trusting that
simulator's output, the project needed a way to check it that didn't itself depend on simulation — a
number computed by direct formula, so if the simulator's Monte Carlo average and the closed-form value
disagree beyond sampling noise, that's a real bug, not "which one do we believe."

**Improvement:** Gave every later task a hard, independent ground truth to cross-check against, rather
than a second simulation that could share the same bug as the first.

## Task 2 — Shared deterministic simulation kernel

**Built:** `recovery/core/` — four `typing.Protocol` interfaces (`Context`, `Policy`, `Environment`,
`Classifier`) and one shared episode loop, `run_episodes`, that every experiment in the project runs
through. A dedicated RNG design (`core/rng.py::SeedStreams`) splits one integer seed into four
independent streams (types, classifier noise, policy randomness, reward sampling) up front, so nothing is
re-derived mid-run.

**Why:** Before this, every experiment script would have needed its own episode loop, each one a fresh
opportunity for a subtle bug. Worse, without a structural guarantee, a policy's `Context` could
accidentally end up carrying the *true* failure type instead of just the classifier's *predicted* type —
which would let a "blind" policy silently cheat.

**Improvement:** `Context` has no `true_type` field at all — not "please don't read it," but structurally
impossible to read, because it isn't there. Every experiment after this task, across all 16 tasks, runs
through this one audited loop instead of its own copy.

## Task 3 — Runtime invariant guards

**Built:** `recovery/guards/invariants.py` — assertions that fire immediately if a result is
mathematically impossible (e.g. a policy scoring above the analytic oracle ceiling from Task 1).

**Why:** A silently-wrong number that merely looks plausible is far more dangerous than a script that
crashes — it can sit in a results table for weeks before anyone notices the ranking it implies is
impossible.

**Improvement:** Converts "checked by hand, eventually, if someone remembers" into "checked automatically,
every run, immediately." Complements Task 1's analytic layer: Task 1 gives you a value to compare against,
Task 3 makes sure a violation can't be missed.

## Task 4 / 4A — Table anomaly investigation and independent regeneration

**Built:** A from-scratch regeneration of a previously published table through the entirely new Task 2/3
kernel, run independently of the original pipeline that produced the historical numbers.

**Why:** One published table (Table VII) had a numerical anomaly worth investigating before it went any
further into the paper.

**Improvement:** The honest outcome here is itself the point: the investigation found **no reproducible
bug in the live simulator**, but also **could not fully explain** the historical archived result. That
finding is reported as an open discrepancy in `docs/anomalies.md` and `docs/reconciliation_stationary.md`
— a cell-by-cell comparison of every regenerated value against the original — rather than silently
patched over or hidden. This set the project's standard for the rest of its life: an unresolved anomaly
gets written down, not smoothed away.

## Task 5 — Statistics / effect-size metrics layer

**Built:** `recovery/metrics/` — bootstrap confidence intervals (10,000 resamples), Cohen's d, and Cliff's
delta (with the Romano et al. 2006 magnitude thresholds), deliberately with **no p-values, t-tests, or
Holm correction implemented at all**.

**Why:** `docs/statistics.md` lays out the specific problem: every "trial" in this project is a re-run of
the *same fully-known* deterministic-given-its-seed program (same matrix, same policy logic, same
classifier) — there is no population of real agent deployments being sampled. In that setting, a p-value
computed over simulation seeds is degenerate: run enough seeds and the p-value for *any* nonzero true
difference, however practically meaningless, drifts toward zero. It answers "how many seeds did you run,"
not "does this difference matter."

**Improvement:** Replaced a statistically misleading tool with the right one for this setting — a
magnitude with a confidence range, instead of a binary significant/not-significant verdict that can be
inflated just by running more seeds.

## Task 6 — The horizon sweep

**Built:** `experiments/horizon_sweep.py` / `horizon_analysis.py` — the same methods and environments,
run once and evaluated at 7 orders-of-magnitude-spanning interaction budgets (100 through 100,000
episodes), instead of the single 500-episode budget every earlier result used.

**Why:** The paper's stationary-world claim ("a static registry beats every adaptive method") was measured
at exactly one budget. Thompson Sampling's regret bound is `O(K log T / gap)` — at `T=500` a meaningful
fraction of that budget is still necessarily exploration. Without varying `T`, there was no way to tell
"adaptive selection is worse than a good static rule" apart from "adaptive selection hasn't finished
paying its exploration cost yet." The budget itself was a hidden confound that no earlier phase had even
named as a variable.

**Improvement:** Made the interaction budget an explicit, tested axis instead of an invisible constant —
turning "static beats adaptive" from a claim about one specific `T=500` measurement into a claim that's
been checked across nearly three orders of magnitude of budgets.

## Task 7 — Prior-Guided Thompson Sampling (PGTS)

**Built:** `recovery.policies.hybrid.PriorInitializedThompson` (PGTS) and its discounted variant
`DiscountedPriorThompson` — a Beta-Bernoulli bandit whose posterior is *initialized* from a hand-designed
recovery table's recommendation (with tunable prior strength `s` and prior mean `m`), then updated with
real evidence exactly like stock Thompson Sampling.

**Why:** The paper's own discussion had speculated, but never tested, that a bandit initialized from a
blind hand-designed table could combine the static table's strong stationary performance with an
adaptive method's ability to recover under drift — rather than being forced to choose one or the other.

**Improvement:** Turned a speculative discussion point into an actual, evaluatable algorithm — one that
Tasks 8 and 9 then go on to characterize precisely (when the prior helps, when it actively hurts, and how
it behaves under drift).

## Task 8 / 8A — Drift taxonomy

**Built:** `recovery/envs/` drift models — five physically-motivated non-stationarity mechanisms
(`PermutationDrift`, `MonotoneDecayDrift`, `PartialReorderDrift`, `GradualRotationDrift`, and an
`AnnouncedDrift` decorator), plus `experiments/drift_taxonomy.py` / `drift_analysis.py` to sweep across
them.

**Why:** Every non-stationary result in the project before this task used exactly one drift mechanism —
randomly permuting a matrix row's strategy assignment — which is the worst case for a fixed mapping (it
can simultaneously demote the previous best arm to worst and promote the previous worst to best, in one
event). Real production drift is rarely this adversarial: APIs degrade gradually, rankings reshuffle only
at the top, migrations roll out over a window, deployments get announced ahead of time.

**Improvement:** Non-stationary claims no longer rest on a single, deliberately-worst-case drift
mechanism — they're evaluated across a taxonomy that includes gradual, partial, and announced drift too,
so "adaptive methods recover under drift" is a claim tested against multiple realistic failure modes, not
one adversarial one.

## Task 9 — Prior-sensitivity study

**Built:** The full quality-×-strength analysis behind `experiments/prior_grid.py` /
`prior_analysis.py` — sweeping the hand-designed table's *quality* (how often it recommends the true-best
strategy) against PGTS's *prior strength* `s`, across the environments and drift models from Tasks 7 and
8, producing the negative-transfer surface, optimal-strength-per-quality table, and (under drift) a
recovery-half-life measurement for the discounted variant.

**Why:** Task 7 showed a prior *can* help; it didn't establish exactly when it helps, when the benefit
disappears, or when it becomes actively harmful. A confidently-wrong prior is a real risk in practice —
this task quantifies it directly rather than leaving it as an intuition.

**Improvement:** Produced a concrete, reusable answer for practitioners: how good does a hand-designed
table need to be before trusting it strongly is worth it, and how much does that answer change once the
environment can drift. (This same analysis surfaced again during Task 16 — see that section — when its
production tables were found to be three days stale relative to the underlying grid and were regenerated
and re-verified.)

## Task 10 / 10B — Analytic crossover predictor

**Built:** `recovery/theory/crossover.py` — an analytic estimate of the horizon `T*` at which learning
online (Thompson Sampling) is expected to overtake a fixed recovery table in total accumulated reward,
combining Task 1's exact static-regret formula with Thompson Sampling's asymptotic (Lai-Robbins) regret
bound, calibrated against Task 6's empirical horizon-sweep measurements. Task 10B adds the
presentation/validation layer on top — `crossover_calibration.py`, `crossover_extension.py`,
`crossover_reliability.py`, `crossover_figures.py` — a per-cell calibration summary, a
`prediction_status`/`confidence_level` classification, and figures, all derived from Task 10's data
**without** changing the underlying model, fitted constant, or any predicted value.

**Why:** Task 6 established *empirically* that a crossover exists somewhere in the tested horizon range.
Task 10 asks a sharper question: can that crossover point be *predicted* from a closed-form formula,
calibrated once, rather than only measured by re-running the full sweep every time?

**Improvement:** Turned an empirical observation (the crossover happens somewhere between these tested
horizons) into a calibrated, reusable predictive model — while being explicit, in its own documentation,
about exactly where that prediction is known to be unreliable (it rests on an asymptotic bound with a
free constant calibrated against one environment family, on one taxonomy size).

## Task 11 — Blind-Design Elicitation Study

**Built:** `study/` — a standalone, technically-enforced-blind web instrument (the server cannot ever
emit the ground-truth matrix — enforced by both a response allowlist and a static AST audit, both under
test) plus a parallel LLM-as-designer arm (`experiments/llm_designers.py`) running the *identical*
protocol against three real, independent model families (Groq Llama, Google Gemini, NVIDIA-hosted
Mistral), with real API calls.

**Why:** Every earlier phase treated the canonical blind rule (6/7 correct) as a single, fixed `n=1`
input. The paper's own limitations section had already flagged this: how good or bad does an independent
blind designer's table tend to be, as a *distribution*, not one draw?

**Improvement:** The LLM arm is real and complete (63/63 samples attempted, 61 parsed, 3 model families)
and found a genuine, reproducible pattern: `malformed_output` has the lowest raw per-type accuracy in
every family tested — though Task 13 later found this holds up to formal statistical testing in only 1 of
the 3 families, and the LLM arm also surfaced a real diversity problem in one family (later investigated
further by Task 15). The human arm — the paper's actual target distribution — is honestly reported as
`n=0`, pending institutional ethics approval, not estimated or simulated in its place.

## Task 12 — SUCCESS_MATRIX sensitivity & robustness analysis

**Built:** `experiments/matrix_robustness.py` — three perturbation schemes (Gaussian noise, rank-breaking
permutation, rank-preserving permutation) applied to the canonical success matrix, evaluated across 2,204
perturbed worlds, plus a check against two structured alternative matrices (uniform-difficulty,
single-dominant-strategy).

**Why:** Every phase before this one conditions on exactly one hand-authored success matrix — authored by
hand, not measured or literature-derived. The headline claim ("a blind static rule beats every adaptive
method in a stationary world") could, in principle, be an artifact of that one specific matrix's
structure rather than a general property.

**Improvement:** Found real, quantified breakdown thresholds (Gaussian noise σ=0.15; rank-breaking
permutation fraction=0.25) — and reported an honest negative result rather than a reassuring one: 2 of the
3 structured alternative matrices break the headline ranking outright. The claim is now correctly scoped
as conditional on the canonical matrix's specific heterogeneous per-type structure, not universally true
of any matrix — a materially more honest and more useful statement than the unqualified version.

## Task 13 — Statistical rigor audit

**Built:** No new experiment — a full audit of every statistical claim already made in the project,
recorded in `docs/statistics_registry.md`.

**Why:** With 9 phases and multiple prior tasks each making their own statistical claims by that point,
nothing had checked them all against one consistent standard.

**Improvement, concretely:**
- **Found and fixed a real non-determinism bug**: one experiment's per-episode seed formula used Python's
  built-in `hash()` on algorithm names, which is randomized per-process unless `PYTHONHASHSEED` is fixed —
  meaning "the same" experiment could silently produce different numbers across separate re-runs. Fixed
  with a deterministic `zlib.crc32`-based offset, with a regression test that runs the sweep in two
  independent subprocesses and asserts identical output.
- **Weakened an overclaimed finding to what the data actually supports**: Task 11's `malformed_output`
  weak-point finding was re-tested formally (Holm-Bonferroni corrected, per-type-vs-rest binomial tests)
  and found significant in only 1 of the 3 LLM families after correction — the project's language was
  corrected everywhere to this weaker, properly-tested claim.
- Re-ran every confidence interval that had been computed from fewer than 30 seeds at `n_seeds=30`.
- Settled the project on exactly one family of tests that should carry a multiple-comparison correction
  at all (the real-sample LLM per-type accuracy tests) — everything else is a simulation-seed comparison
  already reported via Task 5's effect size + bootstrap CI approach, correctly, not via a p-value.

## Task 14 — Runtime, cost, and provenance instrumentation

**Built:** `experiments/runtime_benchmark.py` (per-decision wall-clock and peak-memory benchmarking,
cross-validated against the real unmodified kernel), `experiments/llm_cost_accounting.py` (backfills
prompt/completion token counts from already-logged raw API responses — zero new paid calls — with
real, dated, per-vendor pricing), and `experiments/capture_environment.py` (real `pip freeze`, hardware/OS
capture, per-family LLM model-pin table).

**Why:** A paper reporting "this method is fast" or "this cost this much" needs real, measured numbers
behind those claims, not estimates — and reviewers need to know exactly what hardware, software versions,
and model IDs produced every reported number.

**Improvement:** Every runtime and cost figure in the paper is now backed by an actual measurement rather
than an assumption, including honestly-reported gaps: NVIDIA NIM pricing is left as `NaN` (not silently
zeroed) since no primary pricing source could be confirmed, and Google's Gemini API was found to echo back
a literal model *alias* rather than a resolved, dated model string — meaning exact model pinning is
provably not achievable from the API's own data for that one family, documented as a real limitation
instead of glossed over.

## Task 15 — Prompt-robustness supplementary study

**Built:** `experiments/prompt_robustness.py` — two independent sub-experiments, kept structurally
separate from Task 11's primary results (their own table/figure files, explicitly labeled
"SUPPLEMENTARY"): paraphrasing the blind-design instrument two different ways across all 3 LLM families,
and randomizing the per-sample presentation order for the one family (Google Gemini) that had shown
unusually low design diversity.

**Why:** Task 13's audit had already surfaced a striking pattern — Google Gemini produced only 2 distinct
table designs across 21 identically-worded, independent API calls. Was that genuine low-diversity model
behavior, or an artifact of sending the exact same prompt 21 times in a row?

**Improvement:** A clear, well-evidenced answer, not a guess: randomizing the presentation order alone
took distinct mappings from 2 up to 9 out of 21 (Mann-Whitney U=325.5, p=0.0016) — the original "only 2
designs" finding was **at least partly a prompt-repetition artifact**, not simply "this model always
converges on one answer no matter how it's asked." This is exactly the kind of methodological limitation
worth surfacing about LLM-as-designer studies in general, and it's reported here rather than left
unexamined.

## Task 16 — One-command reproducibility pipeline

**Built:** `backend/reproduce.py` — a single entry point (`python reproduce.py --mode validation|paper`)
that regenerates every offline analysis, publication figure, and publication table through 7 stages
(verify environment -> verify raw artifacts -> regenerate analyses -> regenerate figures -> regenerate
tables -> regenerate doc artifacts -> validate every output against the committed production version),
plus `backend/requirements.lock.txt` (an exact pinned environment) and `docs/reproducibility.md` /
`docs/experiments/instrumentation.md` (setup instructions and a full figure/table provenance map, each
entry verified by actually running the command, not inferred from source code).

**Why:** By Task 15, the project had 15 tasks' worth of experiment scripts, each independently runnable
but with no single command that proved the whole set of paper artifacts could be regenerated from
committed raw data by someone who wasn't in the room when it was built — the actual bar for
reproducibility.

**Improvement, including a real bug this task caught:** the pipeline's own Stage 7 validation — comparing
every regenerated output against the committed production file — caught two of Task 9's prior-sensitivity
tables (`prior_optimal_strength.parquet`, `prior_negative_transfer_surface.parquet`) that had gone stale,
three days out of sync with the canonical prior grid (missing an entire environment, wrong quality/horizon
bins), undetected until this pipeline actually executed and compared rather than trusting the files were
current. They were regenerated from the canonical data, confirmed bitwise-identical to what the pipeline
independently computed, and re-verified end to end. The two genuinely expensive full-scale sweeps
(`horizon_sweep.py` ≈73h, `prior_grid.py` ≈66h) are treated as canonical and skipped by default with a
loud, explicit banner, per the project's standing rule to never re-run expensive experiments unless
strictly necessary. The full backend test suite (319 tests) passes with zero regressions after this task.

---

## The cumulative picture

Read top to bottom, the 16 tasks form one coherent arc: **build a trustworthy simulator with an
independent way to check it (1–3) → catch and honestly report a discrepancy rather than hide it (4) → fix
the statistics to match what the data can actually support (5) → remove a hidden confound in the headline
result (6) → build and characterize a genuinely new algorithm (7, 9) → stop testing non-stationarity
against only the worst case (8) → turn an empirical observation into a calibrated predictive model (10) →
replace a single `n=1` assumption with real, tested data, human and LLM (11) → check how much the whole
result depends on one hand-authored input (12) → audit every statistical claim made so far and fix what
was wrong (13) → back every performance/cost number with a real measurement (14) → stress-test whether an
LLM finding was genuine or a prompt artifact (15) → prove, by actually running it, that a stranger could
regenerate every result from scratch (16).** Each task either closes a gap the previous ones left open, or
checks a claim the project was at risk of overstating.
