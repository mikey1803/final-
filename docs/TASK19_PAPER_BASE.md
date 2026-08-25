# Task 19 — Paper Base Document

**Purpose of this file.** This is a reference document, not paper prose. It
exists so that writing the paper doesn't require re-deriving definitions or
re-finding results scattered across a dozen source documents and a long
working session. It explains the whole project briefly for context, then
covers Task 19 (both halves — 19A and 19B) in full: every key term defined,
every method described, every result reported with its actual caveats, and
every source file named so a claim can be traced back to where it was
produced. Nothing here should be copied into a paper verbatim without
checking it against the underlying doc it cites — this file summarizes and
organizes, the underlying docs (`docs/theory/*.md`, `docs/statistics_registry.md`,
`docs/experiments/instrumentation.md`) remain the source of record.

---

## Part 0 — Project Context (for readers unfamiliar with the whole project)

### 0.1 What the project is

The project studies **failure recovery for autonomous AI agents** as a
learning problem, not an exception-handling problem. Agents that use tools
fail in structured, recurring ways — a tool times out, a tool errors, the
agent picks the wrong tool, the agent hallucinates an output, output comes
back malformed, the agent loses context, or an instruction is ambiguous.
Instead of one fixed rule per failure type, the project treats each
`(failure_type, recovery_strategy)` pair as an option whose value must be
learned from experience, using multi-armed-bandit algorithms (UCB1,
Epsilon-Greedy, Thompson Sampling) to select recovery strategies and improve
over time.

Full plain-language explanation of every component (failure injection,
taxonomy, classifier, recovery strategy bank, adaptive selector, persistent
memory, evaluation harness, dashboard): **`docs/PROJECT_EXPLAINED.md`**.
Full build history of Tasks 1–16 (what each task built and why it mattered,
in order): **`docs/TASK_HISTORY.md`**. This file does not repeat that
content — it picks up where Task 16 (the reproducibility pipeline) left
off, at **Task 19**.

### 0.2 The 7-type failure taxonomy (used throughout)

| Failure Type | Meaning |
|---|---|
| `tool_timeout` | A tool call takes too long or never returns. |
| `tool_error` | A tool returns an error, exception, or service failure. |
| `wrong_tool_selected` | The agent chooses an inappropriate tool. |
| `hallucinated_output` | The agent produces unsupported/fabricated information. |
| `malformed_output` | Output has the wrong format (invalid JSON, incomplete structure). |
| `context_loss` | The agent forgets earlier instructions or state. |
| `ambiguous_instruction` | The instruction is unclear, multiple valid interpretations. |

`N = 7` throughout Task 19 (this taxonomy size), `K = 6` strategies
(`retry`, `replan`, `switch_tool`, `re_ground`, `verify`, `clarify`).

### 0.3 The canonical environment

`recovery/envs/canonical.py` defines `MATRIX` (a 7×6 matrix, `P[type,
strategy]` = success probability of that strategy for that failure type)
and `BLIND_RULE` (a fixed, hand-authored mapping from type to strategy — the
project's one static baseline table, correct on 6 of 7 types against
`MATRIX`'s own argmax — "quality 6/7"). Everything in Task 19 that is not
explicitly a swept/sampled matrix uses this canonical pair.

### 0.4 What Task 10 built (prerequisite for everything in Task 19)

Task 10 built **`recovery/theory/crossover.py`**: an analytic (closed-form
where possible, numerically solved where not) predictor of the horizon
`T*` at which Thompson Sampling's cumulative reward first overtakes a
static table's. Task 19 extends this predictor; understanding it is a
prerequisite. Full detail: `docs/theory/crossover.md`. The essentials:

- **`rho(P, pi, p)`** — a static table's constant per-episode regret rate,
  computed EXACTLY (no approximation): `oracle_rate(P,p) - policy_rate(P,pi,p)`.
  A static table pays this penalty every episode forever, so its cumulative
  regret is the straight line `rho * T`.
- **`R_TS(T)`** — Thompson Sampling's cumulative regret, an ASYMPTOTIC
  (`T→∞`) upper bound derived from the Lai-Robbins theorem, carrying an
  unknown multiplicative constant `c` (later renamed `kappa` in Task 19)
  that this project can only fit empirically, never derive from first
  principles. Two functional forms exist: `form="kl"` (uses the Bernoulli
  KL divergence between arms) and `form="gap"` (uses the raw gap `Δ`
  between arms — cheaper, less accurate).
- **`T*`** — the crossover horizon, the smallest `T` solving `R_TS(T) =
  rho * T`. Computed numerically (`predict_crossover`): scan for the first
  sign change of `h(T) = R_TS(T) - rho*T`, refine with `brentq`. `h` is
  provably unimodal, so this scan can never find a spurious second root.
- **Calibration result (Task 10, `experiments/crossover_calibration.py`)**:
  fitting `c` on 56 empirical cells gives `c* ≈ 0.00452` (a ~220× downward
  correction from the literal `c=1.0`), and even then the predictor is
  reliable ONLY in a narrow regime — poor tables (quality ≤ 1/7) at small
  gaps (~0.05). At the project's own canonical table (quality 6/7), the
  calibrated predictor underestimates the true crossover by ~50×. This
  narrow reliability is the single most important fact to carry into Task
  19: **the predictor Task 19 extends was never fully validated even in
  its original, simpler (no classifier noise) form.**

---

## Part 1 — Glossary of Key Terms (Task 19-specific)

Terms are grouped by topic. Each entry gives the plain-language meaning,
the formal definition where one exists, and the file where it's implemented
or defined.

### 1.1 Core theory objects

| Term | Meaning |
|---|---|
| **Confusion matrix (`confusion`)** | `confusion[t, t'] = P(classifier predicts t' \| true type is t)`. A real classifier is never perfect; this matrix describes its error pattern. `confusion = I` (identity) means a perfect classifier. |
| **`C=I` / `C≠I`** | Shorthand used throughout for "classifier is the identity matrix (perfect)" vs. "classifier is not the identity (has real confusion)" — the two conditions Task 19A compares. |
| **`P_eff` (effective mixture matrix)** | The success-probability matrix `P` re-expressed in PREDICTED-type space instead of true-type space, via Bayes' rule: `P_eff[t',k] = (Σ_t p[t]·confusion[t,t']·P[t,k]) / p'[t']`. This is the central mathematical object of Task 19A — it says "given the classifier predicted type `t'`, what is the effective success probability of strategy `k`, averaged over which true type it actually was." |
| **`p'` (predicted-type marginal)** | `p'[t'] = Σ_t p[t]·confusion[t,t']` — how often the classifier predicts each bucket, marginalizing over true types. |
| **`effective_matrix_under_confusion`** | The function that computes `(P_eff, p')` from `(P, confusion, p)`. Exact Bayes computation, no approximation. Reduces exactly to `(P, p)` when `confusion = I`. |
| **`static_rate_confused` / `rho_confused`** | The classifier-noise-aware version of `rho`: a static table's regret rate when it only ever sees the classifier's *predicted* type, not the true one. Computed EXACTLY (composition of two existing exact functions), not asymptotic. |
| **`ts_cumulative_regret_confused`** | The classifier-noise-aware version of `R_TS(T)`: literally `ts_cumulative_regret(P_eff, T, N, p', form, kappa)` — Task 10's unmodified regret-bound formula, just evaluated on the mixture `(P_eff, p')` instead of `(P, p)`. |
| **`crossover_horizon_confused`** | The classifier-noise-aware version of `T*`: solves `ts_cumulative_regret_confused(...) = rho_confused(...) * T`. **Not** a black-box call to `predict_crossover` — it mixes two different index spaces (`rho_confused` in true-type space, the regret side in predicted-type space) that `predict_crossover` cannot handle internally, so it's a separate implementation reusing the same numerical solving strategy. |
| **`kappa`** | The Task-19 name for Task 10's free constant `c`, applied to `crossover_horizon_confused`. Never derivable from theory — must be empirically fit. This is the central unknown the whole kappa-validation effort (Part 3 below) is about pinning down. |

### 1.2 The two noise channels

| Term | Meaning |
|---|---|
| **Ceiling reduction (Channel 1)** | `oracle_rate(P,p) - oracle_rate(P_eff,p')` — how much reward-ceiling is lost purely because merging true types into indistinguishable predicted buckets destroys information, independent of what policy is used. Provably ≥ 0, exactly 0 at `confusion=I`. |
| **Gap shrinkage (Channel 2)** | For each predicted bucket, how much the (best − second-best) value gap changes relative to the true rows feeding into it — a measure of how much HARDER the arm-identification problem becomes under mixing (learnability, not reward ceiling). |
| **Argmax inversion** | A hypothetical failure mode where a predicted bucket's own best action, under `P_eff`, is optimal for NONE of the true types that feed into that bucket — the mixture "invents" a recommendation no true type actually wanted. Checked explicitly; does not occur at the real classifier's measured confusion, though a constructed counter-example proves the check isn't vacuous. |

### 1.3 Kappa-validation-protocol terms (this session's core work)

| Term | Meaning |
|---|---|
| **In-sample vs. held-out (out-of-sample)** | In-sample: kappa is fit and evaluated on the SAME data. Held-out: kappa is fit on one set of environments (TRAIN) and evaluated on a disjoint set (TEST) never used in fitting. Only held-out evaluation is real validation evidence; in-sample numbers are "uninterpretable as validation in principle," regardless of how good they look. |
| **Environment-level split** | Splitting TRAIN/TEST by which environment (which `P` matrix) a cell comes from, not by individual `(env, classifier)` cell — necessary because cells sharing an environment are not statistically independent (same underlying matrix). |
| **Branch A (transfer)** | Fit `kappa_A` on TRAIN environments' `C=I` cells, apply it UNCHANGED to TEST environments' `C≠I` cells. Tests whether a kappa fit under one classifier condition transfers to another. |
| **Branch B (in-condition ceiling)** | Fit `kappa_B` on the SAME TRAIN environments' `C≠I` cells (not `C=I`), apply to the SAME TEST environments' `C≠I` cells. The best achievable same-condition fit — the benchmark Branch A is compared against. |
| **`R = E_A / E_B`** | The decision statistic: how much worse transfer (Branch A) performs than the best achievable same-condition fit (Branch B), on IDENTICAL held-out data. `R=1` means transfer is exactly as good as fitting directly on the target condition; `R>1` means transfer degrades performance. |
| **Case 1** | `R ≤ 1.5`: one shared kappa suffices — the `P_eff` reduction captures the classifier-noise effect, and residual prediction error is inherited from Task 10's own bound, not new error this extension introduces. |
| **Case 2** | `R > 1.5`: the confusion-aware extension introduces its own model error beyond what Task 10's bound already carries; a kappa fit under one condition does not reliably transfer. Ships as a documented non-result if this happens — no rescue attempt. |
| **Pre-registration** | Writing the exact test, split, threshold, decision rule, and — critically — the EXACT CLAIM WORDING for every possible outcome, BEFORE the data that will be evaluated exists. Prevents "shaping the protocol to the answer" after seeing results. Document: `docs/theory/kappa_preregistration.md`. |
| **Power floor** | The pre-committed minimum sample size (5 finite cells) below which no Case 1/Case 2 conclusion is drawn at all — the result is reported as underpowered/inconclusive rather than forced into one of the two cases. |
| **Bootstrap (environment-level)** | Resampling ENVIRONMENTS (not individual cells) with replacement to estimate a confidence interval — respects the non-independence of cells sharing an environment. Two separate bootstraps were run: held-out error uncertainty (resample TEST envs, reuse already-fitted kappas) and fit uncertainty (resample TRAIN envs, refit kappa each time). |
| **Calibrated predictor vs. validated bound** | A pre-committed naming distinction: if the fitted kappa falls within one order of magnitude of Task 10's own `best_fit_c≈0.0045` (i.e. in `[0.00045, 0.045]`), the extension is described as a "calibrated predictor" (an empirical fit with theoretical motivation) rather than a "validated bound" (which would imply mathematical/statistical certainty the evidence does not support). |
| **PARTIALLY-SUPPORTED** | This project's classification vocabulary has three tiers, not two: VALIDATED / PARTIALLY-SUPPORTED / UNVALIDATED-PRELIMINARY. Used specifically when a pre-committed decision rule is technically satisfied by a point estimate, but the same pre-registered uncertainty quantification (bootstrap CI) does not confirm that satisfaction is robust — a middle category that resists rounding a fragile result up to a clean "validated." |

### 1.4 Infrastructure / process terms (relevant to a Methods or Reproducibility section)

| Term | Meaning |
|---|---|
| **RunStore** | `recovery/core/parallel.py`'s content-addressed cache: each simulated cell (`method, env_id, seed, classifier_id, ...`) hashes to a filename; a cell already on disk is never recomputed. This is the entire resume mechanism for long sweeps — a killed/interrupted run loses at most the cells in flight at kill time. |
| **`--estimate`** | A flag on sweep scripts (`horizon_sweep.py`) that calibrates single-threaded throughput on a short sample, then projects total wall-clock. **Known defect** (this session): the projection divides by `n_jobs`, assuming linear parallel speedup — an assumption that does not hold on real hardware and was empirically wrong by ~2× to ~9× across three separate attempts this session. |
| **Effective parallelism** | The empirically MEASURED ratio of (calibrated single-threaded CPU-work) to (actual wall-clock elapsed) for a real, executed batch of work — as opposed to the `--estimate` flag's assumed (but wrong) `n_jobs`-linear figure. Measured twice this session at 1.90× and 2.74× (see Part 4). |
| **Measurement hygiene** | The general finding that any timing number collected while other foreground compute/tool-use is happening on the same machine is contaminated by unrecorded contention — applies retroactively to Task 14's own runtime-benchmark table (whose collection conditions are now stated as UNKNOWN, not assumed clean) and prospectively to any future timed run. |

---

## Part 2 — Task 19A: The Classifier-Noise-Aware Crossover Extension

**Source of record: `docs/theory/crossover_derivation.md`.**

### 2.1 The one-sentence idea

Task 10 assumes a policy always acts on the TRUE failure type. Verified
directly at the implementation level (not assumed from documentation — the
exact call chain was traced through `episode.py` → `adapters.py` →
`selector.py` → `matrix_env.py`), every policy in this repository actually
only ever sees the classifier's PREDICTED type — `Context` has no field a
policy could even read a true type from. Task 19A asks Task 10's same
question (when does adaptive learning overtake a static table?) under this
structurally-verified fact.

### 2.2 The five numbered assumptions (A1–A5)

- **A1** — predicted-type-bucket posterior structure. Verified at the
  implementation level, not an assumption in the usual sense — a structural
  fact about the simulator.
- **A2** — conditionally Bernoulli rewards (inherited unchanged from Task 10).
- **A3** — stationary confusion matrix (no classifier drift within the
  horizon analyzed). Optimistic if the classifier itself degrades over time.
- **A4** — asymptotic regret bound (inherited unchanged from Task 10) —
  valid as `T→∞`, not a finite-time guarantee, carries the uncalibrated
  `kappa`.
- **A5** — scalar accuracy as a convenience (`uniform_confusion`) when a
  full confusion matrix isn't available — a real simplification, validated
  primarily against the REAL trained classifier's actual (non-uniform)
  confusion matrix specifically to avoid over-relying on this convenience.

### 2.3 The derivation, equation by equation

```
p'[t']      = Σ_t  p[t] · confusion[t, t']                                  (1)
P_eff[t',k] = ( Σ_t p[t] · confusion[t, t'] · P[t, k] ) / p'[t']            (2)
rho_confused(P, pi, confusion, p) = oracle_rate(P,p) − policy_rate_under_confusion(P,pi,confusion,p)   (3)
R_TS_confused(P,T,confusion,p,form,kappa) = ts_cumulative_regret(P_eff, T, N, p', form, kappa)          (4)
T* solves: R_TS_confused(...) = rho_confused(...) · T                       (5)
ceiling_reduction(P,confusion,p) = oracle_rate(P,p) − oracle_rate(P_eff,p') (6)
```

- Equations (1)–(2): exact Bayes computation, no approximation.
- Equation (3): exact composition of two existing, unmodified exact functions.
- Equation (4): Task 10's `ts_cumulative_regret` called unmodified, just
  evaluated on the mixture — inherits ALL of Task 10's own limitations
  (asymptotic-only, uncalibrated constant, no per-arm gap factor — see Part
  0.4).
- Equation (5): a NEW implementation (not a wrapped call to
  `predict_crossover`), because it mixes true-type-space `rho_confused`
  with predicted-type-space `R_TS_confused` — something `predict_crossover`
  cannot do internally since it computes its own `rho` from the same triple
  it uses for the regret side.
- Equation (6): exact math, a diagnostic report, not itself a prediction.

**Exact reductions, proved and unit-tested**: at `confusion = I`, `p'=p`,
`P_eff=P` exactly (equations 1–2), and `crossover_horizon_confused` returns
BIT-FOR-BIT what `predict_crossover` returns (equation 5) — the extension
is a strict generalization, not a divergent reimplementation.

### 2.4 What is exact math vs. what is inherited approximation

This distinction matters enormously for how a paper should describe Task
19A's contribution — it is NOT "we built a new asymptotic bound." It is
"we built exact machinery around an existing asymptotic bound, and
carefully separated which parts are exact from which parts inherit the
old bound's known unreliability."

| Component | Nature |
|---|---|
| `effective_matrix_under_confusion` (eq 1–2) | Exact math |
| `static_rate_confused` / `rho_confused` (eq 3) | Exact math |
| `ceiling_reduction` (eq 6) | Exact math |
| `gap_shrinkage_report` | Exact math (a report, not a prediction) |
| `argmax_inversion_check` | Exact math (a check, not a prediction) |
| `ts_cumulative_regret_confused` (eq 4) | Inherited asymptotic approximation |
| `crossover_horizon_confused` (eq 5) | Inherited asymptotic approximation |

### 2.5 The two-channel decomposition — measured result

At the real trained classifier's empirical confusion (mean diagonal
accuracy **0.939**, computed via
`src.evaluation.classifier_context.compute_empirical_confusion_matrix`):

- **Channel 1 (ceiling reduction): 0.0120**
- **Channel 2 (mean gap shrinkage across 7 buckets): 0.0020**

Ceiling reduction is **~6× larger** than gap shrinkage at this classifier's
actual accuracy — most of the real classifier's measured degradation is
reward-ceiling loss (information destroyed by merging types), not
learnability loss (harder arm identification), though both are present and
neither is zero.

### 2.6 Argmax-inversion check — measured result

**Does not occur, in any of the 7 predicted buckets**, at the real
classifier's measured confusion (full per-bucket table in
`crossover_derivation.md` §5). A constructed 3-type counter-example proves
the check is not a dead code path — it correctly fires when its structural
condition (two comparable-weight contributors with genuinely different
argmaxes, no single dominant contributor) is actually met.

### 2.7 T* non-monotonicity in accuracy — a genuine mathematical finding

`T*` is **not monotonic in classifier accuracy in general** — verified with
a clean 2-type/2-strategy construction
(`tests/test_crossover_theory.py::test_crossover_horizon_confused_not_generally_monotonic_in_accuracy`).
A predicted bucket's own best arm can flip identity as accuracy drops
through some threshold, making that bucket's top-2 gap V-shaped in accuracy
(large at both extremes, exactly zero at the flip point) rather than
monotonically shrinking. `rho`/`ceiling_reduction` ARE provably monotonic
in accuracy; `T*` is not. This is a property of equation (5)'s shape,
established independently of any calibration — it is true regardless of
what `kappa` turns out to be.

---

## Part 3 — Task 19A Empirical Validation: Full Methodology and Results

This is the part of Task 19A that changed the most during this session, and
is almost certainly the part a paper's Results section needs to describe
most carefully — because the headline result is genuinely nuanced (a
pre-committed threshold satisfied by a point estimate, but not confirmed
robust by the same protocol's own uncertainty quantification).

### 3.1 Tier 2 (secondary, static-side only) — the clean validation

**`static_rate_confused` (equation 3) validates tightly.** Table VII
(`results/tables/table_vii.parquet`, 6 noise levels × 50 already-simulated
seeds each) has no adaptive-method trajectory, so it validates the STATIC
side only — never used to claim the adaptive theory is validated.

| noise_level (β) | predicted rate | empirical rate | abs. error |
|---:|---:|---:|---:|
| 0.0 | 0.4408 | 0.4306 | 0.0102 |
| 0.1 | 0.4691 | 0.4582 | 0.0109 |
| 0.3 | 0.5256 | 0.5112 | 0.0144 |
| 0.5 | 0.5821 | 0.5744 | 0.0077 |
| 0.8 | 0.6669 | 0.6584 | 0.0085 |
| 1.0 | 0.7234 | 0.7156 | 0.0078 |

**Median absolute error: 0.0093** (0.93 percentage points) — the strongest
validation result anywhere in Task 19, consistent with this component being
exact closed-form math rather than an asymptotic approximation.

### 3.2 Tier 1 (primary, adaptive side) — history of this validation, in order

**Stage 1 (original, now superseded): 6 cells, `kappa=1.0`, uncalibrated.**
Only 3 environments (`canonical`, `gap0.20_idx00`, `gap0.20_idx01`) had ever
been simulated at the time. At `kappa=1.0` these numbers are NOT
interpretable as validation at all — `kappa` is a free constant, so error
at an unfitted value says nothing about whether the prediction's shape is
right.

**Stage 2 (correction, still superseded): the same 3 environments, kappa
fit IN-SAMPLE.** Fit kappa on the 3 `C=I` cells, freeze, apply to the 3
`C≠I` cells: median error rose from 52.5% (own-fit) to 138.8% (frozen
transfer). Fit kappa independently on `C≠I`: 46.1% own-fit error. This
whole stage is now understood to be **uninterpretable as validation in
principle** — kappa was fit and evaluated on the identical 3-cell subset
throughout; there was no held-out data at all. Retained in git history for
transparency, not reported as a result any longer.

**Stage 3 (current, the actual result): pre-registered, environment-level
held-out protocol, executed against the complete 26-environment /
52-cell dataset.** This is the part described in full detail below.

### 3.3 Why a genuinely held-out test was needed

The core problem with Stage 2: with kappa fit and evaluated on the SAME
3 cells in every row, a low error proves nothing about generalization —
it's definitionally impossible to distinguish "the model is right" from
"the model was tuned to these exact 3 points." Fixing this required (a)
enough environments to make a real train/test split possible, and (b) a
protocol written and committed BEFORE any of that new data existed, so the
test itself couldn't be shaped to produce a favorable answer after the
fact.

### 3.4 Closing the data gap — provenance and cost

**Finding (forensic investigation, read-only, no simulation):** the
committed `horizon_sweep.parquet` had only ever contained 3 of the intended
26 environments. Exhaustive inspection of all 617 files in the
`results/runs/horizon_sweep/` RunStore (every file loaded, not sampled)
confirmed the other 23 environments had ZERO trace anywhere — not a lost
RunStore state, not a filtering bug, a genuine never-simulated gap. Traced
to a stale project-status note (`PROJECT_STATUS.md`) that had propagated
into a false "26 environments" belief; the true, documented, deliberate
scope was a cost-gated reduction after the full grid was measured at ~73
hours.

**Closing the gap required a real, new simulation run.** Filtered to the 2
methods Tier 1 actually needs (`static_blind`, `thompson`; the other 9 of
11 default methods were out of scope) — 312 cells total (26 environments ×
2 classifiers × 2 methods × 3 seeds), 204 new cells needed. Full incident
detail (three attempts, a disproven BLAS-oversubscription hypothesis, two
independent estimator misses, a corrected stop-rule decision, and the
final measured-parallelism-based reprojection) is in
`docs/experiments/instrumentation.md` and summarized in Part 4 below.

**Result: `results/tables/horizon_sweep_full26.parquet`, 26 environments ×
2 classifiers, 52 cells, 2,184 rows across 7 horizons.** This is the
dataset the held-out protocol below runs against.

### 3.5 The pre-registered protocol, exactly as specified

**Document: `docs/theory/kappa_preregistration.md`, written and committed
BEFORE the 26-environment data existed.**

- **Split (`KAPPA_SPLIT_SEED=19001`)**: 25 gap-matrix environments split
  17-train/8-test via a fixed seed computation
  (`rng.choice(25, size=8, replace=False)`); `canonical` placed in TRAIN
  unconditionally (structural reason: it's the only non-gap-family
  environment, holding it out would test a different, separately-flagged
  question). **18 TRAIN environments, 8 TEST environments.**
  - TEST: `gap0.20_idx00, idx01, idx06, idx08, idx11, idx16, idx17, idx24`
  - Disclosed in advance: `idx00`/`idx01` (the two environments whose bad
    `kappa=1.0` errors were already known from the old Stage-1 analysis)
    coincidentally landed in TEST — named explicitly, seed not redrawn
    because of it (redrawing on observing this would itself be the
    seed-shopping the pre-registration exists to prevent).
- **Fitting method**: identical to Task 10's own `fit_best_c` — bounded 1-D
  least-squares minimizing mean squared LOG-space error, bounds
  `(log(1e-3), log(1e4))`, `form="kl"` fixed (not varied).
- **Branch A (transfer)**: fit `kappa_A` on TRAIN environments' `C=I`
  cells, apply UNCHANGED to TEST environments' `C≠I` cells. `E_A` = median
  absolute relative error on those held-out predictions.
- **Branch B (in-condition ceiling)**: fit `kappa_B` on the SAME TRAIN
  environments' `C≠I` cells, apply to the SAME TEST environments' `C≠I`
  cells. `E_B` analogously.
- **Decision statistic**: `R = E_A / E_B`.
- **Decision rule (fixed before any data existed)**: `R ≤ 1.5` → Case 1;
  `R > 1.5` → Case 2. `1.5` justified as a round, moderate middle point
  between a too-lenient factor-3 bar and a too-strict factor-1.0 bar —
  explicitly disclosed as a judgment call, not derived.
- **Power floor**: fewer than 5 of 8 TEST cells finite (or fewer than 5 of
  18 TRAIN cells feeding either branch) → no Case 1/Case 2 conclusion is
  drawn at all, reported as underpowered.
- **Bootstrap**: `BOOTSTRAP_SEED=19002`, `n_boot=2000`, environment-level
  resampling — (1) held-out error uncertainty (resample TEST envs, reuse
  already-fitted kappas), (2) fit uncertainty (resample TRAIN envs, refit
  kappa per resample, separately for each branch).
- **Pre-committed claim wording** for all three outcomes (Case 1, Case 2,
  underpowered) — written in advance so the outcome could not shape the
  write-up after the fact. See `kappa_preregistration.md` §7 for exact text.
- **Calibrated-predictor naming rule**: if the operative fitted kappa falls
  in `[0.00045, 0.045]` (within one order of magnitude of Task 10's own
  `best_fit_c≈0.0045`), describe the result as a "calibrated predictor,"
  not a "validated bound" — decided before seeing the fitted value.
- **Hard commitment**: this is the ONLY kappa specification that will be
  fit and reported. No second split, form, threshold, or fitting bound
  tried after seeing results, regardless of outcome.

### 3.6 The result

Script: `backend/scratch/task19_kappa/fit_kappa_preregistered.py`. Outputs:
`results/phase10/crossover_theory/kappa_preregistered_*.parquet`.

**Power check: passed.** 18/18 TRAIN cells and 8/8 TEST cells finite, both
branches — well above the floor of 5.

| | Branch A (transfer) | Branch B (in-condition) |
|---|---:|---:|
| Fitted kappa | `kappa_A = 0.044381` | `kappa_B = 0.042559` |
| n (TRAIN cells used) | 18 | 18 |
| Median abs. rel. error on held-out TEST | **59.7%** | **40.5%** |

**`R = E_A / E_B = 0.5970 / 0.4053 = 1.4729`.**

**Decision: Case 1** (`R ≤ 1.5`) — satisfied by **1.8%** margin.

**Uncertainty (pre-registered bootstrap, not optional context — required
reporting):** 95% CI for `R` = **[1.00, 9.55]**. This reaches nearly 6.4×
past the decision threshold — deep into Case-2 territory. Fit-uncertainty
bootstrap: `kappa_A` 95% CI `[0.0369, 0.0544]`, `kappa_B` 95% CI `[0.0105,
0.0426]` (notably wider and lower — the `C≠I` fit is less stable than the
transferred one).

**Per-cell detail, Branch A on held-out TEST:**

| env_id | ratio | signed rel. error |
|---|---:|---:|
| gap0.20_idx00 | 1.487 | +48.7% |
| gap0.20_idx01 | 1.527 | +52.7% |
| gap0.20_idx06 | 1.283 | +28.3% |
| gap0.20_idx08 | 5.756 | +475.6% |
| gap0.20_idx11 | 0.675 | −32.5% |
| gap0.20_idx16 | 0.333 | −66.7% |
| gap0.20_idx17 | 5.752 | +475.2% |
| gap0.20_idx24 | 1.694 | +69.4% |

Highly heterogeneous — two cells (`idx08`, `idx17`) are large
over-predictions (~5.75×) under both kappas; `idx16` is a consistent ~3×
under-prediction. No single scalar kappa corrects this spread; no attempt
was made to (pre-registration's no-rescue commitment).

**Calibrated-predictor naming: applies.** `kappa_A = 0.0444` falls inside
`[0.00045, 0.045]` — at the top edge of the range.

**Pre-committed Case-1 claim, used verbatim:** *"`P_eff` captures the
classifier-noise effect; the residual prediction error is INHERITED Task 10
asymptotic-bound bias, not error introduced by this extension."* This is a
claim RELATIVE to Task 10's existing bound — the absolute error (59.7% /
40.5%) remains large; Task 10's own bound is not newly accurate because of
this result. The test quantifies a pre-existing limitation transferring
cleanly; it does not resolve that limitation.

### 3.7 Classification: PARTIALLY-SUPPORTED

**Not VALIDATED. Not UNVALIDATED/PRELIMINARY.** Case 1 is the pre-committed
reported outcome — the point estimate satisfies the rule, and this is real
held-out evidence for the first time, a genuine improvement over the
in-sample Stage 2 result. But the fragility is real and is reported
alongside the pass, not hidden: a **1.8% margin** and a bootstrap CI
spanning **[1.00, 9.55]** mean a different draw of held-out environments
could plausibly have produced Case 2. Both facts belong in the same
sentence in a paper — "satisfies the pre-registered criterion, but not
robustly" is the accurate one-line summary, not "validated."

**For a paper**, the honest framing is: *this is the first genuinely
held-out evidence for the classifier-noise extension, it clears the
pre-committed bar, and the same protocol's own uncertainty quantification
shows that bar-clearing is not decisively robust — reported as a
calibrated predictor with disclosed fragility, not a validated bound.*

---

## Part 4 — The Sweep: Process, Instrumentation, and Reproducibility Findings

This part is aimed at a paper's **Methods** or **Threats to Validity /
Reproducibility** section — these are genuine, dated findings about how
compute-cost estimation and long-running-experiment monitoring behaved in
practice on real hardware, independent of the crossover theory itself.
Full detail: `docs/experiments/instrumentation.md`.

### 4.1 The estimator defect

`horizon_sweep.py --estimate` calibrates single-threaded throughput on a
short sample, then **divides by `n_jobs`, assuming linear parallel
speedup** — stated explicitly in the function's own docstring. This
assumption does not hold on the machine used for this project (4 physical /
8 logical cores) and was wrong, in the same direction (optimistic), across
**three separate attempts**:

| Attempt | Estimate | What actually happened |
|---|---:|---|
| 1 (11-method, 1,518-cell scope) | much smaller than observed | Killed after 9.64h at 387/1,518 cells |
| 2 (2-method, filtered scope) | 38.4 min | Early-rate check: ~86.9 min projected (~2.3× miss) |
| 3 (resumed, same scope) | (n/a, same 38.4 min estimate) | Actual: **~81.9 min total** (~2.13× miss) |

### 4.2 A hypothesis proposed, tested, and DISPROVEN — reported as a positive example of self-correction

**Initial hypothesis**: no `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS` limiting anywhere in the codebase was letting each `loky`
worker process oversubscribe the 4 physical cores via its own internal BLAS
thread pool. **Directly tested** (`backend/scratch/task19_kappa/verify_blas_threads.py`,
dispatching through the exact `joblib.Parallel(backend="loky")` call
`run_grid` uses, reading `threadpoolctl.threadpool_info()` from inside a
worker process) and found **FALSE**: `loky` already auto-computes and sets
these env vars to 1 per worker with zero manual configuration, confirmed at
both `n_jobs=4` and `n_jobs=8`. The hypothesis was retracted immediately
upon disproof, with the disproof reported alongside it rather than left
standing. **The underlying estimator defect (the `n_jobs`-linear-scaling
assumption) was explicitly NOT retracted along with the mechanism** — a
4-physical-core machine under hyperthreading is sufficient explanation on
its own, no BLAS claim required.

### 4.3 Measured effective parallelism — a real, machine-specific number, with real variance

Derived as (calibrated CPU-seconds of work) ÷ (measured wall-clock),
directly from RunStore file-write timestamps or the script's own internal
timer — a MEASUREMENT, categorically different from a `--estimate`
projection.

| Measurement | Value | Basis |
|---|---:|---|
| Attempt 2 | **1.90×** | 60 cells, 2,843 CPU-sec / 1,492s wall-clock (RunStore timestamps) |
| Attempt 3 | **2.74×** | 144 cells, 9,212 CPU-sec / 3,367.45s wall-clock (script's own `run_grid` timer) |

**These two measurements disagree by ~44%**, on the same machine, same
workload shape, ~45 minutes apart. No controlled A/B was run to attribute
the gap; it is left as an open finding rather than smoothed into one
number, consistent with the measurement-hygiene point below (concurrent
foreground diagnostic work during Attempt 2's window is the leading
candidate explanation, not confirmed).

**Implication for the paper's cost figures**: `reproduce.py`'s ~73h
(`horizon_sweep.py`, full unfiltered scope) and ~66h (`prior_grid.py`)
headline costs rest on the identical, now-empirically-falsified linear-`n_jobs`
assumption. Both remain un-re-derived and suspect in BOTH directions
(the estimator over-assumes achievable speedup, making them optimistic;
nothing rules out contention effects at that much larger scale making the
true figure worse still) — flagged as follow-up work, not corrected here.

### 4.4 Measurement hygiene — a general finding, not sweep-specific

Concurrent foreground compute/tool-use on the same machine during a timed
run is an unrecorded confound that applies to **every timing number this
project has ever reported**, including Task 14's own per-decision
runtime-benchmark table. **Whether Task 14's benchmark was collected under
clean conditions is now stated as UNKNOWN, not assumed clean** — no
process-isolation protocol was documented at the time, and nothing in
`runtime_benchmark.py` records contemporaneous machine load. A 4-point
protocol for future timed runs is recorded in `instrumentation.md`: no
concurrent compute during the timed window; no parallel agent/tool calls
during the window; record machine load state alongside timing output; take
multiple `--estimate` calibration draws (a 3.4× spread was observed between
two single draws under otherwise-identical conditions).

### 4.5 Non-randomized cell ordering biases early-progress projections

`build_grid()` orders cells `env → classifier → method → seed`, not
randomized. With a large, known cost asymmetry between classifier
conditions (184,000 eps/sec `perfect` vs. 850 eps/sec `real`, ~218×), every
environment's cheap cells complete before its expensive cells even begin —
so any cumulative-average projection taken early in a run is
**systematically optimistic**, only converging toward the true rate once
the cells-seen-so-far mix approaches the true overall mix. Two fixes
recorded (neither implemented): shuffle cell order before dispatch, or
always project per-population (cost-dominant dimension) rather than
pooled.

### 4.6 A stop-rule correction — a genuine self-correction worth naming for a Methods section

An operational stop-threshold (kill the run if a reprojection exceeds ~90
minutes) was applied literally to a reprojection based on a genuine
MEASUREMENT (1.90× effective parallelism from 60 real completions), and the
run was killed. This was subsequently identified as a **scope error**: the
threshold existed to catch a third bad *estimate*, in a context where much
larger estimates (4.43h, 13.4h) had already been accepted — 96 minutes was
never actually over budget, only over a number that was never scoped to
mean "budget," and a measurement should have been treated differently from
an estimate on principle, not merely evaluated against the same numeric
threshold. The run was relaunched immediately (resuming cleanly from
RunStore-banked cells, confirming the content-addressed cache's resume
guarantee held under a real interruption) and allowed to run to completion
without a new stop threshold. Recorded here because it is a genuine example
of a process correction mid-experiment, transparently reported rather than
silently absorbed.

### 4.7 Final measured wall-clock, full reconciliation

| | Value |
|---|---:|
| Original `--estimate` (204-cell filtered scope) | 38.4 min |
| Actual total (Attempt 2 partial + Attempt 3 completion) | **≈81.9 min** |
| Ratio | **≈2.13×** |
| Consistent with Attempt 2's own early-rate miss? | Yes (~2.3×, same direction) |

---

## Part 5 — Task 19B: The Drift-Crossover Negative Result

**Source of record: `docs/theory/drift_crossover_negative.md`.** Scope note
from `crossover_derivation.md`: 19A and 19B are entirely separate
extensions of Task 10 (classifier noise vs. environment drift) — nothing in
one affects the other's validity. 19B's own assumption numbering (A6–A9)
is kept deliberately separate from 19A's (A1–A5).

### 5.1 The question

Given Task 8's drift taxonomy (`PermutationDrift`, `MonotoneDecayDrift`,
`PartialReorderDrift`, `GradualRotationDrift`) and its dense pre-simulated
grid (12,800 cells), could a low-dimensional summary statistic predict the
SIGN of `adaptive_advantage` (whether Thompson Sampling beats a static
table) under drift, out of sample?

**Falsifiable structure, fixed before any attempt**: a candidate statistic
must predict the sign better than a majority-class guess, leave-one-cell-out.
Three candidates were pre-registered, tested in order; a fourth was
explicitly ruled out in advance if the third failed.

### 5.2 Three attempts, all insufficient as quantitative sign rules

| Attempt | Statistic | R² / accuracy | Verdict |
|---|---|---|---|
| 1 | `(s, g)` — survival, gap-change | R²=0.012 | Insufficient; `g` near-tautological in this taxonomy |
| 2 | `(d, τ)` — magnitude, recovery-ratio | R²=0.579–0.735 | **Retracted** — conceptually circular (τ reuses the theory under test) |
| 3 | `(D, R)` — environment-level regret, reversion fraction | R²=0.676; sign accuracy 79.7% = majority baseline | Mechanistically correct qualitative story; no quantitative sign RULE survives leave-one-cell-out |

**Attempt 2's circularity, explained**: `τ`'s denominator IS
`predict_crossover` — Task 10's own predictor. Testing whether τ predicts
`adaptive_advantage` was substantially testing whether Task 10's existing
bound has SOME validity applied per-row to drift — not independent
confirmation of a new descriptor. Statistical circularity (data leakage)
was absent; conceptual circularity (reusing the theory under test as an
input) was present and material. **Caught and retracted before being
reported as a finding** — named explicitly as a credibility asset, not
buried.

### 5.3 The magnitude-aggregation artifact

`d`/`s`/`g` were computed as means over TOUCHED rows only, blind to how
many of an environment's rows were touched — `magnitude`'s apparent
contribution in Attempt 2 (+0.156 R²) was substantially an artifact of this
exclusion. Fixing the aggregation level (`D`, environment-level
prior-weighted regret, verified numerically identical to the EXISTING
`static_regret_rate` function to 2.78e-16 as a construction-identity check,
never used predictively) collapsed the apparent gain from +0.156 to +0.006.
**General lesson for future work**: a statistic computed at the wrong level
of aggregation will masquerade as needing an extra explanatory variable;
the fix is recomputing at the right level, not adding a patch variable.

### 5.4 CONFIRMED POSITIVE FINDING — reversion, not magnitude, governs sign

Reported as a standalone headline result, separated from the failed
quantitative rules: **the sign of `adaptive_advantage` under drift is
governed by whether staleness damage REVERTS, not by its magnitude.**

- All **13/13** negative-advantage cells sit at `R ≥ 0.68` (reversion
  fraction), spanning 260 independent seed draws.
- All **32/32** `R≈0` cells are positive, spanning 640 independent seed
  draws.
- `monotone_decay`'s `R ≈ 0`, 95% CI `[0.0000, 0.0160]`, n=236 — precise,
  with the small-sample caveat stated as part of the finding, not a
  footnote.

**This qualitative pattern is real and robust even though no quantitative
threshold built on top of it survives out-of-sample testing** — a
correlational, mechanism-consistent pattern can hold while a specific
numerical boundary does not.

### 5.5 This explains a previously-unexplained inversion

Static beats adaptive in 10/16 `partial_reorder` cells (from Task 8's own
analysis). The reversion structure explains this: best↔second swaps
REVERT (period-2 mechanism — event 2 returns to event 0's state, not event
1's), so a static table's stale action retains near-optimal value most of
the time, while an adaptive learner pays real exploration cost chasing a
target that keeps flipping back to where it already was. **Explicitly
stated as a correlational-plus-mechanism explanation, not a causal claim**
— `partial_reorder`'s mechanism was designed before this analysis and
never independently varied; a genuine causal test would need drift
processes matched on `D` (damage magnitude) with independently varied `R`
(reversion), not run here.

### 5.6 Honest scope statement

`drift_type` fixed effects retain **+0.092 R²** beyond `(D,R)` — real,
unaccounted structure. Named candidates (not operationalized, discipline of
stopping at three attempts): disruption abruptness, within-row breadth,
available recovery time relative to shift frequency. **Explicit
conclusion**: the sign of `adaptive_advantage` under drift is **not
reducible to a two-statistic summary** across these four mechanistically
distinct drift regimes, given the data and methods available.

---

## Part 6 — Final Classification Summary (paper-ready table)

| Component | Status | One-line justification |
|---|---|---|
| `effective_matrix_under_confusion` (eq 1–2) | **VALIDATED** | Exact math, proved, unit-tested |
| `static_rate_confused` (eq 3) | **VALIDATED** | Exact math; empirically validated, median abs. error 0.93pp (Tier 2, n=6×50 seeds) |
| `ceiling_reduction` (eq 6) | **VALIDATED** | Exact math, proved, unit-tested |
| `gap_shrinkage_report` | **VALIDATED** | Exact math (a report, not a prediction), unit-tested |
| `argmax_inversion_check` | **VALIDATED** | Exact math (a check), unit-tested; constructed counter-example proves it's not vacuous |
| `T*` non-monotonicity in accuracy | **VALIDATED** (as a shape finding) | Proved with a clean construction, unit-tested — not a claim that predicted values are accurate |
| `ts_cumulative_regret_confused` (eq 4) | **PARTIALLY-SUPPORTED** | Inherits eq 5's status (only reachable empirically through it) |
| `crossover_horizon_confused` (eq 5) / `kappa` | **PARTIALLY-SUPPORTED** | Pre-registered held-out test: Case 1 by point estimate (R=1.4729≤1.5), bootstrap CI [1.00,9.55] does not confirm robustness; described as a calibrated predictor (kappa_A≈0.044, within Task-10-adjacent range), not a validated bound |
| Drift-crossover unified law (19B) | **NEGATIVE RESULT (documented)** | Three pre-registered attempts insufficient as quantitative sign rules; one confirmed qualitative finding (reversion governs sign) survives, no numerical rule does |

**Scope limitation applying to all of Task 19A** (verbatim, from
`crossover_derivation.md`): *"The empirical Tier 1 validation, as currently
available, is confined to one matrix family at a single gap level
(gap=0.20). Generalization across other gap structures and independently
generated environment families is therefore untested."*

---

## Part 7 — Source File Index (for citations)

| Topic | File |
|---|---|
| Whole-project explainer | `docs/PROJECT_EXPLAINED.md` |
| Task-by-task build history (1–16) | `docs/TASK_HISTORY.md` |
| Task 10 base crossover predictor | `docs/theory/crossover.md`, `recovery/theory/crossover.py` |
| Task 19A derivation + all results | `docs/theory/crossover_derivation.md` |
| Kappa pre-registration protocol | `docs/theory/kappa_preregistration.md` |
| Task 19B drift negative result | `docs/theory/drift_crossover_negative.md` |
| Statistical claims registry (all tasks) | `docs/statistics_registry.md` (§6 = 19B, §7 = 19A) |
| Sweep/estimator/measurement-hygiene incident record | `docs/experiments/instrumentation.md` |
| Paper outline + draft prose (19A section) | `docs/PAPER_PREP.md` |
| Task 20 scoping (not implemented) | `docs/experiments/agent_trace_validation.md` |
| Kappa-fitting scripts | `backend/scratch/task19_kappa/fit_kappa.py` (superseded, in-sample), `fit_kappa_preregistered.py` (current, held-out) |
| Final validated dataset | `results/tables/horizon_sweep_full26.parquet` |
