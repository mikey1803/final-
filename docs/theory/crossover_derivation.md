# The Classifier-Noise-Aware Crossover Predictor (Task 19A)

This document derives and validates the classifier-noise-aware extension to
Task 10's analytic crossover predictor
([`crossover.md`](crossover.md)/`recovery.theory.crossover`). It is written
for a reviewer, not for the author. Every assumption is numbered; every
number in the validation section was produced by actually executing
`experiments/crossover_theory_validation.py`, not inferred from the code.

**Scope note.** This document covers ONLY the stationary, classifier-noise
extension (19A). The non-stationary (drift) extension was attempted and
found not to admit a low-dimensional summary statistic — see
[`drift_crossover_negative.md`](drift_crossover_negative.md) for that
negative result, including its own assumption numbering (A6–A9), kept
entirely separate from the numbering below to avoid implying the two
extensions share a derivation.

**Headline status (read this before anything else in this document).**
This extension bundles components with very different validation status,
and they must not be conflated:

- **VALIDATED / exact:** `effective_matrix_under_confusion` (§3.1),
  `static_rate_confused` (§3.2, empirically validated to median abs. error
  0.0093 — Tier 2, §7 "Tier 2"), the two-channel decomposition (§4), the
  `P_eff` argmax-inversion check (§5), the exact reduction to Task 10 at
  `C=I` (§3.1/§3.4), and the `T*` non-monotonicity-in-accuracy finding
  (§6) — all exact math, unit-tested, and (where an empirical claim is
  made at all) validated against real data.
- **PARTIALLY-SUPPORTED:** `crossover_horizon_confused` (§3.4, §7.1) — the
  pre-registered, environment-level held-out kappa protocol
  (`docs/theory/kappa_preregistration.md`) landed **Case 1** (`R = 1.4729
  ≤ 1.5`): a kappa fit on C=I training environments transfers to held-out
  C≠I environments without materially worse degradation than fitting C≠I
  directly. This is real held-out evidence, not the in-sample `n=3` number
  this document previously reported — but the result sits **1.8% below
  the decision threshold**, and its own pre-registered bootstrap puts
  `R`'s 95% CI at **[1.00, 9.55]**, which reaches deep into Case-2
  territory. The point estimate satisfies the pre-committed rule; the
  bootstrap does not confirm that satisfaction is robust to which
  environments happen to be sampled. Absolute error at the fitted kappa
  also remains large (median 59.7% / 40.5% on the two branches) — Task
  10's own asymptotic bound is not newly accurate in absolute terms
  because of this result. Full accounting in §7.1; this component is
  **not** classified VALIDATED by this document.

---

## 1. What this extension adds, in one sentence

Task 10 predicts the horizon at which Thompson Sampling's cumulative regret
first drops below a static table's, assuming the policy always observes the
*true* failure type. This extension asks the same question when the policy
only ever observes a classifier's *predicted* type — which, per the
verified implementation trace below, is what every policy in this
repository actually does.

## 2. Numbered assumptions

- **A1 — predicted-type-bucket posterior structure.** Verified at the
  implementation level (not inferred from documentation), tracing the exact
  call chain `recovery/core/episode.py` → `recovery/policies/adapters.py`
  → `src/recovery/selector.py::ThompsonSamplingSelector` →
  `recovery/envs/matrix_env.py`: `Context` carries only `predicted_type`
  (no field a policy could read a true type from —
  `recovery/core/protocols.py`); `LegacySelectorAdapter.select`/`.update`
  both key the wrapped selector on `ctx.predicted_type`
  (`adapters.py:141,149`); `ThompsonSamplingSelector`'s Beta posterior
  table is indexed by whatever `failure_type` argument it's called with —
  the predicted type (`selector.py:339,356`); `MatrixEnvironment.sample_reward`
  reads `matrix[true_type, action]`, where `true_type` is `episode.py`'s
  own step-1 local variable, never routed through `Context`
  (`matrix_env.py:87`). Consequently each predicted bucket's Beta posterior
  accumulates Bernoulli draws whose success probability comes from
  whichever true type the classifier mapped into that bucket this episode
  — exactly the mixture `effective_matrix_under_confusion` computes below.
  **Neither conservative nor optimistic** — this is a structural fact about
  the simulator, not an approximation.
- **A2 — conditionally Bernoulli rewards.** Inherited from Task 10
  unchanged: given the true type, reward is Bernoulli with the matrix's
  stated probability. Neither conservative nor optimistic.
- **A3 — stationary confusion matrix.** The confusion matrix does not
  change over the horizon being analyzed (no classifier drift). Optimistic
  if the classifier itself degrades over time (e.g. concept drift in the
  error-text distribution) — out of scope here, not modeled anywhere in
  this repository's kernel.
- **A4 — asymptotic regret bound.** Inherited from Task 10 unchanged: the
  Lai–Robbins-style bound `ts_cumulative_regret` implements is valid as
  `T → ∞`, not a finite-time guarantee, and carries an uncalibrated free
  constant (`kappa`). Optimistic at small `T` (see §6, "Limits").
- **A5 — scalar accuracy as a convenience.** When a full confusion matrix
  isn't available, `confusion = recovery.theory.analytic.uniform_confusion(N,
  accuracy)` (symmetric off-diagonal noise) stands in for it. This is a real
  simplification relative to any actual classifier's confusion structure
  (validated in §5 primarily against the REAL trained classifier's
  empirical confusion matrix, not this convenience case, precisely to avoid
  over-relying on it).

## 3. Derivation

### 3.1 The effective mixture matrix

Bayes-flipping the confusion matrix `confusion[t, t'] = P(predict t' | true
t)` against the type prior `p` gives the predicted-bucket marginal and the
mixture matrix:

```
p'[t']      = sum_t  p[t] * confusion[t, t']                                (1)
P_eff[t',k] = ( sum_t p[t] * confusion[t, t'] * P[t, k] ) / p'[t']          (2)
```

Implemented exactly as written in `effective_matrix_under_confusion` — no
approximation in this step; it is a closed-form Bayes computation. Raises
if any `p'[t'] = 0` (a bucket the classifier never reaches has an
undefined, 0/0 mixture there — not silently defaulted).

**Exact reduction**, proved and tested (`tests/test_crossover_theory.py`):
at `confusion = I`, `p' = p` and `P_eff = P` exactly, because only the
diagonal term survives every sum.

### 3.2 Static side (exact, no approximation)

```
rho_confused(P, pi, confusion, p) = oracle_rate(P, p) - policy_rate_under_confusion(P, pi, confusion, p)   (3)
```

`static_rate_confused` — a one-line composition of two existing,
**unmodified** `recovery.theory.analytic` functions. `pi` here is indexed
by predicted type (`pi[t']` = action played when the classifier predicts
`t'`), matching `policy_rate_under_confusion`'s own convention. Exact, not
asymptotic — this is the same kind of closed-form quantity `static_regret_rate`
already was.

### 3.3 Adaptive side (exact composition of an inherited asymptotic bound)

```
R_TS_confused(P, T, confusion, p, form, kappa) = ts_cumulative_regret(P_eff, T, N, p', form, kappa)   (4)
```

`ts_cumulative_regret` is called **unmodified** — every assumption and edge
case documented on it (A4 above; small-`T` clipping; tied-arm handling)
applies here unchanged, just evaluated on `(P_eff, p')` instead of `(P,
p)`. This function performs no new arithmetic beyond computing the mixture.

### 3.4 The crossover equation, and why it is NOT a black-box call to `predict_crossover`

```
T* solves:  R_TS_confused(P, T, confusion, p, form, kappa) = rho_confused(P, pi, confusion, p) * T   (5)
```

Equation (5) mixes two different index spaces: `rho_confused` (equation 3)
is computed in TRUE-type space; `R_TS_confused` (equation 4) is computed in
PREDICTED-type space. `predict_crossover` cannot be called as a black box
here because it computes its own `rho` internally from the same `(P, pi,
p)` triple it uses for the regret side — the wrong `rho` for this equation.
`crossover_horizon_confused` therefore reuses `predict_crossover`'s exact
numerical **strategy** (geomspace scan for the first sign change of `h(T) =
R_TS(T) - rho*T`, `brentq` refinement, never raises on a bracketing
failure) as a new implementation, not a wrapped call.

**Exact reduction**, proved and tested: at `confusion = I`, both sides of
(5) coincide bit-for-bit with the unconfused equation, so
`crossover_horizon_confused(P, pi, I, p, form, kappa)` returns exactly what
`predict_crossover(P, pi, p, form, c=kappa)` returns.

## 4. The two noise channels (Amendment B, part 1)

Classifier noise degrades performance through two distinct, separately
measurable mechanisms that should not be collapsed into one number:

**Channel 1 — ceiling reduction** (policy-independent):

```
ceiling_reduction(P, confusion, p) = oracle_rate(P, p) - oracle_rate(P_eff, p')   (6)
```

Even a hypothetical policy that perfectly knows `P_eff` and always plays
each bucket's own best arm cannot reach the true oracle rate — merging true
types into indistinguishable buckets destroys reward information before
any policy, static or adaptive, gets to act on it. Does not depend on
`pi`. Provably `>= 0` (a bucket's rate under a convex-combination mixture
can never exceed the best of its contributing true rows' own rates — see
`ceiling_reduction`'s docstring) and exactly 0 at `confusion = I`.

**Channel 2 — gap shrinkage** (learnability): for every predicted bucket
`t'`, compares `P_eff[t']`'s own (best − second-best) gap against the top-2
gaps of the true rows contributing to it. This channel is about how much
*harder* Thompson Sampling's own arm-identification problem becomes under
mixing — independent of the reward ceiling. `gap_shrinkage_report` returns
one row per bucket with both the mixture's own gap and each contributor's
gap, so a shrinking-vs-widening judgment can be made per bucket, not just
in aggregate.

**Measured at the real trained classifier's empirical confusion** (mean
diagonal accuracy 0.939, computed via
`src.evaluation.classifier_context.compute_empirical_confusion_matrix`):

- Channel 1 (ceiling reduction): **0.0120**
- Channel 2 (mean gap shrinkage across the 7 predicted buckets): **0.0020**

At this classifier's actual (fairly high) accuracy, both channels are
small, and the ceiling-reduction channel is roughly 6× larger than the
mean gap-shrinkage channel — most of this classifier's real degradation is
reward-ceiling loss, not learnability loss, though both are present and
neither dominates entirely.

## 5. Argmax-inversion proposition (Amendment B, part 2)

**Question:** can a predicted bucket's own best arm (by `P_eff`) differ
from *every* contributing true row's own best arm — a genuine inversion
where the mixture recommends an option that was optimal for none of the
true types feeding into it?

**At the real trained classifier's measured confusion (mean diagonal
0.939): no inversion occurs, in any of the 7 predicted buckets.**
Full detail (bucket, `argmax(P_eff)`, contributing true rows, each
contributor's own argmax):

| bucket | argmax(P_eff) | contributors | contributor argmaxes | inversion? |
|---|---|---|---|---|
| 0 | 0 | [0, 1, 4] | [0, 2, 4] | No |
| 1 | 2 | [1, 4] | [2, 4] | No |
| 2 | 2 | [1, 2] | [2, 2] | No |
| 3 | 4 | [3, 4] | [4, 4] | No |
| 4 | 4 | [4] | [4] | No |
| 5 | 3 | [5] | [3] | No |
| 6 | 5 | [6] | [5] | No |

**The condition under which it would occur.** Inversion requires at least
two contributors with comparable weight and genuinely different individual
argmaxes (no single dominant contributor, i.e. no near-diagonal confusion
mass for that bucket), whose *non-optimal* arms happen to agree on a third
option decent for both — violating a kind of single-peakedness across
contributors. **Constructed counter-example, proving the detector fires
when the condition is genuinely met** (not a dead code path — verified in
`tests/test_crossover_theory.py::test_inversion_detected_in_constructed_example`):
3 types, 3 strategies,

```
P = [[0.65, 0.10, 0.60],    # true type 0, argmax = 0
     [0.10, 0.65, 0.60],    # true type 1, argmax = 1
     [0.50, 0.50, 0.50]]    # filler row
confusion: types 0 and 1 both map ~50/50 into predicted bucket 2
           (type 2 contributes negligible mass there)
```

`P_eff[2, :] ≈ 0.5·[0.65,0.10,0.60] + 0.5·[0.10,0.65,0.60] = [0.375, 0.375,
0.60]` — bucket 2's own best arm is **2**, which is optimal for *neither*
contributing true type (each individually prefers arm 0 or arm 1, at
0.65). This is not a hypothetical impossibility; it simply did not occur at
this project's real classifier's actual, fairly diagonal-dominant confusion
levels.

## 6. Limits of this analysis

- **A4's asymptotic bound is loose at `kappa=1.0`, and the fitted-kappa
  correction does not fully resolve this — see §7.1 for the full numbers.**
  Executed as a pre-registered, environment-level held-out test (18 train /
  8 test environments, `docs/theory/kappa_preregistration.md`): fitting
  `kappa_A` on the 18 TRAIN environments' `C=I` cells and applying it
  UNCHANGED to the 8 held-out TEST environments' `C≠I` cells gives median
  abs. rel. error `E_A=59.7%`; fitting `kappa_B` directly on the TRAIN
  environments' `C≠I` cells and applying it to the SAME held-out cells
  gives `E_B=40.5%`. `R = E_A/E_B = 1.4729`, **1.8% under** the
  pre-committed `R ≤ 1.5` Case-1 threshold — but the pre-registered
  bootstrap CI for `R` is `[1.00, 9.55]`, reaching deep into Case-2
  territory, so this Case-1 result is fragile, not robust, and is reported
  as such rather than as a clean pass. `crossover_horizon_confused` is
  accordingly classified **PARTIALLY-SUPPORTED**, not VALIDATED and not
  UNVALIDATED/PRELIMINARY, in §8's summary table — real held-out evidence
  exists for the first time, it satisfies the pre-committed rule by point
  estimate, and it is described as a **calibrated predictor** (`kappa_A
  ≈ 0.0444`, within the pre-registered near-Task-10 range) rather than a
  validated bound, given the absolute error (~40–60%) and the wide
  bootstrap CI.
- **Amendment C's asymptotic-regime diagnostic**: correlation between
  `observed_T` and absolute relative error across the 6 Tier 1 cells is
  **−0.79** (Pearson) / **−0.89** (on `log(observed_T)`) — smaller observed
  crossover horizons are systematically associated with LARGER prediction
  error. The two `gap0.20_idx00`/`gap0.20_idx01` environments, where
  `BLIND_RULE` happens to have very low incidental quality (0/7 and 1/7
  correct against these randomly-sampled matrices, versus 6/7 on
  canonical) and correspondingly tiny observed crossover horizons (41, 53,
  65, 167 episodes), show the worst errors (up to ~99×). This is reported
  explicitly and is not smoothed over — it is exactly the kind of
  systematic small-`T` breakdown A4 itself warns about, now confirmed
  empirically rather than only asserted theoretically. n=6 is too small to
  treat this correlation as more than suggestive.
- **T\* is not monotonic in classifier accuracy in general** — a genuine
  mathematical finding, verified with a clean 2-type/2-strategy
  construction (`tests/test_crossover_theory.py::test_crossover_horizon_confused_not_generally_monotonic_in_accuracy`):
  a predicted bucket's own best arm can flip identity as accuracy drops
  through some threshold, making that bucket's own top-2 gap V-shaped in
  accuracy (large at both extremes, exactly zero at the flip point) rather
  than monotonically shrinking. `rho` (equation 3) and `ceiling_reduction`
  (equation 6) ARE both provably monotonic in accuracy (tested); `T*`
  itself inherits the gap's non-monotonicity near any such flip and is
  therefore NOT asserted to be monotonic anywhere in this codebase.
- **Tier 1's validation set was 6 cells, not 52, for most of this task's
  history — confirmed, after a dedicated provenance investigation, to be
  an intentional and permanent (at the time) scope reduction, not data
  loss — and was subsequently closed by a dedicated sweep.** The
  originally-committed `results/tables/horizon_sweep.parquet` contained 3
  environments (`canonical`, `gap0.20_idx00`, `gap0.20_idx01`); the other
  23 environments were confirmed never simulated at all (exhaustive
  RunStore check, 617 files, exactly 3 distinct `env_id` values). A
  dedicated, explicitly-approved sweep (`--n-seeds 3 --n-gap-matrices 25`,
  filtered to the 2 methods Tier 1 needs) closed this gap; the new
  `results/tables/horizon_sweep_full26.parquet` contains all 26
  environments × 2 classifiers (52 cells, 2,184 rows across 7 horizons).
  See §7.2 for the full provenance record (both the original gap and the
  run that closed it) and the exact measured cost. This full dataset is
  what §7.1's pre-registered kappa protocol runs against. Every cell's
  presence is verified directly from the output parquet, not assumed.
- **The empirical Tier 1 validation, as currently available, is confined
  to one matrix family at a single gap level (gap=0.20). Generalization
  across other gap structures and independently generated environment
  families is therefore untested.**
- **Scalar-accuracy `confusion` (A5) is a convenience, not a validated
  model of any real classifier.** Validated primarily against the real
  classifier's actual (non-uniform) confusion matrix wherever possible
  (§4, §5, Tier 1's `classifier_id="real"` cells); `uniform_confusion` is
  used only in the unit tests and Tier 1's `classifier_id="perfect"`
  sanity check (`C=I`, not actually a noise case).

## 7. Empirical validation

Produced by `experiments/crossover_theory_validation.py`, reading only
already-committed production parquet files
(`results/tables/horizon_sweep.parquet`, `results/tables/table_vii.parquet`)
plus one deterministic classifier-inference pass
(`compute_empirical_confusion_matrix`) — no new simulation anywhere.

### Tier 1 (primary): `horizon_sweep_full26.parquet`

**Available: 52 of 52 expected cells** (26 environments × 2 classifiers;
the 23-environment gap identified in the prior version of this document
was closed by a dedicated sweep — see §7.2 for the full provenance/cost
record of that gap and the run that closed it). The now-superseded `n=3`
in-sample analysis (3 environments, `horizon_sweep.parquet`) is no longer
reported as a validation number; it was in-sample by construction (kappa
fit and evaluated on the same 3 cells in every row) and is retained only
in git history, not below.

At `kappa=1.0` (uncalibrated, still not interpretable as validation on its
own — `kappa` is a free multiplicative constant): median abs. rel. error
across all 26 environments is large and in the same direction as the
earlier 3-environment sample (over-prediction dominates), consistent with
Task 10's own uncalibrated predictor carrying the same bias. §7.1 fits it
properly, using a real held-out split this time.

### 7.1 Kappa calibration — pre-registered, environment-level held-out protocol (result)

Executed exactly as pre-registered in
[`kappa_preregistration.md`](kappa_preregistration.md), written and
committed BEFORE this data existed: `KAPPA_SPLIT_SEED=19001` (18 TRAIN /
8 TEST environments, `canonical` fixed in TRAIN), `form="kl"`, bounded
log-space least-squares fitting, `R ≤ 1.5` decision threshold, `n≥5`
power floor, `BOOTSTRAP_SEED=19002`/`n_boot=2000`. Script:
`backend/scratch/task19_kappa/fit_kappa_preregistered.py`; outputs in
`results/phase10/crossover_theory/kappa_preregistered_*.parquet`. No
parameter in this protocol was chosen or adjusted after seeing this
result.

**Power check (passed):** 8/8 TEST cells and 18/18 TRAIN cells per branch
were finite — well above the pre-registered floor of 5. This result is
not underpowered.

**Branch A (transfer) — fit `kappa_A` on the 18 TRAIN environments'
`C=I` cells, apply UNCHANGED to the 8 TEST environments' `C≠I` cells:**

| env_id | predicted_T | observed_T | ratio | signed rel. error |
|---|---:|---:|---:|---:|
| gap0.20_idx00 | 78.8 | 53 | 1.487 | +48.7% |
| gap0.20_idx01 | 99.3 | 65 | 1.527 | +52.7% |
| gap0.20_idx06 | 82.1 | 64 | 1.283 | +28.3% |
| gap0.20_idx08 | 80.6 | 14 | 5.756 | +475.6% |
| gap0.20_idx11 | 73.6 | 109 | 0.675 | −32.5% |
| gap0.20_idx16 | 10.0 | 30 | 0.333 | −66.7% |
| gap0.20_idx17 | 63.3 | 11 | 5.752 | +475.2% |
| gap0.20_idx24 | 89.8 | 53 | 1.694 | +69.4% |

`kappa_A = 0.044381` (n=18 TRAIN cells used). **`E_A` (median abs. rel.
error on held-out `C≠I` TEST) = 59.7%.**

**Branch B (in-condition ceiling) — fit `kappa_B` on the SAME 18 TRAIN
environments' `C≠I` cells, apply to the SAME 8 TEST environments:**

`kappa_B = 0.042559` (n=18). **`E_B` (median abs. rel. error on the same
held-out cells) = 40.5%.**

**Result: `R = E_A / E_B = 0.5970 / 0.4053 = 1.4729`.**

**Decision (pre-committed rule, applied without adjustment): `R ≤ 1.5` →
Case 1.** `R = 1.4729` satisfies the rule — **1.8% below the threshold**.

**This margin is thin, and the pre-registered bootstrap does not confirm
robustness.** Resampling the 8 TEST environments with replacement (2,000
draws, reusing the already-fitted `kappa_A`/`kappa_B` per §5.1 of the
pre-registration) gives a 95% CI for `R` of **[1.00, 9.55]** — a range
that reaches deep into Case-2 territory (up to ~6.4× the decision
threshold). The point estimate satisfies the pre-committed rule and Case 1
is therefore the reported outcome, exactly as pre-registered; but the
bootstrap makes clear this is a **fragile** Case 1, not a robust one — a
different draw of held-out environments could plausibly have produced
`R > 1.5`. Both facts are reported together, per §5 of the pre-registration
(the bootstrap CI is required reporting, not optional context).

**Pre-committed claim (Case 1 wording, from `kappa_preregistration.md`
§7, used verbatim):** *"`P_eff` captures the classifier-noise effect; the
residual prediction error is INHERITED Task 10 asymptotic-bound bias, not
error introduced by this extension."* This is a claim relative to Task
10's existing bound, not a claim that the bound is now accurate in
absolute terms: **the absolute error at the fitted kappa remains large
(59.7% / 40.5% median on the two branches)**, and Task 10's own asymptotic
bound remains poorly calibrated in absolute terms — this test quantifies
that pre-existing limitation transferring cleanly across classifier-noise
conditions; it does not resolve it.

**Calibrated-predictor naming (pre-committed, `kappa_preregistration.md`
§7).** The operative kappa under this Case 1 outcome is `kappa_A =
0.044381` (the transfer branch's fitted value), which falls inside the
pre-registered range `[0.00045, 0.045]` (within one order of magnitude of
Task 10's `best_fit_c ≈ 0.0045`) — right at the top edge of that range.
Per the pre-committed rule, `crossover_horizon_confused` (and, by
inheritance, `predict_crossover`) is described as **a calibrated
predictor**: an empirical fit with theoretical motivation from the
regret-bound derivation, not a validated asymptotic bound in the
mathematical sense.

**Diagnostic (not part of the decision rule).** Applying `kappa_A`
unchanged to the 8 TEST environments' held-out **`C=I`** cells (a sanity
check that the fit itself is not degenerate) gives median abs. rel. error
49.1% — similar order to `E_A`, i.e. the fit is not obviously broken, just
imprecise. Fit-uncertainty bootstrap (resampling the 18 TRAIN
environments, 2,000 draws, refitting each time): `kappa_A` 95% CI
`[0.0369, 0.0544]`, `kappa_B` 95% CI `[0.0105, 0.0426]` — `kappa_B`'s
interval is markedly wider and extends much lower than `kappa_A`'s,
another sign that the C≠I fit is less stable than the transferred one.
Per-cell residuals are highly heterogeneous: `gap0.20_idx08` and
`gap0.20_idx17` are large over-predictions under both kappas (ratio
~5.4–5.8×), while `gap0.20_idx16` is a consistent ~3× under-prediction —
no single scalar kappa corrects this cell-to-cell spread, and no attempt
was made to (per the pre-registration's no-rescue commitment).

**Classification: PARTIALLY-SUPPORTED**, not VALIDATED, not
UNVALIDATED/PRELIMINARY. Case 1 is the reported outcome under the
pre-committed rule, and it moves `crossover_horizon_confused` off pure
in-sample-only evidence for the first time — but the fragility disclosed
above (thin margin, bootstrap CI spanning both cases, large absolute
error) means this is not a clean validation, and is not reported as one.
No second kappa specification, re-split, or rescue attempt was made or
will be made in response to this result, per the pre-registration's hard
commitment (§8 there).

### 7.2 Provenance of the missing 46 cells, and the cost of closing the gap

A dedicated forensic investigation (read-only; no simulation, no
materialization) was run to resolve a prior conflicting claim about
whether `horizon_sweep.parquet` had ever contained the full 26-environment
grid.

- **What was inspected:** `results/tables/horizon_sweep.parquet` (3
  `env_id` values, 1,386 rows, mtime 2026-08-05 14:45:10) and every one of
  the 617 files in the `results/runs/horizon_sweep/` RunStore (loaded
  exhaustively, not sampled; RunStore file mtimes span 2026-08-04 20:18:08
  to 2026-08-05 14:44:41 — the table was materialized ~30 seconds after
  the RunStore's last write, i.e. it is not stale relative to the store).
- **RunStore key scheme:** `RunConfig.content_key()` = first 24 hex chars
  of `sha256(json({method, env_id, seed, n_episodes, classifier_id,
  prior_id}, sort_keys=True))`; each file also embeds `env_id` (and the
  other key fields) as literal columns, so env identity is directly
  readable from file content, not only inferable from the hash.
- **Finding: the 23 missing environments have zero trace anywhere in the
  RunStore.** All 617 files decompose as `canonical` (485 files),
  `gap0.20_idx00` (66), `gap0.20_idx01` (66) — nothing else. Since the
  store is content-addressed and files are never deleted (file presence
  *is* the resume mechanism), this rules out both a filtering bug in the
  materialization step and a lost-but-still-present RunStore state:
  **rematerialization is not possible** because there is nothing to
  rematerialize from.
- **Reconciled against `docs/experiments/instrumentation.md`** (dated
  2026-08-06, after the separately-documented `--smoke`-overwrite
  incident described in `docs/PROJECT_STATUS.md` had been resolved for
  this file): the current 3-environment file is the intended, final,
  documented output of `python experiments/horizon_sweep.py --n-seeds 3
  --n-gap-matrices 2`, an explicit **REDUCED (documented deviation)**
  scope after the true full grid was measured (via the script's own
  `--estimate`) at ~73 hours on this hardware and cost-gated out of
  `reproduce.py --mode paper`. The earlier "26 environments" belief traces
  to `PROJECT_STATUS.md`'s now-superseded note about a full-scale
  regeneration "actively running in the background" at the time of the
  smoke-overwrite incident — a job the later, authoritative record shows
  was abandoned in favor of the documented reduced scope, never completed.
- **Cost to close the gap.** Filling the 23 missing environments (46
  `(env, classifier)` cells) requires a genuine new simulation run — not a
  rerun of anything cached. Using `horizon_sweep.py`'s own `--estimate`
  facility (calibrated throughput per classifier, scaled by total episodes
  requested), restricted to exactly the missing cells (`gap0.20_idx02`
  through `gap0.20_idx24`, both classifiers, all 11 methods):

  | config | new cells | total episodes | estimated wall-clock (n_jobs=8) |
  |---|---:|---:|---:|
  | matches current production (`n_seeds=3`) | 1,518 | 75,900,000 | **~13.4 hours** (~804 min) |
  | full documented scale (`n_seeds=30`) | 15,180 | 759,000,000 | **~125.5 hours** |

  Both estimates are almost entirely `classifier_id="real"` time (the
  trained classifier runs ~200 episodes/sec vs. ~50,000–64,000/sec for the
  synthetic "perfect" classifier — 2-3 orders of magnitude slower, per
  `TrainedClassifierAdapter`'s own documented characteristic): at
  `n_seeds=3`, `perfect` costs ~3.2 min and `real` costs ~801 min of the
  ~804 total.

  **Exact command to run** (content-addressed RunStore skips the 3 cells
  already computed automatically; a non-default `--output` is used
  deliberately, per `docs/reproducibility.md`'s documented
  `--smoke`/full-mode default-output-collision lesson):

  ```
  python experiments/horizon_sweep.py --n-seeds 3 --n-gap-matrices 25 \
      --output results/tables/horizon_sweep_full26.parquet
  ```

  This run **was subsequently launched, with explicit approval, and
  completed** — but not on the first attempt, and not at the scope shown
  above. Full incident-level detail (three attempts, a disproven
  BLAS-oversubscription hypothesis, two independent estimator misses, and
  a measured-effective-parallelism correction) is recorded in
  `docs/experiments/instrumentation.md`; summarized here for the
  cost/provenance record:

  - The `--n-seeds 3` figure above (**1,518 cells, 11 methods, ~13.4h
    estimate**) was the cost of closing the gap for ALL methods. The
    actual run was scoped down to the 2 methods Tier 1 needs
    (`static_blind`, `thompson`) — **312 cells total, 204 new** — with a
    fresh `--estimate` of **~38.4 min** for that reduced scope. The other
    9 methods' cells for these 23 environments remain unfilled; they are
    a separate, smaller future cost if ever needed (not gated here).
  - **Actual measured wall-clock to complete the 204 new cells: ≈81.9
    minutes** (Attempt 2: 60 cells in a measured 1,492s span before being
    killed on a since-corrected stop-rule misapplication; Attempt 3,
    resumed from the banked 168 cells: 144 cells in a script-timed
    3,367.45s `run_grid` pass + 54.43s of horizon evaluation, `Wrote 2184
    rows ... in 3421.88s total`) — **≈2.13× the 38.4 min estimate**, the
    same direction and a comparable magnitude to the two earlier
    in-flight misses on this same estimator (Attempt 1's much larger miss
    at 11-method scope, and Attempt 2's own ~2.3× early-rate miss at this
    scope), now confirmed as a real total-run result rather than a
    projection.
  - **Effective parallelism, measured twice, did not agree**: ≈1.90×
    (Attempt 2, from RunStore write timestamps) vs. ≈2.74× (Attempt 3,
    from the script's own `run_grid` timer) — a ~44% difference between
    two measurements of nominally the same quantity, on the same machine,
    for the same workload shape, taken under different concurrent
    foreground load. No controlled comparison was run to attribute the
    gap; it is recorded as open, consistent with the measurement-hygiene
    finding in `instrumentation.md` rather than smoothed into one number.
  - `reproduce.py`'s ~73h (`horizon_sweep.py`, full un-filtered scope) and
    ~66h (`prior_grid.py`) headline cost figures remain **un-re-derived**
    and suspect in both directions for the same reason established during
    this run: they rest on the identical `n_jobs`-linear-scaling
    assumption this run's own estimator missed twice.

### Tier 2 (secondary, static-side only): `table_vii.parquet`

**Table VII has no adaptive-method trajectory** (its `method` column is
`"Fixed Rule"` exclusively) — this tier validates `static_rate_confused`
only, and is never used to claim the adaptive crossover theory is
validated.

| noise_level (β) | n seeds | predicted rate | empirical rate | abs. error |
|---:|---:|---:|---:|---:|
| 0.0 | 50 | 0.4408 | 0.4306 | 0.0102 |
| 0.1 | 50 | 0.4691 | 0.4582 | 0.0109 |
| 0.3 | 50 | 0.5256 | 0.5112 | 0.0144 |
| 0.5 | 50 | 0.5821 | 0.5744 | 0.0077 |
| 0.8 | 50 | 0.6669 | 0.6584 | 0.0085 |
| 1.0 | 50 | 0.7234 | 0.7156 | 0.0078 |

**Median absolute error: 0.0093** — the static-side (exact, non-asymptotic)
prediction tracks real simulated rates closely across the full noise
range. This is the strongest validation result in this document,
consistent with §3.2 being an exact closed-form composition rather than an
asymptotic approximation.

## 8. Exact math vs. inherited approximation vs. empirical validation — summary

| Component | Category | Status | Classification |
|---|---|---|---|
| `effective_matrix_under_confusion` (eq. 1–2) | Exact math | Proved, unit-tested | **VALIDATED** |
| `static_rate_confused` (eq. 3) | Exact math | Proved; empirically validated, median abs. error 0.0093 (Tier 2, n=6 noise levels × 50 seeds) | **VALIDATED** |
| `ts_cumulative_regret_confused` (eq. 4) | Inherited asymptotic approximation (A4) | Unit-tested for correct composition; NOT independently re-derived; not separately empirically validated (only reachable empirically through eq. 5) | **PARTIALLY-SUPPORTED** (inherits eq. 5's status) |
| `crossover_horizon_confused` (eq. 5) | Inherited asymptotic approximation (A4) | Pre-registered, environment-level held-out protocol (§7.1, `n=26` envs, 18 train/8 test): Case 1 by point estimate (`R=1.4729 ≤ 1.5`), but the pre-registered bootstrap CI for `R` is `[1.00, 9.55]` — reaches deep into Case-2 territory. Absolute median error remains large (59.7%/40.5%, two branches). Described as a **calibrated predictor** (fitted `kappa_A=0.0444`, within the pre-registered `[0.00045,0.045]` range), not a validated bound | **PARTIALLY-SUPPORTED** |
| `ceiling_reduction` (eq. 6) | Exact math | Proved, unit-tested | **VALIDATED** |
| `gap_shrinkage_report` | Exact math (a report, not a prediction) | Unit-tested | **VALIDATED** |
| `argmax_inversion_check` | Exact math (a check, not a prediction) | Unit-tested; does not occur at the real classifier's measured confusion | **VALIDATED** |
| `T*` non-monotonicity in accuracy (§6) | Exact mathematical finding (a property of eq. 5, established independently of any calibration) | Proved with a clean 2-type construction, unit-tested | **VALIDATED** (as a finding about eq. 5's shape, not as a claim that eq. 5's predicted values are accurate) |
