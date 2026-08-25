# Why a Unified Drift-Crossover Law Was Not Derivable (A Negative Result)

This document reports the outcome of an attempt to extend Task 10's stationary
analytic crossover predictor (`recovery.theory.crossover.predict_crossover`) to
non-stationary (drifting) environments. Three successive, pre-registered
attempts at a low-dimensional summary statistic capable of predicting when an
adaptive policy overtakes a static one under drift were tested against Task 8's
drift taxonomy and found insufficient. The negative result — and one genuine,
mechanism-grounded positive finding buried inside it — are reported here in
full, including a self-caught methodological error, because the process of
ruling attempts out is itself the scientific content of this document.

---

## 1. The question

Task 10 answers, for a **stationary** environment: given a static recovery
table with per-episode regret rate `rho` and Thompson Sampling's asymptotic
cumulative-regret bound, at what horizon `T*` does learning online overtake
the static table? The natural next question, given Task 8's drift taxonomy
(`recovery.envs.drift`: `PermutationDrift`, `MonotoneDecayDrift`,
`PartialReorderDrift`, `GradualRotationDrift`) and its dense, already-cached
grid (`drift_taxonomy.py`, 12,800 simulated cells across 4 drift types × 4
frequencies × 4 magnitudes × 20 seeds × 10 methods), was whether the same kind
of closed-form-or-numerical prediction could tell us, under drift, when an
adaptive policy's advantage over a static one is *positive* at all — a sign
prediction, prior to any attempt at predicting its magnitude.

This looked derivable a priori. Task 1 supplies an exact static regret rate.
Task 10 supplies an asymptotic Thompson Sampling regret bound. Task 8 supplies
four independently-motivated, physically-interpretable drift mechanisms and a
grid large enough to calibrate against. The ingredients for a Task-10-style
derivation all appeared to be already sitting in the repository.

**The falsifiable structure adopted before any attempt:** a candidate summary
statistic must (a) separate the four drift types in a way that (b) predicts
the *sign* of `adaptive_advantage` (defined throughout as `stock_thompson`'s
minus `static_rule`'s late-stage time-varying normalized score,
`results/tables/drift_per_cell_metrics.parquet`) better than a majority-class
guess, **out of sample** (leave-one-cell-out). Three candidates were
pre-registered and tested in this order. A fourth was explicitly ruled out in
advance once the third failed.

---

## 2. Attempt 1 — (s, g): survival and gap-change

**Motivation.** The most direct translation of "does the argmax survive a
shift, and does the gap change" into two summary numbers: `s` = fraction of
(shift event, touched row) instances where the pre-shift optimal action is
still optimal after the shift; `g` = mean change in the (best − second-best)
value gap, each side computed on its own post-event ranking.

**Method.** Environments were reconstructed deterministically (no new
simulation) via `experiments.drift_taxonomy.build_environment`, replaying
`.step()` to recover each cell's compact `matrix_history()`. `s` and `g` were
pooled per drift type and per `(drift_type, frequency, magnitude)` cell.

**Result.** `R²(s, g)` jointly = **0.012** — essentially no explanatory power.
`g` proved near-tautological in this specific taxonomy: three of the four
drift models (`permutation`, `partial_reorder`, and `gradual_rotation`'s
target construction, which is itself a row permutation of the base) preserve
each touched row's value *multiset* by construction, so `g ≈ 0` for all three
is close to a structural fact about this taxonomy, not an emergent empirical
finding. `partial_reorder` sits at the single deterministic point `(s=0,
g=0)` — distinguishable from the others, but at a point that carries no
sign-predictive information on its own.

**Verdict: insufficient.**

---

## 3. Attempt 2 — (d, τ): magnitude and a theory-derived recovery ratio

**Motivation.** `s` is binary (did the argmax move); `d` is continuous (how
much is lost per decision by *not* moving — a synthetic 1-row
oracle-minus-stale-value quantity). `τ` was constructed to normalize available
recovery time (mean inter-shift interval) against a theoretical relearning
horizon, so that comparisons across drift types with very different damage
magnitudes would sit on a common scale.

**Method.** `d_row = P_after[row, argmax_k P_after[row,k]] − P_after[row,
stale]`, `stale = argmax_k P_before[row,k]`. `τ`'s denominator was computed by
calling `predict_crossover` on a synthetic 1-row matrix `P = P_after[row:row+1,
:]`, `pi = [stale]`, `κ = 1.0` (uncalibrated).

**Result.** `R²(d, τ)` jointly = **0.579**; adding `magnitude` as a third
regressor raised it to **0.735** — a substantial-looking improvement over
Attempt 1.

### 3.1 The τ circularity self-audit

Before drawing any conclusion from the numbers above, τ's construction was
audited and found conceptually flawed, and the flaw is reported here
explicitly rather than left implicit in a footnote.

`τ`'s denominator **is `predict_crossover`** — Task 10's own crossover-horizon
predictor, the exact class of deliverable a drift-aware crossover theory was
supposed to derive and independently validate. `τ > 1` is, by construction,
*definitionally* "the existing stationary crossover theory, applied to a
single row's post-shift state, predicts recovery within the available
window." Testing whether `τ` predicts observed `adaptive_advantage` is
therefore, to a substantial degree, testing whether Task 10's existing
asymptotic bound has *some* validity when applied per-row to a drifting
setting — not independent confirmation of a new environmental descriptor.

Two distinct notions of circularity were separated explicitly:

- **Statistical circularity** (data leakage, shared fitted parameters between
  predictor and target): **absent**. `τ`'s computation never touches
  `adaptive_advantage`'s data source (simulated RunStore logs); `κ` was fixed
  at 1.0, never fit to the target.
- **Conceptual circularity** (the predictor is an application of the same
  theoretical machinery the analysis is supposed to validate):
  **present**, and material.

**The R² numbers in this section are retracted as validation evidence.** They
are kept in this document only as the historical record of what was tried and
why it does not count. A non-circular replacement (`log_exposure =
log(inter_shift_interval) + log(d)`, touching no crossover-theory function at
all) was substituted for the sign-prediction test that followed, and it too
failed to beat the majority baseline (79.7%, in-sample and leave-one-cell-out,
identical misclassified set both times — the fitted threshold landed below
every data point, collapsing to "always predict positive").

Catching and retracting this before it was reported as a finding is stated
here as a credibility asset, not buried: a reader who sees this project catch
its own circular reasoning should trust the surviving claims in this document
more, not less.

**Verdict: insufficient, and the apparent R² was largely an artifact of reusing
the theory under test as an input feature.**

---

## 4. The magnitude aggregation artifact

Diagnosis: `d` (and `s`, `g`) were computed as means over *touched rows only*,
blind to how many of the environment's `N` rows were touched at a given
event. `magnitude`'s apparent contribution in Attempt 2 (`R²` +0.156, from
0.579 to 0.735) was substantially an artifact of this exclusion — a
per-touched-row statistic will always look like it needs a "how many rows"
correction, because it discards exactly the information that correction
supplies.

**The fix:** `D`, an environment-level, prior-weighted regret,

```
D = sum_t p[t] * d_t,   d_t = 0 automatically for any untouched row
```

(`p` uniform, `1/N`), verified numerically identical to the **existing**
`recovery.theory.crossover.static_regret_rate(P_after, pi_stale, p)` to
**2.78e-16** across all 16,640 events checked — an *identity* check on `D`'s
construction (`D` and `static_regret_rate` compute the same quantity by
definition, given the full-`N` `pi_stale` vector), not a predictive use of
theory code; `static_regret_rate` is never used anywhere as a feature.

Once `D` is computed at the correct (environment) level: `R²(D, R) = 0.676`,
`R²(D, R, magnitude) = 0.682` — **Δ collapses from +0.156 to +0.006**. The
general lesson, stated for future work: a summary statistic computed at the
wrong level of aggregation will masquerade as needing an extra explanatory
variable; the correct fix is recomputing it at the right level, not adding a
patch variable on top.

---

## 5. Attempt 3 — (D, R): magnitude and reversion

**Motivation** (from the reviewing party's diagnosis): `s`/`d` measure
*whether/how much* damage occurs at a shift, but not whether it *persists*.
Persistence, not magnitude, should govern whether tracking a moving target is
worth an adaptive learner's exploration cost.

**R's precise definition.** Population restricted to `(row, event)` pairs
where the argmax **actually changed** (`best_after ≠ best_before`) — a strict
subset of `D`'s population, which also includes zero-change events (trivially
`d=0`). `R` = fraction of those change-events where the new optimum had been
optimal at **any earlier point in that row's history within the run** —
*not* restricted to the immediately-previous state.

This lookback choice was fixed before computing anything, for a specific,
checkable reason: `PartialReorderDrift`'s best↔second swap has **period 2**.
Event 1: `A→B` (B becomes optimal). Event 2, if the same two columns remain
top-2 (they do, since the swap only exchanged their values): the swap
reverses, `B→A` — event 2's new optimum equals event **0**'s optimum, not
event 1's. A strict "immediately-previous-only" (lookback = 1) definition
would compare event 2's result against event 1's optimum (`B`), find no
match, and wrongly report `R ≈ 0` for `partial_reorder` — the opposite of
what the mechanism actually does. Using the full available history is the
only definition under which `partial_reorder`'s oscillation is even visible,
and it is the one definition used, unchanged, across every drift type.

`gradual_rotation` has exactly one transition (no repeated shocks), so its
"history" contains only the base matrix's optimum. `R` is mathematically
**well-defined** for it (`P(target's optimum == base's optimum)`) — not
undefined or degenerate — but it measures a structurally different quantity:
a one-shot coincidence rate with the base matrix, not a genuine
repeated-oscillation signal, estimated from far fewer independent instances
per cell than the other three types.

**Pre-registered predictions, verbatim, and their results:**

- **P1** (`R` high for `partial_reorder`, near zero for `monotone_decay`,
  intermediate for `permutation`): **partially holds.** Correct ordinal
  ranking (`monotone_decay ≈ 0 < permutation 0.7217 < partial_reorder
  0.9412`), but `permutation` sits much closer to `partial_reorder`'s "high"
  value than to a true midpoint.
- **P2:** see Section 6 — confirmed cleanly, promoted to a standalone finding.
- **P3** (a two-variable `(D, R)` sign rule beats the 79.7% majority baseline
  out of sample): **fails.** Pre-registered rule `sign(D·(1−R) − θ)`, one
  free parameter: in-sample **79.7% (51/64)**, leave-one-cell-out **79.7%
  (51/64)** — exactly the majority baseline, zero improvement, identical
  13-cell misclassified set both times. `θ` fit below the entire range of
  `D·(1−R)` values, again collapsing to "always predict positive."
  Diagnosis: high `R` is necessary but not sufficient for a negative outcome
  — `partial_reorder` at `frequency=500` (`R≈0.667`, still fairly high) is
  positive in all 4 of its magnitude cells, while `partial_reorder` at
  `frequency ≤ 250` (similar-to-higher `R`) is negative. Available recovery
  time is doing real, separate work that a formula omitting the inter-shift
  interval entirely cannot capture.

---

## 6. CONFIRMED POSITIVE FINDING — reversion, not magnitude, governs sign

Reported as its own headline result, separated from the failed quantitative
sign rules above: **the sign of adaptive advantage under drift is governed by
whether staleness damage reverts, not by its magnitude.**

- All **13 of 13** negative-`adaptive_advantage` cells sit at `R ≥ 0.68` (10
  `partial_reorder`, `R` ∈ [0.857, 0.974]; 3 `permutation`, `R` ∈ [0.683,
  0.845]).
- All **32 of 32** `R ≈ 0` cells (16 `monotone_decay` + 16
  `gradual_rotation`) are positive, ranging 0.031 to 0.754.
- **Independent seeds underlying this split:** every cell in the grid is
  built from `DEFAULT_N_SEEDS = 20` independent seeds
  (`base_seed=20000 + seed_idx`, `seed_idx ∈ [0,20)`), shared as Common
  Random Numbers across methods within a cell. The 13 negative cells rest on
  **260 independent seed-level environment draws** (13 × 20); the 32 `R≈0`
  positive cells rest on **640** (32 × 20). These are the same seeds that
  produced both the `(D, R)` statistics and the `adaptive_advantage`
  measurements for their respective cells — one shared draw per cell, not two
  independent samples.
- `monotone_decay`'s `R` is stated precisely, not as exactly zero: `R ≈ 0`,
  **95% CI [0.0000, 0.0160], n = 236** (the reversion-eligible population —
  only events where the argmax actually moved — which is small for this type
  precisely because it moves so rarely). The small-sample caveat is part of
  the finding, not a footnote to discard.

This qualitative pattern is real and robust even though no quantitative
threshold on it (Section 5, P3) survives out-of-sample testing — a
correlational, mechanism-consistent pattern can hold while a specific
numerical boundary built on top of it does not.

---

## 7. The reversion structure explains the inversion

This connects directly to a previously unexplained observation from Task 8's
own analysis: static beats adaptive in 10 of 16 `partial_reorder` cells (the
"partial-reorder inversion").

**The reversion structure explains the inversion**: best↔second swaps revert
(the period-2 mechanism in Section 5), so the stale action the static table
keeps playing retains near-optimal value most of the time, while the adaptive
learner pays real exploration cost chasing a target that keeps flipping back
to where it already was.

This is stated as an explanation grounded in a verified mechanism plus a
strong correlational pattern (Section 6) — **not as a causal claim, and not
as the output of an intervention experiment.** The evidence available here is
observational: `partial_reorder`'s specific swap mechanism was designed
before this analysis and was not varied independently of its resulting `R`
value. A genuine causal test would require constructing drift processes
**matched on `D`** (same average per-event damage magnitude) while
**independently varying `R`** (holding damage magnitude fixed, varying only
whether it reverts or persists), then comparing `adaptive_advantage` under
those controlled conditions. That experiment was not run and is out of scope
for this document; the claim here is limited to "this mechanism, evidenced
correlationally and consistent with the verified reversion structure,
explains the inversion" — not "reversion causes the inversion" in the sense
a controlled intervention would be needed to establish.

---

## 8. Exploratory observation (clearly separated — not a validated rule)

> **Everything in this box is post-hoc, hypothesis-generating, and explicitly
> not a candidate for implementation.**

A 2-parameter grid search over a rule combining `R` with the inter-shift
interval (`predict negative iff R > θ_R AND interval < θ_I`) was run *after*
seeing which cells broke the pre-registered `(D,R)` rule, specifically to
characterize the failure, not to produce a fourth parameterization:

- In-sample: **92.2% (59/64)**, best at `θ_R = 0.697, θ_I = 500`.
- Leave-one-cell-out: **87.5% (56/64)** — a real 4.7-point drop from
  in-sample, the signature of genuine overfitting risk.

Both numbers exceed the 79.7% baseline, unlike every pre-registered attempt.
This is reported because it suggests what a real drift law probably needs to
combine (persistence *and* available recovery time, jointly) — not because it
constitutes evidence. It was fitted after seeing which cells break sign
classification; it has 2 free parameters against 64 cells that are not
independent draws (20 seeds pooled per cell, heavy within-type structural
correlation, 13 negatives concentrated in only 2 of 4 drift types). It is not
promoted to a finding anywhere else in this document, and it is not
implemented.

---

## 9. Honest scope statement

Three attempts, in order, with what each ruled out:

| Attempt | R² (sign accuracy) | Verdict |
|---|---|---|
| `(s, g)` | 0.012 (baseline) | Insufficient; `g` near-tautological in this taxonomy |
| `(d, τ)` | 0.579–0.735 (retracted) | Conceptually circular — `τ` reuses the theory under test |
| `(D, R)` | 0.676 (79.7% sign acc = baseline) | Mechanistically correct qualitative story (P2); no quantitative sign rule survives LOO |

`drift_type` fixed effects retain `R²` delta **+0.092** beyond `(D, R)` —
real, unabsorbed structure. Candidate unaccounted properties are named
without being operationalized, per the discipline that stopped this
investigation at three attempts: **disruption abruptness** (instantaneous
jump vs. `gradual_rotation`'s multi-episode ramp), **within-row breadth**
(`monotone_decay` perturbs one column; `permutation`/`partial_reorder`
perturb the whole row at once), and **available recovery time relative to
shift frequency** (Section 5's frequency-dependent sign flip within
`partial_reorder`, and Section 8's exploratory signal, both pointing at the
same gap independently).

**Explicit statement:** the sign of `adaptive_advantage` under drift is **not
reducible to a two-statistic summary** across these four mechanistically
distinct drift regimes, given the data and methods available in this
repository as of this task.

---

## 10. What a proper drift-crossover theory would require

- **Per-mechanism derivation, not a unified summary statistic.** Each drift
  model likely needs its own closed-form or semi-closed-form treatment —
  `partial_reorder`'s period-2 structure suggests a Markov-chain/renewal-
  process treatment qualitatively different from `monotone_decay`'s
  monotone-approach-to-a-floor structure.
- **A denser grid with independent environments per cell.** The 64-cell grid
  used throughout this document shares one canonical `MATRIX` and one
  `env_seed`-driven affected-row draw per cell; it is not 64 independent
  draws of "a drift regime," and several of the statistics above (especially
  `monotone_decay`'s `R`, n=236) are limited by this.
- **Interval/frequency as a genuine first-class variable**, not smuggled in
  via a circular theory-derived ratio (Section 3.1) — Sections 5 and 8 both
  point at this independently.
- **A controlled intervention design** (Section 7) — drift processes matched
  on `D`, independently varied in `R` — to move the reversion-explains-
  inversion claim from correlational-plus-mechanism to causal.

This document does not attempt any of the above. It documents why the
attempted low-dimensional summary-statistic approach did not succeed, as a
negative result in its own right.

---

## Related documents

- [`crossover.md`](crossover.md) — the stationary (Task 10) analytic
  crossover predictor this document's Section 3.1 audits.
- [`crossover_derivation.md`](crossover_derivation.md) — the 19A
  classifier-noise-aware extension (stationary case), unaffected by any
  finding in this document.
- [`docs/statistics_registry.md`](../statistics_registry.md) — the P2 claim,
  the retracted `(d,τ)` numbers, and the exploratory `(R, interval)` result
  are registered there, tagged by claim type.
