# Kappa Held-Out Validation Protocol for `crossover_horizon_confused` — Pre-Registration

**Status: PRE-REGISTERED. Written and posted BEFORE the 46-cell horizon-sweep
run is launched, and before any cell from that run is analyzed.** Nothing
below was written or adjusted after seeing any new data. The only numbers
computed so far are the deterministic environment split (§2) and the
bootstrap seed (§5) — both fixed by seed choices stated here, not by
outcome.

## 1. Why this document exists

The prior kappa fit (`docs/theory/crossover_derivation.md` §7.1) fit kappa
and evaluated it on the *same* 3-cell subset in every comparison — an
in-sample number at `n=3`, which is uninterpretable as validation evidence
in principle, regardless of what the resulting error looked like. This
document fixes the protocol that will be used once the 46 missing cells
land, so that the resulting evaluation has actual held-out structure. If
this document is written *after* the data, the same circularity problem
resurfaces in a new form (a protocol shaped to the answer). It is therefore
committed now.

## 2. Train/test split — by ENVIRONMENT, fixed by seed, computed now

The full Tier 1 grid (after the sweep) will have 26 environments
(`canonical` + `gap0.20_idx00`...`gap0.20_idx24`) × 2 classifier conditions
= up to 52 cells. Cells sharing an environment are not independent (same
underlying `P` matrix), so the split is done at the environment level, and
no environment appears in both sides.

**`canonical` is placed in TRAIN unconditionally.** It is the only
environment in this grid that is not a draw from the `gap=0.20` sampled-
matrix family (`src.evaluation.matrix_family.sample_matrix`) — it is a
fixed, hand-designed matrix with no other examples of its own family to
generalize to. Holding it out would test generalization to a family
represented by a single point, a different (and separately already-flagged)
question from the one this protocol targets. This is a structural
decision, made before any of the 25 gap-matrices' data exists, not a
data-driven one.

The 25 `gap0.20_idx*` environments are split 17 (train) / 8 (test) via a
fixed seed, computed now:

```python
KAPPA_SPLIT_SEED = 19001
rng = np.random.default_rng(KAPPA_SPLIT_SEED)
held_out_gap_idx = sorted(rng.choice(25, size=8, replace=False).tolist())
```

Result (computed, not adjustable after this point):

- **TEST (held out, 8 environments):** `gap0.20_idx00`, `gap0.20_idx01`,
  `gap0.20_idx06`, `gap0.20_idx08`, `gap0.20_idx11`, `gap0.20_idx16`,
  `gap0.20_idx17`, `gap0.20_idx24`
- **TRAIN (18 environments):** `canonical`, `gap0.20_idx02`,
  `gap0.20_idx03`, `gap0.20_idx04`, `gap0.20_idx05`, `gap0.20_idx07`,
  `gap0.20_idx09`, `gap0.20_idx10`, `gap0.20_idx12`, `gap0.20_idx13`,
  `gap0.20_idx14`, `gap0.20_idx15`, `gap0.20_idx18`, `gap0.20_idx19`,
  `gap0.20_idx20`, `gap0.20_idx21`, `gap0.20_idx22`, `gap0.20_idx23`

**Disclosed coincidence, for transparency, not for adjustment:**
`gap0.20_idx00` and `gap0.20_idx01` — the two gap-family environments
already present in the currently-committed reduced dataset, whose
`kappa=1.0` errors are already known from the earlier (now-superseded)
Tier 1 analysis (90–99× ratios, the worst of the 6 cells examined then) —
both landed in TEST under this seed. This is a mechanical consequence of a
seed fixed before this document existed, not a choice; the seed is not
being redrawn because of it. Redrawing on observing this would itself be
exactly the seed-shopping this pre-registration exists to prevent. This is
named here so the eventual result cannot be read as if the coincidence
were undisclosed.

`KAPPA_SPLIT_SEED = 19001` will not be changed for any reason once the
sweep data lands.

## 3. The primary test (exactly as specified, restated precisely)

All fits use `method="thompson"`, `criterion` = cumulative (`time_to_parity`
median across seeds), and `form="kl"` — the same form used throughout
Task 19A and Task 10; **`form` will not be varied or selected after seeing
results.**

- **Branch A (transfer).** Fit a single `kappa_A` on the C=I cells of the
  18 TRAIN environments only (via `fit_best_kappa`, identical method to
  `experiments/crossover_calibration.py::fit_best_c`: bounded 1-D
  least-squares in `log(kappa)`, bounds `(log(1e-3), log(1e4))`). Apply
  `kappa_A` UNCHANGED to the C≠I cells of the 8 TEST environments. Compute
  `E_A` = median absolute relative error (`|predicted_T/observed_T − 1|`)
  across those held-out C≠I cells.
- **Branch B (in-condition ceiling).** Fit `kappa_B` on the C≠I cells of
  the SAME 18 TRAIN environments. Apply `kappa_B` to the C≠I cells of the
  SAME 8 TEST environments. Compute `E_B` analogously.
- **The result is `R = E_A / E_B`** — how much worse transfer performs than
  the best achievable same-condition fit, on identical held-out data.
  Both branches are evaluated on exactly the same 8 (or fewer, if any are
  non-finite — see §4) held-out cells, so `R` isolates the transfer gap
  from any difference in which cells were scored.

Secondary/diagnostic only (not part of the decision rule): `kappa_A`'s
error on held-out **C=I** cells of the TEST environments (a sanity check
that the fit itself is not degenerate); median signed relative error and
median log-ratio for both branches, reported alongside `E_A`/`E_B` for
transparency.

## 4. Pre-committed decision rule

- **Case 1 (one shared kappa suffices — the `P_eff` reduction captures the
  classifier-noise effect; residual error is inherited Task 10 asymptotic
  bias):** `R ≤ 1.5`.
- **Case 2 (the confusion-aware extension adds its own model error beyond
  what Task 10's bound already carries):** `R > 1.5`.

**Threshold justification, stated now.** `R` measures relative degradation
between two already-fitted-kappa error levels on identical held-out data —
a different kind of comparison from Task 10's existing "within factor 3"
validated-region bar (`crossover_calibration.py::VALIDATED_WITHIN_FACTOR3_THRESHOLD`),
which judges raw prediction accuracy against observation, not one fit's
degradation relative to another. Reusing factor-3 here would be far too
lenient — it would classify almost any transfer result as "fine." `R ≤
1.0` (zero tolerance for any degradation) is too strict given ordinary
finite-sample fitting noise between two disjoint training subsets, which
will produce *some* nonzero `R` even under a genuinely correct shared-kappa
model. `1.5` — no more than 50% relative degradation in the held-out error
— is chosen as a round, moderate middle point between these two, before
any held-out numbers exist. It is a judgment call, made in advance, not
derived from data; it is disclosed as such rather than presented as
principled from first principles.

## 5. Bootstrap procedure — environment-level resampling

Two separate bootstraps, both resampling ENVIRONMENTS (not cells) with
replacement, at `BOOTSTRAP_SEED = 19002`, `n_boot = 2000`, computed once
and not redrawn:

1. **Held-out error uncertainty.** Resample the (up to 8) TEST
   environments with replacement, recompute `E_A`, `E_B`, and `R` on each
   resample using the ALREADY-FITTED `kappa_A`/`kappa_B` (the fit itself is
   not redone per resample here — this bootstrap characterizes sampling
   noise in which held-out cells were scored, not in the fit). Report the
   2.5/97.5 percentiles of `R`'s bootstrap distribution alongside the
   point estimate.
2. **Fit uncertainty.** Resample the 18 TRAIN environments with
   replacement, separately for the C=I subset (re-fit `kappa_A` on each
   resample) and the C≠I subset (re-fit `kappa_B`), report the 2.5/97.5
   percentiles of the resulting `kappa_A`/`kappa_B` distributions. This
   is diagnostic (parallel to the `n=3` bootstrap already reported in
   §7.1 of the derivation doc, now done properly at environment-level with
   a real train set), not part of the Case 1/Case 2 decision rule, which
   uses only the point-estimate `R` and the held-out-resampling CI from
   (1).

With 18 train / 8 test environments, `18^18` and `8^8` unique resamples
respectively — no longer the ~27-resample degenerate regime of the earlier
`n=3` bootstrap; these intervals are treated as informative, not merely
disclosed-for-transparency.

## 6. Underpowered condition — stated in advance

If, after the sweep, fewer than **5** of the 8 pre-registered TEST
environments yield a finite `predicted_T`/`observed_T` pair for the C≠I
condition (e.g. due to `time_to_parity` never crossing within `t_max`, or
a truncated `crossover_horizon_confused` prediction), **no Case 1/Case 2
conclusion will be drawn**. The result will be reported as inconclusive/
underpowered, with the exact finite count stated, and `crossover_horizon_confused`
will remain classified UNVALIDATED/PRELIMINARY by default in that event —
not because Case 2 was concluded, but because the pre-registered test
could not be run at adequate power. The same floor applies to the 18 TRAIN
environments feeding each fit: if fewer than 5 TRAIN environments yield a
finite fit input for either branch, that branch's kappa is not fit at all
and the test is reported as not executable, not substituted with a smaller
ad hoc fit.

## 7. Addendum (2026-08-18) — pre-committed CLAIMS, not just the test

Written before the sweep landed and before any of its cells were
analyzed, at the same time as the protocol above was extended. §4
pre-registered the TEST; this section pre-registers what will actually be
WRITTEN into `docs/theory/crossover_derivation.md` and the report under
each of the three possible outcomes, so that outcome cannot shape the
write-up after the fact.

- **If `R ≤ 1.5` (Case 1):** The claim will be stated as: *"`P_eff`
  captures the classifier-noise effect; the residual prediction error is
  INHERITED Task 10 asymptotic-bound bias, not error introduced by this
  extension."* This is a claim RELATIVE to Task 10's existing bound, not a
  claim that the bound itself is now accurate in absolute terms — the
  absolute error at the fitted kappa (whatever it turns out to be) will be
  stated explicitly alongside this claim, and the write-up will state
  in the same paragraph that Task 10's own asymptotic bound remains
  poorly calibrated in absolute terms; this test QUANTIFIES that
  pre-existing limitation transferring cleanly, it does not RESOLVE it.
- **If `R > 1.5` (Case 2):** The claim will be stated as: *"The
  confusion-aware extension introduces error beyond what Task 10's bound
  already carries; a kappa fit under one classifier condition does not
  reliably transfer to another."* `crossover_horizon_confused` ships as a
  **documented non-result**, reported with the same care as a positive
  result, alongside `static_rate_confused` (§3.2/Tier 2), which remains
  validated on its own merits regardless of this outcome — the two
  components' validation status is independent and this outcome does not
  downgrade `static_rate_confused`. **No rescue attempt**: no alternative
  kappa specification, no re-split, no reweighting will be tried in
  response to this outcome (§ hard commitment below).
- **If underpowered (`<5` finite held-out cells, §6):** The claim will be
  stated as: *"The pre-registered held-out test could not be run at
  adequate power; no Case 1/Case 2 conclusion is drawn."*
  `crossover_horizon_confused` remains classified UNVALIDATED/PRELIMINARY
  by default — not because Case 2 was concluded, but because the test
  that would distinguish the two cases did not execute. **No fourth
  analysis** will be run to recover power (e.g. no loosening of the `n≥5`
  floor, no merging held-out cells across conditions).

**Bound vs. calibrated predictor — decided now, not after seeing the
fitted value.** Task 10's own `best_fit_c ≈ 0.0045` (a ~111× downward
correction on the stationary, noiseless predictor) is already, by the
project's own prior documentation, an empirically-fit multiplicative
constant layered on top of a Lai–Robbins-style asymptotic regret bound —
not a value derivable from the bound's own theory. **If this protocol's
fitted `kappa` (either `kappa_A` or `kappa_B`, whichever is reported as
the operative shared value under the eventual Case 1/Case 2 outcome)
falls within one order of magnitude of `0.0045` — i.e. in `[0.00045,
0.045]`** — both `crossover_derivation.md` and the report will describe
`crossover_horizon_confused` (and, by inheritance, `predict_crossover`
itself) as **a calibrated predictor**: an empirical fit with theoretical
motivation from the regret-bound derivation, not a validated asymptotic
bound in the mathematical sense. **Outside that range**, the fitted
constant will be reported as such without adopting either label
by default, and the magnitude itself will be discussed as a finding.
This naming decision is orthogonal to the Case 1/Case 2 decision rule in
§4 — a shared kappa can transfer cleanly (Case 1) and still be, in
magnitude, a large empirical correction rather than a near-1.0 constant
that would read as merely "confirming" the bound's own leading constant;
both facts will be reported together, not one instead of the other. This
threshold and the "calibrated predictor" language are fixed now and will
not be adjusted to fit whatever kappa is actually found.

## 8. Hard commitment — no second kappa specification

This is the only kappa specification that will be fit and reported.
`form="kl"`, the bounded log-space least-squares method, the fitting
bounds `(1e-3, 1e4)`, the 17/8 train/test split under `KAPPA_SPLIT_SEED=19001`,
the `R ≤ 1.5` threshold, and the `n≥5` power floor are all fixed by this
document. If the pre-registered protocol yields `R > 1.5` (Case 2), or
yields an underpowered result (§6), that is the reported finding.
**No alternative split, form, threshold, or fitting bound will be tried
after seeing the held-out numbers, and no such alternative will be
substituted into the derivation document even if it would have produced
a more favorable result.** This mirrors the hard-stop discipline already
applied to the 19B drift-crossover negative result
([`drift_crossover_negative.md`](drift_crossover_negative.md)) — a
pre-registered test that fails is a reported result, not a prompt for a
new attempt.
