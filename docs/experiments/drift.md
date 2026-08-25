# A Taxonomy of Drift

## Motivation

Every non-stationary result in this repository before Task 8 used exactly
one drift mechanism: randomly permute the strategy assignment inside a
subset of matrix rows (`PermutationDrift` below, unchanged in semantics).
That is the worst case for a fixed mapping -- a permutation can
simultaneously demote the previous optimum to the new worst arm and
promote the previous worst arm to optimal, in one event. Real production
drift is rarely this adversarial: APIs degrade gradually, rankings
reshuffle only at the top, migrations roll out over a window, and
deployments are often announced ahead of time. This task adds four more
physically-motivated drift models plus a signal decorator, so
non-stationary experiments can be evaluated across a taxonomy instead of
against a single worst-case mechanism.

## Mathematical definitions

Let `P` be the `(N, K)` success matrix (`P[t, k] = ` success probability
of strategy `k` for failure type `t`).

**PermutationDrift.** At each scheduled episode, for every affected row
`t`: `P[t, :] <- P[t, sigma]` for a random permutation `sigma` of
`{0, ..., K-1}`. The multiset of row `t`'s values is exactly preserved;
only the column->value assignment changes.

**MonotoneDecayDrift.** For one `(row, strategy)` pair per affected row
`t`: `P(e) = floor + (P0 - floor) * 2^(-(e - e0) / half_life)`, evaluated
at each scheduled episode `e` (`e0` = the first scheduled episode,
`P0 = ` the ORIGINAL base value). Strictly decreasing toward `floor` as
`e` grows (requires `floor <= P0`).

**PartialReorderDrift.** At each scheduled episode, for every affected
row `t`: let `best = argmax_k P[t, k]`, `second` its runner-up. Swap
`P[t, best]` and `P[t, second]`. The row's value multiset is exactly
preserved; only which column is optimal changes (from `best` to
`second`).

**GradualRotationDrift.** `P(e) = base + (target - base) * clip((e -
start) / window, 0, 1)`, evaluated continuously (every episode). Reaches
`base` exactly at `e <= start` and `target` exactly at `e >= start +
window`.

**AnnouncedDrift.** A decorator: exposes `shift_signal` (alias
`drift_signal`), `True` for exactly the episode immediately following a
`step()` call that produced a new compact history entry in the wrapped
model. No other behavior changes.

## Intuition

Four physically distinct mechanisms cover a spectrum from adversarial to
gentle:

| Model | What changes | How much |
|---|---|---|
| Permutation | every strategy's rank, in one or more rows | maximal (worst case) |
| Monotone decay | one `(row, strategy)` cell, toward a floor | targeted, one strategy at a time |
| Partial reorder | only the top-2 strategies in a row | minimal (value gap = original 1st-2nd gap) |
| Gradual rotation | the whole matrix, continuously | total, but announced/predictable in shape |

## Real-world interpretation

- **Permutation** -- a total behavioral change with no warning: e.g. a
  backing model is swapped for a fundamentally different one, and
  every failure type's best remedy changes at once.
- **Monotone decay** -- API degradation, tool degradation, service
  instability: one specific remedy gets steadily less reliable, other
  remedies are untouched.
- **Partial reorder** -- a minor version bump or a small ranking
  adjustment: the top choice and the runner-up trade places, nothing
  else about the environment changes.
- **Gradual rotation** -- a phased migration or progressive rollout:
  the whole environment changes, but smoothly, over a known window.
- **Announced drift** -- a deployment that IS announced ahead of time
  (a changelog, a feature flag, a maintenance window) -- `AnnouncedDrift`
  models a policy that COULD observe this (via `Context.drift_signal`),
  even though none of the policies evaluated in this task's grid
  actually read it (see "Assumptions" below).

## Why permutation is the worst case

A permutation can move any row's optimal strategy an ARBITRARY distance
in value in one event -- from the row's best value straight to its
worst. No other model in this taxonomy can do that in a single event:
monotone decay only ever touches one column at a time and only ever
moves toward a floor (never past it); partial reorder is bounded by
construction to swap only the top two values (the new optimum's value
gap from the old optimum is at most the original best-vs-second-best
gap); gradual rotation reaches an arbitrary target matrix too, but only
by moving through it continuously, giving an adaptive policy the entire
window to track the change rather than facing an instantaneous jump.
This is a claim about the drift MODEL's structural worst-case reach --
see `docs/experiments/drift_validation.md` for why it does not
automatically follow that permutation produces the largest measured
*adaptive-advantage* in every experimental condition (frequency and
settling time matter too, and the validation run's own numbers show
gradual rotation and monotone decay both producing a larger advantage
than permutation once those are accounted for).

## Why static normalization becomes invalid

`recovery.theory.analytic.normalize` computes `(rate - random_rate(P)) /
(oracle_rate(P) - random_rate(P))` for ONE fixed `P`. After a drift
event, both `oracle_rate` and `random_rate` change (they are properties
of whichever matrix is currently active) -- normalizing a post-shift
episode's reward against the PRE-shift matrix's floor/ceiling scores it
against a scale that no longer describes what is achievable. Depending
on the direction of drift this can make an unchanged policy look like it
improved or got worse purely as an artifact of measuring against the
wrong endpoints.
`recovery.metrics.drift_metrics.time_varying_normalize` fixes this by
reconstructing, from a compact `matrix_history`, which matrix was
actually active at each episode, and normalizing that episode against
ITS OWN oracle/random rates -- reducing to the ordinary
`analytic.normalize` in the one-matrix (stationary) case (see
`test_stationary_normalization_identical_to_task5` in
`tests/test_drift.py`).

## Interpretation of recovery metrics

- **`post_shift_window_score`** -- mean time-varying-normalized score in
  a window right after a shift; a snapshot of "how well is this policy
  doing immediately after the environment changed."
- **`recovery_delta`** -- `pre_score - post_score`: how much normalized
  performance DROPPED across the shift. Positive means the shift hurt;
  negative means it helped (e.g. `GradualRotationDrift` toward a
  strictly better target).
- **`adaptation_half_life`** -- episodes from the shift until a rolling
  normalized-score estimate recovers to within `target_fraction` (default
  0.5, the classic "half-life") of the pre-shift baseline, searching
  forward from the post-shift TROUGH (not from episode 0 of the
  post-shift window) so a single lucky/unlucky early episode cannot
  masquerade as "recovered." Returns `float("inf")` if recovery never
  happens within the available data -- never a number that would imply
  recovery occurred when it did not.
- **`instantaneous_drift_regret` / `cumulative_drift_regret` /
  `normalized_drift_regret`** (`recovery.metrics.drift_regret`, Task 8A,
  Correction 2) -- the drift-aware generalization of Task 5's
  `recovery.metrics.regret`: regret at episode `e` is always computed
  against the oracle of the matrix ACTIVE at `e`, reconstructed from
  `matrix_history` exactly like `time_varying_normalize` does, never
  against a single global (and, in Task 8's first implementation,
  incorrectly final) matrix. Reduces exactly to
  `recovery.metrics.regret`'s functions in the one-matrix case.

## Why static-rule recovery statistics are N/A

`adaptation_half_life`/`recovery_delta` measure how a policy's LEARNED
BEHAVIOR responds to an environment change. `StaticRulePolicy` and
`UniformRandomPolicy` never learn -- their action-selection is either a
fixed table lookup or i.i.d. random, identical before and after any
drift event by construction. Computing a "recovery time" for them is a
category error, not merely an uninteresting number: there is no
adaptation process to time. `experiments/drift_taxonomy.py` names these
`NON_ADAPTIVE_METHODS`, and `experiments/drift_analysis.py` reports the
literal text `"N/A"` for their adaptation statistics (in
`drift_recovery_delta.csv`/`drift_adaptation_half_life.csv`) rather than
computing and reporting a number that would misleadingly imply
otherwise. Their normalized-score numbers are still reported in full --
only the two ADAPTATION-specific statistics are withheld.

## Assumptions

- No policy in this task's grid reads `Context.drift_signal` -- it is
  WIRED THROUGH the kernel (`recovery/core/protocols.py`,
  `recovery/core/episode.py`) and available to any future
  drift-aware policy, but every policy evaluated here (Thompson
  Sampling, UCB1, epsilon-greedy, PGTS, static/random baselines)
  ignores it, exactly like `tool_id`/`retry_count`. Whether a policy
  that DID react to an explicit announcement recovers meaningfully
  faster is a natural follow-up this task's infrastructure supports but
  does not itself answer.
- `PerfectClassifier` only -- classifier noise is already characterized
  in Tasks 5/7; this task isolates the drift-model axis.
- `frequency`/`magnitude` are given one explicit, documented meaning per
  drift type (see `experiments/drift_taxonomy.py`'s module docstring) --
  `magnitude` is always "how many rows are affected," `frequency` is
  "episodes between repeated events" for the three event-driven models
  and "interpolation window length" for the one continuous model.

## Limitations

- **The settling-time confound (not fixed -- see "Scientific
  Corrections" below).** Comparing "adaptive advantage" across drift
  models in this taxonomy mixes two effects: how much the drift model
  changes the environment (its severity/reach) and how much
  UNINTERRUPTED time an adaptive policy gets to respond before the next
  change. `frequency` means "episodes between repeated events" for the
  three event-driven models but "ramp length" for the one continuous
  model (`gradual_rotation`), so the same nominal `frequency` value does
  not buy the same amount of settling time across drift types -- this is
  a genuine confound in the taxonomy's current design, not resolved by
  any of Task 8A's corrections. See "Scientific Corrections" for the
  full discussion and why it was not (yet) controlled for.
- Even after Correction 1 (exact gradual-rotation normalization) and
  Correction 2 (time-varying regret), every metric in this taxonomy is
  still computed from ONE realized trajectory per (drift_type,
  frequency, magnitude, method, seed) cell -- statistical uncertainty is
  quantified via bootstrap CIs across SEEDS (never fabricated as
  larger/smaller than the seed count supports).
- The full default grid (20 seeds, all 4 magnitudes) is specified at
  12,800 cells; see `docs/experiments/drift_validation.md` for exactly
  which grid this document's numbers are drawn from and whether the full
  grid completed.

## Computational complexity

`DriftingEnvironment` stores one `DriftEvent` per actual change, never
one per episode, for three of the four models: `PermutationDrift`,
`MonotoneDecayDrift`, and `PartialReorderDrift` are `O(number of
shifts)`, not `O(n_episodes)`, exactly as originally designed.
`GradualRotationDrift` is the ONE deliberate, documented exception
(Task 8A, Correction 1): its history is `O(window)` -- one entry per
episode strictly inside the ramp, needed for EXACT time-varying
normalization/regret during the ramp itself. `window` is a bounded
constructor parameter independent of the surrounding experiment's total
episode count, so this is still far from `O(n_episodes)` for any
realistic ramp length, but it is denser than the other three models'
history. Per-`step()` cost is `O(1)` for every model (a schedule lookup
plus, at most, one `_next_matrix`/closed-form recomputation call -- never
a loop over episodes). See `docs/experiments/drift_validation.md` for
measured throughput per model.

## Reproducibility

```
# Unit/property/stress/regression tests (39 tests, incl. 5 Task 8A corrections tests)
python -m pytest tests/test_drift.py -q

# Smoke grid (<90s)
python experiments/drift_taxonomy.py --smoke

# Full grid (Task 8A, Correction 5): all 4 drift types x 4 frequencies x
# 4 magnitudes x 10 methods x 20 seeds, T=2000 -- 12,800 cells
python experiments/drift_taxonomy.py

# Analysis (tables + figures), against whichever grid was run
python experiments/drift_analysis.py --input results/tables/drift_taxonomy.parquet \
    --store-root results/runs/drift_taxonomy --output-dir results/tables
```

Every RNG source in `recovery/envs/drift.py` is an explicit
`numpy.random.Generator` supplied by the caller (no module-level RNG),
and `experiments/drift_taxonomy.py` derives every cell's environment seed
from `(drift_type, frequency, magnitude, seed)` alone (never from
`method`), so every method compared at a given seed faces the identical
drift schedule and content -- Common Random Numbers, exactly like
`experiments/prior_grid.py`'s `build_environment`.

## Scientific Corrections (Task 8A)

A supervisor review of the original Task 8 implementation identified
four methodological weaknesses that needed correcting before these
results could be used in the paper. This section explains each
correction; concrete before/after numbers are in
`docs/experiments/drift_validation.md`.

### 1. Exact normalization for gradual rotation

**Problem.** `GradualRotationDrift`'s `matrix_history()` originally only
recorded the ramp's start and completion. `time_varying_normalize`
reconstructing "the active matrix at episode e" from that history alone
therefore treated the ENTIRE ramp as governed by the base matrix's
oracle/random rates -- an approximation acceptable for development, not
for a headline result.

**Fix.** `GradualRotationDrift` now records one compact history entry
for every episode strictly inside the ramp (`start < e <= start +
window`) -- see its class docstring and `docs/experiments/
drift_design.md`. Memory is `O(window)`, a bounded constructor
parameter, never `O(n_episodes)`. Live simulation
(`sample_reward`/`current_matrix`) was ALWAYS exact (recomputed from the
closed-form formula every `step()`); only the RETROSPECTIVE
reconstruction used by `time_varying_normalize`/`drift_regret` was
approximate, and is now exact too --
`test_gradual_rotation_history_equals_true_matrix_at_every_episode`
(`tests/test_drift.py`) checks this at EVERY episode of a ramp, not a
handful of probes. The preferred (exact) solution proved fully
achievable, so the quantized-K-steps alternative offered as a fallback
was not needed.

**Effect on results.** In this task's 3,200-cell validation grid,
`gradual_rotation`'s ramp always completes well before the "late stage"
window used to compute `normalized_score`/effect sizes (`start_episode +
window <= 1000` at every grid frequency, vs. the late-stage window
starting at episode 1600 of 2000) -- so the headline adaptive-advantage
finding (deliverable 5) is numerically UNCHANGED by this correction: the
old (approximate) and new (exact) `drift_effect_sizes.parquet` are
bit-identical for every row (max abs difference 0.0). **What DID change
materially**: `recovery_delta`/`adaptation_half_life`, which anchor
their "shift_episode" reference to `matrix_history()`'s first
post-episode-0 entry. Under the OLD sparse history that was the ramp's
COMPLETION episode; under the NEW dense history it is the ramp's very
FIRST episode (`start + 1`) -- a more scientifically meaningful
reference point (recovery is now measured from when drift BEGINS, not
from when it stops), and it materially changed the measured half-life
(e.g. `stock_thompson` under `gradual_rotation`: 321 episodes old vs.
446 episodes new -- see `drift_validation.md` for the full table). This
difference is reported explicitly, not hidden, per the correction's own
instruction. The subsequent full 12,800-cell grid (Correction 5, run
entirely under the corrected code) reproduces the same ranking and the
same qualitative conclusions as this 3,200-cell paired comparison,
which is independent evidence that the scoped grid's conclusions were
not an artifact of its smaller magnitude/seed subset.

### 2. Time-varying regret

**Problem.** Every regret figure Task 8 originally reported
(`final_cumulative_regret`, via Task 5's `summarize()`) was computed
against the FINAL matrix of a drifted run -- mathematically incorrect
for any episode played under an earlier regime.

**Fix.** `recovery/metrics/drift_regret.py` adds
`instantaneous_drift_regret`/`cumulative_drift_regret`/
`normalized_drift_regret`, each reconstructing "the matrix active at
episode e" from `matrix_history` via the SAME `matrix_segments`
primitive `time_varying_normalize` uses (never a second, independent
implementation of "which matrix was active when").
`experiments/drift_analysis.py` now stores the corrected
`final_cumulative_regret_tv`/`normalized_drift_regret_tv` in every
cell's output, alongside the old final-matrix value under an explicit
`_final_matrix_only` name (kept for comparison, used by no deliverable).
No deliverable in Task 8's original 5-item analysis actually READ the
regret column (all five use `late_stage_normalized_tv`), so this
correction has no effect on the headline findings -- but the STORED
per-cell regret numbers themselves were wrong before and are correct
now, which matters for any future analysis that does use them.

### 3. The settling-time confound

**Problem.** Comparing "adaptive advantage" across drift models
conflates two effects: how severely a drift model changes the
environment, and how much uninterrupted time an adaptive policy gets to
respond before the next change. `gradual_rotation` structurally gives
adaptive policies the LONGEST uninterrupted post-event stationary
stretch of any model in this taxonomy (once the ramp completes, nothing
changes again); `permutation` at high frequency gives the LEAST (a
reshuffle every 50 episodes leaves no time to fully converge). This
taxonomy's own `frequency` parameter means different things for
different drift types (see "Mathematical definitions"), which makes the
confound worse, not better -- it is not simply "the same knob turned to
different values."

**Decision: not controlled for.** A genuinely controlled comparison
would need a common, drift-type-independent definition of "cumulative
variation" (e.g. total expected reward lost, relative to a fixed
reference policy, integrated over the run) and then calibrated
frequency/magnitude/window values PER drift type to equalize it before
any cross-model comparison would be apples-to-apples. Building and
validating that calibration is a substantial, separate piece of
experiment design -- estimating it honestly, it is comparable in scope
to designing a new drift model, which Task 8A's own instructions say not
to do. Per the task's explicit instruction ("if this requires
substantial redesign, do not implement it"), this was NOT implemented.

**How this affects interpretation.** Every "does the adaptive advantage
survive" claim in `docs/experiments/drift_validation.md` should be read
as "survives under THIS taxonomy's specific frequency/magnitude
parameterization," not as a universal statement about drift severity.
The observed ranking (`gradual_rotation > monotone_decay ~ permutation >
partial_reorder`) is very plausibly explained, at least in part, by
`gradual_rotation` receiving the most settling time and high-frequency
`permutation` cells receiving the least -- the per-frequency breakdowns
in `drift_validation.md` support this reading directly (every drift
model's advantage over `static_rule` widens as its own settling time
increases).

**Future work.** A principled fix would (a) define cumulative variation
independent of drift mechanism, (b) calibrate each model's
frequency/magnitude/window to match a target variation level, and (c)
re-run the comparison at matched variation -- isolating "how gentle is
this drift mechanism, holding total change constant" from "how much time
did the policy get to respond." A cheaper partial mitigation, also not
implemented here, would be to report advantage as an explicit function
of measured post-event settling episodes (directly, per cell) rather
than of the raw `frequency` parameter, sidestepping the need to define a
cross-model variation metric at all.

### 4. Stationary compatibility proof

**Problem.** Task 8's original "proof" that `Context` gaining a
`drift_signal` field did not change stationary kernel behavior relied on
the existing test suite's tight statistical tolerances (200,000-episode
tests passing unchanged) -- suggestive, but not a direct digest/array
comparison.

**Fix.** `tests/test_drift.py::test_stationary_digest_equality_before_and_after_task8`
runs the SAME stationary configuration through a frozen, byte-for-byte
reference copy of the pre-Task-8 kernel loop and today's kernel, and
asserts every `EpisodeLog` array is element-wise identical AND that
SHA256 digests of their concatenated bytes match exactly. See
`docs/experiments/drift_validation.md` for the reported digest values
and a larger (5,000/50,000-episode) standalone run.
