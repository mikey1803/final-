# Drift Taxonomy: Validation Results

Concrete numbers from this task's own test runs and validation grids.
See `docs/experiments/drift.md` for the mathematical/scientific
write-up (including the "Scientific Corrections" section this document
supports with numbers) and `docs/experiments/drift_design.md` for
architecture rationale.

**This document covers both Task 8 (original taxonomy) and Task 8A
(scientific corrections after supervisor review).** Numbers explicitly
marked "Task 8A" are new; everything else is retained from Task 8 for
continuity, with corrections layered on top and old-vs-new differences
called out explicitly wherever a correction changed a number.

## Test results

```
python -m pytest tests/test_drift.py -q
-> 39 passed
```

34 from Task 8 (matrix shape preserved, values always in `[0,1]`, base
matrix never mutates, gradual-interpolation endpoints,
partial-reorder-changes-argmax-only, monotone-decay
monotonic/floor-respecting, announcement-only-on-shift-episodes,
same-seed-same-history, stationary-normalization-identical-to-Task-5,
half-life-returns-infinity, permutation-reproduces-original-setup, the
100+ random-matrix stress test, matrix-replay-reconstructs-every-drift,
no-NaN/Inf-ever, deterministic-for-identical-seeds, `--smoke` end-to-end,
and the `static_rule`/`uniform_random` "N/A" reporting behavior) **plus
5 new Task 8A regression tests**:

1. `test_gradual_rotation_history_equals_true_matrix_at_every_episode` --
   Correction 1: exact reconstruction at EVERY episode of a ramp, not
   just start/end probes.
2. `test_drift_regret_equals_ordinary_regret_when_stationary` --
   Correction 2: `drift_regret` functions are bit-identical to
   `recovery.metrics.regret`'s in the one-matrix case.
3. `test_changing_future_matrix_does_not_alter_past_regret` -- Correction
   2: regret before a shift is unaffected by what that shift's target
   matrix turns out to be.
4. `test_stationary_digest_equality_before_and_after_task8` -- Correction
   4 (see below).
5. `test_matrix_history_replay_reconstructs_gradual_rotation_exactly` --
   companion to Task 8's `test_matrix_history_replay_reconstructs_every_drift_exactly`
   (which covered only the event-driven models), now covering the dense
   `GradualRotationDrift` case too.

All 34 Task 8 tests continue to pass unchanged (Task 8A's required test
6, "old Task 8 tests must continue passing") -- one,
`test_edge_case_window_longer_than_run`, needed its assertion UPDATED
(not its intent) because Correction 1 deliberately changed
`GradualRotationDrift`'s history density; this is called out explicitly
in the test itself and in `drift_design.md`, not a silently broken test.

## Full backend regression suite

```
python -m pytest -q
-> 183 passed, 1 failed, 1 warning in 643.14s
```

The 1 failure is `test_stationary_experiments.py::test_smoke_mode_end_to_end[classifier_noise_sweep.py-...]`
-- the same well-documented smoke-timing flake seen in Tasks 4A, 6, 7,
and 8 (unrelated to any file this task touches), this time measuring
12.89s against a 10.0s budget. This run happened to execute CONCURRENTLY
with the Correction 5 full 12,800-cell grid run in the background (both
kicked off in the same session to save wall-clock time), which is
genuine additional CPU contention beyond the environment's normal
jitter. Re-run in ISOLATION after the grid finished:

```
python -m pytest tests/test_stationary_experiments.py::test_smoke_mode_end_to_end -q
-> 3 passed in 18.81s
```

183 + 1 (confirmed non-regression) = 184 = 179 (Task 8 baseline: 145
pre-existing + 34 Task 8 tests) + 5 new Task 8A regression tests. Zero
genuine failures.

## Stationary digest proof (Task 8A, Correction 4)

Ran the SAME stationary configuration (`MatrixEnvironment(MATRIX)`,
`ThompsonPolicy`, `PerfectClassifier`) through a frozen, byte-for-byte
reference copy of the PRE-Task-8 kernel loop (identical to today's
`recovery.core.episode.run_episodes` except for the omitted
`drift_signal=getattr(...)` kwarg in `Context` construction) and
TODAY's actual kernel, at two scales:

| n_episodes | seed | digest before | digest after | identical | element-wise array equality |
|---|---|---|---|---|---|
| 5,000 | 77 | `dff143fe18d2f132a9efdc1563ef4b511cf69e631a9a9e41cdf5bba75ed9d32f` | `dff143fe18d2f132a9efdc1563ef4b511cf69e631a9a9e41cdf5bba75ed9d32f` | **True** | **True** |
| 50,000 | 4242 | `602506225d31bb8979fa6af636ff8af7fd6bf3494500ecd92ae9d84f258aa6f9` | `602506225d31bb8979fa6af636ff8af7fd6bf3494500ecd92ae9d84f258aa6f9` | **True** | **True** |

Digest = SHA256 of the concatenated bytes of all five `EpisodeLog`
arrays (`episode`, `true_type`, `predicted_type`, `action`, `reward`),
in that order. Element-wise array equality was checked independently
via `numpy.array_equal` on each of the five arrays (not just via the
digest) -- both checks agree at both scales. This is now also a
permanent regression test
(`tests/test_drift.py::test_stationary_digest_equality_before_and_after_task8`,
run at 5,000 episodes/seed=77 for fast CI turnaround; the 50,000-episode
run above was a one-off, larger-scale confirmation, not added to the
test suite as a matter of runtime cost, not because it's less
trustworthy).

This supersedes Task 8's original "digest proof" (which relied on
existing tight-tolerance statistical tests passing unchanged, a
suggestive but indirect argument) with the direct comparison Correction
4 required.

## Performance

### Stationary-path overhead (the <5% target)

Unaffected by any Task 8A correction (none of the five corrections touch
the stationary path) -- retained from Task 8:

```
pre-Task-8 (no getattr)        best=5.1055s   (100,000 episodes, best-of-5)
post-Task-8 (current)          best=5.2949s
Slowdown: +3.71%
```

Within the <5% target.

### Per-drift-model throughput (Task 8A: re-measured after Correction 1)

`GradualRotationDrift` now does more work per `step()` during a ramp
(one `DriftEvent` append + one matrix copy per ramp episode, vs. only at
two sparse checkpoints before Correction 1) -- re-benchmarked (50,000
episodes, `ThompsonPolicy`, best-of-6, run in isolation after both
background jobs completed):

| Environment | episodes/sec | vs. `MatrixEnvironment` |
|---|---:|---:|
| `MatrixEnvironment` (baseline) | 21,799 | -- |
| `PermutationDrift` | 21,319 | +2.25% |
| `MonotoneDecayDrift` | 21,668 | +0.61% |
| `PartialReorderDrift` | 21,646 | +0.71% |
| `GradualRotationDrift` | 12,441 | +75.22% |
| `AnnouncedDrift(PermutationDrift)` | 20,922 | +4.20% vs. PermutationDrift alone |

`GradualRotationDrift`'s slowdown increased modestly from Task 8's
original +67.77% to +75.22% -- the direct, expected cost of Correction
1's denser per-ramp-episode history logging. Still `O(1)` per `step()`
call (one array copy + one list append, not a loop), just with a larger
constant factor than the sparse-history version. The other three
event-driven models are within measurement noise of Task 8's original
numbers (all comfortably under a few percent).

## Old vs. new: Correction 1 (exact gradual-rotation normalization)

Comparison performed on the IDENTICAL 3,200-cell simulation data (same
stored per-episode logs, same RNG-derived trajectories -- Correction 1
changes only how `matrix_history()` is populated, never any RNG draw, so
re-running `experiments/drift_analysis.py` against the SAME
`results/runs/drift_taxonomy` store with the corrected `drift.py`
isolates exactly Correction 1's effect).

**Effect sizes (`drift_effect_sizes.parquet`): bit-identical, old vs.
new, max abs difference = 0.0, every row.** This grid's `gradual_rotation`
cells always complete their ramp (`start_episode + window <= 1000`) well
before the "late stage" window used for `normalized_score`/effect sizes
(episodes 1600-2000 of 2000) -- so the approximation Correction 1 fixed
never actually reached the episodes these deliverables read from, in
this specific grid's parameter range. **The headline "does the adaptive
advantage survive" finding (deliverable 5) is therefore UNCHANGED by
Correction 1** in this validation grid.

**Recovery statistics DID change materially.** `recovery_delta`/
`adaptation_half_life` anchor their `shift_episode` reference to
`matrix_history()`'s first post-episode-0 entry -- under the OLD sparse
history that was the ramp's COMPLETION episode; under the NEW dense
history it is the ramp's very FIRST episode. This is a more
scientifically meaningful reference point (recovery measured from when
drift begins, not when it stops), and it materially increased measured
half-life for `gradual_rotation`:

| method | old mean half-life (episodes) | new mean half-life (episodes) | change |
|---|---:|---:|---:|
| `stock_thompson` | 321.4 | 446.0 | +38.8% |
| `pgts_s5` | 250.4 | 410.4 | +63.9% |
| `pgts_s20` | 287.0 | 445.3 | +55.1% |
| `pgts_s50` | 352.4 | 462.6 | +31.3% |
| `discounted_thompson` | 331.1 | 449.3 | +35.7% |
| `discounted_pgts` | 329.7 | 436.6 | +32.4% |
| `epsilon_greedy` | 410.7 | 544.6 | +32.6% |
| `ucb1` | 564.8 | 471.2 | -16.6% |

(`static_rule`/`uniform_random` remain `"N/A"` in both.) This is reported
in full per the correction's explicit instruction ("if the reported
adaptive advantage changes materially, state that explicitly") -- note
the finding it actually changes is the RECOVERY-TIME statistic, not the
adaptive-advantage statistic, and the change is explained by a genuine,
intentional shift in what `shift_episode` means for this one drift
model, not by numerical noise.

## Regret validation (Task 8A, Correction 2)

`recovery.metrics.drift_regret` passes both required regression tests
(`test_drift_regret_equals_ordinary_regret_when_stationary`,
`test_changing_future_matrix_does_not_alter_past_regret`) plus every
manual check performed during implementation: bit-identical to
`recovery.metrics.regret` when `matrix_history` has one entry; past
regret provably unaffected by swapping out a future segment's matrix
(verified on a real drifted run, not just synthetic data).

**Old (final-matrix) vs. new (time-varying) regret, mean over every
non-static-rule method, 3,200-cell grid:**

| drift_type | old mean final regret | new mean final regret (correct) | change |
|---|---:|---:|---:|
| `gradual_rotation` | 358.9 | 292.3 | -18.5% |
| `permutation` | 488.9 | 425.9 | -12.9% |
| `monotone_decay` | 268.6 | 249.5 | -7.1% |
| `partial_reorder` | 328.3 | 307.7 | -6.3% |

The OLD (final-matrix) figure systematically OVER-stated regret in every
drift type: episodes played under an EARLIER matrix regime were being
scored against the final matrix's oracle, which is a different (and, for
these grid conditions, generally harder-to-match) ceiling than the
matrix that was actually active when those earlier episodes were played.
**No deliverable in Task 8's original 5-item analysis actually read the
regret column** (all five use `late_stage_normalized_tv`), so this
correction changes the STORED per-cell numbers (now correct) without
changing any of Task 8's headline conclusions -- but it was still
necessary: those stored numbers are scientifically wrong before this fix
and are used nowhere else in this codebase yet, which does not make them
harmless to leave wrong.

## Matrix-history / normalization validation

- `test_gradual_rotation_history_equals_true_matrix_at_every_episode` --
  every episode of a 120-episode window (not just start/end) reconstructs
  exactly.
- `test_matrix_history_replay_reconstructs_gradual_rotation_exactly` --
  same property, independent test, different window/parameters.
- `test_time_varying_normalize_uses_active_matrix_not_preshift_endpoints`
  (Task 8) -- still passes; unaffected by Task 8A (this test already used
  a single-event, non-gradual history).
- `test_stationary_normalization_identical_to_task5` (Task 8) -- still
  passes; `time_varying_normalize` continues to reduce exactly to
  `analytic.normalize` in the one-segment case.

## Settling-time confound (Task 8A, Correction 3)

Not implemented as a controlled experiment -- see `drift.md`'s
"Scientific Corrections" section, item 3, for the full reasoning (a
principled cross-drift-model "cumulative variation" calibration was
judged substantial enough in scope to fall under the task's explicit
"if this requires substantial redesign, do not implement it" instruction).
The limitation is documented in both `drift.md`'s new "Scientific
Corrections" section and its "Limitations" section, with a concrete
description of how it affects interpretation of every finding below and
what a principled fix would require.

## Full grid execution (Task 8A, Correction 5)

Task 8's original validation used a scoped 3,200-cell grid (2 of 4
magnitudes, 10 of 20 seeds). Correction 5 required the complete
specified grid: 4 drift types x 4 frequencies x 4 magnitudes x 10
methods x 20 seeds, `T=2000` = **12,800 cells**.

```
python experiments/drift_taxonomy.py
-> 12800 cells (9,600 newly simulated -- the 3,200-cell scoped grid's
   cells were already in the RunStore and skipped, per its normal
   skip-if-already-run behavior), wrote in 686.90s (~11.5 min)

python experiments/drift_analysis.py --input results/tables/drift_taxonomy.parquet \
    --store-root results/runs/drift_taxonomy --output-dir results/tables
-> per-cell metric computation for all 12,800 cells + 6 analysis tables
   (incl. the new drift_summary.parquet) + 4 figures in 1240.4s (~20.7 min)
```

**Estimated vs. actual runtime.** Extrapolating from the 3,200-cell
scoped run (115s grid + ~330s analysis average), a naive 4x scale-up
predicted roughly 460s grid + ~1,320s analysis. Actual: 687s grid (the
extra time reflects `GradualRotationDrift`'s higher per-cell cost after
Correction 1, which the 3,200-cell estimate under-weighted) and 1,240s
analysis, both in the same order of magnitude as the estimate. **100% of
the specified grid completed** (12,800/12,800 cells; 320 seeds-worth of
data per (drift_type, method) pair, i.e. exactly 4 frequencies x 4
magnitudes x 20 seeds) -- no partial grid was published.

### Updated findings (deliverables 1-5), full 12,800-cell grid

The full grid CONFIRMS the same qualitative ranking the 3,200-cell scoped
grid found, now with every magnitude and all 20 seeds:

```
Observed ranking by adaptive advantage (best adaptive method vs. static_rule), highest first:
gradual_rotation (0.3842) > monotone_decay (0.1306) > permutation (0.0686) > partial_reorder (-0.0055)

- gradual_rotation: best adaptive method = pgts_s20 (mean normalized 0.8663 vs.
  static_rule 0.4821) -> advantage = 0.3842 (SURVIVES, 560.5% of permutation's advantage)
- monotone_decay: best adaptive method = stock_thompson (mean normalized 0.9117 vs.
  static_rule 0.7810) -> advantage = 0.1306 (SURVIVES, 190.5% of permutation's advantage)
- partial_reorder: best adaptive method = pgts_s5 (mean normalized 0.7777 vs.
  static_rule 0.7831) -> advantage = -0.0055 (DOES NOT SURVIVE, -8.0% of permutation's advantage)
- permutation: best adaptive method = stock_thompson (mean normalized 0.5368 vs.
  static_rule 0.4682) -> advantage = 0.0686 (SURVIVES, reference: permutation drift itself)
```

The full interpretation report (mechanism notes + per-frequency trends
for every drift type) is in `results/tables/drift_interpretation_report.txt`.

**1. Normalized score per drift model**
(`results/tables/drift_normalized_score.parquet`, `results/figures/drift_taxonomy.png`,
`n_cells=320` per row -- every (drift_type, method) pair now backed by
the full 4x4x20 sweep). `uniform_random` scores essentially at the random
floor (~-0.01) in every drift type, as expected; `static_rule` ranges
from 0.468 (`permutation`) to 0.783 (`partial_reorder`), directly
reflecting how much of the matrix each model actually disturbs.

**2/3. Recovery delta and adaptation half-life**
(`results/tables/drift_recovery_delta.csv`, `results/tables/drift_adaptation_half_life.csv`,
`results/figures/recovery_half_life.png`). Mean finite half-life for
`stock_thompson`: 462 (`gradual_rotation`), 437 (`monotone_decay`), 434
(`partial_reorder`), 608 (`permutation`) -- `permutation` remains the
slowest to recover on average, consistent with its high-frequency cells
denying settled time. `fraction_never_recovered` stays below 2.2% for
`stock_thompson`/`pgts_s20` across every drift type in the full grid.
`static_rule`/`uniform_random` report `"N/A"` throughout, confirmed by
`test_static_policies_report_na_for_adaptation_statistics`.

**4. Effect sizes**
(`results/tables/drift_effect_sizes.parquet`, `results/figures/effect_sizes.png`,
`n_sample=n_baseline=320` per row). `gradual_rotation`: LARGE positive
Cliff's delta for `stock_thompson` (0.734) and `pgts_s20` (0.747).
`monotone_decay`: MEDIUM positive for the Thompson family (0.45-0.46),
MEDIUM NEGATIVE for `ucb1` (-0.43). `partial_reorder`: negligible for
the Thompson/PGTS family, LARGE NEGATIVE for `ucb1` (-0.67) --
`static_rule` clearly outperforms `ucb1` there. `permutation`: negligible
for every method except `uniform_random` (large negative, as expected).

**5. Does the adaptive advantage survive under realistic drift?** See
the interpretation report above -- unchanged in direction and magnitude
order from the 3,200-cell scoped grid (a good sign that the scoped
grid's earlier conclusions were not an artifact of the smaller
magnitude/seed subset).

## Scope notes

- No policy in the grid reads `Context.drift_signal` (see `drift.md`,
  "Assumptions") -- whether an explicitly drift-aware policy recovers
  faster than one that infers drift purely from reward statistics is
  future work this task's kernel wiring enables but does not itself
  measure.
- The settling-time confound (Correction 3) means every finding in this
  document is scoped to this taxonomy's specific frequency/magnitude
  parameterization, not a universal statement about drift severity --
  see `drift.md` for the full discussion.
