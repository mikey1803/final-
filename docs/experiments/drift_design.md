# Drift Taxonomy: Architecture Design Notes

Companion to `docs/experiments/drift.md` (the mathematical/scientific
write-up) -- this document explains the ENGINEERING decisions behind
`recovery/envs/drift.py`, `recovery/metrics/drift_metrics.py`, and the
`experiments/drift_taxonomy.py`/`experiments/drift_analysis.py` split,
and why each was made.

## Why a `DriftingEnvironment` base class, not five independent classes

Every event-driven drift model (`PermutationDrift`, `MonotoneDecayDrift`,
`PartialReorderDrift`) shares identical bookkeeping: never mutate the
original matrix, allocate a fresh matrix per event, keep a compact
history, satisfy the `Environment` protocol. Factoring this into
`DriftingEnvironment._advance`/`_next_matrix` means each concrete model
implements ONLY its own mathematical transform (`_next_matrix`), never
the scheduling/history/protocol plumbing -- and a bug fix or
optimization to that plumbing (e.g. the `_validate_matrix` call inside
`_advance`) automatically applies to all three without touching their
files.

`GradualRotationDrift` does NOT fit the "apply a transform at scheduled
episodes" shape (its drift is continuous, not event-driven), so it
overrides `_advance` directly instead of implementing `_next_matrix` --
inheritance is used where the shared shape genuinely fits, not forced
onto a model that does not.

## Why `AnnouncedDrift` is composition, not another subclass

`AnnouncedDrift` needs to wrap ANY of the four models (or, in principle,
any future drift model satisfying the same duck-typed shape) without
knowing its internals -- a decorator over an opaque object, not a
subclass that would need to know each wrapped class's specific
`_next_matrix`/`_advance` override. This also keeps the "no other
behavior changes" guarantee mechanical: every method except `step()`
(and the two new attributes) is a one-line delegation to `self._wrapped`,
so there is no way for `AnnouncedDrift` to accidentally change
`sample_reward`'s behavior -- there is no reward-related code in the
class at all.

## Why `drift_signal` (kernel) and `shift_signal` (module) are two names for one attribute

The task specification names the kernel-facing wiring precisely
(`getattr(env, "drift_signal", False)` in
`recovery/core/episode.py`) and separately describes `AnnouncedDrift`'s
job as exposing `shift_signal`. Rather than picking one name and
silently ignoring the other half of the specification, `AnnouncedDrift`
stores the flag once (`self.drift_signal`) and exposes `shift_signal` as
a read-only property returning it -- both names are always in agreement
by construction (a property can only read the same underlying value),
so there is no risk of the two drifting apart.

## The episode-0 announcement problem, and how it was solved without changing the kernel

The kernel builds `Context` -- and therefore reads `env.drift_signal` --
BEFORE calling `env.step()` for episode 0 (see
`recovery/core/episode.py`'s eight-step loop). A schedule entry at
episode 0 is applied inside the wrapped model's OWN `__init__` (per
`DriftingEnvironment`'s "apply anything due at or before episode 0"
rule), before `AnnouncedDrift` ever sees it -- so naively initializing
`drift_signal = False` would make an episode-0 shift permanently
unannounceable to episode 0's own `Context`, since no `step()` call
would ever occur early enough to flip it.

The fix does not touch the kernel (which the task does not permit
modifying beyond the one specified line): `AnnouncedDrift.__init__`
inspects the wrapped model's `matrix_history()` immediately -- if it
already has more than its initial entry (meaning a schedule entry at
episode <= 0 was already applied during construction), `drift_signal` is
set to `True` right there, before any `step()` call. This is the only
point in the object's lifetime where that information is still
available, and it is exactly what makes
`test_edge_case_shift_at_episode_zero_is_announced_immediately` in
`tests/test_drift.py` pass without any kernel change.

## `GradualRotationDrift`'s history: originally an approximation, now exact (Task 8A, Correction 1)

The task's general complexity constraint ("matrix history complexity
O(number of shifts), not O(episodes)") was originally in direct tension
with a model whose matrix changes on every episode of its window. Task
8's first resolution: keep SIMULATION exact (the closed-form
`_matrix_at(episode)` recomputed fresh every `step()` call) while
keeping the retrospective `matrix_history()` compact (two entries: ramp
start, ramp complete) -- paying the cost entirely at ANALYSIS time
(`time_varying_normalize` treating the whole ramp as base-matrix-governed
until the target was reached).

A supervisor review (Task 8A) flagged this as unacceptable for a
headline result: an approximation during exactly the episodes where the
matrix is most actively changing is the worst place to have one. The
fix: log ONE compact history entry per episode strictly inside the ramp
(`start < e <= start + window`) -- `window` is a bounded constructor
parameter, so this is `O(window)`, a deliberate, DOCUMENTED exception to
the general `O(shifts)` rule (not a silent violation of it), and still
nowhere near `O(n_episodes)` for any experiment whose horizon is many
times its ramp length (true of every cell in this task's grid: horizons
of 2,000 against ramps of at most 500). The alternative considered and
rejected in Task 8's first pass -- logging every `k`-th episode as a
checkpoint -- is no longer necessary: the exact, dense-during-the-ramp
design is both correct and cheap enough (see
`docs/experiments/drift_validation.md`'s per-drift-model throughput
table) that there was no need to settle for an intermediate
quantization.

One side effect of this fix, also documented rather than hidden:
`AnnouncedDrift` wrapping a `GradualRotationDrift` now reports
`drift_signal=True` on EVERY episode inside the ramp (since every one of
those episodes genuinely adds a new history entry), not just at the
ramp's start and end. This is arguably the mathematically correct
reading of "did the matrix change this episode" for a continuously
drifting environment -- a continuous ramp IS changing every episode --
rather than an artifact of the fix.

## Why `recovery/metrics/drift_metrics.py`, not `recovery/envs/drift.py`

`recovery/envs/drift.py` may import only `numpy` and the `Environment`
protocol (an explicit architecture constraint) -- it cannot import
`recovery.theory.analytic`, so it cannot itself compute oracle/random
rates. `recovery/metrics/drift_metrics.py` lives in the metrics package
specifically so it CAN reuse `oracle_rate`/`random_rate` from
`recovery.theory.analytic` (never reimplementing them) and
`bootstrap_ci`/`paired_bootstrap_diff` from `recovery.metrics.bootstrap`
at the analysis layer -- consistent with the metrics package's existing
rule ("may import only numpy, pandas, and recovery.theory.analytic").
`matrix_history` is consumed structurally there (any sequence of
`(episode, matrix)` pairs works), so `drift_metrics.py` still never
imports `recovery.envs` -- it does not need to know `DriftEvent` exists.

## Why `drift_taxonomy.py` stores full logs and computes nothing, and `drift_analysis.py` computes everything

Task 7's `experiments/prior_grid.py` split its grid into a summary-only
main path (light storage, large grid) and a full-log path only for the
sub-grid that specifically needed per-episode data
(`build_discounted_grid`/`run_discounted_grid`). Task 8's fixed `T=2000`
horizon (much smaller than Task 7's up-to-50,000) makes full-log storage
affordable for the WHOLE grid, so `drift_taxonomy.py` always stores full
logs via the unmodified `recovery.core.parallel.RunStore`, and
`results/tables/drift_taxonomy.parquet` is a light metadata/index table
-- `drift_analysis.py` reads logs back and computes Task 5's
`summarize()` plus every `drift_metrics` function EXACTLY ONCE, for
every cell, in one place. This also means a cell's `matrix_history` is
never persisted redundantly: `drift_analysis.py` rebuilds it by
replaying the SAME deterministic `build_environment(...)` factory
`drift_taxonomy.py` used (an `O(n_episodes)` loop of cheap `step()`
calls, no reward sampling), reusing `RunConfig`'s established
"factories are dummy for content-addressing purposes" reconstruction
trick from `experiments/prior_grid.py::recovery_half_life`.

## Why some analysis tables are CSV, not parquet

`recovery_delta_with_ci`/`adaptation_half_life_summary` deliberately mix
floats with the literal string `"N/A"` in the same column (for
`NON_ADAPTIVE_METHODS`) -- Arrow's columnar parquet writer rejects a
column mixing `float` and `str` (`pyarrow.lib.ArrowTypeError`). Rather
than coercing `"N/A"` into `NaN` (indistinguishable, on read, from "a
number that happened to be missing" -- exactly what the task's "never
fabricate values" instruction warns against conflating) or storing every
numeric value as a string (losing the column's numeric type for every
adaptive-method row), these two small, human/report-facing SUMMARY
tables are written as CSV, where mixed content is unremarkable. The
LARGE per-cell table (`drift_per_cell_metrics.parquet`) stays fully
numeric (`NaN`, never a string) plus a separate `is_adaptive` boolean
column -- the systematic data table a downstream consumer would load
into a numeric analysis never has a stray string in a float column to
trip over. `results/tables/drift_summary.parquet` (Task 8A, Correction
5's consolidated headline table) follows the SAME numeric-plus-flag
convention as `drift_per_cell_metrics.parquet`, not the CSV convention
-- it is explicitly specified as a `.parquet` deliverable, so
`build_summary_table` uses `NaN` + `is_adaptive` rather than the literal
string `"N/A"`.
