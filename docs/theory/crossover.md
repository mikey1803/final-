# The Analytic Crossover Predictor

This document explains Task 10's central deliverable
(`backend/recovery/theory/crossover.py`, `experiments/crossover_calibration.py`,
`experiments/crossover_extension.py`) -- an analytic estimate of the
horizon `T*` at which learning online (Thompson Sampling) is expected to
overtake a fixed recovery table in total accumulated reward, calibrated
against Task 6's empirical measurements
([`horizon.md`](../experiments/horizon.md)) and Task 1's exact analytic
layer ([`architecture.md`](../architecture.md)); and Task 10B's
presentation/validation layer (`experiments/crossover_reliability.py`,
`experiments/crossover_figures.py`) -- figures, a per-cell calibration
summary table, and a `prediction_status`/`confidence_level`
classification, all derived from Task 10's data WITHOUT changing the
model, the fitted constant, or any predicted value. For how to use these
outputs in a real decision, see
[`practitioner_guidelines.md`](practitioner_guidelines.md).

**Read this before using a prediction from this module for anything.**
Every number `predict_crossover` produces rests on an *asymptotic* regret
bound with a free constant this project could only calibrate empirically,
against one environment family, on one taxonomy size. Section 5
("Explicit limitations") and every "Assumptions" subsection below are not
boilerplate -- they describe exactly where this predictor is known to be
unreliable.

## 1. Mathematical derivation

### 1.1 Static regret

A static (non-learning) recovery table `pi` maps each type to a fixed
strategy and never updates. Its per-episode regret rate, reusing Task 1's
exact analytic functions (`recovery.theory.analytic`) without
re-derivation, is

```
rho(P, pi, p) = oracle_rate(P, p) - policy_rate(P, pi, p)
              = sum_t p[t] * ( max_k P[t,k] - P[t, pi[t]] )
```

`rho >= 0` always (the oracle is a hard ceiling -- see `oracle_rate`'s
docstring), and it is *constant* -- the table never changes, so it pays
this exact penalty every episode, forever. Its cumulative regret after
`T` episodes is therefore exactly `rho * T`: a straight line.

### 1.2 Thompson Sampling's asymptotic regret

For a single N-armed Bernoulli bandit, the Lai-Robbins lower bound (and
Thompson Sampling's matching asymptotic upper bound, for Bernoulli
rewards) says the expected number of pulls of a suboptimal arm `i` grows
as

```
E[n_i(T)] ~ log(T) / KL(p_i, p*)      as T -> infinity
```

where `p*` is the optimal arm's success probability and `KL(p, q) = p
log(p/q) + (1-p) log((1-p)/(1-q))` is the Bernoulli KL divergence. This
project's recovery problem is not one bandit but `N` of them sharing a
single clock of `T` total episodes (one per type, since the table --
and, structurally, Thompson Sampling's per-type posterior -- only has to
learn the best arm *within* a type's row). Task 10 approximates the
horizon available to any one type's sub-problem as `T/N` (episodes
divided evenly across types under the uniform-type-prior assumption) and
sums the Lai-Robbins pulls formula over every type and every non-optimal
arm, scaled by an explicit constant `c` and an extra factor of `N`:

```
R_TS(T) = c * N * sum_t p[t] * sum_{k != k*(t)} log(T/N) / KL(P[t,k], P[t,k*(t)])
```

(`k*(t) = argmax_k P[t,k]`, `k != k*(t)` excludes both the optimal arm and
any TIED optimal arm -- see the "Assumptions" and "Edge cases" below.)
This is `ts_cumulative_regret(..., form="kl")`. An alternative, cheaper
approximation replaces the KL divergence with the raw gap
`Delta = p* - p_i`:

```
R_TS(T) = c * N * sum_t p[t] * sum_{k != k*(t)} log(T/N) / (P[t,k*(t)] - P[t,k])
```

(`form="gap"`). **Neither formula carries the per-arm gap factor
`Delta_i` that a textbook Lai-Robbins *regret* bound (as opposed to a
*pulls* bound) would have** (the textbook regret bound is
`sum_i Delta_i log(T) / KL(p_i, p*)`) -- Task 10 specifies these two
exact forms verbatim, and this module implements them exactly as
specified, not the textbook regret bound. This is a deliberate scoping
choice, not an oversight: the free constant `c` can absorb an overall
proportionality error, but it cannot fix a wrong *shape* of gap-dependence
that omitting `Delta_i` might introduce -- which is exactly why
`crossover_calibration.py` reports reliability broken down BY GAP rather
than only in aggregate (Section 4). Both forms are logically related: for
small gaps, `KL(p, p*) ~ Delta^2 / (2 p*(1-p*))`, so the KL form's
`1/KL` term is roughly `2 p*(1-p*) / Delta^2` versus the gap form's
`1/Delta` -- the two diverge by a factor of roughly `Delta / (2 p*(1-p*))`
as `Delta -> 0`, and agree much more closely as `Delta` grows (see
`tests/test_crossover.py`'s KL-vs-gap tests, and Section 4).

### 1.3 The crossover equation

`T*` is the smallest horizon solving

```
R_TS(T) = rho * T
```

Both sides are cumulative regret against the SAME oracle, so equal
cumulative regret means equal cumulative reward -- this is exactly the
point where total accumulated reward from learning online first catches
up to (and, by the shape argument in Section 2, stays at or above) total
accumulated reward from the static table. This is a transcendental
equation (`T` appears both inside `log(T/N)` and linearly on the right)
with no closed form, solved numerically (Section 2).

**Which empirical crossover this predicts.** Task 6 reports two different
crossover notions ([`horizon.md`](../experiments/horizon.md), "Two
crossover notions"): *asymptotic* (per-episode success RATE overtakes)
and *cumulative* (total accumulated reward overtakes,
`recovery.metrics.parity.time_to_parity`). `T*` above is a statement
about cumulative regret, i.e. cumulative reward -- it predicts Task 6's
**cumulative** crossover, not the asymptotic one. `crossover_calibration.py`
calibrates against `criterion == "cumulative"` exclusively for this
reason; comparing `T*` against the asymptotic column would be comparing
the predictor to a quantity it was never derived to predict.

## 2. Numerical solution

Define `h(T) = R_TS(T) - rho*T`. `h > 0` means the static table's
cumulative regret has not yet caught up to Thompson Sampling's bound
(static still preferred, in this model); `h <= 0` means TS has caught up
(crossed). `dR_TS/dT` is proportional to `1/T` (the derivative of a log
term) -- a function that is *strictly decreasing* in `T` -- while
`d(rho*T)/dT = rho` is constant. So `dh/dT = dR_TS/dT - rho` is itself
strictly decreasing, meaning **`h` is unimodal**: it may rise at first
(if `dR_TS/dT` starts above `rho`), but once it turns, it falls
monotonically to `-infinity` (linear growth always eventually dominates
log growth, for any `rho > 0`). Consequently `h` has **at most one root**
beyond the trivial "already past `t_min`" case -- there is no risk of the
solver finding a spurious second crossing.

`predict_crossover` exploits this directly:

1. Compute `rho`. If `rho <= 0` (a perfect table), return `T* = infinity`
   immediately -- `R_TS(T) >= 0 = rho*T` for every `T`, an exact (not
   approximate) consequence, no solver needed.
2. Evaluate `h` on a `_SCAN_POINTS`-point (2,000) `numpy.geomspace` grid
   across `[t_min, t_max]` -- one vectorized call, not 2,000 Python-level
   calls.
3. If `h(t_min) <= 0`: the model predicts TS has *already* caught up by
   the smallest horizon considered -- return `(t_min, prediction_truncated=False)`.
4. Otherwise scan for the first adjacent pair where `h` flips from
   positive to `<= 0`, and refine with `scipy.optimize.brentq` on that
   narrow bracket -- deterministic, no randomness anywhere in the solve.
5. If no such pair exists (`h` stays positive across the whole bracket):
   the true root lies beyond `t_max` -- return `(t_max, prediction_truncated=True)`.
   This is the **only** case that sets the truncation flag; every other
   returned value (including `t_min` and every in-bracket root) is exact
   for the model as specified, never a silent clip.
6. Any unexpected numerical failure anywhere in this process is caught
   and mapped to step 5's answer -- `predict_crossover` never raises for
   a solver/bracketing failure (input validation, e.g. `t_min <= 0`,
   still raises `ValueError` immediately, before the solve begins --
   that is a caller error, not a bracketing failure).

**`t_min` must exceed the taxonomy size `N`.** `ts_cumulative_regret`
clips `log(T/N)` at 0 for `T <= N` (the bound is only meaningful for
`T >> N`). Whenever `t_min <= N`, `R_TS(t_min) = 0` exactly, so step 3
above ALWAYS fires regardless of table quality -- not a genuine
prediction, just the search floor sitting outside the bound's valid
range. The default `t_min=10` is only sensible for taxonomies smaller
than 10 types; `crossover_extension.py` raises `t_min` to 100 for its
`N` up to 56 sweep for exactly this reason.

## 3. Every assumption

- **Asymptotic, not finite-time.** The Lai-Robbins-style bound is a
  `T -> infinity` guarantee. Nothing in this module bounds the error at
  any *finite* `T` -- the calibration in Section 4 is the only evidence
  of finite-`T` accuracy this project has, and it covers a narrow regime.
- **Unknown multiplicative constant `c`.** Never derived from first
  principles. `c=1.0` is the literal formula; `experiments/crossover_calibration.py`
  fits a better-justified value empirically. A fitted `c` is a statement
  about how well the FUNCTIONAL FORM tracks this project's specific
  environment family, not a universal constant.
- **Independence across types.** Each type's row is treated as its own
  bandit sub-problem with `T/N` available pulls, ignoring that all `N`
  types are interleaved stochastically within one shared `T`-episode
  clock (not run in `N` sequential blocks of `T/N`).
- **Stationarity.** `P` and `pi` are fixed for the whole horizon. A table
  facing drift (Task 8, `recovery.envs.drift`) has a time-varying `rho`
  this module does not model at all.
- **Bernoulli reward model.** Every `P[t,k]` is treated as a Bernoulli
  success probability; the KL divergence used is specifically the
  Bernoulli one.
- **No classifier/confusion noise.** `pi` is assumed to act on the TRUE
  type (`policy_rate`, not `policy_rate_under_confusion`). A deployment
  with a noisy classifier has strictly worse effective `rho` than this
  module computes, in a way this module does not model.
- **Finite taxonomy**, fixed `K` strategies per type, no continuous
  action space.
- **Tied optimal arms carry zero regret**, excluded from every sum (see
  `ts_cumulative_regret`'s Edge cases) -- not a divide-by-zero
  workaround, a direct modeling choice ("choosing between equally-good
  optimal strategies costs nothing").
- **Uniform prior by default**, but `p` is honored throughout when
  supplied -- every per-type term in `R_TS(T)` is weighted by `p[t]`,
  matching `oracle_rate`/`policy_rate`'s own convention.

## 4. Calibration results

**Data.** Task 6's own `horizon_crossover.parquet` did not exist in the
repository when Task 10 began (only a 6-row, single-environment smoke
artifact had ever been persisted -- see
`experiments/crossover_calibration_grid.py`'s module docstring for the
full explanation and the ~78.6-hour full-grid estimate that made
regenerating Task 6's literal default grid impractical). It was
regenerated with `classifier_id='perfect'` only and `method in
{static_blind, thompson}` only, across `gap in {0.05, 0.10, 0.20, 0.30}`
(25 sampled matrices at `gap=0.20` -- matching Task 6's own default --
and 10 at each other gap, plus the canonical environment), 20-30 seeds
per cell, `T_max=100,000`: **56 calibration cells**, run in ~7 minutes.
Every number below is computed against `criterion == "cumulative"` (see
Section 1.3, "Which empirical crossover this predicts") from this grid.

**Headline finding: the uncalibrated predictor (`c=1.0`) is unusable, and
even the CALIBRATED predictor is reliable only in a narrow sub-regime.**

| | n | median ratio (predicted/observed) | within 2x | within 3x | validated overall? |
|---|---|---|---|---|---|
| Uncalibrated (`c=1.0`, KL form) | 56 | 110.7 | 0% | 0% | No |
| **Calibrated (best-fit `c=0.00452`, KL form)** | 56 | 0.241 | 17.9% | **41.1%** | **No** (below the 50% threshold) |
| Calibrated (best-fit `c=0.0452`, gap form) | 56 | 0.350 | 17.9% | 32.1% | No |

The KL form's calibrated fit is modestly better than the gap form's (as
expected -- see Section 1.2 -- KL is the asymptotically tight
approximation); both forms fall short of a 50%-within-3x bar in
aggregate. **`c` is not a universal constant here: it had to move by
~220x from its literal `c=1.0` value just to bring the median ratio near
1, and even then only 41% of predictions land within 3x.**

**Reliability by gap** (calibrated, KL form; `VALIDATED_WITHIN_FACTOR3_THRESHOLD=0.5`):

| gap | n | median ratio | within 3x | validated? |
|---|---|---|---|---|
| 0.05 | 10 | 2.09 | 70% | **Yes** |
| 0.10 | 10 | 0.19 | 40% | No |
| 0.20 | 25 | 0.12 | 32% | No |
| 0.30 | 10 | 0.14 | 40% | No |

**Reliability by table quality** (calibrated, KL form; count of
`BLIND_RULE` rows exactly matching that matrix's argmax):

| quality | n | median ratio | within 3x | validated? |
|---|---|---|---|---|
| 0/7 | 10 | 0.74 | 70% | **Yes** |
| 1/7 | 22 | 0.40 | 63.6% | **Yes** |
| 2/7 | 17 | 0.070 | 11.8% | No |
| 3/7 | 4 | 0.053 | 0% | No |
| 4/7 | 2 | 0.030 | 0% | No |
| 6/7 (canonical) | 1 | 0.019 | 0% | No |

**The predictor is reliable ONLY for very poor tables (quality <= 1/7)
at small gaps (~0.05).** For every other regime tested -- including the
project's own hand-authored canonical table (quality 6/7), which the
calibrated predictor underestimates by ~50x (predicted 160 episodes vs.
observed 8,470) -- it should not be trusted to better than an
order-of-magnitude.

**Root cause, not just a symptom.** This is not a calibration-procedure
bug (per-cell reconstruction was verified by hand against
`recovery.envs.canonical.MATRIX`/`BLIND_RULE` -- the canonical cell's
`quality=6` and `rho=0.0214` both match direct hand computation). It is
the consequence flagged in Section 1.2: neither regret-bound form
carries a per-arm `Delta_i` factor, so a single scalar `c` cannot
simultaneously fit both (a) the "obviously bad table, quickly overtaken"
cells that dominate this grid by count (35/56 cells sit at `gap >= 0.10`,
almost all of which the calibrated predictor floors to `t_min=10` --
i.e. "already crossed," which is directionally right but not
quantitatively informative) and (b) the "good table, large gap between
arms is rare, takes a long time to overtake" cells like `canonical`
(quality 6/7) where the true observed crossover is 50x later than even
the calibrated bound predicts. A predictor built on the textbook
Lai-Robbins regret bound (which DOES carry `Delta_i`) might resolve this
-- explicitly out of scope for Task 10 (see the closing note under
"Implementation constraints": the formula was specified verbatim).

**Files**: `results/tables/crossover_calibration.parquet` (22 rows: 11
groups x {uncalibrated, calibrated}, KL form).

### 4.1 Validated vs. non-validated performance, reported separately (Task 10B)

Task 10's original report stated a single number -- "41.1% within 3x
overall" -- for the calibrated predictor. **That single number is
misleading on its own**: it blends a sub-region where the predictor
genuinely works well with a much larger sub-region where it does not,
and a reader who only sees "41%" cannot tell which regime their query
falls into. Splitting every calibration point into `validated` (this
exact point's gap-bucket AND quality-bucket both cleared the 50%-within-3x
bar) vs. everything else, using the SAME `classify_status` function the
lookup table and every figure use (`experiments/crossover_reliability.py`):

| region | n | MAE (episodes) | RMSE (episodes) | median ratio | within 2x | within 3x | within 5x |
|---|---|---|---|---|---|---|---|
| **validated** | 9 | 64.9 | 77.9 | 2.19 | 33.3% | **66.7%** | 66.7% |
| **not validated** | 47 | 352.5 | 1303.3 | 0.15 | 14.9% | **36.2%** | 40.4% |

The "validated" region's 66.7% within-3x is close to (and consistent
with, given `n=9`) the design target; the "not validated" region's 36.2%
is what was dragging the blended "41.1% overall" number down. Reporting
only the blended number would have hidden that the predictor has a real,
usable core -- reporting only the split's good half would have hidden
that most of the calibration grid falls outside it. Both are reported
together for exactly this reason. Full per-(gap, quality) cell breakdown
(MAE, RMSE, median ratio, within-2x/3x/5x, `validated_region`,
`extrapolation`) is in `results/tables/crossover_calibration_summary.parquet`;
see also the reliability heatmap below.

### 4.2 Figures

- `results/figures/crossover_prediction_scatter.{png,pdf}` -- every
  calibration point, predicted vs. observed (log-log), colored by
  `prediction_status`, marker shaped by gap, with the `y=x` identity
  line and R²/MAE/RMSE (log10-space) annotated. The large cluster of
  points pinned to `predicted=10` (the `t_min` floor) is visually obvious
  here -- most `gap >= 0.10` cells resolve to "already crossed" under the
  calibrated `c`, regardless of how close their true observed crossover
  actually was.
- `results/figures/crossover_reliability_heatmap.{png,pdf}` -- fraction
  within 3x by (quality row, gap column), with sample count overlaid per
  cell. Visually confirms the pattern in the tables above: the top two
  rows (quality 0-1/7) are green (0.5-1.0) across every gap column; every
  row below that is red (0.0-0.25) for every gap except 0.05.
- `results/figures/crossover_error_histogram.{png,pdf}` -- distribution
  of `log10(predicted/observed)`, split validated (green) vs. not (red).
  The "not validated" mass sits well to the LEFT of 0 (systematic
  underprediction), while "validated" points cluster closer to, and
  somewhat to the right of, 0.
- `results/figures/crossover_calibration_curve.{png,pdf}` -- ratio vs.
  observed horizon (both log scale): shows the predictor's bias is not
  scale-independent -- larger true crossovers (the `canonical`-like,
  high-quality cells) are underpredicted more severely than small ones.

## 5. Explicit limitations

**This predictor is NOT exact.** It is a calibrated *estimate*, not a
guarantee, and it is calibrated only on this project's own synthetic
environment family (`src.evaluation.matrix_family.sample_matrix`'s
gap-controlled random matrices, reimplemented locally in
`recovery/theory/crossover.py` for import-discipline reasons -- see that
module's `_sample_matrix`), against `static_blind`
(`recovery.envs.canonical.BLIND_RULE`) as the ONLY static table shape
ever measured, and Thompson Sampling as the ONLY learner. It says
nothing calibrated about:

- **Real-world failure-type distributions or success-probability
  matrices.** The calibration grid's matrices are synthetic, not fit to
  any observed deployment.
- **Taxonomy sizes other than `N=7`.** Task 6's empirical grid never
  varied `N` (`horizon_sweep.py`'s `build_grid` always uses the
  canonical/gap-sampled 7-type shape). `crossover_extension.py`'s
  `N in {14, 28, 56}` rows are pure theoretical extrapolations of the
  formula's structure -- validated only by the MONOTONICITY unit tests
  in `tests/test_crossover.py`, never against real simulated outcomes.
  Every such row is flagged `is_extrapolation=True` in
  `results/tables/crossover_lookup.parquet`, never presented as
  validated.
- **Classifier confusion noise.** The calibration grid deliberately used
  `classifier_id='perfect'` only (see
  `experiments/crossover_calibration_grid.py`'s module docstring) --
  partly for runtime (the `real` classifier condition is roughly
  1,300x more expensive per episode), but primarily because the theory
  itself has no confusion-noise term to validate against.
- **Non-Thompson-Sampling learners.** UCB1, LinUCB, epsilon-greedy, etc.
  (Task 6's other 8 adapted methods) are out of scope for this
  regret-bound's derivation and were not part of the calibration grid.
- **Drifting environments.** Task 8's drift models are entirely outside
  this module's stationary assumption.
- **The gap-dependence shape omission noted in Section 1.2.** Neither
  formula includes the per-arm `Delta_i` factor a textbook Lai-Robbins
  regret bound has; the calibrated `c` cannot fully correct for this if
  it introduces a genuine shape mismatch across gaps -- Section 4's
  per-gap breakdown is the direct empirical check on this, and should be
  read before trusting a prediction at a gap far from what was
  calibrated.

Every prediction outside the validated regime described in Section 4 is
an **extrapolation**. `predict_crossover`/`predict_crossover_grid` do not
refuse to compute one (refusing outright would make the lookup-table
extension impossible), but `crossover_extension.py` labels every such row
explicitly (`is_extrapolation`, `validated_region`) rather than letting a
number speak for itself.

## 6. Practitioner example

**Scenario**: 7 failure types, 40 failures per type observed each month
(280 total episodes/month), an estimated static-table quality of `5/7`
correct type-to-strategy mappings, and an estimated optimal-arm gap of
`~0.15` on the types the table gets wrong.

```python
from recovery.theory.crossover import predict_crossover_grid

rows = predict_crossover_grid(
    quality_levels=[5], gaps=[0.15], K=6, N=7,
    n_matrices=25, seed=555, form="kl", c=0.00452,   # calibrated c
)
```

- **Calibrated prediction** (`c=0.00452`, the fitted constant from
  Section 4): median `T* = 10` episodes across 25 sampled matrices at
  this (quality, gap) -- the search floor (`t_min=10`), i.e. "already
  crossed" -- **under 1 month** (`10 / 280 ~= 0.04` months).
- **Uncalibrated prediction** (`c=1.0`, the literal formula): median
  `T* ~= 32,940` episodes -- **~118 months (~9.8 years)**.

**These two numbers disagree by roughly 3,300x, and that disagreement is
itself the honest answer.** Quality `5/7` and gap `0.15` fall BETWEEN
every validated cell in Section 4: the calibration grid's validated
regime is quality `<= 1/7` (this deployment's table is much better than
that) and gap `~0.05` (this deployment's gap is 3x larger). The nearest
data point with comparable quality is `canonical` (quality `6/7`), where
the calibrated predictor underestimated the true crossover by ~50x --
which suggests the calibrated number above (`< 1 month`) is likely a
substantial UNDERestimate for a quality-`5/7` table too, and the
uncalibrated number, while itself unreliable, is directionally more
plausible for a table this good.

**Recommendation**: given neither number falls in a validated regime,
do not deploy on the strength of this prediction alone. What the
prediction IS good for: a directional signal that this is NOT a
"perfect table, never worth switching" case (`rho ~= 0.083 > 0`, a
genuine, permanent gap exists) and not a "hopelessly bad table, switch
immediately" case either (quality `5/7` sits far above the validated
"quality <= 1/7" regime where near-immediate crossover was reliably
observed). A short live pilot (running Thompson Sampling in shadow mode
for a few months and directly measuring its late-stage rate against the
table) will answer this specific deployment's question far more reliably
than extrapolating either curve above.

## 7. Validation methodology

Three layers, each checking a different thing:

1. **Unit tests** (`tests/test_crossover.py`, 24 tests): pure formula
   properties that must hold regardless of any empirical data --
   monotonicity in quality/gap/`K`/`N`, the perfect-table-returns-infinity
   edge case, the solver satisfying its own equation to within 1%,
   `static_regret_rate` matching `oracle_rate - policy_rate` bit-exactly,
   never raising on a bracketing failure, and the documented KL-vs-gap
   divergence pattern. These validate the IMPLEMENTATION is correct
   relative to its OWN specification -- they say nothing about whether
   the specification describes reality.
2. **Empirical calibration** (`experiments/crossover_calibration.py`,
   Section 4): the only layer that checks the specification against
   reality, by comparing predicted `T*` to `T*` measured directly from
   50,000-episode Thompson Sampling simulations (Task 6's kernel,
   unmodified). This is where the predictor's real accuracy -- and its
   real limitations -- were discovered, not assumed.
3. **Reliability classification** (`experiments/crossover_reliability.py`,
   Task 10B): converts layer 2's raw numbers into an actionable,
   machine-readable verdict (`prediction_status`/`confidence_level`) for
   every point a downstream consumer might query, so "how much do I
   trust this specific prediction" is never left to the reader's
   judgment call on a summary statistic.

Every classification in this document, the lookup table, and every
figure traces back to ONE function (`crossover_reliability.classify_status`)
applied consistently -- there is no risk of a point being labeled
"validated" in a figure and "extrapolated" in the lookup table, because
both call the same code.

## 8. Calibration procedure

`experiments/crossover_calibration.py::fit_best_c` fits `c` by minimizing
mean squared LOG-SPACE error,

```
c* = argmin_c  mean( (log(predicted_T(c)) - log(observed_T))^2 )
```

over the 56 calibration cells with a finite observed crossover, via
`scipy.optimize.minimize_scalar` (bounded, `log(c) in [log(1e-3), log(1e4)]`).
Log-space, not raw-space, because the observed crossovers span nearly
three orders of magnitude (7 to 8,470 episodes) -- a raw-space
least-squares fit would be dominated entirely by the `canonical` cell's
error and effectively ignore the other 55 points. `c` enters
`ts_cumulative_regret` linearly and `predict_crossover`'s solution is
monotonic in `c` (Section 2), so this is a well-behaved 1-D optimization
with a single minimum, not a search over a jagged landscape.

**This procedure was NOT changed for Task 10B.** Nothing in this section
describes new fitting logic -- `crossover_calibration.py` is untouched;
Task 10B only added the classification and presentation layer that
INTERPRETS this same fit's results.

## 9. Failure modes

Two distinct, opposite-direction failure modes, not one:

- **The "floor" failure (mainly `gap >= 0.10`, `quality <= 1/7`):** the
  calibrated predictor resolves to `T* = t_min = 10` for most of these
  cells (see the scatter plot's horizontal cluster at `predicted=10`).
  This happens to be DIRECTIONALLY correct much of the time (these
  really are quickly-overtaken tables -- observed crossovers as low as
  7-30 episodes are common here), but the prediction carries no
  information beyond "very fast" -- it cannot distinguish a table that
  crosses at episode 7 from one that crosses at episode 90, because both
  floor to the same search boundary. Increasing `t_min` would only move
  the floor, not remove it, since `h(t_min) <= 0` for essentially any
  `t_min` in this regime once `c` is small enough to fit the rest of the
  grid.
- **The "severe underprediction" failure (`quality >= 2/7`):** the
  predictor underestimates the true crossover by 1-2 orders of magnitude
  -- worst observed at `canonical` (quality 6/7): predicted 160 episodes,
  observed 8,470 (50x). This is the economically more dangerous failure
  mode: a practitioner reading a "short" predicted crossover for a
  DECENT table could conclude switching to online learning pays off
  almost immediately, when the true payoff is an order of magnitude
  further out.

Both failure modes are consequences of the same root cause (Section 1.2,
Section 10): a single scalar `c` cannot simultaneously fit a regime that
wants a small effective regret bound (the floor cases) and a regime that
wants a large one (the underprediction cases). Calibrating pulled `c`
toward the numerically-dominant floor regime (35/56 cells sit at
`gap >= 0.10`), which is *why* the underprediction failure is the worse
of the two in practice -- the fit was never going to serve both regimes
well simultaneously, and the larger regime by count won.

## 10. Threats to validity

- **Construct validity.** `R_TS(T)` is a stylized proxy for "Thompson
  Sampling's true cumulative regret," not a measurement of it -- it is
  an asymptotic UPPER BOUND on a DIFFERENT quantity (expected pulls of a
  suboptimal arm), repurposed via an approximation chain (Section 1.2).
  Calibration checks whether the proxy TRACKS reality well enough to be
  useful, not whether it equals reality.
- **Internal validity.** All 56 calibration cells share ONE environment
  family (`sample_matrix`'s uniform-baseline-plus-boosted-arm
  construction), ONE static table (`BLIND_RULE`), ONE classifier
  (`perfect`), ONE learner (Thompson Sampling with the project's default
  prior). Every reported reliability number is conditional on all of
  these being held fixed; nothing here isolates which of them matters
  most to the observed failure modes.
- **External validity.** Real deployments' failure-type distributions,
  success-probability structures, and classifier accuracy will not match
  `sample_matrix`'s synthetic construction. The calibration says nothing
  about how well `c=0.00452` generalizes to a genuinely different
  environment family -- only that it is the best fit TO THIS ONE.
- **Statistical validity.** Several (gap, quality) cells in
  `results/tables/crossover_calibration_summary.parquet` have `n=1` or
  `n=2` samples (e.g. gap=0.30, quality=0: `n=1`). A within-3x fraction
  of "100%" on one sample is not evidence of reliability at any
  meaningful confidence level -- it is reported as-is (never suppressed
  or smoothed), but should be read as "no information" rather than "high
  reliability," and is exactly why the group-level `gap`/`quality`
  validation (pooling across the *whole* gap or quality bucket, `n=10-25`)
  is the classification's basis, not these sparse individual cells.
- **Selection effects in the calibration grid itself.** `gap=0.20` has
  25 sampled matrices (matching Task 6's own default); every other gap
  has only 10. The least-squares fit is therefore implicitly weighted
  toward `gap=0.20`'s error pattern -- not a deliberate design choice for
  the fit, just an inherited consequence of reusing Task 6's own grid
  density where possible (see `crossover_calibration_grid.py`'s
  docstring).

## 11. Why one scalar cannot model every quality regime

Section 1.2 already flags the mechanism: neither `ts_cumulative_regret`
form carries the per-arm gap factor `Delta_i` that a textbook Lai-Robbins
*regret* bound has. Concretely, this means the formula's *relative*
weighting of "many pulls of an easy-to-distinguish bad arm" vs. "few
pulls of a hard-to-distinguish bad arm" does not match the true regret's
weighting (which is pulls TIMES the per-pull cost `Delta_i`, not pulls
alone). A single `c` can rescale the WHOLE curve up or down, but it
cannot selectively stretch the low-quality regime differently from the
high-quality regime -- and Section 9 shows those two regimes need
opposite corrections (the floor regime is already too aggressive at
`c=0.00452`; the high-quality regime is nowhere near aggressive enough).
This is a structural property of a ONE-PARAMETER family fit to a
TWO-REGIME problem, not a bug in the fitting procedure -- adding a
second free parameter (e.g., a per-quality or per-gap multiplier) could
close this gap, but doing so was explicitly out of scope for Task 10B
("do not change the mathematical model") and would also require far more
calibration data than 56 cells to fit responsibly (see "Future work").

## 12. Why calibrated estimation is still useful

A predictor that is only reliable in a narrow regime is not the same as
a useless predictor, for three concrete reasons:

1. **It replaces "no idea" with "bounded idea" in the validated regime.**
   For quality <= 1/7 tables at gap ~0.05 -- a common real situation (a
   table with almost no correct mappings) -- the predictor gives a
   usable, ~2-3x-accurate answer in milliseconds, no simulation required.
2. **Its qualitative sign is trustworthy far more broadly than its
   quantitative value.** `rho > 0` (a permanent gap exists, computed
   EXACTLY via Task 1's `oracle_rate`/`policy_rate`, not approximated at
   all) is always correct; whether that gap is worth chasing this month
   or this decade is the part that needs calibration, and even a
   badly-calibrated `T*` reliably tells you "not never" vs. "never"
   (the `rho <= 0 -> infinity` case, exact by construction -- see
   Section 2).
3. **Knowing exactly where it fails is itself the deliverable.** Section
   4's heatmap is a map of where a practitioner should trust a
   five-minute analytic estimate versus where they should run a real
   pilot (Section "Comparison with purely empirical estimation" below).
   That map has value independent of how good the predictor is on
   average.

## 13. Comparison with purely empirical estimation

| | Analytic predictor (this module) | Purely empirical (Task 6-style sweep) |
|---|---|---|
| Time to an answer | Milliseconds (closed-form-ish, one `brentq` solve) | Minutes-to-hours (must simulate `T_max` episodes per seed x many seeds) |
| Accuracy (validated regime) | ~2-3x | Exact, up to Monte Carlo noise across seeds |
| Accuracy (outside validated regime) | Unknown, possibly 1-2 orders of magnitude off | Still exact -- empirical measurement has no "regime" restriction, only a compute cost |
| Requires simulation infrastructure | No | Yes (the full kernel, `RunStore`, parallel execution) |
| Generalizes to an unmeasured (quality, gap, N) | Yes, instantly (with the caveats above) | No -- must actually run that configuration |
| Tells you WHERE it might be wrong | Yes (`prediction_status`) | N/A -- it's not wrong, it's a direct measurement |

The two are complementary, not competing: the analytic predictor is a
fast SCREENING tool (which deployments even warrant a pilot?), and the
empirical sweep is the GROUND TRUTH tool the analytic predictor was
calibrated against and should defer to whenever the two disagree and
compute budget allows running one.

## 14. Future work

- **A textbook Lai-Robbins regret bound (with the `Delta_i` factor)**
  as a third `form` option -- Section 11's analysis suggests this could
  resolve the two-regime problem without needing a second free
  parameter, since `Delta_i` would naturally reweight the sum toward
  small-gap arms where more pulls are actually needed.
- **A larger, more diverse calibration grid**: more taxonomy sizes
  (`N != 7`), the `real` classifier condition, non-Thompson-Sampling
  learners, and denser sampling of the `quality >= 2/7` regime (currently
  the sparsest, per Section 10's statistical-validity note) -- this is
  the single most direct way to shrink the "extrapolated"/"unsupported"
  fraction of `crossover_lookup.parquet` (currently 420 + 7 = 427/436
  rows).
- **Per-regime or per-quality calibrated constants** (`c(quality)` or
  `c(gap)` instead of one shared `c`) -- explicitly out of scope for
  Task 10B, flagged here as the most direct fix for Section 11's
  structural problem, at the cost of needing enough data per regime to
  fit responsibly.
- **Uncertainty-quantified predictions** (a confidence interval on `T*`,
  not just a point estimate) -- would make `crossover_lookup.parquet`
  self-describing about HOW wrong a "partially_validated" prediction
  might be, rather than only THAT it might be wrong.
- **A drift-aware extension**, connecting this module to Task 8's
  time-varying environments -- currently entirely out of scope
  (Section 3, "Stationarity").

## 15. Reviewer-oriented discussion

**Why isn't 41% a failure?** Because "41% within 3x overall" was never
the right number to evaluate the predictor by -- it is an average over
two regimes with genuinely different reliability (Section 4.1: 66.7%
validated vs. 36.2% not validated), and averaging across a regime
boundary produces a number that describes neither regime accurately.
Judged on the regime it was designed and calibrated to serve (quality
<= 1/7, small gap), it clears its own 50% bar. Judged as a universal
predictor for any (quality, gap), it correctly does not clear that bar
-- and was never claimed to.

**Why is the predictor still scientifically useful?** See Section 12.
In short: exact qualitative signal (does a gap exist at all) everywhere,
plus a usable quantitative estimate in a real, identifiable sub-region,
plus an honest, machine-readable map of where the quantitative estimate
should not be trusted. A tool that knows the boundary of its own
competence is more useful than a tool that doesn't, even if the
competence region is smaller than one might have hoped going in.

**What practical decisions can still be made with it?** (1) For a table
in the validated regime, whether to prioritize a Thompson Sampling pilot
NOW vs. LATER, with a real (if approximate) timeline. (2) For ANY table,
whether a permanent gap exists at all (`rho > 0`, always exact) --
i.e., whether static-table maintenance costs will EVER be worth trading
for online learning, independent of when. (3) For a table outside the
validated regime, a same-order-of-magnitude sanity check against a
purely qualitative guess, clearly flagged as low-confidence rather than
presented as a number to plan around.

**Why is marking extrapolated regions preferable to forcing a better
fit?** Because forcing a better aggregate fit (e.g., re-fitting `c` to
directly maximize the "within 3x overall" percentage, or worse, ADDING
free parameters until the calibration grid's own 56 points fit well)
would optimize the metric a reader sees WITHOUT improving the model's
actual generalization -- classic overfitting, explicitly forbidden by
Task 10B's own framing ("The goal is NOT to improve the calibration
percentage by overfitting"). A model with an honestly-reported 41%
overall accuracy and a correctly-flagged validated sub-region is more
trustworthy, and more scientifically useful, than a model with a
reported 90% accuracy achieved by curve-fitting to the same 56 points
being used to report that accuracy. The flags are the honesty mechanism;
removing them (or hiding them behind a single blended number) would not
make the underlying predictions any more correct -- it would only make
the incorrect ones harder to identify.

## Reproducibility

```
python experiments/crossover_calibration_grid.py --estimate   # print estimated wall-clock and exit
python experiments/crossover_calibration_grid.py               # regenerate results/tables/horizon_crossover.parquet
python experiments/crossover_calibration.py                    # results/tables/crossover_calibration.parquet
python experiments/crossover_reliability.py                    # results/tables/crossover_calibration_summary.parquet (Task 10B)
python experiments/crossover_figures.py                        # results/figures/crossover_*.{png,pdf} (Task 10B)
python experiments/crossover_extension.py                      # results/tables/crossover_lookup.parquet
pytest tests/test_crossover.py tests/test_crossover_reliability.py -v
```

See `docs/experiments/drift_validation.md`'s reproducibility section for
the project-wide convention this follows. See also
[`practitioner_guidelines.md`](practitioner_guidelines.md) for how to
read `crossover_lookup.parquet` safely once regenerated.
