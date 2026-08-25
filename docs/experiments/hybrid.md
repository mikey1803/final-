# Prior-Guided Thompson Sampling (PGTS)

This document describes Task 7's algorithmic contribution: **Prior-Guided
Thompson Sampling (PGTS)**, implemented as
`recovery.policies.hybrid.PriorInitializedThompson` (and its discounted
variant, `DiscountedPriorThompson`). Throughout this document the
algorithm is called PGTS; the implementation class name is
`PriorInitializedThompson`.

## Motivation

The paper's Discussion speculates: *"A bandit initialized from a blind
hand-designed recovery table could combine the strong stationary
performance of static policies with the adaptability of online
learning."* This was never evaluated. PGTS is that evaluation: a bandit
that starts from a hand-designed table's belief and updates it with real
evidence, rather than choosing between "trust the table forever"
(a static policy) and "trust nothing, learn everything from scratch"
(stock Thompson Sampling).

## Mathematical derivation

For a recovery table `pi : failure type -> recommended strategy`, prior
strength `s >= 0`, and prior mean `m in (0, 1)`:

```
alpha0[t, k] = 1 + s*m,       beta0[t, k] = 1 + s*(1-m)     if k == pi[t]
alpha0[t, k] = 1,             beta0[t, k] = 1               otherwise
```

Selection, for the PREDICTED failure type `t`: sample
`theta_k ~ Beta(alpha[t,k], beta[t,k])` for every strategy `k` (one
vectorized `rng.beta(...)` call across all `k`, never a Python loop --
see "Performance" below), then play `argmax_k theta_k`.

Update, on the arm actually played: `alpha[t, action] += reward`,
`beta[t, action] += 1 - reward`. Nothing else changes -- in particular,
`update()` reads only `ctx.predicted_type` (the `Context` protocol
structurally has no true-type field to read from at all, see
`recovery.core.protocols.Context`), so PGTS conditions on the same
information every other policy in this codebase does.

`s` and `m` together parameterize how much of a "head start" the
recommended arm gets: `s` is a pseudo-observation COUNT (how many
imaginary trials worth of confidence the prior represents), `m` is what
those imaginary trials are believed to average out to.

## Limiting cases: PGTS interpolates between two families the paper already studies

### `s = 0`: identical to stock Thompson Sampling

At `s=0`, `alpha0[t,k] = 1 + 0*m = 1` and `beta0[t,k] = 1` for EVERY
`(t, k)`, including the recommended arm -- the table has been given zero
weight and has literally no effect on the resulting `(alpha, beta)`
arrays. This isn't an approximation: `tests/test_hybrid.py::test_strength_zero_is_table_independent_and_matches_stock_thompson`
verifies it two ways -- two `PriorInitializedThompson` instances built
from DIFFERENT tables at `s=0` produce bitwise-identical action
sequences (the table cannot be influencing anything), and a from-scratch
minimal vectorized Beta(1,1) Thompson Sampling implementation, run
through the same kernel and seed, matches too.

### `s -> infinity`: identical to the static recovery table

As `s` grows, the recommended arm's posterior variance
(`alpha*beta / ((alpha+beta)^2*(alpha+beta+1))`) shrinks toward zero and
its mean is pinned at `m` -- PGTS increasingly plays `pi[t]` for every
predicted type `t`, exactly what `recovery.policies.adapters.StaticRulePolicy`
does. `strength=float("inf")` is REJECTED by the constructor rather than
approximated (`1 + inf*m` is not a number `rng.beta` can use) -- callers
wanting the literal limit should use `StaticRulePolicy(table)` directly,
which implements it exactly, with no floating-point approximation
involved. `experiments/prior_grid.py`'s grid represents this limit point
with `StaticRulePolicy`, labeled `method="pgts_static_limit"`.

### A subtlety in the large-strength limit

The "matches the static rule" claim above is about the MEAN and
VARIANCE of the recommended arm's own posterior -- it does NOT mean a
large-but-finite-strength PGTS instance's ACTIONS are guaranteed to
match `StaticRulePolicy`'s from episode zero. Every OTHER (non-recommended)
arm still starts at literal `Beta(1,1)` = Uniform(0,1); with `K`
strategies and prior mean `m`, the probability that at least one of the
`K-1` other arms outsamples the tightly-concentrated recommended arm on
a given early episode is approximately `1 - m^(K-1)` -- non-negligible
for `m` around the everyday default of `0.7` (e.g. `~83%` for `K=6`
first-episode probability with `m=0.7`). This is NOT a bug: it is the
Bayesian machinery correctly reflecting that a not-yet-explored arm's
uncertainty is total, however implausible its (temporary) success looks
against a well-informed competitor. Once a wrong arm has been sampled a
few times, real evidence pulls its own posterior down and it stops
winning by chance. Practically: with `prior_mean` closer to 1
(`0.95`-`0.99`), the per-episode divergence probability shrinks sharply,
and `PriorInitializedThompson` at `strength=1e6` tracks
`StaticRulePolicy` with >99% action agreement over the final quarter of
a 5,000-episode run (see `tests/test_hybrid.py::test_very_large_strength_behaves_like_static_rule`)
-- "behaves like the static rule during early learning" (this task's own
acceptance-criteria phrasing) rather than "is bitwise identical to it
from episode zero", which the algorithm as specified does not actually
guarantee.

## Prior quality and `from_quality`

`PriorInitializedThompson.from_quality(matrix, n_correct, rng)` builds a
table with exactly `n_correct` of `matrix`'s rows mapped to their true
best (argmax) strategy, and every other row mapped to a uniformly random
NON-best strategy -- by reusing
`src.evaluation.static_rule_sweep.generate_static_policy` (the same
quality-controlled generator Task 4/5's static-rule sweeps already use),
converted to the array-index convention every `recovery/` policy expects.
`n_correct=0` (a fully-incorrect table) is supported -- `random.sample`
of zero correct rows is well-defined, and this is an explicit grid point
(`experiments/prior_grid.py`'s `DEFAULT_QUALITIES` includes `0`) used to
characterize negative transfer.

**Tie-breaking**: `matrix.argmax(axis=1)` (used internally) resolves
ties -- multiple strategies achieving a row's maximum success
probability -- by taking the FIRST such strategy in `STRATEGY_NAMES`
order, matching `numpy.argmax`'s documented convention (the same
convention `recovery.theory.analytic` already commits to for "the true
best strategy" generally).

## Posterior entropy

`posterior_entropy()` returns the analytic DIFFERENTIAL entropy of
`Beta(alpha, beta)` for every `(type, strategy)`:

```
H = ln B(alpha, beta) - (alpha-1)*psi(alpha) - (beta-1)*psi(beta) + (alpha+beta-2)*psi(alpha+beta)
```

(`psi` = digamma; `B` = the Beta function -- `scipy.special.betaln`/`digamma`,
not a Monte Carlo estimate). This CAN be negative -- it is differential
entropy of a continuous density, which has no sign constraint, unlike
discrete Shannon entropy.

A high-strength prior's recommended arm starts with LOW entropy (already
concentrated near `prior_mean`); stock Thompson Sampling's every arm
starts at `Beta(1,1)`, the MAXIMUM-entropy point on `[0,1]`. As learning
proceeds, entropy generally decreases (the posterior concentrates as
pseudo-count accumulates) -- `tests/test_hybrid.py::test_posterior_entropy_monotonically_decreases_under_sustained_learning`
verifies this at coarse (250-episode) checkpoints for a heavily-exploited
arm in a stationary environment; individual episodes can occasionally
tick entropy up slightly (a surprising outcome widens the posterior
before it narrows again), which is why the test checks a coarse trend,
not every single episode.

## Convergence behavior

`experiments/prior_analysis.py::posterior_convergence` measures, per
strength, the episode at which the recommended arm's posterior mean
first comes within `epsilon` of the environment's TRUE best-arm success
probability and stays there. Higher strength converges (trivially)
faster in this specific sense, because it starts closer -- the
interesting comparison is not "does higher strength converge faster"
(it starts ahead by construction when the prior is correct) but how much
of the STOCK Thompson Sampling exploration budget a correct prior
actually saves, which is exactly what this measurement quantifies.

## Negative transfer

**Negative transfer** = a PGTS configuration performing WORSE than stock
Thompson Sampling (`strength=0`) at the same environment/horizon.
`experiments/prior_analysis.py::negative_transfer_region` computes this
explicitly (not just a plot): for every `(env_id, n_episodes, quality,
strength)` cell, the delta between its mean `normalized_score` and stock
Thompson's mean at the same `(env_id, n_episodes)`.

**Mechanism**: negative transfer requires BOTH a low-quality table (the
recommendation is often wrong) AND high strength (the wrong
recommendation is asserted with a large, hard-to-overturn pseudo-count).
Neither alone is sufficient -- a low-quality table at `strength=0`
degrades gracefully to stock Thompson (ignored entirely); a high-strength
CORRECT table only helps. It is the PRODUCT of wrongness and confidence
that produces harm: a confidently-wrong prior needs real evidence to
overcome its elevated pseudo-count before the policy will even try the
actually-better arm, and every episode spent failing to override that
confidence is an episode stock Thompson Sampling (which never starts
with false confidence about anything) would likely not have wasted.
`tests/test_hybrid.py::test_negative_transfer_exists_for_bad_prior_under_drift`
confirms this region is reachable and would FAIL the build if it were
not -- per Task 7's own instruction, a PGTS implementation with no
negative-transfer region anywhere in its parameter space would
contradict the intended algorithmic behavior, not validate it.

## Discounted PGTS

`DiscountedPriorThompson` is `PriorInitializedThompson` with `gamma`
defaulting to `0.99` instead of `1.0` -- no logic is duplicated; the
discount mechanism already lives in the base class's `update()`
(`gamma=1.0` is an exact identity, verified by
`tests/test_hybrid.py::test_base_class_gamma_one_is_a_no_op_discount`).
Every `update()` call discounts the ENTIRE row of the type just observed
(not only the arm played) toward `Beta(1,1)`:

```
alpha[t, :] <- 1 + gamma*(alpha[t, :] - 1)
beta[t, :]  <- 1 + gamma*(beta[t, :] - 1)
```

before applying the new observation. Repeated without reinforcement,
this sends `alpha, beta -> 1` GEOMETRICALLY (rate `gamma` per update) --
toward the UNINFORMATIVE prior, not back toward the original informed
one. This is why an incorrect prior can "decay away" under drift: the
elevated pseudo-count the table's recommendation started with is not
sticky. Every observation of that failure type -- regardless of which
arm was actually played -- erodes it a little. Once the environment
shifts and the previously-good arm stops paying off, each new
observation both (a) shrinks the stale confidence in the old
recommendation and (b) adds fresh evidence, so real signal about the new
regime regains influence faster than under `gamma=1` (where stale
evidence never fades and must instead be out-VOTED by an ever-growing
count of new observations). This is the same mechanism
`DiscountedThompsonSamplingSelector` (`src/recovery/selector.py`,
untouched by this task) already uses for stock Thompson Sampling,
applied here on top of an informed starting point instead of an
uninformative one.

`experiments/prior_analysis.py::recovery_half_life` quantifies this: the
episode count, after a scheduled shift, for a rolling-window success
rate to recover halfway from its post-shift trough back to its pre-shift
level, as a function of `strength` and `gamma`.

## Environments: stationary, single-shift, repeated-shift

`experiments/prior_grid.py::DriftEnvironment` implements a
`recovery.core.protocols.Environment` whose success matrix is permuted
at scheduled episodes -- conceptually the same "permute the affected
rows' strategy->probability mapping" mechanism as
`src.evaluation.nonstationarity.build_shifted_matrix` (row difficulty is
preserved; only which strategy achieves which probability is scrambled),
reimplemented against the array-indexed matrix convention `recovery/`
uses throughout rather than reused directly (that function operates on
`FailureType`-keyed dicts and would need a per-call enum/index
conversion inside the kernel's hot loop to reuse as-is). A single
scheduled shift is the "single_shift" condition; several equally-spaced
shifts is "repeated_shift" -- both use the same class with a different
schedule. Neither is added to `recovery/envs/` (Task 7's file list does
not include a new environment module, and no existing file was modified)
-- it is a self-contained `experiments/`-local class satisfying the
`Environment` protocol structurally, exactly as
`recovery.core.episode.run_episodes` requires.

**A documented approximation**: per-cell metrics
(`recovery.metrics.summary.summarize`) are computed against
`env.success_matrix()`'s state AFTER the run completes -- for a drifted
run, this is the FINAL (post-all-shifts) matrix. `late_stage_rate` and
`worst_type_rate` (both computed on the run's tail) are therefore
evaluated against the SAME regime that produced them and are trustworthy;
regret-based metrics (`final_cumulative_regret`,
`normalized_cumulative_regret`), which sum over the WHOLE run spanning
pre- and post-shift episodes, are only APPROXIMATE for drift conditions
-- they compare pre-shift actions/rewards against a post-shift ceiling
that did not apply to them at the time. This is a genuine limitation of
using a single-`P` metrics API (`recovery.metrics`, unmodified per this
task's constraints) against a time-varying ground truth, stated here
rather than silently producing a number that looks precise but isn't.

## Performance

`select()` samples every strategy's `theta` in ONE vectorized
`rng.beta(alpha[t], beta[t])` call (`alpha[t]`/`beta[t]` are already
`(n_strategies,)`-shaped row slices) -- never a Python loop over
strategies, per this task's explicit requirement.

## Practitioner recommendations

- **`s=0` (stock Thompson)**: no table available, or the table is not
  trusted at all.
- **Small `s` (e.g. `5`)**: a LOW-COMMITMENT nudge -- biases early
  exploration toward the table without making it hard to overturn (a
  handful of contradicting observations already dominate the prior's
  pseudo-count). Reasonable whenever the table's quality is uncertain, or
  the environment may drift and a static commitment would be costly to
  unwind.
- **Large `s` (e.g. `50`+)**: a HIGH-COMMITMENT bet -- appropriate only
  when the table is trusted (high quality, e.g. hand-verified by a
  domain expert) AND the environment is believed stationary. See
  `experiments/prior_analysis.py::factor_contribution`'s descriptive
  spread table for a rough sense of how much `strength` vs `quality` vs
  drift each moved outcomes in a given run -- a guide to which factor is
  actually worth being confident about before committing to a large `s`.
- **`gamma < 1` (Discounted PGTS)**: whenever drift is plausible.
  Costs a small amount of stationary-case performance (a converged,
  correct posterior keeps getting nudged back toward less-certain) in
  exchange for materially faster recovery after a shift (see
  `recovery_half_life`) -- a straightforward stability/plasticity
  trade-off, not a free lunch.

## Reproducing the experiment

```
python experiments/prior_grid.py                         # main grid (all defaults)
python experiments/prior_grid.py --discounted             # discounted grid (recovery half-life)
python experiments/prior_analysis.py \
    --discounted-input results/tables/discounted_prior_grid.parquet
```

`--smoke` on `prior_grid.py` (both modes) runs a small representative
subset in well under 60 seconds, for CI/quick iteration; full defaults
match this document's grid (strengths `{0,1,2,5,10,20,50,100,500,inf}`,
qualities `{0,3,4,5,6,7}`, prior means `{0.6,0.7,0.8}`, all three
environments, horizons `{500,5000,50000}`, both classifier conditions,
30 seeds).
