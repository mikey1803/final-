# The Horizon Sweep

This document explains Task 6's central experiment
(`backend/experiments/horizon_sweep.py`, `horizon_analysis.py`) and how to
read its output. It builds on the shared kernel (Task 2,
[`architecture.md`](../architecture.md)) and the metrics layer (Task 5,
[`statistics.md`](../statistics.md)).

## The confound this experiment removes

The paper's stationary claim -- "static registries dominate stationary
environments" -- was measured at exactly one interaction budget: 500
episodes over 7 types, roughly 12 pulls per arm. Thompson Sampling's
regret bound is `O(K log T / gap)`: at `T=500` with `K=6` strategies, the
`log T` term is still small and a meaningful fraction of the budget is
necessarily still exploration. A comparison run at that one budget cannot
distinguish "adaptive selection is worse than a good static rule" from
"adaptive selection hasn't finished paying its exploration cost yet." The
original result is a statement about `T=500`, not a statement about
static-vs-adaptive selection as a category -- the budget itself was a
hidden confound, present in every reported number but never varied or
even named as a variable.

This sweep makes `T` an explicit axis: 7 orders-of-magnitude-spanning
horizons (`100` through `100,000`), the same methods, the same
environments, run once and evaluated at every horizon. If the paper's
ranking reverses somewhere in that range, the "static dominates" claim
was a budget artifact; if it holds at `T=100,000` too, that is a much
stronger claim than the original 500-episode measurement ever supported
either way.

## Prefix evaluation, and why it makes the curve paired

The kernel (`recovery/core/episode.py::run_episodes`) expands one integer
seed into four independent RNG streams (`recovery/core/rng.py`) ONCE, at
the top of the function -- never re-derived partway through, never a
function of `n_episodes`. Every episode then draws from those streams
strictly sequentially. Two consequences fall out of this for free:

1. **The first `T` episodes of a `T_max`-episode run are identical to a
   fresh `T`-episode run at the same seed.** Nothing about episode `i`'s
   true type, classifier prediction, policy action, or reward depends on
   how many MORE episodes are about to happen -- only on the sequence of
   draws up to and including episode `i`. This holds for stateful,
   learning policies too: a policy only ever conditions on its OWN past
   `(context, action, reward)` history, which is byte-identical for
   episodes `0..T-1` regardless of what (if anything) comes after.
   `tests/test_horizon.py`'s prefix-equivalence tests verify this
   directly, for both a memoryless policy (`StaticRulePolicy`) and a
   stateful one (`ThompsonPolicy`): exact array equality, zero tolerance,
   on every logged field.
2. **This makes "evaluate every horizon" nearly free.** Run `T_max` once
   per `(method, environment, seed)` cell, then take `recovery.metrics.prefix.prefix_view(log, T)`
   for each `T` in the grid -- an array slice, not a simulation. Against
   naively re-running once per horizon, this is roughly 7x cheaper (one
   run instead of seven) for exactly this sweep's grid.

The more important consequence is statistical, not computational: because
every horizon for a given seed is a PREFIX of the SAME trajectory, all
seven points on that seed's curve share the exact same random draws for
every episode they have in common. There is no possibility of, say, the
`T=1000` point happening to land on an easy run of types while the
`T=3000` point (a different simulation, under the naive approach) happens
to land on a hard one -- the curve is internally consistent by
construction. This is what makes it meaningful to ask "where does this
seed's curve cross the baseline's" (`crossover_horizon`,
`recovery/analysis/crossover_empirical.py`) rather than only being able
to compare noisy independent estimates at each horizon separately.

## Two crossover notions

This sweep reports two different answers to "when does an adaptive
method catch up to `static_blind`," because they answer different
questions and can differ by orders of magnitude:

- **Asymptotic crossover** (`horizon_analysis.py`, criterion=`"asymptotic"`):
  where the PER-EPISODE success rate (`late_stage_rate` at each horizon
  `T`) overtakes `static_blind`'s and stays there for every larger `T` in
  the grid. Computed by `crossover_horizon`, log-linearly interpolating
  between the two bracketing grid points. This answers: "once the
  adaptive method has converged, is it better per-interaction than the
  static rule?"
- **Cumulative crossover** (criterion=`"cumulative"`): where TOTAL
  accumulated reward (summed over every episode from 0) overtakes
  `static_blind`'s and stays there. This is exactly
  `recovery.metrics.parity.time_to_parity` (Task 5), computed once per
  `(method, env_id, classifier_id, seed)` from the full `T_max`
  trajectory at per-episode granularity (not interpolated on the coarse
  horizon grid). This answers: "has the adaptive method paid back its
  early exploration deficit yet?"

### Why they differ, and why a q<1 static rule must still eventually lose asymptotically

Once an adaptive method's per-episode regret has converged below the
static rule's fixed gap (`q_adaptive > q_static` in the late-stage
sense), the ASYMPTOTIC crossover has already happened -- by definition.
But the CUMULATIVE crossover requires paying back whatever deficit was
accumulated during the (typically worse-than-static) exploration phase
before that point. A static rule with success rate `q < 1` (i.e. any
rule that isn't the argmax action for every type) is guaranteed to be
overtaken asymptotically: the adaptive method's per-episode regret
relative to the oracle shrinks toward zero as `T -> infinity` (standard
bandit regret bounds), while the static rule's per-episode gap to the
oracle stays fixed forever. Summed to infinity, a strictly smaller
per-episode gap eventually wins -- this is a mathematical certainty, not
an empirical finding.

What is NOT guaranteed is that the cumulative crossover happens within
any horizon a real deployment would ever run for. The adaptive method's
early deficit is paid back at a rate proportional to its LATE per-episode
advantage, which can be small; if the static rule's gap to the adaptive
method's ceiling is narrow, or the exploration deficit was large, the
break-even point can sit at `T = 10^7` or beyond even though the
asymptotic crossover already happened at `T = 3,000`. This is exactly
why both numbers are reported, and why `horizon_analysis.py` prints their
ratio: a large cumulative/asymptotic ratio is direct evidence that
"eventually better" and "worth deploying" are different claims for that
specific method/environment.

## How to interpret `inf` crossovers

Both `crossover_horizon` (single seed) and `crossover_distribution`
(across seeds) can report `math.inf` -- "never crosses within the tested
horizon," not "never crosses, period." Two policies never produce `inf`
by construction and are worth knowing as reference points:

- `naive_retry` and `uniform_random`: these do not adapt at all in a way
  that improves on a fixed rule for this environment, so an asymptotic OR
  cumulative crossover against `static_blind` may genuinely never occur
  even in the limit -- `inf` here can be a real (not just
  horizon-truncated) answer.
- The `q=7/7` validation cell (`tests/test_horizon.py`'s
  `test_q7_of_7_static_rule_never_crossed_by_thompson_within_1e5`): a
  static rule constructed to play the exact per-type-optimal action
  (`MATRIX.argmax(axis=1)`) achieves `oracle_rate(MATRIX)` exactly, the
  hard ceiling every policy is bounded by (`recovery.theory.analytic.oracle_rate`'s
  docstring) -- `inf` here is the ONLY correct answer, for any adaptive
  method, at any horizon, because there is nothing above the ceiling to
  cross into.

`crossover_distribution` never reports a mean over these values, and
never silently drops them: it reports `fraction_never_crossed` alongside
`median`/IQR/min/max computed over the finite entries only (see
[`statistics.md`](../statistics.md), "Time-to-parity and why infinite
values must always be reported" -- the same reasoning applies here
verbatim). A method with `fraction_never_crossed=0.4` and a finite median
of 2,000 is a very different result from one with
`fraction_never_crossed=0.9` and the same finite median -- collapsing
either down to a single number that hides the censoring would be
actively misleading. `horizon_analysis.py`'s printed summary always shows
both.

## Kernel throughput

Task 6 profiled `run_episodes` (a prerequisite for a `T_max=100,000`
sweep being tractable at all) and found it running at ~10,100
episodes/second -- below the 15,000/second target. Profiling
(`cProfile`) attributed the gap to two things, both fixed without
changing any observable behavior (`tests/test_horizon.py` and the
existing Task 2 suite both verify this, including an explicit
bit-for-bit regression fixture against the pre-optimization
implementation):

1. **Per-episode `Context` allocation.** `Context` was a
   `@dataclass(frozen=True)`; it is now a `typing.NamedTuple` -- both are
   immutable and support the same attribute access every policy already
   uses, but a `NamedTuple` avoids a dataclass instance's `__dict__` and
   heavier `__init__`, and one `Context` is constructed per episode.
2. **Local variable binding.** `env.sample_reward`, `policy.select`,
   `classifier.predict`, and the four RNG streams were re-looked-up via
   attribute access on every one of `n_episodes` iterations; binding them
   to local variables once, before the loop, removes that repeated
   lookup cost (a standard CPython optimization -- local variable access
   is faster than attribute access).
3. A minor addition: when `type_prior` is `None` (the common, uniform
   case), `streams.types.integers(0, n_types)` replaces
   `streams.types.choice(n_types, p=None)` -- verified to produce a
   BIT-IDENTICAL draw sequence for the same seed (both ultimately
   dispatch to the same integer-generation primitive), while avoiding
   `numpy.random.Generator.choice`'s internal shape-introspection
   overhead (`numpy.prod` on the array form), which profiling showed
   consuming roughly a third of total runtime on its own.

Combined, these bring throughput to ~29,600 episodes/second on the
profiling machine -- see `recovery/core/episode.py`, `recovery/core/log.py`,
and `recovery/core/protocols.py`'s `Context` for the changes themselves.

## The exact command to regenerate the sweep

```
python experiments/horizon_sweep.py --estimate         # print an estimated wall-clock and exit
python experiments/horizon_sweep.py                     # full sweep, all defaults
python experiments/horizon_analysis.py                   # crossover table + stdout summary
```

Defaults match this document's grid: all 11 methods (10 adapted policies
plus `oracle` as a normalizer), both classifier conditions (`perfect`,
`real`), the canonical matrix plus 25 sampled matrices at
optimal-arm gap 0.20, 30 seeds, `T_max=100,000`. Any dimension can be
overridden (`--methods`, `--classifiers`, `--n-seeds`, `--t-max`,
`--horizons`, `--n-gap-matrices`, `--no-canonical`, `--n-jobs`) --
useful for a smaller validation run, since the full default grid
(particularly the `real` classifier condition, at roughly 2ms/inference
call -- see `recovery.classify.adapter.TrainedClassifierAdapter`'s
docstring -- multiplied across `T_max=100,000`) is a multi-hour
undertaking even at 8-way parallelism. `--smoke` (`T_max=2,000`, 3 seeds,
3 methods, canonical + perfect classifier only) runs in well under 60
seconds and exercises the full pipeline -- grid construction, `run_grid`,
prefix evaluation, and output schema -- for CI/quick iteration.

`run_grid` writes to a content-addressed `RunStore`
(`results/runs/horizon_sweep/` by default): re-running the same command
after an interruption skips every cell already on disk, so a partial run
resumes rather than restarting from scratch.
