# Shared Simulation Kernel Architecture

This document describes the kernel introduced in Task 2:
`recovery/core/` (protocols, RNG, logging, the episode loop),
`recovery/envs/matrix_env.py`, `recovery/classify/`, and
`recovery/policies/adapters.py`. It builds on Task 1's independent analytic
layer (`recovery/theory/analytic.py`, `recovery/envs/canonical.py`) and the
Table VII investigation (`docs/anomalies.md`), which found no currently
reproducible bug in the live simulator but could not fully explain a
historical archived result. This task is an architectural improvement, not
a bug fix: it establishes one shared, reproducible episode loop that every
experiment (existing and future) runs through, wrapping existing selector
and classifier code rather than rewriting it.

## The four protocols

Defined in `recovery/core/protocols.py`, as `typing.Protocol` classes
(structural typing -- implementations satisfy them by having the right
methods, not by inheriting from them):

- **`Context`** (frozen dataclass): the per-episode observation handed to a
  policy. Fields: `predicted_type`, `tool_id`, `retry_count`, `episode`.
  It has no `true_type` field -- a policy that wanted to see the true type
  would have nowhere to read it from. Being `frozen=True` means attempting
  to mutate it raises `dataclasses.FrozenInstanceError` at runtime, not
  just "please don't" in a docstring.
- **`Policy`**: `name: str`, `select(ctx, rng) -> int`,
  `update(ctx, action, reward) -> None`, `reset() -> None`.
- **`Environment`**: `n_types: int`, `n_strategies: int`,
  `success_matrix() -> ndarray` (for analytic cross-checks only),
  `sample_reward(true_type, action, rng) -> int`, `step(episode) -> None`
  (a no-op drift hook for stationary environments; a future non-stationary
  environment would mutate its own internal matrix here).
- **`Classifier`**: `predict(true_type, rng) -> int`.

## The episode lifecycle

`recovery/core/episode.py::run_episodes` is the one shared loop. Every
episode executes these eight steps, in this exact order:

```
1. true_type  = streams.types.choice(env.n_types, p=type_prior)
2. predicted  = classifier.predict(true_type, streams.classifier)
3. ctx        = Context(predicted_type=predicted, episode=i)
4. action     = policy.select(ctx, streams.policy)
5. reward     = env.sample_reward(true_type, action, streams.reward)
6. policy.update(ctx, action, reward)
7. env.step(i)
8. log.record(i, true_type, predicted, action, reward)
```

`policy.reset()` is called exactly once, before the loop starts. Actions
are validated against `[0, env.n_strategies)` before being handed to
`sample_reward`; `type_prior` (if given) is validated for shape and that
it sums to 1 (atol 1e-9); `n_episodes < 1` raises before any RNG streams
are even constructed.

Note step 1 -> step 5: `true_type` is a local Python variable read
directly from step 1 into step 5. It is never written into `Context`
(step 3 only stores `predicted`), so there is no path by which step 4's
policy, or anything downstream of it, can see or influence which row of
the matrix step 5 reads from. See "Preventing Reward-Path Errors by
Design" below.

## RNG stream design and Common Random Numbers

`recovery/core/rng.py::from_seed(seed)` expands one integer seed into four
*independent* `numpy.random.Generator` streams via
`numpy.random.SeedSequence(seed).spawn(4)`:

| Stream | Drives |
|---|---|
| `types` | which true failure type occurs each episode |
| `classifier` | the classifier's prediction (if stochastic) |
| `reward` | the environment's Bernoulli reward draw |
| `policy` | a (native) policy's action selection / exploration |

`SeedSequence.spawn` derives statistically independent child seeds from
one parent -- not four offsets into one shared stream -- so each
`Generator` has fully separate internal state. This buys Common Random
Numbers (CRN), a standard simulation variance-reduction technique:

- **Cross-policy CRN**: at a fixed seed, the sequence of true types
  (`streams.types`) is identical no matter which policy is plugged in,
  because a policy only ever touches `streams.policy`. Two policies
  compared at the same seed face an *identical* sequence of environment
  conditions, so a measured difference in outcome is attributable to the
  policy, not to one of them happening to get an easier draw of failure
  types. `tests/test_episode.py::test_stream_independence_true_type_sequence_unaffected_by_policy`
  checks this directly.
- **Cross-classifier CRN**: swapping which `Classifier` is plugged in does
  not change `streams.types` or `streams.reward` -- only the *observed*
  label changes. This is what makes the confusion-noise integration test
  (`tests/test_kernel_vs_analytic.py`) a clean like-for-like comparison
  against the noiseless one.

**Caveat (documented, not silently assumed away):** `streams.policy` is
correctly used by the two *native* policies (`StaticRulePolicy`,
`OraclePolicy`), but every *legacy-wrapped* selector
(`recovery/policies/adapters.py::LegacySelectorAdapter` and everything
built on it -- UCB1, epsilon-greedy, decaying epsilon-greedy, Thompson,
Discounted Thompson, LinUCB, Contextual Cost-Aware, Uniform Random,
One-Shot LLM) owns its own internal `random.Random`, seeded once at
construction, predating this kernel. `LegacySelectorAdapter.select`
accepts `rng` (to satisfy the `Policy` protocol) but ignores it. CRN
*across two different legacy-wrapped policies'* own exploration noise
therefore requires giving them matching `seed=` constructor arguments,
not relying on `streams.policy` -- rewriting their internals to accept an
externally supplied generator was out of scope ("wrap, don't rewrite").
This does not weaken CRN for the environment/classifier streams, which
every policy (native or wrapped) is equally blind to.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `recovery/core/protocols.py` | The four protocols. No concrete logic. |
| `recovery/core/rng.py` | `SeedStreams`, `from_seed`. |
| `recovery/core/log.py` | `EpisodeLog`: preallocated columnar arrays, `to_dataframe`, `success_rate`, `per_type_rate`. |
| `recovery/core/episode.py` | `run_episodes`: the one shared loop. |
| `recovery/envs/matrix_env.py` | `MatrixEnvironment`: an (N,K)-matrix-backed `Environment`. |
| `recovery/classify/synthetic.py` | `PerfectClassifier`, `ConfusionClassifier`: synthetic `Classifier`s for controlled-noise experiments. |
| `recovery/classify/adapter.py` | `TrainedClassifierAdapter`: wraps the real, already-trained `FailureClassifier` (text-in, type-out) behind the `Classifier` protocol. |
| `recovery/policies/adapters.py` | `LegacySelectorAdapter` (generic wrapper) + one named factory per existing selector, plus the native `StaticRulePolicy`/`OraclePolicy`. |
| `recovery/guards/invariants.py` | `InvariantViolation` + six runtime guard functions + `run_all_guards`: post-hoc scientific assertions over a completed simulation's results. |

`recovery/core/` depends on nothing outside the standard library, numpy,
and pandas -- specifically never on `recovery/policies/`, `recovery/envs/`,
or `recovery/classify/` (verified: `grep` for those import paths under
`recovery/core/` returns nothing). `recovery/theory/analytic.py` is
unchanged from Task 1 and remains numpy-only; no simulation module
(old or new) imports it, preserving its role as an independent reference.

## Preventing Reward-Path Errors by Design

The historical concern (`docs/anomalies.md`) was that reward could end up
sampled using a PREDICTED failure type instead of the TRUE one -- Task 1's
diagnostics did not find this bug currently reproducible in the live
simulator, but could not rule out that it existed at some point, or could
be reintroduced by a future refactor. Rather than "fix" specific arithmetic
(there was nothing currently broken to fix), this task makes the entire
bug *class* structurally unrepresentable:

1. **`Environment.sample_reward(true_type, action, rng)` has no parameter
   for a predicted type.** There is no argument slot to accidentally pass
   the wrong thing into -- the function signature itself only accepts a
   true type. Compare this to the pre-kernel harness's
   `_simulate_episode`, where `obs_type` and `failure_type` were two local
   variables of the same type that a future edit could swap by accident
   (see `docs/repo_inventory.md` item 5) -- that was possible *because*
   both flowed through the same function scope with similar names. Here,
   `sample_reward` structurally cannot see a predicted type at all: it is
   not in scope.
2. **`Context` -- the only thing a `Classifier`'s output feeds into --
   carries `predicted_type` and nothing that could stand in for a true
   type.** A policy cannot leak a true type into its action even if it
   wanted to, because it was never given one.
3. **`run_episodes`'s local variable `true_type` (step 1) is read directly
   into `sample_reward` (step 5), five lines away, entirely bypassing
   `Context`.** To reintroduce the bug, a future contributor would have to
   deliberately pass `predicted` (or `ctx.predicted_type`) into
   `env.sample_reward` in place of `true_type` -- an active, visible edit
   to the one shared loop, not an easy-to-miss typo, and not something any
   individual `Environment`, `Policy`, or `Classifier` implementation
   could cause on its own (they don't control the order steps happen in,
   or which variable is threaded into which call).
4. **The regression test exists and is deliberately hard to defeat by
   accident.** `tests/test_episode.py::test_reward_path_guard_reward_uses_true_type_not_predicted_type`
   builds a maximally-adversarial setup: an asymmetric 2x2 matrix
   (`P = [[0.9, 0.1], [0.1, 0.9]]`) and a classifier that ALWAYS predicts
   type 0. Because the policy under test is a constant function of the
   predicted type, the ACTION is identical every episode -- so this
   isolates exactly one variable: which row `sample_reward` actually
   reads from. The two hypotheses produce very different, easily
   distinguishable numbers (0.5 if reward is correctly keyed on the true
   type; 0.9 if it were keyed on the predicted type), so 100,000 episodes
   gives a test with enormous statistical power (~6 standard errors of
   separation) to catch a regression, not a coin flip. This test would
   have failed loudly had the historical bug hypothesis (docs/anomalies.md,
   Hypothesis A) been the actual explanation for Table VII.

Together, these four points hold regardless of whether the historical bug
ever existed in this exact form: the architecture removes the *capacity*
to express it, not just today's instance of it.

## Guard layer

Task 3 (`recovery/guards/invariants.py`) adds a complementary,
independent layer: where the previous section makes one class of bug
structurally impossible to *write*, the guard layer catches whatever
still slips through -- any bug, in any experiment, old or new -- by
checking a completed run's results against the same analytic bounds the
rest of this architecture is built on.

### Dependency graph

```
recovery/theory/analytic.py  (numpy only; Task 1; unchanged)
            |
            v
recovery/guards/invariants.py  (numpy + recovery.theory.analytic; nothing else)
```

`recovery/guards/` imports only `numpy` and `recovery.theory.analytic` --
never `recovery.core`, `recovery.envs`, `recovery.policies`, or
`recovery.classify`. This is deliberate and verified (`grep` for those
import paths under `recovery/guards/` returns nothing): guards must be
able to check a kernel run's output without depending on the kernel
itself, so that a bug *in the kernel* can never simultaneously corrupt
the thing checking the kernel's output. The theory layer -> guard layer
arrow is one-directional, matching the theory layer -> (nothing) rule
Task 1 established: `analytic.py` has no callers inside `recovery/`
other than guards and tests, preserving its role as an independent
reference implementation neither the simulator nor the guards can
accidentally entangle with simulation state.

### Guards never depend on the kernel

Concretely, this means a guard call is just a function of numbers you
already have once a run finishes -- `run_all_guards(rate, P, n_episodes,
pi=..., C=..., normalized=...)` -- not something that needs an
`EpisodeLog`, a `Policy`, or an `Environment` instance passed in. Task 4
can integrate this with a single call after any experiment (existing or
new) completes, without the guard module ever importing
`recovery.core.episode` or anything it produces. This task deliberately
does not add that call anywhere yet (see `docs/anomalies.md`,
"Automated Prevention" -- integration is scoped to Task 4).

### Guards raise instead of warn

Every guard is scoped as a scientific assertion, not a lint warning:
on violation it raises `InvariantViolation` immediately, and never
prints, logs, or continues. The distinction from `ValueError` is
deliberate and enforced throughout `recovery/guards/invariants.py`:
malformed input (bad shapes, `n_episodes < 1`, non-positive `sigmas`,
an out-of-range policy index) is always a `ValueError` -- a programming
error, caught before any statistical comparison is attempted -- while
`InvariantViolation` is reserved for the case where the *simulation
result itself* is inconsistent with what the analytic layer says is
possible. Six guards cover the full picture: a hard `[0, 1]` bound on
any rate; upper/lower statistical bounds against `oracle_rate` and a
newly-derived `worst_case` floor (applicable to every policy, fixed or
adaptive); a fixed-policy-only ceiling against `policy_rate`; a two-sided
check against `policy_rate_under_confusion`; and a hard sanity range for
already-computed normalized scores. All statistical bounds use
`slack = sigmas * sqrt(0.25 / n_episodes)`, so tolerance shrinks as an
experiment's episode count grows -- see `recovery/guards/invariants.py`'s
module docstring and `docs/anomalies.md`'s "Automated Prevention" section
for why this, and not a fixed epsilon, is the statistically correct
model, and for the specific episode-count design decision the Table VII
regression test required.
