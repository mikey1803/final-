# Statistics: What This Repository's Numbers Do and Don't Mean

This document explains the reasoning behind Task 5's metrics layer
(`backend/recovery/metrics/`) and, more importantly, the limits of what
its numbers can be used to claim. It complements
[`docs/architecture.md`](architecture.md) (how results are produced) and
[`docs/reconciliation_stationary.md`](reconciliation_stationary.md) (a
worked example of the Monte Carlo reasoning below being applied to real
numbers).

## Why p-values computed over simulation seeds are not inferential

A p-value is only meaningful relative to a stated null hypothesis and a
sampling process from some population. In this repository, "more samples"
means "more simulation seeds" -- and every seed is drawn from the SAME
fixed, fully-known process: the same success matrix `P`, the same policy
logic, the same classifier. There is no population of real-world agent
deployments being sampled from; there is one deterministic-given-its-seed
program being re-run.

This matters because, in that setting, a p-value's behavior is
degenerate: given two policies with ANY nonzero true difference in
expected reward, running enough seeds drives the p-value for "these two
policies differ" toward zero, even if the true difference is
practically irrelevant (say, 0.0001 success rate). The p-value answers
"is this difference distinguishable from Monte Carlo noise at this seed
count," which is a question about how many seeds were run, not about
whether the difference matters. `src/evaluation/stats_utils.py`'s
`holm_bonferroni` (and any `p_value`/`significant` columns computed
upstream of it, e.g. in `src/evaluation/phase3c_stats_upgrade.py`) are
therefore **deprecated for new work** as of this document -- left
unmodified (Task 5 does not touch existing files), but callers writing
new experiments should use this package's effect sizes
(`recovery/metrics/bootstrap.py::cohens_d`, `cliffs_delta`, `magnitude`)
and bootstrap intervals instead, which report a MAGNITUDE with a
confidence range rather than a binary, sample-size-inflatable
significant/not-significant verdict. `recovery/metrics/` implements no
p-values, t-tests, or Holm correction at all (see its `bootstrap.py`
module docstring).

## Why bootstrap confidence intervals quantify Monte Carlo uncertainty only

`recovery/metrics/bootstrap.py::bootstrap_ci` resamples (with replacement)
from the values it is given and reports the spread of a statistic across
those resamples. What that spread reflects depends entirely on what the
input values ARE:

- Bootstrapping across **per-seed** results (e.g. one late-stage rate per
  simulation seed) quantifies how much the reported number would move if
  a different set of seeds had been drawn -- purely an artifact of this
  repository's own RNG, not of anything in the real world.
- Bootstrapping across **per-episode** results within a single run (as
  `recovery/metrics/summary.py::summarize` does over one log's late-stage
  reward window) quantifies uncertainty from that one run's finite
  episode count.

Neither is a claim about sampling variability in a population of real
agent deployments -- there is no such population being sampled here. This
is the same caveat `src/evaluation/stats_utils.py::MC_VARIATION_NOTE`
already states for the pre-Task-5 harness; `recovery/metrics/` widens its
scope (per-episode as well as per-seed) but the underlying limitation is
identical: a bootstrap CI here answers "how much would this specific
number move if I reran the simulation," never "how much would this move
if I redeployed the real system."

## Why cumulative regret and late-stage success answer different questions

`late_stage_rate` (mean reward over the final `late_stage_frac` of a run)
answers: **once a policy has finished learning, how good is it?** It is
an asymptotic, steady-state metric -- exactly what the paper's tables
report, and exactly what `recovery.theory.analytic` validates against.

`cumulative_regret` (`recovery/metrics/regret.py`) answers a different
question: **what did it cost, in total, to get there?** A policy that
converges to an excellent late-stage rate but spends its first 200
episodes exploring badly has paid a real, one-time cost that late-stage
success is structurally blind to (the window doesn't even include those
episodes). In an actual deployment, every one of those early episodes is
a real user experiencing a real failure -- the exploration cost is not
academic. Two policies can have identical late-stage rates and wildly
different cumulative regret; only the second metric would show that.

## Why a static rule with `q < 1` must eventually lose asymptotically, while the cumulative crossover may lie beyond any realistic horizon

Consider a static (non-learning) rule whose late-stage success rate is
some `q` strictly below what an adaptive policy will eventually converge
to. Per-episode, once the adaptive policy has converged, it incurs
strictly less instantaneous regret than the static rule, episode after
episode, forever. Summed to infinity, that per-episode gap makes the
adaptive policy's cumulative reward overtake the static rule's
**eventually, with mathematical certainty** -- this is `time_to_parity`
(`recovery/metrics/parity.py`) converging as the horizon grows, and it is
guaranteed by nothing more than "a strictly positive per-episode gap,
summed forever, diverges."

But "eventually" carries no information about magnitude. The adaptive
policy typically pays exploration cost UP FRONT (worse-than-static
performance during learning) before its per-episode advantage kicks in.
The crossover episode is the point where the accumulated early deficit is
paid back by the accumulated later surplus -- and depending on how large
that deficit was and how small the later per-episode advantage is, that
point can be episode 40, or it can be an episode number far past any
horizon a real deployment would ever run for. A mathematical guarantee of
eventual superiority is not evidence of practical superiority within a
deployment's actual lifetime -- which is exactly why
`recovery/metrics/parity.py::time_to_parity` (and `summarize_parity`'s
`fraction_never_crossed`) exist as a metric distinct from the
asymptotic guarantee itself: they measure whether parity happens within
the horizon actually tested, not whether it happens in principle.

## Time-to-parity and why infinite values must always be reported

`time_to_parity` returns `math.inf` when the learner never overtakes the
baseline (and stays overtaken) within the tested horizon. It would be easy
to compute a summary by silently filtering these out and averaging the
rest -- but doing so introduces a **survivorship bias**: if parity is hard
to reach, the runs that DID reach it are, by construction, an
unrepresentative, optimistic subset of all runs, and reporting only their
average manufactures a rosier picture than the full experiment supports.

`summarize_parity` (`recovery/metrics/parity.py`) instead always reports
`fraction_never_crossed` alongside `median_finite`, so a reader sees both
"how fast, when it happens" and "how often it doesn't happen at all
within this horizon" -- two numbers that, together, are honest about what
was actually observed; either one alone can mislead.
