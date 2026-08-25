# Pre-registration -- Blind-Design Elicitation Study

Frozen at `PROTOCOL_VERSION = "1.0.0"` (see [`version.py`](version.py)).
This document exists to be written BEFORE data collection and left
unedited afterward -- if a later protocol version changes any of the
below, that is a NEW pre-registration under a new protocol version, not
an edit to this one. Any post-hoc deviation from what is frozen here
must be disclosed as such in `docs/experiments/human_study.md`, not
folded silently into "the plan."

## 1. Hypotheses / research questions

- **RQ1**: What is the distribution of hand-designed recovery-table
  quality `q` (count of correct rows out of 7) across independent, blind
  human designers?
- **RQ2**: Do LLM designers, given the identical blind protocol, achieve
  higher `q` than human designers?
- **RQ3**: Is the paper's "second-best" error hypothesis (a plausible
  human/LLM mistake picks the SECOND-best strategy for a failure type,
  not an arbitrary one) supported by the data?
- **RQ4**: At realistic deployment budgets, what fraction of real
  (human) designers' tables would still be preferred over an adaptive
  bandit, per Task 10's calibrated crossover predictor?

No other hypotheses are tested; any other reported statistic in
`docs/experiments/human_study.md` is explicitly labeled EXPLORATORY, not
confirmatory.

## 2. Scoring rule (frozen)

```
q_count(mapping, P) = |{ t : mapping[t] achieves max_k P[t, k] }|
q_ratio             = q_count / N
```

Implemented in `recovery.analysis.human_study.score_submission`. `P` is
`recovery.envs.canonical.MATRIX`, unchanged from Tasks 1-10. Ties (a
non-chosen strategy achieving the exact same value as the chosen one)
count as correct -- "achieves the max," not "is the unique argmax."

## 3. Inclusion / exclusion criteria (frozen)

- **Include**: age 18+, any engineering background, completed the full
  task (all 7 rows answered) and confirmed the Lock & Submit screen.
- **Exclude**: any prior exposure to `MATRIX`/`BLIND_RULE`/this project's
  results (self-reported, unverifiable -- disclosed as a limitation, not
  silently assumed away); incomplete submissions (impossible to record
  by construction, see `study/protocol.md`).
- No exclusion based on background category (experience, LLM
  familiarity, agent-building history) -- these are measured, not
  screened on.

## 4. Sample size

Target `n` in the 10-30 range (opportunistic/convenience recruitment;
see `study/protocol.md`'s recruitment section). No stopping rule beyond
"recruitment window closes" -- this is explicitly NOT an
optional-stopping / peeking-at-significance design (this project reports
no p-values anywhere, per `docs/statistics.md`, so there is no
significance threshold to peek at). Collection stops at the recruitment
deadline recorded in `docs/experiments/human_study.md`, whatever `n`
that yields.

## 5. Analysis plan (frozen)

Run, in this order, on whatever `n` was actually collected -- no
analysis is contingent on the value of another (e.g. the LLM comparison
runs regardless of what RQ1's distribution looks like):

1. `recovery.analysis.human_study.study_summary(submissions, MATRIX)` --
   RQ1: mean/median/min/max/histogram of `q`, bootstrap 95% CI on the
   mean (reported as undefined, not fabricated, if `n < 2`).
2. `recovery.analysis.human_study.second_best_error_test(scored_submissions)`
   -- RQ3: mean rank of the chosen strategy among INCORRECT rows, and the
   fraction that is EXACTLY rank 2, with a bootstrap CI.
3. `experiments/llm_designers.py` against the model families in
   `conf/llm_designers.yaml`, then
   `recovery.analysis.human_study.compare_human_llm_q` -- RQ2: Cliff's
   delta with a bootstrap CI (never a p-value).
4. `experiments/study_analysis.py`'s crossover-budget analysis -- RQ4:
   for each budget `T` in `{100, 1_000, 10_000, 100_000}` episodes,
   the fraction of REAL (human) submissions whose predicted crossover
   `T*` (via `recovery.theory.crossover.predict_crossover`, using Task
   10's already-calibrated `c` -- never refit here) exceeds `T` (i.e.
   the static table is still ahead of the bandit at that budget).
5. Bias/agreement analyses (Fleiss' kappa, pairwise agreement, per-type
   entropy, position bias, confidence calibration, time-vs-`q`,
   experience-vs-`q`) -- all EXPLORATORY (see Section 1), reported for
   transparency, not as tests of a pre-registered hypothesis.

No statistic anywhere in this plan is a p-value or a significance test
(project-wide convention, `docs/statistics.md`) -- every comparison is
reported as an effect size (Cliff's delta, correlation coefficient, mean
difference) with a bootstrap confidence interval quantifying resampling
uncertainty over the finite collected sample, explicitly NOT inferential
uncertainty about a hypothetical population of all possible designers.

## 6. What would change this pre-registration

Any of: a wording change to Table I/II's descriptions, a change to the
scoring rule, a change to inclusion/exclusion criteria, or a change to
the analysis plan above. Any such change requires a new
`PROTOCOL_VERSION` and a new pre-registration (this file is never edited
in place after the study version it documents has begun collecting
data) -- see `study/README.md`'s institutional-approval re-submission
rule, which applies identically here.
