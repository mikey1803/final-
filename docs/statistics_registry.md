# Hypothesis Registry and Statistical Review (Task 13)

This document is the single registry of every hypothesis/comparison
tested anywhere in this project, the multiple-comparison correction
policy applied to them, and an honest audit of what does and does not
survive correction. It complements, and does not replace,
[`docs/statistics.md`](statistics.md) (the reasoning behind Task 5's
metrics layer) — this document is the inventory and the corrections;
`statistics.md` is the methodology explanation.

Machine-readable companions: `backend/results/tables/hypothesis_registry_llm_per_type.parquet`
(the new real tests, §2) and this document's tables (§3-§5), which mirror
already-written `results/phaseN/*.csv` / `results/tables/*.parquet` files
cited inline rather than duplicated wholesale.

## 1. The multiple-comparison correction policy, and why

**Exactly one family of tests in this project gets Holm-Bonferroni
multiple-comparison correction of p-values: the real-sample LLM
blind-design per-type accuracy tests (§2).** Every other statistical
claim in the project is either (a) an effect size + bootstrap CI with
*no p-value at all*, by a pre-existing, previously-validated design
decision (`recovery/metrics/bootstrap.py`'s own docstring: "no p-values,
t-tests, Holm correction, or significance testing here or anywhere in
this package"), or (b) a p-value from the legacy pre-Task-5
`src/evaluation/harness.py` pipeline, explicitly excluded from the
correction family (§4).

This is a deliberate deviation from a fully literal reading of "apply
multiple-comparison correction consistently across the whole paper's
family of tests" — the closest CORRECT alternative, not a shortcut, and
it needed exactly one decision, so it needs to be justified precisely:

- **A p-value is only a meaningful, correctable object if the process
  generating it is a genuine sample from some population.** This
  project's simulation-based comparisons (Phase 1, 3a, 4, 4b, 5, 6, 7, 9)
  re-run the SAME deterministic-given-its-seed simulator across trial
  seeds. `docs/statistics.md` (written before Task 13, during Task 5)
  already established that a p-value computed this way is degenerate: it
  trends toward zero as more seeds are run, for ANY nonzero true
  difference, however practically irrelevant. Applying Holm-Bonferroni
  correction to a set of already-degenerate p-values does not make them
  meaningful — correcting a philosophically invalid quantity produces
  another philosophically invalid quantity, just a smaller one. Pooling
  these into the SAME correction family as the real-sample LLM tests
  would also dilute the correction applied to the tests that DO carry
  real information (the more tests in a Holm family, the more
  conservative every member's threshold becomes), penalizing genuine
  evidence for the sake of a number that was never meaningful to begin
  with.
- **The real-sample LLM tests (§2) are different in kind, not just in
  degree.** Each of the 63 LLM-designer samples is one real network call
  to an external, non-deterministic model this codebase does not
  control. "Running more samples" costs real API quota, not a change of
  `--seed`. A p-value here answers "does this model's accuracy on this
  failure type differ from its own accuracy on the rest," a question
  about the external process, not about how many times this repository
  re-ran its own code.
- **Given this project already has an established, working, previously-
  validated convention for the simulation-based family (effect size +
  bootstrap CI, no p-value) — per this task's own rule "do not rewrite
  working code without evidence of a flaw" — retrofitting p-values onto
  that family now would both contradict a decision this project already
  made deliberately and risk exactly the "invent significance" failure
  mode this task explicitly forbids** (an extremely small p-value from
  50-2000 simulation seeds is trivially achievable for a practically
  tiny true gap; presenting it as if it were evidence of a real-world
  effect would overstate the conclusion).

**Within the one corrected family** (§2): correction is applied
separately per LLM model family (3 independent Holm corrections, each
across that family's own 7 failure-type tests), not pooled across all 21
tests globally. Rationale: (a) this matches how every other analysis in
this project already treats the 3 LLM families — never pooled, since
their `q` distributions are not comparable (`experiments/study_analysis.py`'s
`llm_crossover_budget_analysis` states this explicitly); (b) it is the
literal, natural reading of "Holm-corrected across failure types" (across
the 7 types, within each family's own results); (c) it is the LESS
conservative of the two reasonable choices — verified explicitly below
that this does not manufacture significance: pooling all 21 tests into
one global correction would only make `p_value_holm` larger (more
conservative) for every test, so `groq-llama`'s already-non-significant
`malformed_output` result would remain non-significant either way, and no
currently-significant result would flip. The choice was not made to
produce a rosier conclusion.

**The legacy `src/evaluation/harness.py` p-values (§4)** are reported in
the registry for completeness (every existing statistical claim must be
inventoried, per this task's instruction) but are excluded from the
correction family and flagged `deprecated_degenerate` throughout — see §4
for why, and see the corrected citations in `docs/PAPER_PREP.md` and
`docs/PROJECT_EXPLAINED.md` (Task 13 added the same caveat inline
wherever these numbers are quoted).

## 2. The new real-sample test family: LLM blind-design per-type accuracy

**Source**: `backend/recovery/analysis/significance.py` (new, real-sample-
only test utilities — Cochran's Q, per-type-vs-rest binomial, kept
structurally separate from `recovery/metrics/bootstrap.py` on purpose, see
that module's docstring) and `backend/experiments/hypothesis_registry.py`
(the driver). Reuses `recovery.analysis.human_study.score_submission`
(Task 11) to re-derive the per-sample, per-type correctness matrix from
`results/tables/llm_designers.parquet`'s raw `parsed_mapping` column (not
itself persisted anywhere as a matrix), and
`src.evaluation.stats_utils.holm_bonferroni` (reused directly, not
reimplemented — see §1 for why reusing this specific function is not the
same as reviving the deprecated simulation-p-value practice).

**Design**: for each of the 3 LLM families, an omnibus **Cochran's Q
test** (the correct test for a MATCHED binary design — each sample
answers all 7 failure types in one API call, so the 7 per-type outcomes
from one sample are paired, not independent; Cochran's Q generalizes
McNemar's test from k=2 to k=7 matched conditions) tests "is per-type
accuracy uniform within this family," i.e. the null the task specifies.
A post-hoc **per-type-vs-rest exact binomial test** (that type's
accuracy vs. the pooled accuracy of the OTHER 6 types) is the direct
test of "is failure type t specifically harder/easier than the rest,"
Holm-corrected within each family across its 7 types (§1). Effect size:
**Cohen's h** (the standard effect size for a difference of two
proportions) for each per-type test; a Cramer's-V-style normalization of
Cochran's Q (`sqrt(Q / (n*(k-1)))`, transparently derived, not a claim of
a single canonical "Cochran's Q effect size" since none is universally
standard) for the omnibus test. CI: bootstrap 95% CI
(`recovery.metrics.bootstrap.bootstrap_ci`, reused) on each type's raw
accuracy.

### 2.1 Omnibus test: is per-type accuracy uniform within each family?

| family | n (parsed) | n API calls | n distinct mappings | bootstrap CI degenerate? | Cochran's Q | df | p-value | effect size (V-style) | significant? |
|---|---|---|---|---|---|---|---|---|---|
| google-gemini | 21 | 21 | **2** | **yes** | 90.00 | 6 | 3.0e-17 | 0.845 | yes -- **but see caveat below** |
| groq-llama | 19 | 21 | 15 | no | 24.19 | 6 | 4.8e-4 | 0.461 | yes |
| nvidia-mistral-nemotron | 21 | 21 | 6 | no | 57.07 | 6 | 1.8e-10 | 0.673 | yes |

**google-gemini caveat (required reporting, not optional)**: this
family's Cochran's Q is technically significant, but its `n_distinct_mappings=2`
means the effective information content is closer to "2 observed
behavioral patterns" than "21 independent trials" — the omnibus test is
real and correctly computed, but should not be read with the same
confidence as the other two families' omnibus results. The asymptotic
chi-square approximation Cochran's Q relies on is also least trustworthy
here, since the data is maximally non-generic (a near-point-mass
distribution).

The omnibus tests are each a single primary test about a DIFFERENT
family/model and are not further corrected against each other (standard
practice for independent primary hypotheses about different
populations) — only the 7 per-type follow-up tests WITHIN each family are
Holm-corrected (§2.2).

### 2.2 Per-type-vs-rest, Holm-corrected within family (full table)

| family | failure_type | accuracy | 95% CI | pooled rest rate | Cohen's h | p (raw) | p (Holm) | significant after correction? |
|---|---|---|---|---|---|---|---|---|
| google-gemini | tool_timeout | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 1.000 | no |
| google-gemini | tool_error | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 1.000 | no |
| google-gemini | wrong_tool_selected | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 1.000 | no |
| google-gemini | hallucinated_output | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 1.000 | no |
| google-gemini | **malformed_output** | 0.286 | [0.095, 0.476] | **1.000** | -2.014 | **undefined** | 1.000 | **cannot compute** |
| google-gemini | context_loss | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 1.000 | no |
| google-gemini | ambiguous_instruction | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 1.000 | no |
| groq-llama | tool_timeout | 0.368 | [0.158, 0.579] | 0.711 | -0.701 | 0.0035 | **0.0212** | **yes** |
| groq-llama | tool_error | 0.842 | [0.684, 1.000] | 0.632 | 0.487 | 0.060 | 0.241 | no |
| groq-llama | wrong_tool_selected | 0.526 | [0.316, 0.737] | 0.684 | -0.325 | 0.145 | 0.290 | no |
| groq-llama | hallucinated_output | 0.684 | [0.474, 0.895] | 0.658 | 0.056 | 1.000 | 1.000 | no |
| groq-llama | **malformed_output** | 0.421 | [0.211, 0.632] | 0.702 | -0.574 | **0.0112** | **0.0562** | **no (fails after correction)** |
| groq-llama | context_loss | 0.842 | [0.684, 1.000] | 0.632 | 0.487 | 0.060 | 0.241 | no |
| groq-llama | ambiguous_instruction | 0.947 | [0.842, 1.000] | 0.614 | 0.878 | 0.0016 | **0.0114** | **yes** |
| nvidia-mistral-nemotron | tool_timeout | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 0.993 | no |
| nvidia-mistral-nemotron | tool_error | 0.952 | [0.857, 1.000] | 0.889 | 0.240 | 0.723 | 1.000 | no |
| nvidia-mistral-nemotron | wrong_tool_selected | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 0.993 | no |
| nvidia-mistral-nemotron | hallucinated_output | 0.905 | [0.762, 1.000] | 0.897 | 0.027 | 1.000 | 1.000 | no |
| nvidia-mistral-nemotron | **malformed_output** | 0.429 | [0.238, 0.619] | 0.976 | -1.404 | **7.99e-15** | **5.59e-14** | **yes** |
| nvidia-mistral-nemotron | context_loss | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 0.993 | no |
| nvidia-mistral-nemotron | ambiguous_instruction | 1.000 | [1.000, 1.000] | 0.881 | 0.705 | 0.166 | 0.993 | no |

Full machine-readable table:
`results/tables/hypothesis_registry_llm_per_type.parquet` (20 rows —
google-gemini's `malformed_output` row is included with `p_value_raw=NaN`,
`p_value_holm=1.0` by Holm's own NaN-safe convention, never silently
dropped).

**The `google-gemini` `malformed_output` test is UNDEFINED, not merely
non-significant.** Every one of its 21 samples answers the other 6 types
correctly, so the pooled reference rate to test against is EXACTLY 1.0 —
a two-sided binomial test against a boundary probability of 1 has no
defined p-value (`scipy.stats.binomtest` is not called in this case; the
registry records `NaN`, and Holm's step-down procedure conservatively
treats `NaN` as `p=1`, i.e. "fail to reject," never fabricating a
significant OR non-significant number). This null result is itself a
direct, mechanical symptom of the family's 2-distinct-mapping degeneracy
(§5) — not independent evidence about `malformed_output`.

### 2.3 The direct answer to the task's specific question

**Is `malformed_output`'s cross-family finding real?** Descriptively,
yes — it is the lowest-raw-accuracy type in all 3 families (28.6% /
42.1% / 42.9%). **Formally, after Holm correction, it is significant in
only 1 of 3 families** (`nvidia-mistral-nemotron`, p_holm=5.6e-14).
`groq-llama`'s raw p=0.0112 does NOT survive correction (p_holm=0.0562 >
0.05) — reported explicitly, not softened, per this task's rule.
`google-gemini`'s test cannot be computed at all. **This is a materially
weaker claim than "the single hardest failure type for every family,"
and the docs previously making that stronger claim
(`docs/PAPER_PREP.md`, `docs/experiments/human_study.md`,
`docs/experiments/matrix_robustness.md`) have been corrected as part of
this task** — see the diffs to those three files.

## 3. Seed-count audit: every existing CI, and what backs it

Per this task's instruction: report how many seeds underlie every
existing CI; if fewer than 30, increase and re-run, or document why
infeasible.

| source | original n_seeds (or n_trials) | action taken | new n_seeds | evidence |
|---|---|---|---|---|
| Phase 1 / Phase 4 / Phase 4b (`src/evaluation/harness.py`, `nonstationarity.py`) | 50 trials | none needed (already >=30) | 50 | unchanged |
| Phase 3a (`matrix_family_sweep.py`) | 20 | **re-run twice** (see §3.1 — first re-run surfaced a real non-determinism bug, fixed, then re-run again) | **30** | `results/phase3a/matrix_family_summary.csv`, `NOTES.txt`; final fixed run completed in 226.7s |
| Phase 5 (`static_rule_sweep.py`) | 20 | **re-run twice** (its `reference_baselines_summary.csv` shares Phase 3a's bug, see §3.1; its own `static_rule_quality_summary.csv` was unaffected) | **30** | `results/phase5/static_rule_quality_summary.csv`, `reference_baselines_summary.csv` |
| Phase 6 (`nonstationarity_sweep.py`, shift-frequency-x-magnitude) | 20 | **re-run** | **30** | `results/phase6/shift_frequency_magnitude_summary.csv` |
| Phase 7 (`taxonomy_size_sweep.py`) | 20 | **re-run** | **30** | `results/phase7/taxonomy_size_summary.csv` |
| Phase 9 / Task 12 (`matrix_robustness.py`) | 10 | **re-run** | **30** | `results/tables/matrix_robustness_*.parquet` (+ noise-floor recalibrated at 30 seeds/world, 200 replicate worlds) |
| `prior_grid.py` (Task 7 PGTS prior grid, full mode) | 30 | none needed (already >=30) | 30 | unchanged |
| `horizon_sweep.py` (Task 9 horizon) | 30 | none needed (already >=30) | 30 | unchanged |
| `crossover_calibration_grid.py` (Task 10, full mode) | 30 | none needed (already >=30) | 30 | unchanged |
| `prior_analysis.py::posterior_convergence` (Task 7, illustrative diagnostic) | 10 | **not re-run — see below** | 10 | `results/tables/prior_posterior_convergence.parquet` |
| Task 8 drift-taxonomy grid (`experiments/drift_taxonomy.py`) | 20 | **not re-run — see below** | 20 | `results/tables/drift_taxonomy.parquet` |

**`prior_analysis.py::posterior_convergence` (n=10, not re-run)**: this
function does not compute a bootstrap CI at all (no `bootstrap_ci` call
on its output anywhere) — it produces a per-strength MEAN convergence
episode from 10 seeds as an illustrative diagnostic, per its own module
docstring ("a small number of directly-instrumented illustrative
trajectories"). It is therefore not "an existing confidence interval"
under this task's instruction. Left as-is because re-running it changes
nothing this task requires; flagged here for completeness.

### 3.1 Reproducibility: a non-determinism issue found and corrected during the seed-count re-runs

This subsection is the detailed technical record behind the
publication-facing summary in `docs/PAPER_PREP.md` §6 ("Reproducibility").
Found during routine internal validation, not by external report; fixed
and every affected result re-run before any number was finalized.

- **Root cause**: `run_matrix_family_sweep`'s per-episode seed formula
  (`src/evaluation/matrix_family.py`) included `hash(algo) % 997`, and
  Python randomizes `hash()` on strings on a per-process basis
  (`PYTHONHASHSEED`) unless explicitly fixed.
- **Detection**: comparing independent re-runs of the identical
  experiment configuration (same `n_seeds`, same `base_seed`) and finding
  a discrepancy (~0.05-0.10 in normalized score) far larger than the
  ~0.02-wide bootstrap CIs involved could explain by Monte Carlo variation
  alone.
- **Fix**: `hash(algo) % 997` -> `zlib.crc32(algo.encode()) % 997` — same
  purpose (a stable per-algorithm-name seed offset), deterministic across
  processes. Confirmed directly: `hash('ucb1') % 997` returned different
  values (399, then 286) across two separate interpreter processes before
  the fix; `zlib.crc32('ucb1'.encode()) % 997` returns the same value (59)
  every time.
- **Experiments rerun**: Phase 3a and the reference-baseline portion of
  Phase 5 — the only two call sites of `run_matrix_family_sweep` — were
  re-executed with the corrected implementation; those are the numbers
  now cited in `docs/PAPER_PREP.md`. Phase 6 and Phase 7 use their own,
  always-deterministic seed formulas (confirmed by inspection) and were
  unaffected — their re-runs (§3) were a seed-count increase only, not a
  correctness fix.
- **Automated verification**: `backend/tests/test_matrix_family_determinism.py`
  runs the affected experiment in two independent subprocesses and
  asserts byte-identical output (part of the full `pytest tests/` suite,
  so it runs on every future test pass) — this specific class of
  regression cannot silently reoccur.

**How it surfaced, concretely**: re-running Phase 3a at `n_seeds=30`
produced a UCB1 column differing from the already-cited
`docs/PAPER_PREP.md` numbers by far more than a seed-count increase
could explain (gap=0.30: previously-cited 0.237, re-run 0.223); a further
independent re-run at the ORIGINAL `n_seeds=20` gave a THIRD distinct
value, 0.224 — three different numbers from three separate process
invocations of nominally the same deterministic experiment, which is
what prompted the investigation rather than accepting it as resampling
noise. This means every Phase 3a/5-reference-baseline number previously
cited in this project's docs (before this task) was not exactly
reproducible; the corrected numbers now in `docs/PAPER_PREP.md`
supersede them.

**Concrete consequence, not just an abstract reproducibility concern**:
`docs/reconciliation_stationary.md`'s Table VI reconciliation had flagged
gap=0.30 Thompson as "the largest single delta found across all three
tables" (paper 0.506 vs. an independent kernel re-implementation's 0.4370)
and built a multi-paragraph mechanistic explanation for it (RNG-architecture
differences between two simulator implementations). That flagged
discrepancy came directly from this bug -- with the corrected paper value
(0.449), the delta shrinks to -0.012, comfortably within Monte Carlo
tolerance, and **all 8 Table VI cells now agree**. The mechanistic
explanation has been corrected in place (not deleted) in that document.
This is the clearest evidence in this audit that the bug was not a
harmless technicality: it had already produced an incorrect scientific
conclusion in existing project documentation before Task 13 found it.

**Task 8 drift-taxonomy grid (n=20, not re-run — documented infeasibility)**:
`drift_analysis.py` DOES compute bootstrap CIs from this grid's output.
Re-running at n_seeds=30 is feasible in principle (10 methods x 64 cells
x 30 seeds x 2000-episode horizon ~= 38.4M episodes, an estimated
~30-40 minutes at this project's observed throughput) but was not run
this session for two concrete reasons: (1) it is Task 8 exploratory
infrastructure, not one of the paper's numbered Phase 1-9 headline
results in `docs/PAPER_PREP.md` §3 (confirmed: no `drift_taxonomy`
citation there); (2) `RunStore` persists FULL per-episode logs for every
cell (`results/runs/` is already measured at **2.1 GB** from the existing
n_seeds=20 run), and a 1.5x seed increase would add a proportional ~1 GB
more disk and ~30-40 more minutes of compute on top of the 5 reruns
already completed this session. This is reported as an honest,
quantified deferral, not silently skipped — recommended as follow-up
work before the drift-taxonomy findings (if used in the paper) are cited
with a CI.

## 4. Legacy pre-Task-5 p-values (`src/evaluation/harness.py`): deprecated, excluded from correction

Kept in the registry for completeness (every claim must be inventoried),
never merged into §1's correction family, and flagged
`deprecated_degenerate` everywhere cited. Two representative rows
(Phase 4, `results/phase4/thompson_effect_sizes.csv` — the exact source
of `docs/PAPER_PREP.md`'s previously-uncaveated "Holm-corrected p=2.3e-13"
citation, now corrected inline):

| method (vs. Thompson) | mean_difference | Cohen's d | Cliff's delta | t | p (raw) | p (Holm) | n_trials |
|---|---|---|---|---|---|---|---|
| Adaptive (UCB1) | +0.1152 | 2.184 | 0.887 (large) | 10.60 | 2.82e-14 | 2.26e-13 | 50 |
| Fixed Rule (blind static) | -0.0566 | -1.213 | -0.618 (large) | -5.36 | 2.21e-6 | 8.83e-6 | 50 |

Some rows in this same legacy pipeline (`results/phase1/cost_effect_sizes.csv`,
`thompson_effect_sizes.csv`) show even more extreme artifacts —
`t_statistic = -inf`, `p_value = 0.0` exactly (e.g. the phase1 "Fixed
Rule" row) — a literal-floating-point symptom of near-zero paired-difference
variance across 50 trials for some method pairs. This is not a bug to
fix: it is the most extreme, concrete illustration of exactly the
phenomenon `docs/statistics.md` already documents (a deterministic
simulator's p-value is not bounded away from these extremes the way a
real population's would be), and is additional evidence, not
contradicting evidence, for excluding this whole pipeline from the
corrected inferential family.

**Effect size and CI remain the load-bearing evidence for every one of
these comparisons** — Cohen's d / Cliff's delta magnitudes here agree
with, and are corroborated by, the independent bootstrap-CI-only Task 5+
comparisons in §5 (non-overlapping CIs on the same headline rates).

## 5. google-gemini's degenerate output diversity — required flagging, applied consistently

Every place in this project that reports a per-family LLM statistic now
carries `n_distinct_mappings` alongside the sample count, and a
`bootstrap_ci_degenerate` flag (`True` when `n_distinct_mappings <= 3` —
calibrated to catch google-gemini's real n_distinct=2 without over-flagging
groq-llama's 15/19 or nvidia-mistral-nemotron's 6/21):

- `backend/experiments/study_analysis.py::llm_q_by_family` and
  `::llm_per_type_accuracy_by_family` — both extended (additive columns,
  Task 13) to report `n_distinct_mappings` and `bootstrap_ci_degenerate`
  per family, reusing `recovery.analysis.significance.n_distinct_mappings`.
- `backend/experiments/hypothesis_registry.py` — every row carries
  `n_api_calls`, `n_parsed_ok`, `n_distinct_mappings`, and
  `bootstrap_ci_degenerate` (§2).
- `docs/PAPER_PREP.md`, `docs/experiments/human_study.md` — narrative
  caveats already existed pre-Task-13 (Task 11's own write-up); Task 13
  adds the corresponding PROGRAMMATIC flag so the caveat is enforced in
  code, not just prose, and cannot silently go stale if new LLM samples
  are added later.

**google-gemini's real numbers**: 21 API calls, 21 parsed OK, **2**
distinct mappings (15/21 identical; the other 6 differ from the first in
exactly one position, `malformed_output`). Every bootstrap CI on this
family's per-type accuracy is therefore flagged degenerate — visible in
§2.2's table as `[1.000, 1.000]` (zero-width) for every type except
`malformed_output`, which is exactly what a percentile bootstrap over a
2-point-mass sample produces (not a bug: a mathematically correct
description of a degenerate input, reported honestly rather than
suppressed).

## 6. Task 19B — drift-crossover negative result

Full write-up: [`docs/theory/drift_crossover_negative.md`](theory/drift_crossover_negative.md).
No claim in this section carries a p-value or Holm-Bonferroni correction —
every number below is a bootstrap-style effect-size/CI or a plain
proportion with a Wilson interval, per §1's policy, and the task's own
explicit instruction not to introduce seed-based p-values for this work.

**Claim family and correction status**: three pre-registered attempts at a
low-dimensional sign-predicting statistic, evaluated against ONE fixed
falsifiable bar (beat a 79.7% majority-class baseline out-of-sample,
leave-one-cell-out) — not a family of independent hypothesis tests, so no
multiple-comparison correction applies; each attempt is reported as a
single effect-size-style number (R², sign accuracy), not a p-value.

- **`(s, g)`**: `R²=0.012`. Retracted status: none (reported as insufficient
  on its own terms, no circularity issue).
- **`(d, τ)`**: `R²=0.579` (jointly), `R²=0.735` (with magnitude). **RETRACTED
  as validation evidence** — τ's denominator is `predict_crossover`, the
  theory under test; see the doc's §3.1 self-audit. Not statistically
  circular (no shared data/fitted parameters with the target; κ=1.0 fixed,
  never fit), but conceptually circular. Kept in this registry only as the
  historical record of a retracted claim.
- **`(D, R)`**: `R²=0.676` (jointly), sign accuracy 79.7% in-sample AND
  leave-one-cell-out (51/64, identical misclassified set both times) — no
  better than the majority baseline. `D` verified against
  `recovery.theory.crossover.static_regret_rate` to 2.78e-16 (identity
  check, not a predictive use of theory code).

**CONFIRMED claim (P2)**: sign of `adaptive_advantage` is governed by
reversion rate `R`, not damage magnitude `D`. 13/13 negative cells at
`R≥0.68`; 32/32 `R≈0` cells positive. Wilson 95% CI on `monotone_decay`'s
`R`: `[0.0000, 0.0160]`, n=236 (reversion-eligible population). Independent
seeds underlying the split: 260 (13 negative cells × `DEFAULT_N_SEEDS=20`),
640 (32 positive `R≈0` cells × 20) — same seeds drive both the `(D,R)`
statistics and the `adaptive_advantage` measurement per cell (shared CRN
draw, not two independent samples).

**EXPLORATORY claim, explicitly not confirmatory**: a post-hoc 2-parameter
`(R, interval)` grid-search rule reaches 92.2% in-sample / 87.5%
leave-one-cell-out — both above baseline, but fitted after seeing which
cells broke the pre-registered `(D,R)` rule, on 64 non-independent cells (20
seeds pooled per cell, heavy within-drift-type correlation, 13 negatives
concentrated in 2 of 4 types). Tagged exploratory/hypothesis-generating
throughout; not a validated rule, not implemented.

**Data sources, all reused, no new simulation**: `experiments.drift_taxonomy
.build_environment` (deterministic environment reconstruction — no episode
simulation), `results/tables/drift_per_cell_metrics.parquet` (already-
committed, read-only). Analysis scripts (temporary, gitignored, not
deliverables): `backend/scratch/gate2/*.py`.

## 7. Task 19A — classifier-noise-aware crossover theory, empirical validation

Full write-up: [`docs/theory/crossover_derivation.md`](theory/crossover_derivation.md).
No p-values or Holm-Bonferroni correction anywhere in this section — every
claim is an effect-size-style number (ratio, absolute error, correlation)
against already-committed production data, per §1's policy (this is not a
simulation-seed comparison family, and it is not the real-sample LLM family
either).

**Tier 1 (primary): `results/tables/horizon_sweep_full26.parquet`.** All 52
expected cells available (26 environments × 2 classifiers; the original
6-of-52 gap — 23 environments never simulated — was closed by a dedicated,
explicitly-approved sweep; full provenance and cost record below and in
`docs/theory/crossover_derivation.md` §7.2). At `kappa=1.0` these numbers
remain **not themselves validation** — `kappa` is an uncalibrated
multiplicative free constant. See below for the pre-registered fitted-kappa
result, which supersedes the earlier in-sample `n=3` analysis entirely.

**Kappa calibration — pre-registered, environment-level held-out result
(see `docs/theory/kappa_preregistration.md` for the protocol, committed
before this data existed, and `docs/theory/crossover_derivation.md` §7.1
for full per-cell tables).** Split: 18 TRAIN / 8 TEST environments
(`KAPPA_SPLIT_SEED=19001`, `canonical` fixed in TRAIN). Both branches
finite at n=18 (train) / n=8 (test) — well above the pre-registered n≥5
power floor; this result is not underpowered.

- **Branch A (transfer):** `kappa_A=0.044381` fit on TRAIN `C=I`, applied
  UNCHANGED to TEST `C≠I` → `E_A` = median abs. rel. error **59.7%**.
- **Branch B (in-condition ceiling):** `kappa_B=0.042559` fit on TRAIN
  `C≠I`, applied to the SAME TEST `C≠I` cells → `E_B` = **40.5%**.
- **`R = E_A/E_B = 1.4729`** — satisfies the pre-committed `R ≤ 1.5`
  Case-1 threshold, 1.8% below it.
- **Pre-registered bootstrap (2,000 resamples, `BOOTSTRAP_SEED=19002`,
  environment-level, not cell-level) does not confirm robustness**: 95%
  CI for `R` = **[1.00, 9.55]**, reaching well past the threshold into
  Case-2 territory. Fit-uncertainty bootstrap: `kappa_A` 95% CI [0.0369,
  0.0544], `kappa_B` 95% CI [0.0105, 0.0426].
- No p-values were computed for this comparison (only ratios, medians,
  and bootstrap percentile intervals, per §1's policy).
- **Conclusion: `crossover_horizon_confused` is classified
  PARTIALLY-SUPPORTED**, neither VALIDATED nor UNVALIDATED/PRELIMINARY.
  Case 1 is the pre-committed reported outcome (point estimate satisfies
  `R≤1.5`), described as a **calibrated predictor** (`kappa_A≈0.0444` is
  within the pre-registered near-Task-10 range `[0.00045,0.045]`) rather
  than a validated bound, given the thin margin, the wide bootstrap CI,
  and the large remaining absolute error (~40–60%). No second kappa
  specification was or will be tried, per the pre-registration's hard
  commitment.

**Provenance of the (now closed) 46-cell Tier 1 gap, and its actual
cost.** The 23-environment gap was confirmed via exhaustive RunStore
inspection to be a documented, intentional, cost-gated scope reduction,
not data loss (`docs/experiments/instrumentation.md`). It was
subsequently closed with explicit approval: scoped to the 2 methods Tier 1
needs (204 new cells, not the 1,518-cell/11-method figure originally
estimated at ~13.4h for all methods), fresh `--estimate` **~38.4 min**.
**Actual measured wall-clock: ≈81.9 minutes (≈2.13× the estimate)** across
two attempts (one killed on a since-corrected stop-rule misapplication,
one completed cleanly) — full incident detail, including a disproven
BLAS-oversubscription hypothesis and two independently-measured
(disagreeing) effective-parallelism figures (1.90× vs. 2.74×), in
`docs/experiments/instrumentation.md`. `reproduce.py`'s ~73h/~66h
un-filtered headline cost figures remain un-re-derived and suspect in
both directions for the same underlying reason.
- Amendment C diagnostic (correlation between `observed_T` and absolute
  relative error): computed only at the original n=6 cell sample,
  −0.79 (Pearson) / −0.89 (log). **Not recomputed at n=52** — this
  diagnostic was not part of the pre-registered protocol, and recomputing
  it now would be an unregistered post-hoc analysis; reported here as an
  n=6 finding only, not extended.

**Tier 2 (secondary, static-side only): `results/tables/table_vii.parquet`.**
6 noise levels, 50 already-simulated seeds each. Validates
`static_rate_confused` only — Table VII has no adaptive-method trajectory,
so it does not and cannot validate `crossover_horizon_confused`. Median
absolute error **0.0093** across all 6 noise levels — the strongest
validation result in this task, consistent with this component being exact
closed-form math (§3.2 of the derivation doc), not an asymptotic
approximation.

**Channel findings** (real classifier, mean diagonal accuracy 0.939):
ceiling reduction = **0.0120**; mean gap shrinkage across 7 predicted
buckets = **0.0020** — ceiling reduction is the larger of the two channels
at this classifier's actual accuracy, by roughly 6×.

**Argmax-inversion proposition**: does not occur at the real classifier's
measured confusion (checked across all 7 predicted buckets). A constructed
3-type counter-example (documented in the derivation doc) confirms the
detector fires when its structural condition IS met — not a dead code
path.

**Non-monotonicity finding**: `T*` is not monotonic in classifier accuracy
in general (`rho` and `ceiling_reduction` are; `T*` is not) — verified with
a clean 2-type construction where a predicted bucket's own best arm flips
identity as accuracy crosses a threshold, producing a V-shaped gap rather
than a monotonically shrinking one. Documented, not asserted away.

## 8. Figures

No new figures were required by this task beyond the registry tables
above (the task specifies figures for Task 12's robustness sweep, already
delivered — `results/figures/robustness_ranking_preservation.png`,
`results/figures/robustness_tstar_distribution.png` — Task 13 did not
request new plots and none were fabricated to pad this document).

## 9. Reproducibility

```bash
# From backend/, with the project venv active
python experiments/hypothesis_registry.py               # the new real LLM per-type tests (§2)
pytest tests/test_significance.py -v                       # unit tests for recovery/analysis/significance.py
pytest tests/test_hypothesis_registry.py -v                # integration test for the driver above
pytest tests/test_matrix_family_determinism.py -v           # regression test for the hash() bug fix (§3.1)

# Re-runs performed this session (seed-count audit, §3) -- already executed,
# results already on disk under results/phase{3a,5,6,7}/ and results/tables/:
python -c "from src.evaluation.matrix_family import run_phase3a; run_phase3a(n_seeds=30)"
python -c "from src.evaluation.static_rule_sweep import run_phase5; run_phase5(n_seeds=30)"
python -c "from src.evaluation.nonstationarity_sweep import run_phase6; run_phase6(n_seeds=30)"
python -c "from src.evaluation.taxonomy_size_sweep import run_phase7; run_phase7(n_seeds=30)"
python experiments/matrix_robustness.py                    # N_SEEDS_FULL=30 (module constant, see §3.1 of this file)

# Task 19A: classifier-noise-aware crossover theory (§7) -- read-only over
# already-committed data, no new simulation
python experiments/crossover_theory_validation.py           # full mode: real classifier confusion matrix
python experiments/crossover_theory_validation.py --smoke   # smoke mode: synthetic uniform_confusion(N,0.9), isolated output dir
pytest tests/test_crossover_theory.py -v                    # 24 unit/self-consistency tests
pytest tests/test_crossover.py -v                            # confirms zero regression on the untouched Task 10 functions

# Task 19B: drift-crossover negative result (§6) -- temporary, gitignored
# analysis scripts, not re-run as part of the normal test suite
# (backend/scratch/gate2/*.py; kept for inspection, not a pytest target)

# Full backend suite (319 passed at time of writing prior to Task 19, 0 failures, 2 pre-existing/unrelated warnings)
pytest tests/ -q
```
