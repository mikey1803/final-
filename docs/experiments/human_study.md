# The Blind-Design Elicitation Study (Task 11)

This document explains Task 11's central deliverable
(`study/`, `backend/recovery/analysis/human_study.py`,
`backend/experiments/llm_designers.py`, `backend/experiments/study_analysis.py`)
-- a study measuring the DISTRIBUTION of hand-designed recovery-table
quality `q` across independent, blind designers (both human and LLM),
supplying the empirical input Task 10's crossover predictor needs `q`
for. See [`study/protocol.md`](../../study/protocol.md) for the complete
protocol and [`study/preregistration.md`](../../study/preregistration.md)
for the frozen scoring/analysis plan.

## Status of this document

**No human data has been collected yet.** Running the study against real
participants requires institutional ethics approval
(`study/README.md`'s "Institutional approval" section), which has not
been obtained as of this writing. Every section below that depends on
human data reports this honestly (`n=0`) rather than being silently
omitted or filled with placeholder numbers. What HAS been done, and is
reported below with real numbers: the full pipeline was built and
independently verified end-to-end (web app -> storage -> scoring ->
analysis -> figures, using synthetic test submissions generated through
the actual API and then discarded -- never presented as findings), and a
small, real (not simulated) validation run was executed against the one
LLM model family this environment has API credentials for.

## The protocol, summarised

See `study/protocol.md` for the full version. In brief: a participant
sees only Table I (7 failure-type descriptions) and Table II (6
recovery-strategy descriptions + costs), in a per-participant randomised
display order, and picks one strategy per failure type, blind to the
hidden success-probability matrix (`recovery.envs.canonical.MATRIX`).
`q_count` is the number of rows where the chosen strategy achieves that
row's maximum success probability.

## The blindness guarantee, and the test that enforces it

Every response from `study/app/backend/main.py` (the standalone study
API) is filtered through an explicit Pydantic field allowlist
(`study/app/backend/schemas.py`), and the module never imports
`recovery.envs.canonical.MATRIX`, `BLIND_RULE`, or `recovery.theory.*`
anywhere in its request-serving path -- scoring happens entirely
offline, in `recovery.analysis.human_study`, after collection. Three
independent tests enforce this (`backend/tests/test_study.py`):

- `test_blindness_no_endpoint_leaks_matrix_values_or_reward_shaped_keys`
  -- hits every endpoint and scans every raw response body for any
  `MATRIX` float value (to 2 decimal places) and any reward/score/matrix-
  shaped key name.
- `test_study_backend_source_never_references_matrix_or_blind_rule` -- a
  static AST audit of `study/app/backend/*.py`'s actual imports/attribute
  accesses/bare names (not string literals, so a docstring warning about
  what NOT to import doesn't trip it) for `MATRIX`/`BLIND_RULE`/
  `recovery.theory`.
- `test_strategy_costs_and_matrix_values_are_disjoint_ranges` -- a
  regression guard behind the value-scanning test's implicit assumption
  (strategy costs are all `>= 1.0`; every `MATRIX` value is `<= 0.80`).

Full endpoint-by-endpoint audit: [`study/blindness_audit.md`](../../study/blindness_audit.md).

## Recruitment record

**Not yet started.** No recruitment channel has been used and no
participant has been invited, pending the institutional approval
described in `study/README.md`. This section will be updated with actual
channels used and dates as recruitment happens, per `study/protocol.md`'s
instruction that the protocol document states the plan and this document
states what actually occurred.

## The measured `q` distribution (human arm)

**`n = 0`.** No distribution to report. `recovery.analysis.human_study.study_summary`
and `experiments/study_analysis.py` both handle this explicitly (printing
"No human submissions collected yet" rather than crashing or fabricating
a distribution) -- verified by `tests/test_study.py::test_study_summary_handles_zero_submissions`
and `test_study_summary_handles_n_equals_1` (the smallest non-degenerate
case: CI correctly reported as `None`/undefined, never a fabricated
interval).

## The second-best-error claim, tested against data

**Pre-registered claim** (`study/preregistration.md`, RQ3): a plausible
design error picks the SECOND-best strategy for a failure type, not an
arbitrary one. This is directly testable on the project's own existing
single instance: `recovery.envs.canonical.BLIND_RULE` scored against
`MATRIX` gets exactly one row wrong (`wrong_tool_selected`), and that
row's chosen strategy (`replan`) is EXACTLY rank 2 (the second-best
option for that row, after `switch_tool`) --

```
score_submission(BLIND_RULE, MATRIX) = {
    "q_count": 6,
    "rank_of_chosen": [1, 1, 2, 1, 1, 1, 1],   # rank 2 at wrong_tool_selected
    ...
}
```

confirmed by `tests/test_study.py::test_score_submission_canonical_blind_rule_is_6_of_7_with_miss_at_rank_2`
and `test_second_best_error_claim_on_the_canonical_miss`
(`second_best_error_test` reports `fraction_rank_2 = 1.0` on this single
data point). **This is `n=1` evidence for the claim, not a validated
population statistic** -- it confirms the claim is at least CONSISTENT
with the one instance the paper's original claim was based on; a
genuine test of the claim requires the human arm's `n=10-30` (pending
approval).

## The LLM arm: what was actually run

`conf/llm_designers.yaml` configures three distinct model families
(`groq-llama` via Groq, `google-gemini` via Google's OpenAI-compatible
endpoint, `nvidia-mistral-nemotron` via NVIDIA NIM's OpenAI-compatible
endpoint -- deliberately a Mistral-lineage model, not a second Llama
derivative, so the three families are genuinely distinct base-model
lineages, not just distinct vendors), each reachable through the same
generic `_call_openai_compatible` client (`experiments/llm_designers.py`)
-- **no model name is hardcoded in code**; `_call_anthropic` remains
implemented and available for any future `provider_type: "anthropic"`
entry, but no currently-configured family uses it. This environment now
has working API keys and working models for all three configured
families.

Getting there surfaced three real, undocumented provider quirks, each
confirmed via a real API call (never assumed from documentation alone)
and each fixed in code/config rather than worked around by hand:

1. **Gemini rejects an unrecognised `seed` field with HTTP 400** rather
   than ignoring it (`"Unknown name \"seed\": Cannot find field"`). Fix:
   `run_one_sample`/`run_model_config` gained a `supports_seed` flag
   (`conf/llm_designers.yaml`'s `google-gemini` entry sets
   `supports_seed: false`); the *recorded* `seed` field is `None` in that
   case, never a fabricated "intended" value -- the same rule already
   applied to the Anthropic provider type.
2. **The originally-configured NIM model,
   `mistralai/mixtral-8x7b-instruct-v0.1`, had reached end-of-life** on
   NIM's hosted endpoint on 2026-07-27 (HTTP 410 Gone, with that exact
   date in the response body). Replaced with `mistralai/mistral-nemotron`.
   Separately, this environment's Google key returned HTTP 429
   `limit: 0` ("no quota") on every dated/versioned Gemini model
   (`gemini-2.0-flash`, `gemini-2.5-pro`, etc.) but HTTP 200 on the
   `-latest` alias models; `gemini-flash-latest` truncated to empty
   content at `max_tokens=512` (apparently spending tokens on internal
   reasoning before answering), while `gemini-flash-lite-latest` returned
   real, parseable content -- that is the model now configured.
3. **NVIDIA NIM's free endpoint has real, load-dependent read-timeout
   flakiness**: four sequential identical calls to the same model
   measured elapsed times of 60.8s (timeout), 62.1s (timeout), 2.3s
   (success), 62.0s (timeout) -- consistent with serverless cold-start
   queueing on a shared free tier, not a persistent outage. `run_one_sample`
   now retries a `requests.exceptions.Timeout` specifically within its
   existing attempt budget (distinct from other, deterministic
   network/HTTP errors -- e.g. a 404 or 410 -- which still propagate
   immediately, since retrying a guaranteed-to-fail call wastes the
   budget). If every attempt times out, the sample is recorded as
   `parse_status="network_timeout"`, deliberately distinct from
   `"parse_failure"` (the model never got a chance to answer, vs.
   answering unparseably -- different, and differently informative,
   failure modes). `main()`'s per-family
   `try/except requests.exceptions.RequestException` isolation (also
   added this session) remains as a second-layer safety net: a
   persistent, non-transient failure for one family is recorded
   distinctly (`errored_families`, printed loudly) and does not block
   the other configured families from running to completion.

A fourth quirk surfaced only at full scale (the 4-sample validation run
never triggered it): **Groq's free tier enforces a 6,000-tokens-per-minute
budget** (`x-ratelimit-limit-tokens: 6000`, confirmed directly from
response headers), and this study's ~500-token prompt makes that budget
bind within the first several of 20 rapid sequential calls, not at 20
calls/minute as the *request*-count limit (`14,400/day`) alone would
suggest. The first full run hit this and the whole `groq-llama` family
aborted after 0 samples (`main()`'s per-family isolation correctly
recorded it as `errored_families` rather than crashing the other two
families, which both completed 21/21 in that same run). Fix:
`run_one_sample` now catches HTTP 429 specifically, parses the
provider's own reported reset window from response headers
(`Retry-After`, or Groq's `x-ratelimit-reset-tokens`/`-requests` in its
own `"6s"`/`"1m42s"`/`"570ms"` format -- a real format observed directly
against Groq's API, not assumed), sleeps that long (capped at 65s), and
retries within the existing attempt budget -- the same treatment already
given to transient read timeouts, and for the same reason: a 429 is a
transient, load-dependent condition when the true bottleneck is a
per-minute budget, not a permanent block. If every attempt is
rate-limited, the sample is recorded as `parse_status="rate_limited"`,
deliberately distinct from both `"network_timeout"` and
`"parse_failure"` -- three different reasons a sample can fail to
produce a mapping, each with different implications, none of them
conflated. A `--families` CLI filter was also added so a family that
fails this way can be retried in isolation without re-querying (and
re-spending real quota on) families that already succeeded; the merge-on-write
logic in `main()` preserves on-disk rows for every family NOT included
in the current `--families` subset, replacing only the rows for
families that actually produced new data this run.

**2026-08-20 provenance note**: `llama-3.1-8b-instant` (the `groq-llama`
family's model) no longer exists on this Groq account — live
`client.models.list()` now returns zero Llama chat models. This was
discovered well after this study's data collection (bounded by the NIM
end-of-life date logged above, 2026-07-27), so it does not affect the
`groq-llama` numbers below — they were collected while the model was live
and stand as-is. Cite `llama-3.1-8b-instant` with this deprecation noted
explicitly wherever these numbers are reported, since a model ID that no
longer resolves is a reproducibility fact a reader needs. Full audit:
`docs/PAPER_PREP.md`'s One-Shot LLM provenance-audit note (same date).

**What was executed**: the full run (no `--validation` flag; 20 samples
at `temperature=1.0` + 1 at `temperature=0.0` per family, the protocol's
specified sample size) against all three configured families -- all
three ran to completion (`groq-llama` via one `--families groq-llama`
retry after the rate-limit fix, merged with the other two families'
results from the initial run):

| family | model | n samples | n parsed OK | n parse failures | n timeouts | n rate-limited | q (parsed samples), mean |
|---|---|---|---|---|---|---|---|
| groq-llama | `llama-3.1-8b-instant` (Groq) | 21 | 19 | 2 | 0 | 0 | mean 4.63, `[3,3,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6]` |
| google-gemini | `gemini-flash-lite-latest` (Google) | 21 | 21 | 0 | 0 | 0 | mean 6.29, `[6×15, 7×6]` |
| nvidia-mistral-nemotron | `mistralai/mistral-nemotron` (NVIDIA NIM) | 21 | 21 | 0 | 0 | 0 | mean 6.29, `[5×2, 6×12, 7×7]` |

**63 real samples total, 61 parsed**, across three genuinely distinct
base model lineages (Llama, Gemini, Mistral) -- the full LLM arm the
protocol specifies. This is the LLM arm only; it says nothing about the
human distribution, which remains `n=0` pending institutional approval.

The `groq-llama` family's two full-run parse failures (2/21 = 9.5% of
its samples) are, on inspection of the raw responses, **two distinct
failure modes**, not one repeated twice -- worth stating precisely
rather than over-generalizing from the validation run's single instance:

1. **Wrong type key** (the same mode seen in the earlier validation
   run): the model used its own shorthand `"hallucination"` instead of
   the canonical `"hallucinated_output"` given verbatim in the prompt.
   This sample's raw response also shows the model attempting an
   inline self-correction ("However, replan is not one of the given
   recovery strategies...") that still leaves `"context_loss"` mapped to
   the non-canonical `"re-ground"` (hyphenated) rather than `"re_ground"`
   -- a self-correction that fixed one problem while leaving another.
2. **Wrong strategy value formatting**: type keys were all correct, but
   every strategy value was returned in Title Case with spaces/hyphens
   (`"Retry"`, `"Switch Tool"`, `"Re-ground"`, `"Verify"`, `"Clarify"`)
   instead of the canonical snake_case strings
   (`retry`/`switch_tool`/`re_ground`/`verify`/`clarify`) given verbatim
   in the prompt.

`parse_mapping_response` (`experiments/llm_designers.py`) does an exact,
case-sensitive string match against `TYPE_NAMES`/`STRATEGY_NAMES` with no
normalization -- confirmed correct, by design: silently accepting
`"Switch Tool"` as `switch_tool` would be exactly the kind of
"helpfully" guessing this project's "never fabricate" rule exists to
prevent, since a normalization rule permissive enough to accept this
case could, in a different response, silently accept a genuinely
ambiguous or wrong answer. After exhausting all 3 retries with a fresh
sample each time, both mappings are correctly recorded as
`parse_status="parse_failure"`, `parsed_mapping=None`, never coerced
into a guess.

**A real, notable cross-family pattern -- descriptively true, formally
significant in only 1 of 3 families (updated by Task 13's statistical
audit)**: `experiments/study_analysis.py`'s `llm_per_type_accuracy_by_family`
table shows `malformed_output` has the lowest raw accuracy in **all
three** families -- google-gemini 28.6%, groq-llama 42.1%,
nvidia-mistral-nemotron 42.9%, each the family's own lowest score by a
wide margin (every other type scores 37-100% across families). Task 13
(`backend/experiments/hypothesis_registry.py`,
`docs/statistics_registry.md`) tested this claim formally: a per-type-vs-
rest exact binomial test (type's accuracy vs. the pooled rate of the
OTHER 6 types), Holm-corrected within each family across its 7 types --

| family | raw p | Holm-corrected p | significant at alpha=0.05? |
|---|---|---|---|
| nvidia-mistral-nemotron | 8.0e-15 | 5.6e-14 | **yes** |
| groq-llama | 0.011 | 0.056 | **no** (fails after correction) |
| google-gemini | undefined | undefined | **cannot be computed** (see below) |

`google-gemini`'s test is undefined, not merely non-significant: every
one of its 21 samples answered all 6 OTHER types correctly, so the
"pooled rest rate" to test `malformed_output` against is exactly 1.0 --
a two-sided binomial test against a reference probability of exactly 1
has no defined p-value, and the pipeline reports this honestly (`NaN`,
Holm-treated conservatively as p=1 / fail-to-reject) rather than
fabricating a number. This is itself a direct symptom of this family's
2-distinct-mapping degeneracy (next paragraph), not independent evidence
against the finding. For `groq-llama`, the type with a statistically
significant corrected deficit is actually `tool_timeout`
(p_holm=0.021), not `malformed_output`.

**Conclusion**: `malformed_output`'s lowest-raw-accuracy pattern is real
and reproduced in all three families' raw numbers, but the paper should
report it as a descriptive pattern with one family's worth of formal
significance (`nvidia-mistral-nemotron`), not as "the single hardest
failure type for every family" -- that stronger phrasing is not
supported by the corrected significance tests and has been corrected
here and in `docs/PAPER_PREP.md`.

**A real limitation, caught by inspecting the raw data rather than
trusting the sample count at face value**: `google-gemini`'s
`temperature=1.0` draws show very LOW sampling diversity -- only **2**
distinct mappings across 21 draws (15/21 identical; the other 6 differ
from the first in exactly one position, `malformed_output`). This was
checked for a script bug first (seed reuse, accidental caching) and
ruled out: `sample_index` increments correctly 0-19, `seed` is
consistently `None` (as expected -- `supports_seed: false` for this
family, recorded honestly, see quirk #1 above), and timestamps show 21
genuinely distinct API calls. This appears to be real behavior of
`gemini-flash-lite-latest` on this structured-JSON task -- a "lite"
model may have an effectively peaked/low-entropy output distribution
here even at nominal `temperature=1.0`. `nvidia-mistral-nemotron` shows
moderate diversity (6 unique mappings/21), `groq-llama` the most (15
unique/19 parsed). **Implication for the paper**: `n=21` per family
should NOT be read as 21 independent draws from a rich underlying
distribution for every family -- for `google-gemini` specifically, it is
closer to "2 distinct designs, sampled with skewed frequency," which
matters for how confidently any per-family variance claim can be made.
State this explicitly wherever `google-gemini`'s `n=21` is cited, and
do not compute a bootstrap CI on its `q` distribution without this
caveat attached.

## Task 10 consuming the empirical q prior

Beyond the headline per-submission analysis above,
`recovery.analysis.human_study.q_prior_crossover_distribution` explicitly
wires `q_prior`'s resampled empirical `q` distribution into Task 10's
predictor (`predict_crossover`), turning each resampled `q` back into a
full synthetic mapping via one documented modeling assumption: the `N - q`
"wrong" rows get that row's RANK-2 (second-best) strategy, per the
pre-registered second-best-error claim tested above -- `q` alone doesn't
determine a mapping's regret rate, so this assumption is necessary and is
stated explicitly rather than left implicit. This is deliberately
distinct from `crossover_budget_analysis` (which uses each real
submission's ACTUAL mapping, no assumption needed) -- the two answer
different questions ("what do real designs predict" vs. "what does the
measured quality DISTRIBUTION predict"). Verified end-to-end against
synthetic scored submissions (2 inputs, 2000 resampled draws): a
perfect-table-only input correctly yields `T*=inf` for all 2000 draws; a
mixed canonical/perfect input yields 957/2000 finite draws with a
bootstrap 95% CI of roughly `[1.02e5, 1.08e5]` episodes -- confirms the
wiring, not a reported finding. At `n=0` real submissions,
`experiments/study_analysis.py` reports this section as "Cannot compute:
q_prior requires at least one scored submission," the same honest-`n=0`
pattern as every other section.

## LLM-arm crossover-budget analysis (computable now; not the paper's headline claim)

`experiments/study_analysis.py:llm_crossover_budget_analysis` applies
Task 10's calibrated predictor to each of the 61 real, parsed LLM
mappings, grouped by family (never pooled -- the families' `q`
distributions are not comparable). Real result:

| family | fraction beats bandit @ 100 ep | @1k | @10k | @100k |
|---|---|---|---|---|
| nvidia-mistral-nemotron | 0.429 | 0.381 | 0.381 | 0.381 |
| google-gemini | 0.286 | 0.286 | 0.286 | 0.286 |
| groq-llama | 0.105 | 0.000 | 0.000 | 0.000 |

`google-gemini`'s flat 0.286 across every budget is the direct,
mechanical consequence of its 2-point-mass output distribution (above):
with only 2 distinct mappings, `T*` takes only 2 distinct values, so the
"fraction beating budget" step function has at most one jump across the
whole budget range tested. **This table answers "what does an LLM's
blind design imply about the static-vs-adaptive crossover," which is a
different question from the paper's actual headline claim** (what a
HUMAN blind design implies) -- report both, but never substitute this
for the (currently `n=0`) human-arm table when it lands.

## Human-vs-LLM comparison

**Cannot be computed**: requires a non-empty human arm.
`recovery.analysis.human_study.compare_human_llm_q` (Cliff's delta with
a bootstrap CI, reused from `recovery.metrics.bootstrap.cliffs_delta` --
never a p-value, per `docs/statistics.md`) is implemented and unit-tested
(`experiments/study_analysis.py`'s own pipeline-verification pass, run
against synthetic test submissions generated through the real API and
then discarded, confirmed `delta=-1.0, magnitude="large"` for
deliberately-random vs. real-LLM mappings -- exactly the expected
direction, evidence the function is wired correctly, NOT a reported
research finding).

## Agreement analysis

`recovery.analysis.human_study.fleiss_kappa`, `pairwise_agreement`, and
`entropy_per_type` are implemented and unit-verified (perfect-agreement
synthetic input gives kappa=1.0/pairwise=1.0/entropy=0 for every type;
uniformly-random synthetic input gives kappa≈0.008 (chance-level) and
entropy near `log2(6)≈2.58` bits, the maximum possible for 6 strategies
-- both exactly as expected, confirming the implementation, not a
reported finding). Cannot be computed on real data: requires `n >= 2`
human submissions.

## Bias analysis

- **Position bias**: display order is randomised per participant and
  recorded (`display_order`) precisely so this is checkable once data
  exists -- `tests/test_study.py::test_display_order_differs_across_tokens_and_is_recorded`
  confirms the mechanism itself works (different tokens get different,
  correctly-recorded orders). The actual bias CHECK (does an early
  display position get chosen disproportionately, controlling for
  content) requires real submissions.
- **Confidence calibration**: `plot_confidence_vs_correctness` in
  `experiments/study_analysis.py` is implemented and figure-tested (see
  below); requires human data to produce a real figure.
- **Completion time vs. `q`**: `plot_time_vs_q`, same status.
- **Experience vs. `q`**: `study_summary`'s `experience_q_correlation`/
  `llm_familiarity_q_correlation` fields, same status.

## Sample size -- honest statement

Per `study/protocol.md`'s pre-registered acknowledgement: even at the
target `n=10-30`, the bootstrap 95% CI on mean `q` is wide (simulated
under two plausible priors: ~1.05→0.64 width at `n=10→30` under a
prior centered near the known instance's `q=6/7`; ~2.65→1.61 under a
conservative uniform prior -- see `study/protocol.md`'s power-analysis
table). **This study, once run, will establish a plausible RANGE for `q`,
not a precise population parameter** -- and currently, at `n=0`, it
establishes nothing empirical about the human population at all. Every
number in this document is either a real (if small/partial) measurement,
explicitly labeled as such, or an explicit "not yet collected," never a
number in between.

## Reproducibility

```bash
# Backend (from repo root)
uvicorn study.app.backend.main:app --reload --port 8100
# Frontend (from study/app/frontend/)
npm install && npm run dev

# LLM arm
python experiments/llm_designers.py --validation   # small real run, whatever keys are set
python experiments/llm_designers.py                 # full run, all 3 families (already executed -- see above)
python experiments/llm_designers.py --families groq-llama   # retry just one family (e.g. after a rate limit),
                                                              # merges into the existing --output file rather
                                                              # than overwriting other families' real data

# Analysis (safe to run at any time, including n=0 human submissions)
python experiments/study_analysis.py

# Tests
pytest tests/test_study.py -v
```
