# Task 15 — Prompt-Robustness Supplementary Experiment

> **SUPPLEMENTARY.** This document and everything it reports is a
> supplementary analysis of the Task 11 blind-design LLM study. It does
> **not** replace, modify, or get pooled with the primary protocol's
> results (`results/tables/llm_designers.parquet`, Phase 8 /
> `docs/PAPER_PREP.md` §3). The primary run remains the paper's headline
> LLM-arm result. Every artifact here is prefixed `prompt_robustness_*`
> and lives in its own table/figure files, kept structurally separate
> from the primary ones — see §5.

## 1. Motivation

Does the LLM arm's headline `q` (design quality, 0–7 correct rows) depend
on the specific wording of the blind-design instrument? And is
`google-gemini`'s striking Task 13 finding — only 2 distinct mappings
across 21 identically-worded repeated calls — an artefact of sending the
exact same prompt every time, or genuine low-diversity model behaviour
that persists even when the prompt varies?

Two independent sub-experiments answer these:

- **Part A (paraphrase stability)**: run 2 differently-worded paraphrases
  of the primary instrument across all 3 LLM families, at reduced `n`,
  and test whether `q` differs from the primary run.
- **Part B (presentation-order randomization, google-gemini only)**:
  re-run the primary wording, but with a per-sample randomized
  failure-type/strategy display order, and compare mapping diversity and
  `q` against the original identical-prompt run.

## 2. Method

### 2.1 Paraphrases

`recovery/envs/paraphrases.py` defines two paraphrases of
`recovery.envs.descriptions.build_task_instructions()`:

- **paraphrase_a** — casual/narrative register ("Your task: build a
  static decision table...").
- **paraphrase_b** — formal/enumerated register ("Task specification:
  construct a static mapping from failure category to recovery
  action...").

Both preserve **exactly**: the 7 failure-type `name`/`label` identifiers,
the 6 strategy `name`/`label` identifiers, the 6 strategy cost values
(imported from `recovery.envs.descriptions.STRATEGY_COSTS`, never
retyped), the task itself (choose exactly one strategy per failure type,
tie-break toward lower cost), the epistemic constraint (no performance
feedback), and the response-format instructions
(`experiments.llm_designers.RESPONSE_FORMAT_INSTRUCTIONS`, appended
identically regardless of variant — never paraphrased). Only the framing
sentences and the per-type/per-strategy description **prose** differ,
deliberately in different directions from each other (not just from the
primary), so a stability finding can't be an artefact of two paraphrases
being suspiciously similar to one another.

### 2.2 Blindness audit

Before any API quota was spent, every variant's full built prompt
(instructions + response-format block) was checked by
`recovery/analysis/prompt_blindness_audit.py::audit_prompt_text` — the
same value/key-pattern check `tests/test_study.py`'s blindness test runs
against the study backend's HTTP responses (`study/blindness_audit.md`):
no substring matching a `MATRIX` value to 2 decimal places, and no
forbidden reward-shaped key (`reward`, `success_rate`, `matrix`, `score`,
`p_success`, case-insensitive).

```
=== Blindness audit (prompt text) ===
  [PASS] primary: 2236 chars, hash=964fa39587e5
  [PASS] paraphrase_a: 2545 chars, hash=bff2996c3923
  [PASS] paraphrase_b: 2676 chars, hash=16422fe04fa8
```

All 3 variants passed. No paraphrase leaks more information than the
primary instrument.

### 2.3 Runs

Part A was executed via `experiments/llm_designers.py` itself (invoked as
a subprocess, per variant, by `experiments/prompt_robustness.py`) —
100% reuse of the existing request/retry/429-handling/parsing logic, with
an explicit non-default `--output` and `--prompt-variant {paraphrase_a,paraphrase_b}`.
`llm_designers.py` structurally **refuses** to write a non-`"primary"`
variant to its default output path
(`results/tables/llm_designers.parquet`) even if `--output` is omitted —
this is enforced in code (`main()`'s argument validation), not just by
convention, specifically so a paraphrase run can never overwrite the
primary results (see the incident disclosed in `docs/PROJECT_STATUS.md`
§5 from the immediately preceding task, which is exactly the failure mode
this guard exists to make structurally impossible).

Reduced sample size: `n_temp1=8, n_temp0=1` (9 samples/family/paraphrase,
vs. the primary run's 20+1=21) — a real reduction, not a token one, to
respect free-tier rate limits while still giving Mann-Whitney U workable
group sizes.

Part B called `experiments.llm_designers.run_one_sample` directly (the
same tested per-call logic Part A's subprocess also runs) in a thin loop
over sample indices, since randomizing per-sample requires a prompt that
varies per call. For each sample, `recovery.envs.display_order.display_order(token, 7, 6)`
— **the exact same deterministic randomiser the human study instrument
uses** (`study/app/backend/main.py`, refactored this task into
`recovery/envs/display_order.py` so both callers share one
implementation, not two) — produced a type/strategy display order seeded
off `token = f"llm-sample-{i:04d}-temp{temperature}"` instead of a
participant token, and `build_task_instructions(type_order, strategy_order)`
built that sample's prompt. Run at `n_temp1=20, n_temp0=1` (n=21),
matching the primary `google-gemini` run's sample count so the
distinct-mapping comparison is apples-to-apples.

## 3. Results — Part A: paraphrase stability

### 3.1 q per (family, prompt variant), with bootstrap 95% CI on the ratio

| Family | Variant | n parsed | n distinct mappings | CI degenerate? | Mean q (/7) | q-ratio 95% CI |
| --- | --- | ---: | ---: | --- | ---: | --- |
| google-gemini | primary | 21 | 2 | Yes | 6.286 | [0.871, 0.925] |
| google-gemini | paraphrase_a | 9 | 3 | Yes | 6.444 | [0.873, 0.968] |
| google-gemini | paraphrase_b | 9 | 2 | Yes | 5.889 | [0.810, 0.857] |
| groq-llama | primary | 19 | 15 | No | 4.632 | [0.609, 0.714] |
| groq-llama | paraphrase_a | 7 | 6 | No | 4.286 | [0.531, 0.673] |
| groq-llama | paraphrase_b | **2** | 2 | Yes | 3.500 | [0.429, 0.571] |
| nvidia-mistral-nemotron | primary | 21 | 6 | No | 6.286 | [0.857, 0.932] |
| nvidia-mistral-nemotron | paraphrase_a | 8 | 2 | Yes | 6.500 | [0.875, 0.982] |
| nvidia-mistral-nemotron | paraphrase_b | 9 | 2 | Yes | 6.778 | [0.921, 1.000] |

"CI degenerate" reuses Task 13's `n_distinct <= 3` flag — a bootstrap
resample of 2–3 point masses cannot represent genuine sampling
uncertainty, so several of these CIs are narrower than they should
honestly be read as.

### 3.2 Does q differ across prompt variants? (Mann-Whitney U, Holm-corrected within family)

| Family | Comparison | U | raw p | median A | median B | Holm p | Significant? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| google-gemini | primary vs. paraphrase_a | 79.5 | 0.422 | 6.0 | 6.0 | 0.422 | No |
| google-gemini | primary vs. paraphrase_b | 129.0 | 0.037 | 6.0 | 6.0 | 0.073 | No |
| google-gemini | paraphrase_a vs. paraphrase_b | 61.0 | 0.024 | 6.0 | 6.0 | 0.073 | No |
| groq-llama | primary vs. paraphrase_a | 81.0 | 0.389 | 5.0 | 4.0 | 0.547 | No |
| groq-llama | primary vs. paraphrase_b | 32.0 | 0.115 | 5.0 | 3.5 | 0.346 | No |
| groq-llama | paraphrase_a vs. paraphrase_b | 11.0 | 0.273 | 4.0 | 3.5 | 0.547 | No |
| nvidia-mistral-nemotron | primary vs. paraphrase_a | 70.0 | 0.459 | 6.0 | 6.5 | 0.540 | No |
| nvidia-mistral-nemotron | primary vs. paraphrase_b | 55.0 | 0.048 | 6.0 | 7.0 | 0.143 | No |
| nvidia-mistral-nemotron | paraphrase_a vs. paraphrase_b | 26.0 | 0.270 | 6.5 | 7.0 | 0.540 | No |

**No pairwise comparison survives Holm-Bonferroni correction within any
family.** Three raw p-values sit below 0.05 (google-gemini
primary-vs-paraphrase_b, google-gemini paraphrase_a-vs-paraphrase_b,
nvidia primary-vs-paraphrase_b) but none survive correction for testing 3
comparisons per family — this is reported as an honest **null result at
this sample size**, not as proof of stability: at `n=8–9` per paraphrase
group, Mann-Whitney U has limited power, and several of the reduced-`n`
groups also have degenerate distinct-mapping counts (§3.1), so a true
small-to-moderate wording effect could exist without this experiment
being powered to detect it after correction. **State honestly**: this
result is consistent with — but does not conclusively establish —
wording-stability of the LLM arm's `q`.

### 3.3 An unplanned, striking finding: paraphrase wording strongly affected **parse reliability**, not just measured q

groq-llama's (`llama-3.1-8b-instant`) rate of producing a **parseable**
JSON response varied enormously across variants:

| Variant | groq-llama parse failures | Failure rate |
| --- | --- | --- |
| primary | 2 / 21 | 9.5% |
| paraphrase_a | 2 / 9 | 22.2% |
| paraphrase_b | **7 / 9** | **77.8%** |

This is a genuine, reportable finding independent of the q-comparison
above: paraphrase_b's more formal, enumerated register ("Task
specification: construct a static mapping...") caused this small (8B)
model to fail to comply with the JSON-only response format the large
majority of the time, while the same model handled the primary and casual
paraphrase_a wording far more reliably. This was **not observed** for the
other two (larger) model families, which parsed at or near 100% under
every variant. No mechanism is claimed here beyond the observation itself
— plausible candidates (more formal register competing with the
instruction-following template a small instruct-tuned model was trained
on, longer/denser sentence structure, different token distribution near
the response-format instructions) are not distinguished by this data. This
is exactly the kind of LLM-as-designer methodology limitation Task 15 was
scoped to surface — reported here, not smoothed over.

## 4. Results — Part B: google-gemini under presentation-order randomization

| | Original (identical prompt) | Randomized order (per-sample) |
| --- | ---: | ---: |
| n | 21 | 21 |
| Distinct mappings | 2 | **9** |
| Mean q | 6.286 | 5.714 |

Mann-Whitney U = 325.5, p = 0.0016 (single comparison, not part of a
multiple-comparison family — no correction applies).

**Interpretation, stated at the appropriate level of confidence:**
randomizing the per-sample presentation order more than quadrupled the
number of distinct mappings google-gemini produced (2 → 9 out of 21),
which directly answers this sub-experiment's motivating question: the
Task 13 "2 distinct mappings" finding is **at least partly an artefact of
sending an identical, repeated prompt** — it is not simply "this model
always converges on one design regardless of how the question is posed."
At the same time, mean `q` under randomization (5.714) is measurably
*lower* than under the original identical prompt (6.286), a real and
significant difference by this test. A plausible reading is that the
original identical-order prompt let the model exploit some
position/order-dependent regularity that happened to correlate with
higher measured quality, and disrupting that regularity via randomization
reveals more genuine (and somewhat worse, more variable) per-sample
design behaviour — but this is offered as a plausible interpretation, not
a confirmed causal mechanism; this experiment was not designed to
isolate *why* order affects the model's answer, only to establish *that*
it does. Either way, the finding itself is unambiguous and reportable:
`google-gemini`'s apparent design "consistency" under the primary protocol
is confounded with the protocol's use of one fixed, repeated prompt
structure.

## 5. Separation from the primary run

| | Primary (Task 11 headline) | Supplementary (Task 15) |
| --- | --- | --- |
| Data file | `results/tables/llm_designers.parquet` (untouched by this task — verified: 63 rows, unchanged, no `prompt_variant` column added) | `results/tables/prompt_robustness_*.parquet` (6 new files) |
| Figures | `results/figures/llm_study_*.png/.pdf`, `human_study_*.png/.pdf` | `results/figures/prompt_robustness_*.png/.pdf` (2 new figures, both titled "SUPPLEMENTARY (Task 15)" directly in the figure) |
| Docs | `docs/PAPER_PREP.md` §3 Phase 8, `docs/statistics_registry.md` | this file |
| Sample size | n=20+1 per family | n=8+1 (paraphrases) / n=20+1 (randomization) per family |

## 6. Limitations of this supplementary experiment itself

- **Reduced n limits statistical power.** Part A's paraphrase groups
  (n=2–9 per family) are small; the null result in §3.2 should be read as
  "not detected at this budget," not "proven absent."
- **Single run, not repeated.** Neither part was repeated at multiple
  reduced-n draws — a second independent reduced-n run of the same
  paraphrase could land on a different p-value by chance at these small
  sample sizes. This was a deliberate cost/rate-limit trade-off (Task
  15's own instruction to respect free-tier limits), not an oversight.
- **groq-llama's paraphrase_b group (n=2) is too small for its bootstrap
  CI or Mann-Whitney comparisons to be informative** — reported anyway
  (§3.1/§3.2), flagged as degenerate, never hidden.
  Part B's causal interpretation (§4) is explicitly offered as plausible,
  not confirmed — this experiment cannot distinguish "position bias" from
  other explanations for why randomization changed the outcome
  distribution.
- Only `google-gemini` was run under Part B (as specified by Task 15);
  whether the same order-sensitivity holds for the other two families is
  untested.

## 7. Reproduce

```bash
cd backend
python experiments/prompt_robustness.py --dry-run   # blindness audit + prompt preview, no API calls
python experiments/prompt_robustness.py              # full run: Part A + Part B + analysis + figures
```
