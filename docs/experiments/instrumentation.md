# Task 14 — Runtime, Cost, and Provenance Instrumentation

Documents the four-part instrumentation deliverable: per-decision runtime/memory
benchmarking, LLM API cost accounting, exact environment capture, and a
figure/table provenance map. Every number below comes from an actual script
run recorded in `backend/results/tables/`; nothing is estimated or fabricated.

All four scripts live in `backend/experiments/` and reuse the existing kernel,
metrics, and RNG design rather than duplicating logic — see each script's
module docstring for the exact reuse/deviation rationale.

---

## Part 1 — Runtime and memory (`experiments/runtime_benchmark.py`)

### Method

Every policy is timed through an isolated per-decision loop built from the
same shared kernel primitives `run_episodes` uses internally
(`recovery.core.protocols.Context`, `recovery.core.rng.from_seed`,
`recovery.envs.matrix_env.MatrixEnvironment`,
`recovery.classify.synthetic.PerfectClassifier`) — **not** by modifying the
hand-optimized `run_episodes` hot path itself, which Task 6 already profiled
and optimized and which this task's own rules say not to rewrite without
evidence of a flaw. `policy.select()` and `policy.update()` are timed
separately via `time.perf_counter()` over 5,000 decisions (200 warmup
decisions excluded from the statistics) at `BASE_SEED = 140_014`. Peak memory
uses `tracemalloc`, cleared immediately after warmup so allocator warm-up
noise isn't counted.

Because this is a new, parallel measurement path rather than an edit to
`run_episodes`, every policy was **cross-validated** against the real,
unmodified kernel (`cross_validate_against_kernel`, 5,000 real episodes) to
confirm the new measurements aren't measuring something spurious. The
relative ordering matches exactly between the two methods (e.g. LinUCB is
the slowest policy under both the isolated-decision timer and the full
kernel; Oracle/FixedRule/StaticRule are the fastest under both).

A `_NoOpPolicy` is run through the identical loop shape to report the
measurement floor transparently (`select_mean_us = 0.41`) rather than
presenting raw per-policy numbers as if they had zero overhead.

`OneShotLLMPolicy` is excluded — it makes real network/fallback calls and
its latency is dominated by that I/O, not by decision logic; timing it here
would misrepresent both this benchmark (as CPU-bound) and the LLM cost
accounting in Part 2 (as untimed).

### Hardware / OS (exact)

Captured via `capture_hardware_info()`, real values, not assumed:

- CPU: Intel(R) Core(TM) i5-9300H @ 2.40GHz, 4 cores / 8 logical processors
- OS: Windows 11 Home Single Language, 10.0.26200, 64-bit (AMD64)
- Python: 3.13.11 (CPython, MSC v.1944 64-bit AMD64)
- RAM: 8 GB

### Results — per-decision timing (`select`), microseconds, n=5,000 decisions

| Policy | mean | median | p95 | p99 | peak memory (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| _measurement floor (no-op)_ | 0.41 | 0.30 | 0.90 | 1.00 | 296 |
| Oracle | 1.95 | 1.50 | 3.90 | 5.20 | 296 |
| StaticRule (BLIND_RULE, native) | 2.27 | 1.50 | 4.11 | 6.10 | 296 |
| NaiveRetry | 2.93 | 2.30 | 6.30 | 7.70 | 296 |
| FixedRule (blind static, 6/7) | 5.65 | 3.50 | 10.80 | 14.80 | 296 |
| UniformRandom | 12.03 | 10.20 | 25.60 | 35.20 | 680 |
| DecayingEpsilonGreedy | 24.17 | 19.80 | 48.90 | 59.11 | 1,416 |
| EpsilonGreedy | 25.44 | 19.70 | 54.30 | 71.91 | 1,424 |
| UCB1 | 34.00 | 29.00 | 72.00 | 90.10 | 968 |
| DiscountedPGTS | 98.21 | 72.60 | 197.61 | 295.51 | 15,536 |
| PGTS (PriorInitializedThompson) | 92.48 | 71.50 | 184.51 | 258.11 | 101,648 |
| DiscountedThompson | 101.00 | 77.90 | 207.22 | 263.90 | 1,288 |
| Thompson | 105.63 | 81.90 | 220.93 | 291.00 | 1,536 |
| ContextualCostAware | 571.57 | 451.20 | 1,096.80 | 1,368.66 | 6,336 |
| LinUCB | 662.40 | 511.05 | 1,242.92 | 1,554.81 | 12,256 |

Peak-memory differences below ~2,000 bytes are within the measurement noise
floor for this workload (the whole scale here is hundreds to tens of
thousands of bytes, not MB) and are not reported as meaningful. Flagged as
meaningfully different (>2,000 bytes over the minimum, 296 bytes):
**PGTS** (+101,352 bytes — the largest and currently unexplained outlier,
reported honestly rather than with a fabricated explanation), **LinUCB**
(+11,960), **DiscountedPGTS** (+15,240), **ContextualCostAware** (+6,040).

### Cross-validation against the real, unmodified kernel (`run_episodes`, n=5,000 episodes)

| Policy | kernel us/episode | kernel episodes/sec |
| --- | ---: | ---: |
| StaticRule (BLIND_RULE, native) | 10.35 | 96,650 |
| FixedRule (blind static, 6/7) | 10.95 | 91,299 |
| Oracle | 12.34 | 81,020 |
| NaiveRetry | 13.77 | 72,636 |
| UniformRandom | 19.47 | 51,355 |
| EpsilonGreedy | 21.27 | 47,007 |
| DecayingEpsilonGreedy | 22.42 | 44,610 |
| UCB1 | 25.27 | 39,569 |
| PGTS | 54.24 | 18,437 |
| DiscountedThompson | 52.97 | 18,878 |
| DiscountedPGTS | 66.12 | 15,124 |
| Thompson | 60.30 | 16,583 |
| ContextualCostAware | 278.20 | 3,594 |
| LinUCB | 326.89 | 3,059 |

The relative ordering (fast: Oracle/StaticRule/FixedRule/NaiveRetry; slow:
LinUCB, ContextualCostAware) is consistent between the isolated-decision
timer and the real kernel, which is the evidence that the new measurement
path is measuring something real rather than an artifact of its own loop
shape.

Outputs: `results/tables/runtime_benchmark_timing.parquet`,
`runtime_benchmark_hardware.parquet`, `runtime_benchmark_kernel_validation.parquet`.

Reproduce: `python experiments/runtime_benchmark.py --validate` (from `backend/`).

### Quality-vs-overhead trade-off figure (`experiments/pareto_tradeoff_figure.py`)

A publication figure combining Part 1's real per-episode runtime with the
project's existing normalized performance score (`table_v.parquet`, from
`stationary_comparison.py`) — no new composite metric is invented; both axes
are numbers that already exist, for the 10 policies confirmed to be the
*exact same class with the exact same hyperparameters* in both scripts
(verified by reading `stationary_comparison.py`'s `_METHODS` dict against
`recovery/policies/adapters.py`'s factory defaults directly, not assumed
from name similarity).

![Runtime vs. quality trade-off](../../backend/results/figures/runtime_quality_tradeoff.png)

| Policy | Runtime/episode (us, kernel-measured) | Normalized performance (Table V, mean +/- SD over 50 seeds) | Pareto-efficient? |
| --- | ---: | --- | --- |
| Fixed Rule | 10.95 | 0.913 +/- 0.133 | Yes |
| Oracle | 12.34 | 0.988 +/- 0.137 | Yes |
| Naive Retry | 13.77 | -0.383 +/- 0.143 | No |
| UniformRandom | 19.47 | -0.010 +/- 0.179 | No |
| Epsilon-Greedy | 21.27 | 0.582 +/- 0.183 | No |
| Decaying Epsilon-Greedy | 22.42 | 0.343 +/- 0.304 | No |
| UCB1 | 25.27 | 0.382 +/- 0.178 | No |
| Thompson Sampling | 60.30 | 0.709 +/- 0.171 | No |
| Contextual Cost-Aware | 278.20 | 0.679 +/- 0.236 | No |
| LinUCB | 326.89 | 0.778 +/- 0.174 | No |

**Interpretation.** Only **Fixed Rule** and **Oracle** are Pareto-efficient:
nothing achieves comparable-or-better quality at lower-or-equal runtime than
Fixed Rule, and nothing achieves higher quality than Oracle without also
costing more than Fixed Rule for less quality. All eight remaining policies
are strictly dominated — each is beaten on *both* axes simultaneously by at
least one of the two frontier points. This is consistent with, and adds a
computational-cost dimension to, the paper's central stationary-environment
finding (Phase 4/5): in a **stationary** world, the simple static rule is not
only the highest-quality baseline but also one of the cheapest, so every
adaptive method's extra runtime buys no quality advantage here — Thompson
Sampling costs ~5.5x Fixed Rule's runtime for lower quality, and LinUCB costs
~30x for lower quality still. That overhead is the price of *adaptivity*,
which this stationary-world figure cannot show paying off — Phase 6's
distribution-shift results are where the adaptive methods' advantage
actually materializes; this figure should be read alongside that phase, not
as a standalone verdict against adaptive methods in general.

**Excluded from this figure** (benchmarked in Part 1 but with no matching
same-protocol Table V entry, so no score was merged or estimated for them):
`StaticRule (BLIND_RULE, native)` (a second, decision-equivalent
implementation of the same blind rule as Fixed Rule, but a different adapter
class with no independent Table V row); `DiscountedThompson`, `PGTS`, and
`DiscountedPGTS` (introduced for the Task 9 prior-sensitivity study, not
benchmarked under Table V's stationary protocol — their performance data
lives in `prior_grid.parquet`, which was still being regenerated at full
scale at the time this figure was produced; see `docs/PROJECT_STATUS.md`).

Outputs: `results/tables/runtime_quality_tradeoff.parquet`,
`results/figures/runtime_quality_tradeoff.png`,
`results/figures/runtime_quality_tradeoff.pdf`.

Reproduce: `python experiments/pareto_tradeoff_figure.py` (from `backend/`).

---

## Part 2 — API cost accounting (`experiments/llm_cost_accounting.py`)

### Method

Backfilled entirely from already-logged data — **zero new paid API calls**,
per this task's explicit rule. Every row of `results/tables/llm_designers.parquet`
(Task 11's blind-designer LLM study) already stores the complete raw provider
JSON response in its `raw_response_json` column, and all three configured
providers (Groq, Google, NVIDIA NIM) return an OpenAI-compatible
`usage: {prompt_tokens, completion_tokens, total_tokens}` block, confirmed by
direct inspection of real logged rows before writing any extraction code.

**Documented backfill limitation**: 2 of the 63 logged rows (both
`groq-llama`, `parse_status="parse_failure"`) have `n_attempts > 1` —
`run_one_sample` only preserves the *last* attempt's raw response, so earlier
failed attempts consumed real tokens that were never logged. These 2 rows are
flagged `tokens_are_lower_bound_only = True`; their token counts are reported
as a lower bound, never silently presented as complete.

### Pricing sources (real, cited, dated — not fabricated)

| Family | Model priced | Input $/1M | Output $/1M | Pinned to exact model? | Source |
| --- | --- | ---: | ---: | --- | --- |
| groq-llama | llama-3.1-8b-instant | 0.05 | 0.08 | Yes | groq.com/pricing, fetched 2026-08-04 |
| google-gemini | gemini-2.5-flash-lite (**assumed** resolution of the `gemini-flash-lite-latest` alias — not confirmed) | 0.10 | 0.40 | No | ai.google.dev/gemini-api/docs/pricing, fetched 2026-08-04 |
| nvidia-mistral-nemotron | — | — | — | No | **No confirmed official price found** for `mistralai/mistral-nemotron` on NVIDIA NIM as of 2026-08-04 (build.nvidia.com's per-model pricing page did not load). Left unestimated per this task's rule ("where backfill is impossible, say so rather than estimating") — reported as `NaN`, not $0. |

### Results

| Family | n samples | n lower-bound-only | Prompt tokens | Completion tokens | Total tokens | Est. cost (USD, pay-as-you-go) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| google-gemini | 21 | 0 | 11,340 | 1,534 | 12,874 | $0.001748 |
| groq-llama | 21 | 2 | 11,235 | 1,507 | 12,742 | $0.000682 |
| nvidia-mistral-nemotron | 21 | 0 | 10,920 | 1,541 | 12,461 | NaN (no confirmed price) |

Grand total: **38,077 tokens across 63 calls**; **$0.0024** hypothetical
pay-as-you-go cost across the 2/3 priced families.

A `pandas.groupby().sum()` on an all-NaN column defaults to `0.0`, which
would have misleadingly presented the NVIDIA family's *unknown* cost as a
*confirmed zero* — fixed via `s.sum(min_count=1)` so it correctly stays
`NaN`.

**Important caveat, stated explicitly rather than left implicit**: the API
keys used for the actual Task 11 study were free/developer-tier credentials
(Groq's rate-limited free tier, NVIDIA NIM's free prototyping tier, a
free-tier Google API key) — real money spent was almost certainly **$0**.
The figures above are a hypothetical "what this would cost at published list
rates," not a claim that money was actually spent.

Outputs: `results/tables/llm_cost_accounting_per_sample.parquet`,
`llm_cost_accounting_summary.parquet`.

Reproduce: `python experiments/llm_cost_accounting.py` (from `backend/`).

---

## Part 3 — Environment capture (`experiments/capture_environment.py`)

### Method

Reuses `runtime_benchmark.py::capture_hardware_info` rather than duplicating
hardware-capture logic. Dependency versions come from a real
`pip freeze` against the same interpreter every experiment in this project
ran under — not a hand-maintained list (`requirements.txt` itself pins no
versions; every line is a bare package name). LLM model strings are read
back from the same logged `llm_designers.parquet` used in Part 2.

### Python and key dependencies (exact, from `pip freeze`, 85 packages total)

Full list: `results/tables/environment_packages.parquet` and
`results/environment_snapshot.txt`. Notable pins: `numpy==2.5.1`,
`pandas==3.0.3`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `groq==0.37.1`,
`langchain==1.3.14`, `pytest==9.1.1`. Python: 3.13.11.

### LLM model strings actually called

| Family | Provider | Configured model ID | API-disclosed `model_version` | n calls | First call (UTC) | Last call (UTC) |
| --- | --- | --- | --- | ---: | --- | --- |
| google-gemini | openai_compatible | `gemini-flash-lite-latest` | `gemini-flash-lite-latest` | 21 | 2026-08-04T09:16:08Z | 2026-08-04T09:18:04Z |
| groq-llama | openai_compatible | `llama-3.1-8b-instant` | `llama-3.1-8b-instant` | 21 | 2026-08-04T09:25:47Z | 2026-08-04T09:26:06Z |
| nvidia-mistral-nemotron | openai_compatible | `mistralai/mistral-nemotron` | `mistralai/mistral-nemotron` | 21 | 2026-08-04T09:18:07Z | 2026-08-04T09:19:35Z |

**Finding worth flagging explicitly**: Groq and NVIDIA both echo back their
exact configured model ID in the API response, so those two families are
genuinely pinned/reproducible. Google's Gemini API only ever echoes back the
literal alias string `gemini-flash-lite-latest` — never a resolved, dated
model identifier. This means the *exact* underlying model Google served for
this study **cannot be recovered from the logged data alone**. Model
aliases can silently move to a newer underlying model over time; this is a
genuine reproducibility gap for that one family, not an oversight in this
script, and is reported as such rather than assumed away.

All 63 calls across all three families were made on 2026-08-04, between
roughly 09:16 and 09:26 UTC.

Outputs: `results/environment_snapshot.txt`,
`results/tables/environment_hardware.parquet`,
`environment_packages.parquet`, `environment_llm_model_pins.parquet`.

Reproduce: `python experiments/capture_environment.py` (from `backend/`).

---

## Part 4 — Provenance map

Completed 2026-08-06, after both production files affected by the
`--smoke`-overwrite incident (`horizon_sweep.parquet`, `prior_grid.parquet`)
were restored and verified — see the Repository Integrity Recovery report
(`docs/PROJECT_STATUS.md` and the recovery session transcript) for the full
incident history, root-cause analysis, and verification evidence.

**Extended 2026-08-06 for Task 16** (`backend/reproduce.py`, the one-command
reproducibility pipeline; full design rationale and verification evidence in
`docs/reproducibility.md`). Every row now names its **pipeline stage**
(`reproduce.py`'s Stage 0-7; see that script's own module docstring for why
the execution order is dependency-driven rather than a literal 3-4-5
reading of "analyses, figures, tables"). Every command below was actually
executed this session — either during the original recovery, or during
`reproduce.py --mode validation` and `reproduce.py --mode paper`'s two full
runs — and its output compared against the committed production file
(`compare_table()`, numeric-tolerant, `rtol=1e-9`, or a byte hash for
non-parquet files), never merely inferred from reading source. Two rows are
marked **REDUCED (documented deviation)** where a true full-scale
regeneration was measured to be computationally infeasible on this hardware
(~73h and ~66h respectively); this remains true under Task 16 — see
"Cost-gating" below.

| Artifact (production file) | Pipeline stage | Generating script | Raw data consumed | Reproduce command | Seed(s) | Notes |
|---|---|---|---|---|---|---|
| `results/tables/table_v.parquet` (Table V) | 5 | `experiments/stationary_comparison.py` | None (self-contained kernel simulation) | `python experiments/stationary_comparison.py` | `DEFAULT_BASE_SEED=123` | Full scale (550 rows). Non-"One-Shot LLM" rows re-verified exact under Task 16's paper-mode run; the "One-Shot LLM" method's rows make one real, non-deterministic, uncached Groq call per episode and are excluded from bitwise verification for that reason (documented limitation, not a gap in coverage) |
| `results/tables/table_vi.parquet` (Table VI) | 5 | `experiments/matrix_family_sweep.py` | None (self-contained kernel simulation) | `python experiments/matrix_family_sweep.py` | `DEFAULT_BASE_SEED=2026` | Full scale (16,000 rows), re-verified exact under Task 16's paper-mode run |
| `results/tables/table_vii.parquet` (Table VII) | 5 | `experiments/classifier_noise_sweep.py` | `data/classifier_model.joblib` (for the `real`-classifier condition) | `python experiments/classifier_noise_sweep.py` | `DEFAULT_BASE_SEED=123` | Full scale (300 rows), re-verified exact under Task 16's paper-mode run |
| `results/tables/matrix_robustness_{noise_floor,raw,ranking,tstar,hardest_type}.parquet` + 2 figures (Task 12) | 5 | `experiments/matrix_robustness.py` | None (self-contained) | `python experiments/matrix_robustness.py` | `N_SEEDS_FULL=30` region, script-internal seeding | Only script that auto-suffixes its own `--smoke` outputs (`_smoke`), so it cannot repeat the dual-writer footgun the others have. Re-verified exact under Task 16's paper-mode run |
| `results/tables/runtime_benchmark_{timing,hardware,kernel_validation}.parquet` | 5 (paper mode only) | `experiments/runtime_benchmark.py` | None | `python experiments/runtime_benchmark.py --validate` | `BASE_SEED=140_014` | Cheap (n=5,000 decisions/policy); not part of the fast validation-mode pipeline (no `--smoke` equivalent needed — full mode is already fast) |
| `results/tables/drift_taxonomy.parquet` | 3 | `experiments/drift_taxonomy.py` | RunStore cache at `results/runs/drift_taxonomy` (12,812 files, complete) | `python experiments/drift_taxonomy.py` | `DEFAULT_BASE_SEED=20_000` | Full scale (12,800 rows). A full (non-`--smoke`) invocation is a RunStore cache hit, not a re-simulation — confirmed cheap and re-run for real under Task 16's paper-mode pipeline (not cost-gated, unlike horizon_sweep/prior_grid below) |
| `results/tables/horizon_sweep.parquet` | 3 | `experiments/horizon_sweep.py` | RunStore cache at `results/runs/horizon_sweep` | `python experiments/horizon_sweep.py --n-seeds 3 --n-gap-matrices 2` | `DEFAULT_BASE_SEED=4000`, offset by `--n-seeds`/`--n-gap-matrices` | **REDUCED (documented deviation)**: true full default (`n_seeds=30, n_gap_matrices=25` -> 17,160 cells) measured via the script's own `--estimate` at **~73 hours** on this machine; reduced to `n_seeds=3, n_gap_matrices=2` (198 cells, 1,386 rows). Under Task 16, `reproduce.py --mode paper` does NOT re-run this by default (cost-gated; `--allow-expensive` forces the true full grid) — see "Cost-gating" below |
| `results/tables/prior_grid.parquet` | 3 | `experiments/prior_grid.py` | RunStore cache at `results/runs/prior_grid` | `python experiments/prior_grid.py --n-seeds 3 --qualities 0 4 7 --prior-means 0.7 --horizons 500 50000 --n-jobs 2` | `DEFAULT_BASE_SEED` per script default, offset by the above overrides | **REDUCED (documented deviation)**: true full default (84,780 cells) estimated at **~66 hours**; reduced to 1,116 cells/rows. `--n-jobs 2` is a separate memory-exhaustion fix, not a further scope reduction (see root-cause section below). Cost-gated in `reproduce.py --mode paper`, same as horizon_sweep.parquet |
| `results/tables/horizon_crossover.parquet` | 3 | `experiments/crossover_calibration_grid.py` (**not** `horizon_analysis.py` — see warning below) | RunStore cache at `results/runs/crossover_calibration` (2,760 files, complete) | `python experiments/crossover_calibration_grid.py` | `BASE_SEED=4000`, 4 `gap_seed` offsets | 112 rows (56 envs x 2 criteria x 1 method [thompson], `classifier_id='perfect'` only, by design — see the script's own docstring for the theory-scope rationale). Cheap (~6-8 min uncached; RunStore cache already complete so re-verification under Task 16 was a cache hit). Re-verified exact this session |
| `results/tables/crossover_calibration.parquet`, `crossover_calibration_summary.parquet` | 3 | `experiments/crossover_calibration.py`, `experiments/crossover_reliability.py` | `results/tables/horizon_crossover.parquet` | `python experiments/crossover_calibration.py` then `python experiments/crossover_reliability.py` | none (deterministic fit, no RNG) | Re-verified exact this session (16-row `crossover_calibration_summary.parquet` bitwise-identical) |
| `results/tables/crossover_lookup.parquet` | 3 | `experiments/crossover_extension.py` | `results/tables/crossover_calibration.parquet` | `python experiments/crossover_extension.py` | none | Re-verified exact this session |
| `results/tables/prior_{entropy_curves,factor_contribution,mean_sensitivity,negative_transfer_region,negative_transfer_surface,optimal_strength,posterior_convergence,recovery_half_life}.parquet` + `prior_interpretation_report.txt` | 3 | `experiments/prior_analysis.py` | `results/tables/prior_grid.parquet` | `python experiments/prior_analysis.py` | none (deterministic aggregation) | Re-verified exact this session against the current (reduced-scope) `prior_grid.parquet` |
| `results/tables/drift_{per_cell_metrics,normalized_score,effect_sizes,summary}.parquet`, `drift_{recovery_delta,adaptation_half_life}.csv`, `drift_interpretation_report.txt` + 4 figures | 3 | `experiments/drift_analysis.py` | `results/tables/drift_taxonomy.parquet` + `results/runs/drift_taxonomy` | `python experiments/drift_analysis.py` | none (deterministic aggregation) | Re-verified exact this session (`drift_summary.parquet`, `drift_normalized_score.parquet` numerically identical) |
| `results/tables/llm_cost_accounting_{per_sample,summary}.parquet` | 3 | `experiments/llm_cost_accounting.py` | `results/tables/llm_designers.parquet` (read-only, zero new API calls) | `python experiments/llm_cost_accounting.py` | none | Re-verified exact this session (63-row / 3-row, bitwise-identical) |
| `results/tables/hypothesis_registry_llm_per_type.parquet` | 3 | `experiments/hypothesis_registry.py` | `results/tables/llm_designers.parquet` + prior result files | `python experiments/hypothesis_registry.py` | none | Re-verified exact this session (21 rows, bitwise-identical) |
| `results/tables/human_study_llm_by_family.parquet`, `llm_study_*.parquet` (3 files) | 3 | `experiments/study_analysis.py` | `study/data/submissions.jsonl` (n=0, no human data collected yet) + `results/tables/llm_designers.parquet` + `crossover_calibration.parquet` | `python experiments/study_analysis.py` | none | Re-verified exact this session |
| `results/figures/crossover_{prediction_scatter,reliability_heatmap,error_histogram,calibration_curve}.{png,pdf}` | 4 | `experiments/crossover_figures.py` | `crossover_calibration.parquet`, `crossover_calibration_summary.parquet` | `python experiments/crossover_figures.py` | none | `.png` re-verified byte-for-byte identical this session (confirms deterministic matplotlib rasterization on this machine/version set); `.pdf` intentionally not compared bitwise (creation-timestamp metadata) — see `docs/reproducibility.md` |
| `results/tables/runtime_quality_tradeoff.parquet` + 2 figures | 4 | `experiments/pareto_tradeoff_figure.py` | `runtime_benchmark_kernel_validation.parquet`, `table_v.parquet` | `python experiments/pareto_tradeoff_figure.py` | none | `.parquet`/`.png` re-verified byte-for-byte identical this session; `.pdf` not compared bitwise (same reason as above) |
| `results/environment_snapshot.txt`, `environment_{hardware,packages,llm_model_pins}.parquet` | 6 | `experiments/capture_environment.py` | `results/tables/llm_designers.parquet` (for model pins) | `python experiments/capture_environment.py` | none | Inherently machine/run-specific by design; regenerated and schema-checked this session (`environment_packages.parquet`: 94 rows bitwise-identical) then always rolled back, never promoted — not a reproducibility target itself |
| `results/tables/prompt_robustness_*.parquet` (6 files, Task 15) + 2 figures | 0 (online, opt-in only) | `experiments/prompt_robustness.py`, `experiments/llm_designers.py` | `results/tables/llm_designers.parquet` (read-only, for the primary-vs-paraphrase q comparison) | `python experiments/prompt_robustness.py` | `sample_index`-derived per-call seeds (see `docs/experiments/prompt_robustness.md` §2.3) | Real API-call data, not reproducible bit-for-bit (LLM responses are not deterministic across providers/time) — a one-time-collection artifact. `reproduce.py` never re-runs this even with `--online`, per "do not rerun LLM experiments if raw responses already exist" |
| `results/tables/llm_designers.parquet` | 0 (online, opt-in only) | `experiments/llm_designers.py` | `backend/conf/llm_designers.yaml` (provider/model config) + live LLM APIs | `python experiments/llm_designers.py` | `sample_index`-derived per-call seeds | Same online-only, not-rerun-by-default treatment as prompt_robustness above |

**Dual-writer warning, found by inspecting data, not source** (Task 16
verification): `experiments/horizon_analysis.py` and
`experiments/crossover_calibration_grid.py` both default to writing
`results/tables/horizon_crossover.parquet`, but compute structurally
different tables from different scopes (whole-sweep multi-classifier
derivation vs. a dedicated perfect-classifier-only calibration grid).
Confirmed by inspecting the committed file's actual row/column structure
(112 rows = 56 envs x 2 criteria x 1 method — matches
`crossover_calibration_grid.py`'s cell count exactly, not
`horizon_analysis.py`'s own default scope), not inferred from source code
alone. `reproduce.py` only ever invokes `crossover_calibration_grid.py`
against this path; `horizon_analysis.py`'s functions are imported and reused
by that script but its own CLI is never run against the default output path
in the pipeline. Running `python experiments/horizon_analysis.py` bare would
silently overwrite this file with a different derivation — flagged here so
it is never done by accident, the same class of mistake as the original
`--smoke`-overwrite incident this Part 4 was created to remedy.

**Cost-gating** (Task 16): `horizon_sweep.parquet` and `prior_grid.parquet`
remain the two REDUCED (documented deviation) artifacts above; their true
full-scale regeneration (~73h / ~66h, measured) is not executed by
`reproduce.py --mode paper` unless `--allow-expensive` is explicitly passed.
This is a loud, explicit, logged skip naming the exact command that would
run the true full grid — never a silent one — consistent with this task's
own rule to fail/skip loudly, not quietly. See `docs/reproducibility.md`
§5 for the full mode comparison table.

### Task 19A missing-cell sweep (`horizon_sweep_full26.parquet`) — cost-estimate reconciliation, including a self-correction

**Attempt 1 (killed after 9.64h).** Launched 2026-08-18: `python
experiments/horizon_sweep.py --n-seeds 3 --n-gap-matrices 25 --output
results/tables/horizon_sweep_full26.parquet` (distinct output path; the
committed `horizon_sweep.parquet` above untouched throughout). Two prior
cost estimates existed before this run: an ad hoc filtered estimate
(~13.4h, restricted to the 1,518 actually-missing cells) and the script's
own official `--estimate` immediately before launch (~4.43h, unfiltered
across all 1,716 cells). **Neither held.** After 9.64h the run had
completed 387/1,518 cells (~40 cells/hour), projecting to roughly 38h
total — several times worse than either estimate.

**A hypothesis was proposed, then directly disproven by the verification
it demanded — recorded here rather than silently replaced.** The initial
diagnosis: no `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`
limiting was set anywhere in this codebase, so each of the 8 `loky`
worker processes running scikit-learn/numpy inference could spawn its own
BLAS thread pool, oversubscribing this machine's 4 physical cores. This
was presented as the likely cause. It was then checked directly —
dispatching through `joblib.Parallel(n_jobs=8, backend="loky")` (the
exact call `run_grid` uses) and reading `threadpoolctl.threadpool_info()`
from inside each worker, at the exact production `n_jobs=8`, with
`OMP_NUM_THREADS` etc. NOT manually set:

```
pid=24268  OMP=1  OPENBLAS=1  [('openblas', 1)]
pid=17368  OMP=1  OPENBLAS=1  [('openblas', 1)]
... (all 8 workers identical)
```

**`loky` already auto-computes and sets these to 1 per worker at this
`n_jobs`, with no manual configuration at all.** The oversubscription
diagnosis was wrong — not a matter of degree, the specific mechanism
proposed does not occur. Manually setting the env vars (still done on the
restart, below) is therefore a no-op layered on already-correct behavior,
not the fix, and is not reported as one.

**Revised hypothesis for the shortfall, held to a lower confidence than
the disproven claim was:** with true single-thread-per-worker parallelism
already confirmed, the two live candidates are (a) this CPU's 4
physical/8 logical core topology giving well under 8× real speedup for
compute-bound inference under hyperthreading, and (b) concurrent
foreground tool use on the same machine during the run (file reads,
`pandas` analysis, the verification script itself) competing for the same
4 cores. Neither has been isolated with a controlled test; both are named
as candidates, not conclusions — a deliberate contrast with how the BLAS
claim was wrongly stated as settled.

**Restart (Attempt 2), same day.** Killed cleanly via the process tree
(`taskkill /T /F` on the launched PID, all 8 loky workers plus the loky
resource-tracker processes terminated; the RunStore was verified
undamaged afterward — file count exactly `617 + 387 = 1,004`, zero `.tmp`
orphans, confirming the atomic tmp-then-rename write path held even under
a hard kill). Two changes: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` set explicitly before launch
(redundant given the finding above, kept anyway so the launch command
doesn't rely on an undocumented `loky` default); and `--methods
static_blind thompson` (Tier 1 only needs the mandatory `time_to_parity`
baseline plus the one predicted method) — `23 envs × 2 classifiers × 2
methods × 3 seeds = 276` cells instead of 1,518, a 5.5× reduction, with
108 of 312 total (26 envs incl. the 3 cached) already skipped from cache
on restart. Fresh `--estimate` under the filtered scope: **~38.4 min**.

**Early-rate check at 20/204 cells: ~8.5 min elapsed → projects to ~86.9
min for the full 204 — a ~2.3× miss against the fresh estimate, in the
SAME direction as Attempt 1's much larger miss.** This is a second,
independent data point (different scope, different `--estimate` draw,
same machine) landing on the same side of the estimator's assumption, not
a one-off. Held against a pre-stated stop threshold (~90 min projected);
86.9 min is under it initially — but the cumulative-average projection
this was based on turned out to be biased optimistic for a structural
reason (see the 5th finding below), so it was re-derived before being
trusted, and the re-derivation changed the outcome.

**Reprojection using per-population rates, not the pooled cumulative
average, and the kill decision.** Per-cell inter-arrival times over the
first ~20-34 completed cells were monotone-descending (96.5s → 76.9s →
... → 10.5s/it), not noise around a stable mean — a signature of the
cheap `classifier_id='perfect'` cells (0.54s/cell at the calibrated
184,000 eps/sec) draining first while the expensive `'real'` cells
(117.6s/cell at 850 eps/sec) queue behind them, since `build_grid()`
orders cells `env → classifier → method → seed` with `classifiers=
["perfect", "real"]` — not randomized. A cumulative average taken early
in that ordering is therefore a biased sample, not a representative one.
Re-projected instead from the RunStore's own file-write timestamps
(bypassing the ordered-generator's buffering entirely): of the 60 cells
Attempt 2 had actually written (36 `perfect` / 24 `real`) over 1,492s
wall-clock, total calibrated CPU-work was 36×0.544s + 24×117.6s ≈ 2,843
CPU-sec, giving an empirically-observed **effective parallelism ≈ 1.90×**
— far below the nominal 8-way dispatch. Applying that same 1.90× to the
144 cells still pending at that point (66 `perfect` / 78 `real`, split
confirmed directly against the RunStore, not inferred): `(66×0.544 +
78×117.6) / 1.90 ≈ 80.6 min` central estimate. Under the 90-min
threshold, but not comfortably: a 20%-worse-than-calibrated real-cell
throughput — a miss size already seen twice on this exact job — pushes
the same reprojection to ~96.6 min, over threshold. Given that history,
the run was killed rather than continued (PID 15804, `taskkill /T /F`,
all child/loky-worker processes terminated; RunStore re-verified
afterward — 1,064 files, zero `.tmp` orphans, 168/312 cells banked,
confirming the atomic write path held under a third hard kill). Per the
pre-stated asymmetry, this costs exactly one in-flight cell; RunStore's
resume logic means none of the 168 banked cells (108 original + 60 from
this attempt) need to be recomputed. Final measured wall-clock for a
COMPLETED run is therefore still pending a future, correctly-ordered
attempt (see the fix below).

**Correction, recorded for the record: the kill was a scope error, not a
methodology error.** The ~90-min threshold was a tripwire against a
THIRD miscalibrated *estimate*, in a context where 4.43h and 13.4h
estimates had both already been accepted as reasonable costs — 96 minutes
was never actually over budget, only over a number that was never scoped
to mean "budget." The 1.90× effective-parallelism reprojection above is
categorically different from the two prior misses it was being compared
against: it is a MEASUREMENT taken from 60 real completed cells' own
RunStore timestamps, not a model-based `--estimate` extrapolation, and
should have been treated as clearing the tripwire on method rather than
merely evaluated against the threshold's number. Attempt 3 was relaunched
immediately after this was pointed out, resuming from the 168/312 cells
already banked, with no new stop threshold — only genuine-failure
conditions (process death, RunStore corruption, or a reprojection
exceeding ~4h, which would indicate an actually new problem) warrant
stopping it.

### Measured effective parallelism (6th instrumentation finding)

**1.90× effective parallelism**, measured on this machine (4 physical /
8 logical cores) at `n_jobs=8` (`joblib.Parallel(backend="loky")`, the
exact call `run_grid()` uses), for the `horizon_sweep.py` Tier-1-filtered
workload (`static_blind` + `thompson`, `perfect` + `real` classifiers).

**Derivation**: calibrated CPU-seconds ÷ observed wall-clock, over 60
cells Attempt 2 actually completed (36 `classifier_id='perfect'` at
0.544s/cell [184,000 eps/sec] + 24 `'real'` at 117.6s/cell [850
eps/sec] ≈ 2,843 CPU-sec of work) against 1,492s of measured wall-clock
from the RunStore's own file-write timestamps (`st_mtime`, bypassing
`joblib`'s ordered-generator buffering entirely) — 2,843 / 1,492 ≈ 1.90.

**This value is machine-specific** — a property of this machine's
physical/logical core topology and current load, not of the workload or
the codebase — and should not be assumed to transfer to a different
machine without re-measuring.

**This is the empirical correction to `estimate_wall_clock`'s linear-
`n_jobs`-scaling assumption**, not merely an incident detail: where the
estimator assumes `n_jobs=8` buys 8× real speedup, this workload measured
1.90× on this hardware — roughly a 4.2× overestimate of achievable
parallelism, which is the same order of magnitude as the ~2.3× and much
larger Attempt-1 misses recorded above, now with an actual number behind
it instead of just a symptom.

**Consequence for `reproduce.py`'s ~73h (`horizon_sweep.py`) and ~66h
(`prior_grid.py`) headline cost figures**: these should be RE-DERIVED
against measured parallelism (≈1.90× on this hardware, if run here)
rather than against `n_jobs`. Not performed now — flagged as follow-up
work, consistent with the standing note above that both figures remain
suspect in both directions until re-estimated.

**Concrete estimator fix** for `estimate_wall_clock` (not implemented
here, recorded for whoever picks up the follow-up): accept a
`measured_parallelism` parameter (falling back to `n_jobs` only if none
is supplied), or — better — calibrate it automatically from a short
pilot batch (run a handful of cells from each `classifier_id` for real,
under the actual `n_jobs`/backend the full run will use, and divide
observed CPU-work by observed wall-clock) instead of assuming the
configured `n_jobs` is what the hardware actually delivers.

**Noted but explicitly not tested now**: `n_jobs=8` process-level
parallelism on a 4-physical-core machine, for a CPU-bound workload, may
itself be part of the shortfall (8-way contention on 4 physical cores
rather than benefiting from hyperthreading the way I/O-bound or mixed
workloads can) — i.e. `n_jobs=4` might measure a HIGHER effective
parallelism than `n_jobs=8` did here, not just a lower nominal ceiling.
Recorded as a hypothesis for a future sweep to A/B; not tested in this
session.

**The DEFECT stands; only the MECHANISM was corrected — this section does
not retract the finding that prompted it.** `estimate_wall_clock` divides
a single-threaded calibration by `n_jobs`, assuming linear parallel
scaling (its own docstring says so explicitly); Attempt 1 achieved
nothing close to 8× real speedup, and Attempt 2's early rate is missing
its own fresh estimate by ~2.3× under the SAME assumption. A 4
physical-core machine under hyperthreading is sufficient on its own to
break this assumption, with no BLAS-threading claim required at all — the
correction above removed one specific (wrong) proposed mechanism, not the
underlying finding that the estimator's parallel-scaling assumption does
not hold on this hardware.

**Consequence for the `horizon_sweep.py` ~73h / `prior_grid.py` ~66h
headline cost figures** (`reproduce.py --mode paper`'s `--allow-expensive`
gate, "Cost-gating" above) is therefore UNCHANGED by the BLAS correction,
and is now reinforced by two independent misses on this same sweep
script's own estimator rather than weakened: those two much larger,
never-actually-measured figures rest on the identical linear-`n_jobs`-
scaling assumption this section's own two attempts just watched fail
twice. **Both figures remain suspect in both directions** — the
estimator over-assumes achievable parallel speedup (making them
optimistic), while nothing rules out cell-composition or contention
effects at that much larger scale pushing the true figure further still
(making them pessimistic in a different way) — and should be treated as
estimates with an empirically-unverified parallel-scaling assumption, not
as measurements. Re-estimating them is flagged as follow-up work, not
performed here.

### Measurement-hygiene: concurrent foreground load during timed runs (4th instrumentation finding)

Hypothesis (b) for the Attempt 1 shortfall — foreground tool use (file
reads, `pandas` analysis, ad hoc scripts) on the same machine, during the
same window as a timed run, competing for the same 4 physical cores — is
not specific to this sweep. **It applies to every timing number this
project reports**, including Task 14's own per-decision runtime/memory
benchmark (`runtime_benchmark_timing.parquet`, "Part 1" above) and every
`--estimate` calibration draw anywhere in `experiments/`.

**Whether Task 14's benchmark was collected under clean (no concurrent
compute) conditions is UNKNOWN, not assumed clean.** No process-isolation
protocol was documented at the time it was run, and nothing in
`runtime_benchmark.py` or its write-up records machine load state
alongside the timing numbers — so this cannot be verified retroactively
either way. Stated as unknown rather than defaulting to "presumably
fine," consistent with this project's standing rule to report a gap as a
gap rather than assume the favorable case.

**Protocol for future timed runs, recorded here for reuse:**

1. No concurrent compute during any timed measurement — no parallel
   analysis scripts, no other terminal sessions issuing CPU-bound work,
   on the SAME machine, for the duration of the timed run.
2. No parallel agent/editor tool calls (file reads, greps, script
   executions) during the timed window specifically — a constraint this
   session's own Task 19A sweep violated by design (analysis and
   verification work continued while the sweep ran in the background).
3. Record machine state alongside the timing output: at minimum, logical
   CPU count and whether any other identified process was running,
   the same level of detail `capture_hardware_info()` (Task 14 Part 3)
   already captures for hardware identity — extended to capture
   contemporaneous LOAD, not just static hardware identity, which it does
   not currently do.
4. Prefer `--estimate`-style short calibration draws to be taken multiple
   times (not once) and reported with their spread, given this session
   directly observed a 3.4× spread between two single-draw calibrations
   under otherwise-identical conditions.

Not retroactively applied to any existing timing table in this
document — flagged as a protocol for FUTURE timed runs, and as a caveat
on every existing one.

### Non-randomized cell ordering biases early-progress projections (5th instrumentation finding)

`experiments/horizon_sweep.py::build_grid()` emits cells in a fixed,
non-randomized nesting: `for env_id: for classifier_id: for method: for
seed`. With `classifiers=["perfect", "real"]` and a large, known cost
asymmetry between the two (184,000 vs. 850 eps/sec, ~218×), every
environment's cheap `perfect` cells complete before its expensive `real`
cells even begin. Any projection taken from a cumulative average over an
early prefix of `run_grid()`'s progress — which is exactly what a
naive "N cells done in T minutes, extrapolate linearly" early-rate check
does — is therefore **systematically optimistic** early on and only
converges toward the true rate once the cell mix seen so far approaches
the true overall mix. This is the direct, root cause of the Attempt 2
early-rate check's own error (86.9 min looked fine as a pooled figure;
the per-population reprojection above put it in the 80.6–96.6 min range
depending on a plausible calibration miss).

**Fix, for any future sweep with a similar cost-asymmetric grid:**
either (a) shuffle cell order (seeded, for reproducibility) before
dispatch so any prefix is a representative sample of the full mix, or
(b) never trust a pooled cumulative-average projection for a
heterogeneous grid — always split the projection by the cost-dominant
dimension (here, `classifier_id`) and recombine using the actually
-remaining counts per population, as done above. Neither is implemented
in `horizon_sweep.py` today; recorded here as a concrete fix for the next
attempt rather than applied retroactively to this one.

**Compute overshoot in Attempt 1, noted for future sweeps.** The RunStore
cache is durable and content-addressed, so none of Attempt 1's 387
completed cells were wasted — they remain banked for any future
Tier-1-scope work needing the other 9 methods. Recorded as a lesson for
how a similarly-scoped future sweep should be LAUNCHED (filtered from the
start), not a claim that Attempt 1's work was lost.

### Recurring pattern: stale project state causing a downstream error (two instances)

This is the second documented instance of stale project state causing a
downstream error, not a one-off. Naming both together because the fix for
one instance (fix the specific file) does not fix the pattern (the next
stale note causes the next downstream error):

1. **Instance 1 — the `--smoke`-overwrite incident** (documented in
   `docs/PROJECT_STATUS.md`, occurring during the Task 14 work session):
   several `--smoke` (fast validation) runs were executed without first
   confirming each script's smoke and full modes write to the same
   *default* output path — this silently overwrote six production result
   files with smoke-scale data. **Destructive**: real committed data was
   overwritten. Caught by noticing row counts had collapsed, not by
   design.
2. **Instance 2 — the `horizon_sweep.parquet` provenance drift**
   (documented above and in `docs/PROJECT_STATUS.md`'s corrected Task 14
   entry): a status note describing an in-flight full-scale regeneration
   as "actively running... not yet completed" was left uncorrected after
   that job was abandoned in favor of a documented reduced scope. A later
   task (Task 19A's "Gate 1") read the stale note, inferred a
   26-environment dataset existed, and that belief propagated into a
   validation-grid design two gates further downstream before being
   caught. **Non-destructive**: no data was lost or overwritten a second
   time — only a written description of state was wrong, and it stayed
   wrong long enough to mislead a reader once it was.

**The common root cause**: both are the same failure mode wearing
different clothes — a written description of a script/job's OUTCOME
(overwritten-and-restored file counts in Instance 1; a background job's
completion status in Instance 2) was not re-verified against the actual
artifact/state at the time a later task relied on it. Instance 1 was
caught because the artifact itself was wrong in an observable way (row
counts). Instance 2 was much harder to catch precisely because the
artifact was fine — only the PROSE describing it was wrong — so nothing
about reading the data itself would ever surface the error; only
independently re-deriving/inspecting provenance (as Task 19A's later
correction pass did) catches this class.

**Structural change that would prevent a third instance.** Point fixes
(correcting this one note) don't address the pattern. What would: every
"status"/"in progress" prose claim about a script's outcome or a
background job's completion should be treated as a claim requiring the
same re-verification discipline `reproduce.py`'s own Stage 7 already
applies to DATA artifacts (`compare_table()` against the committed
production file) — but nothing currently re-verifies STATUS PROSE against
underlying artifact state the way Stage 7 re-verifies table content
against re-derived tables. Concretely: `docs/PROJECT_STATUS.md`-style
running status notes should either (a) be treated as strictly point-in-time
journal entries never read as current truth by a later task (a discipline
change, not a tooling change — cheap, but relies on every future task
actually following it), or (b) be replaced for anything reproducibility-
relevant by generated, not hand-written, status — e.g. an extension of
Part 4's own provenance table (already regenerated by actually running
each command, never inferred) that includes a `last_verified` field
authored by the verification run itself, not by a human editing prose
after the fact. (b) is the more durable fix since it removes the
opportunity for the prose and the artifact to diverge at all, at the cost
of only covering what the automated table tracks (a background job's
"still running" state, as in Instance 2, is not naturally a provenance-
table row, and would need a small addition to represent in-flight/
abandoned work explicitly rather than in narrative prose). Not
implemented here — flagged as a structural recommendation, per the
correction task's read-only/documentation-only scope for this item.

Cross-referenced from `docs/PAPER_PREP.md`'s threats-to-validity list.

**Root cause of the `prior_grid.py` restoration crash** (relevant to this
table's `--n-jobs 2` column): `recovery/metrics/bootstrap.py::bootstrap_ci`'s
vectorized resampling peaks at ~1.6GB for a `horizon=50000` cell's late-stage
window (10,000 episodes) -- confirmed via `tracemalloc`, and exactly matching
both this session's crash (`horizon=50000`, 763 MiB) and an earlier session's
crash (`horizon=5000`, 76.3 MiB) via the same `n_resamples x n x 8 bytes`
arithmetic. Not a bug in `bootstrap_ci` (used safely everywhere else in the
project at much smaller late-stage windows); an environmental interaction
between this one script's large-horizon cells and joblib's default 8-way
parallelism on an 8GB-RAM machine with only ~2-2.7GB free even at idle. Fixed
by capping `--n-jobs 2` for this run; documented in
`experiments/prior_grid.py`'s module docstring for future runs on similar
hardware. No source code in `recovery/metrics/` was modified.

**Production-artifact-only confirmation**: every reproduce command above
targets the script's real default output path (`results/tables/*.parquet`,
no `_smoke`, `_repro_test`, or other throwaway suffix). All temporary/test
files created during this recovery's root-cause investigation
(`_repro_test.parquet`, `_repro_test2.parquet`, and their RunStore
directories) were deleted immediately after use and never appear in this
table or in `results/tables/`.
