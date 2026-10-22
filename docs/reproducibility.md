# Reproducibility Guide

Task 16: the single entry point (`backend/reproduce.py`) that regenerates
every experiment analysis, publication figure, publication table, and
auto-generated documentation artifact behind the IEEE paper, plus what it
deliberately does *not* automate and why.

If you only read one section, read "Two modes" and "Common failure modes."

---

## 1. Repository setup

```
git clone <repo>
cd failure-aware-agent
```

Layout that matters for reproduction:

- `backend/` — everything runnable: `experiments/` (one script per
  table/figure/analysis), `recovery/` (the shared kernel: episode runner,
  metrics, policies, classifiers, RNG), `results/` (committed outputs —
  tables, figures, RunStore caches, legacy phase evidence), `data/`
  (`classifier_model.joblib`), `tests/`.
- `docs/` (repo root) — this file, the provenance map
  (`docs/experiments/instrumentation.md`), per-experiment write-ups.
- `study/` — the (currently empty; no human data collected yet) human-study
  web instrument.

Everything below assumes commands are run **from `backend/`**, using
**the project's own virtual environment interpreter** — not whatever
`python`/`python3` happens to be first on `PATH`. On this project's
reference machine, the system default `python` resolved to an unrelated
3.14 interpreter with none of this project's dependencies installed;
`reproduce.py`'s Stage 1 checks for exactly this mistake and fails loudly
with the correct command rather than producing confusing errors three
stages later.

```
# Windows
..\.venv\Scripts\python.exe reproduce.py --mode validation

# POSIX
../.venv/bin/python reproduce.py --mode validation
```

## 2. Environment creation

Two files, two purposes:

| File | Purpose |
| --- | --- |
| `backend/requirements.txt` | Loose, human-maintained top-level dependency list (unpinned — always existed this way in this repo). |
| `backend/requirements.lock.txt` | **The reproducibility target.** An exact `pip freeze` (94 packages) of the actual `.venv` every experiment in this repository was run under. `pip install -r requirements.lock.txt` reproduces the full dependency closure, not just top-level names. Regenerate with the header comment's own command inside that file. |

Recorded exactly, not estimated:

- **Python**: 3.13.11 (CPython, MSC v.1944 64-bit AMD64)
- **OS**: Windows 11 Home Single Language, 10.0.26200, 64-bit (AMD64)
- **Hardware**: Intel Core i5-9300H @ 2.40GHz, 4 cores / 8 logical processors, 8GB RAM
- **LLM providers/models**: groq-llama (`llama-3.1-8b-instant`), google-gemini (`gemini-flash-lite-latest` — an alias, not a pinned dated model; Google's API never discloses a resolved version, a genuine reproducibility gap for that one family only), nvidia-mistral-nemotron (`mistralai/mistral-nemotron`)
- **Experiment dates**: all 63 LLM study calls on 2026-08-04, 09:16–09:26 UTC; every offline sweep/grid's exact date is in its RunStore cache's file mtimes and in `docs/experiments/instrumentation.md`'s provenance table
- **RNG seeds**: every experiment script's `DEFAULT_BASE_SEED` is listed per-artifact in `docs/experiments/instrumentation.md`'s provenance table (e.g. `stationary_comparison.py`=123, `matrix_family_sweep.py`=2026, `classifier_noise_sweep.py`=123, `drift_taxonomy.py`=20\_000, `horizon_sweep.py`/`crossover_calibration_grid.py`=4000)

Full detail (all 94 pinned packages, hardware capture, LLM model pins) lives
in `results/environment_snapshot.txt` and
`results/tables/environment_{hardware,packages,llm_model_pins}.parquet`,
produced by `experiments/capture_environment.py` (Stage 6 of the pipeline).
**Re-running Stage 6 on a different machine will legitimately produce a
different snapshot** — that file is a record of the machine it ran on, not
a reproducibility target itself, and `reproduce.py` never auto-promotes a
freshly captured snapshot over the committed one for exactly this reason.

## 3. Required API keys

None, for reproducing the paper's analysis, figures, and tables.
`backend/.env.example` documents `GROQ_API_KEY` (also `GOOGLE_API_KEY`,
`NVIDIA_API_KEY` are read if present, see `backend/conf/llm_designers.yaml`)
— these are needed **only** to re-collect fresh LLM-study data
(`llm_designers.py`, `prompt_robustness.py`), which the pipeline does not
do by default (see §4). If a key is present but you don't want any live
calls, unset it or leave `reproduce.py`'s default (`--online` off).

## 4. Offline vs. online workflow

**Offline (default, no keys needed)** — everything except a first-time
collection of LLM study data: all sweeps/grids, all analyses, all figures,
all publication tables, environment capture. This is what
`python reproduce.py --mode validation` and
`python reproduce.py --mode paper` do by default.

**Online (opt-in via `--online`, requires keys)** — re-collecting raw LLM
responses from scratch. The pipeline's Stage 0 does **not** call
`llm_designers.py`/`prompt_robustness.py` even with `--online`, because
`results/tables/llm_designers.parquet` and the 6
`prompt_robustness_*.parquet` files already exist as committed raw data,
and this task's own rule is "do not rerun LLM experiments if the raw
responses already exist." To force a genuine, paid, non-deterministic
re-collection, run those two scripts directly — that is a deliberate
manual action, never something `reproduce.py` does on your behalf.

This means: **someone who clones this repository with no API keys at all
can still regenerate every analysis, every figure, and every table** —
confirmed by actually running the pipeline offline end-to-end (§6).

### Phase 1 baseline table (legacy harness): `experiments/regenerate_phase1_summary.py`

| | |
|---|---|
| What | Regenerates `results/phase1/` (11 methods, 500 episodes × 50 trials) through the legacy `src/evaluation/harness.py`. It reproduces the 2026-07-28 run exactly (verified 2026-10-21). |
| Seeds | `base_seed=123`. Per trial: `trial_seed = 123 + 1000·trial`; the type stream is seeded with `trial_seed` and the outcome stream with `trial_seed + 7`. |
| Offline | Yes. `GROQ_API_KEY` is forced empty, and the One-Shot LLM baseline is pinned to its recovered original 2026-07-28 answers (see `docs/PAPER_PREP.md` §6, 2026-10-21 entry). |
| Why pinning is required | All methods share one outcome stream, so with any other One-Shot answers none of the 11 rows reproduce. |
| One-Shot row | Kept at its original numbers, per `results/phase1/comparison_summary.provenance.json`. |
| Run | `python experiments/regenerate_phase1_summary.py` (full) or `--smoke` (writes to `results/phase1_smoke/`). |
| `reproduce.py` | Stage 2 still treats `results/phase1/` as legacy phase evidence. This script is not yet a pipeline stage. |

### Live data source (not regenerable): the drift-atlas probe (Task 25, C3)

| | |
|---|---|
| What | `probe/run_probe.py` calls 24 real APIs every 6 h from GitHub Actions (`.github/workflows/drift-probe.yml`). |
| Seeds | **None.** The code uses no randomness; the measured world is not deterministic. |
| Dataset | The `data` branch, as of a given commit, filtered to `targets_sha256` = the frozen value in `docs/experiments/drift_atlas_preregistration.md` §1. Cite that commit hash to pin the exact observations used. |
| Re-running | Re-running produces **new** observations of a changed world, not a reproduction. Local runs write to a temp dir and never enter the dataset. |
| `reproduce.py` | **Not registered yet, deliberately.** Calling live APIs would break the offline guarantee above. Task 25 part 2's analysis (offline, over a pinned `data`-branch commit) will be registered in `reproduce.py` with that commit hash as its input. |
| Tests | `backend/tests/test_probe.py` (mocked network) runs in the normal suite. |
| Keys | `GROQ_API_KEY` and `GOOGLE_API_KEY` as GitHub Actions secrets. NVIDIA's model list is public, so no key is needed. |

### Live data source (not regenerable): live pilot v3 (`experiments/run_live_v3.py`)

| | |
|---|---|
| What | One v3 session runs all six injection blocks (T, S, O2, O5, O20, D) against the six real tools, with one uniformly drawn recovery strategy and a paired resend baseline per captured failure. Definitions: `docs/live_pilot/strategy_definitions_v3.md` (frozen, `c1705bf`). Hypotheses and analysis: `docs/live_pilot/v3_preregistration.md` (`4723b2b`). |
| Seeds | Session `s` uses `session_seed = 231000 + s`; `s = 0` is reserved for smoke runs and is never data. Named streams come from `SeedSequence(session_seed, spawn_key=(stream, *keys))` (`recovery/live/v3/seeds.py`): `block_order`, `selection` and `jitter` per failure, and `injection`, `resend`, `outage` and `degradation` per block. Each derived seed is recorded on the record that used it. |
| Run | Data: `python experiments/run_live_v3.py --store results/live_pilot_v3 --session-index <s>`, with s = 1, 2, … in order and never re-run. Smoke: `--smoke --store <dir>_smoke` (each block runs until it has ≥ 3 injected failures, at most 6 passes). |
| Stopping | The count-only `recovery/live/v3/stopping.py`: stop when every cell has ≥ 30 included attempts, or after 60 sessions (pre-reg §2). |
| Report | `python experiments/report_live_v3.py --store <store>` (offline, descriptive). |
| Analysis | `python experiments/analyse_live_v3.py --store results/live_pilot_v3 --out <dir> [--calibration <fa.json>]`. This is the registered P1–P3 analysis, committed alone in `0cc3bac` (pre-reg D-7). It is offline, with bootstrap seed 23001 and false-alarm sample seed 23002. |
| Deviations | D-2 to D-6 are applied and logged in pre-reg §11: the Layer-2 selector, the family split, the judge input, read timeouts and pacing. Selector before/after: `experiments/selector_empty_rate_v3.py` (a live measurement). |
| Libraries | The plausible-wrong libraries in `backend/recovery/live/v3/data/` were captured once by `experiments/capture_v3_libraries.py` and are hash-checked on load. They are never re-captured. |
| Re-running | Re-running produces **new** observations, not a reproduction. Cite the store's session manifests (`v3_sessions/*.json`, with code commit and seeds). |
| `reproduce.py` | **Registered (stage 3).** `analyse_live_v3.py` runs offline over `results/live_pilot_v3` once that store exists; until then the stage logs a loud skip. Collection itself is live and never run by the pipeline. |
| Tests | `backend/tests/test_live_v3_*.py` and `test_live_v1v2_golden.py` (no network). |
| Keys | `GROQ_API_KEY` in `backend/.env`. |

## 5. The two modes

| | `--mode validation` | `--mode paper` |
| --- | --- | --- |
| Purpose | Fast correctness check, CI-suitable | Full paper-scale regeneration |
| Base sweeps (`horizon_sweep.py`, `prior_grid.py`) | `--smoke` (<60s each) | Full default scope, **cost-gated** (see below) |
| `drift_taxonomy.py` | `--smoke` (<90s) | Full — cheap in practice, its RunStore cache (12,812 files) is already complete, so a full run is a cache hit, not a re-simulation |
| `stationary_comparison.py`/`matrix_family_sweep.py`/`classifier_noise_sweep.py`/`matrix_robustness.py` | `--smoke` | Full |
| `crossover_calibration.py`/`crossover_extension.py` | **Skipped, documented** — their real upstream (`crossover_calibration_grid.py`) has no `--smoke` mode; feeding them smoke-scale data produces near-zero finite crossovers and a correct, loud `ValueError`, not a bug | Full |
| `runtime_benchmark.py` (paper only) | not run | Full (n=5,000 decisions/policy) |
| Measured wall-clock (offline stages, this machine) | ~2–3 minutes | Minutes for everything except the two cost-gated sweeps (see below) |
| Writes to production paths? | Never — always an isolated validation directory | Never automatically — compares against production, never overwrites (Part 8 below) |

**Cost gate**: `horizon_sweep.py`'s and `prior_grid.py`'s true full-default
grids are measured (via `horizon_sweep.py --estimate` and a
`build_grid()`-based calculation respectively — not guessed) at **~73
hours** and **~66 hours** on this hardware. `--mode paper` does **not**
execute these by default; it prints a loud, explicit, non-silent skip
banner naming the exact command that would run the true full grid, and
uses the already-committed production tables (which are themselves a
documented reduced-but-real scope — see
`docs/experiments/instrumentation.md`) for every downstream stage. Passing
`--allow-expensive` forces the true full run. This is a deliberate,
documented exception to "regenerate everything," not an oversight — see
§9 Limitations.

## 6. Reproducibility verification performed this session

Ran `python reproduce.py --mode validation` end-to-end from this checkout:
Stages 1–7 all completed; `--smoke` runs of the seven affected scripts
finished in seconds each; the two smoke-scale-incompatible calibration
scripts were skipped with the documented reason above; every other stage
(analyses, both separable figure scripts, all three headline tables plus
matrix-robustness, environment capture) executed for real and its output
was compared against committed production via `compare_table()`
(numeric-tolerant, `rtol=1e-9`) or a byte hash for text files.

Ran `python reproduce.py --mode paper` (offline stages only, i.e. without
`--allow-expensive`) and diffed every regenerated artifact whose script
supports non-default output redirection against the committed production
file. Results:

- **Exact or numeric match** (deterministic offline scripts, same seeds,
  same code): `table_vi.parquet`, `table_vii.parquet`,
  `matrix_robustness_*.parquet`, `horizon_crossover.parquet`
  (via `crossover_calibration_grid.py`, RunStore cache hit),
  `crossover_calibration.parquet`, `crossover_lookup.parquet`,
  `drift_summary.parquet`, `drift_normalized_score.parquet`,
  `prior_optimal_strength.parquet`, `prior_negative_transfer_surface.parquet`.
- **`table_v.parquet`: matched except for the documented, permanent
  exception.** All non-"One-Shot LLM" rows matched exactly. The "One-Shot
  LLM" policy's rows cannot be independently re-verified by re-execution —
  full mode makes one real, non-deterministic, paid Groq call per episode
  with no stored response cache to replay. Rather than spend money for an
  answer that is inherently not reproducible, this pipeline forces the
  same offline fallback `--smoke` already uses (`GROQ_API_KEY=""`) even in
  paper-mode verification, and reports this row set as an explicit,
  documented, non-pass/fail exclusion rather than a silent skip or a false
  pass.
- **`environment_snapshot.txt` / `environment_*.parquet`**: regenerated
  and schema-checked, then unconditionally rolled back — expected to
  differ run-to-run by design (machine/timestamp capture), not a
  reproducibility target.
- **No unexplained numeric drift found.** Every difference observed maps
  to one of the two documented, legitimate causes above (LLM
  non-determinism, environment capture being inherently point-in-time).

**Not verified this session** (by design, per the cost gate in §5):
`horizon_sweep.parquet` and `prior_grid.parquet`'s true full-scale forms.
The committed production files for both are themselves reduced-but-real
scope (see `docs/experiments/instrumentation.md`), so there is no
full-scale artifact to diff against even if the ~73h/~66h runs were
executed — reproducing them exactly would require reproducing the same
documented scope reduction, which the pipeline's `--allow-expensive` flag
does not currently parameterize (a known gap, tracked in §10).

**Figures**: `.png`/`.pdf` outputs were not compared bitwise. PDF export
embeds a creation timestamp in its metadata, so byte-identical PDFs across
two separate runs are not expected even with identical data — this is the
"legitimate source such as font rendering" case Task 16 itself anticipated,
not numerical drift. PNG rasterization is typically deterministic given
identical matplotlib/font versions, but this was not independently
confirmed pixel-by-pixel this session; only that regeneration succeeds
without error and produces a file of plausible size.

## 7. Expected runtime (this machine: i5-9300H, 8GB RAM, 8 logical cores)

Measured directly from two clean `reproduce.py` runs this session (not
estimated):

| Stage | Validation mode (measured) | Paper mode (offline, no `--allow-expensive`) |
| --- | ---: | ---: |
| 1–2 (env + input verification) | <1s | <1s |
| 3 (analyses) | ~2.9 min (33.5s horizon_sweep smoke + 25.0s prior_grid smoke + 12.8s drift_taxonomy smoke + 3.3s horizon_analysis + 41.9s prior_analysis + 9.2s drift_analysis + 3.0s llm_cost_accounting + 21.5s crossover_reliability + 6.6s hypothesis_registry + 14.5s study_analysis) | Longer — includes `crossover_calibration_grid.py`'s cache-warm 2,760-cell grid and full-scale `prior_analysis.py`/`drift_analysis.py` bootstrap CIs; not fully re-timed end-to-end this session (individual sub-stages confirmed to complete, see §6) |
| 4 (figures) | ~41s (32.0s crossover_figures + 8.6s pareto_tradeoff) — note: one earlier run measured crossover_figures.py at 801.7s under heavy concurrent system load (VS Code/browser/other processes active); the clean re-run at 32.0s is the representative figure | similar, offline |
| 5 (tables) | ~43s (5.7s table_vi + 8.9s table_vii + 28.0s matrix_robustness; table_v smoke <10s by its own guarantee) | Longer at full scale (`matrix_robustness.py` full: 200+150 matrices × 30 seeds; `runtime_benchmark.py` adds 5,000 decisions × 14 policies, paper mode only) |
| 6 (docs) | ~12s | ~12s |
| 7 (validate) | <1s | <1s |
| **Total** | **~4.5 min, measured** | **On the order of 15-20 min for the offline stages (not independently re-timed end-to-end as a single wall-clock number this session; every sub-stage's individual completion was confirmed instead — see §6)** |
| `--allow-expensive` (true full `horizon_sweep.py` + `prior_grid.py`) | n/a | **+~139 hours** (~73h + ~66h, measured via `--estimate`/`build_grid()`, not guessed) — not run in this verification pass |

## 8. Hardware assumptions

None of the offline pipeline requires anything beyond a single modern
laptop-class CPU; nothing here uses a GPU. `prior_grid.py` at
`horizon=50,000` under joblib's default full parallelism can transiently
require several GB of RAM (documented in that script's own module
docstring, with the exact `bootstrap_ci` memory-scaling arithmetic and a
`tracemalloc`-measured 1.6GB peak) — on an 8GB machine, pass a lower
`--n-jobs` if regenerating that specific artifact at full horizon under
sustained high concurrency.

## 9. Limitations

- **table_v.parquet's live-LLM row** is fundamentally not bit-for-bit
  reproducible without spending money on a non-deterministic API call —
  a property of the experiment design (Task 5), not a pipeline defect.
- **`horizon_sweep.py`/`prior_grid.py` full-scale forms** were not
  re-verified this session (cost-gated, ~139h combined) — the committed
  production files remain the ones restored and verified during the
  separate repository-integrity recovery pass that preceded this task, at
  a documented reduced-but-real scope, not this pipeline's own full
  default.
- **PDF/PNG figure bitwise equality** was not independently confirmed —
  only that regeneration succeeds and produces a plausible file.
- **`docs/experiments/instrumentation.md`'s provenance table** is
  hand-maintained (an editorial task), not auto-generated by Stage 6 —
  scoped out intentionally, not an oversight.
- **`horizon_analysis.py` vs `crossover_calibration_grid.py`** both
  default to writing `results/tables/horizon_crossover.parquet`, but
  produce structurally different tables (confirmed by inspecting the
  committed file's row/column structure, not inferred from source alone —
  it matches `crossover_calibration_grid.py`'s 56-env/2-criterion/1-method
  cell structure, not `horizon_analysis.py`'s own default scope). This
  pipeline only ever invokes `crossover_calibration_grid.py` against that
  path; `horizon_analysis.py` is used purely as an imported function
  library. Do not run `horizon_analysis.py`'s bare CLI against its default
  output path — it will silently overwrite this file with a different
  derivation.

## 10. Common failure modes / troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Stage 1 fails: "Python X.Y detected... 3.13 expected" | Running with the system/global Python, not the project venv | Use `.venv/Scripts/python.exe` (Windows) / `.venv/bin/python` (POSIX) explicitly |
| Stage 2 fails listing missing `results/runs/*` or `results/phaseN/` | Shallow clone, `.gitignore`'d large files not fetched, or genuinely running against a fresh checkout missing committed raw data | These are NOT regenerable — pull the full repository/LFS objects; this is the pipeline's fail-loud-on-missing-input behavior working as intended |
| `crossover_calibration.py` raises `ValueError: no finite observed crossovers to calibrate against` | Ran it directly against a `--smoke`-scale `horizon_sweep.parquet`/`horizon_analysis.py` derivation | Expected — this script needs `crossover_calibration_grid.py`'s dedicated calibration grid (paper mode only); see §5 |
| A `--smoke` run appears to have overwritten a production table | Six of the seven grid/table scripts default `--output` to the same path whether or not `--smoke` is passed | Always pass an explicit non-default `--output` (as `reproduce.py` always does) — never invoke `--smoke` bare against a checkout with real results you care about |
| `prior_grid.py` raises `numpy._core._exceptions._ArrayMemoryError` at large horizons | `bootstrap_ci`'s vectorized resampling under joblib's default full parallelism on a memory-constrained machine — root-caused and documented in that script's own module docstring | Pass a lower `--n-jobs` (2–3) |
| A long background sweep appears to stall for hours with no CPU activity | Windows power-plan sleep timeout triggering mid-run (confirmed via `powercfg /query`, not assumed) | Adjust the sleep timeout, or run foreground/plugged-in for multi-hour jobs |
| `reproduce.py` reports a `mismatch` not covered by the two documented exclusions | A genuine, unexplained discrepancy — the pipeline never auto-promotes in this case | Investigate before trusting the artifact; production was not overwritten by the comparison run itself |
