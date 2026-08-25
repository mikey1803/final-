# Project Status — Everything Known So Far

*A single up-to-date snapshot of what this project is, what has been built, and where things currently stand. For the plain-language walkthrough of how each component works, see [`PROJECT_EXPLAINED.md`](PROJECT_EXPLAINED.md). For the full research dossier (every experimental phase, exact numbers, paper outline), see [`PAPER_PREP.md`](PAPER_PREP.md). For a task-by-task build log explaining what each of the 16 implementation tasks did and how it improved the project, see [`TASK_HISTORY.md`](TASK_HISTORY.md).*

Last updated: 2026-08-04.

---

## 1. What this project is

**Failure-Aware Adaptive Reasoning Framework for Reliable Autonomous AI Agents.**

Autonomous LLM-agent tool calls fail in structured, recurring ways: timeouts, wrong tool chosen, hallucinated output, malformed output, context loss, ambiguous instructions, tool errors. Instead of hardcoding one recovery rule per failure type, this project frames recovery-strategy selection as a **multi-armed bandit problem**: each `(failure_type, recovery_strategy)` pair has an unknown success probability, and an adaptive selector (Thompson Sampling, UCB1, epsilon-greedy, LinUCB, ...) must learn which strategy works best for each failure type purely from trial and error, without ever seeing the ground-truth success matrix.

**The central empirical finding**, arrived at across nine experimental phases: a well-designed *static* rule table beats every adaptive method in a *stationary* environment, by a margin that grows as the failure taxonomy grows — but the moment the environment drifts (a tool degrades, an API changes), the static rule cannot recover while the adaptive methods do, recovering faster the more practice they've had. The paper is about mapping out **where that crossover lives**, not just "adaptive beats non-adaptive."

Author: Pio Micheal (URK23AI1021), Final-year B.Tech AI & Data Science, Karunya Institute of Technology and Sciences. Guided by Ms. Divina.

---

## 2. The two layers of the codebase

### 2.1 Original system (`backend/src/`) — the working prototype

The end-to-end pipeline: **Failure Injection → Failure Classification → Recovery Selection (bandit) → Recovery Outcome → Persistent Memory (SQLite) → Evaluation / Dashboard.**

- `failures/taxonomy.py` — 7 failure types (`tool_timeout`, `tool_error`, `wrong_tool_selected`, `hallucinated_output`, `malformed_output`, `context_loss`, `ambiguous_instruction`).
- `failures/injector.py` — deterministic, seeded synthetic failure generation for repeatable experiments.
- `failures/classifier.py` — TF-IDF + LinearSVC (plus a logistic model for confidence scores) predicting failure type from raw error text. Reaches 95.00% accuracy / 94.68% F1.
- `recovery/strategies.py` — 6 recovery actions (`retry`, `replan`, `switch_tool`, `re_ground`, `verify`, `clarify`) plus the hidden `SUCCESS_MATRIX` ground truth the selectors must learn without seeing it directly.
- `recovery/selector.py` — the adaptive selectors: Thompson Sampling, UCB1, epsilon-greedy (later extended with LinUCB, contextual cost-aware, prior-initialized/discounted variants — see below).
- `memory/store.py` — SQLite-backed persistent outcome memory, replayable back into a selector.
- `evaluation/harness.py` — runs matched, seeded comparisons across methods and trials; reports cumulative + late-stage success rate, cumulative regret, paired t-tests.
- Two dashboards under `frontend/`: the original Streamlit app (fallback) and a React/Vite port with a live FastAPI-backed demo (`api/main.py`) using the real trained classifier and a real Thompson selector against its own isolated SQLite file.

Validated headline result (500 episodes, 30 trials, late-stage = final 20%):

| Method | Late-Stage Success Rate |
| --- | ---: |
| Thompson Sampling | 65.50% |
| UCB1 | 57.87% |
| Fixed Rule (v1, superseded — see §2.2) | 52.60% |
| One-Shot LLM | 47.47% |
| Random | 41.87% |
| Naive Retry | 31.23% |

Note: this specific table predates the Fixed Rule v1→v2 blind-design correction described next — its Fixed Rule row is not the paper's current headline number.

### 2.2 Reproducible simulation kernel (`backend/recovery/`) — additive research architecture

Built *on top of* the original system, without modifying it, to independently validate and extend the stationary-environment results with a numerically cross-checked, deterministic kernel:

- `theory/analytic.py` + `envs/canonical.py` — a numpy-only closed-form reference (oracle/random/policy rates) for the canonical 7×6 environment, used to cross-check simulation output instead of trusting it blindly.
- `core/` — one shared episode loop (`run_episodes`) every regenerated experiment runs through, with an RNG design (`core/rng.py::SeedStreams`) that keeps the classifier's *predicted* type structurally unable to leak into the reward function (documented in `architecture.md`, "Preventing Reward-Path Errors by Design").
- `guards/invariants.py` — runtime assertions that raise immediately if a result is mathematically impossible (e.g. exceeding the oracle ceiling), instead of silently producing a bad number.
- `policies/`, `envs/matrix_env.py`, `classify/` — thin adapters wrapping every existing selector plus new ones (LinUCB, contextual cost-aware, prior-initialized Thompson/PGTS, discounted variants) and the trained classifier behind common interfaces.
- `analysis/tables.py` — shared table-generation logic reused by every experiment script rather than duplicated per script.

This kernel investigated (and, for one table, could not fully reproduce) a numerical anomaly in a published table — see `anomalies.md` for the investigation and `reconciliation_stationary.md` for a cell-by-cell comparison of every regenerated value against the paper.

---

## 3. The experimental program (`backend/experiments/`)

Nine phases plus three cross-cutting audit tasks, all documented in detail in `PAPER_PREP.md` §3:

| Phase | What it establishes |
| --- | --- |
| 1 | Full 11-method baseline comparison (re-run and citable). |
| 2 | Classifier-noise sweep (superseded by later work). |
| 3A | Randomized reward-matrix family sweep, Thompson vs. UCB1. |
| 3B | Non-stationarity mini-experiment (superseded). |
| 4 | **Key methodological pivot**: discovered the original Fixed Rule baseline (v1) was inadvertently reward-structure-contaminated; built a corrected v2 via a blind-design protocol. Established the core dichotomy: static rules win stationary, adaptive wins under drift. |
| 4b | Reconciles stationary vs. non-stationary results — the paper's central tension. |
| 5 | Static-rule-quality crossover: a hand-designed rule needs ≈64% (4.5/7) quality to match the best adaptive baseline in a stationary world. |
| 6 | Shift frequency × magnitude sweep — quantifies how the adaptive advantage grows under drift, plus recovery half-life. |
| 7 | Taxonomy-size stress test — confirms the stationary-world static advantage is not an artifact of the real taxonomy's small size (7); it holds and strengthens at 4x scale. |
| 8 (Task 11) | **Blind-Design Elicitation Study**: a technically-enforced-blind web instrument plus an identical-protocol LLM arm (3 real model families, real API calls). Found a universal per-failure-type weak point (`malformed_output`) — later re-tested more rigorously in Task 13 (see below). Human arm pending institutional approval. |
| 9 (Task 12) | **SUCCESS_MATRIX sensitivity/robustness analysis**: 2,204 evaluated perturbed worlds across 3 perturbation schemes. Real breakdown thresholds found (Gaussian noise sigma=0.15; rank-breaking permutation fraction=0.25; rank-preserving permutation never breaks the ranking). Honest negative result: 2 of 3 structured alternative matrices break the headline ranking outright — the stationary-world claim is conditional on the canonical matrix's specific structure, not universally true. |

### Cross-cutting audits (not new experimental phases)

- **Task 13 — Statistical rigor audit** (`docs/statistics_registry.md`): audited every statistical claim in the project.
  - Settled on exactly one family of tests getting Holm-Bonferroni correction (the real-sample LLM per-type accuracy tests) — everything else is a simulation-seed comparison already reported via effect size + bootstrap CI, not a p-value.
  - Re-tested the `malformed_output` cross-family finding formally: significant in only 1 of 3 LLM families after correction, not all 3 as originally stated — the Phase 8 language was corrected project-wide to this weaker, tested claim.
  - Re-ran every CI resting on fewer than 30 seeds at `n_seeds=30`.
  - **Found and fixed a real non-determinism bug**: `run_matrix_family_sweep`'s seed formula used Python's per-process-randomized `hash()` on algorithm names, so "the same" experiment silently produced different numbers across process invocations. Fixed with a deterministic `zlib.crc32`-based offset, regression-tested.
  - Added a "Reproducibility" subsection to `PAPER_PREP.md` §6 documenting all of this for the paper.

- **Task 14 — Runtime, cost, and provenance instrumentation** (in progress — see §5 below): per-decision wall-clock/memory benchmarking, LLM API token/cost backfill accounting, exact environment/dependency/model-version capture, and a provenance map from every paper figure/table back to its generating script, raw data, and reproduce command.

---

## 4. Current repository layout

```text
failure-aware-agent/
|-- backend/
|   |-- src/            # original prototype (classifier, injector, selector, harness, memory)
|   |-- recovery/        # additive deterministic kernel (theory/, core/, guards/, envs/, classify/, policies/, analysis/)
|   |-- experiments/     # ~25 scripts: stationary_comparison, matrix_family_sweep, classifier_noise_sweep,
|   |                     #   crossover_*, drift_*, horizon_*, prior_*, matrix_robustness, llm_designers,
|   |                     #   hypothesis_registry, study_analysis, runtime_benchmark, llm_cost_accounting,
|   |                     #   capture_environment
|   |-- tests/           # pytest suite (unit + kernel-vs-analytic cross-checks + determinism regression tests)
|   |-- data/            # trained classifier_model.joblib, evaluation_results.csv, confusion_matrix.png
|   |-- results/         # phase1/ .. phase9/, tables/ (regenerated V/VI/VII + every experiment's parquet output)
|   `-- api/              # FastAPI server backing the React dashboard's live demo
|-- frontend/
|   |-- dashboard/        # original Streamlit app (fallback)
|   `-- dashboard-react/  # React + Vite port, live simulation tab
|-- study/                 # the blind-design elicitation web instrument (Task 11)
`-- docs/
    |-- PROJECT_EXPLAINED.md        # plain-language walkthrough of the original system
    |-- PROJECT_STATUS.md           # this file
    |-- PAPER_PREP.md               # full research dossier — single source of truth for the paper
    |-- architecture.md             # the recovery/ kernel's design
    |-- repo_inventory.md           # inventory of the original codebase the kernel was built on
    |-- anomalies.md                # investigation of a published-table numerical anomaly
    |-- reconciliation_stationary.md # cell-by-cell regenerated-vs-published comparison
    |-- statistics_registry.md      # Task 13 statistical audit — every test, correction, and re-run
    |-- statistics.md
    |-- experiments/                 # per-experiment design notes (drift, horizon, matrix_robustness, human_study, ...)
    |-- theory/
    `-- datasets/
```

**Note on git state**: only 2 commits exist in this repository so far (initial implementation, and a README update). Everything described in §2.2, §3, and §4 above — the entire `recovery/` kernel, the `experiments/` suite, `study/`, and most of `docs/` — exists on disk but is **not yet committed**. `git status` currently shows the old flat `src/`, `tests/`, `dashboard/`, `data/`, and `requirements.txt` at repo root as deleted (they moved under `backend/`), plus a long list of untracked new files/directories. Nothing has been force-cleaned or discarded; this is simply pending a commit that captures the restructure.

---

## 5. Task 14 (in progress) — Runtime, cost, and provenance instrumentation

This is the currently active task, not yet finished. Status as of this writing:

**Done:**
- **Part 1 (runtime/memory)**: `experiments/runtime_benchmark.py` — measures per-decision wall-clock time (mean/median/p95/p99, microseconds) and peak memory (`tracemalloc`) for every policy including fixed-rule baselines, cross-validated against the real unmodified kernel. Real hardware captured: Intel Core i5-9300H @ 2.40GHz, 4C/8T, Windows 11 Home Single Language, Python 3.13.11. Fastest policy (Oracle, ~2us/decision) to slowest (LinUCB, ~660us/decision) characterized; one unexplained memory outlier (PGTS, ~100KB peak) reported honestly without a fabricated explanation.
- **Part 2 (API cost accounting)**: `experiments/llm_cost_accounting.py` — backfills prompt/completion tokens from already-logged raw API responses (zero new paid calls, per the task's rule). Real per-vendor pricing cited from primary sources (groq.com, ai.google.dev) with fetch dates. NVIDIA NIM pricing left explicitly unconfirmed (`NaN`, not zero) since no primary source could be reached. ~38,000 tokens backfilled across 63 logged calls; ~$0.0024 hypothetical pay-as-you-go cost (all calls were actually made on free/dev-tier keys, so real spend was ~$0).
- **Part 3 (environment capture)**: `experiments/capture_environment.py` — real `pip freeze` (85 packages), hardware/OS capture (reused from Part 1), and per-family LLM model-pin table. Found that Google's Gemini API echoes back the literal alias `"gemini-flash-lite-latest"` rather than a resolved dated model string — meaning exact model pinning is **not achievable from the data alone** for that family; documented as an explicit gap, not glossed over. Groq and NVIDIA both echo back their exact configured model ID.

**In progress:**
- **Recovery from an operational incident**: while testing output paths, several `--smoke` (fast validation) runs were executed without first confirming each script's smoke and full modes share the same *default* output file path (they do, inconsistently with `matrix_robustness.py`, which uses separate `_smoke`-suffixed filenames). This overwrote six production result files with smoke-scale data: `table_v.parquet`, `table_vi.parquet`, `table_vii.parquet`, `drift_taxonomy.parquet`, `horizon_sweep.parquet`, `prior_grid.parquet`. No source code was changed — only output data files.
  - **2026-08-20 correction: the root cause described above was never actually closed.** The data was restored (see below), but `stationary_comparison.py`, `classifier_noise_sweep.py`, `matrix_family_sweep.py`, `horizon_sweep.py`, `prior_grid.py`, and `drift_taxonomy.py` all still defaulted `--output` to the same production path regardless of `--smoke`, confirmed by direct code inspection during a labelling-integrity audit that found the identical bug shape in `tests/test_memory_and_harness.py` (see `docs/PAPER_PREP.md`'s provenance-audit note and `docs/reconciliation_stationary.md`'s correction). None of these six files had actually been re-damaged since the restore (all six carried their original 2026-08-04/05/06 restore timestamps, checked directly) — this was a live, unfired landmine, not a second incident. Fixed in all six scripts: a bare `--output`-less `--smoke` invocation now writes to a `*_smoke.parquet` sibling, matching `matrix_robustness.py`'s already-correct pattern. Verified by actually running every script's `--smoke` mode and confirming the real file's mtime didn't change and the `_smoke` sibling did appear.
  - **Restored and verified** (full-scale re-run, values cross-checked against previously-documented numbers, exact matches): `table_v.parquet` (550 rows), `table_vi.parquet` (16,000 rows), `table_vii.parquet` (300 rows), `drift_taxonomy.parquet` (12,800 rows — its underlying per-episode cache was untouched by the incident, so this restore was near-instant).
  - **~~Still restoring~~ — CORRECTED (2026-08-18), see below.** This entry originally said `horizon_sweep.parquet` and `prior_grid.parquet` were still at smoke scale with a *true full-scale* (`n_seeds=30`, `n_gap_matrices=25`) regeneration "actively running in the background," implying it would complete and land at full scale. **That job was never completed — it was abandoned.** The true full grid was separately measured via each script's own `--estimate` at ~73h (`horizon_sweep.py`) / ~66h (`prior_grid.py`) on this hardware and judged infeasible to run to completion at the time; both files were instead regenerated at a smaller, deliberate, documented scope (`horizon_sweep.py --n-seeds 3 --n-gap-matrices 2`; `prior_grid.py --n-seeds 3 --qualities 0 4 7 --prior-means 0.7 --horizons 500 50000 --n-jobs 2`) and marked **REDUCED (documented deviation)** — see `docs/experiments/instrumentation.md` Part 4 for the authoritative, verified final state of both files. This stale note was left uncorrected for over a week: a later task (Gate 1 of Task 19A) read this paragraph, took "actively running... have not yet completed" as evidence a 26-environment `horizon_sweep.parquet` existed or was imminent, and that belief propagated two gates further into a validation-grid design before being caught and corrected. Same failure class as the Task 15 incident this section documents above (documentation drifting out of sync with artifact reality) — non-destructive here (no data was lost or overwritten a second time), but a real instance of the same root cause: a status note describing in-flight work was never updated once that work's actual outcome (abandonment + documented reduced scope) was known. See `docs/experiments/instrumentation.md`'s "Documentation drift incident" note for the corresponding entry on that side.
- **Part 4 (provenance map)**: not yet built. Plan is a table mapping every paper figure/table to its generating script, raw data file(s), exact reproduce command, and seed(s) — verified by actually re-running each command, not by reading the code. Partial verification already done this session (Phases 1, 2, 4, 4b reproduced exactly or with fully-understood/documented deviations); the rest remains.
- Full backend test suite has not yet been re-run since Task 14's script additions; needs a final pass before Task 14 is reported done.
- A consolidated `docs/experiments/instrumentation.md` write-up for all four parts has not yet been created.

---

## 6. How to run things

```bash
# Backend tests
cd backend && python -m pytest

# Original evaluation harness / dashboard
streamlit run frontend/dashboard/app.py          # from repo root
cd frontend/dashboard-react && npm install && npm run dev   # React port

# Live interactive demo (real classifier + real Thompson selector)
cd backend && uvicorn api.main:app --reload --port 8000     # terminal 1
cd frontend/dashboard-react && npm run dev                  # terminal 2

# Regenerated kernel experiments (from backend/)
python experiments/stationary_comparison.py [--smoke]
python experiments/matrix_family_sweep.py [--smoke]
python experiments/classifier_noise_sweep.py [--smoke]
# ... and ~20 more under experiments/, one per phase/task above
```

`--smoke` gives a <60s validation run; omitting it runs the full paper-scale experiment. **Caveat learned the hard way this session**: not every script isolates smoke output to a separate filename — check each script's `DEFAULT_OUTPUT`/`--output` behavior before running `--smoke` against a directory holding real results.

---

## 7. Where to look for what

| Question | File |
| --- | --- |
| "How does the system work, end to end, in plain language?" | `PROJECT_EXPLAINED.md` |
| "What's the exact experimental methodology and numbers for phase X?" | `PAPER_PREP.md` §3 |
| "What are the project's core contributions?" | `PAPER_PREP.md` §4 |
| "What are the known limitations?" | `PAPER_PREP.md` §6 |
| "Is this statistical claim actually valid?" | `statistics_registry.md` |
| "Does the kernel reproduce the published numbers?" | `reconciliation_stationary.md`, `anomalies.md` |
| "How is the `recovery/` kernel designed?" | `architecture.md` |
| "What was the original codebase before the kernel was added?" | `repo_inventory.md` |
| "What's the current overall project status?" | this file |
