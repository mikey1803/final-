# Failure-Aware Adaptive Reasoning Framework for Reliable Autonomous AI Agents

Autonomous LLM-agent tool calls fail in structured, recurring ways: timeouts, wrong tool chosen,
hallucinated output, malformed output, context loss, ambiguous instructions, tool errors. Instead of
hardcoding one recovery rule per failure type, this project frames recovery-strategy selection as a
**multi-armed bandit problem**: each `(failure_type, recovery_strategy)` pair has an unknown success
probability, and an adaptive selector (Thompson Sampling, UCB1, epsilon-greedy, LinUCB, ...) has to learn
which strategy works best for each failure type purely from trial and error, without ever seeing the
ground-truth success matrix.

**Central finding**, established across nine experimental phases plus six follow-up audit/robustness
tasks: a well-designed *static* rule table beats every adaptive method in a **stationary** environment,
by a margin that grows as the failure taxonomy grows — but the moment the environment **drifts** (a tool
degrades, an API changes), the static rule cannot recover while the adaptive methods do, recovering
faster the more practice they've had. The project is about mapping out **where that crossover lives**,
quantifying it, stress-testing it, and making every number behind it independently reproducible — not
just showing "adaptive beats non-adaptive."

Author: **Pio Micheal**, URK23AI1021 — Final-year B.Tech Artificial Intelligence and Data Science,
Karunya Institute of Technology and Sciences. Guided by **Ms. Divina**.

---

## What's in this repository

The project has two layers, built in two stages, plus a large experimental program on top of both.

### 1. The original prototype (`backend/src/`)

The working end-to-end pipeline the whole idea is demonstrated on:

```text
Failure Injection -> Failure Classification -> Recovery Selection (bandit) -> Recovery Outcome
     -> Persistent Memory (SQLite) -> Evaluation / Dashboard
```

- **`failures/taxonomy.py`** — 7 failure types (`tool_timeout`, `tool_error`, `wrong_tool_selected`,
  `hallucinated_output`, `malformed_output`, `context_loss`, `ambiguous_instruction`).
- **`failures/injector.py`** — deterministic, seeded synthetic failure generation for repeatable
  experiments.
- **`failures/classifier.py`** — TF-IDF + LinearSVC (plus a logistic model for confidence scores)
  predicting failure type from raw error text. **95.00% accuracy / 94.68% F1.**
- **`recovery/strategies.py`** — 6 recovery actions (`retry`, `replan`, `switch_tool`, `re_ground`,
  `verify`, `clarify`) plus the hidden `SUCCESS_MATRIX` ground truth the selectors must learn without
  seeing it directly.
- **`recovery/selector.py`** — Thompson Sampling, UCB1, epsilon-greedy (later extended with LinUCB,
  contextual cost-aware, and prior-initialized/discounted variants — see below).
- **`memory/store.py`** — SQLite-backed persistent outcome memory, replayable back into a selector.
- **`evaluation/harness.py`** — matched, seeded comparisons across methods and trials; cumulative +
  late-stage success rate, cumulative regret, paired t-tests.
- **Two dashboards** under `frontend/` (Streamlit fallback + React/Vite port), and a FastAPI server
  (`backend/api/main.py`) backing the React dashboard's *live* demo — real trained classifier, real
  Thompson selector, real SQLite state, not a replay.

### 2. The reproducible simulation kernel (`backend/recovery/`)

Built **on top of** the original system, without modifying it, to independently validate and extend the
results with a numerically cross-checked, deterministic architecture:

- **`theory/analytic.py` + `envs/canonical.py`** — a numpy-only closed-form reference (oracle/random/policy
  rates) for the canonical 7×6 environment, used to cross-check simulation output instead of trusting it
  blindly.
- **`core/`** — one shared episode loop (`run_episodes`) that every regenerated experiment runs through,
  with an RNG design (`core/rng.py::SeedStreams`) that keeps the classifier's *predicted* type
  structurally unable to leak into the reward function (see `docs/architecture.md`, "Preventing
  Reward-Path Errors by Design").
- **`guards/invariants.py`** — runtime assertions that raise immediately if a result is mathematically
  impossible (e.g. exceeding the oracle ceiling), instead of silently producing a bad number.
- **`policies/`, `envs/matrix_env.py`, `classify/`** — thin adapters wrapping every existing selector plus
  new ones (LinUCB, contextual cost-aware, prior-initialized Thompson/PGTS, discounted variants) and the
  trained classifier behind common interfaces.
- **`core/parallel.py` (`RunStore`)** — content-addressed parquet cache (hash of method/env/seed/episode
  count/classifier/prior -> one file per cell) so every expensive grid sweep only ever computes a given
  cell once, ever, across all future re-runs.

This kernel investigated (and, for one published table, could not fully reproduce) a numerical anomaly —
see `docs/anomalies.md` for the investigation and `docs/reconciliation_stationary.md` for a cell-by-cell
comparison of every regenerated value against the original paper draft.

---

## The experimental program

Nine research phases, four follow-up rigor/robustness tasks, and two infrastructure tasks — all under
`backend/experiments/` (24 scripts), documented in full in `docs/PAPER_PREP.md` §3.

| Phase / Task | What it establishes |
| --- | --- |
| 1 | Full 11-method baseline comparison (re-run and citable). |
| 2 | Classifier-noise sweep. |
| 3A | Randomized reward-matrix family sweep, Thompson vs. UCB1. |
| 3B | Non-stationarity mini-experiment. |
| 4 | **Key methodological pivot** — discovered the original Fixed Rule baseline (v1) was inadvertently reward-structure-contaminated; built a corrected v2 via a blind-design protocol. Established the core dichotomy: static rules win stationary, adaptive wins under drift. |
| 4B | Reconciles stationary vs. non-stationary results — the paper's central tension. |
| 5 | Static-rule-quality crossover: a hand-designed rule needs ≈64% (4.5/7) quality to match the best adaptive baseline in a stationary world. |
| 6 | Shift-frequency × magnitude sweep — quantifies how the adaptive advantage grows under drift, plus recovery half-life. |
| 7 | Taxonomy-size stress test — confirms the stationary-world static advantage isn't an artifact of the real taxonomy's small size (7); it holds and strengthens at 4x scale. |
| 8 / Task 11 | **Blind-Design Elicitation Study** — a technically-enforced-blind web instrument (`study/`) plus an identical-protocol LLM arm (3 real model families, real API calls). Found a universal per-failure-type weak point (`malformed_output`). Human arm pending institutional approval. |
| 9 / Task 12 | **SUCCESS_MATRIX sensitivity/robustness analysis** — 2,204 evaluated perturbed worlds across 3 perturbation schemes. Real breakdown thresholds found (Gaussian noise σ=0.15; rank-breaking permutation fraction=0.25). Honest negative result: 2 of 3 structured alternative matrices break the headline ranking outright. |
| Task 13 | **Statistical rigor audit** — every statistical claim in the project re-checked; settled on exactly one test family needing Holm-Bonferroni correction; re-tested the `malformed_output` finding formally (significant in 1 of 3 LLM families after correction, not all 3); found and fixed a real non-determinism bug (Python's randomized `hash()` was silently making one experiment's "seed" not actually deterministic across process runs). |
| Task 14 | **Runtime, cost, and provenance instrumentation** — per-decision wall-clock/memory benchmarking (`runtime_benchmark.py`), LLM API token/cost backfill accounting from already-logged responses (`llm_cost_accounting.py`, zero new paid calls), and exact environment/dependency/model-version capture (`capture_environment.py`). |
| Task 15 | **Repository integrity recovery** — root-caused and fixed an operational incident where `--smoke` (fast validation) runs had silently overwritten several full-scale production result files, because most experiment scripts don't separate smoke and full output paths by default; restored every affected artifact from a clean full-scale re-run and verified against previously-documented numbers. |
| Task 16 | **One-command reproducibility pipeline** — see below. |

### Cross-cutting drift, horizon & prior-strength studies

Beyond the numbered phases, `backend/experiments/` also includes a full sub-program characterizing *when*
and *how much* a prior-informed policy (PGTS) helps or hurts:

- **`drift_taxonomy.py` / `drift_analysis.py`** — classifies drift severity across many environment
  configurations and quantifies adaptive recovery speed.
- **`horizon_sweep.py` / `horizon_analysis.py` / `crossover_calibration_grid.py` /
  `crossover_calibration.py` / `crossover_extension.py`** — finds and calibrates the exact episode-count
  crossover point where adaptive methods overtake the static rule, across many environment/classifier
  combinations, with a fitted, extrapolable model.
- **`prior_grid.py` / `prior_analysis.py`** — sweeps prior "quality" (how correct the hand-designed table
  is) against prior "strength" (how confidently the policy trusts it), mapping out exactly when a prior
  helps versus causes *negative transfer*.
- **`matrix_robustness.py`, `llm_designers.py`, `prompt_robustness.py`** — matrix-family robustness and a
  3-model-family LLM study of how consistently different systems (human and LLM) can hand-design a good
  recovery table.

---

## Key Results

Evaluation was run over 500 episodes and 30 independent trials. The late-stage success rate is measured
over the final 20% of episodes, after learning/convergence effects have had time to appear.

| Method | Late-Stage Success Rate |
| --- | ---: |
| Thompson Sampling | 65.50% |
| UCB1 | 57.87% |
| Fixed Rule | 52.60% |
| One-Shot LLM | 47.47% |
| Random | 41.87% |
| Naive Retry | 31.23% |

> This table is the original headline run. It predates the Fixed Rule v1 -> v2 blind-design correction
> (Phase 4) described above — see `docs/PAPER_PREP.md` §3–4 for the corrected, current numbers and the
> full crossover/robustness results built on top of it.

The failure classifier achieved:

| Metric | Score |
| --- | ---: |
| Accuracy | 95.00% |
| F1 Score | 94.68% |

---

## Task 16 — One-command reproducibility pipeline

Every offline analysis, publication figure, and publication table in the paper can be regenerated with a
single command, from committed raw data, **without any API keys**:

```bash
cd backend
python reproduce.py --mode validation   # ~5 min, --smoke scale, CI-suitable
python reproduce.py --mode paper        # ~65 min, full paper scale (two multi-day sweeps skipped by default)
```

- **7 stages**, each stopping immediately on failure: verify environment -> verify raw artifacts exist ->
  regenerate analyses -> regenerate figures -> regenerate tables -> regenerate doc artifacts -> validate
  every output against the committed production version (numeric-tolerant, not just byte-equality).
- **Safety-first**: regenerated outputs are always written to a timestamped validation directory (or run
  through a snapshot/rollback wrapper) — production artifacts are *never* overwritten unless equivalence
  is explicitly confirmed first.
- **Cost-gated**: the two truly expensive full grids (`horizon_sweep.py` ≈73h, `prior_grid.py` ≈66h) are
  skipped by default with a loud, explicit banner naming the exact command to force them
  (`--allow-expensive`); everything else in paper mode runs for real.
- Full setup, environment pinning (`backend/requirements.lock.txt`), online-vs-offline workflow, expected
  runtimes, and troubleshooting are in **`docs/reproducibility.md`**.
- The complete figure/table provenance map (generating script, required inputs, pipeline stage, exact
  reproduce command, RNG seeds — verified by actually executing each command) is in
  **`docs/experiments/instrumentation.md`**.

---

## Tech Stack

- Groq, Google Gemini, NVIDIA NIM APIs (3 independent LLM model families used across the LLM-arm studies)
- LangChain, LangGraph
- Thompson Sampling, UCB1, Epsilon-Greedy, LinUCB (multi-armed bandit algorithms)
- scikit-learn
- SQLite
- FastAPI, Streamlit, React + Vite, Plotly
- pandas, NumPy, Matplotlib, SciPy, joblib

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd failure-aware-agent
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The Python backend lives under `backend/`:

```bash
pip install -r backend/requirements.txt
```

For an exact, pinned reproduction of the environment this project was developed and verified against, use
the lockfile instead:

```bash
pip install -r backend/requirements.lock.txt
```

### 4. Configure environment variables (optional — only needed for the LLM-arm experiments)

Offline reproduction (all analyses, figures, and tables) needs **no API keys at all** — see
`docs/reproducibility.md`. If you want to re-run the live LLM experiments (`llm_designers.py`,
`prompt_robustness.py`, the "One-Shot LLM" row of Table V), copy the example environment file inside
`backend/` and fill in whichever keys you have:

```bash
cp backend/.env.example backend/.env
```

```text
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here      # optional, Gemini family
NVIDIA_API_KEY=your_nvidia_api_key_here      # optional, NVIDIA NIM family
```

On Windows PowerShell:

```powershell
Copy-Item backend\.env.example backend\.env
```

---

## Running Tests

Run the full test suite from inside `backend/`:

```bash
cd backend
python -m pytest
```

Selected test files (there are 30+):

- `tests/test_injector.py`, `tests/test_classifier.py`, `tests/test_selector.py`,
  `tests/test_memory_and_harness.py` — the original prototype's core components.
- `tests/test_analytic.py`, `tests/test_kernel_vs_analytic.py`, `tests/test_invariants.py`,
  `tests/test_episode.py` — the reproducible kernel cross-checked against the closed-form reference.
- `tests/test_drift.py`, `tests/test_horizon.py`, `tests/test_crossover.py`,
  `tests/test_crossover_reliability.py`, `tests/test_hybrid.py` — the drift/horizon/prior-strength
  sub-program.
- `tests/test_matrix_robustness.py`, `tests/test_matrix_perturbation.py`,
  `tests/test_matrix_family_determinism.py`, `tests/test_significance.py`,
  `tests/test_hypothesis_registry.py` — the robustness and statistical-audit tasks.
- `tests/test_study.py` — the blind-design elicitation instrument's blindness guarantees.

---

## Running the Dashboard

Two frontends are available under `frontend/`: the original Streamlit dashboard (`frontend/dashboard/`)
and a newer React port (`frontend/dashboard-react/`), kept as a fallback of each other.

### Streamlit (run from the repo root)

```bash
streamlit run frontend/dashboard/app.py
```

If `streamlit` is not available as a shell command, run:

```bash
python -m streamlit run frontend/dashboard/app.py
```

The dashboard includes:

- Live simulation across Thompson Sampling, UCB1, Epsilon-Greedy, and baseline methods with cumulative
  success rate and regret curve visualizations.
- Algorithm Race tab with live animated comparison of all 3 bandit algorithms competing on the same
  failure sequence.
- Strategy learning heatmap for `(failure_type, recovery_strategy)` success rates.
- Failure classifier metrics and interactive prediction demo.
- Live agent trace showing failure detection, classification, recovery selection, and outcome.

### React (run from `frontend/dashboard-react/`)

```bash
cd frontend/dashboard-react
npm install
npm run dev
```

### Live Simulation demo (real backend, not a replay)

The React dashboard's **Live Simulation** tab calls a FastAPI server that wraps the actual trained failure
classifier and the actual Thompson Sampling selector (with persistent SQLite state), so every prediction,
strategy pick, and outcome you see is computed live. It needs its own backend process running alongside
the frontend:

```bash
# Terminal 1 — from backend/
cd backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — from frontend/dashboard-react/
cd frontend/dashboard-react
npm run dev
```

Then open `http://localhost:5173`, go to Data Dashboard -> Live Simulation. This uses its own SQLite file
(`backend/data/demo_outcomes.db`), kept separate from `backend/data/outcomes.db` so clicking around the
demo never touches the real evaluation results.

---

## Project Structure

```text
failure-aware-agent/
|-- backend/
|   |-- src/                  # original prototype: agent/, evaluation/, failures/, memory/, recovery/
|   |-- recovery/             # additive deterministic kernel: theory/, core/, guards/, envs/, classify/, policies/, analysis/
|   |-- experiments/          # 24 scripts: stationary_comparison, matrix_family_sweep, classifier_noise_sweep,
|   |                         #   drift_*, horizon_*, crossover_*, prior_*, matrix_robustness, llm_designers,
|   |                         #   prompt_robustness, hypothesis_registry, study_analysis, runtime_benchmark,
|   |                         #   llm_cost_accounting, capture_environment, diagnose_table_vii
|   |-- tests/                # 30+ pytest files: unit tests, kernel-vs-analytic cross-checks, determinism
|   |                         #   regression tests, blindness audits
|   |-- api/                  # FastAPI server backing the React dashboard's live demo
|   |-- data/                 # trained classifier_model.joblib, evaluation_results.csv, confusion_matrix.png
|   |-- results/              # phase1/ .. phase7/, tables/, figures/, runs/ (RunStore cache), diagnostics/
|   |-- reproduce.py          # Task 16: one-command reproducibility pipeline
|   |-- requirements.txt
|   |-- requirements.lock.txt # exact pinned environment (Task 16)
|   |-- .env / .env.example
|   `-- conf/
|-- frontend/
|   |-- dashboard/            # original Streamlit dashboard (fallback)
|   `-- dashboard-react/      # React + Vite dashboard, live simulation tab
|-- study/                    # blind-design elicitation web instrument (Task 11)
|-- docs/
|   |-- PROJECT_EXPLAINED.md      # plain-language walkthrough of the original system
|   |-- PROJECT_STATUS.md         # up-to-date project snapshot
|   |-- PAPER_PREP.md             # full research dossier — exact numbers, methodology, paper outline
|   |-- reproducibility.md        # Task 16: setup, online/offline workflow, runtimes, troubleshooting
|   |-- architecture.md           # the recovery/ kernel's design
|   |-- repo_inventory.md         # inventory of the original codebase the kernel was built on
|   |-- anomalies.md              # investigation of a published-table numerical anomaly
|   |-- reconciliation_stationary.md  # cell-by-cell regenerated-vs-published comparison
|   |-- statistics_registry.md    # Task 13 statistical audit — every test, correction, and re-run
|   |-- statistics.md
|   |-- experiments/              # per-experiment design notes + full provenance map (instrumentation.md)
|   |-- theory/
|   `-- datasets/
|-- .gitignore
`-- README.md
```

---

## Where to look for what

| Question | File |
| --- | --- |
| "How does the system work, end to end, in plain language?" | `docs/PROJECT_EXPLAINED.md` |
| "What did each of the 16 tasks build, and how did it improve the project?" | `docs/TASK_HISTORY.md` |
| "What's the exact experimental methodology and numbers for phase X?" | `docs/PAPER_PREP.md` §3 |
| "What are the project's core contributions?" | `docs/PAPER_PREP.md` §4 |
| "What are the known limitations?" | `docs/PAPER_PREP.md` §6 |
| "Is this statistical claim actually valid?" | `docs/statistics_registry.md` |
| "Does the kernel reproduce the published numbers?" | `docs/reconciliation_stationary.md`, `docs/anomalies.md` |
| "How is the `recovery/` kernel designed?" | `docs/architecture.md` |
| "What was the original codebase before the kernel was added?" | `docs/repo_inventory.md` |
| "How do I regenerate every figure/table from scratch?" | `docs/reproducibility.md` |
| "Where did this specific figure/table come from?" | `docs/experiments/instrumentation.md` |
| "What's the current overall project status?" | `docs/PROJECT_STATUS.md` |

---

## Author

**Mikey**
Pio Micheal, URK23AI1021
Final-year B.Tech Artificial Intelligence and Data Science
Karunya Institute of Technology and Sciences
Guided by **Ms. Divina**
