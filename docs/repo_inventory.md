# Repository Inventory (Part A Reconnaissance)

Read-only survey. Nothing in this document reflects a modification — it is
a report on the repository as found. Repo root (git top level) is
`failure-aware-agent/`; the actual Python package lives under
`failure-aware-agent/backend/`.

## 1. Directory tree (depth 3, excluding node_modules/__pycache__/.git/venv)

```
.
├── .env.example
├── .gitignore
├── README.md
├── backend/
│   ├── .env
│   ├── .env.example
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── tools.py
│   ├── data/
│   │   ├── classifier_model.joblib
│   │   ├── confusion_matrix.png
│   │   ├── convergence_curve.png
│   │   ├── cost_curve.png / cost_efficiency_*.{csv,png}
│   │   ├── demo_outcomes.db(-shm/-wal)
│   │   ├── evaluation_results.csv
│   │   ├── lambda_sweep_results.csv
│   │   ├── method_comparison.png / regret_curve.png
│   │   └── phase4/
│   ├── requirements.txt
│   ├── results/
│   │   ├── phase1/  phase2/  phase3a/  phase3b/  phase3c/
│   │   ├── phase4/  phase4b/  phase5/  phase6/  phase7/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── agent/       (graph.py, llm_client.py)
│   │   ├── evaluation/  (baselines.py, classifier_context.py, harness.py,
│   │   │                 matrix_family.py, nonstationarity*.py,
│   │   │                 static_rule_sweep.py, stats_utils.py,
│   │   │                 taxonomy_size_sweep.py)
│   │   ├── failures/    (classifier.py, injector.py, taxonomy.py)
│   │   ├── memory/      (store.py)
│   │   └── recovery/    (contextual_selector.py, selector.py, strategies.py)
│   └── tests/
│       ├── __init__.py
│       ├── test_classifier.py
│       ├── test_contextual_selector.py
│       ├── test_injector.py
│       ├── test_memory_and_harness.py
│       └── test_selector.py
├── docs/
│   ├── PAPER_PREP.md
│   └── PROJECT_EXPLAINED.md
├── frontend/
│   ├── dashboard/          (app.py — Streamlit)
│   └── dashboard-react/    (Vite/React app)
├── streamlit_stderr.log
└── streamlit_stdout.log
```

Note: `.venv/` (repo-root level) and `.pytest_cache/` directories exist at
both repo root and `backend/` but were excluded per instructions.

## 2. Ground-truth success matrix — every definition site

| File | Line | Matches canonical? | Notes |
|---|---|---|---|
| `backend/src/recovery/strategies.py:49` | `SUCCESS_MATRIX: dict[FailureType, dict[RecoveryStrategy, float]] = {` (values run lines 50–105) | **Yes** — every row/value verified by hand against Part B's canonical MATRIX, in the same type order (tool_timeout, tool_error, wrong_tool_selected, hallucinated_output, malformed_output, context_loss, ambiguous_instruction) and strategy order (retry, replan, switch_tool, re_ground, verify, clarify). | **Single, sole ground-truth definition.** |

**No duplicates found.** Every other file that touches `SUCCESS_MATRIX`
(`baselines.py`, `harness.py`, `matrix_family.py`, `nonstationarity.py`,
`nonstationarity_sweep.py`, `static_rule_sweep.py`, `classifier_context.py`)
*imports* it from `src.recovery.strategies` — none re-declares the values.
`src/evaluation/matrix_family.py:66` renders it into a numpy array
(`np.array([[SUCCESS_MATRIX[ft][rs] for rs in _STRATS] for ft in _TYPES])`)
but reads from the same dict at call time, so it is a projection, not a
second source of truth.

The **blind fixed rule** (task->strategy mapping) also has a single
authoritative definition, at `backend/src/evaluation/baselines.py:82`
(`_FIXED_RULES_V2_BLIND`), matching Part B's `BLIND_RULE` exactly:
retry, switch_tool, replan, verify, verify, re_ground, clarify. An earlier,
answer-key-informed version (`_FIXED_RULES_V1_ANSWER_KEY_INFORMED`,
line 49) is kept only as a documented historical record — `FixedRuleBaseline`
(line 98) uses `_FIXED_RULES = _FIXED_RULES_V2_BLIND` (line 85), not v1. See
`backend/results/phase4/FIXEDRULE_CORRECTION_NOTES.md` for the v1→v2
correction history.

## 3. Episode / simulation loops

| File | Line | Description |
|---|---|---|
| `backend/src/evaluation/harness.py:119` | `def _simulate_episode(...)` | The **single shared** per-episode primitive: selects a strategy conditioned on an (optionally noisy) observed type, samples the outcome from `SUCCESS_MATRIX[true_type]`, updates bandit state. Every sweep below calls this. |
| `backend/src/evaluation/harness.py:326` | `def run_comparison(...)` → loop at line 366 (`for ep in range(num_episodes):`) | Phase 1 head-to-head comparison of all selectors. |
| `backend/src/evaluation/harness.py:729` | `def run_lambda_sweep(...)` → loop at line 786 | Cost-weight (lambda) sweep. |
| `backend/src/evaluation/harness.py:917` | `def run_context_mode_sweep(...)` → loop at line 976 | **This is the Table VII classifier-noise sweep** (see item 5). |
| `backend/src/evaluation/matrix_family.py:81` | `def simulate_selector_on_matrix(...)` → loop at line 110 | Phase 3A/5 randomized-matrix-family sweep; takes an arbitrary `P` matrix rather than the hardwired `SUCCESS_MATRIX`. |
| `backend/src/evaluation/nonstationarity.py:98` | `def run_nonstationarity_experiment(...)` → loop at line 133 | Phase 3B/4b non-stationarity (shifted matrix) experiment. |
| `backend/src/evaluation/nonstationarity_sweep.py:108` | `def simulate_selector_with_shifts(...)` → loop at line 128 | Phase 6 shift-frequency/magnitude sweep. |

**Verdict: the core per-episode loop is defined once and shared.** All
higher-level sweeps (`run_comparison`, `run_lambda_sweep`,
`run_context_mode_sweep`) are thin orchestration around the single
`_simulate_episode()` primitive in `harness.py`. `matrix_family.py` and
`nonstationarity_sweep.py` implement their own `for ep in range(...)` loops
because they operate on a parameterized `P` (randomized matrix family /
shifted matrix) rather than the fixed module-level `SUCCESS_MATRIX`, so they
call the lower-level `execute_strategy`/selector API directly instead of
reusing `_simulate_episode`. No file duplicates `_simulate_episode` itself.

## 4. Classifier: location, training, persisted artifacts

- **Definition**: `backend/src/failures/classifier.py:25`, class
  `FailureClassifier` — TF-IDF text features + `LinearSVC` for hard
  predictions (`predict`, line 144) and a separate `LogisticRegression` for
  calibrated probabilities (`predict_proba`, line 159).
- **Training**: `FailureClassifier.train()` (line 118) fits on a list of
  `FailureEvent`s. `backend/src/evaluation/classifier_context.py:45`
  (`load_or_train_classifier`) is the orchestration entry point used by the
  sweeps — it loads a persisted model if present, else trains fresh on 300
  synthetic injected events (seed 99).
- **Fitted artifact on disk**: **Yes** —
  `backend/data/classifier_model.joblib` exists (loaded via
  `FailureClassifier.load`, `classifier.py:221`, `joblib.load`). Loading it
  currently emits `sklearn.exceptions.InconsistentVersionWarning` (saved
  under scikit-learn 1.7.2, environment has 1.9.0) but loads successfully.
- **Held-out predictions on disk**: **None found.** No CSV/JSON of
  classifier predictions vs. ground truth exists under `backend/data/` or
  `backend/results/**`. `compute_empirical_confusion_matrix()`
  (`classifier_context.py:62`) recomputes the confusion matrix **on the fly**
  each time it's called, from 150 fresh synthetic events per type
  (seed 4242) run through the loaded classifier — it is not cached to disk
  anywhere in the repo.

## 5. Table VII (classifier-noise sweep, β ∈ {1.0,…,0.0}) — script and code path

**Producing script**: `backend/src/evaluation/harness.py:917`,
`run_context_mode_sweep()`. It is exercised for the paper's numbers in
`backend/results/phase4b/` (see `PHASE4B_SUMMARY.md`, "Task 1 — Classifier-
noise sweep") and mirrored in `docs/PAPER_PREP.md` lines 219–232, which is
the literal source of the "0.741 at 93.9% accuracy" figure the objective
refers to (row `β=1.0 | eff. acc.=0.939 | Fixed Rule=0.741`). An older,
apparently-stale run of the same script lives in
`backend/results/phase2/context_sweep_results.csv`, whose `β=1.0` Fixed Rule
value (0.5346) disagrees sharply with phase4b's 0.7408 at the *same* nominal
β/accuracy — consistent with the v1→v2 FixedRule correction (item 2)
happening between the two runs. This is reported for completeness; Part D
of the diagnostic script only reproduces against the current codebase, not
against either stale CSV.

**Reward-sampling code path** — `harness.py:976`–`988`, inside
`run_context_mode_sweep`'s trial loop:

```python
for ep in range(num_episodes):
    true_ft = fts[ep]
    observed_ft = sample_synthetic_context(true_ft, c_beta, context_rng)

    for mname, sel in selectors.items():
        retry_count = outcome_rng.randint(0, 3)
        success, _cost = _simulate_episode(
            sel, true_ft, outcome_rng, ep + 1,
            method_name=mname, retry_count=retry_count,
            observed_failure_type=observed_ft,
        )
```

which calls `_simulate_episode` (`harness.py:119`):

```python
obs_type = failure_type if observed_failure_type is None else observed_failure_type
...
strategy = selector.select_strategy(obs_type, context=context)   # line 142 — ACTION from PREDICTED type
event = FailureEvent(failure_type=failure_type, raw_error="simulated")
outcome = execute_strategy(strategy, event, ctx, rng=rng)         # line 144 — REWARD from TRUE type (event.failure_type)
```

**Explicit answer**: the reward is drawn using the **TRUE** failure type
(`failure_type` / `event.failure_type`, sampled from
`SUCCESS_MATRIX[true_type][strategy]` inside `execute_strategy` →
`_simulate_outcome`, `strategies.py:147`); the **action** (which strategy is
played) is chosen from the **PREDICTED/observed** type
(`obs_type` = `observed_failure_type` when supplied). This is the correct
"action-from-predicted, reward-from-true" wiring — i.e. as read, this
specific code path does not itself contain an obvious action/reward
type-swap bug. See `docs/anomalies.md` for what this implies about the
0.741 figure.

Also notable: at `β=1.0`, `C(beta) = 1.0 * C_emp + 0.0 * Uniform = C_emp`
exactly — i.e. the "β=1.0, 93.9%" row is the *real* classifier's empirical
confusion matrix, not synthetic noise. `effective_accuracy(C_emp)` was
independently recomputed during this reconnaissance and equals
`0.939047619047619` — matching the paper's "93.9%" exactly, confirming the
93.9% figure is the actual fitted classifier's accuracy, not a rounded
synthetic value.

## 6. Python version / dependency manager / lockfile

- **Python**: 3.13.11 (`backend/../.venv/Scripts/python.exe --version`).
- **Dependency manager**: plain `pip` + a single `backend/requirements.txt`
  (langchain, langgraph, langchain-groq, scikit-learn, streamlit,
  python-dotenv, pandas, numpy, matplotlib, joblib, plotly, scipy, fastapi,
  uvicorn[standard]). No `pyproject.toml`, `poetry.lock`, or `Pipfile*`
  anywhere in the repo.
- **Environment**: one `.venv/` at the repo root (`failure-aware-agent/.venv`,
  shared by `backend/`), not the more common per-subproject venv layout.
  `pytest` is **not currently installed** in this venv (`python -m pytest`
  → `No module named pytest`) — required appending `pytest` to
  `backend/requirements.txt` and installing it to satisfy the testing
  requirement (see summary at the end of this task).
