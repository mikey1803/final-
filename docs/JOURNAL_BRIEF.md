# Journal Brief — Failure-Aware Adaptive Reasoning Framework for Reliable Autonomous AI Agents

*A single "everything about this project" file: what it is, what was built, what was found, what is weak, and how to take it to the next level (journal paper, publication, career). Built from the repo's own docs (`README.md`, `docs/PROJECT_STATUS.md`, `docs/PAPER_PREP.md`, `docs/TASK19_PAPER_BASE.md`, `paper/ieee_v3.tex`). When a number matters for a paper, check it against the source doc named next to it.*

---

## 1. Project identity

| Field | Value |
| --- | --- |
| Project | Failure-Aware Adaptive Reasoning Framework for Reliable Autonomous AI Agents |
| Author | Pio Micheal, URK23AI1021 |
| Programme | Final-year B.Tech Artificial Intelligence and Data Science |
| Institution | Karunya Institute of Technology and Sciences, Coimbatore, India |
| Guide | Ms. Divina |
| Current paper title | *When Does Adaptive Recovery Pay? Crossover-Horizon Prediction for Failure Recovery under a Noisy Failure Classifier* |
| Paper draft | `paper/ieee_v3.tex` (IEEE format) → `paper/ieee_v3.pdf`; older draft in `paper/main.tex` |
| Repo location | `E:\final year project\failure-aware-agent` |

---

## 2. The idea in plain words

LLM agents that call tools fail in **recurring, classifiable ways**:

| Failure type | Meaning |
| --- | --- |
| `tool_timeout` | Tool is slow or never returns |
| `tool_error` | Tool returns an error or exception |
| `wrong_tool_selected` | Agent picks an unsuitable tool |
| `hallucinated_output` | Agent fabricates unsupported information |
| `malformed_output` | Wrong format (bad JSON, incomplete structure) |
| `context_loss` | Agent forgets earlier instructions or state |
| `ambiguous_instruction` | Instruction has several valid readings |

And there are 6 recovery strategies: `retry`, `replan`, `switch_tool`, `re_ground`, `verify`, `clarify`.

Normal engineering hardcodes one rule per failure ("timeout → retry"). This project instead treats every `(failure_type, recovery_strategy)` pair as a **multi-armed bandit arm** with an unknown success probability, and lets algorithms (Thompson Sampling, UCB1, epsilon-greedy, LinUCB) *learn* the best recovery from trial and error, without ever seeing the true success matrix.

### The real research question (what makes it a paper, not a demo)

> **When does adaptive recovery actually beat a hand-written rule table, and by how much?**

Answer found: not always.
- **Stationary world** (nothing changes): a well-designed static rule table **beats every adaptive method**, and the margin *grows* as the failure taxonomy gets bigger.
- **Drifting world** (a tool degrades, an API changes): the static rule cannot recover; adaptive methods do, faster with more practice.
- So the contribution is mapping **where the crossover lives** and quantifying it, not "adaptive is better".
- The newest paper (Task 19) adds the realistic wrinkle: the agent doesn't see the true failure type, it sees a **noisy classifier's prediction**. They derive how classifier noise moves the crossover horizon.

---

## 3. System architecture

### Layer 1 — Original prototype (`backend/src/`)

```text
Failure Injection -> Failure Classification -> Recovery Selection (bandit)
   -> Recovery Outcome -> Persistent Memory (SQLite) -> Evaluation / Dashboard
```

- `failures/taxonomy.py` — the 7 types
- `failures/injector.py` — seeded synthetic failure generator
- `failures/classifier.py` — TF-IDF + LinearSVC (plus logistic for confidence): **95.00% accuracy, 94.68% F1**
- `recovery/strategies.py` — 6 strategies + the hidden `SUCCESS_MATRIX`
- `recovery/selector.py` — Thompson, UCB1, epsilon-greedy (+ LinUCB, contextual cost-aware, prior-initialised, discounted variants)
- `memory/store.py` — SQLite outcome memory, replayable into a selector
- `evaluation/harness.py` — matched seeded comparisons, cumulative/late-stage success, regret, paired t-tests

### Layer 2 — Reproducible simulation kernel (`backend/recovery/`)

Added on top, without modifying Layer 1, to independently verify results.
- `theory/analytic.py` + `envs/canonical.py` — closed-form numpy reference to cross-check the simulation
- `core/` — one shared episode loop; `core/rng.py::SeedStreams` structurally stops the classifier's *predicted* type leaking into the reward
- `guards/invariants.py` — runtime assertions that crash on impossible numbers (e.g. beating the oracle)
- `core/parallel.py` (`RunStore`) — content-addressed parquet cache; each grid cell computed once, ever
- `policies/`, `envs/`, `classify/` — adapters behind common interfaces

### Layer 3 — Apps & instruments

| Piece | Location | Purpose |
| --- | --- | --- |
| Streamlit dashboard | `frontend/dashboard/` | Original fallback UI: live simulation, algorithm race, heatmap, classifier demo |
| React + Vite dashboard | `frontend/dashboard-react/` | Newer UI with a **Live Simulation** tab |
| FastAPI server | `backend/api/main.py` | Real classifier + real Thompson selector + real SQLite (`demo_outcomes.db`), not a replay |
| Blind-design study app | `study/` | Web instrument for humans/LLMs to hand-design a recovery table blind |
| Reproducibility pipeline | `backend/reproduce.py` | One command regenerates every analysis, figure, table |

**Tech stack:** Python, scikit-learn, NumPy/pandas/SciPy/Matplotlib/joblib, SQLite, FastAPI, Streamlit, React + Vite, Plotly, LangChain/LangGraph, Groq + Google Gemini + NVIDIA NIM (3 LLM families for the LLM-arm studies), LaTeX (IEEE).

---

## 4. The experimental program (`backend/experiments/`, ~24 scripts)

| Phase / Task | What it establishes |
| --- | --- |
| 1 | 11-method baseline comparison |
| 2 | Classifier-noise sweep |
| 3A / 3B | Random reward-matrix family sweep; first non-stationarity test |
| **4** | **Key pivot:** original Fixed Rule baseline (v1) was contaminated by knowledge of the reward structure → rebuilt as v2 via a *blind-design protocol*. Core dichotomy established |
| 4b | Reconciles stationary vs non-stationary results |
| 5 | Static rule needs ≈ **64% quality (4.5/7 rows right)** to match the best adaptive method in a stationary world |
| 6 | Shift frequency × magnitude sweep + adaptive recovery half-life |
| 7 | Taxonomy-size stress test: the static advantage holds and strengthens at 4× taxonomy |
| 8 / Task 11 | Blind-design elicitation study + LLM arm (3 model families). `malformed_output` is a weak point, but after correction **significant in only 1 of 3 families** |
| 9 / Task 12 | SUCCESS_MATRIX robustness: 2,204 perturbed worlds. Breaks at Gaussian σ=0.15 and rank-breaking fraction 0.25. **2 of 3 structured alternative matrices break the headline ranking** |
| 13 | Statistical rigor audit (Holm-Bonferroni where needed; found and fixed a `hash()` non-determinism bug) |
| 14 | Runtime/memory benchmarks, LLM cost accounting (~$0.0024 hypothetical), environment/provenance capture |
| 15 | Repo integrity recovery after `--smoke` runs overwrote six production result files; root cause actually closed on 2026-08-20 |
| 16 | One-command reproducibility pipeline (`reproduce.py`, validation ≈5 min / paper ≈65 min) |
| **19A** | Classifier-noise-aware crossover extension (exact Bayes mixture matrix `P_eff`) |
| **19B** | Drift-crossover unified law — documented **negative result**; one qualitative finding: *reversion, not magnitude, governs the sign* |

Also: drift taxonomy, horizon sweep, crossover calibration, prior-strength grid (`prior_grid.py`), prompt-robustness, matrix-robustness studies.

---

## 5. Headline results

### Original run (500 episodes, 30 trials, late-stage = last 20%)

| Method | Late-stage success |
| --- | ---: |
| Thompson Sampling | 65.50% |
| UCB1 | 57.87% |
| Fixed Rule (v1, **superseded**) | 52.60% |
| One-Shot LLM | 47.47% |
| Random | 41.87% |
| Naive Retry | 31.23% |

⚠ This table predates the Fixed Rule v1→v2 correction. **Do not put it in a journal paper as the headline.** Use the corrected numbers in `docs/PAPER_PREP.md` §3.

### Task 19 paper claims (`paper/ieee_v3.tex` abstract)

- Mean classifier diagonal accuracy 0.939
- Ceiling loss exceeds learnability loss by about **6:1**
- Classifier noise raises the static table's regret rate by **58%**, yet the predicted crossover only moves **3053 → 3126 episodes**
- The crossover horizon is provably **not monotonic** in classifier accuracy (proved by construction)
- Pre-registered test of the free constant κ: held-out **R = 1.4729** (threshold 1.5) but bootstrap CI **[1.00, 9.55]** → classified **PARTIALLY-SUPPORTED**

### Component status (from `TASK19_PAPER_BASE.md` Part 6)

| Component | Status |
| --- | --- |
| `effective_matrix_under_confusion`, `static_rate_confused`, `ceiling_reduction`, `gap_shrinkage_report`, `argmax_inversion_check`, T* non-monotonicity | VALIDATED |
| `ts_cumulative_regret_confused`, `crossover_horizon_confused` / κ | PARTIALLY-SUPPORTED |
| Drift-crossover unified law (19B) | NEGATIVE RESULT (documented) |

---

## 6. What is genuinely strong

1. **Honest science.** Pre-registration, a caught baseline-contamination bug (v1→v2), a caught non-determinism bug, a disproven hypothesis reported as such, negative results kept. Reviewers respond well to this.
2. **Reproducibility.** One command, no API keys needed for offline reproduction, pinned `requirements.lock.txt`, provenance map for every figure/table, determinism regression tests, runtime invariants.
3. **Real depth.** Closed-form theory cross-checked against simulation, 30+ test files, statistical audit.
4. **Full stack.** Classifier, selectors, memory, API, two dashboards, a study instrument — rare for a student project.
5. **Clear question with a non-obvious answer** ("static rules win until the world drifts").

---

## 7. Known weaknesses (fix or disclose before submission)

| # | Weakness | Why a reviewer will care |
| --- | --- | --- |
| 1 | **Everything is simulated.** Failures are synthetic and the success matrix is hand-authored. No real agent logs. | Biggest threat: external validity |
| 2 | **One canonical 7×6 matrix** drives most claims; 2 of 3 alternative structures break the ranking. | Result is conditional, not general |
| 3 | **Human blind-design arm: n = 0.** Awaiting institutional ethics approval. | The "distribution of rule quality q" is only LLM-sourced |
| 4 | κ validation is **partially supported**; CI [1.00, 9.55] is very wide. Tier-1 validation limited to one matrix family at gap = 0.20. | Core predictive claim is not confirmed |
| 5 | Task 10 predictor reliable only in a narrow regime (N = 7, quality ≤ 1/7); at the canonical table it **underestimates crossover ~50×**. | Predictor Task 19 extends was never fully validated |
| 6 | Classifier trained on synthetic injector text (95% accuracy) — real agent error text is messier. | Accuracy figure may not transfer |
| 7 | Two huge grids (`horizon_sweep` ≈73 h, `prior_grid` ≈66 h) run only at **reduced documented scope**. | Weakens horizon/prior claims |
| 8 | Gemini model pin not recoverable (`gemini-flash-lite-latest` alias); NVIDIA pricing unconfirmed. | Provenance gap |
| 9 | Several figures use an old colour scheme; `paper/ieee_v3.tex` has TODOs (co-author full name/e-mail/department) | Polish |
| 10 | **Git history is thin and uncommitted**: parent repo `master` has no commits; the `failure-aware-agent` content is staged/untracked. Project docs say only 2 commits exist inside it. | Risk of losing work; needed for artifact release |
| 11 | Related-work citations not finalised (`PAPER_PREP.md` §9 says none were fabricated; you must supply them). | Journals require a literature positioning |
| 12 | Stale docs: `PROJECT_STATUS.md` (last updated 2026-08-04) still describes Task 14 as in progress. | Doc drift has already caused two incidents |

---

## 8. How to take it to the next level

### A. Make the journal paper land (highest priority)

1. **Pick the venue first**, then format. Realistic tiers for this work:
   - *Workshops* (NeurIPS/ICML/ICLR agent-reliability or RL-in-practice workshops, AAAI workshops): good fit for a negative-results-plus-rigor paper.
   - *Conferences* (IEEE/Springer applied-AI venues, AAMAS-style).
   - *Journals* (e.g. an IEEE Access-type or Springer applied-AI journal; also reproducibility-oriented venues like ML Reproducibility Challenge-style tracks, or ReScience).
   Verify current scope, page limits and fees on each venue's site yourself — don't rely on this list.
2. **Choose ONE paper story.** The repo supports two:
   - *Paper 1 — "When does adaptive recovery pay?"* (Phases 4–7: static vs adaptive, crossover, drift, taxonomy scaling). Strongest and most complete.
   - *Paper 2 — noisy-classifier crossover theory* (Task 19, the current `ieee_v3.tex`). More theoretical, partially supported.
   Don't cram both. For a journal you can write Paper 1 as the main article and fold Task 19 in as a section, or keep them separate.
3. **Lead with the corrected numbers** (post v2), never the 52.60% Fixed Rule table.
4. **Reframe κ honestly**: "calibrated predictor with a documented validity region" — already the plan; keep it.
5. **Write Related Work** with real citations: Thompson 1933, Auer et al. 2002 (UCB1), Lai & Robbins 1985, Li et al. 2010 (LinUCB), Garivier & Cappé on KL-UCB, non-stationary bandits (discounted/sliding-window UCB, Garivier & Moulines), and agent-failure/reliability work (ReAct, Reflexion, Self-Refine, agent-benchmark papers). Look every one up; do not cite from memory.
6. **Add a threats-to-validity section** from §7 above — most of the text already exists in `PAPER_PREP.md` §6.
7. Fill the `ieee_v3.tex` TODOs, run `paper/scripts/verify_tex.py` and `word_count.py`, regenerate figures with a consistent style.

### B. Close the biggest scientific gap: real data

Pick at least one:
- **Real agent failure traces.** Run a LangGraph/ReAct agent on a public tool-use benchmark (e.g. τ-bench, ToolBench, AgentBench, SWE-style tasks — check what is current) with a real tool-failure injector; log the error text and recovery outcomes; build the success matrix from reality instead of authoring it.
- **Re-train the classifier on real error messages** (and report accuracy honestly, probably lower than 95%).
- **The live pilot** already started in `docs/live_pilot/strategy_definitions.md` — finish it.
- **Human arm of the blind-design study**: get approval from the Karunya ethics/IRB route via Ms. Divina, run ≥ 30 participants (classmates/juniors), preregistered in `study/preregistration.md`.

Even one real-trace experiment moves this from "simulation study" to "validated on real agents" and changes reviewer reception substantially.

### C. Strengthen the statistics & theory

- Run the multi-day grids to full scale (use a cloud box or leave a machine on) and remove the "REDUCED" flags.
- Test more matrix families and gap levels (not only gap = 0.20) to resolve the κ generalisation limit.
- Tighten the κ bootstrap interval with more independent environments (more than 5 finite cells in the test set).
- Add a regret lower bound or a proper non-asymptotic Thompson bound instead of the Lai-Robbins constant fitted empirically.
- Try other adaptive algorithms in the crossover study (sliding-window TS, change-point detection, Bayesian online change detection) — the drift story is where adaptive methods are meant to shine.

### D. Make it a usable product / open-source release

- **Package it**: turn `backend/recovery/` into an installable library (`pip install failure-aware-recovery`) with a 10-line quickstart.
- **LangGraph / LangChain integration**: a `RecoveryNode` that wraps any tool call, classifies the failure, picks a strategy via Thompson Sampling and logs outcomes. This is the single most "real-world" feature you could add.
- **Real strategy executors**: right now strategies are probabilities. Implement actual `retry` (backoff), `replan`, `switch_tool`, `re_ground` (re-retrieve context), `verify`, `clarify` for a real agent.
- **Cost-aware & contextual selection** using task features, tool name, latency, token cost (LinUCB/contextual bandit are already in the repo).
- **Safety**: cap retries, escalate to a human when confidence is low or after N failures.
- **Dashboard → monitoring**: expose `/metrics` for Prometheus/Grafana, deploy the React dashboard + FastAPI (Docker Compose), host a public demo.
- **CI**: GitHub Actions running `pytest` + `reproduce.py --mode validation`; badges in the README; tagged release and a Zenodo DOI for the artifacts.

### E. Housekeeping that protects everything (do this now)

1. **Commit the work.** `git status` shows `failure-aware-agent` only staged/untracked; the parent repo has zero commits. Make clean, logical commits (kernel, experiments, study, docs, paper) and push to GitHub (private first). *Check `.gitignore` covers `.env`, `node_modules`, `*.log`, large parquet caches, `.venv`, before the first push — do not publish API keys.*
2. Refresh `docs/PROJECT_STATUS.md` (still dated 2026-08-04) so it matches reality.
3. Rotate any API key that ever sat in `backend/.env` if the repo is going public.
4. Remove or ignore `streamlit_stdout.log` / `streamlit_stderr.log`.
5. Archive a final tagged snapshot (`v1.0-submission`) that exactly matches the paper's numbers.

### F. Career leverage

- Portfolio: you already have a React portfolio (`../portfoli`). Add this as **Project 01 — featured research**: lead with the finding ("static rules win until the world drifts"), link the paper PDF, the repo, and a live demo. At the moment the portfolio's featured projects are Kintsugi AI, TrustLens+ and the Forex system; this project is the strongest and isn't there.
- Put "pre-registered, reproducible, negative results reported" in your CV/LinkedIn — it's a distinctive signal for ML research or ML-engineering roles and for MS applications.
- Target: workshop paper → arXiv preprint (cs.LG / cs.AI) as soon as the draft is stable; an arXiv link helps applications even before peer review is complete.
- Ask Ms. Divina about co-authorship formalities, institutional affiliation line and any internal publication-approval process.

---

## 9. Suggested roadmap

| When | Goal | Deliverable |
| --- | --- | --- |
| **Week 1** | Safety + clarity | Commit & push repo, refresh status doc, tag snapshot, fix `ieee_v3.tex` TODOs |
| **Weeks 2–3** | Paper structure | Decide venue; choose Paper 1 vs Paper 2; full Related Work with verified citations; regenerate consistent figures |
| **Weeks 3–6** | Real-data experiment | Real agent traces + success matrix + real error text classifier (even a small one) |
| **Weeks 5–8** | Human study | Ethics approval → run blind-design arm, analyse with the existing pipeline |
| **Weeks 6–9** | Library & demo | `pip` package, LangGraph node, Docker Compose, public demo, CI |
| **Weeks 9–10** | Submit | arXiv preprint + venue submission, Zenodo DOI for artifacts |

---

## 10. How to run things

```bash
# Backend tests
cd backend && python -m pytest

# Reproduce every analysis/figure/table (no API keys needed)
cd backend
python reproduce.py --mode validation   # ~5 min
python reproduce.py --mode paper        # ~65 min (two multi-day sweeps skipped by default)

# Dashboards
streamlit run frontend/dashboard/app.py                      # from repo root
cd backend && uvicorn api.main:app --reload --port 8000      # terminal 1 (live demo API)
cd frontend/dashboard-react && npm install && npm run dev    # terminal 2, open http://localhost:5173

# Or both with one command from repo root
npm run dev
```

`--smoke` = fast (<60 s) validation run. Every experiment script now writes `--smoke` output to a `*_smoke.parquet` sibling so production results are not overwritten.

---

## 11. Where to look for what

| Question | File |
| --- | --- |
| Plain-language walkthrough | `docs/PROJECT_EXPLAINED.md` |
| What each of the tasks built | `docs/TASK_HISTORY.md` |
| Exact methodology and numbers per phase | `docs/PAPER_PREP.md` §3 |
| Core contributions | `docs/PAPER_PREP.md` §4 |
| Limitations | `docs/PAPER_PREP.md` §6 |
| Task 19 definitions, equations, results | `docs/TASK19_PAPER_BASE.md` |
| Crossover theory and practitioner guide | `docs/theory/crossover.md`, `docs/theory/practitioner_guidelines.md` |
| Kappa pre-registration | `docs/theory/kappa_preregistration.md` |
| Statistical audit | `docs/statistics_registry.md` |
| Reproduce a figure or table | `docs/reproducibility.md`, `docs/experiments/instrumentation.md` |
| Blind-design study | `docs/experiments/human_study.md`, `study/` |
| Kernel design | `docs/architecture.md` |
| Numerical anomaly investigation | `docs/anomalies.md`, `docs/reconciliation_stationary.md` |
| The paper | `paper/ieee_v3.tex`, `paper/ieee_v3.pdf` |

---

## 12. Related project in the same folder: `portfoli/`

A separate personal portfolio site (React 19, Vite, TypeScript, Tailwind v4, Motion, Lenis; comic/multiverse visual theme; deploys via Vercel). Sections: Hero, Introduction, FeaturedWork, Capabilities, Experience, Education, Certifications, CurrentlyBuilding, BeyondTheCode, Contact. Content lives in `src/data/portfolio.ts`.

Profile data in it: ML Intern at Prodigy Infotech (Jun 2025), Frontend Intern at SwiftAnt Technologies (Jun 2024); AWS Academy certifications (Cloud Data Pipeline Builder, Microservices and CI/CD Pipeline Builder, Apr 2026) and Cisco CCNA: Introduction to Networks (Apr 2025); other projects: Kintsugi AI (mental-wellness app), TrustLens+ (fake-review forensics), Forex Auto-Trading System. **Action:** add this failure-aware-agent research project as the lead case study.

---

## 13. One-paragraph pitch (reusable for abstract, CV, LinkedIn)

> I built and rigorously evaluated a failure-recovery framework for autonomous LLM agents that treats recovery-strategy choice as a multi-armed bandit problem. Across nine experimental phases and a pre-registered theory extension, I showed that a well-designed static rule table beats adaptive learners in stationary environments (by a margin that grows with taxonomy size) while adaptive learners win once the environment drifts, quantified the crossover horizon, and derived how a noisy failure classifier shifts it. The work includes a deterministic, cross-checked simulation kernel, a one-command reproducibility pipeline, a statistical audit that caught and fixed a real non-determinism bug, and a live FastAPI/React demo.
