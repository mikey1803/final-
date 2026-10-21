# CLAUDE.md — Failure-Aware Adaptive Reasoning Framework

Recovery-strategy selection for LLM-agent tool failures, framed as a multi-armed bandit
(7 failure types × 6 recovery strategies). Central finding so far: a well-designed static rule
table beats adaptive methods in a stationary world; adaptive methods win once the world drifts —
the research maps where that crossover lives. Author: Pio Micheal (URK23AI1021), Karunya Institute
of Technology and Sciences; guide Ms. Divina.

**Current work — journal extension "v2"**: C1 diagnosis-aware recovery (Task 23), C2 drift-gated
recovery (Task 24), C3 real-world drift atlas (Task 25), plus supporting Tasks 20–22 and 26–29.
Task 20 scoping (not implemented): `docs/experiments/agent_trace_validation.md`.

## Layout
- `backend/src/` — Layer 1, original prototype (taxonomy, injector, classifier, selectors, SQLite memory, harness). **Read-only.**
- `backend/recovery/` — Layer 2, deterministic kernel: `theory/` (closed-form reference), `core/` (protocols, `SeedStreams`, `run_episodes`, `RunStore`), `guards/`, `envs/`, `classify/`, `policies/`, `analysis/`, `live/`. Design: `docs/architecture.md`.
- `backend/experiments/` — one script per phase/task; `backend/reproduce.py` — one-command pipeline.
- `backend/tests/` — pytest suite; `conftest.py` blocks writes outside `tmp_path`.
- `probe/` — real-world drift probe (Task 25, C3), run by `.github/workflows/drift-probe.yml`; collected data lives on the `data` branch only.
- `frontend/` (Streamlit + React), `backend/api/` (FastAPI demo), `study/` (blind-design instrument), `paper/` (`ieee_v3.tex`).
- **Git-ignored**: `backend/results/`, `backend/scratch/`, `.env`, `backend/data/*.db`. Result artifacts are on disk only, not in git history.

## Commands (from `backend/`, venv at `../.venv`)
```bash
python -m pytest                          # full suite
python reproduce.py --mode validation     # ~5 min — required after every task
python reproduce.py --mode paper          # ~65 min; horizon_sweep/prior_grid skipped unless --allow-expensive
python experiments/<script>.py --smoke    # <60 s, writes *_smoke.parquet
```

## Rules

1. **Never fabricate** numbers, results, datasets, citations, author names or affiliations. Anything not yet measured is written `[RESULT REQUIRED]`; any unverified reference `[CITATION REQUIRED]`. This includes the co-author TODOs in `paper/ieee_v3.tex`. The human study arm is n = 0 — never simulate or estimate it.
2. **Layer 1 (`backend/src/`) is not modified.** New code goes in `backend/recovery/` behind the existing protocols (`Policy`, `Environment`, `Classifier`, `Context`). If a Layer-1 change seems unavoidable (e.g. a bug), stop and ask (rule 10).
   Exception: `probe/` (repo root) is a standalone CI data-collection instrument, not kernel code.
3. **Frozen files are never edited**: `docs/live_pilot/strategy_definitions.md` (FROZEN v1), every pre-registration document (`study/preregistration.md` @ `PROTOCOL_VERSION 1.0.0`, `docs/theory/kappa_preregistration.md`, `docs/experiments/drift_atlas_preregistration.md` + `probe/targets.yaml`, and any new one), and any file a pre-registration names. Changes go in a new versioned file (`…_v2.md`); v1 and v2 data are never merged in one cell.
4. **Pre-registration before data.** Every pre-registration is committed to git — in its own commit — BEFORE the experiment it covers is run. The commit hash is the timestamp. Never run first.
5. **Never overwrite production results.** `--smoke` runs write `*_smoke.parquet` siblings, never production paths. Tests use `tmp_path` only and respect the write guard in `tests/conftest.py` (it does not reach subprocesses — any script a test shells out to must isolate its own output).
6. **Register every new experiment** in `reproduce.py` and `docs/reproducibility.md`, with seeds. Seeds come from `SeedSequence`/`SeedStreams` or `zlib.crc32` — never Python's `hash()` (per-process randomised; this already caused a real bug).
7. **Runtime invariants must hold** (`recovery/guards/invariants.py`): no policy beats the oracle. Keep the reward path structural — `Environment.sample_reward` takes only `true_type`; `Context` never carries the true type. `guards/` imports only numpy + `theory.analytic`; `core/` never imports `policies/`, `envs/`, `classify/`.
8. **Definition of done, every task:** full pytest suite green, `reproduce.py --mode validation` passes, `docs/PROJECT_STATUS.md` updated, changes committed in logical commits with clear messages. **Never push or force-push without asking.**
9. **Secrets**: API keys live only in `backend/.env` (git-ignored) locally and in GitHub Actions secrets in CI. Never print, log, or commit them, or echo `.env` contents.
10. **Open design decisions: stop and ask** with 2–3 options. Don't guess.

## Known traps (from this project's own incident history)
- **Stale status docs have caused two incidents.** When in-flight work finishes, is abandoned, or is reduced in scope, update the doc describing it in the same commit.
- **Superseded numbers — never cite as headline**: README's 500-ep table (Fixed Rule v1, 52.60%), Phase 2's Fixed Rule column, pre-`hash()`-fix Phase 3A values, legacy `harness.py` p-values as inferential evidence. Current numbers: `docs/PAPER_PREP.md` §3.
- **Claim-status vocabulary**: VALIDATED / PARTIALLY-SUPPORTED / UNVALIDATED-PRELIMINARY / NEGATIVE RESULT. Don't round PARTIALLY-SUPPORTED up (κ: R = 1.4729, CI [1.00, 9.55]).
- **Pre-registration timing gap**: the three pre-registrations that existed before v2 were first committed in `3999e27` (2026-08-25), after the κ result files (2026-08-18). Git history cannot evidence that they preceded their data; disclose, don't reword.
- **`--estimate` is unreliable** (assumes linear `n_jobs` speedup; 2–9× optimistic). The ~73 h / ~66 h figures are suspect. No concurrent compute while timing anything.
- **`horizon_sweep` / `prior_grid` are REDUCED scope** (`docs/experiments/instrumentation.md` Part 4) — don't describe them as full-scale.
- **`fit_kappa_preregistered.py` lives in git-ignored `backend/scratch/`** — the κ headline's script is not in git.
- **One-Shot LLM model is unresolved**: `llama-3.1-8b-instant` is gone from Groq; `llm_client.py` is a stopgap on `gpt-oss-20b`. Don't regenerate the One-Shot LLM row or `results/phase1/comparison_summary.csv` (currently smoke data on disk) until the model is chosen.
- **Gemini model pin**: Gemini echoes the `gemini-flash-lite-latest` alias — exact model unrecoverable. Record the resolved model id for any new LLM call.
- **Legacy-wrapped selectors ignore `streams.policy`** — CRN across them needs matching `seed=` args.

## Where things are
Numbers & methods: `docs/PAPER_PREP.md` · Task 19: `docs/TASK19_PAPER_BASE.md` · Kernel design: `docs/architecture.md` · Statistics audit: `docs/statistics_registry.md` · Provenance: `docs/experiments/instrumentation.md` · Reproduction: `docs/reproducibility.md` · Build history: `docs/TASK_HISTORY.md` · Drift atlas: `docs/experiments/drift_atlas.md`
