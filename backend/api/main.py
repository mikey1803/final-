"""Live-demo API: wraps the REAL trained classifier and REAL Thompson Sampling
selector (persistent SQLite state) behind HTTP endpoints for the React
frontend's interactive Live Simulation tab.

Run with:  uvicorn api.main:app --reload --port 8000   (from backend/)

Important: this uses its own SQLite file (data/demo_outcomes.db), never
backend/data/outcomes.db — that file backs the real evaluation results
reported in the paper/README, and a live demo clicking around must never
contaminate it.
"""

import random
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

from src.agent.llm_client import parse_tool_request  # noqa: E402
from src.evaluation.baselines import FixedRuleBaseline, LinUCBBaseline, NaiveRetryBaseline  # noqa: E402
from src.evaluation.classifier_context import load_or_train_classifier  # noqa: E402
from src.evaluation.nonstationarity import build_shifted_matrix, pick_shift_types  # noqa: E402
from src.failures.injector import FailureInjector  # noqa: E402
from src.failures.taxonomy import FailureType  # noqa: E402
from src.memory.store import OutcomeMemory  # noqa: E402
from src.recovery.selector import SelectorFactory, ThompsonSamplingSelector  # noqa: E402
from src.recovery.strategies import SUCCESS_MATRIX, RecoveryStrategy, get_action_cost  # noqa: E402
from api.tools import TOOLS, corrupt_and_call, call_tool  # noqa: E402

_DEMO_DB_PATH = _BACKEND_ROOT / "data" / "demo_outcomes.db"

app = FastAPI(title="Failure-Aware Agent — Live Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DemoState:
    """All live-demo state, held in one place so /reset can cleanly rebuild it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.classifier = load_or_train_classifier()
        self._rebuild_selector()

    def _rebuild_selector(self) -> None:
        self.memory = OutcomeMemory(db_path=_DEMO_DB_PATH)
        self.selector: ThompsonSamplingSelector = SelectorFactory.create(
            "thompson_sampling", memory=self.memory
        )
        self.memory.load_into_selector(self.selector)
        self.episode_number = len(self.memory.get_full_history())
        self.injector = FailureInjector(injection_rate=1.0)

        # Comparison-only selectors for /api/demo/compare-live. Fixed Rule and
        # Naive Retry are stateless; LinUCB learns, so it gets one persistent
        # (in-process, not SQLite-backed) instance that keeps improving across
        # calls the same way state.selector does, purely for a fair side-by-side.
        self.fixed_rule = FixedRuleBaseline()
        self.naive_retry = NaiveRetryBaseline()
        self.linucb = LinUCBBaseline(alpha=1.0)

        # Active SUCCESS_MATRIX for outcome sampling — swapped out by
        # /api/demo/trigger-shift, restored to the real matrix on reset.
        self.active_matrix = SUCCESS_MATRIX
        self.shift_active = False
        self.shifted_types: list[str] = []

    def reset(self) -> None:
        with self.lock:
            self.memory.reset()
            self.memory.close()
            self._rebuild_selector()


state = DemoState()


class InjectManualRequest(BaseModel):
    failure_type: str


class InjectNaturalRequest(BaseModel):
    user_request: str


def _confidences_json(selector: ThompsonSamplingSelector, failure_type: FailureType) -> dict:
    return {s.value: round(v, 4) for s, v in selector.get_confidences(failure_type).items()}


def _select_execute_update(predicted_type: FailureType, true_failure_type: FailureType) -> dict:
    """Core of one attempt: bandit selects (informed by whatever it currently
    believes — including any update from a prior failed attempt), the real
    active matrix decides the outcome, then the selector learns from it.
    Shared by the initial inject pipeline and /api/demo/retry so a retry is
    just this same core run again, not a duplicated implementation."""
    selector = state.selector
    scores_before = _confidences_json(selector, predicted_type)
    chosen_strategy = selector.select_strategy(predicted_type)

    p_success = state.active_matrix[true_failure_type][chosen_strategy]
    success = random.random() < p_success

    with state.lock:
        state.episode_number += 1
        episode_number = state.episode_number
        selector.update(
            predicted_type,
            chosen_strategy,
            success=success,
            cost=get_action_cost(chosen_strategy),
            episode_number=episode_number,
        )

    scores_after = _confidences_json(selector, predicted_type)

    return {
        "strategy_scores_before": scores_before,
        "strategy_scores_after": scores_after,
        "chosen_strategy": chosen_strategy.value,
        "outcome": "success" if success else "fail",
        "episode_number": episode_number,
    }


def _run_recovery_pipeline(
    true_failure_type: FailureType,
    raw_error: str,
    step_context: dict,
) -> dict:
    """Shared core: classify -> select -> sample outcome -> learn. Used by
    both the manual and natural injection endpoints so their behaviour
    (and the meaning of every returned field) is identical."""
    clf = state.classifier

    predicted_type = clf.predict(raw_error, step_context)
    proba = clf.predict_proba(raw_error, step_context)
    classifier_confidence = round(proba.get(predicted_type, 0.0), 4)

    core = _select_execute_update(predicted_type, true_failure_type)

    return {
        "true_failure_type": true_failure_type.value,
        "injected_error_text": raw_error,
        "predicted_failure_type": predicted_type.value,
        "classifier_confidence": classifier_confidence,
        **core,
    }


@app.post("/api/demo/inject-manual")
def inject_manual(body: InjectManualRequest):
    try:
        true_failure_type = FailureType(body.failure_type)
    except ValueError:
        valid = [f.value for f in FailureType]
        raise HTTPException(status_code=400, detail=f"Unknown failure_type. Use one of {valid}")

    step_context = {
        "step_index": state.episode_number + 1,
        "tool_name": random.choice(["web_search", "calculator", "file_reader", "api_caller"]),
        "goal": "demo: manually injected failure",
    }
    event = state.injector.generate_event(true_failure_type, step_context)
    return _run_recovery_pipeline(true_failure_type, event.raw_error, step_context)


@app.post("/api/demo/inject-natural")
def inject_natural(body: InjectNaturalRequest):
    if not body.user_request or not body.user_request.strip():
        raise HTTPException(status_code=400, detail="user_request must not be empty")

    tool_descriptions = [
        {"name": name, "description": t["description"], "args": t["args"]} for name, t in TOOLS.items()
    ]
    try:
        parsed = parse_tool_request(body.user_request, tool_descriptions)
        tool_name = parsed["tool"]
        args = parsed.get("args", {})
        if tool_name not in TOOLS:
            raise ValueError(f"LLM picked unknown tool {tool_name!r}")
    except Exception as exc:
        # LLM call failed or returned something unusable — fall back to a
        # fixed default rather than 500ing the whole demo.
        tool_name, args = "get_weather", {"city": "the requested location"}
        llm_note = f"LLM tool-parsing failed ({exc}); fell back to a default tool call."
    else:
        llm_note = None

    rng = random.Random()
    deliberate_failure = rng.random() < 0.4

    step_context = {
        "step_index": state.episode_number + 1,
        "tool_name": tool_name,
        "goal": body.user_request,
        "tool_args": args,
    }

    if deliberate_failure:
        raw_error, true_failure_type = corrupt_and_call(tool_name, args, rng)
        result = _run_recovery_pipeline(true_failure_type, raw_error, step_context)
        result["tool_called"] = tool_name
        result["tool_args"] = args
        result["agent_note"] = llm_note
        return result

    # No deliberate failure — really call the real tool. It can still
    # organically fail if the LLM supplied bad args; if so we route it
    # through the same pipeline (using the classifier's own prediction as
    # the operative "true" type, since there's no separate ground truth
    # for an unplanned failure).
    try:
        tool_result = call_tool(tool_name, args)
    except Exception as exc:
        raw_error = str(exc)
        step_context_for_predict = step_context
        clf = state.classifier
        predicted_type = clf.predict(raw_error, step_context_for_predict)
        result = _run_recovery_pipeline(predicted_type, raw_error, step_context)
        result["tool_called"] = tool_name
        result["tool_args"] = args
        result["agent_note"] = llm_note or "Tool call failed organically (not a deliberate injection)."
        return result

    return {
        "true_failure_type": None,
        "injected_error_text": None,
        "predicted_failure_type": None,
        "classifier_confidence": None,
        "strategy_scores_before": None,
        "strategy_scores_after": None,
        "chosen_strategy": None,
        "outcome": "success",
        "episode_number": state.episode_number,
        "tool_called": tool_name,
        "tool_args": args,
        "tool_result": tool_result,
        "agent_note": llm_note or "Tool call succeeded - no recovery needed.",
    }


class RetryRequest(BaseModel):
    true_failure_type: str
    predicted_failure_type: str
    attempt_number: int


@app.post("/api/demo/retry")
def retry(body: RetryRequest):
    """Re-runs Layers 2-3-4 (select -> execute -> learn) for the SAME
    already-classified failure, after a prior attempt failed. Attempt 1 is
    always the initial inject-manual/inject-natural call, so this endpoint
    only ever covers attempts 2 and 3 (3 attempts total, then escalate)."""
    try:
        true_failure_type = FailureType(body.true_failure_type)
        predicted_type = FailureType(body.predicted_failure_type)
    except ValueError:
        valid = [f.value for f in FailureType]
        raise HTTPException(status_code=400, detail=f"Unknown failure_type. Use one of {valid}")
    if body.attempt_number not in (2, 3):
        raise HTTPException(
            status_code=400,
            detail="attempt_number must be 2 or 3 (attempt 1 is the initial inject call; max 3 attempts total)",
        )

    core = _select_execute_update(predicted_type, true_failure_type)
    return {
        "true_failure_type": true_failure_type.value,
        "predicted_failure_type": predicted_type.value,
        "attempt_number": body.attempt_number,
        **core,
    }


@app.get("/api/demo/state")
def get_state():
    selector = state.selector
    best_per_type = {}
    all_confidences = {}
    for ft in FailureType:
        confidences = _confidences_json(selector, ft)
        all_confidences[ft.value] = confidences
        best_strategy = max(confidences, key=confidences.get)
        best_per_type[ft.value] = {"strategy": best_strategy, "confidence": confidences[best_strategy]}

    return {
        "episode_number": state.episode_number,
        "best_strategy_per_failure_type": best_per_type,
        "confidences": all_confidences,
        "shift_active": state.shift_active,
        "shifted_types": state.shifted_types,
    }


@app.post("/api/demo/reset")
def reset_demo():
    state.reset()
    return {"status": "reset", "episode_number": state.episode_number}


@app.get("/api/demo/health")
def health():
    return {"status": "ok"}


class CompareLiveRequest(BaseModel):
    failure_type: str


@app.post("/api/demo/compare-live")
def compare_live(body: CompareLiveRequest):
    """Run the same injected failure through 4 methods side by side: the
    real persistent Thompson selector (state.selector — this call feeds its
    learning same as inject-manual), Fixed Rule (blind, stateless), LinUCB
    (its own persistent-in-process instance), and Naive Retry (stateless)."""
    try:
        true_failure_type = FailureType(body.failure_type)
    except ValueError:
        valid = [f.value for f in FailureType]
        raise HTTPException(status_code=400, detail=f"Unknown failure_type. Use one of {valid}")

    matrix = state.active_matrix
    results = []

    ts_strategy = state.selector.select_strategy(true_failure_type)
    ts_success = random.random() < matrix[true_failure_type][ts_strategy]
    with state.lock:
        state.episode_number += 1
        episode_number = state.episode_number
        state.selector.update(
            true_failure_type, ts_strategy, success=ts_success,
            cost=get_action_cost(ts_strategy), episode_number=episode_number,
        )
    results.append({
        "method": "Thompson Sampling", "chosen_strategy": ts_strategy.value,
        "outcome": "success" if ts_success else "fail", "learns": True,
    })

    fr_strategy = state.fixed_rule.select_strategy(true_failure_type)
    fr_success = random.random() < matrix[true_failure_type][fr_strategy]
    results.append({
        "method": "Fixed Rule (blind)", "chosen_strategy": fr_strategy.value,
        "outcome": "success" if fr_success else "fail", "learns": False,
    })

    lu_strategy = state.linucb.select_strategy(true_failure_type)
    lu_success = random.random() < matrix[true_failure_type][lu_strategy]
    state.linucb.update(true_failure_type, lu_strategy, success=lu_success, cost=get_action_cost(lu_strategy))
    results.append({
        "method": "LinUCB", "chosen_strategy": lu_strategy.value,
        "outcome": "success" if lu_success else "fail", "learns": True,
    })

    nr_strategy = state.naive_retry.select_strategy(true_failure_type)
    nr_success = random.random() < matrix[true_failure_type][nr_strategy]
    results.append({
        "method": "Naive Retry", "chosen_strategy": nr_strategy.value,
        "outcome": "success" if nr_success else "fail", "learns": False,
    })

    return {
        "true_failure_type": true_failure_type.value,
        "episode_number": state.episode_number,
        "results": results,
    }


class BeatBanditRequest(BaseModel):
    failure_type: str
    human_strategy: str


@app.post("/api/demo/beat-the-bandit")
def beat_the_bandit(body: BeatBanditRequest):
    """Head-to-head round: a human-chosen strategy vs. the real persistent
    Thompson selector's own pick, both sampled against the same active
    matrix. The bandit only ever learns from its own choice (never the
    human's), same separation principle as /api/demo/compare-live."""
    try:
        true_failure_type = FailureType(body.failure_type)
    except ValueError:
        valid = [f.value for f in FailureType]
        raise HTTPException(status_code=400, detail=f"Unknown failure_type. Use one of {valid}")
    try:
        human_strategy = RecoveryStrategy(body.human_strategy)
    except ValueError:
        valid = [s.value for s in RecoveryStrategy]
        raise HTTPException(status_code=400, detail=f"Unknown human_strategy. Use one of {valid}")

    matrix = state.active_matrix
    human_success = random.random() < matrix[true_failure_type][human_strategy]

    bandit_core = _select_execute_update(true_failure_type, true_failure_type)
    bandit_won = bandit_core["outcome"] == "success"

    if human_success and not bandit_won:
        winner = "human"
    elif bandit_won and not human_success:
        winner = "bandit"
    else:
        winner = "tie"

    return {
        "true_failure_type": true_failure_type.value,
        "human_strategy": human_strategy.value,
        "human_outcome": "success" if human_success else "fail",
        "bandit_strategy": bandit_core["chosen_strategy"],
        "bandit_outcome": bandit_core["outcome"],
        "bandit_strategy_scores_before": bandit_core["strategy_scores_before"],
        "bandit_strategy_scores_after": bandit_core["strategy_scores_after"],
        "winner": winner,
        "episode_number": bandit_core["episode_number"],
    }


@app.post("/api/demo/trigger-shift")
def trigger_shift():
    """Permute SUCCESS_MATRIX for 2 failure types (real logic from
    src/evaluation/nonstationarity.py, used for the Phase 3B/4b experiments)
    and persist it as the active matrix — every subsequent inject-manual,
    inject-natural, compare-live, and batch-run call samples outcomes
    against the shifted matrix until /api/demo/reset."""
    rng = random.Random()
    shift_types = pick_shift_types(rng)
    shifted = build_shifted_matrix(shift_types, rng)
    with state.lock:
        state.active_matrix = shifted
        state.shifted_types = [ft.value for ft in shift_types]
        state.shift_active = True
    return {"status": "shifted", "shifted_types": state.shifted_types}


class BatchRunRequest(BaseModel):
    n_episodes: int = 50


@app.post("/api/demo/batch-run")
def batch_run(body: BatchRunRequest):
    """Rapidly simulate n_episodes of Thompson vs Fixed Rule server-side
    (harness-style loop, reusing the exact selector/baseline classes),
    against whatever matrix is currently active. Runs on fresh, throwaway
    selectors — never touches the persistent demo SQLite state — so
    clicking this doesn't perturb the interactive single-shot demo above.

    Also tracks a shifted-type-only cumulative rate alongside the all-types
    one, same reasoning as src/evaluation/nonstationarity.py's
    plot_nonstationarity_shifted_type_only: a shift only ever affects 2 of 7
    failure types, and failure types are sampled uniformly here, so the
    all-types rate dilutes the recovery signal with 5/7 untouched episodes
    where Fixed Rule is already near-optimal by design. Without this,
    whether the live demo visibly favors the adaptive method after a shift
    is mostly down to how many shifted-type episodes happened to land in
    this particular n_episodes draw, not the actual effect."""
    n = max(1, min(body.n_episodes, 500))
    matrix = state.active_matrix
    shift_active = state.shift_active
    shifted_types = set(state.shifted_types)
    rng = random.Random()
    types = list(FailureType)

    thompson = SelectorFactory.create("thompson_sampling", seed=rng.randint(0, 1_000_000))
    fixed = FixedRuleBaseline()

    thompson_successes = 0
    fixed_successes = 0
    thompson_shifted_successes = 0
    fixed_shifted_successes = 0
    shifted_episodes_seen = 0
    rows = []
    for ep in range(1, n + 1):
        ft = rng.choice(types)
        is_shifted_episode = shift_active and ft.value in shifted_types

        t_strategy = thompson.select_strategy(ft)
        t_success = rng.random() < matrix[ft][t_strategy]
        thompson.update(ft, t_strategy, success=t_success, cost=get_action_cost(t_strategy), episode_number=ep)
        thompson_successes += int(t_success)

        f_strategy = fixed.select_strategy(ft)
        f_success = rng.random() < matrix[ft][f_strategy]
        fixed_successes += int(f_success)

        if is_shifted_episode:
            shifted_episodes_seen += 1
            thompson_shifted_successes += int(t_success)
            fixed_shifted_successes += int(f_success)

        rows.append({
            "episode": ep,
            "thompson_success_rate": round(thompson_successes / ep, 4),
            "fixed_rule_success_rate": round(fixed_successes / ep, 4),
            "thompson_success_rate_shifted": (
                round(thompson_shifted_successes / shifted_episodes_seen, 4) if shifted_episodes_seen else None
            ),
            "fixed_rule_success_rate_shifted": (
                round(fixed_shifted_successes / shifted_episodes_seen, 4) if shifted_episodes_seen else None
            ),
        })

    return {
        "n_episodes": n,
        "shift_active": shift_active,
        "shifted_types": sorted(shifted_types),
        "shifted_episodes_seen": shifted_episodes_seen,
        "rows": rows,
    }


@app.get("/api/demo/explain/{failure_type}")
def explain(failure_type: str):
    """Templated explanation of the persistent Thompson selector's current
    belief for one failure type: pull/success counts per strategy plus the
    current top pick vs runner-up, straight off the live bandit state."""
    try:
        ft = FailureType(failure_type)
    except ValueError:
        valid = [f.value for f in FailureType]
        raise HTTPException(status_code=400, detail=f"Unknown failure_type. Use one of {valid}")

    selector = state.selector
    confidences = _confidences_json(selector, ft)
    ranked = sorted(confidences.items(), key=lambda kv: kv[1], reverse=True)
    top_name = ranked[0][0]
    second_name = ranked[1][0]

    per_strategy = {}
    total_seen = 0
    for rs in RecoveryStrategy:
        arm = selector._table[ft][rs]
        per_strategy[rs.value] = {"pulls": arm.attempts, "successes": arm.successes}
        total_seen += arm.attempts

    top, second = per_strategy[top_name], per_strategy[second_name]
    if total_seen > 0:
        explanation = (
            f"Seen {total_seen} times; {top_name} succeeded {top['successes']}/{top['pulls']} times "
            f"vs next-best {second_name} {second['successes']}/{second['pulls']} times."
        )
    else:
        explanation = f"Not seen yet - no data for {ft.value}."

    return {
        "failure_type": ft.value,
        "total_seen": total_seen,
        "top_strategy": top_name,
        "per_strategy": per_strategy,
        "explanation": explanation,
    }


class DiagnoseRequest(BaseModel):
    raw_error: str
    tool_name: Optional[str] = None


@app.post("/api/demo/diagnose")
def diagnose(body: DiagnoseRequest):
    """Read-only diagnostic: classify arbitrary error text with the real
    trained classifier and recommend the bandit's current best strategy for
    that predicted type. Unlike every other endpoint here, this never
    touches state.lock, episode_number, or selector.update — there is no
    ground truth for text pasted from outside the demo, so nothing here is
    allowed to feed fake learning into the persistent bandit."""
    if not body.raw_error or not body.raw_error.strip():
        raise HTTPException(status_code=400, detail="raw_error must not be empty")

    step_context = {"tool_name": body.tool_name or "unknown_tool"}
    clf = state.classifier
    predicted_type = clf.predict(body.raw_error, step_context)
    proba = clf.predict_proba(body.raw_error, step_context)
    confidences = _confidences_json(state.selector, predicted_type)
    top_strategy, top_confidence = max(confidences.items(), key=lambda kv: kv[1])

    return {
        "predicted_failure_type": predicted_type.value,
        "probabilities": {ft.value: round(p, 4) for ft, p in proba.items()},
        "recommended_strategy": top_strategy,
        "recommended_strategy_confidence": top_confidence,
        "strategy_confidences": confidences,
    }


class WhatIfMatrixRequest(BaseModel):
    matrix: dict[str, dict[str, float]]
    n_episodes: int = 200


@app.post("/api/demo/whatif-matrix")
def whatif_matrix(body: WhatIfMatrixRequest):
    """Race Thompson Sampling, Fixed Rule, LinUCB, and Naive Retry against a
    USER-SUPPLIED success-rate matrix instead of the real SUCCESS_MATRIX --
    makes the paper's Task 9 finding (2 of 3 hand-constructed alternative
    matrices break the headline "Fixed Rule wins stationary" ranking
    outright) something a visitor can test themselves, not just read.
    Fresh throwaway instances only (same pattern as batch_run) -- never
    touches the persistent demo state, and there is no ground truth to
    learn from here anyway since the matrix is arbitrary user input."""
    try:
        parsed: dict[FailureType, dict[RecoveryStrategy, float]] = {}
        for ft_key, row in body.matrix.items():
            ft = FailureType(ft_key)
            parsed[ft] = {}
            for strat_key, value in row.items():
                strat = RecoveryStrategy(strat_key)
                if not (0.0 <= value <= 1.0):
                    raise HTTPException(
                        status_code=400,
                        detail=f"probability out of [0,1] range for {ft_key}/{strat_key}: {value}",
                    )
                parsed[ft][strat] = value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid failure_type or strategy name: {exc}")

    missing = [ft.value for ft in FailureType if ft not in parsed]
    if missing:
        raise HTTPException(status_code=400, detail=f"matrix missing failure types: {missing}")
    for ft in FailureType:
        missing_strats = [s.value for s in RecoveryStrategy if s not in parsed[ft]]
        if missing_strats:
            raise HTTPException(status_code=400, detail=f"matrix missing strategies for {ft.value}: {missing_strats}")

    n = max(1, min(body.n_episodes, 500))
    rng = random.Random()
    types = list(FailureType)

    thompson = SelectorFactory.create("thompson_sampling", seed=rng.randint(0, 1_000_000))
    fixed = FixedRuleBaseline()
    linucb = LinUCBBaseline(alpha=1.0, seed=rng.randint(0, 1_000_000))
    naive = NaiveRetryBaseline()

    successes = {"thompson": 0, "fixed_rule": 0, "linucb": 0, "naive_retry": 0}
    rows = []
    for ep in range(1, n + 1):
        ft = rng.choice(types)

        t_strategy = thompson.select_strategy(ft)
        t_success = rng.random() < parsed[ft][t_strategy]
        thompson.update(ft, t_strategy, success=t_success, cost=get_action_cost(t_strategy), episode_number=ep)
        successes["thompson"] += int(t_success)

        f_strategy = fixed.select_strategy(ft)
        f_success = rng.random() < parsed[ft][f_strategy]
        successes["fixed_rule"] += int(f_success)

        l_strategy = linucb.select_strategy(ft)
        l_success = rng.random() < parsed[ft][l_strategy]
        linucb.update(ft, l_strategy, success=l_success, cost=get_action_cost(l_strategy))
        successes["linucb"] += int(l_success)

        n_strategy = naive.select_strategy(ft)
        n_success = rng.random() < parsed[ft][n_strategy]
        successes["naive_retry"] += int(n_success)

        rows.append({
            "episode": ep,
            "thompson_success_rate": round(successes["thompson"] / ep, 4),
            "fixed_rule_success_rate": round(successes["fixed_rule"] / ep, 4),
            "linucb_success_rate": round(successes["linucb"] / ep, 4),
            "naive_retry_success_rate": round(successes["naive_retry"] / ep, 4),
        })

    final = rows[-1]
    ranking = sorted(
        [
            ("Thompson Sampling", final["thompson_success_rate"]),
            ("Fixed Rule", final["fixed_rule_success_rate"]),
            ("LinUCB", final["linucb_success_rate"]),
            ("Naive Retry", final["naive_retry_success_rate"]),
        ],
        key=lambda kv: kv[1],
        reverse=True,
    )

    return {
        "n_episodes": n,
        "rows": rows,
        "final_ranking": [{"method": m, "success_rate": r} for m, r in ranking],
        "fixed_rule_wins": ranking[0][0] == "Fixed Rule",
    }
