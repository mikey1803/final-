"""Baseline strategy-selection methods for comparative evaluation.

Each baseline implements a ``select_strategy(failure_type) ->
RecoveryStrategy`` interface that mirrors ``AdaptiveSelector``, so every
baseline can be swapped in interchangeably inside the evaluation harness.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.failures.taxonomy import FailureType
from src.recovery.strategies import SUCCESS_MATRIX, RecoveryStrategy


class BaseSelector(ABC):
    """Common interface for all selection methods."""

    @abstractmethod
    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        """Return a recovery strategy for the given failure type (and optional context)."""

    def reset(self) -> None:
        """Reset any internal state (no-op for stateless baselines)."""


# ======================================================================
# 1. NaiveRetryBaseline — always retries
# ======================================================================

class NaiveRetryBaseline(BaseSelector):
    """Always selects RETRY regardless of failure type."""

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        return RecoveryStrategy.RETRY


# ======================================================================
# 2. FixedRuleBaseline — a hand-coded but imperfect heuristic mapping
# ======================================================================

# --- v1: answer-key-informed (kept only as a documented historical record —
# this version was designed by looking at SUCCESS_MATRIX's actual values and
# deliberately engineering a 4-correct/3-near-miss split. NOT used by
# FixedRuleBaseline any more; see results/phase4/comparison_summary_v1_answer_key_informed.csv
# for its measured performance. ---
_FIXED_RULES_V1_ANSWER_KEY_INFORMED: dict[FailureType, RecoveryStrategy] = {
    FailureType.TOOL_TIMEOUT:          RecoveryStrategy.RETRY,        # best=0.70
    FailureType.TOOL_ERROR:            RecoveryStrategy.SWITCH_TOOL,  # best=0.75
    FailureType.CONTEXT_LOSS:          RecoveryStrategy.RE_GROUND,    # best=0.80
    FailureType.AMBIGUOUS_INSTRUCTION: RecoveryStrategy.CLARIFY,      # best=0.80
    FailureType.WRONG_TOOL_SELECTED:   RecoveryStrategy.REPLAN,       # 0.60 vs best 0.75 (SWITCH_TOOL)
    FailureType.HALLUCINATED_OUTPUT:   RecoveryStrategy.RE_GROUND,    # 0.55 vs best 0.75 (VERIFY)
    FailureType.MALFORMED_OUTPUT:      RecoveryStrategy.SWITCH_TOOL,  # 0.60 vs best 0.75 (VERIFY)
}

# --- v2: genuinely blind design. Produced by a fresh Claude subagent with no
# memory of this project and an explicit instruction not to read any files
# (0 tool calls made) — reasoning from the failure/strategy descriptions
# alone, with no access to SUCCESS_MATRIX. Locked in before ever being
# scored; not adjusted afterward. Scored 6/7 correct against SUCCESS_MATRIX
# once evaluated (only wrong_tool_selected -> replan misses the true best,
# switch_tool) — this was NOT engineered, it's what fell out naturally.
#
# Reasoning (verbatim from the blind design pass):
#   tool_timeout -> retry: transient issue, cheapest proportionate response
#     is to retry before assuming anything structural is broken.
#   tool_error -> switch_tool: the specific tool is malfunctioning; the
#     sub-task is still valid, so route it through a different tool.
#   wrong_tool_selected -> replan: a planning-level judgment error, not a
#     tool malfunction — fix belongs one level up, regenerate the approach.
#   hallucinated_output -> verify: a content-correctness problem, not a
#     mechanical failure — run a self-check/verification pass on the claim.
#   malformed_output -> verify: run a structural/format check against the
#     expected shape rather than blindly re-running the action.
#   context_loss -> re_ground: degraded internal state, not a bad plan or
#     tool — re-fetch the original task and prior context to re-anchor.
#   ambiguous_instruction -> clarify: the root cause is underspecified
#     input; only the user can supply the missing specificity.
_FIXED_RULES_V2_BLIND: dict[FailureType, RecoveryStrategy] = {
    FailureType.TOOL_TIMEOUT:          RecoveryStrategy.RETRY,
    FailureType.TOOL_ERROR:            RecoveryStrategy.SWITCH_TOOL,
    FailureType.WRONG_TOOL_SELECTED:   RecoveryStrategy.REPLAN,
    FailureType.HALLUCINATED_OUTPUT:   RecoveryStrategy.VERIFY,
    FailureType.MALFORMED_OUTPUT:      RecoveryStrategy.VERIFY,
    FailureType.CONTEXT_LOSS:          RecoveryStrategy.RE_GROUND,
    FailureType.AMBIGUOUS_INSTRUCTION: RecoveryStrategy.CLARIFY,
}

# FixedRuleBaseline uses the blind version — this is the one that's fair to
# compare against learning methods, since it wasn't designed with access to
# the ground truth it's being scored against.
_FIXED_RULES: dict[FailureType, RecoveryStrategy] = _FIXED_RULES_V2_BLIND


class FixedRuleBaseline(BaseSelector):
    """Picks a strategy via a hand-coded rule table.

    Uses ``_FIXED_RULES_V2_BLIND`` — designed by a fresh agent instance
    reasoning purely from the failure/strategy *descriptions*, with no
    access to SUCCESS_MATRIX (0 file reads during design; see the comment
    block above ``_FIXED_RULES_V2_BLIND`` for the verbatim reasoning and
    ``_FIXED_RULES_V1_ANSWER_KEY_INFORMED`` for the earlier, answer-key-
    informed version this replaced).
    """

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        return _FIXED_RULES.get(failure_type, RecoveryStrategy.RETRY)


# ======================================================================
# 3. RandomBaseline — uniform random pick every time
# ======================================================================

class RandomBaseline(BaseSelector):
    """Picks a uniformly random strategy on every call.

    Labelled "UniformRandom" in comparison tables — the explicit
    achievable *floor*, paired with :class:`OraclePolicy` (the ceiling)
    to normalize every other method's score.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        return self._rng.choice(list(RecoveryStrategy))


# ======================================================================
# 4. OneShotLLMBaseline — real zero-shot Groq call, no learning across episodes
# ======================================================================

# Hand-crafted keyword -> strategy relevance scores, used only as a
# fallback if GROQ_API_KEY isn't configured or a live call fails.
# Higher score = more likely the LLM would suggest this strategy.
_KEYWORD_SCORES: dict[str, dict[RecoveryStrategy, float]] = {
    "timeout": {
        RecoveryStrategy.RETRY: 0.8,
        RecoveryStrategy.SWITCH_TOOL: 0.5,
        RecoveryStrategy.REPLAN: 0.3,
        RecoveryStrategy.RE_GROUND: 0.1,
        RecoveryStrategy.VERIFY: 0.1,
        RecoveryStrategy.CLARIFY: 0.05,
    },
    "error": {
        RecoveryStrategy.RETRY: 0.6,
        RecoveryStrategy.SWITCH_TOOL: 0.7,
        RecoveryStrategy.REPLAN: 0.4,
        RecoveryStrategy.RE_GROUND: 0.2,
        RecoveryStrategy.VERIFY: 0.3,
        RecoveryStrategy.CLARIFY: 0.1,
    },
    "wrong": {
        RecoveryStrategy.RETRY: 0.2,
        RecoveryStrategy.SWITCH_TOOL: 0.8,
        RecoveryStrategy.REPLAN: 0.5,
        RecoveryStrategy.RE_GROUND: 0.3,
        RecoveryStrategy.VERIFY: 0.3,
        RecoveryStrategy.CLARIFY: 0.4,
    },
    "hallucin": {
        RecoveryStrategy.RETRY: 0.1,
        RecoveryStrategy.SWITCH_TOOL: 0.2,
        RecoveryStrategy.REPLAN: 0.4,
        RecoveryStrategy.RE_GROUND: 0.6,
        RecoveryStrategy.VERIFY: 0.7,
        RecoveryStrategy.CLARIFY: 0.5,
    },
    "malform": {
        RecoveryStrategy.RETRY: 0.4,
        RecoveryStrategy.SWITCH_TOOL: 0.3,
        RecoveryStrategy.REPLAN: 0.3,
        RecoveryStrategy.RE_GROUND: 0.2,
        RecoveryStrategy.VERIFY: 0.8,
        RecoveryStrategy.CLARIFY: 0.2,
    },
    "context": {
        RecoveryStrategy.RETRY: 0.1,
        RecoveryStrategy.SWITCH_TOOL: 0.1,
        RecoveryStrategy.REPLAN: 0.4,
        RecoveryStrategy.RE_GROUND: 0.8,
        RecoveryStrategy.VERIFY: 0.2,
        RecoveryStrategy.CLARIFY: 0.5,
    },
    "ambigu": {
        RecoveryStrategy.RETRY: 0.1,
        RecoveryStrategy.SWITCH_TOOL: 0.1,
        RecoveryStrategy.REPLAN: 0.3,
        RecoveryStrategy.RE_GROUND: 0.2,
        RecoveryStrategy.VERIFY: 0.2,
        RecoveryStrategy.CLARIFY: 0.9,
    },
}


class OneShotLLMBaseline(BaseSelector):
    """A real zero-shot LLM call (Groq, via ``src.agent.llm_client``)
    choosing a recovery strategy for a representative error of each
    failure type. Crucially there is **no learning** — every call is an
    independent zero-shot prompt with no memory of past outcomes.

    Responses are cached by error text at the **class level**, shared
    across every instance (i.e. every trial in the harness). Since each
    ``FailureType`` maps to one fixed representative error text, this
    baseline makes at most one real API call per failure type ever —
    typically 7 total for an entire multi-trial run, regardless of how
    many thousands of episodes are simulated.

    If ``GROQ_API_KEY`` isn't configured, or a live call raises, falls
    back to the keyword-heuristic softmax below rather than crashing —
    but this baseline only earns the name "One-Shot LLM" when a real key
    is present; check ``used_real_llm`` after a run to confirm which path
    actually fired.
    """

    _RESPONSE_CACHE: dict[str, RecoveryStrategy] = {}
    _REPRESENTATIVE_ERRORS: dict[FailureType, str] = {}
    used_real_llm: bool = False

    _TAXONOMY_DESCRIPTION = (
        "- tool_timeout: a tool call took too long or never responded\n"
        "- tool_error: a tool returned an error or a garbage/broken response\n"
        "- wrong_tool_selected: the agent picked an inappropriate tool for the subtask\n"
        "- hallucinated_output: the agent's output contains a fabricated/false fact\n"
        "- malformed_output: the output doesn't match the expected structure/format\n"
        "- context_loss: the agent lost track of the goal or earlier steps\n"
        "- ambiguous_instruction: the input was vague and got misinterpreted"
    )

    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        self._temperature = temperature
        self._rng = random.Random(seed)
        self._ensure_representative_errors()

    @classmethod
    def _ensure_representative_errors(cls) -> None:
        """Generate one fixed representative error string per FailureType
        (same text every run, so the class-level cache actually hits)."""
        if cls._REPRESENTATIVE_ERRORS:
            return
        from src.failures.injector import FailureInjector

        injector = FailureInjector(injection_rate=1.0, seed=4242)
        ctx = {
            "step_index": 0, "tool_name": "web_search",
            "goal": "example subtask", "tool_output": "prior output",
        }
        for ft in FailureType:
            event = injector._build_event(ft, ctx)
            cls._REPRESENTATIVE_ERRORS[ft] = event.raw_error

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        raw_error = self._REPRESENTATIVE_ERRORS[failure_type]
        if raw_error in self._RESPONSE_CACHE:
            return self._RESPONSE_CACHE[raw_error]

        strategy = self._call_llm(raw_error, failure_type)
        self._RESPONSE_CACHE[raw_error] = strategy
        return strategy

    def _call_llm(self, raw_error: str, failure_type: FailureType) -> RecoveryStrategy:
        from src.agent.llm_client import classify_recovery_strategy, get_groq_api_key

        if not get_groq_api_key():
            return self._fallback_choice(failure_type)

        try:
            strategy_names = [s.value for s in RecoveryStrategy]
            response_text = classify_recovery_strategy(raw_error, self._TAXONOMY_DESCRIPTION, strategy_names)
            OneShotLLMBaseline.used_real_llm = True
        except Exception as exc:
            print(f"  [OneShotLLMBaseline] live Groq call failed ({exc}); using keyword fallback.")
            return self._fallback_choice(failure_type)

        return self._parse_strategy(response_text)

    @staticmethod
    def _parse_strategy(response_text: str) -> RecoveryStrategy:
        cleaned = response_text.strip().lower()
        for strategy in RecoveryStrategy:
            if strategy.value in cleaned:
                return strategy
        return RecoveryStrategy.RETRY  # safe default if the reply didn't match anything

    def _fallback_choice(self, failure_type: FailureType) -> RecoveryStrategy:
        """Keyword-heuristic softmax choice — used only when no real LLM
        call is available."""
        ft_name = failure_type.value.lower()
        scores: dict[RecoveryStrategy, float] = {}
        for strategy in RecoveryStrategy:
            score = 0.0
            for keyword, strat_scores in _KEYWORD_SCORES.items():
                if keyword in ft_name:
                    score += strat_scores.get(strategy, 0.0)
            scores[strategy] = max(score, 1e-8)  # avoid zero

        import math
        max_s = max(scores.values())
        exp_scores = {k: math.exp((v - max_s) / self._temperature) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = [exp_scores[s] / total for s in RecoveryStrategy]
        return self._rng.choices(list(RecoveryStrategy), weights=probs, k=1)[0]


# ======================================================================
# 5. OraclePolicy — always plays the best strategy for the true failure type
# ======================================================================

_BEST_STRATEGY: dict[FailureType, RecoveryStrategy] = {
    ft: max(rates, key=rates.get) for ft, rates in SUCCESS_MATRIX.items()
}


class OraclePolicy(BaseSelector):
    """Always selects argmax_s SUCCESS_MATRIX[failure_type][s].

    This baseline has privileged access to the hidden ground-truth
    success-rate matrix that no real selector is given — it represents
    the achievable *ceiling*. Labelled "Oracle" in comparison tables,
    paired with :class:`RandomBaseline` (the "UniformRandom" floor) to
    normalize every other method's score.
    """

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        return _BEST_STRATEGY[failure_type]


# ======================================================================
# 6. LinUCBBaseline — plain disjoint linear contextual bandit
# ======================================================================

class LinUCBBaseline(BaseSelector):
    """Plain (disjoint) LinUCB contextual bandit baseline.

    Each recovery strategy is a separate arm with its own ridge-regression
    parameters ``(A, b)`` over a feature vector built from the failure
    type (one-hot) plus whatever incidental context features the harness
    passes in — retry count, tool identity, and time since the last
    failure. This is a genuine *baseline* for comparison, not the
    project's proposed method: unlike ``ContextualCostAwareSelector`` it
    has no cost-awareness term, and unlike Thompson/UCB1 it has no
    privileged access to a clean per-(type, strategy) table — it has to
    learn a shared linear model per arm over noisy incidental features.

    Needs an explicit ``update()`` call after every episode (like
    ``BanditSelector``), so callers must check for that — it isn't a
    ``BanditSelector`` subclass since it's presented as a baseline, not
    an adaptive method, but the interface is the same shape.

    Reference: Li et al., "A Contextual-Bandit Approach to Personalized
    News Article Recommendation" (LinUCB, WWW 2010).
    """

    KNOWN_TOOLS: list[str] = [
        "web_search", "calculator", "file_reader", "api_caller",
        "database_query", "code_executor", "translator", "summariser",
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        """``types``/``strategies`` optionally override the arm/feature index
        space (default: the real ``FailureType``/``RecoveryStrategy`` enums)
        — used by the taxonomy-size stress test to scale the number of
        failure-type arms without touching any other caller's behaviour."""
        self._rng = random.Random(seed)
        self.alpha = alpha
        self._types = list(types) if types is not None else list(FailureType)
        self._strategies = list(strategies) if strategies is not None else list(RecoveryStrategy)
        self._dim = len(self._types) + len(self.KNOWN_TOOLS) + 2 + 1  # +retry +time_since +bias
        self._A: dict[RecoveryStrategy, np.ndarray] = {s: np.eye(self._dim) for s in self._strategies}
        self._b: dict[RecoveryStrategy, np.ndarray] = {s: np.zeros(self._dim) for s in self._strategies}

    def _featurize(self, failure_type: FailureType, context: Optional[dict]) -> np.ndarray:
        ctx = context or {}
        x = np.zeros(self._dim)
        x[self._types.index(failure_type)] = 1.0
        tool = ctx.get("tool_name")
        if tool in self.KNOWN_TOOLS:
            x[len(self._types) + self.KNOWN_TOOLS.index(tool)] = 1.0
        offset = len(self._types) + len(self.KNOWN_TOOLS)
        x[offset] = ctx.get("retry_count", 0) / 3.0
        x[offset + 1] = ctx.get("time_since_last_failure", 0) / 10.0
        x[offset + 2] = 1.0  # bias term
        return x

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        x = self._featurize(failure_type, context)
        best_strategy, best_score = self._strategies[0], -np.inf
        for s in self._strategies:
            A_inv = np.linalg.inv(self._A[s])
            theta = A_inv @ self._b[s]
            score = float(theta @ x) + self.alpha * float(np.sqrt(x @ A_inv @ x))
            if score > best_score:
                best_score, best_strategy = score, s
        return best_strategy

    def update(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool,
        cost: float = 1.0,
        context: Optional[dict] = None,
        episode_number: int = 0,
    ) -> None:
        x = self._featurize(failure_type, context)
        reward = 1.0 if success else 0.0
        self._A[strategy] += np.outer(x, x)
        self._b[strategy] += reward * x
