"""Baseline strategy-selection methods for comparative evaluation.

Each baseline implements a ``select_strategy(failure_type) ->
RecoveryStrategy`` interface that mirrors ``AdaptiveSelector``, so all
five methods can be swapped in interchangeably inside the evaluation
harness.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional

from src.failures.taxonomy import FailureType
from src.recovery.strategies import RecoveryStrategy


class BaseSelector(ABC):
    """Common interface for all selection methods."""

    @abstractmethod
    def select_strategy(self, failure_type: FailureType) -> RecoveryStrategy:
        """Return a recovery strategy for the given failure type."""

    def reset(self) -> None:
        """Reset any internal state (no-op for stateless baselines)."""


# ======================================================================
# 1. NaiveRetryBaseline — always retries
# ======================================================================

class NaiveRetryBaseline(BaseSelector):
    """Always selects RETRY regardless of failure type."""

    def select_strategy(self, failure_type: FailureType) -> RecoveryStrategy:
        return RecoveryStrategy.RETRY


# ======================================================================
# 2. FixedRuleBaseline — a hand-coded but imperfect heuristic mapping
# ======================================================================

_FIXED_RULES: dict[FailureType, RecoveryStrategy] = {
    FailureType.TOOL_TIMEOUT:       RecoveryStrategy.RETRY,
    FailureType.TOOL_ERROR:         RecoveryStrategy.SWITCH_TOOL,
    FailureType.WRONG_TOOL_SELECTED: RecoveryStrategy.RETRY,
    FailureType.HALLUCINATED_OUTPUT: RecoveryStrategy.RE_GROUND,
    FailureType.MALFORMED_OUTPUT:   RecoveryStrategy.RETRY,
    FailureType.CONTEXT_LOSS:       RecoveryStrategy.RE_GROUND,
    FailureType.AMBIGUOUS_INSTRUCTION: RecoveryStrategy.VERIFY,
}


class FixedRuleBaseline(BaseSelector):
    """Picks a strategy via a hand-coded rule table.

    The rules represent a plausible-but-imperfect expert guess — e.g.
    someone who hasn't run experiments yet might think RETRY is always
    reasonable and that VERIFY helps with hallucinations (it does, but
    this mapping misses several better options).
    """

    def select_strategy(self, failure_type: FailureType) -> RecoveryStrategy:
        return _FIXED_RULES.get(failure_type, RecoveryStrategy.RETRY)


# ======================================================================
# 3. RandomBaseline — uniform random pick every time
# ======================================================================

class RandomBaseline(BaseSelector):
    """Picks a uniformly random strategy on every call."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_strategy(self, failure_type: FailureType) -> RecoveryStrategy:
        return self._rng.choice(list(RecoveryStrategy))


# ======================================================================
# 4. OneShotLLMBaseline — keyword heuristic, no learning across episodes
# ======================================================================

# Hand-crafted keyword -> strategy relevance scores.
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
    """Simulates a memoryless LLM choosing a recovery strategy.

    For each call it scores every strategy by summing keyword-match
    relevance weights extracted from the failure-type name, then
    softmax-samples a choice.  Crucially there is **no learning** —
    the LLM sees each failure in isolation with no memory of past
    outcomes, representing a zero-shot prompt-based approach.
    """

    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        self._temperature = temperature
        self._rng = random.Random(seed)

    def select_strategy(self, failure_type: FailureType) -> RecoveryStrategy:
        ft_name = failure_type.value.lower()
        scores: dict[RecoveryStrategy, float] = {}
        for strategy in RecoveryStrategy:
            score = 0.0
            for keyword, strat_scores in _KEYWORD_SCORES.items():
                if keyword in ft_name:
                    score += strat_scores.get(strategy, 0.0)
            scores[strategy] = max(score, 1e-8)  # avoid zero

        # softmax with temperature
        import math
        max_s = max(scores.values())
        exp_scores = {k: math.exp((v - max_s) / self._temperature) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = [exp_scores[s] / total for s in RecoveryStrategy]
        return self._rng.choices(list(RecoveryStrategy), weights=probs, k=1)[0]
