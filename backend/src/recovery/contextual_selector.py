"""Contextual, cost-aware bandit selector (LinUCB-style).

Extends the basic bandit formulation by considering *context* (not just
``failure_type``) and by optimising a combined objective of success rate
AND recovery cost.

Key ideas
---------
1. **Context:** Each episode carries features such as ``retry_count``,
   ``tool_name``, and ``time_since_last_failure``.  A linear model per
   ``(failure_type, strategy)`` pair predicts expected reward from these
   features.

2. **Cost-aware reward:** ``reward = success - lambda_cost * normalized_cost``
   where ``lambda_cost`` (default 0.3) controls the trade-off between
   maximising success and minimising cost.

3. **LinUCB exploration bonus:**  Selection uses an upper-confidence
   bound computed from the ridge-regression covariance matrix, so the
   algorithm explores strategies whose reward estimates are still
   uncertain.
"""

from typing import Optional

import numpy as np

from src.failures.taxonomy import FailureType
from src.recovery.selector import BanditSelector
from src.recovery.strategies import RecoveryStrategy, get_action_cost


# ------------------------------------------------------------------
# Helper: build a feature vector from the context dict
# ------------------------------------------------------------------

_CONTEXT_FEATURE_DIM = 8


def _encode_context(context: Optional[dict]) -> np.ndarray:
    """Encode context dict into a fixed-length feature vector.

    Features:
        0: retry_count (bounded 0-10, normalised /10)
        1: time_since_last_failure (bounded 0-100, normalised /100)
        2-7: one-hot tool_name (max 6 tools, padded)
    """
    vec = np.zeros(_CONTEXT_FEATURE_DIM, dtype=np.float64)

    if context is None:
        context = {}

    # retry_count
    rc = min(context.get("retry_count", 0), 10) / 10.0
    vec[0] = rc

    # time_since_last_failure
    tslf = min(context.get("time_since_last_failure", 0), 100) / 100.0
    vec[1] = tslf

    # one-hot tool_name (bounded to 6 slots)
    tool_map = {
        "web_search": 0, "calculator": 1, "file_reader": 2,
        "api_caller": 3, "database_query": 4, "code_executor": 5,
    }
    tool_name = context.get("tool_name", "unknown")
    if tool_name in tool_map:
        vec[2 + tool_map[tool_name]] = 1.0

    return vec


# ------------------------------------------------------------------
# Normalisation helpers for cost
# ------------------------------------------------------------------

_MAX_STRATEGY_COST = max(get_action_cost(rs) for rs in RecoveryStrategy)


def _normalise_cost(cost: float) -> float:
    """Normalise a strategy cost into [0, 1]."""
    return cost / _MAX_STRATEGY_COST if _MAX_STRATEGY_COST > 0 else 1.0


# ======================================================================
# ContextualCostAwareSelector
# ======================================================================


class ContextualCostAwareSelector(BanditSelector):
    """Contextual bandit with cost-aware reward (LinUCB-style).

    Maintains a ridge-regression linear model per
    ``(failure_type, strategy)`` pair.  The reward is:

        reward = success - lambda_cost * normalised_cost

    where *success* is 0 or 1 and *normalised_cost* is the strategy's
    relative cost mapped to [0, 1].

    Attributes:
        lambda_cost: Trade-off parameter (default 0.3).  Higher values
            penalise expensive strategies more aggressively.
        alpha: Exploration bonus coefficient for the LinUCB confidence
            bound (default 0.25).
    """

    def __init__(
        self,
        lambda_cost: float = 0.3,
        alpha: float = 0.25,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
    ) -> None:
        super().__init__(seed=seed, memory=memory)
        self.lambda_cost = lambda_cost
        self.alpha = alpha

        # Ridge-regression parameters per (ft, rs) pair
        # A = X^T X + lambda_reg * I   (covariance matrix)
        # b = X^T y                     (reward vector)
        # theta = A^{-1} b              (model weights)
        lambda_reg = 1.0

        self._A: dict = {}
        self._b: dict = {}
        self._theta: dict = {}

        for ft in FailureType:
            for rs in RecoveryStrategy:
                key = (ft, rs)
                self._A[key] = lambda_reg * np.eye(_CONTEXT_FEATURE_DIM, dtype=np.float64)
                self._b[key] = np.zeros(_CONTEXT_FEATURE_DIM, dtype=np.float64)
                self._theta[key] = np.zeros(_CONTEXT_FEATURE_DIM, dtype=np.float64)

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        """Pick the strategy with the highest LinUCB upper confidence bound.

        For each strategy, computes:

            mu = theta^T x
            uncertainty = alpha * sqrt(x^T A^{-1} x)
            score = mu + uncertainty

        and returns the strategy with the highest score.
        """
        x = _encode_context(context)

        best_strategy: Optional[RecoveryStrategy] = None
        best_score = -1e9

        for strategy in RecoveryStrategy:
            key = (failure_type, strategy)
            # Compute mu = theta^T x
            mu = float(self._theta[key] @ x)

            # Compute uncertainty = alpha * sqrt(x^T A^{-1} x)
            try:
                A_inv = np.linalg.inv(self._A[key])
                uncertainty = self.alpha * np.sqrt(float(x @ A_inv @ x))
            except np.linalg.LinAlgError:
                uncertainty = 0.0

            score = mu + uncertainty

            if score > best_score:
                best_score = score
                best_strategy = strategy

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
        """Record outcome with a cost-aware reward and update the linear model.

        The composite reward is:

            reward = success - lambda_cost * normalised_cost

        This trains the ridge-regression model to prefer strategies that
        both succeed AND are inexpensive.
        """
        # Update base statistics
        arm = self._table[failure_type][strategy]
        arm.attempts += 1
        arm.successes += int(success)
        self._total_attempts[failure_type] += 1

        # Normalise cost using the action's actual cost
        norm_cost = _normalise_cost(cost)

        # Composite reward: success minus penalised normalised cost
        reward = float(success) - self.lambda_cost * norm_cost
        arm.total_reward += reward

        # Ridge regression update
        x = _encode_context(context).reshape(-1, 1)  # d x 1
        key = (failure_type, strategy)
        self._A[key] += x @ x.T
        self._b[key] += reward * x.flatten()
        try:
            self._theta[key] = np.linalg.solve(self._A[key], self._b[key])
        except np.linalg.LinAlgError:
            pass  # keep previous theta on singular matrix

        # Memory persistence
        ep = episode_number or (context or {}).get("episode_number", 0)
        self._record_memory(failure_type, strategy, success, cost, ep)
