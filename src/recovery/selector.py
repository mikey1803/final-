"""Adaptive recovery-strategy selector using multi-armed bandit algorithms.

The ``AdaptiveSelector`` maintains a table of per-(FailureType,
RecoveryStrategy) statistics and uses UCB1, epsilon-greedy, or Thompson
Sampling to choose which strategy to try next.  Over many episodes it
converges on the best recovery strategy for each failure type,
effectively learning the hidden success-rate matrix in
``strategies.py`` through experience alone.

Optionally accepts an ``OutcomeMemory`` instance so that every outcome
is persisted to SQLite in real time, allowing learning to resume across
runs.
"""

import math
import random
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.failures.taxonomy import FailureType
from src.recovery.strategies import RecoveryStrategy

if TYPE_CHECKING:
    from src.memory.store import OutcomeMemory

_ALGORITHMS = ("ucb1", "epsilon_greedy", "thompson_sampling")


class _ArmStats:
    """Per-(failure_type, strategy) running statistics."""

    __slots__ = ("attempts", "successes", "total_reward")

    def __init__(self) -> None:
        self.attempts: int = 0
        self.successes: int = 0
        self.total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.attempts if self.attempts > 0 else 0.0


class AdaptiveSelector:
    """Learns the best recovery strategy for each failure type via bandits.

    Args:
        algorithm: One of ``"ucb1"``, ``"epsilon_greedy"``, or
            ``"thompson_sampling"``.
        epsilon: Exploration probability for epsilon-greedy (ignored
            for other algorithms).
        seed: Optional RNG seed for reproducibility.
        memory: Optional ``OutcomeMemory`` instance.  When provided,
            every ``update()`` call is automatically persisted to the
            SQLite database so that learning survives across runs.
    """

    def __init__(
        self,
        algorithm: str = "ucb1",
        epsilon: float = 0.15,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
    ) -> None:
        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'; use one of {_ALGORITHMS}."
            )
        self.algorithm = algorithm
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._memory = memory

        # table[FailureType][RecoveryStrategy] -> _ArmStats
        self._table: dict[FailureType, dict[RecoveryStrategy, _ArmStats]] = {
            ft: {rs: _ArmStats() for rs in RecoveryStrategy}
            for ft in FailureType
        }
        # total attempts across *all* arms for a given failure type
        self._total_attempts: dict[FailureType, int] = {ft: 0 for ft in FailureType}

        # Thompson Sampling Beta parameters: alpha = 1 + successes, beta = 1 + failures
        self._beta: dict[FailureType, dict[RecoveryStrategy, dict[str, int]]] = {
            ft: {rs: {"alpha": 1, "beta": 1} for rs in RecoveryStrategy}
            for ft in FailureType
        }

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_strategy(self, failure_type: FailureType) -> RecoveryStrategy:
        """Choose a recovery strategy for the given failure type."""
        dispatch = {
            "ucb1": self._select_ucb1,
            "epsilon_greedy": self._select_epsilon_greedy,
            "thompson_sampling": self._select_thompson,
        }
        return dispatch[self.algorithm](failure_type)

    def _select_ucb1(self, failure_type: FailureType) -> RecoveryStrategy:
        """UCB1: pick the arm with the highest upper confidence bound."""
        best_strategy: Optional[RecoveryStrategy] = None
        best_score = -1.0
        total = self._total_attempts[failure_type]

        for strategy in RecoveryStrategy:
            arm = self._table[failure_type][strategy]
            if arm.attempts == 0:
                return strategy
            avg = arm.avg_reward
            exploration = math.sqrt(2.0 * math.log(total + 1) / arm.attempts)
            score = avg + exploration
            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy  # type: ignore[return-value]

    def _select_epsilon_greedy(self, failure_type: FailureType) -> RecoveryStrategy:
        """Epsilon-greedy: explore randomly with probability epsilon."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(list(RecoveryStrategy))
        best_strategy: Optional[RecoveryStrategy] = None
        best_reward = -1.0
        for strategy in RecoveryStrategy:
            arm = self._table[failure_type][strategy]
            if arm.avg_reward > best_reward:
                best_reward = arm.avg_reward
                best_strategy = strategy
        return best_strategy  # type: ignore[return-value]

    def _select_thompson(self, failure_type: FailureType) -> RecoveryStrategy:
        """Thompson Sampling: sample from each arm's Beta distribution, pick highest."""
        samples: dict[RecoveryStrategy, float] = {}
        for strategy in RecoveryStrategy:
            params = self._beta[failure_type][strategy]
            samples[strategy] = self._rng.betavariate(params["alpha"], params["beta"])
        return max(samples, key=samples.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool,
        cost: float = 1.0,
        episode_number: int = 0,
    ) -> None:
        """Record the outcome of a recovery attempt.

        Args:
            failure_type: The type of failure that was recovered from.
            strategy: The strategy that was applied.
            success: Whether the recovery succeeded.
            cost: An optional cost weight (lower is better).
            episode_number: Episode counter, forwarded to the memory store.
        """
        arm = self._table[failure_type][strategy]
        reward = 1.0 if success else 0.0
        arm.attempts += 1
        arm.successes += int(success)
        arm.total_reward += reward
        self._total_attempts[failure_type] += 1

        # Update Thompson Sampling Beta parameters
        if success:
            self._beta[failure_type][strategy]["alpha"] += 1
        else:
            self._beta[failure_type][strategy]["beta"] += 1

        if self._memory is not None:
            self._memory.record(
                failure_type=failure_type.value,
                strategy=strategy.value,
                success=success,
                cost=cost,
                episode_number=episode_number,
            )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_stats_table(self) -> pd.DataFrame:
        """Return the full success-rate table as a pandas DataFrame."""
        data: dict[str, dict[str, float]] = {}
        for ft in FailureType:
            row: dict[str, float] = {}
            for rs in RecoveryStrategy:
                arm = self._table[ft][rs]
                row[rs.value] = arm.successes / arm.attempts if arm.attempts > 0 else float("nan")
            data[ft.value] = row
        return pd.DataFrame(data).T

    def get_attempts_table(self) -> pd.DataFrame:
        """Return the per-cell attempt counts as a DataFrame."""
        data: dict[str, dict[str, int]] = {}
        for ft in FailureType:
            row: dict[str, int] = {}
            for rs in RecoveryStrategy:
                row[rs.value] = self._table[ft][rs].attempts
            data[ft.value] = row
        return pd.DataFrame(data).T

    def get_best_strategies(self) -> dict[FailureType, RecoveryStrategy]:
        """Return the current best strategy for each failure type based on
        highest average reward (ties broken arbitrarily)."""
        best: dict[FailureType, RecoveryStrategy] = {}
        for ft in FailureType:
            best_rs: Optional[RecoveryStrategy] = None
            best_r = -1.0
            for rs in RecoveryStrategy:
                arm = self._table[ft][rs]
                if arm.avg_reward > best_r:
                    best_r = arm.avg_reward
                    best_rs = rs
            best[ft] = best_rs  # type: ignore[assignment]
        return best
