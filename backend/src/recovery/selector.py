"""Adaptive recovery-strategy selector using multi-armed bandit algorithms.

The ``BanditSelector`` ABC defines the common interface.  Three concrete
implementations — ``UCB1Selector``, ``EpsilonGreedySelector``, and
``ThompsonSamplingSelector`` — capture the classic bandit algorithms.

Use ``SelectorFactory.create(algorithm, **kwargs)`` to obtain the right
instance instead of instantiating subclasses directly.

Optionally accepts an ``OutcomeMemory`` instance so that every outcome
is persisted to SQLite in real time, allowing learning to resume across
runs.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.failures.taxonomy import FailureType
from src.recovery.strategies import RecoveryStrategy

if TYPE_CHECKING:
    from src.memory.store import OutcomeMemory


class BanditSelector(ABC):
    """Abstract base for all bandit-based recovery-strategy selectors.

    Each subclass maintains its own internal statistics and implements
    the selection / update logic for the specific bandit algorithm.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        """``types``/``strategies`` optionally override the arm-table's index
        space (default: the real ``FailureType``/``RecoveryStrategy`` enums)
        — used by the taxonomy-size stress test to scale the number of
        failure-type arms without touching any other caller's behaviour.
        """
        self._rng = random.Random(seed)
        self._memory = memory
        self._types = list(types) if types is not None else list(FailureType)
        self._strategies = list(strategies) if strategies is not None else list(RecoveryStrategy)
        # table[failure_type][strategy] -> _ArmStats
        self._table: dict = {
            ft: {rs: _ArmStats() for rs in self._strategies}
            for ft in self._types
        }
        self._total_attempts: dict = {ft: 0 for ft in self._types}

    @abstractmethod
    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        """Choose a recovery strategy for the given failure type (and optional context)."""

    @abstractmethod
    def update(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool,
        cost: float = 1.0,
        context: Optional[dict] = None,
        episode_number: int = 0,
    ) -> None:
        """Record the outcome of a recovery attempt.

        Args:
            failure_type: The type of failure that was recovered from.
            strategy: The strategy that was applied.
            success: Whether the recovery succeeded.
            cost: An optional cost weight (lower is better).
            context: Optional contextual information about the episode.
            episode_number: Episode counter, forwarded to the memory store.
        """

    # ------------------------------------------------------------------
    # Reporting (shared across all subclasses)
    # ------------------------------------------------------------------

    def get_stats_table(self) -> pd.DataFrame:
        """Return the full success-rate table as a pandas DataFrame."""
        data: dict[str, dict[str, float]] = {}
        for ft in self._types:
            row: dict[str, float] = {}
            for rs in self._strategies:
                arm = self._table[ft][rs]
                row[getattr(rs, "value", rs)] = arm.successes / arm.attempts if arm.attempts > 0 else float("nan")
            data[getattr(ft, "value", ft)] = row
        return pd.DataFrame(data).T

    def get_attempts_table(self) -> pd.DataFrame:
        """Return the per-cell attempt counts as a DataFrame."""
        data: dict[str, dict[str, int]] = {}
        for ft in self._types:
            row: dict[str, int] = {}
            for rs in self._strategies:
                row[getattr(rs, "value", rs)] = self._table[ft][rs].attempts
            data[getattr(ft, "value", ft)] = row
        return pd.DataFrame(data).T

    def get_best_strategies(self) -> dict[FailureType, RecoveryStrategy]:
        """Return the current best strategy for each failure type based on
        highest average reward (ties broken arbitrarily)."""
        best: dict = {}
        for ft in self._types:
            best_rs: Optional[RecoveryStrategy] = None
            best_r = -1.0
            for rs in self._strategies:
                arm = self._table[ft][rs]
                if arm.avg_reward > best_r:
                    best_r = arm.avg_reward
                    best_rs = rs
            best[ft] = best_rs
        return best

    def _record_memory(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool,
        cost: float,
        episode_number: int,
    ) -> None:
        if self._memory is not None:
            self._memory.record(
                failure_type=failure_type.value,
                strategy=strategy.value,
                success=success,
                cost=cost,
                episode_number=episode_number,
            )


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


# ======================================================================
# Concrete implementations
# ======================================================================


class UCB1Selector(BanditSelector):
    """Upper Confidence Bound (UCB1) bandit."""

    def __init__(
        self,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        super().__init__(seed=seed, memory=memory, types=types, strategies=strategies)

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        best_strategy: Optional[RecoveryStrategy] = None
        best_score = -1.0
        total = self._total_attempts[failure_type]

        for strategy in self._strategies:
            arm = self._table[failure_type][strategy]
            if arm.attempts == 0:
                return strategy
            avg = arm.avg_reward
            exploration = math.sqrt(2.0 * math.log(total + 1) / arm.attempts)
            score = avg + exploration
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
        arm = self._table[failure_type][strategy]
        reward = 1.0 if success else 0.0
        arm.attempts += 1
        arm.successes += int(success)
        arm.total_reward += reward
        self._total_attempts[failure_type] += 1

        ep = episode_number or (context or {}).get("episode_number", 0)
        self._record_memory(failure_type, strategy, success, cost, ep)


class EpsilonGreedySelector(BanditSelector):
    """Epsilon-greedy bandit: explore randomly with probability epsilon."""

    def __init__(
        self,
        epsilon: float = 0.15,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        super().__init__(seed=seed, memory=memory, types=types, strategies=strategies)
        self.epsilon = epsilon

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self._strategies)
        best_strategy: Optional[RecoveryStrategy] = None
        best_reward = -1.0
        for strategy in self._strategies:
            arm = self._table[failure_type][strategy]
            if arm.avg_reward > best_reward:
                best_reward = arm.avg_reward
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
        arm = self._table[failure_type][strategy]
        reward = 1.0 if success else 0.0
        arm.attempts += 1
        arm.successes += int(success)
        arm.total_reward += reward
        self._total_attempts[failure_type] += 1

        ep = episode_number or (context or {}).get("episode_number", 0)
        self._record_memory(failure_type, strategy, success, cost, ep)


class DecayingEpsilonGreedySelector(BanditSelector):
    """Epsilon-greedy bandit with a decaying exploration rate.

    Unlike the fixed-epsilon variant, epsilon decays over time so the
    algorithm gradually shifts from exploration to exploitation:

        epsilon_effective = epsilon0 / (1 + episode / decay_rate)

    With the defaults (epsilon0=0.15, decay_rate=100), exploration drops
    to ~7.5 % by episode 100 and ~3 % by episode 400.
    """

    def __init__(
        self,
        epsilon0: float = 0.15,
        decay_rate: float = 100.0,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        super().__init__(seed=seed, memory=memory, types=types, strategies=strategies)
        self.epsilon0 = epsilon0
        self.decay_rate = decay_rate
        self._episode: int = 0  # auto-incremented on update()

    @property
    def epsilon_effective(self) -> float:
        """Current (decayed) epsilon value."""
        return self.epsilon0 / (1.0 + self._episode / self.decay_rate)

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        if self._rng.random() < self.epsilon_effective:
            return self._rng.choice(self._strategies)
        best_strategy: Optional[RecoveryStrategy] = None
        best_reward = -1.0
        for strategy in self._strategies:
            arm = self._table[failure_type][strategy]
            if arm.avg_reward > best_reward:
                best_reward = arm.avg_reward
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
        arm = self._table[failure_type][strategy]
        reward = 1.0 if success else 0.0
        arm.attempts += 1
        arm.successes += int(success)
        arm.total_reward += reward
        self._total_attempts[failure_type] += 1
        self._episode += 1

        ep = episode_number or (context or {}).get("episode_number", 0)
        self._record_memory(failure_type, strategy, success, cost, ep)


class ThompsonSamplingSelector(BanditSelector):
    """Thompson Sampling bandit: sample from Beta posteriors."""

    def __init__(
        self,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        super().__init__(seed=seed, memory=memory, types=types, strategies=strategies)
        # Beta parameters: alpha = 1 + successes, beta = 1 + failures
        self._beta: dict = {
            ft: {rs: {"alpha": 1, "beta": 1} for rs in self._strategies}
            for ft in self._types
        }

    def select_strategy(self, failure_type: FailureType, context: Optional[dict] = None) -> RecoveryStrategy:
        samples: dict[RecoveryStrategy, float] = {}
        for strategy in self._strategies:
            params = self._beta[failure_type][strategy]
            samples[strategy] = self._rng.betavariate(params["alpha"], params["beta"])
        return max(samples, key=samples.get)

    def get_confidences(self, failure_type: FailureType) -> dict[RecoveryStrategy, float]:
        """Posterior mean (alpha/(alpha+beta)) per strategy for a failure type —
        what a UI should display as "confidence"/"belief" (not the same as the
        stochastic sample used internally by ``select_strategy`` for exploration)."""
        out: dict[RecoveryStrategy, float] = {}
        for strategy in self._strategies:
            params = self._beta[failure_type][strategy]
            out[strategy] = params["alpha"] / (params["alpha"] + params["beta"])
        return out

    def update(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool,
        cost: float = 1.0,
        context: Optional[dict] = None,
        episode_number: int = 0,
    ) -> None:
        arm = self._table[failure_type][strategy]
        reward = 1.0 if success else 0.0
        arm.attempts += 1
        arm.successes += int(success)
        arm.total_reward += reward
        self._total_attempts[failure_type] += 1

        if success:
            self._beta[failure_type][strategy]["alpha"] += 1
        else:
            self._beta[failure_type][strategy]["beta"] += 1

        ep = episode_number or (context or {}).get("episode_number", 0)
        self._record_memory(failure_type, strategy, success, cost, ep)


class DiscountedThompsonSamplingSelector(ThompsonSamplingSelector):
    """Thompson Sampling with exponential discounting on the Beta posterior.

    Before incorporating each new observation, every arm's ``(alpha,
    beta)`` for the affected failure type is decayed toward the uniform
    prior by ``gamma``: ``alpha' = 1 + gamma*(alpha-1)`` (similarly for
    ``beta``). ``gamma=1.0`` recovers standard Thompson Sampling exactly.
    Decaying *every* arm (not just the one just played) lets stale
    evidence fade so the selector can re-adapt after the reward
    distribution shifts — e.g. a tool/API degradation mid-run.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        seed: Optional[int] = None,
        memory: Optional["OutcomeMemory"] = None,
        types: Optional[list] = None,
        strategies: Optional[list] = None,
    ) -> None:
        super().__init__(seed=seed, memory=memory, types=types, strategies=strategies)
        self.gamma = gamma

    def update(
        self,
        failure_type: FailureType,
        strategy: RecoveryStrategy,
        success: bool,
        cost: float = 1.0,
        context: Optional[dict] = None,
        episode_number: int = 0,
    ) -> None:
        for params in self._beta[failure_type].values():
            params["alpha"] = 1 + self.gamma * (params["alpha"] - 1)
            params["beta"] = 1 + self.gamma * (params["beta"] - 1)

        arm = self._table[failure_type][strategy]
        reward = 1.0 if success else 0.0
        arm.attempts += 1
        arm.successes += int(success)
        arm.total_reward += reward
        self._total_attempts[failure_type] += 1

        if success:
            self._beta[failure_type][strategy]["alpha"] += 1
        else:
            self._beta[failure_type][strategy]["beta"] += 1

        ep = episode_number or (context or {}).get("episode_number", 0)
        self._record_memory(failure_type, strategy, success, cost, ep)


# ======================================================================
# Factory
# ======================================================================


class SelectorFactory:
    """Factory to create the right BanditSelector subclass by algorithm name.

    Usage::

        selector = SelectorFactory.create("thompson_sampling", seed=42)
        selector = SelectorFactory.create("ucb1", memory=my_memory)
    """

    _REGISTRY: dict[str, type] = {
        "ucb1": UCB1Selector,
        "epsilon_greedy": EpsilonGreedySelector,
        "decaying_epsilon_greedy": DecayingEpsilonGreedySelector,
        "thompson_sampling": ThompsonSamplingSelector,
        "discounted_thompson_sampling": DiscountedThompsonSamplingSelector,
    }

    @staticmethod
    def create(algorithm: str, **kwargs) -> BanditSelector:
        """Return a ``BanditSelector`` instance for the given algorithm name.

        Args:
            algorithm: One of ``"ucb1"``, ``"epsilon_greedy"``, or
                ``"thompson_sampling"``.
            **kwargs: Forwarded to the subclass constructor (e.g. ``seed``,
                ``epsilon``, ``memory``).

        Returns:
            A concrete ``BanditSelector`` subclass instance.

        Raises:
            ValueError: If *algorithm* is unknown.
        """
        cls = SelectorFactory._REGISTRY.get(algorithm)
        if cls is None:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'; "
                f"use one of {list(SelectorFactory._REGISTRY.keys())}."
            )
        return cls(**kwargs)

    @staticmethod
    def register(algorithm: str, cls: type) -> None:
        """Register an additional selector class (e.g. for custom subclasses)."""
        SelectorFactory._REGISTRY[algorithm] = cls


# Backward-compatible alias for code that still uses the old name.
# New code should use SelectorFactory.create() and BanditSelector.
AdaptiveSelector = ThompsonSamplingSelector
