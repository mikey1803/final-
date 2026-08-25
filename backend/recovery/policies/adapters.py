"""Policy adapters wrapping every existing selector, plus two native policies.

No selector's internal algorithm is rewritten here -- every wrapped class
(``src.recovery.selector.*``, ``src.recovery.contextual_selector.*``,
``src.evaluation.baselines.*``) is used exactly as it already exists. This
module only adapts calling convention: translating between the kernel's
``Context``/int-index world and each selector's own
``FailureType``/``RecoveryStrategy`` enum-based ``select_strategy``/
``update`` API.

RNG-stream limitation (read before assuming Common Random Numbers)
--------------------------------------------------------------------
``Policy.select(ctx, rng)`` receives ``streams.policy`` (see
``recovery.core.rng``), and native policies in this module
(``StaticRulePolicy``, ``OraclePolicy``) use it correctly. However, every
*legacy-wrapped* selector below (UCB1, epsilon-greedy, decaying
epsilon-greedy, Thompson, Discounted Thompson, LinUCB, Contextual
Cost-Aware, Uniform Random, One-Shot LLM's keyword fallback) was written
before this kernel existed and owns its OWN internal ``random.Random``
instance, seeded once at construction (``seed=`` kwarg). The kernel's
``rng`` argument is therefore accepted (to satisfy the ``Policy`` protocol
signature) but **ignored** by these adapters -- rewriting the wrapped
selectors' internals to accept an externally-supplied ``numpy.random.Generator``
was explicitly out of scope for this task (see "wrap, don't rewrite").

Practically: Common Random Numbers *for the environment and classifier*
(the ``types``/``classifier``/``reward`` streams) still hold across any
comparison that swaps only the policy, because policies never touch those
streams. Common Random Numbers *for a policy's own exploration noise*
across two different legacy-wrapped policies requires giving them matching
``seed=`` constructor arguments, not relying on ``streams.policy`` -- that
is a real limitation of wrapping pre-existing ``random.Random``-based code,
documented here rather than silently assumed away. See
docs/architecture.md.
"""

from typing import Callable, Optional

import numpy as np

from src.evaluation.baselines import (
    FixedRuleBaseline,
    LinUCBBaseline,
    NaiveRetryBaseline,
    OneShotLLMBaseline,
    RandomBaseline,
)
from src.failures.taxonomy import FailureType
from src.recovery.contextual_selector import ContextualCostAwareSelector
from src.recovery.selector import SelectorFactory
from src.recovery.strategies import RecoveryStrategy, get_action_cost

from recovery.core.protocols import Context

__all__ = [
    "LegacySelectorAdapter",
    "UCB1Policy",
    "EpsilonGreedyPolicy",
    "DecayingEpsilonGreedyPolicy",
    "ThompsonPolicy",
    "DiscountedThompsonPolicy",
    "LinUCBPolicy",
    "ContextualCostAwarePolicy",
    "NaiveRetryPolicy",
    "UniformRandomPolicy",
    "FixedRulePolicy",
    "OneShotLLMPolicy",
    "StaticRulePolicy",
    "OraclePolicy",
]

_TYPES: list[FailureType] = list(FailureType)
_STRATS: list[RecoveryStrategy] = list(RecoveryStrategy)
_STRAT_TO_IDX: dict[RecoveryStrategy, int] = {s: i for i, s in enumerate(_STRATS)}

_TOOL_NAMES: list[str] = [
    "web_search", "calculator", "file_reader", "api_caller",
    "database_query", "code_executor",
]


def _tool_name(tool_id: int) -> str:
    if 0 <= tool_id < len(_TOOL_NAMES):
        return _TOOL_NAMES[tool_id]
    return "unknown"


def _legacy_context(ctx: Context) -> dict:
    """Build the ``context: dict`` legacy selectors expect from a ``Context``.

    The kernel's ``Context`` is deliberately minimal (see
    ``recovery.core.protocols``); it has no ``time_since_last_failure``
    field the way the pre-kernel harness's ad hoc context dict did. Legacy
    selectors that read that key (``LinUCBBaseline``,
    ``ContextualCostAwareSelector``) fall back to their own default (0) via
    ``dict.get(..., 0)``, so this is a documented behavioral simplification,
    not a bug.
    """
    return {
        "retry_count": ctx.retry_count,
        "tool_name": _tool_name(ctx.tool_id),
        "episode_number": ctx.episode,
    }


class LegacySelectorAdapter:
    """Generic wrapper adapting a legacy ``BaseSelector``/``BanditSelector``
    instance to the ``Policy`` protocol.

    Parameters
    ----------
    name : str
        The policy's display name.
    factory : callable
        Zero-argument callable that constructs a fresh instance of the
        wrapped selector. Called once at construction and again on every
        ``reset()`` -- legacy selectors expose no native ``reset()`` that
        restores Beta posteriors / arm tables / internal RNG state, so
        this adapter reconstructs the wrapped selector from scratch
        instead (using the same constructor arguments, captured by the
        closure), which is the only way to fully reset it without
        reaching into (and thus effectively rewriting) its internals.
    updates : bool, default=True
        Whether the wrapped selector defines ``update()``. Several
        baselines (Naive Retry, Uniform Random, Fixed Rule, One-Shot LLM)
        are stateless / non-learning by design and have no ``update()``
        method at all -- set ``updates=False`` for those so
        ``LegacySelectorAdapter.update()`` becomes a documented no-op
        instead of raising ``AttributeError``.
    """

    def __init__(self, name: str, factory: Callable[[], object], updates: bool = True) -> None:
        self.name = name
        self._factory = factory
        self._updates = updates
        self._selector = factory()

    def select(self, ctx: Context, rng: np.random.Generator) -> int:
        """Select an action. See module docstring for the ``rng``-ignoring caveat."""
        del rng  # legacy selector owns its own RNG; see module docstring
        failure_type = _TYPES[ctx.predicted_type]
        strategy = self._selector.select_strategy(failure_type, context=_legacy_context(ctx))
        return _STRAT_TO_IDX[strategy]

    def update(self, ctx: Context, action: int, reward: int) -> None:
        """Forward the outcome to the wrapped selector's ``update()``, if any."""
        if not self._updates:
            return
        failure_type = _TYPES[ctx.predicted_type]
        strategy = _STRATS[action]
        self._selector.update(
            failure_type,
            strategy,
            success=bool(reward),
            cost=get_action_cost(strategy),
            context=_legacy_context(ctx),
            episode_number=ctx.episode,
        )

    def reset(self) -> None:
        """Reconstruct the wrapped selector from scratch (see class docstring)."""
        self._selector = self._factory()


# ======================================================================
# Adapters for src.recovery.selector (BanditSelector family)
# ======================================================================


def UCB1Policy(seed: Optional[int] = None, name: str = "UCB1") -> LegacySelectorAdapter:
    """Adapts ``src.recovery.selector.UCB1Selector``."""
    return LegacySelectorAdapter(
        name, lambda: SelectorFactory.create("ucb1", seed=seed), updates=True
    )


def EpsilonGreedyPolicy(
    epsilon: float = 0.15, seed: Optional[int] = None, name: str = "EpsilonGreedy"
) -> LegacySelectorAdapter:
    """Adapts ``src.recovery.selector.EpsilonGreedySelector``."""
    return LegacySelectorAdapter(
        name,
        lambda: SelectorFactory.create("epsilon_greedy", epsilon=epsilon, seed=seed),
        updates=True,
    )


def DecayingEpsilonGreedyPolicy(
    epsilon0: float = 0.15,
    decay_rate: float = 100.0,
    seed: Optional[int] = None,
    name: str = "DecayingEpsilonGreedy",
) -> LegacySelectorAdapter:
    """Adapts ``src.recovery.selector.DecayingEpsilonGreedySelector``."""
    return LegacySelectorAdapter(
        name,
        lambda: SelectorFactory.create(
            "decaying_epsilon_greedy", epsilon0=epsilon0, decay_rate=decay_rate, seed=seed
        ),
        updates=True,
    )


def ThompsonPolicy(seed: Optional[int] = None, name: str = "Thompson") -> LegacySelectorAdapter:
    """Adapts ``src.recovery.selector.ThompsonSamplingSelector``."""
    return LegacySelectorAdapter(
        name, lambda: SelectorFactory.create("thompson_sampling", seed=seed), updates=True
    )


def DiscountedThompsonPolicy(
    gamma: float = 0.99, seed: Optional[int] = None, name: str = "DiscountedThompson"
) -> LegacySelectorAdapter:
    """Adapts ``src.recovery.selector.DiscountedThompsonSamplingSelector``."""
    return LegacySelectorAdapter(
        name,
        lambda: SelectorFactory.create("discounted_thompson_sampling", gamma=gamma, seed=seed),
        updates=True,
    )


# ======================================================================
# Adapters for src.recovery.contextual_selector / src.evaluation.baselines
# ======================================================================


def LinUCBPolicy(
    alpha: float = 1.0, seed: Optional[int] = None, name: str = "LinUCB"
) -> LegacySelectorAdapter:
    """Adapts ``src.evaluation.baselines.LinUCBBaseline``."""
    return LegacySelectorAdapter(
        name, lambda: LinUCBBaseline(alpha=alpha, seed=seed), updates=True
    )


def ContextualCostAwarePolicy(
    lambda_cost: float = 0.3,
    alpha: float = 0.25,
    seed: Optional[int] = None,
    name: str = "ContextualCostAware",
) -> LegacySelectorAdapter:
    """Adapts ``src.recovery.contextual_selector.ContextualCostAwareSelector``."""
    return LegacySelectorAdapter(
        name,
        lambda: ContextualCostAwareSelector(lambda_cost=lambda_cost, alpha=alpha, seed=seed),
        updates=True,
    )


def NaiveRetryPolicy(name: str = "NaiveRetry") -> LegacySelectorAdapter:
    """Adapts ``src.evaluation.baselines.NaiveRetryBaseline`` (stateless, no update)."""
    return LegacySelectorAdapter(name, lambda: NaiveRetryBaseline(), updates=False)


def UniformRandomPolicy(
    seed: Optional[int] = None, name: str = "UniformRandom"
) -> LegacySelectorAdapter:
    """Adapts ``src.evaluation.baselines.RandomBaseline`` (stateless, no update)."""
    return LegacySelectorAdapter(name, lambda: RandomBaseline(seed=seed), updates=False)


def FixedRulePolicy(name: str = "FixedRule") -> LegacySelectorAdapter:
    """Adapts the EXISTING ``src.evaluation.baselines.FixedRuleBaseline``
    (the blind v2 rule, ``_FIXED_RULES_V2_BLIND``) -- i.e. this wraps the
    legacy ``FailureType``/``RecoveryStrategy``-keyed rule table as-is.

    For the array-based blind rule used with the analytic reference layer
    (``recovery.envs.canonical.BLIND_RULE``), use ``StaticRulePolicy``
    instead -- the two happen to encode the same mapping, but
    ``StaticRulePolicy`` takes it as an explicit ``numpy`` array rather
    than going through this legacy class.
    """
    return LegacySelectorAdapter(name, lambda: FixedRuleBaseline(), updates=False)


def OneShotLLMPolicy(
    temperature: float = 1.0, seed: Optional[int] = None, name: str = "OneShotLLM"
) -> LegacySelectorAdapter:
    """Adapts ``src.evaluation.baselines.OneShotLLMBaseline`` (no learning by design)."""
    return LegacySelectorAdapter(
        name, lambda: OneShotLLMBaseline(temperature=temperature, seed=seed), updates=False
    )


# ======================================================================
# Native policies (not wrapping legacy code)
# ======================================================================


class StaticRulePolicy:
    """Plays a fixed ``rule[predicted_type] -> action`` mapping.

    Native (not a legacy adapter) so it can be driven directly by an
    array like ``recovery.envs.canonical.BLIND_RULE``, making it directly
    comparable to ``recovery.theory.analytic.policy_rate`` without going
    through ``FailureType``/``RecoveryStrategy`` enum plumbing.

    Parameters
    ----------
    rule : numpy.ndarray, shape (n_types,)
        ``rule[t]`` is the action index played when the predicted type is
        ``t``.
    name : str, default="StaticRule"
        The policy's display name.
    """

    def __init__(self, rule: np.ndarray, name: str = "StaticRule") -> None:
        self.name = name
        self._rule = np.asarray(rule)

    def select(self, ctx: Context, rng: np.random.Generator) -> int:
        del rng  # deterministic; no randomness needed
        return int(self._rule[ctx.predicted_type])

    def update(self, ctx: Context, action: int, reward: int) -> None:
        return None  # stateless: this policy never learns

    def reset(self) -> None:
        return None  # stateless: nothing to reset


class OraclePolicy:
    """Always plays ``argmax_k matrix[predicted_type, k]``.

    This exists ONLY as an evaluation normalizer (the achievable ceiling,
    ``recovery.theory.analytic.oracle_rate``), not as a method a real
    system could deploy: it is constructed with privileged access to the
    full ground-truth matrix, which no real policy would have. It still
    only *reads* ``ctx.predicted_type`` at call time -- never a true
    type -- so it does not violate the environment/classifier separation;
    it is only meaningful as the oracle ceiling when run together with a
    ``PerfectClassifier`` (predicted_type == true_type). Run with any
    noisy classifier, it silently degrades to "argmax of the WRONG row" --
    a legitimate simulation, just not the oracle rate.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_types, n_strategies)
        The ground-truth success matrix (e.g.
        ``recovery.envs.canonical.MATRIX``).
    name : str, default="Oracle"
        The policy's display name.
    """

    def __init__(self, matrix: np.ndarray, name: str = "Oracle") -> None:
        self.name = name
        matrix = np.asarray(matrix, dtype=np.float64)
        self._rule = matrix.argmax(axis=1)

    def select(self, ctx: Context, rng: np.random.Generator) -> int:
        del rng
        return int(self._rule[ctx.predicted_type])

    def update(self, ctx: Context, action: int, reward: int) -> None:
        return None

    def reset(self) -> None:
        return None
