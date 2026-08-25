"""Structural protocols for the shared simulation kernel.

This module defines the four interfaces every experiment is built from:
``Context`` (the immutable per-episode observation handed to a policy),
and the ``Policy``, ``Environment``, and ``Classifier`` protocols.

These are ``typing.Protocol`` classes, not ABCs: concrete implementations
(in ``recovery/policies/``, ``recovery/envs/``, ``recovery/classify/``)
satisfy them structurally (duck typing) and do not need to inherit from
them. This module intentionally contains no concrete implementations and
no imports beyond ``dataclasses``, ``typing``, and ``numpy`` -- see the
architecture constraint in docs/architecture.md.

The ``Environment.sample_reward`` signature is the load-bearing design
decision in this module: it accepts only ``true_type``, never a predicted
type. There is no parameter through which a classifier's output could
reach the reward function -- see docs/architecture.md, "Preventing
Reward-Path Errors by Design".
"""

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np

__all__ = ["Context", "Policy", "Environment", "Classifier"]


class Context(NamedTuple):
    """Immutable per-episode observation handed to a policy.

    Deliberately carries only the PREDICTED type -- never the true type.
    A policy that wanted to cheat by reading the true type would have no
    field to read it from.

    A ``NamedTuple`` rather than a ``@dataclass(frozen=True)`` (its
    pre-Task-6 implementation): both are immutable and support the same
    keyword construction and attribute access used everywhere this class
    is read (``ctx.predicted_type`` etc.), but ``NamedTuple`` avoids a
    dataclass instance's per-object ``__dict__`` and its heavier
    ``__init__``, which matters here since one ``Context`` is allocated
    per episode in the kernel's hot loop (see docs/architecture.md and
    docs/experiments/horizon.md, "Kernel throughput"). Attempting to
    mutate an instance still raises (``AttributeError`` instead of
    ``dataclasses.FrozenInstanceError`` -- both are ``Exception``
    subclasses; no caller inspects the specific exception type).

    Parameters
    ----------
    predicted_type : int
        The (possibly noisy) type index a classifier believes occurred.
    tool_id : int, default=0
        Incidental context: which tool was in use. Contextual policies
        (e.g. LinUCB-style adapters) may use this as a feature.
    retry_count : int, default=0
        Incidental context: how many retries have occurred so far.
    episode : int, default=0
        The episode index, 0-based.
    drift_signal : bool, default=False
        Whether the environment announced a drift/shift event as of this
        episode. Always ``False`` for stationary environments and for any
        ``Environment`` that does not expose a ``drift_signal`` attribute
        (see ``recovery.core.episode.run_episodes``, which populates this
        field via ``getattr(env, "drift_signal", False)`` -- a policy
        never needs to know whether the underlying environment supports
        drift announcements at all). Set by
        ``recovery.envs.drift.AnnouncedDrift``; every other environment
        in this codebase leaves it at the default. Purely an incidental
        observation, like ``tool_id``/``retry_count`` -- nothing in the
        kernel requires a policy to read it.
    """

    predicted_type: int
    tool_id: int = 0
    retry_count: int = 0
    episode: int = 0
    drift_signal: bool = False


@runtime_checkable
class Policy(Protocol):
    """A strategy-selection policy.

    Implementations must never mutate the ``Context`` passed to ``select``
    or ``update`` (it is a frozen dataclass, so attempting to would raise
    ``dataclasses.FrozenInstanceError`` -- this is enforced by Python, not
    just convention).
    """

    name: str

    def select(self, ctx: Context, rng: np.random.Generator) -> int:
        """Choose an action (strategy index) given the observed context.

        Parameters
        ----------
        ctx : Context
            The current episode's (predicted-type-only) observation.
        rng : numpy.random.Generator
            The policy's dedicated RNG stream (``SeedStreams.policy``).

        Returns
        -------
        int
            The chosen action index, in ``[0, n_strategies)``.
        """
        ...

    def update(self, ctx: Context, action: int, reward: int) -> None:
        """Incorporate the outcome of one episode into the policy's state.

        Parameters
        ----------
        ctx : Context
            The context that was passed to the ``select`` call this
            outcome corresponds to.
        action : int
            The action that was played.
        reward : int
            The observed reward (0 or 1) for that action.
        """
        ...

    def reset(self) -> None:
        """Restore the policy to its initial (untrained) state."""
        ...


@runtime_checkable
class Environment(Protocol):
    """The reward-generating process. Never receives a predicted type.

    Implementations hold (or generate) an (N, K) success-probability
    structure but expose it only through ``success_matrix()``, which is
    documented as being for analytic cross-checks only -- ``sample_reward``
    itself must not be derivable from anything but ``true_type``,
    ``action``, and its own internal state.
    """

    n_types: int
    n_strategies: int

    def success_matrix(self) -> np.ndarray:
        """Return the (N, K) ground-truth success-probability matrix.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_types, n_strategies)``, values in [0, 1]. For
            analytic cross-checks (e.g. against
            ``recovery.theory.analytic``) only -- no policy or classifier
            may read this at simulation time.
        """
        ...

    def sample_reward(self, true_type: int, action: int, rng: np.random.Generator) -> int:
        """Sample a 0/1 reward for playing ``action`` under ``true_type``.

        Parameters
        ----------
        true_type : int
            The TRUE type index for this episode. Never a prediction.
        action : int
            The action index the policy chose.
        rng : numpy.random.Generator
            The environment's dedicated RNG stream (``SeedStreams.reward``).

        Returns
        -------
        int
            0 or 1.
        """
        ...

    def step(self, episode: int) -> None:
        """Advance any internal drift state. No-op for stationary environments.

        Parameters
        ----------
        episode : int
            The episode index that just completed.
        """
        ...


@runtime_checkable
class Classifier(Protocol):
    """Maps a true type to a (possibly noisy) predicted type.

    Its output feeds ``Context.predicted_type`` -- and, structurally,
    nothing else. There is no path from a ``Classifier`` back into
    ``Environment.sample_reward``.
    """

    def predict(self, true_type: int, rng: np.random.Generator) -> int:
        """Predict a type index given the true type.

        Parameters
        ----------
        true_type : int
            The true type index for this episode.
        rng : numpy.random.Generator
            The classifier's dedicated RNG stream (``SeedStreams.classifier``).
            Implementations should consume a consistent, non-zero number of
            draws from this stream per call (even deterministic
            classifiers) so the classifier stream's own draw-count
            trajectory does not depend on which classifier is plugged in.
            Because each ``SeedStreams`` generator is an independent
            ``numpy.random.Generator`` with its own internal state, this
            does NOT affect the ``types``/``reward``/``policy`` streams
            either way -- it only keeps the classifier stream itself
            comparable/swappable across implementations. See
            ``recovery.classify.synthetic.PerfectClassifier``.

        Returns
        -------
        int
            The predicted type index, in ``[0, n_types)``.
        """
        ...
