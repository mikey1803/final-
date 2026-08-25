"""The one shared episode loop.

Every experiment in this repository should ultimately call
``run_episodes``. It is the single place the eight-step episode lifecycle
is implemented (see docs/architecture.md for the full walkthrough); no
other module in this task defines a competing loop.
"""

from typing import Optional

import numpy as np

from recovery.core.log import EpisodeLog
from recovery.core.protocols import Classifier, Context, Environment, Policy
from recovery.core.rng import from_seed

__all__ = ["run_episodes"]


def run_episodes(
    env: Environment,
    policy: Policy,
    classifier: Classifier,
    n_episodes: int,
    seed: int,
    type_prior: Optional[np.ndarray] = None,
) -> EpisodeLog:
    """Run ``n_episodes`` through the shared kernel loop.

    Loop body, executed in exactly this order, every episode:

    1. ``true_type = streams.types.choice(env.n_types, p=type_prior)``
    2. ``predicted = classifier.predict(true_type, streams.classifier)``
    3. ``ctx = Context(predicted_type=predicted, episode=i, drift_signal=getattr(env, "drift_signal", False))``
    4. ``action = policy.select(ctx, streams.policy)``
    5. ``reward = env.sample_reward(true_type, action, streams.reward)``
    6. ``policy.update(ctx, action, reward)``
    7. ``env.step(i)``
    8. ``log`` records ``true_type``, ``predicted``, ``action``, ``reward`` at row ``i``

    ``true_type`` is read directly from step 1 into step 5's
    ``sample_reward`` call -- it is never routed through ``Context`` or
    through the classifier, so there is no way for a predicted type to
    reach the reward function through this loop.

    Parameters
    ----------
    env : Environment
        The reward-generating process.
    policy : Policy
        The strategy-selection policy under test.
    classifier : Classifier
        Maps true types to (possibly noisy) predicted types.
    n_episodes : int
        Number of episodes to run (must be >= 1).
    seed : int
        Top-level seed; expanded into four independent streams via
        ``recovery.core.rng.from_seed``.
    type_prior : numpy.ndarray, shape (env.n_types,), optional
        Probability of each true type. Defaults to uniform (numpy's
        default for ``Generator.choice`` when ``p=None``).

    Returns
    -------
    EpisodeLog
        The recorded outcome of every episode.

    Raises
    ------
    ValueError
        If ``n_episodes < 1``; if ``type_prior`` has the wrong shape or
        does not sum to 1 (atol 1e-9); if a policy selects an action
        outside ``[0, env.n_strategies)``.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    if type_prior is not None:
        type_prior = np.asarray(type_prior, dtype=np.float64)
        if type_prior.shape != (env.n_types,):
            raise ValueError(
                f"type_prior must have shape ({env.n_types},), got shape {type_prior.shape}"
            )
        total = float(type_prior.sum())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"type_prior must sum to 1 (atol=1e-9), got sum={total!r}")

    streams = from_seed(seed)
    log = EpisodeLog(n_episodes)
    policy.reset()

    # Local variable binding for the hot loop (avoid a `.`-attribute
    # lookup on env/policy/classifier/streams/log every one of
    # n_episodes iterations -- see docs/experiments/horizon.md, "Kernel
    # throughput"). Purely a CPython dispatch-cost optimization: every
    # name below is bound to the exact same underlying object/method the
    # unabbreviated form would have resolved to, so this changes nothing
    # about which values are computed or drawn.
    types_rng = streams.types
    classifier_rng = streams.classifier
    reward_rng = streams.reward
    policy_rng = streams.policy
    predict = classifier.predict
    select = policy.select
    sample_reward = env.sample_reward
    update = policy.update
    step = env.step
    n_strategies = env.n_strategies
    log_true_type = log.true_type
    log_predicted_type = log.predicted_type
    log_action = log.action
    log_reward = log.reward

    n_types = env.n_types
    use_uniform_draw = type_prior is None
    type_indices = np.arange(n_types)  # only used on the non-uniform-prior path

    for i in range(n_episodes):
        if use_uniform_draw:
            # types_rng.integers(0, n_types) draws from the identical
            # distribution as types_rng.choice(n_types, p=None) -- verified
            # bit-for-bit identical for a fixed seed (both ultimately
            # dispatch to the same underlying integer-generation
            # primitive) -- and is markedly faster, since `choice` incurs
            # `np.prod`/shape-introspection overhead per call.
            true_type = int(types_rng.integers(0, n_types))
        else:
            true_type = int(types_rng.choice(type_indices, p=type_prior))
        predicted = predict(true_type, classifier_rng)
        ctx = Context(
            predicted_type=predicted, episode=i, drift_signal=getattr(env, "drift_signal", False)
        )
        action = select(ctx, policy_rng)
        if not (0 <= action < n_strategies):
            raise ValueError(
                f"action out of range: got {action}, expected [0, {n_strategies})"
            )
        reward = sample_reward(true_type, action, reward_rng)
        update(ctx, action, reward)
        step(i)
        log_true_type[i] = true_type
        log_predicted_type[i] = predicted
        log_action[i] = action
        log_reward[i] = reward

    return log
