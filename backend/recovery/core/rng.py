"""RNG stream construction for the shared simulation kernel.

A single integer seed is expanded into four *independent* streams via
``numpy.random.SeedSequence.spawn`` -- one each for type sampling,
classification, reward sampling, and policy decisions/exploration.

Why independent streams give Common Random Numbers (CRN)
----------------------------------------------------------
Each stream is its own ``numpy.random.Generator`` with fully independent
internal state (``SeedSequence.spawn`` derives four statistically
independent child seeds from one parent seed -- not four different
starting offsets into the same stream). Concretely this buys two
properties experiments in this repository rely on:

1. **Cross-policy CRN.** At a fixed seed, ``streams.types`` produces the
   exact same sequence of true failure types regardless of which
   ``Policy`` is plugged in, because the policy only ever touches
   ``streams.policy``, never ``streams.types``. Two policies compared at
   the same seed therefore face an *identical* sequence of environment
   conditions -- differences in measured outcome are attributable to the
   policy, not to variance in which failures happened to occur. This is
   the classic simulation variance-reduction technique of Common Random
   Numbers.
2. **Cross-classifier CRN.** Symmetrically, swapping ``Classifier``
   implementations (``streams.classifier``) does not change the sequence
   of true types (``streams.types``) or the sequence of reward draws
   (``streams.reward``) a policy experiences for a given action -- only
   the *observed* label changes. This is what makes
   ``tests/test_episode.py``'s stream-independence test possible, and
   why config comparisons in ``experiments/`` (see Task 1) are meaningful
   rather than being confounded by "got a different random environment."

Because a policy's own exploration randomness lives in ``streams.policy``,
separate from ``streams.reward`` and ``streams.types``, two *different*
policies run at the same seed also do not have their exploration draws
interfere with which types/rewards occur -- each stream advances only
when its own protocol method is called.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["SeedStreams", "from_seed"]

_N_STREAMS = 4


@dataclass
class SeedStreams:
    """Four independent RNG streams derived from one integer seed.

    Attributes
    ----------
    types : numpy.random.Generator
        Drives sampling of the true failure type each episode.
    classifier : numpy.random.Generator
        Drives the ``Classifier``'s prediction (if stochastic).
    reward : numpy.random.Generator
        Drives the ``Environment``'s reward sampling.
    policy : numpy.random.Generator
        Drives the ``Policy``'s action selection/exploration.
    """

    types: np.random.Generator
    classifier: np.random.Generator
    reward: np.random.Generator
    policy: np.random.Generator


def from_seed(seed: int) -> SeedStreams:
    """Build four independent RNG streams from a single integer seed.

    Parameters
    ----------
    seed : int
        The top-level experiment seed.

    Returns
    -------
    SeedStreams
        Four independent ``numpy.random.Generator`` instances, spawned via
        ``numpy.random.SeedSequence(seed).spawn(4)``.
    """
    parent = np.random.SeedSequence(seed)
    types_ss, classifier_ss, reward_ss, policy_ss = parent.spawn(_N_STREAMS)
    return SeedStreams(
        types=np.random.default_rng(types_ss),
        classifier=np.random.default_rng(classifier_ss),
        reward=np.random.default_rng(reward_ss),
        policy=np.random.default_rng(policy_ss),
    )
