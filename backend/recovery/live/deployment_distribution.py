"""Estimate the deployment distribution p over failure types from live-pilot
data -- and ONLY from ``targeted=False`` (natural/untargeted) tasks.

Two distributions must stay structurally separate:

- The confusion-matrix DESIGN target: deliberately balanced across types.
  ``targeted=True`` tasks exist specifically to force enough samples of
  rare agent-side types to reach the n=30-45 floor -- see
  ``recovery.live.pilot_report``.
- The DEPLOYMENT distribution p: what ``recovery.theory.analytic``'s ``p``
  parameter needs for the crossover analysis -- what an agent actually
  encounters when task selection isn't steered toward a target type.

Feeding the balanced/targeted counts into p would bias the crossover
estimate toward whatever mix of types was easiest to elicit by design,
not what happens in practice. This module is the one sanctioned way to
compute p from live-pilot data; it structurally cannot include a targeted
task's events, by construction, not by caller discipline.
"""

from __future__ import annotations

import numpy as np

from recovery.envs.canonical import TYPE_NAMES
from recovery.live.event_schema import Task
from recovery.live.event_store import EventStore

__all__ = ["DeploymentDistributionError", "estimate_deployment_distribution"]


class DeploymentDistributionError(ValueError):
    """Raised when there's nothing legitimate to estimate p from."""


def estimate_deployment_distribution(tasks: list[Task], store: EventStore) -> np.ndarray:
    """p[i] = fraction of labelled events from untargeted tasks whose
    true_type == TYPE_NAMES[i] (same order recovery.envs.canonical.MATRIX
    uses). Raises DeploymentDistributionError rather than returning a
    fabricated uniform fallback if there is nothing untargeted to measure
    from -- an unmeasured p reported as measured is worse than an error.
    """
    natural_task_ids = {t.task_id for t in tasks if not t.targeted}
    if not natural_task_ids:
        raise DeploymentDistributionError(
            "no targeted=False tasks in the given task list -- p must be estimated "
            "from natural/untargeted tasks only, never from the balanced targeted set"
        )

    counts = np.zeros(len(TYPE_NAMES), dtype=np.float64)
    n = 0
    for event in store.iter_events():
        if event.true_type is None or event.task_id not in natural_task_ids:
            continue
        counts[TYPE_NAMES.index(event.true_type)] += 1
        n += 1

    if n == 0:
        raise DeploymentDistributionError(
            "no labelled events from untargeted tasks yet -- cannot estimate p "
            "from zero observations"
        )
    return counts / n
