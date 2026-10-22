"""Runtime invariant guards: scientific assertions over simulation results.

Depends only on ``numpy`` and ``recovery.theory.analytic`` -- never on the
simulation kernel, policies, environments, or classifiers (see
docs/architecture.md, "Guard layer"). Guards are pure, deterministic, and
side-effect free: they either return ``None`` or raise. They never print,
warn, or continue past a violated invariant.

That rule covers ``invariants.py``. ``runtime.py`` (Task 22) is a separate
sub-layer of runtime guardrails for the ``with_recovery`` wrapper: it acts
during an agent run, and imports only ``Budget`` / ``TOOL_CALL_RESERVE_MS``
from ``recovery.live.executors.base`` -- see docs/architecture.md.
"""
