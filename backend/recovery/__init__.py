"""Independent analytic reference layer for the recovery-strategy environment.

This package is deliberately separate from ``src.recovery`` (the simulator's
selector/strategy implementation). It exists to provide a closed-form,
simulation-independent way to compute the same quantities the simulator
estimates by sampling, so the two can be cross-checked against each other.
"""
