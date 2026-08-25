"""Closed-form success-rate formulas for the recovery-strategy environment.

This module is a pure-numpy analytic reference layer. It MUST NOT import
``recovery.envs.canonical`` or any simulation module — every function here
operates on arbitrary ``(P, pi, C, p)`` arrays passed in by the caller, so it
can be used to validate a simulator without depending on it.

Notation
--------
P : ndarray, shape (N, K)
    ``P[t, k]`` = P(success | true type ``t``, strategy ``k``).
pi : ndarray, shape (N,)
    Strategy played when the (possibly noisy) *predicted* type is ``t'``:
    ``pi[t']`` is the strategy index chosen for predicted type ``t'``.
C : ndarray, shape (N, N)
    Row-stochastic confusion matrix: ``C[t, t']`` = P(predict ``t'`` | true ``t``).
p : ndarray, shape (N,)
    Type prior. Defaults to uniform (``1/N`` each) when omitted.

All argmax operations (``oracle_rate``, and implicitly anywhere a "best
strategy" is picked) use ``numpy``'s convention: ties are broken by taking
the *first* maximum along the axis.
"""

from typing import Optional

import numpy as np

__all__ = [
    "oracle_rate",
    "random_rate",
    "policy_rate",
    "policy_rate_under_confusion",
    "uniform_confusion",
    "normalize",
]


def _validate(
    P: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
) -> None:
    """Validate whichever of ``P``, ``pi``, ``C``, ``p`` are provided.

    Parameters
    ----------
    P : ndarray, optional
        Success-probability matrix to validate (2-D, values in [0, 1]).
    pi : ndarray, optional
        Policy vector to validate against ``P``'s shape, if ``P`` is given.
    C : ndarray, optional
        Confusion matrix to validate (square, row-stochastic).
    p : ndarray, optional
        Type prior to validate (correct shape, sums to 1).

    Raises
    ------
    ValueError
        If any provided argument fails its shape/value/stochasticity check.
        The message names the offending argument and its shape or value.
    """
    n_types: Optional[int] = None
    n_strats: Optional[int] = None

    if P is not None:
        if P.ndim != 2:
            raise ValueError(f"P must be 2-D, got shape {P.shape}")
        if np.any(P < 0.0) or np.any(P > 1.0):
            raise ValueError(
                f"P must contain only values in [0, 1], got min={P.min()!r}, max={P.max()!r}"
            )
        n_types, n_strats = P.shape

    if pi is not None:
        if n_types is not None and pi.shape != (n_types,):
            raise ValueError(f"pi must have shape ({n_types},), got shape {pi.shape}")
        if pi.ndim != 1:
            raise ValueError(f"pi must be 1-D, got shape {pi.shape}")
        if n_strats is not None and (np.any(pi < 0) or np.any(pi >= n_strats)):
            raise ValueError(
                f"pi contains an out-of-range strategy index; must be in "
                f"[0, {n_strats}), got min={int(pi.min())}, max={int(pi.max())}"
            )

    if C is not None:
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"C must be square 2-D, got shape {C.shape}")
        if n_types is not None and C.shape != (n_types, n_types):
            raise ValueError(f"C must have shape ({n_types}, {n_types}), got shape {C.shape}")
        row_sums = C.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"C must be row-stochastic (each row sums to 1), got row sums {row_sums}"
            )

    if p is not None:
        if n_types is not None and p.shape != (n_types,):
            raise ValueError(f"p must have shape ({n_types},), got shape {p.shape}")
        if p.ndim != 1:
            raise ValueError(f"p must be 1-D, got shape {p.shape}")
        total = float(p.sum())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"p must sum to 1 (atol=1e-9), got sum={total!r}")


def _default_prior(n_types: int) -> np.ndarray:
    return np.full(n_types, 1.0 / n_types, dtype=np.float64)


def oracle_rate(P: np.ndarray, p: Optional[np.ndarray] = None) -> float:
    """Expected success rate of the per-type-optimal policy.

    ``oracle_rate(P, p) = sum_t p[t] * max_k P[t, k]``.

    This is the hard ceiling for *any* policy under *any* confusion matrix
    (see ``policy_rate_under_confusion``'s docstring) — no way of mapping
    (possibly noisy) observations to actions can beat always knowing the
    true type and playing its best strategy.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.

    Returns
    -------
    float
        The oracle success rate.

    Raises
    ------
    ValueError
        If ``P`` or ``p`` fail validation (see ``_validate``).

    Notes
    -----
    Ties in ``max_k P[t, k]`` do not affect this function's value, but any
    caller that also needs *which* ``k`` achieves the max (e.g. via
    ``P.argmax(axis=1)``) should note numpy's argmax breaks ties by taking
    the first maximum along the axis.
    """
    _validate(P=P, p=p)
    prior = _default_prior(P.shape[0]) if p is None else p
    return float(prior @ P.max(axis=1))


def random_rate(P: np.ndarray, p: Optional[np.ndarray] = None) -> float:
    """Expected success rate of picking a strategy uniformly at random.

    ``random_rate(P, p) = sum_t p[t] * mean_k P[t, k]``.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.

    Returns
    -------
    float
        The uniform-random-strategy success rate (the normalization floor).

    Raises
    ------
    ValueError
        If ``P`` or ``p`` fail validation (see ``_validate``).
    """
    _validate(P=P, p=p)
    prior = _default_prior(P.shape[0]) if p is None else p
    return float(prior @ P.mean(axis=1))


def policy_rate(P: np.ndarray, pi: np.ndarray, p: Optional[np.ndarray] = None) -> float:
    """Expected success rate of a fixed policy under perfect type information.

    ``policy_rate(P, pi, p) = sum_t p[t] * P[t, pi[t]]``.

    This is a HARD CEILING for policy ``pi``: it is what ``pi`` achieves
    when it always observes the *true* type (no classifier/observation
    noise). Equivalently it equals
    ``policy_rate_under_confusion(P, pi, np.eye(N), p)`` — the identity
    confusion matrix. It is a hard ceiling in this noiseless sense, but it
    is *not* necessarily a ceiling once ``pi`` is driven by a *noisy*
    predicted type: see ``policy_rate_under_confusion``'s Notes for why
    confusion can occasionally reroute a suboptimal ``pi`` entry to a
    per-type-better action.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    pi : ndarray, shape (N,)
        Strategy index played for each true type.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.

    Returns
    -------
    float
        The noiseless policy success rate.

    Raises
    ------
    ValueError
        If ``P``, ``pi``, or ``p`` fail validation (see ``_validate``).
    """
    _validate(P=P, pi=pi, p=p)
    prior = _default_prior(P.shape[0]) if p is None else p
    return float(prior @ P[np.arange(P.shape[0]), pi])


def policy_rate_under_confusion(
    P: np.ndarray,
    pi: np.ndarray,
    C: np.ndarray,
    p: Optional[np.ndarray] = None,
) -> float:
    """Expected success rate of policy ``pi`` when acting on a noisy type.

    ``policy_rate_under_confusion(P, pi, C, p)
        = sum_t p[t] * sum_t' C[t, t'] * P[t, pi[t']]``.

    The ACTION index comes from the PREDICTED type ``t'`` (``pi[t']``); the
    REWARD row comes from the TRUE type ``t`` (``P[t, .]``). Implemented as
    ``R = P[:, pi]; return p @ (C * R).sum(axis=1)`` — do not reverse which
    axis indexes true vs. predicted type.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    pi : ndarray, shape (N,)
        Strategy index played for each *predicted* type.
    C : ndarray, shape (N, N)
        Row-stochastic confusion matrix, ``C[t, t'] = P(predict t' | true t)``.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.

    Returns
    -------
    float
        The confusion-aware policy success rate.

    Raises
    ------
    ValueError
        If ``P``, ``pi``, ``C``, or ``p`` fail validation (see ``_validate``).

    Notes
    -----
    This quantity is always bounded above by ``oracle_rate(P, p)``, for any
    row-stochastic ``C`` whatsoever: per true type ``t``,
    ``sum_t' C[t, t'] * P[t, pi[t']] <= max_k P[t, k]`` because every term
    averaged is itself at most the per-type max and the weights sum to 1.

    It is NOT, in general, bounded above by ``policy_rate(P, pi)`` (the
    noiseless rate for the same ``pi``). If ``pi`` is not the per-type
    argmax policy (i.e. it is suboptimal for some true type ``t``), a
    confusion matrix that happens to reroute type ``t`` to a *different*
    predicted type ``t'`` whose ``pi[t']`` is a better action for the true
    type ``t`` than ``pi[t]`` itself will *raise* that type's contribution
    above its noiseless value. ``policy_rate(P, pi)`` is therefore only a
    ceiling in the noiseless/identity-confusion case, not under arbitrary
    confusion — only ``oracle_rate`` is a universal ceiling.
    """
    _validate(P=P, pi=pi, C=C, p=p)
    prior = _default_prior(P.shape[0]) if p is None else p
    R = P[:, pi]
    return float(prior @ (C * R).sum(axis=1))


def uniform_confusion(n_types: int, accuracy: float) -> np.ndarray:
    """Build a symmetric confusion matrix with uniform off-diagonal noise.

    Diagonal entries equal ``accuracy``; each off-diagonal entry in a row
    equals ``(1 - accuracy) / (n_types - 1)``, so every row sums to 1 and
    all non-diagonal confusions are equally likely.

    Parameters
    ----------
    n_types : int
        Number of types, N (must be >= 2).
    accuracy : float
        Diagonal (correct-prediction) probability, in [0, 1].

    Returns
    -------
    ndarray, shape (N, N)
        The row-stochastic uniform confusion matrix, dtype float64.

    Raises
    ------
    ValueError
        If ``n_types < 2`` or ``accuracy`` is outside [0, 1].
    """
    if n_types < 2:
        raise ValueError(f"n_types must be >= 2, got {n_types}")
    if not (0.0 <= accuracy <= 1.0):
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy!r}")
    off_diagonal = (1.0 - accuracy) / (n_types - 1)
    C = np.full((n_types, n_types), off_diagonal, dtype=np.float64)
    np.fill_diagonal(C, accuracy)
    return C


def normalize(rate: float, P: np.ndarray, p: Optional[np.ndarray] = None) -> float:
    """Affine-normalize a rate onto the [random_rate, oracle_rate] -> [0, 1] scale.

    ``normalize(rate, P, p) = (rate - random_rate(P, p)) / (oracle_rate(P, p) - random_rate(P, p))``.

    Parameters
    ----------
    rate : float
        The raw success rate to normalize (e.g. a ``policy_rate`` or
        ``policy_rate_under_confusion`` result, or a simulator estimate).
    P : ndarray, shape (N, K)
        Success-probability matrix, used to compute the floor/ceiling.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.

    Returns
    -------
    float
        The normalized score (0.0 at the random floor, 1.0 at the oracle
        ceiling; can fall outside [0, 1] if ``rate`` does).

    Raises
    ------
    ValueError
        If ``P`` or ``p`` fail validation, or if
        ``oracle_rate(P, p) - random_rate(P, p) < 1e-12`` (degenerate ``P``
        for which the floor/ceiling are indistinguishable).
    """
    _validate(P=P, p=p)
    o = oracle_rate(P, p)
    r = random_rate(P, p)
    denom = o - r
    if denom < 1e-12:
        raise ValueError(
            f"oracle_rate - random_rate must be >= 1e-12 to normalize, got {denom!r} "
            f"(oracle_rate={o!r}, random_rate={r!r})"
        )
    return float((rate - r) / denom)
