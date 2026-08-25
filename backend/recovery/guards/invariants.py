"""Runtime invariant guards over simulation results.

These are scientific assertions, not debugging helpers: they encode what
is mathematically IMPOSSIBLE for a correctly-implemented simulation to
produce, not what is merely surprising. On violation they raise
immediately and preserve the evidence in the exception message -- they
never print, warn, or continue.

The paper's historical 0.741 value (docs/anomalies.md) is exactly the
class of anomaly ``check_static_policy_ceiling`` exists to catch.

Statistical tolerance
----------------------
A simulation's empirical success rate is a Monte Carlo estimate of a true
underlying rate, not the rate itself. Its sampling error shrinks with the
number of episodes it was measured over -- comparing an empirical rate to
an analytic bound therefore needs a tolerance that SHRINKS as
``n_episodes`` grows, not a fixed epsilon. A fixed tolerance is
statistically incorrect in both directions: too loose at large
``n_episodes`` (it would let a real bug slip through once enough episodes
make the true deviation obvious), and too tight at small ``n_episodes``
(it would flag ordinary sampling noise as a violation).

Every guard here uses the same tolerance model, a normal approximation to
the worst-case Bernoulli standard error:

    slack(sigmas, n_episodes) = sigmas * sqrt(0.25 / n_episodes)

``0.25`` is ``p(1-p)`` maximized over ``p in [0, 1]`` (at ``p=0.5``) --
i.e. the largest possible per-episode Bernoulli variance, so this is a
conservative (upper) bound on the true standard error regardless of the
actual underlying rate. Default ``sigmas=3.0`` for one-sided checks;
``check_confusion_prediction`` is a two-sided comparison and defaults to
``sigmas=4.0`` (a two-sided test needs a wider band than a one-sided test
at the same false-positive rate).

Edge cases
----------
- ``n_episodes=1`` gives an enormous ``slack`` (e.g. 1.5 at sigmas=3),
  so these guards become very weak at tiny sample sizes -- this is
  correct, not a bug: with one episode you cannot statistically
  distinguish a real anomaly from ordinary noise, and a guard that
  claimed otherwise would itself be making a statistically unjustified
  claim.
- Degenerate matrices (e.g. every row identical, so
  ``oracle_rate == random_rate == worst_case``) do not cause division by
  zero or special-cased behavior here, because no guard in this module
  divides by ``oracle - random`` (unlike ``analytic.normalize``, which
  guards never call -- ``check_normalized_range`` takes an
  already-computed normalized value as input instead).
- Invalid policy indices (``pi`` containing an out-of-range action) are
  caught by ``recovery.theory.analytic``'s own validation (raises
  ``ValueError``) when a guard calls ``policy_rate``/
  ``policy_rate_under_confusion`` -- consistent with this module's rule
  that malformed INPUT is a ``ValueError``, never an
  ``InvariantViolation``.
"""

from typing import Optional

import numpy as np

from recovery.theory.analytic import (
    oracle_rate,
    policy_rate,
    policy_rate_under_confusion,
)

__all__ = [
    "InvariantViolation",
    "check_rate_in_unit_interval",
    "check_bounded_by_oracle",
    "check_above_worst_case",
    "check_static_policy_ceiling",
    "check_confusion_prediction",
    "check_normalized_range",
    "run_all_guards",
]

_MAX_BERNOULLI_VARIANCE = 0.25
DEFAULT_SIGMAS = 3.0
DEFAULT_CONFUSION_SIGMAS = 4.0
DEFAULT_NORMALIZED_LOWER = -1.0
DEFAULT_NORMALIZED_UPPER = 1.05


class InvariantViolation(Exception):
    """A simulation result violates a mathematically necessary bound.

    Raised only for scientific violations -- never for malformed input
    (that is always a ``ValueError``, raised before any statistical
    comparison is attempted).
    """


# ======================================================================
# Internal helpers
# ======================================================================


def _slack(sigmas: float, n_episodes: int) -> float:
    """Statistical tolerance: ``sigmas * sqrt(0.25 / n_episodes)``.

    Parameters
    ----------
    sigmas : float
        Number of standard errors of tolerance (must be > 0).
    n_episodes : int
        Number of episodes the empirical rate was measured over (must be
        >= 1).

    Returns
    -------
    float
        The tolerance to add/subtract around an analytic bound.

    Raises
    ------
    ValueError
        If ``sigmas <= 0`` or ``n_episodes < 1``.
    """
    if sigmas <= 0:
        raise ValueError(f"sigmas must be > 0, got {sigmas}")
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    return float(sigmas * np.sqrt(_MAX_BERNOULLI_VARIANCE / n_episodes))


def _validate_matrix(P: np.ndarray) -> np.ndarray:
    """Minimal shape/range validation for a success matrix (not a rate formula)."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D, got shape {P.shape}")
    if np.any(P < 0.0) or np.any(P > 1.0):
        raise ValueError(
            f"P must contain only values in [0, 1], got min={P.min()!r}, max={P.max()!r}"
        )
    return P


def _resolve_prior(P: np.ndarray, p: Optional[np.ndarray]) -> np.ndarray:
    """Validate an explicit prior, or return the uniform default."""
    n_types = P.shape[0]
    if p is None:
        return np.full(n_types, 1.0 / n_types, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n_types,):
        raise ValueError(f"p must have shape ({n_types},), got shape {p.shape}")
    total = float(p.sum())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"p must sum to 1 (atol=1e-9), got sum={total!r}")
    return p


def _violation(
    check_name: str,
    observed: float,
    bound: float,
    tolerance: str,
    n_episodes: Optional[int],
    detail: str = "",
) -> InvariantViolation:
    msg = (
        f"{check_name}: observed value {observed!r} violates bound {bound!r} "
        f"(tolerance={tolerance}, n_episodes={n_episodes!r})"
    )
    if detail:
        msg += f" -- {detail}"
    return InvariantViolation(msg)


# ======================================================================
# Guards
# ======================================================================


def check_rate_in_unit_interval(rate: float, n_episodes: Optional[int] = None) -> None:
    """A success rate must lie in [0, 1]. Hard bound -- no tolerance.

    Parameters
    ----------
    rate : float
        The empirical success rate to check.
    n_episodes : int, optional
        Included only for the exception message, if this check fails.

    Raises
    ------
    InvariantViolation
        If ``rate`` is outside [0, 1].
    """
    if not (0.0 <= rate <= 1.0):
        raise _violation(
            "check_rate_in_unit_interval", rate, "[0, 1]", "0 (hard bound)", n_episodes
        )


def check_bounded_by_oracle(
    rate: float,
    P: np.ndarray,
    n_episodes: int,
    p: Optional[np.ndarray] = None,
    sigmas: float = DEFAULT_SIGMAS,
) -> None:
    """Verify ``rate <= oracle_rate(P, p) + slack``. Applies to every policy.

    No policy -- fixed, adaptive, or anything else -- can ever exceed
    ``oracle_rate``: for any action played for true type ``t``, the
    per-episode expected reward is at most ``max_k P[t, k]``, so the
    aggregate can never exceed the weighted average of those per-type
    maxima.

    Parameters
    ----------
    rate : float
        The empirical success rate.
    P : numpy.ndarray, shape (N, K)
        The ground-truth success matrix.
    n_episodes : int
        Number of episodes ``rate`` was measured over.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    sigmas : float, default=3.0
        Statistical tolerance in standard errors.

    Raises
    ------
    ValueError
        If ``P``/``p``/``n_episodes``/``sigmas`` are invalid.
    InvariantViolation
        If ``rate`` exceeds the oracle ceiling by more than ``slack``.
    """
    ceiling = oracle_rate(P, p)
    slack = _slack(sigmas, n_episodes)
    bound = ceiling + slack
    if rate > bound:
        raise _violation(
            "check_bounded_by_oracle",
            rate,
            bound,
            f"{sigmas}-sigma (slack={slack!r})",
            n_episodes,
            detail=f"oracle_rate={ceiling!r}",
        )


def check_above_worst_case(
    rate: float,
    P: np.ndarray,
    n_episodes: int,
    p: Optional[np.ndarray] = None,
    sigmas: float = DEFAULT_SIGMAS,
) -> None:
    """Verify ``rate >= worst_case - slack``. Catches inverted reward logic.

    ``worst_case = sum_t p[t] * min_k P[t, k]`` is the dual of
    ``oracle_rate`` (max -> min) -- ``recovery.theory.analytic`` does not
    expose this as a public function, so it is computed inline here
    rather than duplicating ``oracle_rate``'s implementation; the
    mathematical justification is identical (per true type, the
    per-episode expected reward for ANY action lies in
    ``[min_k P[t, k], max_k P[t, k]]``, so the aggregate can never fall
    below the weighted average of the per-type minima). A rate that
    somehow falls below this floor is characteristic of e.g. a reward
    computed as ``1 - success`` instead of ``success``.

    Parameters
    ----------
    rate : float
        The empirical success rate.
    P : numpy.ndarray, shape (N, K)
        The ground-truth success matrix.
    n_episodes : int
        Number of episodes ``rate`` was measured over.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    sigmas : float, default=3.0
        Statistical tolerance in standard errors.

    Raises
    ------
    ValueError
        If ``P``/``p``/``n_episodes``/``sigmas`` are invalid.
    InvariantViolation
        If ``rate`` falls below the worst-case floor by more than ``slack``.
    """
    P_arr = _validate_matrix(P)
    prior = _resolve_prior(P_arr, p)
    worst_case = float(prior @ P_arr.min(axis=1))
    slack = _slack(sigmas, n_episodes)
    bound = worst_case - slack
    if rate < bound:
        raise _violation(
            "check_above_worst_case",
            rate,
            bound,
            f"{sigmas}-sigma (slack={slack!r})",
            n_episodes,
            detail=f"worst_case_rate={worst_case!r}",
        )


def check_static_policy_ceiling(
    rate: float,
    P: np.ndarray,
    pi: np.ndarray,
    n_episodes: int,
    p: Optional[np.ndarray] = None,
    sigmas: float = DEFAULT_SIGMAS,
) -> None:
    """Verify ``rate <= policy_rate(P, pi, p) + slack``.

    This is the guard that would have caught the historical Table VII
    anomaly (docs/anomalies.md): the paper's reported 0.741 for the blind
    6/7-correct rule exceeds ``policy_rate(MATRIX, BLIND_RULE) = 0.735714``
    by more than statistical noise can explain at a properly-powered
    episode count.

    ONLY VALID FOR FIXED POLICIES -- i.e. ``pi`` must be a deterministic
    ``predicted_type -> action`` mapping that does not change during the
    run (``recovery.policies.adapters.StaticRulePolicy``,
    ``OraclePolicy``). NEVER use this for adaptive policies (Thompson
    Sampling, UCB1, LinUCB, ...): their effective policy changes every
    episode as they learn, so no single ``pi`` vector bounds their
    achievable rate, and applying this check to one would produce a
    meaningless (usually far too tight) bound.

    Parameters
    ----------
    rate : float
        The empirical success rate.
    P : numpy.ndarray, shape (N, K)
        The ground-truth success matrix.
    pi : numpy.ndarray, shape (N,)
        The fixed policy's action for each (here: predicted) type.
    n_episodes : int
        Number of episodes ``rate`` was measured over.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    sigmas : float, default=3.0
        Statistical tolerance in standard errors.

    Raises
    ------
    ValueError
        If ``P``/``pi``/``p``/``n_episodes``/``sigmas`` are invalid.
    InvariantViolation
        If ``rate`` exceeds the fixed-policy ceiling by more than ``slack``.
    """
    ceiling = policy_rate(P, pi, p)
    slack = _slack(sigmas, n_episodes)
    bound = ceiling + slack
    if rate > bound:
        raise _violation(
            "check_static_policy_ceiling",
            rate,
            bound,
            f"{sigmas}-sigma (slack={slack!r})",
            n_episodes,
            detail=f"policy_rate={ceiling!r} (noiseless ceiling for this fixed policy)",
        )


def check_confusion_prediction(
    rate: float,
    P: np.ndarray,
    pi: np.ndarray,
    C: np.ndarray,
    n_episodes: int,
    p: Optional[np.ndarray] = None,
    sigmas: float = DEFAULT_CONFUSION_SIGMAS,
) -> None:
    """Verify ``abs(rate - policy_rate_under_confusion(P, pi, C, p)) <= slack``.

    Two-sided: an empirical rate that is surprisingly far BELOW the
    confusion-aware prediction is just as diagnostic of a bug (e.g. the
    wrong confusion matrix being applied, or a policy/environment
    mismatch) as one surprisingly far above it. Defaults to
    ``sigmas=4.0`` (wider than the one-sided checks' 3.0) since a
    two-sided test needs a larger multiplier for a comparable
    false-positive rate.

    Parameters
    ----------
    rate : float
        The empirical success rate.
    P : numpy.ndarray, shape (N, K)
        The ground-truth success matrix.
    pi : numpy.ndarray, shape (N,)
        The fixed policy's action for each predicted type.
    C : numpy.ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    n_episodes : int
        Number of episodes ``rate`` was measured over.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    sigmas : float, default=4.0
        Statistical tolerance in standard errors (two-sided).

    Raises
    ------
    ValueError
        If ``P``/``pi``/``C``/``p``/``n_episodes``/``sigmas`` are invalid.
    InvariantViolation
        If ``rate`` deviates from the confusion-aware prediction by more
        than ``slack``.
    """
    prediction = policy_rate_under_confusion(P, pi, C, p)
    slack = _slack(sigmas, n_episodes)
    diff = abs(rate - prediction)
    if diff > slack:
        raise _violation(
            "check_confusion_prediction",
            rate,
            f"{prediction!r} +/- {slack!r}",
            f"{sigmas}-sigma (slack={slack!r}, two-sided)",
            n_episodes,
            detail=f"analytic_prediction={prediction!r}, abs_diff={diff!r}",
        )


def check_normalized_range(
    normalized: float,
    lower: float = DEFAULT_NORMALIZED_LOWER,
    upper: float = DEFAULT_NORMALIZED_UPPER,
    n_episodes: Optional[int] = None,
) -> None:
    """Verify ``lower <= normalized <= upper``. Hard sanity bound, not statistical.

    Default bounds are ``[-1.0, 1.05]``, not ``[0, 1]``:

    - Negative normalized scores are VALID and expected. ``normalized =
      (rate - random_rate) / (oracle_rate - random_rate)``, so any policy
      that performs WORSE than uniform-random scores below 0 -- this is
      not a bug. ``NaiveRetryBaseline`` (always plays RETRY regardless of
      type) is the canonical example: it can easily score well below the
      random floor when RETRY is a poor action for most failure types,
      giving a normalized score around -0.37 in this repo's canonical
      environment (see docs/anomalies.md) -- a real, valid, and
      informative result, not an error.
    - The upper bound is ``1.05`` rather than exactly ``1.0`` to allow a
      small, FIXED margin for finite-sample overshoot right at the oracle
      ceiling (unlike the other guards, this bound intentionally does not
      shrink with ``n_episodes`` -- normalized scores are typically
      reported as one summary number per experiment configuration rather
      than tied to a single well-defined sample size the way a raw rate
      is, so a conservative fixed margin is used instead of statistical
      scaling). Values further above 1.05 indicate a genuine problem
      (e.g. exceeding the oracle ceiling by more than sampling noise can
      explain), not measurement noise.

    Parameters
    ----------
    normalized : float
        The normalized score to check.
    lower : float, default=-1.0
        Lower bound.
    upper : float, default=1.05
        Upper bound.
    n_episodes : int, optional
        Included only for the exception message, if this check fails.

    Raises
    ------
    InvariantViolation
        If ``normalized`` is outside ``[lower, upper]``.
    """
    if not (lower <= normalized <= upper):
        raise _violation(
            "check_normalized_range",
            normalized,
            f"[{lower!r}, {upper!r}]",
            "0 (hard bound)",
            n_episodes,
        )


def run_all_guards(
    rate: float,
    P: np.ndarray,
    n_episodes: int,
    pi: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    normalized: Optional[float] = None,
    sigmas: float = DEFAULT_SIGMAS,
    confusion_sigmas: float = DEFAULT_CONFUSION_SIGMAS,
) -> None:
    """Run every guard applicable to the data supplied.

    Always runs ``check_rate_in_unit_interval``, ``check_bounded_by_oracle``,
    and ``check_above_worst_case``. Additionally runs
    ``check_static_policy_ceiling`` if ``pi`` is given, and
    ``check_confusion_prediction`` if BOTH ``pi`` and ``C`` are given.
    Runs ``check_normalized_range`` if ``normalized`` is given. Never
    silently skips a check that could run given the arguments provided --
    the presence/absence of ``pi``/``C``/``normalized`` is the only thing
    that determines which checks fire.

    Parameters
    ----------
    rate : float
        The empirical success rate to validate.
    P : numpy.ndarray, shape (N, K)
        The ground-truth success matrix.
    n_episodes : int
        Number of episodes ``rate`` was measured over.
    pi : numpy.ndarray, shape (N,), optional
        A fixed policy's action for each predicted type. If given, enables
        ``check_static_policy_ceiling`` (see ITS docstring: only valid for
        fixed policies).
    C : numpy.ndarray, shape (N, N), optional
        Row-stochastic confusion matrix. If given together with ``pi``,
        enables ``check_confusion_prediction``.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    normalized : float, optional
        A normalized score to validate via ``check_normalized_range``, if
        given.
    sigmas : float, default=3.0
        Statistical tolerance (in standard errors) for the one-sided checks.
    confusion_sigmas : float, default=4.0
        Statistical tolerance for ``check_confusion_prediction``.

    Raises
    ------
    ValueError
        If any input is malformed.
    InvariantViolation
        If any applicable guard's bound is violated.
    """
    check_rate_in_unit_interval(rate, n_episodes=n_episodes)
    check_bounded_by_oracle(rate, P, n_episodes, p=p, sigmas=sigmas)
    check_above_worst_case(rate, P, n_episodes, p=p, sigmas=sigmas)
    if pi is not None:
        check_static_policy_ceiling(rate, P, pi, n_episodes, p=p, sigmas=sigmas)
    if pi is not None and C is not None:
        check_confusion_prediction(rate, P, pi, C, n_episodes, p=p, sigmas=confusion_sigmas)
    if normalized is not None:
        check_normalized_range(normalized, n_episodes=n_episodes)
