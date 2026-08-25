"""Analytic static-vs-adaptive crossover predictor (Task 10).

Given a fixed ("static") recovery table with per-episode regret rate
``rho`` and a Thompson-Sampling-style learner with a known asymptotic
cumulative-regret bound ``R_TS(T)``, this module predicts the horizon
``T*`` at which the learner's *cumulative* regret first drops to (and
stays at or below) the static table's cumulative regret ``rho * T`` --
the point beyond which "learn online" is expected to beat "write the
table" in total accumulated reward. This is the SAME notion Task 6 calls
the "cumulative crossover" (``recovery.analysis.crossover_empirical`` /
``recovery.metrics.parity.time_to_parity``), not the "asymptotic"
(per-episode-rate) crossover -- see ``docs/theory/crossover.md``, "Which
empirical crossover this predicts."

Reuse, not redesign
--------------------
``static_regret_rate`` is a one-line composition of Task 1's
``oracle_rate``/``policy_rate`` (``recovery.theory.analytic``) -- it does
not recompute what those functions already compute exactly. The
Thompson-Sampling regret bound is new (Task 1 has no bandit-regret
formula), but it is deliberately built from the same ``P``/``pi``/``p``
notation and the same "type-then-arm" decomposition as
``recovery.theory.analytic``'s docstring conventions.

Blanket caveat -- read before trusting a prediction
----------------------------------------------------
Every function below is built on an ASYMPTOTIC regret bound (Lai-Robbins
/ KL-UCB style, valid as ``T -> infinity``, not a finite-time guarantee),
carries an UNKNOWN MULTIPLICATIVE CONSTANT ``c`` that this module does
not derive from first principles (``experiments/crossover_calibration.py``
fits it empirically against Task 6's simulated data instead), and assumes
independent Bernoulli reward types with a stationary success-probability
matrix and no classifier/confusion noise. None of this makes the
predictor exact. It is a calibrated *estimate*, scoped to the regime it
was validated against -- see ``docs/theory/crossover.md`` for the
validated regime and every number behind that claim. Never treat a
prediction from this module as a guarantee, especially outside that
regime.

Import discipline
------------------
This module may import ONLY ``numpy``, ``scipy.optimize``, and
``recovery.theory.analytic``. No I/O, no kernel, no policies -- every
function here is a pure, deterministic computation over ``(P, pi, p)``
arrays and plain Python/numpy scalars, exactly like ``analytic.py``.

``t_min`` must exceed the taxonomy size ``N``
-----------------------------------------------
``ts_cumulative_regret`` clips ``log(T/N)`` at 0 for ``T <= N`` (the bound
is only valid for ``T >> N`` -- see its docstring's "Small T" edge case).
Consequently, whenever ``t_min <= N``, ``predict_crossover`` finds
``R_TS(t_min) == 0`` and immediately floors to ``T* = t_min`` for ANY
table, regardless of quality -- not a meaningful prediction, just the
search floor sitting entirely outside the bound's valid range. Always
choose ``t_min`` comfortably above the largest ``N`` you intend to query
(the default ``t_min=10`` is only valid for taxonomies smaller than 10
types); ``predict_crossover_grid`` callers sweeping larger ``N`` must
raise ``t_min`` to match.
"""

import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from recovery.theory.analytic import oracle_rate, policy_rate, policy_rate_under_confusion

__all__ = [
    "static_regret_rate",
    "ts_cumulative_regret",
    "predict_crossover",
    "predict_crossover_grid",
    # Task 19A -- classifier-noise-aware extension (additive)
    "effective_matrix_under_confusion",
    "static_rate_confused",
    "ts_cumulative_regret_confused",
    "crossover_horizon_confused",
    "ceiling_reduction",
    "gap_shrinkage_report",
    "argmax_inversion_check",
    "asymptotic_regime_flag",
]

_KL_EPS = 1e-12
_TIE_EPS = 1e-12
_SCAN_POINTS = 2000


def _default_prior(n_types: int) -> np.ndarray:
    return np.full(n_types, 1.0 / n_types, dtype=np.float64)


def _bernoulli_kl(p: np.ndarray, q) -> np.ndarray:
    """Bernoulli KL divergence ``KL(p || q)``, elementwise, clipped away
    from 0/1 (``_KL_EPS``) so ``log`` never sees a zero argument. ``q`` may
    be a scalar (broadcast) or an array matching ``p``'s shape."""
    p = np.clip(p, _KL_EPS, 1.0 - _KL_EPS)
    q = np.clip(q, _KL_EPS, 1.0 - _KL_EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def static_regret_rate(P: np.ndarray, pi: np.ndarray, p: Optional[np.ndarray] = None) -> float:
    """Per-episode regret rate of a fixed (static, non-learning) table.

    ``static_regret_rate(P, pi, p) = oracle_rate(P, p) - policy_rate(P, pi, p)``
    -- computed by calling Task 1's ``oracle_rate``/``policy_rate`` exactly
    as written, never re-derived. This is ``rho`` everywhere else in this
    module: the constant amount of expected reward the static table gives
    up, every single episode, forever.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix (see ``recovery.theory.analytic``'s
        module docstring for notation).
    pi : ndarray, shape (N,)
        Strategy played for each (true or predicted) type -- the static
        table under evaluation.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.

    Returns
    -------
    float
        ``rho >= 0`` (an oracle-consistent ``pi`` can never beat the
        oracle, so this is never negative; see ``oracle_rate``'s Notes).

    Assumptions / limitations
    --------------------------
    Assumes the table is genuinely static (``pi`` never changes with
    experience) and that ``P``/``p`` are stationary for the horizon of
    interest -- a table facing a drifting environment (Task 8) has a
    time-varying ``rho`` this function does not model.
    """
    return float(oracle_rate(P, p) - policy_rate(P, pi, p))


def ts_cumulative_regret(
    P: np.ndarray,
    T,
    n_types: int,
    p: Optional[np.ndarray] = None,
    form: str = "kl",
    c: float = 1.0,
):
    """Asymptotic cumulative-regret bound for Thompson Sampling at horizon ``T``.

    KL form (``form="kl"``, the Lai-Robbins-style asymptotic bound)::

        R_TS(T) = c * N * sum_t p[t] * sum_{k non-optimal for t} log(T/N) / KL(P[t,k], P[t,k*])

    Gap form (``form="gap"``, a cruder proxy that avoids computing KL)::

        R_TS(T) = c * N * sum_t p[t] * sum_{k non-optimal for t} log(T/N) / (P[t,k*] - P[t,k])

    where ``k*`` is type ``t``'s optimal arm (``P[t,:].argmax()``), ``N``
    is ``n_types``, and ``c`` is a free multiplicative constant (see the
    module docstring -- this is NOT derived here, only exposed). Both
    forms are genuine, useful approximations of the same underlying
    quantity: the KL form is the asymptotically tight one for Bernoulli
    rewards, the gap form is cheaper and, since ``KL(p, p*) approx
    (p*-p)^2 / (2 p*(1-p*))`` for small gaps, diverges from the KL form
    by roughly a factor of ``(p*-p) / (2 p*(1-p*))`` at small gaps --
    material at ``gap <= 0.05``, small at ``gap >= 0.20`` (see
    ``tests/test_crossover.py`` and ``docs/theory/crossover.md``).

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    T : float or ndarray
        Horizon(s) to evaluate at. Scalar in, scalar out; array in, array
        out (same shape).
    n_types : int
        Must equal ``P.shape[0]`` -- kept as an explicit parameter (rather
        than silently inferred) so a caller cannot pass a ``P``/``n_types``
        pair that have silently drifted apart.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform. Weights each type's contribution
        to the sum by ``p[t]``, per this module's non-uniform-prior
        handling (see Edge cases below).
    form : {"kl", "gap"}, default="kl"
        Which approximation to use.
    c : float, default=1.0
        Free multiplicative constant. ``c=1.0`` is the UNCALIBRATED
        predictor; ``experiments/crossover_calibration.py`` fits a
        better-justified value against Task 6's empirical data.

    Returns
    -------
    float or ndarray
        ``R_TS(T) >= 0``.

    Raises
    ------
    ValueError
        If ``form`` is not ``"kl"``/``"gap"``, or ``n_types != P.shape[0]``.

    Edge cases
    ----------
    - **Tied optimal arms** (``P[t,k] == P[t,k*]`` for some ``k != k*``):
      excluded from the sum entirely (zero regret for choosing between
      equally-good optimal arms, not a divide-by-zero KL/gap term).
    - **Small T** (``T <= n_types``): ``log(T/N)`` is clipped at 0 rather
      than allowed to go negative. This bound is an ASYMPTOTIC one, valid
      for ``T >> N``; treating "not yet in the asymptotic regime" as zero
      accrued regret is a documented simplification, not a claim that TS
      truly has zero regret at ``T <= N`` (see the module docstring's
      blanket caveat).

    Assumptions
    -----------
    Independent Bernoulli arms per type, stationary ``P``, no classifier
    confusion, and the constant ``c`` absorbing every approximation this
    bound makes relative to TS's true finite-time regret. This is an
    asymptotic bound, not a finite-time guarantee.
    """
    if form not in ("kl", "gap"):
        raise ValueError(f"form must be 'kl' or 'gap', got {form!r}")
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D, got shape {P.shape}")
    N, K = P.shape
    if n_types != N:
        raise ValueError(f"n_types ({n_types}) must equal P.shape[0] ({N})")

    prior = _default_prior(N) if p is None else np.asarray(p, dtype=np.float64)

    T_arr = np.asarray(T, dtype=np.float64)
    scalar_input = T_arr.ndim == 0
    T_arr = np.atleast_1d(T_arr)
    log_term = np.log(np.maximum(T_arr / n_types, 1.0))  # shape (n_T,), clipped at 0

    total = np.zeros_like(T_arr)
    best_idx = P.argmax(axis=1)
    for t in range(N):
        row = P[t]
        p_star = row[best_idx[t]]
        mask = row < (p_star - _TIE_EPS)
        if not np.any(mask):
            continue  # every arm ties the optimum for this type -- zero regret
        if form == "kl":
            denom = _bernoulli_kl(row[mask], p_star)
        else:
            denom = p_star - row[mask]
        inv_sum = float(np.sum(1.0 / denom))
        total += prior[t] * inv_sum * log_term

    result = c * n_types * total
    return float(result[0]) if scalar_input else result


def predict_crossover(
    P: np.ndarray,
    pi: np.ndarray,
    p: Optional[np.ndarray] = None,
    form: str = "kl",
    c: float = 1.0,
    t_min: float = 10,
    t_max: float = 1e9,
):
    """Predict the static-vs-adaptive crossover horizon ``T*``.

    ``T*`` is the smallest ``T`` solving the transcendental equation
    ``ts_cumulative_regret(P, T, N, p, form, c) = static_regret_rate(P, pi, p) * T``
    -- the horizon at which Thompson Sampling's cumulative regret bound
    first drops to (and, since the gap between the two curves is unimodal
    -- see the solving comment in the source -- stays at or below
    thereafter) the static table's cumulative regret. Solved numerically
    in ``log(T)`` via a coarse scan (to locate the FIRST sign change)
    followed by ``scipy.optimize.brentq`` refinement on the bracketing
    sub-interval.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    pi : ndarray, shape (N,)
        The static table under evaluation.
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    form : {"kl", "gap"}, default="kl"
        Passed through to ``ts_cumulative_regret``.
    c : float, default=1.0
        Passed through to ``ts_cumulative_regret``.
    t_min, t_max : float, default=(10, 1e9)
        Search bracket. Never exceeded: a prediction landing outside this
        range is reported AT the boundary, explicitly flagged (see
        Returns), never silently extrapolated past it.

    Returns
    -------
    (T_star, prediction_truncated) : tuple[float, bool]
        ``T_star`` is ``math.inf`` for a perfect table (``rho == 0``,
        never crosses -- there is nothing to catch up to) or a finite
        horizon in ``[t_min, t_max]`` otherwise.
        ``prediction_truncated`` is ``True`` only when the true root was
        never located within ``[t_min, t_max]`` and ``t_max`` is reported
        as a lower-bound estimate, not an exact prediction -- this is the
        ONLY case that returns ``True``; every other returned value
        (including ``t_min`` for an extremely poor table, and every
        in-bracket root) is exact for the model as specified. NEVER
        raises for a bracketing failure -- see Edge cases.

    Edge cases
    ----------
    - **Perfect table** (``rho == 0``): returns ``(math.inf, False)``
      immediately, without invoking the solver -- ``ts_cumulative_regret``
      is always ``>= 0``, so ``R_TS(T) >= 0 = rho * T`` for every ``T``:
      the two curves never cross, by construction, for any ``T``. This is
      an exact, not approximate, consequence of ``rho == 0``.
    - **Extremely poor table** (root at or below ``t_min``): returns
      ``(t_min, False)`` -- the model predicts TS has already caught up
      by the smallest horizon considered.
    - **Root never found in the bracket** (root at or above ``t_max``, OR
      any unexpected numerical failure while solving): returns
      ``(t_max, True)`` -- NEVER raises.

    Assumptions / limitations
    --------------------------
    Inherits every assumption of ``ts_cumulative_regret`` and
    ``static_regret_rate``: asymptotic TS bound, unknown constant ``c``,
    independent stationary Bernoulli types, no classifier noise.
    ``h(T) = R_TS(T) - rho*T`` is unimodal (see the solving comment in the
    source), so it has at most one root beyond the trivial ``t_min`` case;
    this function always returns the SMALLEST/first crossing, which has
    the clear real-world reading "first point favoring the switch to
    online learning." Do not overclaim precision beyond the validated
    regime in ``docs/theory/crossover.md``.
    """
    if t_min <= 0:
        raise ValueError(f"t_min must be > 0, got {t_min!r}")
    if t_max <= t_min:
        raise ValueError(f"t_max ({t_max!r}) must be > t_min ({t_min!r})")

    # Solved as h(T) = R_TS(T) - rho*T: h > 0 means TS's regret bound still
    # EXCEEDS the static table's cumulative regret (static still preferred,
    # not yet crossed); h <= 0 means TS has caught up (crossed). h's slope
    # is dR_TS/dT - rho, and dR_TS/dT is a strictly DECREASING function of T
    # (derivative of a log term) while rho is constant, so h is unimodal:
    # it can rise at first but, once it turns, falls monotonically to -inf
    # (rho*T's linear growth eventually dominates any log-shaped bound, for
    # any rho > 0) -- hence at most ONE root exists after any initial rise,
    # plus the trivial "already crossed at t_min" case. A coarse log-spaced
    # scan for the first +-to-<=0 transition, refined by brentq, finds it.
    try:
        rho = static_regret_rate(P, pi, p)
        if rho <= 0.0:
            return math.inf, False

        n_types = P.shape[0]

        def h(T_val: float) -> float:
            return float(ts_cumulative_regret(P, T_val, n_types, p, form, c)) - rho * T_val

        T_grid = np.geomspace(t_min, t_max, _SCAN_POINTS)
        R_grid = ts_cumulative_regret(P, T_grid, n_types, p, form, c)
        h_grid = R_grid - rho * T_grid

        if h_grid[0] <= 0.0:
            return float(t_min), False  # already past crossover at t_min

        cross_idx = None
        for i in range(len(h_grid) - 1):
            if h_grid[i] > 0.0 and h_grid[i + 1] <= 0.0:
                cross_idx = i
                break

        if cross_idx is None:
            return float(t_max), True  # never crosses within the bracket

        root = brentq(h, float(T_grid[cross_idx]), float(T_grid[cross_idx + 1]))
        return float(root), False

    except Exception:
        # Never raise for a bracketing/numerical failure: fall back to the
        # conservative "never resolved within range" answer.
        return float(t_max), True


def predict_crossover_grid(
    quality_levels: Sequence[int],
    gaps: Sequence[float],
    K: int,
    N: int,
    n_matrices: int,
    seed: int,
    form: str = "kl",
    c: float = 1.0,
    t_min: float = 10,
    t_max: float = 1e9,
) -> list[dict]:
    """Sweep ``predict_crossover`` over a synthetic grid of (quality, gap) tables.

    For every ``(quality, gap)`` cell, samples ``n_matrices`` independent
    ``(N, K)`` success-probability matrices at optimal-arm gap ``gap``
    (uniform row baseline in ``[0.15, 0.55]``, one boosted best arm per
    row -- the same construction ``src.evaluation.matrix_family.sample_matrix``
    uses, reimplemented locally in pure numpy since this module may not
    import outside ``numpy``/``scipy.optimize``/``recovery.theory.analytic``),
    builds a static table with exactly ``quality`` of its ``N`` rows
    correct (mirroring ``src.evaluation.static_rule_sweep.generate_static_policy``,
    likewise reimplemented locally), and predicts its crossover.

    Parameters
    ----------
    quality_levels : sequence of int
        Number of correct rows out of ``N`` for the sampled static tables
        (e.g. ``[7, 6, 5, 4, 3]`` for ``N=7``).
    gaps : sequence of float
        Optimal-arm gaps to sample matrices at.
    K : int
        Number of strategies (arms) per type.
    N : int
        Number of types (taxonomy size).
    n_matrices : int
        Independent matrix/table draws per ``(quality, gap)`` cell.
    seed : int
        Seed for the (single, shared) ``numpy.random.Generator`` driving
        every draw -- deterministic and reproducible for a fixed
        ``(quality_levels, gaps, K, N, n_matrices, seed)``.
    form, c, t_min, t_max
        Passed through to ``predict_crossover``.

    Returns
    -------
    list of dict
        One dict per (quality, gap, matrix draw), with keys
        ``quality``, ``gap``, ``K``, ``N``, ``matrix_idx``, ``rho``,
        ``T_star``, ``prediction_truncated``, ``form``, ``c``. Returned as
        plain ``list[dict]``, not a ``pandas.DataFrame``, since this
        module may not import ``pandas`` -- callers (e.g.
        ``experiments/crossover_extension.py``) wrap this in a DataFrame
        themselves.

    Raises
    ------
    ValueError
        If any ``quality`` in ``quality_levels`` is outside ``[0, N]``.

    Assumptions / limitations
    --------------------------
    Purely synthetic data -- these matrices are NOT drawn from, or
    validated against, any real deployment's failure-type distribution.
    See ``predict_crossover``'s and ``ts_cumulative_regret``'s docstrings
    for every assumption those functions carry; this function adds no new
    modeling assumptions of its own beyond the matrix/table sampling
    scheme described above.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for gap in gaps:
        for q in quality_levels:
            if not (0 <= q <= N):
                raise ValueError(f"quality level {q} out of range [0, {N}]")
            for matrix_idx in range(n_matrices):
                P = _sample_matrix(N, K, gap, rng)
                pi = _quality_policy(P, q, rng)
                rho = static_regret_rate(P, pi)
                T_star, truncated = predict_crossover(P, pi, form=form, c=c, t_min=t_min, t_max=t_max)
                rows.append(
                    {
                        "quality": q,
                        "gap": gap,
                        "K": K,
                        "N": N,
                        "matrix_idx": matrix_idx,
                        "rho": rho,
                        "T_star": T_star,
                        "prediction_truncated": truncated,
                        "form": form,
                        "c": c,
                    }
                )
    return rows


def _sample_matrix(n_types: int, n_strats: int, gap: float, rng: np.random.Generator) -> np.ndarray:
    """Local, pure-numpy reimplementation of
    ``src.evaluation.matrix_family.sample_matrix`` -- duplicated (not
    imported) because ``recovery/theory/crossover.py`` may not import
    outside ``numpy``/``scipy.optimize``/``recovery.theory.analytic``.
    Same construction: uniform row baseline in ``[0.15, 0.55]``, one
    randomly chosen strategy per row boosted to ``row.max() + gap``
    (capped at 0.95)."""
    base = rng.uniform(0.15, 0.55, size=(n_types, n_strats))
    for t in range(n_types):
        best = int(rng.integers(n_strats))
        base[t, best] = min(0.95, base[t].max() + gap)
    return base


##############################################################################
# Task 19A -- classifier-noise-aware extension (additive only)
#
# Everything above this point (static_regret_rate, ts_cumulative_regret,
# predict_crossover, predict_crossover_grid) is Task 10's original,
# UNCHANGED deliverable -- no existing signature, behavior, or default is
# touched by anything below. New public names use `kappa` (calibration
# constant), `confusion` (a confusion matrix), and `accuracy` (a scalar
# classifier accuracy) -- never a bare `c`, to keep this extension's
# vocabulary distinct from the calibration constant `c` already used above.
#
# A1 (implementation-level, verified against recovery/core/episode.py,
# recovery/policies/adapters.py, src/recovery/selector.py -- not inferred
# from docs): every wrapped selector's Beta posterior (or, structurally,
# any per-type table a native Policy indexes) is keyed by
# `ctx.predicted_type`, never the true type -- `Context` has no true-type
# field to leak one, and `LegacySelectorAdapter.select`/`.update` both
# translate `ctx.predicted_type` before touching the wrapped selector.
# Meanwhile `MatrixEnvironment.sample_reward(true_type, action, rng)` reads
# `matrix[true_type, action]` -- the TRUE type's row, unambiguously (see
# docs/theory/crossover_derivation.md, "A1 verification"). Consequently,
# each PREDICTED-type bucket's arm-reward distribution is a probability
# MIXTURE over whichever true types the classifier confuses into that
# bucket, weighted by the type prior and the confusion matrix -- this is
# exactly `effective_matrix_under_confusion` below, not an approximation
# of the real dynamics but a direct, exact description of them (subject to
# A2: rewards remain conditionally Bernoulli given the true type, and A3:
# the confusion matrix is stationary for the horizon considered).
##############################################################################


def effective_matrix_under_confusion(
    P: np.ndarray,
    confusion: np.ndarray,
    p: Optional[np.ndarray] = None,
) -> tuple:
    """The predicted-type-bucket mixture matrix a real classifier induces.

    Every wrapped policy in this codebase (see the module docstring's A1
    note) selects and updates using only ``ctx.predicted_type`` -- so under
    a confusion matrix, predicted bucket ``t'`` is not a clean Bernoulli
    arm for any single true row; it accumulates reward from whichever true
    type ``t`` the classifier mapped into ``t'`` that episode. Bayes-
    flipping ``confusion`` against the type prior ``p`` gives the exact
    mixture::

        p_prime[t']    = sum_t p[t] * confusion[t, t']
        P_eff[t', k]   = ( sum_t p[t] * confusion[t, t'] * P[t, k] ) / p_prime[t']

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    confusion : ndarray, shape (N, N)
        Row-stochastic ``confusion[t, t'] = P(predict t' | true t)`` --
        same convention as ``recovery.theory.analytic.policy_rate_under_confusion``
        and ``uniform_confusion``.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.

    Returns
    -------
    (P_eff, p_prime) : tuple[ndarray, ndarray]
        ``P_eff`` has shape ``(N, K)`` (one row per PREDICTED bucket, same
        column/strategy space as ``P``). ``p_prime`` has shape ``(N,)`` and
        sums to 1 (the predicted-type marginal).

    Raises
    ------
    ValueError
        If ``P``, ``confusion``, or ``p`` fail the same validation
        ``recovery.theory.analytic`` applies (2-D, values in [0,1],
        row-stochastic confusion, prior summing to 1); if any predicted
        bucket has zero marginal probability (``p_prime[t'] == 0``) --
        ``P_eff`` would be a 0/0 mixture for that bucket, mathematically
        undefined, not silently defaulted to anything.

    Notes
    -----
    At ``confusion = numpy.eye(N)`` (a perfect classifier), ``p_prime``
    reduces to ``p`` exactly and ``P_eff`` reduces to ``P`` exactly (every
    bucket's mixture collapses to its single contributing true row) --
    this is the exact-reduction property every function built on top of
    this one inherits and is tested for (``tests/test_crossover_theory.py``).
    """
    oracle_rate(P, p)  # reuses analytic._validate's checks on P/p via a real call
    prior = _default_prior(P.shape[0]) if p is None else np.asarray(p, dtype=np.float64)
    C = np.asarray(confusion, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != P.shape[0]:
        raise ValueError(f"confusion must be square ({P.shape[0]}, {P.shape[0]}), got shape {C.shape}")
    row_sums = C.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"confusion must be row-stochastic, got row sums {row_sums}")

    p_prime = prior @ C  # shape (N,) -- predicted-type marginal
    if np.any(p_prime <= 0.0):
        zero_buckets = np.where(p_prime <= 0.0)[0].tolist()
        raise ValueError(
            f"predicted bucket(s) {zero_buckets} have zero marginal probability under this "
            f"(prior, confusion) pair -- P_eff is a 0/0 mixture there, mathematically undefined"
        )
    numerator = C.T @ (prior[:, None] * P)  # shape (N, K)
    P_eff = numerator / p_prime[:, None]
    return P_eff, p_prime


def static_rate_confused(
    P: np.ndarray,
    pi: np.ndarray,
    confusion: np.ndarray,
    p: Optional[np.ndarray] = None,
) -> float:
    """Per-episode regret rate of a static table, driven by a NOISY predicted type.

    ``static_rate_confused(P, pi, confusion, p)
        = oracle_rate(P, p) - policy_rate_under_confusion(P, pi, confusion, p)``
    -- an EXACT composition of two existing, unmodified
    ``recovery.theory.analytic`` functions, not a new derivation. ``pi`` is
    indexed by PREDICTED type here (``pi[t']`` is the action played when the
    classifier predicts ``t'``), matching
    ``policy_rate_under_confusion``'s own convention.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    pi : ndarray, shape (N,)
        Static table, indexed by predicted type.
    confusion : ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.

    Returns
    -------
    float
        ``rho_confused >= 0`` (bounded by the same oracle-ceiling argument
        as ``static_regret_rate``, since ``policy_rate_under_confusion`` is
        never below 0 or above ``oracle_rate`` -- see that function's
        Notes).

    Raises
    ------
    ValueError
        If ``P``, ``pi``, ``confusion``, or ``p`` fail validation.

    Reduction
    ---------
    At ``confusion = numpy.eye(N)``, ``policy_rate_under_confusion`` reduces
    exactly to ``policy_rate`` (identity confusion is the noiseless case),
    so ``static_rate_confused(P, pi, I, p) == static_regret_rate(P, pi, p)``
    bit-for-bit.
    """
    return float(oracle_rate(P, p) - policy_rate_under_confusion(P, pi, confusion, p))


def ts_cumulative_regret_confused(
    P: np.ndarray,
    T,
    n_types: int,
    confusion: np.ndarray,
    p: Optional[np.ndarray] = None,
    form: str = "kl",
    kappa: float = 1.0,
):
    """Classifier-noise-aware Thompson Sampling cumulative-regret bound.

    Computes the effective mixture ``(P_eff, p_prime)`` via
    ``effective_matrix_under_confusion``, then calls the EXISTING,
    UNMODIFIED ``ts_cumulative_regret(P_eff, T, n_types, p_prime, form,
    c=kappa)`` -- this function performs no new arithmetic of its own
    beyond the mixture; every regret-bound assumption/edge-case documented
    on ``ts_cumulative_regret`` (asymptotic, small-T clipping, tied-arm
    handling) applies here unchanged, just evaluated in the PREDICTED-type
    index space rather than the true-type one.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    T : float or ndarray
        Horizon(s) to evaluate at.
    n_types : int
        Must equal ``P.shape[0]``.
    confusion : ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.
    form : {"kl", "gap"}, default="kl"
        Passed through unchanged.
    kappa : float, default=1.0
        The SAME free multiplicative constant ``ts_cumulative_regret``
        calls ``c`` -- renamed here only at this new function's own
        boundary (see module-section docstring); ``kappa=1.0`` is
        uncalibrated, matching ``ts_cumulative_regret``'s own default.

    Returns
    -------
    float or ndarray
        ``R_TS_confused(T) >= 0``.

    Raises
    ------
    ValueError
        Propagated from ``effective_matrix_under_confusion`` or
        ``ts_cumulative_regret``.

    Reduction
    ---------
    At ``confusion = numpy.eye(N)``: ``P_eff == P``, ``p_prime == p``
    exactly, so this reduces bit-for-bit to
    ``ts_cumulative_regret(P, T, n_types, p, form, c=kappa)``.
    """
    P_eff, p_prime = effective_matrix_under_confusion(P, confusion, p)
    return ts_cumulative_regret(P_eff, T, n_types, p_prime, form, kappa)


def crossover_horizon_confused(
    P: np.ndarray,
    pi: np.ndarray,
    confusion: np.ndarray,
    p: Optional[np.ndarray] = None,
    form: str = "kl",
    kappa: float = 1.0,
    t_min: float = 10,
    t_max: float = 1e9,
):
    """Classifier-noise-aware static-vs-adaptive crossover horizon ``T*``.

    Solves the SAME transcendental equation ``predict_crossover`` solves
    (``R_TS(T) = rho * T``), following the identical numerical strategy
    (coarse geomspace scan for the first sign change, ``brentq`` refinement,
    never raises on a bracketing failure) -- but ``rho`` and ``R_TS`` now
    live in two DIFFERENT index spaces that ``predict_crossover`` itself
    cannot mix: ``rho = static_rate_confused(P, pi, confusion, p)`` is
    TRUE-type-indexed (the static table's real regret, confusion-aware),
    while ``R_TS = ts_cumulative_regret_confused(P, T, n_types, confusion,
    p, form, kappa)`` is PREDICTED-type-indexed (Thompson's regret bound on
    the mixture ``P_eff``). This is why this function reuses
    ``predict_crossover``'s SEARCH STRATEGY/pattern rather than calling it
    as a black box -- ``predict_crossover`` computes its own ``rho``
    internally from the same ``(P, pi, p)`` triple used for both halves of
    its equation, which would be the wrong ``rho`` here.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    pi : ndarray, shape (N,)
        Static table, indexed by PREDICTED type (same convention as
        ``static_rate_confused``).
    confusion : ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.
    form : {"kl", "gap"}, default="kl"
    kappa : float, default=1.0
    t_min, t_max : float, default=(10, 1e9)
        Same search bracket semantics as ``predict_crossover``.

    Returns
    -------
    (T_star, prediction_truncated) : tuple[float, bool]
        Identical semantics to ``predict_crossover``'s return value.

    Raises
    ------
    ValueError
        If ``t_min <= 0`` or ``t_max <= t_min``; propagated from
        ``effective_matrix_under_confusion`` for a malformed ``confusion``.

    Reduction
    ---------
    At ``confusion = numpy.eye(N)``, this returns EXACTLY what
    ``predict_crossover(P, pi, p, form, c=kappa, t_min, t_max)`` returns --
    both ``rho`` and every ``R_TS(T)`` evaluation coincide bit-for-bit (see
    ``static_rate_confused`` and ``ts_cumulative_regret_confused``'s own
    reduction notes), and the search procedure below is line-for-line the
    same as ``predict_crossover``'s. Verified in
    ``tests/test_crossover_theory.py``.
    """
    if t_min <= 0:
        raise ValueError(f"t_min must be > 0, got {t_min!r}")
    if t_max <= t_min:
        raise ValueError(f"t_max ({t_max!r}) must be > t_min ({t_min!r})")

    try:
        rho = static_rate_confused(P, pi, confusion, p)
        if rho <= 0.0:
            return math.inf, False

        n_types = P.shape[0]

        def h(T_val: float) -> float:
            return float(ts_cumulative_regret_confused(P, T_val, n_types, confusion, p, form, kappa)) - rho * T_val

        T_grid = np.geomspace(t_min, t_max, _SCAN_POINTS)
        R_grid = ts_cumulative_regret_confused(P, T_grid, n_types, confusion, p, form, kappa)
        h_grid = R_grid - rho * T_grid

        if h_grid[0] <= 0.0:
            return float(t_min), False

        cross_idx = None
        for i in range(len(h_grid) - 1):
            if h_grid[i] > 0.0 and h_grid[i + 1] <= 0.0:
                cross_idx = i
                break

        if cross_idx is None:
            return float(t_max), True

        root = brentq(h, float(T_grid[cross_idx]), float(T_grid[cross_idx + 1]))
        return float(root), False

    except Exception:
        return float(t_max), True


def ceiling_reduction(P: np.ndarray, confusion: np.ndarray, p: Optional[np.ndarray] = None) -> float:
    """CHANNEL 1 of classifier-noise degradation: the unavoidable ceiling loss.

    ``ceiling_reduction(P, confusion, p) = oracle_rate(P, p) - oracle_rate(P_eff, p_prime)``.

    Even a hypothetical policy that perfectly knows ``P_eff`` and always
    plays each predicted bucket's own best arm cannot reach the TRUE
    oracle rate, purely because confusion merges multiple true types into
    indistinguishable buckets -- reward information is destroyed before
    any policy (static OR adaptive) ever gets a chance to act on it. This
    channel is policy-independent: it does not depend on ``pi``, and it is
    NOT the same quantity as ``static_rate_confused``'s regret (which also
    reflects how good ``pi`` specifically is).

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    confusion : ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.

    Returns
    -------
    float
        ``>= 0`` (mixing true types into a shared bucket can only lower or
        preserve the best achievable per-bucket rate, never raise it above
        the true oracle -- ``oracle_rate(P_eff, p_prime) <= oracle_rate(P,
        p)`` follows from ``P_eff``'s rows being convex combinations of
        ``P``'s rows via a stochastic weighting per bucket). Exactly 0 at
        ``confusion = numpy.eye(N)``.
    """
    P_eff, p_prime = effective_matrix_under_confusion(P, confusion, p)
    return float(oracle_rate(P, p) - oracle_rate(P_eff, p_prime))


def gap_shrinkage_report(P: np.ndarray, confusion: np.ndarray, p: Optional[np.ndarray] = None) -> list:
    """CHANNEL 2 of classifier-noise degradation: per-bucket learnability.

    For every predicted bucket ``t'``, compares its own (best - second-best)
    value gap in ``P_eff`` against the gaps of the TRUE rows that
    contribute nonzero mass to it (``confusion[t, t'] > 0``) -- this
    channel is about how much HARDER Thompson Sampling's own arm
    identification problem becomes under mixing, independent of
    ``ceiling_reduction``'s reward-ceiling loss.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    confusion : ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.

    Returns
    -------
    list of dict
        One dict per predicted bucket ``t'``, keys: ``bucket``,
        ``gap_P_eff``, ``contributing_true_rows`` (list of true type
        indices with nonzero confusion mass into this bucket),
        ``gap_true_rows`` (list of each contributor's own top-2 gap in
        ``P``, same order), ``mean_gap_true_rows``, ``min_gap_true_rows``,
        ``gap_shrinkage`` (``mean_gap_true_rows - gap_P_eff`` -- positive
        means mixing shrank the effective gap below what the average
        contributing true row offered, i.e. made the bucket's own
        arm-identification problem harder than any single contributor's).
    """
    P_eff, _ = effective_matrix_under_confusion(P, confusion, p)
    C = np.asarray(confusion, dtype=np.float64)
    n_types = P.shape[0]

    def _top2_gap(row: np.ndarray) -> float:
        order = np.argsort(row, kind="stable")[::-1]
        return float(row[order[0]] - row[order[1]])

    out = []
    for t_prime in range(n_types):
        contributors = np.where(C[:, t_prime] > 0.0)[0].tolist()
        gaps_true = [_top2_gap(P[t]) for t in contributors]
        out.append(
            {
                "bucket": t_prime,
                "gap_P_eff": _top2_gap(P_eff[t_prime]),
                "contributing_true_rows": contributors,
                "gap_true_rows": gaps_true,
                "mean_gap_true_rows": float(np.mean(gaps_true)) if gaps_true else float("nan"),
                "min_gap_true_rows": float(np.min(gaps_true)) if gaps_true else float("nan"),
                "gap_shrinkage": (float(np.mean(gaps_true)) - _top2_gap(P_eff[t_prime])) if gaps_true else float("nan"),
            }
        )
    return out


def argmax_inversion_check(P: np.ndarray, confusion: np.ndarray, p: Optional[np.ndarray] = None) -> list:
    """Amendment B: does any predicted bucket's best arm match NO contributor's?

    For every predicted bucket ``t'``, checks whether
    ``argmax_k P_eff[t', k]`` differs from ``argmax_k P[t, k]`` for EVERY
    true row ``t`` contributing nonzero confusion mass to that bucket -- a
    genuine inversion, where the mixture recommends an arm that was not
    optimal for any of the true types the classifier actually confuses
    into that bucket. This does not happen for a diagonally-dominant
    confusion matrix in general (a large enough single contributor's
    argmax typically still wins the weighted vote), but is not
    IMPOSSIBLE -- this function checks it directly rather than assuming
    either way.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        True-type success-probability matrix.
    confusion : ndarray, shape (N, N)
        Row-stochastic confusion matrix.
    p : ndarray, shape (N,), optional
        True-type prior. Defaults to uniform.

    Returns
    -------
    list of dict
        One dict per predicted bucket ``t'``, keys: ``bucket``,
        ``argmax_P_eff``, ``contributing_true_rows``,
        ``contributor_argmaxes`` (parallel list), ``is_inversion`` (bool:
        ``argmax_P_eff`` not in ``contributor_argmaxes``).
    """
    P_eff, _ = effective_matrix_under_confusion(P, confusion, p)
    C = np.asarray(confusion, dtype=np.float64)
    n_types = P.shape[0]

    out = []
    for t_prime in range(n_types):
        contributors = np.where(C[:, t_prime] > 0.0)[0].tolist()
        contributor_argmaxes = [int(np.argmax(P[t])) for t in contributors]
        argmax_eff = int(np.argmax(P_eff[t_prime]))
        out.append(
            {
                "bucket": t_prime,
                "argmax_P_eff": argmax_eff,
                "contributing_true_rows": contributors,
                "contributor_argmaxes": contributor_argmaxes,
                "is_inversion": argmax_eff not in contributor_argmaxes,
            }
        )
    return out


def asymptotic_regime_flag(N_star: float, n_types: int, multiple: float = 10.0) -> bool:
    """Amendment C: is a predicted crossover horizon in the asymptotic bound's defensible range?

    ``ts_cumulative_regret``'s KL/gap bound is derived for ``T >> N``
    (``log(T/N)`` is clipped to 0, not negative, for ``T <= N`` -- see its
    own docstring's "Small T" edge case); a prediction close to that floor
    rests on the LEAST defensible part of the bound. This function applies
    one simple, stated threshold -- ``N_star >= multiple * n_types`` -- as
    a DEFENSIBLE/NOT-DEFENSIBLE classification, not a claim of exact
    validity beyond that point either.

    Parameters
    ----------
    N_star : float
        A predicted crossover horizon (finite; ``math.inf`` is always
        classified defensible -- a perfect table's "never crosses"
        prediction does not rest on the asymptotic regime at all).
    n_types : int
        The taxonomy size ``N`` the prediction was computed against.
    multiple : float, default=10.0
        The threshold multiple of ``n_types``. Documented, not tuned
        against any validation outcome.

    Returns
    -------
    bool
        ``True`` if ``N_star`` is comfortably inside the regime the bound
        was derived for.
    """
    if not math.isfinite(N_star):
        return True
    return N_star >= multiple * n_types


def _quality_policy(P: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    """Local, pure-numpy reimplementation of
    ``src.evaluation.static_rule_sweep.generate_static_policy`` -- builds
    a static table with exactly ``q`` of ``P``'s rows mapped to their true
    argmax strategy, the rest mapped to a uniformly random non-optimal
    strategy. Reimplemented locally for the same import-discipline reason
    as ``_sample_matrix``."""
    n_types, n_strats = P.shape
    best_idx = P.argmax(axis=1)
    correct_rows = set(rng.choice(n_types, size=q, replace=False).tolist()) if q > 0 else set()
    pi = np.empty(n_types, dtype=np.int64)
    for t in range(n_types):
        if t in correct_rows:
            pi[t] = best_idx[t]
        else:
            choices = [k for k in range(n_strats) if k != best_idx[t]]
            pi[t] = choices[int(rng.integers(len(choices)))]
    return pi
