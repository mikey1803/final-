"""Prior-Guided Thompson Sampling (PGTS).

The implementation class is named ``PriorInitializedThompson`` (plus the
``DiscountedPriorThompson`` subclass); throughout documentation and
experimental analysis (``docs/experiments/hybrid.md``,
``experiments/prior_grid.py``, ``experiments/prior_analysis.py``) the
algorithm itself is referred to as **Prior-Guided Thompson Sampling
(PGTS)**.

PGTS initializes a Thompson Sampling bandit's Beta posteriors from a
hand-designed recovery table (instead of the uninformative ``Beta(1,1)``
prior every arm normally starts at), then updates online exactly like
stock Thompson Sampling. It is a NATIVE ``recovery.core.protocols.Policy``
implementation (not a legacy-selector wrapper, unlike most of
``recovery/policies/adapters.py``) so that Beta sampling can be fully
vectorized across strategies via ``rng.beta(...)`` -- see "Performance"
below.

Mathematical limits (see docs/experiments/hybrid.md for the full
derivation)
--------------------------------------------------------------------------
- ``strength=0`` -> every arm's prior is exactly ``Beta(1, 1)``, so PGTS
  is mathematically IDENTICAL to stock Thompson Sampling: the recovery
  table has no influence whatsoever (``tests/test_hybrid.py`` verifies
  this by checking two PGTS instances built from DIFFERENT tables at
  ``strength=0`` produce bitwise-identical action sequences).
- ``strength -> infinity`` -> the recommended arm's posterior mean is
  pinned at ``prior_mean`` with vanishing variance, so PGTS increasingly
  behaves like ``recovery.policies.adapters.StaticRulePolicy`` playing
  ``table``. This is a LARGE-STRENGTH-FINITE-HORIZON statement, not a
  literal ``s=inf`` construction (see "Why ``strength=inf`` is rejected"
  below, and docs/experiments/hybrid.md's "A subtlety in the
  large-strength limit" for why exact-from-episode-zero matching is not
  guaranteed even at very large finite strength, and how large that
  strength needs to be as a function of ``prior_mean``).

Why ``strength=inf`` is rejected
---------------------------------
``alpha0 = 1 + inf*prior_mean`` is not a finite number Beta sampling can
use. Rather than silently clamping or producing NaN, the constructor
raises ``ValueError`` and points callers at ``StaticRulePolicy`` --
which already exactly implements the "always follow the table, never
learn" policy this limit describes, with no approximation involved.
"""

import math
from typing import Optional

import numpy as np
from scipy.special import betaln, digamma

from recovery.core.protocols import Context

__all__ = ["PriorInitializedThompson", "DiscountedPriorThompson"]


def _validate_table(table: np.ndarray, n_strategies: int) -> np.ndarray:
    table = np.asarray(table)
    if table.ndim != 1:
        raise ValueError(f"table must be 1-D, got shape {table.shape}")
    if table.size == 0:
        raise ValueError("table must have at least one failure type")
    if np.any(table < 0) or np.any(table >= n_strategies):
        raise ValueError(
            f"table contains an out-of-range strategy index; must be in "
            f"[0, {n_strategies}), got min={int(table.min())}, max={int(table.max())}"
        )
    return table


class PriorInitializedThompson:
    """Prior-Guided Thompson Sampling (PGTS) -- see module docstring.

    Parameters
    ----------
    table : numpy.ndarray, shape (n_types,)
        ``table[t]`` is the recommended strategy index for failure type
        ``t`` (e.g. ``recovery.envs.canonical.BLIND_RULE``, or a table
        built by ``PriorInitializedThompson.from_quality``).
    n_strategies : int
        Number of strategies (``K``); must be consistent with ``table``'s
        values.
    strength : float, default=20.0
        Prior pseudo-count ``s >= 0`` placed on the recommended arm. See
        the module docstring for the ``s=0``/``s->infinity`` limits.
    prior_mean : float, default=0.7
        Believed success probability ``m in (0, 1)`` of the recommended
        arm, used to split ``strength`` into
        ``alpha0 = 1 + s*m``, ``beta0 = 1 + s*(1-m)``.
    gamma : float, default=1.0
        Posterior discount factor applied every ``update()`` call to the
        UPDATED failure type's entire row (every strategy, not just the
        one played): ``alpha <- 1 + gamma*(alpha-1)``,
        ``beta <- 1 + gamma*(beta-1)``. ``gamma=1.0`` is an exact identity
        (no discounting) -- this is what makes ``PriorInitializedThompson``
        and ``DiscountedPriorThompson`` the same class with a different
        default, rather than duplicated update logic (see
        ``DiscountedPriorThompson``). Repeated discounting decays ANY
        accumulated posterior -- including the informed prior itself --
        toward ``Beta(1,1)``, not toward the informed prior: see
        ``DiscountedPriorThompson``'s docstring.
    name : str, default="PGTS"
        Display name.

    Raises
    ------
    ValueError
        If ``strength`` is negative or infinite/NaN; if ``prior_mean`` is
        not in the OPEN interval ``(0, 1)``; if ``gamma`` is not in
        ``(0, 1]``; if ``table`` is not 1-D, is empty, or contains an
        out-of-range strategy index.
    """

    def __init__(
        self,
        table: np.ndarray,
        n_strategies: int,
        strength: float = 20.0,
        prior_mean: float = 0.7,
        gamma: float = 1.0,
        name: str = "PGTS",
    ) -> None:
        if not math.isfinite(strength):
            raise ValueError(
                f"strength must be finite, got {strength!r} -- for an infinitely "
                f"strong (always-follow-the-table) prior, use "
                f"recovery.policies.adapters.StaticRulePolicy directly instead of "
                f"approximating it here"
            )
        if strength < 0:
            raise ValueError(f"strength must be >= 0, got {strength!r}")
        if not (0.0 < prior_mean < 1.0):
            raise ValueError(f"prior_mean must be in the open interval (0, 1), got {prior_mean!r}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")

        table = _validate_table(table, n_strategies)

        self.name = name
        self.table = table.copy()
        self.n_types: int = table.shape[0]
        self.n_strategies: int = n_strategies
        self.strength = float(strength)
        self.prior_mean = float(prior_mean)
        self.gamma = float(gamma)

        self._alpha0, self._beta0 = self._build_prior()
        self._alpha = self._alpha0.copy()
        self._beta = self._beta0.copy()

    def _build_prior(self) -> tuple[np.ndarray, np.ndarray]:
        """Pure function of ``(table, n_strategies, strength, prior_mean)``:
        every call with the same inputs returns the same arrays, and
        nothing here reads or mutates instance state besides those four
        already-validated attributes."""
        alpha0 = np.ones((self.n_types, self.n_strategies), dtype=np.float64)
        beta0 = np.ones((self.n_types, self.n_strategies), dtype=np.float64)
        rows = np.arange(self.n_types)
        alpha0[rows, self.table] = 1.0 + self.strength * self.prior_mean
        beta0[rows, self.table] = 1.0 + self.strength * (1.0 - self.prior_mean)
        if not (np.all(np.isfinite(alpha0)) and np.all(np.isfinite(beta0))):
            raise ValueError(
                f"strength={self.strength!r} combined with prior_mean={self.prior_mean!r} "
                f"overflows float64 during prior construction (alpha0/beta0 would contain inf) "
                f"-- use a smaller finite strength, or StaticRulePolicy for the s->infinity limit"
            )
        return alpha0, beta0

    def select(self, ctx: Context, rng: np.random.Generator) -> int:
        """Sample ``theta ~ Beta(alpha, beta)`` for every strategy of the
        PREDICTED failure type in one vectorized call, and return
        ``argmax(theta)``. Never loops over strategies in Python."""
        t = ctx.predicted_type
        theta = rng.beta(self._alpha[t], self._beta[t])
        return int(np.argmax(theta))

    def update(self, ctx: Context, action: int, reward: int) -> None:
        """Discount the PREDICTED type's row (identity if ``gamma==1.0``),
        then apply the Bernoulli update to the played arm only:
        ``alpha[t, action] += reward``, ``beta[t, action] += 1 - reward``.

        Conditions on ``ctx.predicted_type`` -- never a true type, which
        ``Context`` structurally cannot carry (see
        ``recovery.core.protocols.Context``) -- exactly like every other
        policy in this codebase.
        """
        t = ctx.predicted_type
        if self.gamma != 1.0:
            self._alpha[t] = 1.0 + self.gamma * (self._alpha[t] - 1.0)
            self._beta[t] = 1.0 + self.gamma * (self._beta[t] - 1.0)
        self._alpha[t, action] += reward
        self._beta[t, action] += 1 - reward

    def reset(self) -> None:
        """Restore the ORIGINAL informed prior (``alpha0``/``beta0``) --
        NOT uniform ``Beta(1,1)``. A fresh run of a strength>0 PGTS
        instance should start from the same informed belief every time,
        exactly like its first construction."""
        self._alpha = self._alpha0.copy()
        self._beta = self._beta0.copy()

    # ------------------------------------------------------------------
    # Diagnostics -- read-only; must never influence select()/update().
    # ------------------------------------------------------------------

    def posterior_mean(self) -> np.ndarray:
        """``alpha / (alpha + beta)`` for every (type, strategy), shape
        ``(n_types, n_strategies)``."""
        return self._alpha / (self._alpha + self._beta)

    def posterior_variance(self) -> np.ndarray:
        """Analytic ``Beta(alpha, beta)`` variance,
        ``alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))``, for every
        (type, strategy).

        Raises
        ------
        ValueError
            If ``(alpha+beta)**2`` overflows float64 -- reachable only at
            pathologically large ``strength`` (alpha0/beta0 themselves
            stay finite well beyond any value this codebase's experiment
            grid uses; squaring them in the variance formula is what
            overflows first). Raised rather than silently returning
            ``inf``/``nan``.
        """
        a, b = self._alpha, self._beta
        s = a + b
        with np.errstate(over="raise"):
            try:
                denom = s * s * (s + 1.0)
            except FloatingPointError as exc:
                raise ValueError(
                    "posterior_variance overflowed float64 -- (alpha+beta) is too large "
                    "(pathologically large `strength`); this is a numerical limit, not a "
                    "meaningful variance"
                ) from exc
        return (a * b) / denom

    def posterior_entropy(self) -> np.ndarray:
        """Analytic differential entropy of ``Beta(alpha, beta)``, for
        every (type, strategy):
        ``ln B(a,b) - (a-1)*psi(a) - (b-1)*psi(b) + (a+b-2)*psi(a+b)``,
        using ``scipy.special.betaln``/``digamma`` (the standard closed
        form -- not Monte Carlo estimated). Can be negative (this is
        DIFFERENTIAL entropy of a continuous density, not discrete
        Shannon entropy, which has no sign constraint). Used by
        ``experiments/prior_analysis.py`` to show entropy COLLAPSE as a
        strong prior's tight initial concentration is either confirmed
        (stays low) or overturned (rises, then falls again as new
        evidence accumulates) by real outcomes.
        """
        a, b = self._alpha, self._beta
        return betaln(a, b) - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) + (a + b - 2.0) * digamma(a + b)

    def effective_prior_strength(self) -> np.ndarray:
        """``alpha0 + beta0 - 2`` for every (type, strategy): the ORIGINAL
        prior pseudo-count still in effect at each arm (``strength`` on
        the recommended arm, ``0`` everywhere else) -- constant over the
        policy's lifetime (it is a property of initialization, not of
        the current posterior); use ``posterior_confidence()`` for the
        CURRENT (evidence-inclusive) pseudo-count."""
        return self._alpha0 + self._beta0 - 2.0

    def posterior_confidence(self) -> np.ndarray:
        """``alpha + beta`` for every (type, strategy): the CURRENT total
        pseudo-count (prior + accumulated evidence) -- higher means a
        tighter, more confident posterior. Unlike
        ``effective_prior_strength()``, this changes as ``update()`` is
        called (and, under discounting, can shrink as well as grow)."""
        return self._alpha + self._beta

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_quality(
        matrix: np.ndarray,
        n_correct: int,
        rng,
    ) -> np.ndarray:
        """Build a recovery table with exactly ``n_correct`` rows mapped
        to their true best (argmax) strategy under ``matrix``, and every
        other row mapped to a uniformly random NON-best strategy.

        Reuses ``src.evaluation.static_rule_sweep.generate_static_policy``
        (the same quality-controlled rule generator Task 4/5's static-rule
        sweeps already use) rather than reimplementing quality-controlled
        rule generation -- converts its ``FailureType -> RecoveryStrategy``
        dict output to the index-array form every ``recovery/`` policy
        expects (``recovery.envs.canonical.STRATEGY_NAMES`` ordering,
        which is verified identical to ``RecoveryStrategy``'s enum order --
        see ``recovery/envs/canonical.py``'s module docstring cross-check).

        Parameters
        ----------
        matrix : numpy.ndarray, shape (n_types, n_strategies)
            The ground-truth success matrix ``n_correct`` is measured
            against (``argmax(axis=1)`` per row).
        n_correct : int
            Number of rows (out of ``matrix.shape[0]``) to assign their
            true best strategy; must be in ``[0, n_types]``. ``0`` is
            supported (a fully-incorrect table -- see Edge Cases in
            docs/experiments/hybrid.md).
        rng : int or random.Random
            Seed or already-constructed ``random.Random`` instance (the
            wrapped generator's own RNG type -- this factory does not
            introduce a second RNG convention).

        Returns
        -------
        numpy.ndarray, shape (n_types,), dtype int64
            The recovery table, ready to pass as ``PriorInitializedThompson``'s
            ``table`` argument.

        Raises
        ------
        ValueError
            If ``n_correct`` is outside ``[0, n_types]`` (raised by
            ``random.sample`` inside ``generate_static_policy`` for an
            out-of-range count).

        Notes
        -----
        Tie-breaking: ``matrix.argmax(axis=1)`` (used internally by
        ``generate_static_policy``) resolves ties -- multiple strategies
        achieving the row's maximum success probability -- by taking the
        FIRST such strategy in ``STRATEGY_NAMES``/``RecoveryStrategy``
        order, matching ``numpy.argmax``'s documented convention (the
        same convention ``recovery.theory.analytic``'s docstring already
        commits to for the "true best" strategy generally).
        """
        import random as _random

        from src.evaluation.matrix_family import _STRATS, _TYPES
        from src.evaluation.static_rule_sweep import generate_static_policy

        py_rng = rng if isinstance(rng, _random.Random) else _random.Random(rng)
        policy_dict = generate_static_policy(matrix, num_correct=n_correct, rng=py_rng)
        table = np.array([_STRATS.index(policy_dict[ft]) for ft in _TYPES], dtype=np.int64)
        return table


class DiscountedPriorThompson(PriorInitializedThompson):
    """Prior-Guided Thompson Sampling with posterior discounting enabled
    by default (``gamma=0.99`` instead of ``1.0``) -- referred to as
    **Discounted PGTS** in documentation/analysis.

    Identical to ``PriorInitializedThompson`` in every other respect (no
    logic is duplicated -- this subclass only changes ``gamma``'s
    default); the discounting mechanism itself already lives in the base
    class's ``update()`` (active whenever ``gamma != 1.0``).

    Why this lets an incorrect prior decay away under drift
    -----------------------------------------------------------
    Repeatedly applying ``alpha <- 1 + gamma*(alpha-1)`` without
    reinforcing evidence sends ``alpha -> 1`` geometrically (rate
    ``gamma`` per update) -- toward the UNINFORMATIVE ``Beta(1,1)``, not
    toward the original informed prior. This means an incorrect
    hand-designed recommendation is not "sticky": every update to a
    failure type's row erodes the ORIGINAL prior belief a little,
    regardless of whether that update's evidence came from the
    recommended arm or a different one. Under environmental drift (the
    previously-best strategy stops being best), each fresh outcome nudges
    the whole row's posterior back toward blank-slate uncertainty, after
    which ordinary Thompson exploration can find the new best arm --
    exactly the mechanism ``DiscountedThompsonSamplingSelector``
    (``src/recovery/selector.py``, untouched by this task) already uses
    for stock Thompson Sampling, applied here on top of an informed
    starting point instead of an uninformative one.
    """

    def __init__(
        self,
        table: np.ndarray,
        n_strategies: int,
        strength: float = 20.0,
        prior_mean: float = 0.7,
        gamma: float = 0.99,
        name: str = "DiscountedPGTS",
    ) -> None:
        super().__init__(table, n_strategies, strength=strength, prior_mean=prior_mean, gamma=gamma, name=name)
