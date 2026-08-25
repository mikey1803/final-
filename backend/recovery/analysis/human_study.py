"""Scoring and analysis for the blind-design elicitation study (Task 11).

Scoring happens ENTIRELY OFFLINE, here, after collection -- never inside
``study/app/backend``'s request-serving path (see that module's
docstring and ``study/protocol.md``'s blindness guarantee). This module
is free to import ``recovery.envs.canonical.MATRIX`` and
``recovery.theory.crossover`` (unlike ``study/``, it is not part of the
blind request path), and reuses Task 1's exact analytic functions and
Task 5/8A's bootstrap/effect-size machinery rather than re-deriving any
of them.
"""

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

from recovery.envs.canonical import TYPE_NAMES
from recovery.metrics.bootstrap import bootstrap_ci, cliffs_delta, magnitude
from recovery.theory.crossover import predict_crossover, static_regret_rate

__all__ = [
    "score_submission",
    "study_summary",
    "q_prior",
    "q_prior_crossover_distribution",
    "second_best_error_test",
    "bootstrap_correlation_ci",
    "compare_human_llm_q",
    "fleiss_kappa",
    "pairwise_agreement",
    "entropy_per_type",
]

YEARS_EXPERIENCE_LEVELS = ("0-2", "3-5", "6-10", "10+")
LLM_FAMILIARITY_LEVELS = ("none", "some", "expert")


def _rank_of_chosen(row: np.ndarray, chosen: int) -> int:
    """1-indexed rank of ``row[chosen]`` within ``row`` (1 = optimal / tied
    for optimal, ``K`` = worst). Competition ranking: ties share the best
    rank among them (``1 + count(strictly greater values)``)."""
    return int(1 + np.sum(row > row[chosen]))


def score_submission(mapping: Sequence[int], P: np.ndarray, p: Optional[np.ndarray] = None) -> dict:
    """Score one participant's ``mapping`` (a length-``N`` list of chosen
    strategy indices, one per failure type, in ``TYPE_NAMES`` order)
    against the hidden success-probability matrix ``P``.

    Parameters
    ----------
    mapping : sequence of int, length N
        ``mapping[t]`` is the strategy index the participant chose for
        failure type ``t``.
    P : ndarray, shape (N, K)
        The hidden success-probability matrix the participant never saw
        (``recovery.envs.canonical.MATRIX`` for the canonical study).
    p : ndarray, shape (N,), optional
        Type prior. Defaults to uniform (matches every other Task 1-10
        analytic function's default).

    Returns
    -------
    dict with keys:
        ``q_count`` (int): number of rows where the chosen strategy is
            (tied for) optimal, i.e. ``per_row_correct.sum()``.
        ``q_ratio`` (float): ``q_count / N``.
        ``per_row_correct`` (list of bool, length N): whether each row's
            chosen strategy achieves that row's max.
        ``rank_of_chosen`` (list of int, length N): 1-indexed rank of the
            chosen strategy within its row (1 = optimal).
        ``regret_rate`` (float): ``oracle_rate(P, p) - policy_rate(P, mapping, p)``
            -- computed via ``recovery.theory.crossover.static_regret_rate``,
            never re-derived (see that function's docstring).

    Raises
    ------
    ValueError
        If ``mapping`` has a length other than ``P.shape[0]``, or any
        entry is outside ``[0, P.shape[1])``.
    """
    P = np.asarray(P, dtype=np.float64)
    N, K = P.shape
    mapping_arr = np.asarray(mapping, dtype=np.int64)
    if mapping_arr.shape != (N,):
        raise ValueError(f"mapping must have length {N} (P.shape[0]), got shape {mapping_arr.shape}")
    if np.any(mapping_arr < 0) or np.any(mapping_arr >= K):
        raise ValueError(f"mapping entries must be in [0, {K}), got {mapping}")

    best_idx = P.argmax(axis=1)
    per_row_correct = [bool(mapping_arr[t] == best_idx[t] or P[t, mapping_arr[t]] == P[t, best_idx[t]]) for t in range(N)]
    rank_of_chosen = [_rank_of_chosen(P[t], int(mapping_arr[t])) for t in range(N)]

    return {
        "q_count": int(sum(per_row_correct)),
        "q_ratio": float(sum(per_row_correct) / N),
        "per_row_correct": per_row_correct,
        "rank_of_chosen": rank_of_chosen,
        "regret_rate": static_regret_rate(P, mapping_arr, p),
    }


def bootstrap_correlation_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Optional[tuple]:
    """Percentile bootstrap CI for the Spearman correlation between ``x``
    and ``y``, resampling PAIRS jointly (same pattern as
    ``recovery.metrics.bootstrap.paired_bootstrap_diff``, generalized to
    a correlation statistic rather than a mean difference -- that
    function does not cover this case, so this is new code, not a
    duplicate).

    Returns
    -------
    (correlation, ci_low, ci_high) or None
        ``None`` if fewer than 3 paired observations are available
        (Spearman correlation and its bootstrap are not meaningful below
        that) -- never raises, since this is a descriptive, not
        load-bearing, statistic in ``study_summary``.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = xa.size
    if n < 3 or ya.size != n:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        point = float(spearmanr(xa, ya).statistic)
    if np.isnan(point):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    # A resample can legitimately land on a constant x or y (e.g. every
    # resampled confidence rating happens to be "3") at small n --
    # spearmanr correctly reports this as undefined (NaN) with a warning;
    # already handled below by filtering NaNs out, so the warning itself
    # is expected noise, not a bug, and is suppressed here rather than
    # left to alarm a caller running this at the study's small sample size.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        resampled = np.array([spearmanr(xa[i], ya[i]).statistic for i in idx], dtype=np.float64)
    resampled = resampled[~np.isnan(resampled)]
    if resampled.size < n_resamples // 2:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo, hi = float(np.quantile(resampled, alpha)), float(np.quantile(resampled, 1.0 - alpha))
    return point, lo, hi


def second_best_error_test(scored_submissions: list[dict]) -> dict:
    """Directly test the paper's claim that a plausible human error picks
    the SECOND-best strategy: aggregate ``rank_of_chosen`` over every
    INCORRECT row (``per_row_correct is False``) across every submission.

    Parameters
    ----------
    scored_submissions : list of dict
        Each dict as returned by :func:`score_submission`.

    Returns
    -------
    dict with keys ``n_incorrect_rows``, ``mean_rank_of_incorrect``,
    ``mean_rank_ci`` (bootstrap 95% CI, or ``None`` if ``n < 2``),
    ``fraction_rank_2`` (fraction of incorrect rows where the chosen
    strategy was EXACTLY second-best -- the direct test of the claim).
    """
    incorrect_ranks = [
        rank
        for sub in scored_submissions
        for correct, rank in zip(sub["per_row_correct"], sub["rank_of_chosen"])
        if not correct
    ]
    n = len(incorrect_ranks)
    if n == 0:
        return {
            "n_incorrect_rows": 0,
            "mean_rank_of_incorrect": float("nan"),
            "mean_rank_ci": None,
            "fraction_rank_2": float("nan"),
        }
    ranks_arr = np.asarray(incorrect_ranks, dtype=np.float64)
    ci = bootstrap_ci(ranks_arr) if n >= 2 else None
    return {
        "n_incorrect_rows": n,
        "mean_rank_of_incorrect": float(ranks_arr.mean()),
        "mean_rank_ci": ci,
        "fraction_rank_2": float(np.mean(ranks_arr == 2)),
    }


def _experience_ordinal(background: dict) -> Optional[int]:
    years = background.get("years_experience")
    return YEARS_EXPERIENCE_LEVELS.index(years) if years in YEARS_EXPERIENCE_LEVELS else None


def _llm_familiarity_ordinal(background: dict) -> Optional[int]:
    fam = background.get("llm_familiarity")
    return LLM_FAMILIARITY_LEVELS.index(fam) if fam in LLM_FAMILIARITY_LEVELS else None


def study_summary(submissions: list[dict], P: np.ndarray, p: Optional[np.ndarray] = None) -> dict:
    """Full descriptive analysis of the human-study arm.

    Parameters
    ----------
    submissions : list of dict
        RAW submission records (as read from ``study.storage.SubmissionStore``),
        each with at least ``mapping``, ``confidence`` (length-N list of
        1-5 ints), and ``background`` (dict). Each is scored internally
        via :func:`score_submission` -- callers never need to pre-score.
    P : ndarray, shape (N, K)
        The hidden matrix (``recovery.envs.canonical.MATRIX`` for the
        canonical study).
    p : ndarray, optional
        Type prior for scoring. Defaults to uniform.

    Returns
    -------
    dict -- see the module-level fields documented inline below. Every
    bootstrap CI is ``None`` (not a crash, not a fabricated interval) when
    ``n < 2``, per Task 11's ``study_summary handles n=1`` requirement.
    """
    N = P.shape[0]
    scored = [score_submission(s["mapping"], P, p) for s in submissions]
    n = len(scored)
    qs = np.array([s["q_count"] for s in scored], dtype=np.float64)

    q_ci = bootstrap_ci(qs, statistic=np.mean) if n >= 2 else None

    histogram = {int(k): int(np.sum(qs == k)) for k in range(N + 1)}

    per_type_correct = np.zeros(N, dtype=np.float64)
    for s in scored:
        per_type_correct += np.asarray(s["per_row_correct"], dtype=np.float64)
    per_type_accuracy = {TYPE_NAMES[t]: float(per_type_correct[t] / n) if n else float("nan") for t in range(N)}

    second_best = second_best_error_test(scored)

    all_confidence: list[float] = []
    all_correct: list[float] = []
    for sub, sc in zip(submissions, scored):
        conf = sub.get("confidence") or []
        for c, correct in zip(conf, sc["per_row_correct"]):
            all_confidence.append(float(c))
            all_correct.append(1.0 if correct else 0.0)
    confidence_correctness_corr = bootstrap_correlation_ci(all_confidence, all_correct)

    exp_years = [_experience_ordinal(s.get("background") or {}) for s in submissions]
    exp_pairs = [(e, q) for e, q in zip(exp_years, qs) if e is not None]
    experience_q_corr = (
        bootstrap_correlation_ci([e for e, _ in exp_pairs], [q for _, q in exp_pairs]) if len(exp_pairs) >= 3 else None
    )

    fam = [_llm_familiarity_ordinal(s.get("background") or {}) for s in submissions]
    fam_pairs = [(f, q) for f, q in zip(fam, qs) if f is not None]
    familiarity_q_corr = (
        bootstrap_correlation_ci([f for f, _ in fam_pairs], [q for _, q in fam_pairs]) if len(fam_pairs) >= 3 else None
    )

    return {
        "n": n,
        "q_mean": float(qs.mean()) if n else float("nan"),
        "q_median": float(np.median(qs)) if n else float("nan"),
        "q_min": float(qs.min()) if n else float("nan"),
        "q_max": float(qs.max()) if n else float("nan"),
        "q_mean_ci": q_ci,
        "q_histogram": histogram,
        "per_type_accuracy": per_type_accuracy,
        "second_best_error_test": second_best,
        "confidence_correctness_correlation": confidence_correctness_corr,
        "experience_q_correlation": experience_q_corr,
        "llm_familiarity_q_correlation": familiarity_q_corr,
        "scored_submissions": scored,
    }


def q_prior(submissions: list[dict]) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Build an empirical-distribution sampler over ``q_count`` suitable
    for feeding Task 10's crossover predictor
    (``recovery.theory.crossover.predict_crossover``), so a downstream
    caller (``experiments/study_analysis.py``) can report ``T*`` as a
    DISTRIBUTION (one draw per sampled ``q``) rather than a single point.

    Parameters
    ----------
    submissions : list of dict
        Already-SCORED submissions -- each dict must carry a ``q_count``
        key (i.e. the output of :func:`score_submission`, or
        ``study_summary(...)["scored_submissions"]``). This function does
        not score anything itself; it only resamples an already-measured
        empirical distribution.

    Returns
    -------
    callable ``sampler(rng, n) -> ndarray[int]``
        Draws ``n`` i.i.d. samples of ``q`` from the empirical
        distribution (simple categorical resampling WITH replacement --
        this is Monte Carlo resampling of a finite empirical sample, not
        a smoothed/parametric density estimate; see
        ``docs/experiments/human_study.md`` for why that distinction
        matters at this study's sample size). The raw observed values are
        also exposed as ``sampler.empirical_values`` for callers that want
        the PMF directly instead of resampling it.

    Raises
    ------
    ValueError
        If ``submissions`` is empty.
    """
    if not submissions:
        raise ValueError("q_prior requires at least one scored submission")
    qs = np.array([s["q_count"] for s in submissions], dtype=np.int64)

    def sampler(rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.choice(qs, size=n, replace=True)

    sampler.empirical_values = qs  # type: ignore[attr-defined]
    return sampler


def q_prior_crossover_distribution(
    submissions: list[dict],
    P: np.ndarray,
    c: float,
    form: str = "kl",
    n_draws: int = 2000,
    seed: int = 0,
) -> dict:
    """Feed Task 10's crossover predictor from the empirical ``q`` PRIOR
    (:func:`q_prior`), rather than from individual real mappings, so the
    predictor can be asked "what does the empirically MEASURED distribution
    of designer quality imply about ``T*``," as a distribution.

    This is deliberately distinct from
    ``experiments/study_analysis.py:crossover_budget_analysis``, which
    feeds Task 10's predictor each REAL submission's actual mapping
    directly (exact -- no modeling assumption needed, because the real
    mapping fully determines ``static_regret_rate``). Here, ``q_prior``
    only resamples the scalar ``q_count``, which does not by itself
    determine a mapping (two mappings with the same ``q`` can have
    different regret rates depending on how bad the wrong rows are), so
    turning a resampled ``q`` back into a mapping requires one explicit,
    documented modeling assumption: each of the ``N - q`` "wrong" rows is
    assigned that row's RANK-2 (second-best) strategy, per the
    pre-registered second-best-error claim
    (``study/preregistration.md``, RQ3; :func:`second_best_error_test`)
    rather than an arbitrary wrong choice. Which ``q`` rows are "correct"
    is chosen uniformly at random per draw.

    Parameters
    ----------
    submissions : list of dict
        Already-scored submissions, each carrying a ``q_count`` key (see
        :func:`q_prior`).
    P : ndarray, shape (N, K)
        Hidden success-probability matrix.
    c : float
        Task 10's already-calibrated constant -- reused, never refit here.
    form : {"kl", "gap"}, default="kl"
        Passed through to :func:`recovery.theory.crossover.predict_crossover`.
    n_draws : int, default=2000
        Number of synthetic mappings resampled from the empirical q prior.
    seed : int, default=0
        Seed for the resampling RNG (deterministic and reproducible).

    Returns
    -------
    dict with keys:
        ``T_star_samples`` (ndarray, length ``n_draws``): predicted
            crossover horizon for each synthetic mapping (``math.inf``
            entries possible for a perfect, ``q == N`` draw).
        ``T_star_median`` (float)
        ``T_star_ci`` (tuple or None): bootstrap 95% CI over the FINITE
            samples, via :func:`recovery.metrics.bootstrap.bootstrap_ci`;
            ``None`` if fewer than 2 finite samples were drawn.
        ``n_finite`` (int): number of draws that did not hit ``math.inf``.
        ``n_input_submissions`` (int)
        ``n_draws`` (int)

    Raises
    ------
    ValueError
        If ``submissions`` is empty (propagated from :func:`q_prior`).
    """
    P = np.asarray(P, dtype=np.float64)
    N = P.shape[0]
    sampler = q_prior(submissions)
    rng = np.random.default_rng(seed)
    qs = sampler(rng, n_draws)

    row_rank1 = np.argmax(P, axis=1)
    row_rank2 = np.argsort(-P, axis=1)[:, 1]

    T_stars = np.empty(n_draws, dtype=np.float64)
    for i, q in enumerate(qs):
        q_i = int(np.clip(q, 0, N))
        mapping = row_rank2.copy()
        if q_i > 0:
            correct_rows = rng.choice(N, size=q_i, replace=False)
            mapping[correct_rows] = row_rank1[correct_rows]
        T_star, _truncated = predict_crossover(P, mapping, form=form, c=c)
        T_stars[i] = T_star

    finite = T_stars[np.isfinite(T_stars)]
    ci = bootstrap_ci(finite) if finite.size >= 2 else None
    return {
        "T_star_samples": T_stars,
        "T_star_median": float(np.median(T_stars)),
        "T_star_ci": ci,
        "n_finite": int(finite.size),
        "n_input_submissions": len(submissions),
        "n_draws": n_draws,
    }


def compare_human_llm_q(human_q: Sequence[float], llm_q: Sequence[float]) -> dict:
    """Human-vs-LLM designer comparison: Cliff's delta (effect size) with
    a bootstrap CI, NOT a p-value (project-wide convention -- see
    docs/statistics.md; also reuses ``recovery.metrics.bootstrap.cliffs_delta``
    exactly, never re-implemented).

    Returns
    -------
    dict with ``delta``, ``delta_ci`` (bootstrap 95% CI on Cliff's delta,
    or ``None`` if either sample has fewer than 2 observations),
    ``magnitude`` (Romano et al. qualitative label), ``n_human``, ``n_llm``.
    """
    human_arr = np.asarray(human_q, dtype=np.float64)
    llm_arr = np.asarray(llm_q, dtype=np.float64)
    delta = cliffs_delta(human_arr, llm_arr)

    delta_ci = None
    if human_arr.size >= 2 and llm_arr.size >= 2:
        rng = np.random.default_rng(0)
        n_resamples = 10_000
        idx_h = rng.integers(0, human_arr.size, size=(n_resamples, human_arr.size))
        idx_l = rng.integers(0, llm_arr.size, size=(n_resamples, llm_arr.size))
        deltas = np.array(
            [cliffs_delta(human_arr[idx_h[i]], llm_arr[idx_l[i]]) for i in range(n_resamples)], dtype=np.float64
        )
        delta_ci = (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))

    return {
        "delta": delta,
        "delta_ci": delta_ci,
        "magnitude": magnitude(delta),
        "n_human": int(human_arr.size),
        "n_llm": int(llm_arr.size),
    }


def _mapping_matrix(submissions: list[dict], n_types: int) -> np.ndarray:
    """``(n_submissions, n_types)`` array of chosen strategy indices."""
    return np.array([s["mapping"] for s in submissions], dtype=np.int64).reshape(len(submissions), n_types)


def fleiss_kappa(submissions: list[dict], n_types: int, n_strategies: int) -> Optional[float]:
    """Fleiss' kappa for inter-designer agreement: how much more do
    independent designers agree with each other, choosing among
    ``n_strategies`` strategies for each of ``n_types`` failure types,
    than chance agreement would predict?

    Standard formula (Fleiss, 1971): each failure type is an "item," each
    submission a "rater," each strategy a "category." Returns ``None`` if
    fewer than 2 submissions are available (agreement is undefined for a
    single rater) -- never fabricates a value.

    Returns
    -------
    float or None
        In roughly ``[-1, 1]``; ``1`` = perfect agreement, ``0`` =
        agreement no better than chance, negative = worse than chance.
    """
    n = len(submissions)
    if n < 2:
        return None
    mapping = _mapping_matrix(submissions, n_types)  # (n, n_types)

    # counts[t, k] = number of submissions choosing strategy k for type t
    counts = np.zeros((n_types, n_strategies), dtype=np.float64)
    for t in range(n_types):
        for k in range(n_strategies):
            counts[t, k] = np.sum(mapping[:, t] == k)

    p_j = counts.sum(axis=0) / (n_types * n)  # overall proportion choosing each category
    P_i = (np.sum(counts * counts, axis=1) - n) / (n * (n - 1))  # per-item agreement
    P_bar = float(np.mean(P_i))
    P_e = float(np.sum(p_j * p_j))
    if np.isclose(1.0 - P_e, 0.0):
        return None  # denominator undefined (every rater picked the same category for everything)
    return (P_bar - P_e) / (1.0 - P_e)


def pairwise_agreement(submissions: list[dict], n_types: int) -> Optional[dict]:
    """Average fraction of failure-type rows on which two INDEPENDENT
    submissions agree (chose the same strategy), averaged over every
    pair of submissions -- a more intuitive, less parametric companion
    to :func:`fleiss_kappa`.

    Returns
    -------
    dict or None
        ``{"mean_agreement": float, "ci": (lo, hi) or None, "n_pairs": int}``,
        or ``None`` if fewer than 2 submissions.
    """
    n = len(submissions)
    if n < 2:
        return None
    mapping = _mapping_matrix(submissions, n_types)
    agreements = []
    for i in range(n):
        for j in range(i + 1, n):
            agreements.append(float(np.mean(mapping[i] == mapping[j])))
    agreements = np.asarray(agreements, dtype=np.float64)
    ci = bootstrap_ci(agreements) if agreements.size >= 2 else None
    return {"mean_agreement": float(agreements.mean()), "ci": ci, "n_pairs": int(agreements.size)}


def entropy_per_type(submissions: list[dict], n_types: int, n_strategies: int) -> dict:
    """Shannon entropy (base 2, bits) of the distribution of CHOSEN
    strategies for each failure type across submissions -- a
    per-failure-type measure of how much designers disagree with each
    other about that specific type (independent of whether they were
    RIGHT; a type could have low entropy because everyone agrees on the
    same WRONG strategy). ``0`` bits = every submission chose the same
    strategy; ``log2(n_strategies)`` bits = uniform disagreement across
    every strategy.

    Returns
    -------
    dict[str, float]
        Keyed by ``recovery.envs.canonical.TYPE_NAMES``. Empty
        submissions list -> every entropy is ``NaN`` (undefined, never 0).
    """
    n = len(submissions)
    if n == 0:
        return {name: float("nan") for name in TYPE_NAMES}
    mapping = _mapping_matrix(submissions, n_types)
    out = {}
    for t in range(n_types):
        counts = np.bincount(mapping[:, t], minlength=n_strategies).astype(np.float64)
        probs = counts / counts.sum()
        nonzero = probs[probs > 0]
        out[TYPE_NAMES[t]] = float(-np.sum(nonzero * np.log2(nonzero)))
    return out
