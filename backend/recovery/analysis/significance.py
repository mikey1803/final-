"""Task 13: real-sample inferential hypothesis testing.

This module is deliberately SEPARATE from ``recovery/metrics/bootstrap.py``,
whose own docstring states "no p-values, t-tests, Holm correction, or
significance testing here or anywhere in this package" -- a documented,
previously-validated design decision (see ``docs/statistics.md``) that
this task's own rules ("do not rewrite working code without evidence of a
flaw") forbid overturning.

That design decision is specific to SIMULATION-SEED data: ``docs/statistics.md``
explains that a p-value computed across repeated re-runs of the SAME
deterministic-given-its-seed program is degenerate (more seeds trivially
drive it toward zero for any nonzero true effect, however small).

This module exists for the ONE category of data in this repository where
that objection does not apply: REAL, finite samples from an external,
non-deterministic process this codebase does not control -- specifically
the LLM blind-design study's API responses (``experiments/llm_designers.py``,
Task 11). Each sample there is one real network call to a real model,
not a re-run of a known program with a different seed; running "more
samples" costs real API quota and time, and the resulting p-value answers
a legitimate question (does this model's per-type accuracy differ from
chance/from its own other types), not merely "how many times did we
re-run our own simulator."

Multiple-comparison correction: reuses ``src.evaluation.stats_utils.holm_bonferroni``
directly rather than reimplementing Holm's step-down procedure --
that function is a generic, always-mathematically-valid algorithm
independent of what feeds it; what ``docs/statistics.md`` deprecates is
the PRACTICE of computing p-values over simulation seeds upstream of it,
not the correction procedure itself. See ``docs/statistics_registry.md``
for the full multiple-comparison-family policy this module's output feeds
into.
"""

from typing import Sequence

import numpy as np
from scipy.stats import binomtest, chi2, mannwhitneyu

from src.evaluation.stats_utils import holm_bonferroni  # noqa: E402 -- reused, not reimplemented

__all__ = [
    "cochrans_q_test",
    "cohens_h",
    "per_type_vs_rest_binomial_tests",
    "n_distinct_mappings",
    "variant_q_comparison_test",
]


def cochrans_q_test(binary_matrix: np.ndarray) -> dict:
    """Cochran's Q test: are the per-column success probabilities equal,
    for a MATCHED binary design (same ``n`` subjects each answering all
    ``k`` conditions)?

    This is the statistically appropriate omnibus test here (not a plain
    chi-square goodness-of-fit test), because each LLM sample answers all
    7 failure types in the SAME API call -- the 7 per-type correctness
    outcomes from one sample are paired, not independent draws, exactly
    the repeated-measures binary design Cochran's Q is built for (it
    generalizes McNemar's test from k=2 to k>=2 matched conditions).

    Parameters
    ----------
    binary_matrix : ndarray, shape (n_subjects, k_conditions), dtype bool/int
        ``binary_matrix[i, j] = 1`` iff subject ``i`` succeeded on
        condition ``j``.

    Returns
    -------
    dict with ``Q`` (test statistic), ``df`` (``k - 1``), ``p_value``
    (upper-tail chi-square survival, asymptotic approximation), ``n``,
    ``k``, and ``effect_size_v`` (a Cramer's-V-style normalization of
    ``Q``, ``sqrt(Q / (n * (k - 1)))`` -- there is no single universally
    standard effect size for Cochran's Q; this is a transparent, clearly
    labeled derivation from the same chi-square statistic, not a claim
    of a canonical "Cochran's Q effect size").

    Notes
    -----
    The chi-square approximation to Cochran's Q's null distribution is
    asymptotic; with ``n`` in the 19-21 range and ``k=7`` used here
    (``n*k`` ~= 133-147), this sits within commonly-cited rules of thumb
    for the approximation's adequacy, but this is not a guarantee at any
    given alpha -- reported honestly, not as an exact test.
    """
    X = np.asarray(binary_matrix, dtype=np.float64)
    n, k = X.shape
    if n < 2 or k < 2:
        raise ValueError(f"Cochran's Q requires n>=2 subjects and k>=2 conditions, got n={n}, k={k}")
    col_totals = X.sum(axis=0)
    row_totals = X.sum(axis=1)
    grand_total = X.sum()
    numerator = (k - 1) * (k * np.sum(col_totals**2) - grand_total**2)
    denominator = k * np.sum(row_totals) - np.sum(row_totals**2)
    if denominator == 0:
        # Every subject scored identically on every condition (all-correct
        # or all-incorrect rows only) -- Q is undefined, not zero; report
        # honestly rather than fabricating a p-value.
        return {"Q": float("nan"), "df": k - 1, "p_value": float("nan"), "n": n, "k": k,
                "effect_size_v": float("nan"), "degenerate": True}
    Q = float(numerator / denominator)
    df = k - 1
    p_value = float(chi2.sf(Q, df))
    effect_size_v = float(np.sqrt(max(Q, 0.0) / (n * df))) if n * df > 0 else float("nan")
    return {"Q": Q, "df": df, "p_value": p_value, "n": n, "k": k,
            "effect_size_v": effect_size_v, "degenerate": False}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h: the standard effect size for a difference between two
    proportions, ``2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))``. Conventional
    magnitude thresholds (Cohen 1988): ~0.2 small, ~0.5 medium, ~0.8 large."""
    return float(2 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0))) - 2 * np.arcsin(np.sqrt(np.clip(p2, 0.0, 1.0))))


def per_type_vs_rest_binomial_tests(binary_matrix: np.ndarray, type_names: Sequence[str]) -> list[dict]:
    """For each of the ``k`` conditions (failure types), an exact binomial
    test of that condition's success count against the POOLED success
    rate of the OTHER ``k-1`` conditions (excluding the type itself, to
    avoid testing a type's rate against a pooled rate that already
    includes it) -- the post-hoc test for "is failure type t specifically
    harder/easier than the rest," as distinct from Cochran's Q's omnibus
    "is there any per-type heterogeneity at all."

    Returns one dict per type (NOT yet Holm-corrected -- correction is
    applied by the caller across whatever family of tests is appropriate,
    per ``docs/statistics_registry.md``'s multiple-comparison policy),
    each with: ``type_name``, ``k_successes``, ``n``, ``accuracy``,
    ``pooled_rest_rate``, ``p_value`` (two-sided exact binomial),
    ``cohens_h`` (accuracy vs. pooled_rest_rate).
    """
    X = np.asarray(binary_matrix, dtype=np.float64)
    n, k = X.shape
    if k != len(type_names):
        raise ValueError(f"binary_matrix has {k} columns but {len(type_names)} type_names given")
    col_totals = X.sum(axis=0)
    grand_total = X.sum()
    rows = []
    for j, type_name in enumerate(type_names):
        k_successes = int(col_totals[j])
        rest_successes = grand_total - col_totals[j]
        rest_n = n * (k - 1)
        pooled_rest_rate = float(rest_successes / rest_n) if rest_n > 0 else float("nan")
        accuracy = float(k_successes / n)
        test = binomtest(k_successes, n, pooled_rest_rate, alternative="two-sided") if 0 < pooled_rest_rate < 1 else None
        rows.append({
            "type_name": type_name,
            "k_successes": k_successes,
            "n": n,
            "accuracy": accuracy,
            "pooled_rest_rate": pooled_rest_rate,
            "p_value": float(test.pvalue) if test is not None else float("nan"),
            "cohens_h": cohens_h(accuracy, pooled_rest_rate),
        })
    return rows


def n_distinct_mappings(mappings: Sequence[Sequence[int]]) -> int:
    """Number of DISTINCT mapping tuples among ``mappings`` -- required
    reporting alongside every LLM-family sample count per Task 13's
    instruction (``google-gemini`` produced only 2 distinct mappings
    across 21 API calls; any resampling-based CI on such data must be
    flagged as degenerate, since a bootstrap of a near-point-mass
    distribution cannot represent genuine sampling uncertainty)."""
    return len({tuple(int(x) for x in m) for m in mappings})


def variant_q_comparison_test(q_a: Sequence[float], q_b: Sequence[float]) -> dict:
    """Task 15: does per-sample design quality ``q`` (0-7 correct rows out
    of 7, one value per independent API call) differ between two prompt
    variants -- e.g. the primary instrument vs. a paraphrase, within one
    LLM family?

    A two-sided Mann-Whitney U test on the raw ``q`` values (NOT a
    per-(sample, failure-type) trial-level test like
    :func:`per_type_vs_rest_binomial_tests`) -- because the 7 per-type
    answers within one sample are correlated (one designer's coherent
    choice, the same reason :func:`cochrans_q_test` treats them as a
    matched design), while different SAMPLES under different prompt
    variants are genuinely independent API calls. Treating the 7 answers
    within a sample as 7 independent trials here would silently inflate
    the effective ``n`` and understate the true p-value. Mann-Whitney is
    used (rather than a t-test) because ``q`` is a small-range, bounded
    integer count and Task 15 runs at reduced ``n`` -- nonparametric is
    the more conservative, defensible choice at this sample size, and
    scipy's ``method="auto"`` uses the exact permutation null when ``n``
    is small enough for it to be tractable.

    Falls back to reporting ``p_value=nan`` (never raises, never
    fabricates a value) when either group has fewer than 2 observations.

    Multiple-comparison correction is the CALLER's responsibility (per
    ``docs/statistics_registry.md``'s policy: apply Holm-Bonferroni across
    whichever family of comparisons is being run together -- for Task 15
    that is the 3 pairwise variant comparisons within one LLM family, an
    identical correction *scope* to Task 13's "within-family across its 7
    failure types," just a different set of 3 vs. 7 comparisons).
    """
    a = np.asarray(q_a, dtype=np.float64)
    b = np.asarray(q_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return {
            "u_statistic": float("nan"), "p_value": float("nan"), "n_a": int(a.size), "n_b": int(b.size),
            "median_a": float(np.median(a)) if a.size else float("nan"),
            "median_b": float(np.median(b)) if b.size else float("nan"),
        }
    result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return {
        "u_statistic": float(result.statistic), "p_value": float(result.pvalue),
        "n_a": int(a.size), "n_b": int(b.size),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
    }
