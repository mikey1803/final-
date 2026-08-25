"""Static perturbations of a success-probability matrix, for building a
FAMILY of independent stationary worlds around one hand-authored matrix
(Task 12's SUCCESS_MATRIX sensitivity/robustness analysis).

Distinct from ``recovery.envs.drift``, which perturbs a matrix WITHIN one
simulation run (an ``Environment`` whose active matrix changes over
episodes). Every function here is a pure, one-shot transformation: given
a base matrix and an explicit ``numpy.random.Generator``, return ONE new,
independent, still-stationary matrix. Building a FAMILY of worlds means
calling these functions repeatedly with fresh draws from the same (or a
seeded, reproducible) generator -- there is no module-level RNG anywhere
in this file, matching ``recovery.envs.drift``'s convention.

Architecture constraint: this module may import only ``numpy`` -- no
``recovery.core``, ``recovery.metrics``, ``recovery.policies``,
``recovery.theory``, or ``experiments`` -- matching
``recovery.envs.drift``'s and ``recovery.envs.canonical``'s import
discipline, so the analytic/kernel layers can validate against these
matrices without any circular dependency.

Two notions of "hardest failure type" -- do not conflate them
--------------------------------------------------------------
``hardest_type_by_ceiling`` below computes a MATRIX-ONLY property: the
type whose best-available strategy still has the lowest success
probability (``argmin_t max_k P[t, k]``). This is NOT the same construct
as Task 11's finding that ``malformed_output`` has the lowest BLIND-
DESIGNER identification accuracy across three independent LLM families --
that is a property of how designers read the plain-English descriptions,
not a property of ``P`` alone, and cannot be recomputed for a perturbed
matrix without running a new blind-design elicitation study against it
(out of scope here -- blind designers must never see ``P``, so a new
matrix requires new participants). See
``docs/experiments/matrix_robustness.md`` for the full discussion,
including that ``malformed_output`` is in fact NOT the ceiling-hardest
type in the original canonical ``MATRIX`` (``tool_timeout`` is, at 0.70).
"""

from typing import Optional

import numpy as np

__all__ = [
    "gaussian_perturb",
    "permute_row_entries",
    "hardest_type_by_ceiling",
    "structured_uniform_difficulty",
    "structured_single_dominant_strategy",
    "structured_malformed_not_hardest",
]


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Minimal shape/range validation shared by every function here.

    Reimplemented locally (rather than imported) because this module may
    not import ``recovery.envs.matrix_env`` or any metrics module -- see
    the module docstring's architecture constraint. Identical in spirit
    to ``recovery.envs.drift._validate_matrix``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("matrix must not contain NaN or Inf values")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(
            f"matrix must contain only values in [0, 1], got min={matrix.min()!r}, max={matrix.max()!r}"
        )
    return matrix


def gaussian_perturb(P: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Scheme 1 (Task 12): independent truncated-Gaussian noise per entry.

    ``P'[t, k] = clip(P[t, k] + N(0, sigma^2), 0, 1)`` -- each of the
    ``N*K`` entries perturbed independently and identically distributed,
    then clipped (not resampled) at the ``[0, 1]`` boundary, i.e. this is
    a truncated-at-the-boundary Gaussian in the sense of clipping the
    realized value, not a rejection-sampled truncated normal distribution
    -- the distinction matters near the boundary at large ``sigma``: mass
    that would fall outside ``[0, 1]`` piles up exactly AT the boundary
    rather than being redistributed inside it, which is the standard,
    simplest interpretation of "clipped to [0,1]" and is stated explicitly
    here since the task specification did not give a formula.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Base success-probability matrix.
    sigma : float
        Standard deviation of the per-entry noise. Must be ``>= 0``.
        ``sigma=0`` returns a copy of ``P`` unchanged (identity
        perturbation, useful as a sanity-check baseline).
    rng : numpy.random.Generator
        Source of randomness. No module-level RNG -- the caller owns and
        seeds this.

    Returns
    -------
    ndarray, shape (N, K)
        A new, independent matrix (``P`` itself is never mutated).

    Raises
    ------
    ValueError
        If ``P`` fails validation, or ``sigma < 0``.
    """
    P = _validate_matrix(P)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma!r}")
    if sigma == 0:
        return P.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=P.shape)
    return np.clip(P + noise, 0.0, 1.0)


def permute_row_entries(
    P: np.ndarray,
    fraction: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scheme 2 (Task 12): permute a controlled fraction of each row's entries.

    Operationalization (the task specification described the desired
    BEHAVIOR -- "rank-breaking permutations should be able to change the
    optimal strategy, rank-preserving ones should not" -- but not an
    algorithm; this is the exact, explicit definition used here):

    - ``mode="rank_breaking"``: for each row ``t``, choose
      ``round(fraction * K)`` column positions uniformly at random from
      ALL ``K`` positions (the row's current argmax position is eligible
      to be chosen), and permute the values at those positions among
      themselves. Because the argmax position is eligible, this CAN
      relocate the row's maximum value to a different strategy -- i.e.
      change which strategy is optimal for that failure type.
    - ``mode="rank_preserving"``: identical, except the row's CURRENT
      argmax position is explicitly EXCLUDED from the eligible set before
      choosing positions to permute. The row's maximum value therefore
      never moves, so the optimal strategy for that type is unchanged BY
      CONSTRUCTION, regardless of ``fraction`` -- this is not an
      empirical property to be checked, it is guaranteed by this
      function's definition, and downstream analysis should treat
      ``mode="rank_preserving"`` as a built-in negative control, not as
      an independent empirical result.

    A row with fewer than 2 eligible chosen positions (e.g. ``fraction``
    small enough that ``round(fraction * K) < 2``) is left unpermuted for
    that row -- there is nothing to swap with a single position. **This
    is a real, reportable edge case for ``K=6`` strategies**:
    ``fraction=0.1`` gives ``round(0.6) = 1`` selected position for every
    row, which permutes NOTHING (every row is returned unchanged) --
    ``fraction=0.1`` is a de facto no-op perturbation at this action-space
    size, not a "mild" one. The smallest ``fraction`` that permutes
    anything at ``K=6`` is ``fraction >= 0.25`` (``round(1.5) = 2``,
    Python/NumPy round-half-to-even, so exactly ``0.25`` rounds to ``2``).
    This is documented here rather than silently patched (e.g. by forcing
    a minimum of 2 swapped positions), because silently changing what
    "fraction" means would make the reported perturbation magnitude not
    match what was actually done to the matrix.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Base success-probability matrix.
    fraction : float
        Fraction of each row's ``K`` entries to select for permutation,
        in ``[0, 1]``. Converted to a position count via
        ``round(fraction * K)`` (NumPy round-half-to-even), then clipped
        to the number of eligible positions for the given ``mode``.
    mode : {"rank_preserving", "rank_breaking"}
        See above.
    rng : numpy.random.Generator
        Source of randomness for both position selection and the
        permutation itself.

    Returns
    -------
    ndarray, shape (N, K)
        A new, independent matrix (``P`` itself is never mutated). The
        MULTISET of values in every row is exactly preserved (a
        permutation only reassigns which column holds which value) --
        only ``policy_rate``/``oracle_rate`` for policies that reference
        the moved positions can change, never ``random_rate`` (the row
        mean is permutation-invariant).

    Raises
    ------
    ValueError
        If ``P`` fails validation, ``fraction`` is outside ``[0, 1]``, or
        ``mode`` is not one of the two recognized values.
    """
    P = _validate_matrix(P)
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")
    if mode not in ("rank_preserving", "rank_breaking"):
        raise ValueError(f"mode must be 'rank_preserving' or 'rank_breaking', got {mode!r}")

    new = P.copy()
    n_types, n_strats = P.shape
    n_select = int(round(fraction * n_strats))

    for t in range(n_types):
        row = new[t]
        argmax_idx = int(np.argmax(row))
        if mode == "rank_preserving":
            eligible = [k for k in range(n_strats) if k != argmax_idx]
        else:
            eligible = list(range(n_strats))

        n_this_row = min(n_select, len(eligible))
        if n_this_row < 2:
            continue  # nothing to swap with a single (or zero) selected position

        chosen = rng.choice(np.asarray(eligible, dtype=np.int64), size=n_this_row, replace=False)
        values = row[chosen].copy()
        row[chosen] = rng.permutation(values)

    return new


def hardest_type_by_ceiling(P: np.ndarray, type_names: Optional[list] = None):
    """The type with the lowest ACHIEVABLE success ceiling, ``argmin_t max_k P[t, k]``.

    This is a property of the matrix alone -- see the module docstring's
    "Two notions of hardest failure type" section for why this is NOT
    the same construct as Task 11's blind-designer-accuracy finding.

    Parameters
    ----------
    P : ndarray, shape (N, K)
        Success-probability matrix.
    type_names : list of str, optional
        If given, also returns the name at the resolved index.

    Returns
    -------
    int, or (int, str)
        The row index of the ceiling-hardest type (ties broken by
        ``numpy.argmin``'s first-minimum convention), and its name if
        ``type_names`` was given.

    Raises
    ------
    ValueError
        If ``P`` fails validation.
    """
    P = _validate_matrix(P)
    ceilings = P.max(axis=1)
    idx = int(np.argmin(ceilings))
    if type_names is not None:
        return idx, type_names[idx]
    return idx


# ======================================================================
# Scheme 3 (Task 12): structured alternative matrices, hand-authored.
# ======================================================================
#
# All three share the canonical (7, 6) shape (TYPE_NAMES x STRATEGY_NAMES
# order, see recovery.envs.canonical) so BLIND_RULE and every existing
# analytic/kernel function apply unmodified. Every value below was chosen
# by hand to satisfy the STRUCTURAL property named by the function, not
# sampled -- consistent with the task specification's "construct ... by
# hand."


def structured_uniform_difficulty() -> np.ndarray:
    """Scheme 3(a): every failure type has an IDENTICAL achievable ceiling.

    Construction: one fixed 6-value "difficulty profile" (max=0.75,
    min=0.20, matching the canonical matrix's own overall value range),
    cyclically rotated by one position per row across all 7 rows. Every
    row is therefore a permutation of the exact same multiset of values
    -- same max, same min, same shape -- so
    ``hardest_type_by_ceiling`` is undefined-by-tie (every row ties at
    the same ceiling) rather than picking out any one type as hardest.
    """
    profile = np.array([0.30, 0.45, 0.55, 0.65, 0.75, 0.20], dtype=np.float64)
    n_types = 7
    rows = [np.roll(profile, shift) for shift in range(n_types)]
    return np.stack(rows, axis=0)


def structured_single_dominant_strategy() -> np.ndarray:
    """Scheme 3(b): one strategy (index 4, "verify") is optimal for every type.

    Every row has strategy 4 fixed at 0.85 (comfortably the row max);
    the other 5 strategies vary by row (hand-chosen, in [0.10, 0.65]) so
    the matrix is not otherwise degenerate -- only the "which strategy
    wins" dimension collapses to a single answer for all 7 types.
    """
    return np.array(
        [
            [0.40, 0.30, 0.55, 0.20, 0.85, 0.35],
            [0.45, 0.35, 0.50, 0.25, 0.85, 0.40],
            [0.20, 0.55, 0.60, 0.30, 0.85, 0.45],
            [0.15, 0.40, 0.35, 0.50, 0.85, 0.45],
            [0.35, 0.45, 0.50, 0.30, 0.85, 0.25],
            [0.10, 0.50, 0.25, 0.60, 0.85, 0.40],
            [0.15, 0.40, 0.20, 0.35, 0.85, 0.65],
        ],
        dtype=np.float64,
    )


def structured_malformed_not_hardest() -> np.ndarray:
    """Scheme 3(c): a matrix where ``malformed_output`` is NOT the
    ceiling-hardest type -- in fact made comfortably one of the EASIEST
    (ceiling 0.95), with ``ambiguous_instruction`` made the ceiling-hardest
    type instead (ceiling 0.20).

    Starts from the canonical ``MATRIX`` (see
    ``recovery.envs.canonical``) and hand-edits exactly two rows (index 4,
    ``malformed_output``, and index 6, ``ambiguous_instruction``) so the
    comparison to the canonical matrix's other 5 unchanged rows is as
    direct as possible. Every other row is copied byte-for-byte from
    ``MATRIX``.

    Note (see module docstring): under the ACHIEVABLE-CEILING definition
    of "hardest" used throughout this module, the ORIGINAL canonical
    ``MATRIX`` already does NOT have ``malformed_output`` as its hardest
    type -- ``tool_timeout`` is (ceiling 0.70). This function still
    constructs the requested "malformed_output is not hardest" matrix
    for completeness, and to test the ceiling-hardest LABEL's sensitivity
    to authored values, but see
    ``docs/experiments/matrix_robustness.md`` for why this does not, and
    cannot, directly re-test Task 11's distinct designer-accuracy finding.
    """
    P = np.array(
        [
            [0.70, 0.50, 0.60, 0.15, 0.30, 0.10],  # tool_timeout (unchanged)
            [0.65, 0.55, 0.75, 0.20, 0.35, 0.25],  # tool_error (unchanged)
            [0.15, 0.60, 0.75, 0.45, 0.40, 0.55],  # wrong_tool_selected (unchanged)
            [0.10, 0.45, 0.30, 0.55, 0.75, 0.50],  # hallucinated_output (unchanged)
            [0.50, 0.60, 0.70, 0.45, 0.95, 0.40],  # malformed_output -- ceiling raised to 0.95
            [0.10, 0.55, 0.25, 0.80, 0.40, 0.50],  # context_loss (unchanged)
            [0.05, 0.15, 0.10, 0.10, 0.05, 0.20],  # ambiguous_instruction -- ceiling lowered to 0.20
        ],
        dtype=np.float64,
    )
    return P
