"""A taxonomy of non-stationary drift models over the (N, K) success matrix.

Task 6/7's non-stationary result used exactly one drift mechanism:
randomly permuting the strategy->probability mapping inside a subset of
rows (``PermutationDrift`` below reproduces that mechanism exactly). That
is the WORST case for a fixed mapping -- every strategy's rank can change
simultaneously and unpredictably. Real production drift is rarely this
adversarial: APIs degrade gradually, rankings shuffle only at the top,
migrations roll out over a window, and deployments are often announced in
advance. This module adds four more physically-motivated drift models
(``MonotoneDecayDrift``, ``PartialReorderDrift``, ``GradualRotationDrift``)
plus a signal decorator (``AnnouncedDrift``) so experiments can compare
adaptive methods against a REALISTIC drift taxonomy, not just the worst
case.

Architecture constraint: this module may import only ``numpy`` and
``recovery.core.protocols.Environment`` (for structural typing/duck-typing
reference) -- never ``recovery.metrics``, ``recovery.policies``,
``experiments``, or ``recovery.core.episode`` (the kernel). It defines no
module-level RNG: every source of randomness is an explicit
``numpy.random.Generator`` the caller supplies.

Every concrete drift model is a ``DriftingEnvironment`` subclass and
therefore satisfies ``recovery.core.protocols.Environment`` structurally
(``n_types``, ``n_strategies``, ``success_matrix()``, ``sample_reward()``,
``step()``) -- any of them can be dropped into
``recovery.core.episode.run_episodes`` in place of
``recovery.envs.matrix_env.MatrixEnvironment``.

Matrix history and complexity
------------------------------
``DriftingEnvironment`` never allocates one matrix per episode. It stores
a COMPACT history: one ``DriftEvent(episode, matrix)`` entry per actual
change, so history storage and the per-``step()`` bookkeeping cost are
both ``O(number of shifts)``, never ``O(n_episodes)``. The one class this
does not trivially apply to is ``GradualRotationDrift`` (drift is
continuous across its interpolation window, not a discrete jump) -- see
its docstring for how it stays compact without losing simulation
accuracy.
"""

from typing import NamedTuple, Sequence

import numpy as np

from recovery.core.protocols import Environment

__all__ = [
    "DriftEvent",
    "DriftingEnvironment",
    "PermutationDrift",
    "MonotoneDecayDrift",
    "PartialReorderDrift",
    "GradualRotationDrift",
    "AnnouncedDrift",
]


class DriftEvent(NamedTuple):
    """One compact matrix-history entry.

    Attributes
    ----------
    episode : int
        The episode index at which ``matrix`` became the active matrix
        (inclusive -- ``sample_reward`` calls for this episode and every
        later one, up to the next event, use ``matrix``).
    matrix : numpy.ndarray, shape (N, K)
        A defensive copy of the success matrix that became active.
    """

    episode: int
    matrix: np.ndarray


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Minimal shape/range validation shared by every drift model.

    Reimplemented locally (rather than imported) because this module may
    not import ``recovery.envs.matrix_env`` or any metrics module --
    see the module docstring's architecture constraint.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("matrix must not contain NaN or Inf values")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(
            f"matrix must contain only values in [0, 1], got min={matrix.min()!r}, "
            f"max={matrix.max()!r}"
        )
    return matrix


def _resolve_base_matrix(base) -> np.ndarray:
    """Accept either a raw ``(N, K)`` array or an ``Environment``-like
    object exposing ``success_matrix()`` (e.g.
    ``recovery.envs.matrix_env.MatrixEnvironment``) -- "wraps an existing
    stationary environment" is satisfied either way: if given an
    ``Environment``, its ``success_matrix()`` (already documented there as
    returning a defensive copy) becomes this drift model's pre-drift
    baseline.
    """
    if hasattr(base, "success_matrix"):
        return np.asarray(base.success_matrix(), dtype=np.float64)
    return np.asarray(base, dtype=np.float64)


class DriftingEnvironment:
    """Base class for every drift model in this module.

    Wraps a stationary base matrix (or an ``Environment`` exposing
    ``success_matrix()``) and maintains:

    - ``base_matrix()`` -- the ORIGINAL matrix, never mutated after
      construction (every method that could expose it returns a copy).
    - ``current_matrix()`` / ``success_matrix()`` -- the matrix currently
      active, for downstream analytic normalization (see
      ``recovery.metrics.drift_metrics``).
    - ``matrix_history()`` -- the compact list of ``DriftEvent`` change
      events (never one entry per episode).

    Subclasses implement ``_next_matrix(current, due_episode)``, called
    once per scheduled event, to compute the FRESH matrix that becomes
    active at that event (a fresh ``numpy`` array is always allocated --
    ``current`` is never mutated in place). ``GradualRotationDrift``
    overrides ``_advance`` directly instead, since its drift is continuous
    rather than event-scheduled (see its docstring).

    Parameters
    ----------
    base_matrix : numpy.ndarray or Environment
        The pre-drift success matrix, or an object exposing
        ``success_matrix()``.
    schedule : sequence of int, default=()
        Episode indices at which a new matrix becomes active. Need not be
        sorted or unique on input (sorted internally; duplicate entries
        are applied in ascending order, each producing its own history
        entry -- see "Composition order" below).

    Raises
    ------
    ValueError
        If the resolved base matrix is not 2-D, contains non-finite
        values, or contains values outside [0, 1].

    Composition order
    ------------------
    If several schedule entries share the same episode ("overlapping
    shifts"), they are applied in ascending-episode, then
    insertion-order, sequence -- each producing its own
    ``_next_matrix`` call (fed the PREVIOUS event's output) and its own
    compact history entry. A schedule entry at episode 0 (or any
    non-positive episode) is applied during ``__init__``, before the
    first simulated episode -- see "Edge cases" in
    ``docs/experiments/drift.md`` for why this means an episode-0 shift
    cannot be *announced* via ``AnnouncedDrift`` in episode 0's own
    ``Context`` (though ``current_matrix()``/``sample_reward()`` reflect
    it correctly from episode 0 onward) unless the model is wrapped in
    ``AnnouncedDrift`` before any ``step()`` call is made -- which
    ``AnnouncedDrift`` handles automatically at construction time.
    """

    def __init__(self, base_matrix, schedule: Sequence[int] = ()) -> None:
        resolved = _validate_matrix(_resolve_base_matrix(base_matrix))
        self._base_matrix = resolved.copy()
        self.n_types: int = resolved.shape[0]
        self.n_strategies: int = resolved.shape[1]
        self._pending: list = sorted(int(e) for e in schedule)
        self._current = self._base_matrix.copy()
        self._history: list = [DriftEvent(0, self._current.copy())]
        self._advance(-1)  # apply anything scheduled at/before episode 0

    def _next_matrix(self, current: np.ndarray, due_episode: int) -> np.ndarray:
        """Compute the fresh matrix for a scheduled event. Must be
        overridden by every event-driven subclass (everything except
        ``GradualRotationDrift``, which overrides ``_advance`` instead).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _next_matrix or override _advance"
        )

    def _advance(self, completed_episode: int) -> None:
        """Apply every pending schedule entry now due.

        "Due" means the NEXT episode (``completed_episode + 1``) has
        reached or passed the scheduled episode -- matching the timing
        convention used throughout this codebase's drift/kernel
        interaction (``recovery.core.episode.run_episodes`` calls
        ``sample_reward`` for episode ``i`` BEFORE ``step(i)``, so a
        matrix scheduled for episode ``e`` must already be active by the
        end of ``step(e - 1)``).
        """
        while self._pending and completed_episode + 1 >= self._pending[0]:
            due_episode = self._pending.pop(0)
            new_matrix = _validate_matrix(self._next_matrix(self._current, due_episode))
            self._current = new_matrix
            self._history.append(DriftEvent(due_episode, new_matrix.copy()))

    def base_matrix(self) -> np.ndarray:
        """Return a defensive copy of the ORIGINAL (never-drifted) matrix."""
        return self._base_matrix.copy()

    def current_matrix(self) -> np.ndarray:
        """Return a defensive copy of the currently active matrix.

        For downstream analytic normalization
        (``recovery.metrics.drift_metrics``) -- callers must never mutate
        the returned array in place (it would not affect this
        environment's internal state, but callers should not rely on
        that).
        """
        return self._current.copy()

    def success_matrix(self) -> np.ndarray:
        """``Environment`` protocol method: alias for ``current_matrix()``."""
        return self.current_matrix()

    def matrix_history(self) -> list:
        """Return a shallow copy of the compact change-event history.

        Returns
        -------
        list of DriftEvent
            Always starts with ``DriftEvent(0, base_matrix)`` (or the
            already-drifted matrix if a shift was scheduled at episode
            <= 0). ``O(number of shifts)`` entries, never
            ``O(n_episodes)``.
        """
        return list(self._history)

    def sample_reward(self, true_type: int, action: int, rng: np.random.Generator) -> int:
        """Sample a 0/1 reward from the CURRENTLY active matrix.

        Identical contract to
        ``recovery.envs.matrix_env.MatrixEnvironment.sample_reward``.
        """
        if not (0 <= true_type < self.n_types):
            raise ValueError(
                f"true_type out of range: got {true_type}, expected [0, {self.n_types})"
            )
        if not (0 <= action < self.n_strategies):
            raise ValueError(
                f"action out of range: got {action}, expected [0, {self.n_strategies})"
            )
        return int(rng.random() < self._current[true_type, action])

    def step(self, episode: int) -> None:
        """Advance drift state. See ``_advance`` for exact timing semantics."""
        self._advance(episode)


# ==========================================================================
# 1. PermutationDrift
# ==========================================================================


class PermutationDrift(DriftingEnvironment):
    """Randomly permute the strategy assignment inside affected rows.

    Mathematical definition
    ------------------------
    At each scheduled episode, for every affected row ``t``:
    ``matrix[t, :] <- matrix[t, sigma]`` for a random permutation
    ``sigma`` of ``{0, ..., K-1}`` drawn fresh from ``rng`` at that event.
    The MULTISET of values in row ``t`` is exactly preserved; only which
    column (strategy) achieves which value changes.

    This reproduces the paper's original non-stationary drift mechanism
    exactly (the same "permute affected rows' strategy->probability
    mapping" construction ``experiments/prior_grid.py``'s local
    ``DriftEnvironment`` already uses for Task 7's single/repeated-shift
    conditions) -- see ``tests/test_drift.py``'s
    ``test_permutation_reproduces_original_paper_setup``.

    Why this is the worst case for a fixed mapping: a permutation can
    simultaneously demote the PREVIOUSLY-best strategy to the WORST one
    for that row and promote the previous worst to best, in the same
    event -- no other drift model in this module can move a row's
    optimal action that far in one step.

    Parameters
    ----------
    base_matrix : numpy.ndarray or Environment
        Pre-drift success matrix.
    schedule : sequence of int
        Episodes at which a new permutation is drawn.
    affected_rows : sequence of int
        Row (failure-type) indices to permute. May include every row
        (``magnitude == n_types``) or none (a degenerate but valid no-op
        drift model).
    rng : numpy.random.Generator
        Source of randomness for every permutation drawn (no module-level
        RNG -- the caller owns and seeds this).
    """

    def __init__(
        self,
        base_matrix,
        schedule: Sequence[int],
        affected_rows: Sequence[int],
        rng: np.random.Generator,
    ) -> None:
        self._affected_rows = [int(t) for t in affected_rows]
        self._rng = rng
        super().__init__(base_matrix, schedule)
        for t in self._affected_rows:
            if not (0 <= t < self.n_types):
                raise ValueError(f"affected row {t} out of range [0, {self.n_types})")

    def _next_matrix(self, current: np.ndarray, due_episode: int) -> np.ndarray:
        new = current.copy()
        for t in self._affected_rows:
            new[t] = self._rng.permutation(new[t])
        return new


# ==========================================================================
# 2. MonotoneDecayDrift
# ==========================================================================


class MonotoneDecayDrift(DriftingEnvironment):
    """Exponentially decay one strategy's success probability toward a floor.

    Mathematical definition
    ------------------------
    For every affected row ``t`` and the fixed ``strategy`` column ``k``:
    ``P(e) = floor + (P0 - floor) * 2^(-(e - e0) / half_life)``, where
    ``P0 = base_matrix[t, k]``, ``e0`` is the FIRST scheduled episode, and
    ``e`` is the current scheduled episode. Evaluated at each entry of
    ``schedule`` (not continuously -- unlike ``GradualRotationDrift``,
    which interpolates every episode).

    Represents: API degradation, tool degradation, service instability --
    processes that get monotonically worse (never spontaneously better)
    and asymptotically approach some floor reliability rather than
    collapsing to exactly 0.

    Parameters
    ----------
    base_matrix : numpy.ndarray or Environment
        Pre-drift success matrix.
    strategy : int
        The column (strategy) index that decays.
    rows : sequence of int
        Row (failure-type) indices affected; every other row is untouched.
    half_life : float
        Number of episodes for the GAP above ``floor`` to halve. Must be
        > 0.
    schedule : sequence of int
        Episodes at which the decay formula is (re-)evaluated.
    floor : float
        Asymptotic floor probability, in [0, 1]. Must be <=
        ``base_matrix[t, strategy]`` for every affected row ``t`` (this
        model represents DEGRADATION -- a floor above the starting value
        would describe improvement, which is out of scope for this
        class; see ``GradualRotationDrift`` for arbitrary directional
        drift).

    Raises
    ------
    ValueError
        If ``half_life <= 0``; if ``floor`` is outside [0, 1]; if
        ``floor`` exceeds the base value for any affected row.
    """

    def __init__(
        self,
        base_matrix,
        strategy: int,
        rows: Sequence[int],
        half_life: float,
        schedule: Sequence[int],
        floor: float,
    ) -> None:
        if half_life <= 0:
            raise ValueError(f"half_life must be > 0, got {half_life!r}")
        if not (0.0 <= floor <= 1.0):
            raise ValueError(f"floor must be in [0, 1], got {floor!r}")
        resolved = _validate_matrix(_resolve_base_matrix(base_matrix))
        rows = [int(t) for t in rows]
        strategy = int(strategy)
        if not (0 <= strategy < resolved.shape[1]):
            raise ValueError(f"strategy {strategy} out of range [0, {resolved.shape[1]})")
        for t in rows:
            if not (0 <= t < resolved.shape[0]):
                raise ValueError(f"row {t} out of range [0, {resolved.shape[0]})")
            if floor > resolved[t, strategy] + 1e-12:
                raise ValueError(
                    f"floor={floor!r} exceeds base value {resolved[t, strategy]!r} for row {t} "
                    f"-- MonotoneDecayDrift only represents DEGRADATION (floor <= base value)"
                )

        self._strategy = strategy
        self._rows = rows
        self._half_life = float(half_life)
        self._floor = float(floor)
        self._start_episode: "int | None" = None
        super().__init__(resolved, schedule)

    def _next_matrix(self, current: np.ndarray, due_episode: int) -> np.ndarray:
        if self._start_episode is None:
            self._start_episode = due_episode
        elapsed = due_episode - self._start_episode
        decay = 0.5 ** (elapsed / self._half_life)
        new = current.copy()
        for t in self._rows:
            base_value = self._base_matrix[t, self._strategy]
            new[t, self._strategy] = self._floor + (base_value - self._floor) * decay
        return new


# ==========================================================================
# 3. PartialReorderDrift
# ==========================================================================


class PartialReorderDrift(DriftingEnvironment):
    """Swap only the best and second-best strategy in affected rows.

    Mathematical definition
    ------------------------
    At each scheduled episode, for every affected row ``t``: let
    ``best = argmax_k matrix[t, k]`` and ``second`` the runner-up
    (``argmax`` over the row with ``best`` excluded; ties broken by
    lowest index, matching ``numpy.argmax``'s first-maximum convention).
    Swap ``matrix[t, best]`` and ``matrix[t, second]``. Every other entry
    of row ``t``, and every other row, is untouched.

    The row's VALUE DISTRIBUTION (multiset of probabilities) is exactly
    preserved -- only which column is optimal changes (from ``best`` to
    ``second``). This is a strictly gentler drift than
    ``PermutationDrift``: at most one strategy's rank changes at the top,
    everything else (including the relative order of the remaining
    ``K - 2`` strategies) is undisturbed.

    Represents: minor version changes, ranking adjustments -- the kind of
    small reordering a routine update to an underlying tool or model might
    cause, without a wholesale re-shuffling of every strategy's
    effectiveness.

    Parameters
    ----------
    base_matrix : numpy.ndarray or Environment
        Pre-drift success matrix. Must have at least 2 strategies
        (columns) -- a "best vs. second-best" swap is undefined for
        ``K < 2``.
    rows : sequence of int
        Row (failure-type) indices to reorder.
    schedule : sequence of int
        Episodes at which the swap is (re-)applied. Reapplying at the SAME
        rows again is a no-op relative to the pre-shift-of-this-event
        matrix only if no other event intervened (it swaps back), which
        is a legitimate (if unusual) way to model rankings oscillating.

    Raises
    ------
    ValueError
        If ``base_matrix`` has fewer than 2 strategies.
    """

    def __init__(self, base_matrix, rows: Sequence[int], schedule: Sequence[int]) -> None:
        resolved = _validate_matrix(_resolve_base_matrix(base_matrix))
        if resolved.shape[1] < 2:
            raise ValueError(
                f"PartialReorderDrift requires at least 2 strategies, got {resolved.shape[1]}"
            )
        self._rows = [int(t) for t in rows]
        super().__init__(resolved, schedule)
        for t in self._rows:
            if not (0 <= t < self.n_types):
                raise ValueError(f"row {t} out of range [0, {self.n_types})")

    def _next_matrix(self, current: np.ndarray, due_episode: int) -> np.ndarray:
        new = current.copy()
        for t in self._rows:
            row = new[t]
            order = np.argsort(row, kind="stable")[::-1]  # descending; ties -> lowest index first
            best, second = int(order[0]), int(order[1])
            row[best], row[second] = row[second], row[best]
        return new


# ==========================================================================
# 4. GradualRotationDrift
# ==========================================================================


class GradualRotationDrift(DriftingEnvironment):
    """Linearly interpolate from a base matrix to a target matrix over a window.

    Mathematical definition
    ------------------------
    ``matrix(e) = base + (target - base) * clip((e - start) / window, 0, 1)``
    -- evaluated CONTINUOUSLY (every episode, via ``step``), not just at
    discrete schedule points. Reaches ``base`` exactly at ``e <= start``
    and ``target`` exactly at ``e >= start + window``.

    Represents: slow migrations, model rollouts, progressive deployment --
    a new backing system phased in gradually rather than a single
    instantaneous cutover.

    Exact history during the ramp (Task 8A, Correction 1)
    --------------------------------------------------------
    ``sample_reward``/``current_matrix`` were always EXACT (the live
    matrix is recomputed from the closed-form formula above on every
    ``step()`` call, an ``O(1)`` operation -- never approximated).
    ``matrix_history()`` originally only gained new entries at the
    ramp's start and completion, which made ``recovery.metrics.
    drift_metrics`` functions reconstructing "the matrix active at
    episode e" APPROXIMATE during the ramp itself (they would treat the
    whole ramp as still governed by the base matrix). Task 8A's
    scientific review flagged this as unacceptable for a headline
    result: ``matrix_history()`` now gains ONE entry per episode for
    every episode strictly inside the ramp (``start < e <=
    start + window``), so retrospective normalization is exact
    everywhere, not just live simulation. This is the one deliberate,
    documented exception to this module's general
    ``O(number of shifts)`` history-complexity rule (see the module
    docstring) -- memory here is ``O(window)``, never ``O(n_episodes)``,
    since ``window`` is a bounded constructor parameter independent of
    how many episodes the surrounding experiment runs for. A side
    effect, also documented rather than hidden: ``AnnouncedDrift``
    wrapping a ``GradualRotationDrift`` instance will report
    ``drift_signal=True`` on EVERY episode inside the ramp, not just at
    its start/end -- which is in fact the mathematically correct
    reading of "did the matrix change this episode" for a continuously
    drifting environment.

    Parameters
    ----------
    base_matrix : numpy.ndarray or Environment
        Starting matrix.
    target_matrix : numpy.ndarray
        Ending matrix. Must have the same shape as ``base_matrix``.
    start_episode : int
        Episode at which interpolation begins (``alpha=0`` for every
        episode ``<= start_episode``). Must be ``>= 0``.
    window : int
        Number of episodes over which the interpolation completes. Must
        be ``>= 1`` (a window of 0 would be a discontinuous jump, out of
        scope for a "gradual" model -- use a 1-entry ``schedule`` on
        ``DriftingEnvironment`` directly, or ``PermutationDrift``/
        ``PartialReorderDrift``, for an instantaneous change).

    Raises
    ------
    ValueError
        If ``target_matrix``'s shape does not match ``base_matrix``'s, or
        contains values outside [0, 1]; if ``start_episode < 0``; if
        ``window < 1``.
    """

    def __init__(
        self,
        base_matrix,
        target_matrix: np.ndarray,
        start_episode: int,
        window: int,
    ) -> None:
        resolved_base = _validate_matrix(_resolve_base_matrix(base_matrix))
        target_matrix = _validate_matrix(np.asarray(target_matrix, dtype=np.float64))
        if target_matrix.shape != resolved_base.shape:
            raise ValueError(
                f"target_matrix shape {target_matrix.shape} must match base_matrix shape "
                f"{resolved_base.shape}"
            )
        if start_episode < 0:
            raise ValueError(f"start_episode must be >= 0, got {start_episode!r}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window!r}")

        self._target = target_matrix.copy()
        self._start = int(start_episode)
        self._window = int(window)
        super().__init__(resolved_base, schedule=())

    def _matrix_at(self, episode: int) -> np.ndarray:
        if episode <= self._start:
            alpha = 0.0
        elif episode >= self._start + self._window:
            alpha = 1.0
        else:
            alpha = (episode - self._start) / self._window
        return self._base_matrix + (self._target - self._base_matrix) * alpha

    def _advance(self, completed_episode: int) -> None:
        """Recompute the exact live matrix every call (``O(1)``), and --
        Task 8A, Correction 1 -- append a COMPACT history entry for every
        episode strictly inside the ramp (``self._start < next_episode <=
        self._start + self._window``), so ``matrix_history()`` alone is
        enough to reconstruct the exact active matrix at any episode, not
        just before the ramp starts or after it completes. Outside the
        ramp, no new entry is needed: the base matrix's single episode-0
        entry (from ``DriftingEnvironment.__init__``) covers everything
        up to ``start``, and the ramp's final entry (at
        ``start + window``) covers everything after -- both already
        exact without further logging.
        """
        next_episode = completed_episode + 1
        self._current = _validate_matrix(self._matrix_at(next_episode))
        if self._start < next_episode <= self._start + self._window:
            self._history.append(DriftEvent(next_episode, self._current.copy()))


# ==========================================================================
# 5. AnnouncedDrift
# ==========================================================================


class AnnouncedDrift:
    """Decorator: wrap ANY drift model (or, structurally, anything
    exposing ``step``/``matrix_history``) and expose a ``shift_signal``
    boolean, true for exactly the episode immediately following a
    ``step()`` call that produced a NEW compact history entry -- i.e.
    exactly on shift episodes. No other behavior changes: every other
    method delegates, unmodified, to the wrapped model.

    Two attribute names for the same flag
    ---------------------------------------
    ``shift_signal`` is a read-only property (the name this module's
    specification uses). ``drift_signal`` is the underlying plain
    attribute -- the EXACT name
    ``recovery.core.episode.run_episodes`` reads via
    ``getattr(env, "drift_signal", False)`` to populate
    ``Context.drift_signal``. Both always agree; ``shift_signal`` exists
    purely as a more descriptive alias for direct inspection (e.g. in
    tests or analysis code) of the same underlying flag.

    The episode-0 edge case
    -------------------------
    If the wrapped model already applied a schedule entry at episode <= 0
    during ITS OWN construction (before being handed to ``AnnouncedDrift``
    here), that would normally be invisible to the FIRST ``Context`` the
    kernel builds (which reads ``env.drift_signal`` before ever calling
    ``step()``). ``AnnouncedDrift.__init__`` detects this case (the
    wrapped model's ``matrix_history()`` already has more than its
    initial entry) and sets ``drift_signal=True`` immediately at
    construction, so episode 0's ``Context`` still sees the announcement
    -- the only way it structurally can, given the kernel calls
    ``Context(...)`` before ``step()`` every episode.

    Parameters
    ----------
    wrapped : DriftingEnvironment (or duck-typed equivalent)
        Any object exposing ``n_types``, ``n_strategies``,
        ``success_matrix()``, ``current_matrix()``, ``matrix_history()``,
        ``sample_reward()``, and ``step()``.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self.n_types: int = wrapped.n_types
        self.n_strategies: int = wrapped.n_strategies
        initial_history = wrapped.matrix_history()
        self._last_history_len = len(initial_history)
        # See "The episode-0 edge case" above.
        self.drift_signal: bool = len(initial_history) > 1

    @property
    def shift_signal(self) -> bool:
        """Read-only alias for ``drift_signal`` -- see class docstring."""
        return self.drift_signal

    def base_matrix(self) -> np.ndarray:
        return self._wrapped.base_matrix()

    def current_matrix(self) -> np.ndarray:
        return self._wrapped.current_matrix()

    def success_matrix(self) -> np.ndarray:
        return self._wrapped.success_matrix()

    def matrix_history(self) -> list:
        return self._wrapped.matrix_history()

    def sample_reward(self, true_type: int, action: int, rng: np.random.Generator) -> int:
        return self._wrapped.sample_reward(true_type, action, rng)

    def step(self, episode: int) -> None:
        """Advance the wrapped model, then recompute ``drift_signal`` by
        comparing its history length before and after -- a length
        increase means THIS ``step()`` call produced a genuine shift
        event (whether from one schedule entry or several sharing the
        same due episode -- "overlapping shifts" collapse to a single
        True announcement for that episode, documented here rather than
        distinguishing event COUNT, which ``Context.drift_signal`` (a
        single bool) cannot represent anyway).
        """
        self._wrapped.step(episode)
        new_len = len(self._wrapped.matrix_history())
        self.drift_signal = new_len > self._last_history_len
        self._last_history_len = new_len
