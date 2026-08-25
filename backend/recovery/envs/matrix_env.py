"""A stationary environment backed by an (N, K) success-probability matrix.

Satisfies the ``recovery.core.protocols.Environment`` protocol.
"""

import numpy as np

__all__ = ["MatrixEnvironment"]


class MatrixEnvironment:
    """Reward-generating process defined by a fixed (N, K) success matrix.

    ``sample_reward`` depends on nothing but ``true_type``, ``action``,
    and ``rng`` -- there is no parameter or attribute through which a
    predicted type could influence the reward.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (N, K)
        ``matrix[t, k]`` = P(success | true type ``t``, action ``k``).

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D, or contains values outside [0, 1].
    """

    def __init__(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
        if np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise ValueError(
                f"matrix must contain only values in [0, 1], got min={matrix.min()!r}, "
                f"max={matrix.max()!r}"
            )
        self._matrix = matrix
        self.n_types: int = matrix.shape[0]
        self.n_strategies: int = matrix.shape[1]

    def success_matrix(self) -> np.ndarray:
        """Return a defensive copy of the (N, K) success-probability matrix.

        For analytic cross-checks only (e.g. against
        ``recovery.theory.analytic``) -- no policy or classifier reads
        this at simulation time. A copy is returned so callers cannot
        mutate the environment's internal state through the reference.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_types, n_strategies)``.
        """
        return self._matrix.copy()

    def sample_reward(self, true_type: int, action: int, rng: np.random.Generator) -> int:
        """Sample a 0/1 reward from ``matrix[true_type, action]``.

        Parameters
        ----------
        true_type : int
            The true type index, in ``[0, n_types)``.
        action : int
            The action index, in ``[0, n_strategies)``.
        rng : numpy.random.Generator
            Source of randomness for the Bernoulli draw.

        Returns
        -------
        int
            1 with probability ``matrix[true_type, action]``, else 0.

        Raises
        ------
        ValueError
            If ``true_type`` or ``action`` is out of range.
        """
        if not (0 <= true_type < self.n_types):
            raise ValueError(
                f"true_type out of range: got {true_type}, expected [0, {self.n_types})"
            )
        if not (0 <= action < self.n_strategies):
            raise ValueError(
                f"action out of range: got {action}, expected [0, {self.n_strategies})"
            )
        return int(rng.random() < self._matrix[true_type, action])

    def step(self, episode: int) -> None:
        """No-op drift hook: this environment is stationary.

        Parameters
        ----------
        episode : int
            Ignored.
        """
        return None
