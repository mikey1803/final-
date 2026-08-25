"""Synthetic classifiers: perfect (oracle) typing and confusion-matrix noise.

Both satisfy ``recovery.core.protocols.Classifier`` structurally.
"""

import numpy as np

__all__ = ["PerfectClassifier", "ConfusionClassifier"]


class PerfectClassifier:
    """Always predicts the true type -- i.e. "no classifier noise" mode."""

    def predict(self, true_type: int, rng: np.random.Generator) -> int:
        """Return ``true_type`` unchanged.

        Parameters
        ----------
        true_type : int
            The true type index.
        rng : numpy.random.Generator
            Consumed exactly once (one ``random()`` draw) even though the
            value is discarded, so this classifier's stream-consumption
            shape matches noisy classifiers (``ConfusionClassifier``) --
            see ``recovery.core.protocols.Classifier.predict``.

        Returns
        -------
        int
            ``true_type``, unchanged.
        """
        _ = rng.random()  # deliberately unused; keeps classifier-stream consumption uniform
        return true_type


class ConfusionClassifier:
    """Predicts a type drawn from a fixed row-stochastic confusion matrix.

    Parameters
    ----------
    C : numpy.ndarray, shape (N, N)
        Row-stochastic confusion matrix: ``C[t, t']`` = P(predict ``t'`` |
        true ``t``).

    Raises
    ------
    ValueError
        If ``C`` is not square 2-D, or its rows do not each sum to 1.
    """

    def __init__(self, C: np.ndarray) -> None:
        C = np.asarray(C, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"C must be square 2-D, got shape {C.shape}")
        row_sums = C.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"C must be row-stochastic (each row sums to 1), got row sums {row_sums}"
            )
        self._C = C
        self.n_types: int = C.shape[0]

    def predict(self, true_type: int, rng: np.random.Generator) -> int:
        """Draw a predicted type from ``C[true_type]``.

        Parameters
        ----------
        true_type : int
            The true type index, in ``[0, n_types)``.
        rng : numpy.random.Generator
            Source of randomness.

        Returns
        -------
        int
            The predicted type index, drawn from ``C[true_type]``.

        Raises
        ------
        ValueError
            If ``true_type`` is out of range.
        """
        if not (0 <= true_type < self.n_types):
            raise ValueError(
                f"true_type out of range: got {true_type}, expected [0, {self.n_types})"
            )
        return int(rng.choice(self.n_types, p=self._C[true_type]))
