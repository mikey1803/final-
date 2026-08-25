"""Preallocated, columnar episode log for the shared simulation kernel.

Deliberately avoids lists-of-dicts / dicts-of-lists: every column is a
preallocated ``numpy`` array of length ``n_episodes``, written into by
index during the episode loop, so logging 200k episodes allocates memory
exactly once.
"""

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["EpisodeLog"]


class EpisodeLog:
    """Columnar record of every episode in a ``run_episodes`` call.

    Parameters
    ----------
    n_episodes : int
        Number of episodes this log will hold (must be >= 1).

    Raises
    ------
    ValueError
        If ``n_episodes < 1``.
    """

    def __init__(self, n_episodes: int) -> None:
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
        self.n_episodes = n_episodes
        # record() always sets episode[i] = i, so this is precomputed here
        # rather than written on every record() call -- identical final
        # content, one fewer array write per episode in the hot loop.
        self.episode: np.ndarray = np.arange(n_episodes, dtype=np.int32)
        self.true_type: np.ndarray = np.zeros(n_episodes, dtype=np.int8)
        self.predicted_type: np.ndarray = np.zeros(n_episodes, dtype=np.int8)
        self.action: np.ndarray = np.zeros(n_episodes, dtype=np.int8)
        self.reward: np.ndarray = np.zeros(n_episodes, dtype=np.int8)

    def record(
        self,
        i: int,
        true_type: int,
        predicted_type: int,
        action: int,
        reward: int,
    ) -> None:
        """Write one episode's outcome into row ``i`` of every column.

        Parameters
        ----------
        i : int
            Row index, in ``[0, n_episodes)``.
        true_type : int
            The true type index for this episode.
        predicted_type : int
            The (possibly noisy) predicted type index for this episode.
        action : int
            The action index played.
        reward : int
            The observed reward (0 or 1).

        Raises
        ------
        ValueError
            If ``i`` is out of ``[0, n_episodes)``.
        """
        if not (0 <= i < self.n_episodes):
            raise ValueError(f"i must be in [0, {self.n_episodes}), got {i}")
        self.episode[i] = i
        self.true_type[i] = true_type
        self.predicted_type[i] = predicted_type
        self.action[i] = action
        self.reward[i] = reward

    def to_dataframe(self) -> pd.DataFrame:
        """Return the log as a ``pandas.DataFrame`` with one row per episode.

        Returns
        -------
        pandas.DataFrame
            Columns: ``episode``, ``true_type``, ``predicted_type``,
            ``action``, ``reward``.
        """
        return pd.DataFrame(
            {
                "episode": self.episode,
                "true_type": self.true_type,
                "predicted_type": self.predicted_type,
                "action": self.action,
                "reward": self.reward,
            }
        )

    def success_rate(self, start_frac: float = 0.0, end_frac: float = 1.0) -> float:
        """Mean reward over the episode window ``[start_frac, end_frac)``.

        Parameters
        ----------
        start_frac : float, default=0.0
            Start of the window, as a fraction of ``n_episodes``, in [0, 1].
        end_frac : float, default=1.0
            End of the window, as a fraction of ``n_episodes``, in [0, 1].

        Returns
        -------
        float
            The mean reward over rows ``[floor(start_frac*n), floor(end_frac*n))``.

        Raises
        ------
        ValueError
            If ``start_frac``/``end_frac`` are outside [0, 1] or
            ``start_frac >= end_frac``.
        """
        if not (0.0 <= start_frac <= 1.0):
            raise ValueError(f"start_frac must be in [0, 1], got {start_frac}")
        if not (0.0 <= end_frac <= 1.0):
            raise ValueError(f"end_frac must be in [0, 1], got {end_frac}")
        if start_frac >= end_frac:
            raise ValueError(
                f"start_frac must be < end_frac, got start_frac={start_frac}, end_frac={end_frac}"
            )
        start = int(start_frac * self.n_episodes)
        end = int(end_frac * self.n_episodes)
        window = self.reward[start:end]
        return float(window.mean()) if window.size > 0 else float("nan")

    def per_type_rate(self) -> np.ndarray:
        """Mean reward broken down by TRUE type.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_types,)`` where ``n_types = true_type.max() + 1``,
            float64. Types never seen as a true type in this log get
            ``nan``.
        """
        n_types = int(self.true_type.max()) + 1 if self.n_episodes > 0 else 0
        rates = np.full(n_types, np.nan, dtype=np.float64)
        for t in range(n_types):
            mask = self.true_type == t
            if np.any(mask):
                rates[t] = float(self.reward[mask].mean())
        return rates

    def __len__(self) -> int:
        return self.n_episodes
