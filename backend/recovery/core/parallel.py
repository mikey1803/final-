"""Parallel execution of a grid of kernel runs, with content-addressed,
skip-if-already-run storage (``RunStore``).

``experiments/`` scripts build a ``list[RunConfig]`` and hand it to
``run_grid`` -- this is the only place ``joblib`` and on-disk run storage
are wired together, so experiment scripts stay free of both parallelism
and storage logic (see docs/experiments/horizon.md).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

from recovery.core.episode import run_episodes
from recovery.core.protocols import Classifier, Policy

__all__ = ["RunConfig", "RunStore", "run_grid"]


@dataclass(frozen=True, eq=False)
class RunConfig:
    """One (method, environment, seed, ...) cell to simulate.

    ``eq=False``: a frozen dataclass would otherwise auto-generate
    ``__eq__``/``__hash__`` from ALL fields, including ``matrix`` (a
    ``numpy.ndarray``) -- comparing or hashing that raises ("ambiguous
    truth value" / "unhashable type"). Falls back to identity-based
    equality/hash, which is all ``run_grid``/``RunStore`` ever need
    (cells are addressed by ``content_key()``, not by ``==``).

    ``env_id``/``classifier_id``/``prior_id`` are short LABELS used for
    content-addressing (hashing a matrix's float array is fragile across
    platforms/numpy versions; a caller-assigned label -- e.g.
    ``"canonical"``, ``"gap0.20_idx03"`` -- is stable and matches how
    Table VI (Task 4A) already identifies matrices by index rather than
    by value). It is the CALLER's responsibility to keep these labels
    unique per distinct configuration -- two cells that hash to the same
    key are treated as the same computation and one will be skipped.

    Parameters
    ----------
    method : str
        Display name for this policy configuration (e.g. ``"thompson"``).
    env_id : str
        Label identifying the environment/matrix (e.g. ``"canonical"``).
    matrix : numpy.ndarray, shape (N, K)
        The actual success matrix to run against.
    seed : int
        Top-level kernel seed (passed to ``recovery.core.episode.run_episodes``).
    n_episodes : int
        Number of episodes to run (``T_max`` for a horizon-sweep cell).
    policy_factory : callable
        Zero-argument callable building a FRESH ``Policy`` instance (called
        inside the worker, once per cell).
    classifier_factory : callable
        Zero-argument callable building a FRESH ``Classifier`` instance.
    classifier_id : str, default="perfect"
        Label for content-addressing (e.g. ``"perfect"``, ``"real"``).
    type_prior : numpy.ndarray, optional
        Type prior; ``None`` = uniform.
    prior_id : str, default="uniform"
        Label for content-addressing.
    """

    method: str
    env_id: str
    matrix: np.ndarray
    seed: int
    n_episodes: int
    policy_factory: Callable[[], Policy]
    classifier_factory: Callable[[], Classifier]
    classifier_id: str = "perfect"
    type_prior: Optional[np.ndarray] = None
    prior_id: str = "uniform"

    def content_key(self) -> str:
        """Deterministic hash identifying this cell's RunStore address.

        Depends only on the labeling/identity fields (never on ``matrix``
        values or the factories themselves), so it is stable across
        Python processes, ``n_jobs``, and completion order.
        """
        payload = {
            "method": self.method,
            "env_id": self.env_id,
            "seed": self.seed,
            "n_episodes": self.n_episodes,
            "classifier_id": self.classifier_id,
            "prior_id": self.prior_id,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


class RunStore:
    """Content-addressed on-disk store for completed cells' full episode logs.

    One parquet file per cell, named by ``RunConfig.content_key()``, under
    ``root``. Re-running ``run_grid`` over a grid that overlaps a
    previous run skips any cell whose file already exists -- this file
    presence check IS the entire "resume" mechanism; there is no separate
    index/manifest that could go stale relative to the files on disk.

    Parameters
    ----------
    root : pathlib.Path
        Directory the store writes into (created if missing).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, config: RunConfig) -> Path:
        """The content-addressed path this cell's output belongs at."""
        return self.root / f"{config.content_key()}.parquet"

    def exists(self, config: RunConfig) -> bool:
        """Whether this cell has already been run and stored."""
        return self.path_for(config).exists()

    def write(self, config: RunConfig, log) -> Path:
        """Write one cell's full episode log, plus its identifying
        columns, to its content-addressed path.

        Writes to a ``.tmp`` sibling first and renames into place --
        ``Path.replace`` is atomic on the same filesystem, so a worker
        killed mid-write never leaves a partial/corrupt file at the final
        path for a later run to mistake for a completed cell.

        Parameters
        ----------
        config : RunConfig
            The cell this log belongs to.
        log : EpisodeLog-like
            Exposes ``.to_dataframe()`` (e.g.
            ``recovery.core.log.EpisodeLog``).

        Returns
        -------
        pathlib.Path
            The path written to (``path_for(config)``).
        """
        df = log.to_dataframe()
        df["method"] = config.method
        df["env_id"] = config.env_id
        df["seed"] = config.seed
        df["classifier_id"] = config.classifier_id
        df["prior_id"] = config.prior_id
        path = self.path_for(config)
        tmp_path = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
        return path

    def read(self, config: RunConfig) -> pd.DataFrame:
        """Read back one cell's stored log as a tidy per-episode DataFrame."""
        return pd.read_parquet(self.path_for(config))


def _run_one_cell(config: RunConfig, store_root: Path) -> Path:
    """Worker entry point: run one cell's kernel trajectory and store it.

    Module-level (not a closure) so it pickles cleanly regardless of
    which joblib backend is selected. Re-checks ``store.exists`` inside
    the worker (not just in ``run_grid``'s caller-side filter) so two
    overlapping ``run_grid`` invocations racing against the same store
    still each skip work the other has already finished, up to the usual
    filesystem-existence-check race window.
    """
    from recovery.envs.matrix_env import MatrixEnvironment

    store = RunStore(store_root)
    if store.exists(config):
        return store.path_for(config)

    env = MatrixEnvironment(config.matrix)
    policy = config.policy_factory()
    classifier = config.classifier_factory()
    log = run_episodes(
        env, policy, classifier,
        n_episodes=config.n_episodes, seed=config.seed, type_prior=config.type_prior,
    )
    return store.write(config, log)


def run_grid(
    cells: list[RunConfig],
    store_root: Path,
    n_jobs: int = -1,
    backend: str = "loky",
    show_progress: bool = True,
) -> list[Path]:
    """Run every cell in ``cells`` (skipping ones already in the
    RunStore), in parallel via ``joblib``, and return each cell's output
    path.

    Determinism: each cell's output depends only on its own
    ``RunConfig`` (seed, matrix, factories) -- workers share no mutable
    state (``backend="loky"`` is process-based, so there is no
    accidental cross-worker state even in principle), and
    ``recovery.core.rng.from_seed`` derives fully independent RNG
    streams from each cell's own ``seed``. The result for a given cell
    is therefore identical regardless of ``n_jobs`` or the order cells
    happen to complete in. The RETURNED LIST is always in ``cells``
    order (not completion order) for the same reason.

    Parameters
    ----------
    cells : list of RunConfig
        The grid to run.
    store_root : pathlib.Path
        Root directory of the ``RunStore``.
    n_jobs : int, default=-1
        Passed to ``joblib.Parallel`` (``-1`` = use all cores).
    backend : str, default="loky"
        Passed to ``joblib.Parallel``. Process-based backends
        (``"loky"``, the default) are required for factories built as
        closures/lambdas to be usable at all (``loky`` pickles via
        ``cloudpickle``); a thread-based backend would also work but
        gains nothing since the kernel is CPU-bound pure Python/numpy.
    show_progress : bool, default=True
        Whether to render a ``tqdm`` progress bar (to stderr).

    Returns
    -------
    list of pathlib.Path
        One path per cell, in ``cells`` order -- the cell's RunStore
        location (whether freshly run this call or already present).
    """
    store = RunStore(store_root)
    pending = [c for c in cells if not store.exists(c)]
    skipped = len(cells) - len(pending)
    if skipped:
        print(
            f"run_grid: {skipped}/{len(cells)} cell(s) already in the RunStore at "
            f"{store_root}, skipping",
            file=sys.stderr,
        )

    if pending:
        parallel = joblib.Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")
        results = parallel(joblib.delayed(_run_one_cell)(config, store_root) for config in pending)
        if show_progress:
            from tqdm import tqdm

            results = tqdm(results, total=len(pending), desc="run_grid", file=sys.stderr)
        for _ in results:
            pass  # each cell writes itself to the store; nothing to collect here

    return [store.path_for(c) for c in cells]
