#!/usr/bin/env python
"""Grid experiment for Prior-Guided Thompson Sampling (PGTS) --
``recovery.policies.hybrid.PriorInitializedThompson``/``DiscountedPriorThompson``.

Sweeps prior strength, prior table quality, prior mean, environment
(stationary / single-shift / repeated-shift), horizon, and classifier
condition, recording Task 5's metrics for every cell. This script
contains no bandit/prior-construction logic of its own -- it only builds
configurations and composes ``recovery.core.episode.run_episodes``,
``recovery.policies.hybrid``, and ``recovery.metrics.summary.summarize``.

Infrastructure reuse
---------------------
Storage/content-addressing reuses Task 6's ``recovery.core.parallel.RunStore``
UNCHANGED. Task 6's ``run_grid``/``RunConfig`` are NOT reused for the
drift conditions: ``RunConfig`` always builds a
``recovery.envs.matrix_env.MatrixEnvironment`` internally (hardcoded in
``recovery/core/parallel.py::_run_one_cell``, which this task may not
modify), so it cannot run a cell against ``DriftEnvironment`` below. The
stationary condition COULD go through ``run_grid`` directly, but this
script uses one uniform code path (``_run_one_prior_cell`` + a local
``joblib.Parallel`` call, mirroring ``run_grid``'s own
skip-if-already-stored / deterministic-regardless-of-n_jobs structure)
for all three environment conditions instead of splitting the grid
across two different runners.

Usage
-----
    python experiments/prior_grid.py [--smoke] [options]

Known environmental constraint -- memory pressure at large horizons under
high parallelism
------------------------------------------------------------------------
``recovery.metrics.summary.summarize`` calls ``recovery.metrics.bootstrap.bootstrap_ci``
over the late-stage window (``late_stage_frac`` fraction of ``n_episodes``,
default 20%). ``bootstrap_ci`` is deliberately fully vectorized (see its
own docstring -- a documented, previously-validated speed/memory
trade-off used safely everywhere else in this project, where late-stage
windows are at most a few thousand episodes). For a cell with
``horizon=50_000`` (this script's own default includes it), the late-stage
window is 10,000 episodes, and one ``bootstrap_ci`` call transiently peaks
around 1.6 GB (measured via ``tracemalloc``: two ``(n_resamples, n)`` =
(10000, 10000) arrays, ~763 MiB each). This is NOT a flaw in
``bootstrap_ci`` -- it is a correct, fast function meeting an unusually
large input for the first time in this codebase.

Under joblib's default ``n_jobs=-1`` (full parallelism -- 8 workers on an
8-core/8GB machine), if several workers happen to hit their
``bootstrap_ci`` call at nearly the same moment, peak combined memory can
exceed available RAM, raising ``numpy._core._exceptions._ArrayMemoryError``
(confirmed twice on an 8GB-RAM machine with only ~2-2.7GB free even at
idle: once at horizon=5000 -- "Unable to allocate 76.3 MiB for an array
with shape (10000, 1000)" -- and once at horizon=50000 -- "...763. MiB for
an array with shape (10000, 10000)", both exactly matching this
mechanism's arithmetic). This is timing-dependent, not deterministic: two
small (16-cell) targeted reproductions at ``n_jobs=8`` with horizon=50000
completed without crashing, so it manifests specifically under the
SUSTAINED high-concurrency operation of a full/near-full grid, not
reliably at small scale.

On memory-constrained machines, pass a lower ``--n-jobs`` (e.g. 2-3) when
the grid includes a large horizon -- this trades wall-clock time for
reliability without changing any algorithm or result. No source change to
``bootstrap_ci`` is warranted: it has no defect, and touching its
vectorized implementation would risk every other already-validated call
site across this project for a problem specific to this one script's
large-horizon default on one resource-constrained machine.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.episode import run_episodes  # noqa: E402
from recovery.core.parallel import RunStore  # noqa: E402
from recovery.core.protocols import Policy  # noqa: E402
from recovery.envs.canonical import MATRIX  # noqa: E402
from recovery.envs.matrix_env import MatrixEnvironment  # noqa: E402
from recovery.metrics.prefix import PrefixLog  # noqa: E402
from recovery.metrics.summary import summarize  # noqa: E402
from recovery.policies.adapters import StaticRulePolicy, ThompsonPolicy  # noqa: E402
from recovery.policies.hybrid import DiscountedPriorThompson, PriorInitializedThompson  # noqa: E402

DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "prior_grid.parquet"
DEFAULT_STORE_ROOT = _BACKEND_DIR / "results" / "runs" / "prior_grid"

DEFAULT_STRENGTHS: list[float] = [0, 1, 2, 5, 10, 20, 50, 100, 500, float("inf")]
DEFAULT_QUALITIES: list[int] = [0, 3, 4, 5, 6, 7]
DEFAULT_PRIOR_MEANS: list[float] = [0.6, 0.7, 0.8]
DEFAULT_ENVIRONMENTS: list[str] = ["stationary", "single_shift", "repeated_shift"]
DEFAULT_HORIZONS: list[int] = [500, 5000, 50000]
DEFAULT_CLASSIFIERS: list[str] = ["perfect", "real"]
DEFAULT_N_SEEDS = 30
DEFAULT_BASE_SEED = 9000
N_SHIFT_TYPES = 2  # how many of the 7 failure types drift, per src.evaluation.nonstationarity's convention
REPEATED_SHIFT_PERIOD_FRAC = 0.2  # repeated_shift re-permutes every 20% of the horizon

SMOKE_STRENGTHS: list[float] = [0, 20, float("inf")]
SMOKE_QUALITIES: list[int] = [0, 7]
SMOKE_PRIOR_MEANS: list[float] = [0.7]
SMOKE_ENVIRONMENTS: list[str] = ["stationary", "single_shift"]
SMOKE_HORIZONS: list[int] = [500]
SMOKE_CLASSIFIERS: list[str] = ["perfect"]
SMOKE_N_SEEDS = 2
SMOKE_BASE_SEED = 9000


# ======================================================================
# Drift environment (Environment protocol; not added to recovery/envs --
# see module docstring)
# ======================================================================


class DriftEnvironment:
    """A ``recovery.core.protocols.Environment`` whose success matrix is
    permuted at scheduled episodes.

    Conceptually the same "permute the affected rows' strategy->probability
    mapping" drift mechanism as
    ``src.evaluation.nonstationarity.build_shifted_matrix`` (magnitude/
    difficulty of each affected row is preserved -- only which strategy
    achieves which probability is scrambled) -- reimplemented here against
    the array-indexed matrix convention ``recovery/`` uses throughout
    (that function operates on ``FailureType``-keyed dicts and would need
    a per-call enum/index round-trip to reuse directly for a per-episode
    hot-loop hook).

    A single entry in ``shift_episodes`` is a "single shift" condition; several
    equally-spaced entries is "repeated shift" -- both use this same class.

    Parameters
    ----------
    base_matrix : numpy.ndarray, shape (N, K)
        The pre-shift success matrix.
    shift_episodes : list of int
        Episode indices at which a NEW permutation of ``affected_types``'
        rows takes effect (see ``step``'s docstring for exact timing).
    affected_types : list of int
        Which type indices drift; other rows are never touched.
    shift_seed : int
        Seeds the RNG that decides each shift's permutation -- entirely
        separate from the kernel's ``streams.reward``/``streams.types``
        (drift SCHEDULE/CONTENT is a property of the environment's own
        construction, not of any one run's reward-sampling stream).
    """

    def __init__(
        self,
        base_matrix: np.ndarray,
        shift_episodes: list[int],
        affected_types: list[int],
        shift_seed: int,
    ) -> None:
        base_matrix = np.asarray(base_matrix, dtype=np.float64)
        self.n_types, self.n_strategies = base_matrix.shape
        self._shift_episodes = sorted(shift_episodes)
        shift_rng = np.random.default_rng(shift_seed)

        self._states: list[np.ndarray] = [base_matrix.copy()]
        current = base_matrix.copy()
        for _ in self._shift_episodes:
            current = current.copy()
            for t in affected_types:
                current[t] = shift_rng.permutation(current[t])
            self._states.append(current)

        self._state_idx = 0
        self._active = self._states[0]

    def success_matrix(self) -> np.ndarray:
        return self._active.copy()

    def sample_reward(self, true_type: int, action: int, rng: np.random.Generator) -> int:
        if not (0 <= true_type < self.n_types):
            raise ValueError(f"true_type out of range: got {true_type}, expected [0, {self.n_types})")
        if not (0 <= action < self.n_strategies):
            raise ValueError(f"action out of range: got {action}, expected [0, {self.n_strategies})")
        return int(rng.random() < self._active[true_type, action])

    def step(self, episode: int) -> None:
        """Advance the active matrix if the NEXT episode (``episode + 1``)
        has reached a scheduled shift.

        ``run_episodes`` calls ``sample_reward(i, ...)`` BEFORE
        ``step(i)`` for every episode ``i`` (see
        ``recovery.core.episode.run_episodes``'s docstring, steps 5 and
        7) -- so for episode ``i == shift_episodes[k]`` to see the
        post-shift matrix, the switch must already have happened by the
        time episode ``i`` calls ``sample_reward``, i.e. at the END of
        ``step(i - 1)``.
        """
        while self._state_idx < len(self._shift_episodes) and episode + 1 >= self._shift_episodes[self._state_idx]:
            self._state_idx += 1
            self._active = self._states[self._state_idx]


def _pick_shift_types(rng: np.random.Generator, n_types: int, n_shift: int) -> list[int]:
    return sorted(int(x) for x in rng.choice(n_types, size=min(n_shift, n_types), replace=False))


def build_environment(condition: str, matrix: np.ndarray, n_episodes: int, env_seed: int):
    """Build the ``Environment`` for ``condition in {"stationary",
    "single_shift", "repeated_shift"}``.

    ``env_seed`` seeds BOTH which types drift and the drift permutations
    themselves -- deterministic given ``(condition, n_episodes, env_seed)``.
    """
    if condition == "stationary":
        return MatrixEnvironment(matrix)

    rng = np.random.default_rng(env_seed)
    affected = _pick_shift_types(rng, matrix.shape[0], N_SHIFT_TYPES)
    if condition == "single_shift":
        shift_episodes = [n_episodes // 2]
    elif condition == "repeated_shift":
        period = max(1, int(n_episodes * REPEATED_SHIFT_PERIOD_FRAC))
        shift_episodes = list(range(period, n_episodes, period))
    else:
        raise ValueError(f"unknown environment condition {condition!r}")
    return DriftEnvironment(matrix, shift_episodes, affected, shift_seed=env_seed)


# ======================================================================
# Grid configuration
# ======================================================================


@dataclass(frozen=True, eq=False)
class PriorCellConfig:
    """One grid cell. Mirrors ``recovery.core.parallel.RunConfig``'s
    shape/purpose (content-addressed identity + factories) but carries an
    ``environment_factory`` instead of a raw ``matrix``, so it can build
    ``DriftEnvironment`` as well as ``MatrixEnvironment`` -- see the
    module docstring for why ``RunConfig`` itself isn't reusable here.
    Exposes the same ``method``/``env_id``/``seed``/``classifier_id``/
    ``prior_id`` attributes ``RunStore.write`` reads (duck-typed; no
    change to ``RunStore`` was needed).
    """

    method: str
    env_id: str
    seed: int
    n_episodes: int
    environment_factory: Callable[[], object]
    policy_factory: Callable[[], Policy]
    classifier_factory: Callable[[], object]
    classifier_id: str = "perfect"
    prior_id: str = "uniform"
    strength: float = 0.0
    quality: int = 0
    prior_mean: float = 0.7
    gamma: float = 1.0

    def content_key(self) -> str:
        import hashlib
        import json

        payload = {
            "method": self.method, "env_id": self.env_id, "seed": self.seed,
            "n_episodes": self.n_episodes, "classifier_id": self.classifier_id,
            "prior_id": self.prior_id, "strength": self.strength, "quality": self.quality,
            "prior_mean": self.prior_mean, "gamma": self.gamma,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


def _classifier_factory(classifier_id: str):
    if classifier_id == "perfect":
        return lambda: PerfectClassifier()
    if classifier_id == "real":
        import warnings

        from recovery.classify.adapter import TrainedClassifierAdapter
        from src.evaluation.classifier_context import load_or_train_classifier

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = load_or_train_classifier()
        return lambda: TrainedClassifierAdapter(clf)
    raise ValueError(f"unknown classifier_id {classifier_id!r}")


def _strength_label(strength: float) -> str:
    return "inf" if strength == float("inf") else f"{strength:g}"


def build_grid(
    strengths: list[float],
    qualities: list[int],
    prior_means: list[float],
    environments: list[str],
    horizons: list[int],
    classifiers: list[str],
    n_seeds: int,
    base_seed: int,
    matrix: np.ndarray = MATRIX,
    include_stock_thompson: bool = True,
) -> list[PriorCellConfig]:
    """Build the full grid of ``PriorCellConfig`` cells.

    For ``strength in {0, inf}``, ``prior_mean`` has no effect on the
    resulting policy (``strength=0`` erases the prior; ``strength=inf``
    is represented by ``StaticRulePolicy``, which never reads
    ``prior_mean`` at all) -- these two boundary strengths are run ONCE
    per (quality, environment, horizon, classifier, seed), not once per
    ``prior_means`` entry, to avoid redundant, wasted cells.
    """
    n_types = matrix.shape[0]
    cells: list[PriorCellConfig] = []

    for env_id in environments:
        for horizon in horizons:
            for classifier_id in classifiers:
                classifier_factory = _classifier_factory(classifier_id)

                if include_stock_thompson:
                    for seed_idx in range(n_seeds):
                        seed = base_seed + seed_idx
                        cells.append(
                            PriorCellConfig(
                                method="stock_thompson", env_id=env_id, seed=seed, n_episodes=horizon,
                                environment_factory=(
                                    lambda m=matrix, h=horizon, s=seed, e=env_id: build_environment(e, m, h, s)
                                ),
                                policy_factory=(lambda s=seed: ThompsonPolicy(seed=s, name="stock_thompson")),
                                classifier_factory=classifier_factory, classifier_id=classifier_id,
                                strength=0.0, quality=7, prior_mean=0.7, gamma=1.0, prior_id="stock",
                            )
                        )

                for quality in qualities:
                    for strength in strengths:
                        means_for_strength = [prior_means[0]] if strength in (0.0, float("inf")) else prior_means
                        for prior_mean in means_for_strength:
                            for seed_idx in range(n_seeds):
                                seed = base_seed + seed_idx
                                table_seed = base_seed + quality * 10_000 + seed_idx
                                table = PriorInitializedThompson.from_quality(matrix, quality, table_seed)
                                prior_id = f"q{quality}_s{_strength_label(strength)}_m{prior_mean:g}"

                                if strength == float("inf"):
                                    policy_factory = (lambda t=table: StaticRulePolicy(t, name="pgts_static_limit"))
                                    method = "pgts_static_limit"
                                else:
                                    policy_factory = (
                                        lambda t=table, s=strength, m=prior_mean, n=n_types, k=matrix.shape[1]:
                                        PriorInitializedThompson(t, k, strength=s, prior_mean=m, name="pgts")
                                    )
                                    method = "pgts"

                                cells.append(
                                    PriorCellConfig(
                                        method=method, env_id=env_id, seed=seed, n_episodes=horizon,
                                        environment_factory=(
                                            lambda m=matrix, h=horizon, s=seed, e=env_id: build_environment(e, m, h, s)
                                        ),
                                        policy_factory=policy_factory, classifier_factory=classifier_factory,
                                        classifier_id=classifier_id, strength=strength, quality=quality,
                                        prior_mean=prior_mean, gamma=1.0, prior_id=prior_id,
                                    )
                                )
    return cells


def build_discounted_grid(
    strengths: list[float],
    qualities: list[int],
    gammas: list[float],
    environments: list[str],
    horizons: list[int],
    classifiers: list[str],
    n_seeds: int,
    base_seed: int,
    matrix: np.ndarray = MATRIX,
    prior_mean: float = 0.7,
) -> list[PriorCellConfig]:
    """Discounted PGTS cells (recovery half-life analysis): sweeps
    ``strength x gamma`` at a fixed ``prior_mean`` (drift conditions are
    where discounting matters -- see docs/experiments/hybrid.md,
    "Discounted PGTS")."""
    n_types = matrix.shape[0]
    cells: list[PriorCellConfig] = []
    finite_strengths = [s for s in strengths if s != float("inf")]

    for env_id in environments:
        for horizon in horizons:
            for classifier_id in classifiers:
                classifier_factory = _classifier_factory(classifier_id)
                for quality in qualities:
                    for strength in finite_strengths:
                        for gamma in gammas:
                            for seed_idx in range(n_seeds):
                                seed = base_seed + seed_idx
                                table_seed = base_seed + quality * 10_000 + seed_idx
                                table = PriorInitializedThompson.from_quality(matrix, quality, table_seed)
                                prior_id = f"q{quality}_s{_strength_label(strength)}_g{gamma:g}"
                                cells.append(
                                    PriorCellConfig(
                                        method="discounted_pgts", env_id=env_id, seed=seed, n_episodes=horizon,
                                        environment_factory=(
                                            lambda m=matrix, h=horizon, s=seed, e=env_id: build_environment(e, m, h, s)
                                        ),
                                        policy_factory=(
                                            lambda t=table, s=strength, g=gamma, k=matrix.shape[1]:
                                            DiscountedPriorThompson(t, k, strength=s, prior_mean=prior_mean, gamma=g)
                                        ),
                                        classifier_factory=classifier_factory, classifier_id=classifier_id,
                                        strength=strength, quality=quality, prior_mean=prior_mean, gamma=gamma,
                                        prior_id=prior_id,
                                    )
                                )
    return cells


# ======================================================================
# Execution
# ======================================================================


def _run_one_prior_cell(config: PriorCellConfig, store_root: Path) -> Path:
    """Worker entry point: run one cell, compute Task 5 metrics, store a
    ONE-ROW tidy parquet (mirrors ``recovery.core.parallel._run_one_cell``
    but builds ``config.environment_factory()`` instead of a hardcoded
    ``MatrixEnvironment`` -- see module docstring).

    Metrics use ``env.success_matrix()`` AFTER the run completes -- for
    ``DriftEnvironment``, this is the FINAL (post-all-shifts) matrix, not
    the matrix that was active for the whole run. This is the most
    defensible single ground truth for a drifted run: ``late_stage_rate``
    and ``worst_type_rate`` (both computed on the run's tail) are
    evaluated against the SAME regime that produced them; regret-based
    metrics (which sum over the WHOLE run, spanning pre- and post-shift
    episodes) are consequently only APPROXIMATE for drift conditions --
    documented explicitly in docs/experiments/hybrid.md rather than
    silently presented as exact.
    """
    store = RunStore(store_root)
    if store.exists(config):
        return store.path_for(config)

    env = config.environment_factory()
    policy = config.policy_factory()
    classifier = config.classifier_factory()
    log = run_episodes(env, policy, classifier, n_episodes=config.n_episodes, seed=config.seed)

    matrix_for_metrics = env.success_matrix()
    prefix_log = PrefixLog(log.true_type, log.action, log.reward, log.predicted_type)
    metrics = summarize(prefix_log, matrix_for_metrics)

    row = {
        "method": config.method, "env_id": config.env_id, "classifier_id": config.classifier_id,
        "seed": config.seed, "n_episodes": config.n_episodes, "strength": config.strength,
        "quality": config.quality, "prior_mean": config.prior_mean, "gamma": config.gamma,
        "prior_id": config.prior_id,
        "late_stage_rate": metrics["late_stage_rate"],
        "normalized_score": metrics["normalized_score"],
        "final_cumulative_regret": metrics["final_cumulative_regret"],
        "normalized_cumulative_regret": metrics["normalized_cumulative_regret"],
        "total_cost": metrics["total_cost"],
        "cost_normalized_reward": metrics["cost_normalized_reward"],
        "cost_aware_oracle_rate": metrics["cost_aware_oracle_rate"],
        "worst_type_rate": metrics["worst_type_rate"],
        "per_type_rates": metrics["per_type_rates"].tolist(),
        "bootstrap_ci_lo": metrics["bootstrap_ci"][0],
        "bootstrap_ci_hi": metrics["bootstrap_ci"][1],
    }
    df = pd.DataFrame([row])
    path = store.path_for(config)
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
    return path


def run_prior_grid(
    cells: list[PriorCellConfig],
    store_root: Path,
    n_jobs: int = -1,
    backend: str = "loky",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run every cell (skipping ones already in the ``RunStore``), in
    parallel via ``joblib``, and return the concatenated tidy DataFrame.

    Determinism: identical to ``recovery.core.parallel.run_grid`` -- each
    cell's output depends only on its own ``PriorCellConfig``
    (deterministic factories + seed), workers share no state
    (``backend="loky"`` is process-based), so results do not depend on
    ``n_jobs`` or completion order.
    """
    store = RunStore(store_root)
    pending = [c for c in cells if not store.exists(c)]
    skipped = len(cells) - len(pending)
    if skipped:
        print(f"run_prior_grid: {skipped}/{len(cells)} cell(s) already in the RunStore, skipping", file=sys.stderr)

    if pending:
        parallel = joblib.Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")
        results = parallel(
            joblib.delayed(_run_one_prior_cell)(config, store_root) for config in pending
        )
        if show_progress:
            from tqdm import tqdm

            results = tqdm(results, total=len(pending), desc="prior_grid", file=sys.stderr)
        for _ in results:
            pass

    frames = [store.read(c) for c in cells]
    return pd.concat(frames, ignore_index=True)


def _run_one_discounted_cell(config: PriorCellConfig, store_root: Path) -> Path:
    """Worker for ``run_discounted_grid``: stores the FULL per-episode log
    via ``RunStore.write`` (Task 6's designed behavior, reused as-is) --
    unlike ``_run_one_prior_cell`` (which stores only a summary row for
    the much larger main grid), the discounted grid is small enough that
    full-log storage is affordable, and ``recovery_half_life`` (in
    ``experiments/prior_analysis.py``) needs per-episode granularity a
    summary row cannot provide."""
    store = RunStore(store_root)
    if store.exists(config):
        return store.path_for(config)
    env = config.environment_factory()
    policy = config.policy_factory()
    classifier = config.classifier_factory()
    log = run_episodes(env, policy, classifier, n_episodes=config.n_episodes, seed=config.seed)
    return store.write(config, log)


def run_discounted_grid(
    cells: list[PriorCellConfig],
    store_root: Path,
    n_jobs: int = -1,
    backend: str = "loky",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run every ``build_discounted_grid`` cell, storing FULL per-episode
    logs (see ``_run_one_discounted_cell``), and return a tidy metadata
    DataFrame (identity columns only -- callers needing per-episode data
    read it back via ``RunStore`` directly, e.g.
    ``experiments/prior_analysis.py::recovery_half_life``)."""
    store = RunStore(store_root)
    pending = [c for c in cells if not store.exists(c)]
    skipped = len(cells) - len(pending)
    if skipped:
        print(
            f"run_discounted_grid: {skipped}/{len(cells)} cell(s) already in the RunStore, skipping",
            file=sys.stderr,
        )

    if pending:
        parallel = joblib.Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")
        results = parallel(
            joblib.delayed(_run_one_discounted_cell)(config, store_root) for config in pending
        )
        if show_progress:
            from tqdm import tqdm

            results = tqdm(results, total=len(pending), desc="discounted_grid", file=sys.stderr)
        for _ in results:
            pass

    rows = [
        {
            "method": c.method, "env_id": c.env_id, "classifier_id": c.classifier_id, "seed": c.seed,
            "n_episodes": c.n_episodes, "strength": c.strength, "quality": c.quality,
            "prior_mean": c.prior_mean, "gamma": c.gamma, "prior_id": c.prior_id,
        }
        for c in cells
    ]
    return pd.DataFrame(rows)


DEFAULT_DISCOUNTED_OUTPUT = _BACKEND_DIR / "results" / "tables" / "discounted_prior_grid.parquet"
DEFAULT_DISCOUNTED_STORE_ROOT = _BACKEND_DIR / "results" / "runs" / "discounted_prior_grid"
DEFAULT_GAMMAS: list[float] = [1.0, 0.99, 0.9, 0.7]
SMOKE_GAMMAS: list[float] = [1.0, 0.9]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast validation run (<60s).")
    parser.add_argument(
        "--discounted", action="store_true",
        help="Run the DiscountedPriorThompson grid (recovery-half-life analysis) instead of the main grid.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=None)
    parser.add_argument("--qualities", type=int, nargs="+", default=None)
    parser.add_argument("--prior-means", type=float, nargs="+", default=None)
    parser.add_argument("--gammas", type=float, nargs="+", default=None)
    parser.add_argument("--environments", type=str, nargs="+", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--classifiers", type=str, nargs="+", default=None, choices=["perfect", "real"])
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="loky")
    args = parser.parse_args()

    if args.smoke:
        strengths = args.strengths or SMOKE_STRENGTHS
        qualities = args.qualities or SMOKE_QUALITIES
        prior_means = args.prior_means or SMOKE_PRIOR_MEANS
        gammas = args.gammas or SMOKE_GAMMAS
        environments = args.environments or SMOKE_ENVIRONMENTS
        horizons = args.horizons or SMOKE_HORIZONS
        classifiers = args.classifiers or SMOKE_CLASSIFIERS
        n_seeds = args.n_seeds or SMOKE_N_SEEDS
        base_seed = args.base_seed or SMOKE_BASE_SEED
    else:
        strengths = args.strengths or DEFAULT_STRENGTHS
        qualities = args.qualities or DEFAULT_QUALITIES
        prior_means = args.prior_means or DEFAULT_PRIOR_MEANS
        gammas = args.gammas or DEFAULT_GAMMAS
        environments = args.environments or DEFAULT_ENVIRONMENTS
        horizons = args.horizons or DEFAULT_HORIZONS
        classifiers = args.classifiers or DEFAULT_CLASSIFIERS
        n_seeds = args.n_seeds or DEFAULT_N_SEEDS
        base_seed = args.base_seed or DEFAULT_BASE_SEED

    if args.discounted:
        # See stationary_comparison.py's identical fix note: --smoke must
        # never default to the same path as a full run (Task 14 incident).
        output = args.output or (
            DEFAULT_DISCOUNTED_OUTPUT.with_name(DEFAULT_DISCOUNTED_OUTPUT.stem + "_smoke.parquet")
            if args.smoke else DEFAULT_DISCOUNTED_OUTPUT
        )
        store_root = args.store_root or DEFAULT_DISCOUNTED_STORE_ROOT
        drift_environments = [e for e in environments if e != "stationary"] or ["single_shift"]
        cells = build_discounted_grid(
            strengths=strengths, qualities=qualities, gammas=gammas, environments=drift_environments,
            horizons=horizons, classifiers=classifiers, n_seeds=n_seeds, base_seed=base_seed,
        )
        print(
            f"Discounted prior grid ({'smoke' if args.smoke else 'full'} mode): {len(cells)} cells "
            f"(strengths={strengths}, qualities={qualities}, gammas={gammas}, "
            f"environments={drift_environments}, horizons={horizons}, classifiers={classifiers}, "
            f"n_seeds={n_seeds})"
        )
        t0 = time.time()
        df = run_discounted_grid(cells, store_root=store_root, n_jobs=args.n_jobs, backend=args.backend)
        elapsed = time.time() - t0
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        print(f"Wrote {len(df)} rows to {output} in {elapsed:.2f}s (full logs in {store_root})")
        return

    # See stationary_comparison.py's identical fix note: --smoke must never
    # default to the same path as a full run (Task 14 incident -- this
    # exact file, prior_grid.parquet, was one of the six damaged in that
    # incident).
    output = args.output or (
        DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_smoke.parquet")
        if args.smoke else DEFAULT_OUTPUT
    )
    store_root = args.store_root or DEFAULT_STORE_ROOT
    cells = build_grid(
        strengths=strengths, qualities=qualities, prior_means=prior_means, environments=environments,
        horizons=horizons, classifiers=classifiers, n_seeds=n_seeds, base_seed=base_seed,
    )
    print(
        f"Prior grid ({'smoke' if args.smoke else 'full'} mode): {len(cells)} cells "
        f"(strengths={strengths}, qualities={qualities}, prior_means={prior_means}, "
        f"environments={environments}, horizons={horizons}, classifiers={classifiers}, n_seeds={n_seeds})"
    )

    t0 = time.time()
    df = run_prior_grid(cells, store_root=store_root, n_jobs=args.n_jobs, backend=args.backend)
    elapsed = time.time() - t0

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} rows to {output} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
