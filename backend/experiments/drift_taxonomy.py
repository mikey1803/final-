#!/usr/bin/env python
"""Grid experiment for the Task 8 drift taxonomy
(``recovery.envs.drift``): sweeps drift type, drift frequency, drift
magnitude, method, and seed, at a fixed horizon (``T=2000`` by default),
storing every cell's FULL per-episode log via
``recovery.core.parallel.RunStore`` (reused unchanged).

This script only builds configurations and composes
``recovery.core.episode.run_episodes`` with
``recovery.envs.drift``/``recovery.policies.*`` -- it contains no metric
computation of its own. ``results/tables/drift_taxonomy.parquet`` is
therefore a light METADATA/index table (one row per cell: which
drift_type/frequency/magnitude/method/seed it is, plus its
``RunStore`` content key) -- ``experiments/drift_analysis.py`` reads the
full per-episode logs back and computes every metric (Task 5's
``summarize`` plus ``recovery.metrics.drift_metrics``) at analysis time.
This mirrors ``experiments/prior_grid.py``'s
``build_discounted_grid``/``run_discounted_grid`` split (full-log
storage, metadata-only return value) -- applied to the WHOLE grid here
since this task's fixed ``T=2000`` horizon keeps full-log storage
affordable everywhere, not just for a small sub-grid.

Drift-type -> taxonomy mapping
---------------------------------
The grid's "drift type" axis is the four SUBSTANTIVE models
(``permutation``, ``monotone_decay``, ``partial_reorder``,
``gradual_rotation``); ``recovery.envs.drift.AnnouncedDrift`` (the
taxonomy's fifth class) is a behavior-preserving decorator, not a
generator of its own drift, so EVERY cell is wrapped in it uniformly
(harmless by construction -- see its docstring's "no other behavior
changes" guarantee) rather than treated as a separate grid value.

"Frequency" and "magnitude" are given a single, explicit, per-drift-type
meaning (documented in ``build_environment`` below) so that both axes
mean "how often" / "how much" consistently across the taxonomy:

- ``magnitude`` is ALWAYS the number of affected rows (failure types),
  out of ``n_types`` -- ``magnitude == n_types`` (7 in the canonical
  matrix) is a valid, deliberately-included grid point ("every type
  drifts").
- ``frequency`` is the number of episodes between repeated drift events
  for the three EVENT-DRIVEN models (``permutation``, ``partial_reorder``,
  ``monotone_decay``, which also uses ``frequency`` as its
  ``half_life``, so decay meaningfully progresses within its own
  re-evaluation period); for the one CONTINUOUS model
  (``gradual_rotation``, a single ramp, not a repeating event),
  ``frequency`` maps to the interpolation WINDOW length instead --
  documented explicitly here rather than silently reused with a
  different meaning.

Common Random Numbers
-----------------------
``env_seed`` (which affects both which rows drift and the drift content
itself -- e.g. the ``PermutationDrift`` RNG) is set to the cell's own
``seed``, identically for EVERY method -- exactly one environment
construction recipe per (drift_type, frequency, magnitude, seed), shared
across every method compared at that seed (mirroring
``experiments/prior_grid.py``'s ``build_environment``). A classifier
axis is deliberately OUT of scope here (Tasks 5/7 already characterize
classifier noise); every cell uses ``PerfectClassifier``.

Usage
-----
    python experiments/drift_taxonomy.py [--smoke] [options]
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
from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.envs.drift import (  # noqa: E402
    AnnouncedDrift,
    GradualRotationDrift,
    MonotoneDecayDrift,
    PartialReorderDrift,
    PermutationDrift,
)
from recovery.policies.adapters import (  # noqa: E402
    DiscountedThompsonPolicy,
    EpsilonGreedyPolicy,
    StaticRulePolicy,
    ThompsonPolicy,
    UCB1Policy,
    UniformRandomPolicy,
)
from recovery.policies.hybrid import DiscountedPriorThompson, PriorInitializedThompson  # noqa: E402

DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "drift_taxonomy.parquet"
DEFAULT_STORE_ROOT = _BACKEND_DIR / "results" / "runs" / "drift_taxonomy"

DEFAULT_DRIFT_TYPES: list[str] = ["permutation", "monotone_decay", "partial_reorder", "gradual_rotation"]
DEFAULT_FREQUENCIES: list[int] = [50, 100, 250, 500]
DEFAULT_MAGNITUDES: list[int] = [1, 2, 4, 7]
DEFAULT_METHODS: list[str] = [
    "static_rule", "uniform_random", "stock_thompson", "discounted_thompson",
    "ucb1", "epsilon_greedy", "pgts_s5", "pgts_s20", "pgts_s50", "discounted_pgts",
]
DEFAULT_N_SEEDS = 20
DEFAULT_BASE_SEED = 20_000
DEFAULT_HORIZON = 2000

# Methods that never learn/adapt their behavior -- `experiments/drift_analysis.py`
# reports "N/A" (never a fabricated number) for their recovery/adaptation
# statistics. Exported so the analysis script has one shared source of truth.
NON_ADAPTIVE_METHODS: frozenset = frozenset({"static_rule", "uniform_random"})

# Prior quality for every PGTS/DiscountedPGTS cell (5/7 correct -- a good
# but imperfect table, so PGTS has something genuine to demonstrate; not
# a swept axis in this experiment, unlike Task 7's prior_grid).
PGTS_N_CORRECT = 5
PGTS_PRIOR_MEAN = 0.7

SMOKE_DRIFT_TYPES: list[str] = ["permutation", "gradual_rotation"]
SMOKE_FREQUENCIES: list[int] = [50]
SMOKE_MAGNITUDES: list[int] = [2]
SMOKE_METHODS: list[str] = ["static_rule", "stock_thompson", "pgts_s20"]
SMOKE_N_SEEDS = 2
SMOKE_BASE_SEED = 20_000
SMOKE_HORIZON = 300


# ======================================================================
# Environment construction (shared across every method -- CRN)
# ======================================================================


def build_environment(
    drift_type: str,
    frequency: int,
    magnitude: int,
    matrix: np.ndarray,
    n_episodes: int,
    env_seed: int,
) -> AnnouncedDrift:
    """Build the (``AnnouncedDrift``-wrapped) drift environment for one
    ``(drift_type, frequency, magnitude, env_seed)`` cell. See the module
    docstring for the frequency/magnitude -> parameter mapping.
    """
    rng = np.random.default_rng(env_seed)
    n_types, n_strategies = matrix.shape
    rows = sorted(int(x) for x in rng.choice(n_types, size=min(magnitude, n_types), replace=False))

    if drift_type == "permutation":
        schedule = list(range(frequency, n_episodes, frequency))
        inner = PermutationDrift(matrix, schedule=schedule, affected_rows=rows, rng=rng)
    elif drift_type == "partial_reorder":
        schedule = list(range(frequency, n_episodes, frequency))
        inner = PartialReorderDrift(matrix, rows=rows, schedule=schedule)
    elif drift_type == "monotone_decay":
        schedule = list(range(frequency, n_episodes, frequency))
        strategy = int(rng.integers(0, n_strategies))
        base_value = float(min(matrix[r, strategy] for r in rows)) if rows else 0.0
        floor = base_value * 0.1
        inner = MonotoneDecayDrift(
            matrix, strategy=strategy, rows=rows, half_life=float(frequency),
            schedule=schedule, floor=floor,
        )
    elif drift_type == "gradual_rotation":
        target = matrix.copy()
        for r in rows:
            target[r] = rng.permutation(target[r])
        start_episode = n_episodes // 4
        inner = GradualRotationDrift(matrix, target, start_episode=start_episode, window=frequency)
    else:
        raise ValueError(f"unknown drift_type {drift_type!r}")

    return AnnouncedDrift(inner)


def _env_id(drift_type: str, frequency: int, magnitude: int) -> str:
    return f"{drift_type}_f{frequency}_m{magnitude}"


# ======================================================================
# Policy construction
# ======================================================================


def _policy_factory(method: str, matrix: np.ndarray, seed: int) -> Callable[[], Policy]:
    n_strategies = matrix.shape[1]

    if method == "static_rule":
        return lambda: StaticRulePolicy(BLIND_RULE, name="static_rule")
    if method == "uniform_random":
        return lambda: UniformRandomPolicy(seed=seed, name="uniform_random")
    if method == "stock_thompson":
        return lambda: ThompsonPolicy(seed=seed, name="stock_thompson")
    if method == "discounted_thompson":
        return lambda: DiscountedThompsonPolicy(gamma=0.99, seed=seed, name="discounted_thompson")
    if method == "ucb1":
        return lambda: UCB1Policy(seed=seed, name="ucb1")
    if method == "epsilon_greedy":
        return lambda: EpsilonGreedyPolicy(seed=seed, name="epsilon_greedy")
    if method.startswith("pgts_s"):
        strength = float(method[len("pgts_s"):])
        table = PriorInitializedThompson.from_quality(matrix, PGTS_N_CORRECT, seed)
        return lambda t=table, s=strength: PriorInitializedThompson(
            t, n_strategies, strength=s, prior_mean=PGTS_PRIOR_MEAN, name=method
        )
    if method == "discounted_pgts":
        table = PriorInitializedThompson.from_quality(matrix, PGTS_N_CORRECT, seed)
        return lambda t=table: DiscountedPriorThompson(
            t, n_strategies, strength=20.0, prior_mean=PGTS_PRIOR_MEAN, gamma=0.99, name=method
        )
    raise ValueError(f"unknown method {method!r}")


def _prior_id(method: str) -> str:
    if method.startswith("pgts_s") or method == "discounted_pgts":
        return f"q{PGTS_N_CORRECT}_m{PGTS_PRIOR_MEAN:g}"
    return "n/a"


# ======================================================================
# Grid configuration
# ======================================================================


@dataclass(frozen=True, eq=False)
class DriftCellConfig:
    """One grid cell. Mirrors ``recovery.core.parallel.RunConfig``'s
    content-addressing shape but carries an ``environment_factory``
    (built from ``drift_type``/``frequency``/``magnitude``/``seed``,
    never from a raw ``matrix``) so it can run against any of the four
    drift models via ``build_environment`` -- ``RunConfig`` itself always
    builds a stationary ``MatrixEnvironment`` internally and may not be
    modified (see ``recovery/core/parallel.py::_run_one_cell``).
    """

    method: str
    drift_type: str
    frequency: int
    magnitude: int
    seed: int
    n_episodes: int
    environment_factory: Callable[[], object]
    policy_factory: Callable[[], Policy]
    classifier_factory: Callable[[], object] = PerfectClassifier
    classifier_id: str = "perfect"
    prior_id: str = "n/a"

    @property
    def env_id(self) -> str:
        return _env_id(self.drift_type, self.frequency, self.magnitude)

    def content_key(self) -> str:
        import hashlib
        import json

        payload = {
            "method": self.method, "env_id": self.env_id, "seed": self.seed,
            "n_episodes": self.n_episodes, "classifier_id": self.classifier_id,
            "prior_id": self.prior_id,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


def build_cell(
    method: str, drift_type: str, frequency: int, magnitude: int, seed: int,
    n_episodes: int, matrix: np.ndarray = MATRIX,
) -> DriftCellConfig:
    """Build one ``DriftCellConfig`` -- the single place ``env_seed ==
    seed`` (Common Random Numbers, shared across every method) is
    enforced."""
    return DriftCellConfig(
        method=method, drift_type=drift_type, frequency=frequency, magnitude=magnitude,
        seed=seed, n_episodes=n_episodes,
        environment_factory=(
            lambda m=matrix, dt=drift_type, f=frequency, mag=magnitude, n=n_episodes, s=seed:
            build_environment(dt, f, mag, m, n, s)
        ),
        policy_factory=_policy_factory(method, matrix, seed),
        prior_id=_prior_id(method),
    )


def build_grid(
    drift_types: list[str], frequencies: list[int], magnitudes: list[int],
    methods: list[str], n_seeds: int, base_seed: int, n_episodes: int,
    matrix: np.ndarray = MATRIX,
) -> list[DriftCellConfig]:
    """Build the full ``drift_type x frequency x magnitude x method x seed`` grid."""
    cells: list[DriftCellConfig] = []
    for drift_type in drift_types:
        for frequency in frequencies:
            for magnitude in magnitudes:
                for seed_idx in range(n_seeds):
                    seed = base_seed + seed_idx
                    for method in methods:
                        cells.append(
                            build_cell(method, drift_type, frequency, magnitude, seed, n_episodes, matrix)
                        )
    return cells


# ======================================================================
# Execution
# ======================================================================


def _run_one_cell(config: DriftCellConfig, store_root: Path) -> Path:
    """Worker: run one cell and store its FULL per-episode log (see
    module docstring for why -- ``drift_metrics`` functions need
    per-episode granularity that no one-row summary could provide)."""
    store = RunStore(store_root)
    if store.exists(config):
        return store.path_for(config)

    env = config.environment_factory()
    policy = config.policy_factory()
    classifier = config.classifier_factory()
    log = run_episodes(env, policy, classifier, n_episodes=config.n_episodes, seed=config.seed)
    return store.write(config, log)


def run_drift_grid(
    cells: list[DriftCellConfig],
    store_root: Path,
    n_jobs: int = -1,
    backend: str = "loky",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run every cell (skipping ones already in the ``RunStore``), in
    parallel via ``joblib``, and return a tidy METADATA DataFrame (one
    row per cell: identity columns only -- see module docstring for why
    metrics are computed later, in ``experiments/drift_analysis.py``).
    """
    store = RunStore(store_root)
    pending = [c for c in cells if not store.exists(c)]
    skipped = len(cells) - len(pending)
    if skipped:
        print(f"run_drift_grid: {skipped}/{len(cells)} cell(s) already in the RunStore, skipping", file=sys.stderr)

    if pending:
        parallel = joblib.Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")
        results = parallel(joblib.delayed(_run_one_cell)(config, store_root) for config in pending)
        if show_progress:
            from tqdm import tqdm

            results = tqdm(results, total=len(pending), desc="drift_taxonomy", file=sys.stderr)
        for _ in results:
            pass

    rows = [
        {
            "method": c.method, "drift_type": c.drift_type, "frequency": c.frequency,
            "magnitude": c.magnitude, "env_id": c.env_id, "classifier_id": c.classifier_id,
            "seed": c.seed, "n_episodes": c.n_episodes, "prior_id": c.prior_id,
            "content_key": c.content_key(),
        }
        for c in cells
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast validation run (<90s).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--drift-types", type=str, nargs="+", default=None)
    parser.add_argument("--frequencies", type=int, nargs="+", default=None)
    parser.add_argument("--magnitudes", type=int, nargs="+", default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="loky")
    args = parser.parse_args()

    if args.smoke:
        drift_types = args.drift_types or SMOKE_DRIFT_TYPES
        frequencies = args.frequencies or SMOKE_FREQUENCIES
        magnitudes = args.magnitudes or SMOKE_MAGNITUDES
        methods = args.methods or SMOKE_METHODS
        n_seeds = args.n_seeds or SMOKE_N_SEEDS
        base_seed = args.base_seed or SMOKE_BASE_SEED
        horizon = args.horizon or SMOKE_HORIZON
    else:
        drift_types = args.drift_types or DEFAULT_DRIFT_TYPES
        frequencies = args.frequencies or DEFAULT_FREQUENCIES
        magnitudes = args.magnitudes or DEFAULT_MAGNITUDES
        methods = args.methods or DEFAULT_METHODS
        n_seeds = args.n_seeds or DEFAULT_N_SEEDS
        base_seed = args.base_seed or DEFAULT_BASE_SEED
        horizon = args.horizon or DEFAULT_HORIZON

    # See stationary_comparison.py's identical fix note: --smoke must never
    # default to the same path as a full run (Task 14 incident -- this
    # exact file, drift_taxonomy.parquet, was one of the six damaged in
    # that incident).
    output = args.output or (
        DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_smoke.parquet")
        if args.smoke else DEFAULT_OUTPUT
    )
    store_root = args.store_root or DEFAULT_STORE_ROOT

    cells = build_grid(drift_types, frequencies, magnitudes, methods, n_seeds, base_seed, horizon)
    print(
        f"Drift taxonomy grid ({'smoke' if args.smoke else 'full'} mode): {len(cells)} cells "
        f"(drift_types={drift_types}, frequencies={frequencies}, magnitudes={magnitudes}, "
        f"methods={methods}, n_seeds={n_seeds}, horizon={horizon})"
    )

    t0 = time.time()
    df = run_drift_grid(cells, store_root=store_root, n_jobs=args.n_jobs, backend=args.backend)
    elapsed = time.time() - t0

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df)} rows to {output} in {elapsed:.2f}s (full logs in {store_root})")


if __name__ == "__main__":
    main()
