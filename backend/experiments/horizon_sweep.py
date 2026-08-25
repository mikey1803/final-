#!/usr/bin/env python
"""Horizon sweep: how every method's performance -- and the
static-vs-adaptive crossover -- depend on the interaction budget T.

This is the project's central experiment (see docs/experiments/horizon.md
for the full write-up). The paper's stationary claim was measured at a
single fixed budget (500 episodes); this sweep makes the budget an
explicit axis instead.

KEY DESIGN: PREFIX EVALUATION, NOT PER-HORIZON SIMULATION
-----------------------------------------------------------
For each (method, environment, seed, classifier) cell, exactly ONE
trajectory of length ``--t-max`` is run through the shared kernel (via
``recovery.core.parallel.run_grid`` -- this script contains no
simulation logic of its own). Every horizon in the grid is then evaluated
as a PREFIX of that one trajectory (``recovery.metrics.prefix``), never
as a separate simulation -- ~7x cheaper than one run per horizon, and
critically, this makes every horizon's curve PAIRED: all horizons for a
given seed share the same random draws, so the curve has no
cross-horizon noise (see ``tests/test_horizon.py``'s prefix-equivalence
test).

MEMORY
------
``run_grid`` writes each cell's full episode log to disk as soon as its
worker finishes (``RunStore.write``), so the parallel phase never holds
more than ``n_jobs`` logs in memory at once. The subsequent
``evaluate_horizons`` pass (single-process, after ``run_grid`` returns)
processes one ``(env_id, classifier_id, seed)`` group at a time and holds
at most two full trajectories in memory per group (that group's
``static_blind`` baseline, plus whichever other method is currently
being scored) -- never the whole grid's logs simultaneously, regardless
of total grid size.

Usage
-----
    python experiments/horizon_sweep.py [--smoke] [--estimate] [options]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.adapter import TrainedClassifierAdapter  # noqa: E402
from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.parallel import RunConfig, RunStore, run_grid  # noqa: E402
from recovery.core.protocols import Policy  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.metrics.parity import time_to_parity  # noqa: E402
from recovery.metrics.pertype import worst_type_rate  # noqa: E402
from recovery.metrics.prefix import PrefixLog  # noqa: E402
from recovery.metrics.prefix import late_stage_rate as prefix_late_stage_rate  # noqa: E402
from recovery.metrics.prefix import prefix_view  # noqa: E402
from recovery.metrics.regret import final_cumulative_regret, normalized_cumulative_regret  # noqa: E402
from recovery.policies.adapters import (  # noqa: E402
    ContextualCostAwarePolicy,
    DecayingEpsilonGreedyPolicy,
    DiscountedThompsonPolicy,
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    NaiveRetryPolicy,
    OraclePolicy,
    StaticRulePolicy,
    ThompsonPolicy,
    UCB1Policy,
    UniformRandomPolicy,
)
from recovery.theory.analytic import normalize  # noqa: E402
from src.evaluation.classifier_context import load_or_train_classifier  # noqa: E402
from src.evaluation.matrix_family import sample_matrix  # noqa: E402

DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "horizon_sweep.parquet"
DEFAULT_STORE_ROOT = _BACKEND_DIR / "results" / "runs" / "horizon_sweep"

HORIZONS: list[int] = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
LATE_STAGE_FRAC = 0.2
TIME_TO_PARITY_STABILITY_WINDOW = 50

DEFAULT_T_MAX = 100_000
DEFAULT_N_SEEDS = 30
DEFAULT_BASE_SEED = 4000
DEFAULT_METHODS: list[str] = [
    "static_blind", "thompson", "discounted_thompson", "ucb1", "linucb",
    "epsilon_greedy", "decaying_epsilon", "naive_retry", "uniform_random",
    "contextual_cost_aware", "oracle",
]
DEFAULT_CLASSIFIERS: list[str] = ["perfect", "real"]
DEFAULT_N_GAP_MATRICES = 25
DEFAULT_GAP = 0.20

SMOKE_HORIZONS: list[int] = [100, 300, 1000, 2000]
SMOKE_T_MAX = 2000
SMOKE_N_SEEDS = 3
SMOKE_BASE_SEED = 4000
SMOKE_METHODS: list[str] = ["static_blind", "thompson", "ucb1"]
SMOKE_CLASSIFIERS: list[str] = ["perfect"]
SMOKE_N_GAP_MATRICES = 0  # canonical only -- keeps --smoke well under 60s


def _policy_factory(method: str, seed: int, matrix: np.ndarray) -> Callable[[], Policy]:
    """Build a zero-arg factory for ``method``, closing over ``seed``/``matrix``.

    A factory, not an instance, because ``run_grid`` calls it INSIDE the
    worker (once per cell) -- ``recovery.core.episode.run_episodes`` needs
    a fresh, unused ``Policy`` (it calls ``policy.reset()`` itself, but a
    stateful legacy-wrapped policy still must not be shared/mutated
    across cells running concurrently in different processes).
    """
    if method == "static_blind":
        return lambda: StaticRulePolicy(BLIND_RULE, name="static_blind")
    if method == "oracle":
        return lambda: OraclePolicy(matrix, name="oracle")
    if method == "thompson":
        return lambda: ThompsonPolicy(seed=seed, name="thompson")
    if method == "discounted_thompson":
        return lambda: DiscountedThompsonPolicy(seed=seed, name="discounted_thompson")
    if method == "ucb1":
        return lambda: UCB1Policy(seed=seed, name="ucb1")
    if method == "linucb":
        return lambda: LinUCBPolicy(seed=seed, name="linucb")
    if method == "epsilon_greedy":
        return lambda: EpsilonGreedyPolicy(seed=seed, name="epsilon_greedy")
    if method == "decaying_epsilon":
        return lambda: DecayingEpsilonGreedyPolicy(seed=seed, name="decaying_epsilon")
    if method == "naive_retry":
        return lambda: NaiveRetryPolicy(name="naive_retry")
    if method == "uniform_random":
        return lambda: UniformRandomPolicy(seed=seed, name="uniform_random")
    if method == "contextual_cost_aware":
        return lambda: ContextualCostAwarePolicy(seed=seed, name="contextual_cost_aware")
    raise ValueError(f"unknown method {method!r}")


def _classifier_factory(classifier_id: str) -> Callable[[], object]:
    """Build a zero-arg classifier factory. For ``"real"``, the trained
    classifier is loaded/trained ONCE here (not once per cell) and closed
    over by the returned factory -- ``run_grid`` still calls the factory
    fresh inside each worker (constructing a new lightweight
    ``TrainedClassifierAdapter`` wrapper per cell), but every cell reuses
    the SAME already-loaded model rather than re-loading it from disk."""
    if classifier_id == "perfect":
        return lambda: PerfectClassifier()
    if classifier_id == "real":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn version-mismatch warning on the pickled artifact
            clf = load_or_train_classifier()
        return lambda: TrainedClassifierAdapter(clf)
    raise ValueError(f"unknown classifier_id {classifier_id!r}")


def build_grid(
    methods: list[str],
    classifiers: list[str],
    n_seeds: int,
    base_seed: int,
    t_max: int,
    include_canonical: bool = True,
    n_gap_matrices: int = 0,
    gap: float = DEFAULT_GAP,
    gap_seed: Optional[int] = None,
) -> list[RunConfig]:
    """Build the full ``list[RunConfig]`` grid: every
    (method, environment, classifier, seed) combination.

    Parameters
    ----------
    methods : list of str
        Method labels (see ``_policy_factory``).
    classifiers : list of str
        ``"perfect"`` and/or ``"real"``.
    n_seeds : int
        Seeds per cell, drawn as ``base_seed + seed_idx``.
    base_seed : int
        First seed.
    t_max : int
        Episodes to run per cell (the longest horizon evaluated).
    include_canonical : bool, default=True
        Whether to include the canonical MATRIX environment.
    n_gap_matrices : int, default=0
        Number of additional sampled matrices at optimal-arm ``gap`` to
        include (Task 4A's ``src.evaluation.matrix_family.sample_matrix``).
    gap : float, default=0.20
        Optimal-arm gap for the sampled matrices.
    gap_seed : int, optional
        Seed for sampling the gap matrices themselves (distinct from the
        per-cell simulation seeds). Defaults to ``base_seed + 999_999``.

    Returns
    -------
    list of RunConfig
    """
    envs: list[tuple[str, np.ndarray]] = []
    if include_canonical:
        envs.append(("canonical", MATRIX))
    if n_gap_matrices > 0:
        matrix_rng = np.random.default_rng(gap_seed if gap_seed is not None else base_seed + 999_999)
        for idx in range(n_gap_matrices):
            m = sample_matrix(gap=gap, rng=matrix_rng)
            envs.append((f"gap{gap:.2f}_idx{idx:02d}", m))

    cells: list[RunConfig] = []
    for env_id, matrix in envs:
        for classifier_id in classifiers:
            classifier_factory = _classifier_factory(classifier_id)
            for method in methods:
                for seed_idx in range(n_seeds):
                    seed = base_seed + seed_idx
                    cells.append(
                        RunConfig(
                            method=method,
                            env_id=env_id,
                            matrix=matrix,
                            seed=seed,
                            n_episodes=t_max,
                            policy_factory=_policy_factory(method, seed, matrix),
                            classifier_factory=classifier_factory,
                            classifier_id=classifier_id,
                        )
                    )
    return cells


def _read_log(store: RunStore, config: RunConfig) -> PrefixLog:
    df = store.read(config)
    return PrefixLog(
        true_type=df["true_type"].to_numpy(),
        action=df["action"].to_numpy(),
        reward=df["reward"].to_numpy(),
    )


def evaluate_horizons(
    cells: list[RunConfig],
    paths: list[Path],
    horizons: list[int],
    store_root: Path,
) -> pd.DataFrame:
    """Evaluate every horizon in ``horizons`` as a prefix of each cell's
    stored T_max trajectory, producing one tidy row per
    (method, env_id, classifier_id, seed, T).

    Processes ONE (env_id, classifier_id, seed) group at a time, and
    within a group holds at most two full trajectories in memory at once
    (the group's ``static_blind`` baseline, plus whichever OTHER method's
    log is currently being scored) -- never the whole grid's raw logs
    simultaneously, regardless of how many cells are in ``cells`` (see
    "Memory" in the module docstring / docs/experiments/horizon.md).
    Everything else this function produces (the returned rows) is tiny
    (a handful of floats per horizon).

    ``time_to_parity`` (vs. that group's ``static_blind`` baseline) is
    computed ONCE per (method, env_id, classifier_id, seed) from the
    FULL T_max trajectories -- it is a property of the pair of complete
    trajectories, not something that is re-derived per horizon -- and
    broadcast across that group's T rows (see docs/experiments/horizon.md,
    "Two crossover notions").

    Raises
    ------
    ValueError
        If any horizon in ``horizons`` exceeds a cell's stored
        ``n_episodes`` (never silently truncated -- see
        ``recovery.metrics.prefix.prefix_view``), or if a non-"static_blind"
        method's ``time_to_parity`` baseline is missing for a given
        (env_id, classifier_id, seed).
    """
    store = RunStore(store_root)

    max_t_max = max(config.n_episodes for config in cells)
    for T in horizons:
        if T > max_t_max:
            raise ValueError(
                f"horizon T={T} exceeds the largest T_max simulated ({max_t_max}); "
                f"prefix evaluation never silently truncates"
            )

    groups: dict[tuple, list[RunConfig]] = {}
    for config in cells:
        group_key = (config.env_id, config.classifier_id, config.seed)
        groups.setdefault(group_key, []).append(config)

    rows: list[dict] = []
    for (env_id, classifier_id, seed), group_cells in groups.items():
        baseline_config = next((c for c in group_cells if c.method == "static_blind"), None)
        non_baseline_methods = [c.method for c in group_cells if c.method != "static_blind"]
        if non_baseline_methods and baseline_config is None:
            raise ValueError(
                f"time_to_parity requires a 'static_blind' baseline at "
                f"(env_id={env_id!r}, classifier_id={classifier_id!r}, seed={seed}), but "
                f"none was found in this grid -- include 'static_blind' in --methods"
            )
        baseline_log = _read_log(store, baseline_config) if baseline_config is not None else None

        for config in group_cells:
            matrix = config.matrix
            n_types = matrix.shape[0]
            log = baseline_log if config.method == "static_blind" else _read_log(store, config)

            if config.method == "static_blind" or baseline_log is None:
                crossing = float("nan")
            else:
                crossing = time_to_parity(
                    log, seed, baseline_log, seed, stability_window=TIME_TO_PARITY_STABILITY_WINDOW
                )

            for T in horizons:
                prefix = prefix_view(log, T)
                rate = prefix_late_stage_rate(prefix, late_stage_frac=LATE_STAGE_FRAC)
                rows.append(
                    {
                        "method": config.method,
                        "env_id": env_id,
                        "classifier_id": classifier_id,
                        "seed": seed,
                        "T": T,
                        "late_stage_rate": rate,
                        "normalized": normalize(rate, matrix),
                        "cumulative_regret": final_cumulative_regret(prefix, matrix),
                        "normalized_cumulative_regret": normalized_cumulative_regret(prefix, matrix),
                        "time_to_parity": crossing,
                        "worst_type_rate": worst_type_rate(prefix, n_types=n_types),
                    }
                )
            # `log` (a non-baseline method's full trajectory) goes out of
            # scope at the end of this loop body and is discarded before
            # the next method in the group is read -- only `baseline_log`
            # persists for the group's duration.

    return pd.DataFrame(rows)


def estimate_wall_clock(
    cells: list[RunConfig], n_jobs: int, calibration_episodes: int = 2000
) -> None:
    """Print an estimated total wall-clock time for running ``cells`` and exit.

    Calibrates throughput separately per ``classifier_id`` (a real
    classifier is ~2-3 orders of magnitude slower per episode than the
    synthetic ones -- see ``recovery.classify.adapter.TrainedClassifierAdapter``'s
    docstring) using a short representative run, then scales linearly by
    total episodes and divides by ``n_jobs``.
    """
    import os

    from recovery.core.episode import run_episodes
    from recovery.envs.matrix_env import MatrixEnvironment

    effective_n_jobs = os.cpu_count() if n_jobs in (-1, 0) else n_jobs
    print(f"Estimating wall-clock for {len(cells)} cells at n_jobs={effective_n_jobs}...")

    by_classifier: dict[str, int] = {}
    for c in cells:
        by_classifier[c.classifier_id] = by_classifier.get(c.classifier_id, 0) + c.n_episodes

    total_seconds = 0.0
    for classifier_id, total_episodes in by_classifier.items():
        sample_cell = next(c for c in cells if c.classifier_id == classifier_id)
        env = MatrixEnvironment(sample_cell.matrix)
        policy = sample_cell.policy_factory()
        classifier = sample_cell.classifier_factory()
        n_calib = min(calibration_episodes, sample_cell.n_episodes)
        t0 = time.perf_counter()
        run_episodes(env, policy, classifier, n_episodes=n_calib, seed=sample_cell.seed)
        elapsed = time.perf_counter() - t0
        throughput = n_calib / elapsed if elapsed > 0 else float("inf")
        classifier_seconds = total_episodes / throughput / effective_n_jobs
        total_seconds += classifier_seconds
        print(
            f"  classifier_id={classifier_id!r}: {throughput:,.0f} episodes/sec (calibrated on "
            f"{n_calib} episodes), {total_episodes:,} total episodes across all cells "
            f"-> ~{classifier_seconds / 60:.1f} min at n_jobs={effective_n_jobs}"
        )

    print(f"\nEstimated total wall-clock: ~{total_seconds / 60:.1f} min (~{total_seconds / 3600:.2f} hours)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast validation run (<60s).")
    parser.add_argument("--estimate", action="store_true", help="Print estimated wall-clock and exit.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--classifiers", nargs="+", default=None, choices=["perfect", "real"])
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--n-gap-matrices", type=int, default=None)
    parser.add_argument("--no-canonical", action="store_true", help="Exclude the canonical environment.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="loky")
    args = parser.parse_args()

    if args.output is None:
        # See stationary_comparison.py's identical fix note: --smoke must
        # never default to the same path as a full run (Task 14 incident --
        # this exact file, horizon_sweep.parquet, was one of the six
        # damaged in that incident).
        args.output = (
            DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_smoke.parquet")
            if args.smoke else DEFAULT_OUTPUT
        )

    if args.smoke:
        methods = args.methods or SMOKE_METHODS
        classifiers = args.classifiers or SMOKE_CLASSIFIERS
        n_seeds = args.n_seeds or SMOKE_N_SEEDS
        base_seed = args.base_seed or SMOKE_BASE_SEED
        t_max = args.t_max or SMOKE_T_MAX
        horizons = args.horizons or SMOKE_HORIZONS
        n_gap_matrices = args.n_gap_matrices if args.n_gap_matrices is not None else SMOKE_N_GAP_MATRICES
    else:
        methods = args.methods or DEFAULT_METHODS
        classifiers = args.classifiers or DEFAULT_CLASSIFIERS
        n_seeds = args.n_seeds or DEFAULT_N_SEEDS
        base_seed = args.base_seed or DEFAULT_BASE_SEED
        t_max = args.t_max or DEFAULT_T_MAX
        horizons = args.horizons or HORIZONS
        n_gap_matrices = args.n_gap_matrices if args.n_gap_matrices is not None else DEFAULT_N_GAP_MATRICES

    for T in horizons:
        if T > t_max:
            raise ValueError(f"--horizons contains T={T} > --t-max={t_max}; horizons cannot exceed t_max")

    cells = build_grid(
        methods=methods, classifiers=classifiers, n_seeds=n_seeds, base_seed=base_seed,
        t_max=t_max, include_canonical=not args.no_canonical, n_gap_matrices=n_gap_matrices,
    )
    print(
        f"Horizon sweep ({'smoke' if args.smoke else 'full'} mode): {len(cells)} cells "
        f"(methods={methods}, classifiers={classifiers}, n_seeds={n_seeds}, t_max={t_max}, "
        f"n_gap_matrices={n_gap_matrices})"
    )

    if args.estimate:
        estimate_wall_clock(cells, n_jobs=args.n_jobs)
        return

    t0 = time.time()
    paths = run_grid(cells, store_root=args.store_root, n_jobs=args.n_jobs, backend=args.backend)
    run_elapsed = time.time() - t0
    print(f"run_grid: {len(paths)} cells ready in {run_elapsed:.2f}s")

    df = evaluate_horizons(cells, paths, horizons, args.store_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    total_elapsed = time.time() - t0
    print(f"Wrote {len(df)} rows to {args.output} in {total_elapsed:.2f}s total")


if __name__ == "__main__":
    main()
