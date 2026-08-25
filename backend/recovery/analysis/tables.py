"""Shared per-trial runner and table-writing helpers for the Task 4A
stationary-experiment scripts (Tables V, VI, VII).

This is the ONE place ``recovery.core.episode.run_episodes`` is called from
outside the kernel itself and its own tests -- every experiment script in
``experiments/`` goes through ``run_trial`` here rather than re-implementing
any part of the episode loop, so there is exactly one call site to audit
for "experiment scripts contain zero simulation logic" (see
docs/architecture.md). Analytic normalization always uses
``recovery.theory.analytic`` (Task 1's closed forms), never a simulated
Oracle/UniformRandom endpoint. Every trial is validated with
``recovery.guards.invariants.run_all_guards`` (Task 3) before being
returned -- a violated invariant raises and aborts the run rather than
silently producing a row.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recovery.core.episode import run_episodes
from recovery.core.protocols import Classifier, Policy
from recovery.envs.matrix_env import MatrixEnvironment
from recovery.guards.invariants import run_all_guards
from recovery.theory.analytic import normalize

from src.evaluation.stats_utils import bootstrap_ci as _bootstrap_ci

__all__ = [
    "LATE_STAGE_FRAC",
    "N_BOOTSTRAP",
    "bootstrap_ci",
    "run_trial",
    "write_tidy_table",
    "aggregate_and_check",
]

LATE_STAGE_FRAC = 0.2
N_BOOTSTRAP = 5000


def bootstrap_ci(values, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI over ``values``, 5000 resamples.

    Reuses ``src.evaluation.stats_utils.bootstrap_ci`` (already-validated
    percentile bootstrap implementation) rather than re-implementing it,
    pinned to this task's required sample count and confidence level. No
    p-values or significance testing -- out of scope for Task 4A.

    Parameters
    ----------
    values : sequence of float
        Per-seed values to bootstrap over.
    seed : int, default=0
        Seed for the bootstrap resampling RNG (not the simulation).

    Returns
    -------
    tuple[float, float]
        (lower, upper) 95% percentile bootstrap interval.
    """
    return _bootstrap_ci(values, n_resamples=N_BOOTSTRAP, ci=0.95, seed=seed)


def run_trial(
    matrix: np.ndarray,
    policy: Policy,
    classifier: Classifier,
    n_episodes: int,
    seed: int,
    pi: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    late_stage_frac: float = LATE_STAGE_FRAC,
    check_guards: bool = True,
) -> dict:
    """Run one (matrix, policy, classifier, seed) trial through the kernel.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (N, K)
        The ground-truth success matrix for this trial.
    policy : recovery.core.protocols.Policy
        The policy under test (a fresh instance, or one that has not yet
        been run -- ``run_episodes`` calls ``policy.reset()`` internally).
    classifier : recovery.core.protocols.Classifier
        The classifier supplying (possibly noisy) predicted types.
    n_episodes : int
        Total episodes to run.
    seed : int
        Top-level seed, expanded into the kernel's four independent RNG
        streams (see ``recovery.core.rng``).
    pi : numpy.ndarray, shape (N,), optional
        If this trial's policy is a KNOWN fixed policy (e.g.
        ``recovery.envs.canonical.BLIND_RULE`` or an oracle argmax rule),
        pass its action vector here to enable
        ``check_static_policy_ceiling``. Omit for adaptive/randomized
        policies (see that guard's docstring: only valid for fixed
        policies).
    C : numpy.ndarray, shape (N, N), optional
        If ``classifier`` is a fixed confusion matrix (e.g.
        ``ConfusionClassifier``), pass it here (together with ``pi``) to
        enable ``check_confusion_prediction``.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform (both in the kernel and in the
        analytic normalization).
    late_stage_frac : float, default=0.2
        Fraction of the run's tail used for the reported rate (paper's
        "late-stage success": final 20%).
    check_guards : bool, default=True
        Whether to run Task 3's guards on this trial's result before
        returning it. Always ``True`` in practice; the flag exists for
        exceptional/diagnostic use, not for routine experiment code.

    Returns
    -------
    dict
        ``{"raw_rate": float, "normalized": float, "n_episodes": int}``.
        Callers add their own ``method``/condition/``seed`` columns.

    Raises
    ------
    recovery.guards.invariants.InvariantViolation
        If ``check_guards`` is True and any applicable guard's bound is
        violated -- propagates uncaught, aborting the run (guards raise
        instead of warn; see docs/architecture.md, "Guard layer").

    Notes
    -----
    ``check_normalized_range`` is deliberately NOT run here, even though
    ``check_guards=True``. Its bound (default ``[-1.0, 1.05]``) is fixed
    rather than shrinking with sample size -- by design (see its
    docstring: "normalized scores are typically reported as one summary
    number per experiment configuration"). At a single trial's late-stage
    sample size (e.g. 100 episodes at ``n_episodes=500``,
    ``late_stage_frac=0.2``), a normalized score's standard error is large
    enough (empirically observed: a single Table V trial reached
    ``normalized=1.071`` from ordinary sampling noise around the oracle
    ceiling) that this fixed bound produces false positives when checked
    per-trial. This is a property of HOW ``check_normalized_range`` is
    being invoked here, not a flaw in the guard -- see
    ``aggregate_and_check`` below, which runs it correctly, on the
    across-seed aggregate the guard was designed for.
    """
    env = MatrixEnvironment(matrix)
    log = run_episodes(env, policy, classifier, n_episodes=n_episodes, seed=seed, type_prior=p)

    start_frac = 1.0 - late_stage_frac
    raw_rate = log.success_rate(start_frac=start_frac, end_frac=1.0)
    normalized = normalize(raw_rate, matrix, p)

    if check_guards:
        late_n = n_episodes - int(start_frac * n_episodes)
        run_all_guards(raw_rate, matrix, late_n, pi=pi, C=C, p=p, normalized=None)

    return {"raw_rate": raw_rate, "normalized": normalized, "n_episodes": n_episodes}


def aggregate_and_check(
    df: pd.DataFrame,
    group_cols: list[str],
    matrix: np.ndarray,
    pi_by_group: Optional[dict] = None,
    C: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    late_stage_frac: float = LATE_STAGE_FRAC,
) -> pd.DataFrame:
    """Aggregate ``df`` across seeds per ``group_cols`` and re-validate.

    Every row in ``df`` passed here must share the SAME ground-truth
    ``matrix`` (a single P) -- callers with per-matrix conditions (e.g.
    Table VI's ``matrix_idx``) must call this once per matrix, including
    ``matrix_idx`` (or an equivalent single-matrix selector) in
    ``group_cols``, not mix multiple matrices into one call.

    This is where ``check_normalized_range`` is actually exercised (see
    ``run_trial``'s Notes): each group's MEAN ``raw_rate``/``normalized``
    across its seeds is a well-conditioned aggregate statistic --
    ``n_seeds`` late-stage windows pooled together -- unlike a single
    trial's late-stage rate.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy per-trial rows (as produced by repeated ``run_trial`` calls),
        containing at least ``group_cols``, ``raw_rate``, ``normalized``,
        ``n_episodes``.
    group_cols : list of str
        Columns identifying one (matrix, condition) cell to aggregate
        over seeds within -- e.g. ``["method"]`` for Table V,
        ``["algorithm"]`` for one Table VI matrix, ``["noise_level"]`` for
        Table VII.
    matrix : numpy.ndarray, shape (N, K)
        The single ground-truth matrix shared by every row in ``df``.
    pi_by_group : dict, optional
        Maps a group key (the ``group_cols`` values, as a scalar if
        ``len(group_cols) == 1`` else a tuple) to a fixed-policy action
        vector, enabling ``check_static_policy_ceiling`` for that group.
        Groups absent from this dict get ``pi=None``.
    C : numpy.ndarray, shape (N, N), optional
        Confusion matrix shared by every row, if applicable.
    p : numpy.ndarray, shape (N,), optional
        Type prior. Defaults to uniform.
    late_stage_frac : float, default=0.2
        Must match what ``run_trial`` used to produce ``df``'s rates.

    Returns
    -------
    pandas.DataFrame
        One row per group: ``group_cols`` + ``raw_rate`` (mean),
        ``normalized`` (mean), ``n_seeds``.

    Raises
    ------
    recovery.guards.invariants.InvariantViolation
        If any group's aggregate violates an applicable guard.
    """
    agg = (
        df.groupby(group_cols)
        .agg(raw_rate=("raw_rate", "mean"), normalized=("normalized", "mean"), n_seeds=("seed", "count"))
        .reset_index()
    )
    late_episodes_per_seed = int(late_stage_frac * int(df["n_episodes"].iloc[0]))

    for _, row in agg.iterrows():
        key = row[group_cols[0]] if len(group_cols) == 1 else tuple(row[c] for c in group_cols)
        pi = (pi_by_group or {}).get(key)
        total_late_n = int(row["n_seeds"]) * late_episodes_per_seed
        run_all_guards(
            float(row["raw_rate"]), matrix, total_late_n,
            pi=pi, C=C, p=p, normalized=float(row["normalized"]),
        )

    return agg


def write_tidy_table(rows: list[dict], path: Path) -> pd.DataFrame:
    """Assemble tidy rows into a DataFrame and write it to a parquet file.

    Parameters
    ----------
    rows : list of dict
        One dict per row (method, condition columns, seed, raw_rate,
        normalized, n_episodes -- no run_digest, no Hydra metadata).
    path : pathlib.Path
        Output path; parent directories are created if needed.

    Returns
    -------
    pandas.DataFrame
        The assembled (and already-written) table.
    """
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
