#!/usr/bin/env python
"""Table V: stationary method comparison, regenerated through the Task 2
shared simulation kernel.

Every method from the historical Phase 1 comparison
(``src/evaluation/harness.py::_make_selectors``), run against the canonical
matrix under perfect typing (no classifier noise), 500 episodes, 50 seeds,
late-stage success = final 20%. Contains zero simulation logic itself --
every trial goes through ``recovery.analysis.tables.run_trial``, which is
the sole caller of ``recovery.core.episode.run_episodes`` and
``recovery.guards.invariants.run_all_guards`` in this task. See
docs/reconciliation_stationary.md for how these numbers compare to the
historical `results/phase1/comparison_summary.csv`.

Usage
-----
    python experiments/stationary_comparison.py [--smoke] [--output PATH]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.analysis.tables import aggregate_and_check, run_trial, write_tidy_table  # noqa: E402
from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.protocols import Policy  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX, STRATEGY_NAMES  # noqa: E402
from recovery.policies.adapters import (  # noqa: E402
    ContextualCostAwarePolicy,
    DecayingEpsilonGreedyPolicy,
    EpsilonGreedyPolicy,
    FixedRulePolicy,
    LinUCBPolicy,
    NaiveRetryPolicy,
    OneShotLLMPolicy,
    OraclePolicy,
    StaticRulePolicy,
    ThompsonPolicy,
    UCB1Policy,
    UniformRandomPolicy,
)

DEFAULT_OUTPUT = _BACKEND_DIR / "results" / "tables" / "table_v.parquet"
DEFAULT_N_EPISODES = 500
DEFAULT_N_SEEDS = 50
DEFAULT_BASE_SEED = 123

# Smoke mode reduces the number of SEEDS (breadth), not the episode count:
# check_normalized_range (recovery.guards.invariants) uses a fixed, non-
# shrinking bound ([-1.0, 1.05]) rather than a statistical one, so a very
# small late-stage window (e.g. n_episodes=20 -> a 4-episode late-stage
# slice) can legitimately swing outside that bound on pure sampling noise
# even for a correct implementation. Keeping n_episodes at the full paper
# scale (500 -> a 100-episode late-stage window) keeps every guard's
# statistical assumptions valid while --smoke still finishes in seconds by
# cutting seeds from 50 to 2.
SMOKE_N_EPISODES = DEFAULT_N_EPISODES
SMOKE_N_SEEDS = 2

_RETRY_ACTION = STRATEGY_NAMES.index("retry")
_NAIVE_RETRY_PI = np.full(MATRIX.shape[0], _RETRY_ACTION, dtype=np.int64)
_ORACLE_PI = MATRIX.argmax(axis=1)


def _disable_live_llm_calls() -> None:
    """Force OneShotLLMPolicy onto its keyword fallback (no network call).

    A live Groq call takes ~14s -- far outside the <10s smoke budget.
    ``src.agent.llm_client.get_groq_api_key()`` reads ``os.environ`` (via
    ``python-dotenv``, which does not override already-set variables), so
    setting the key to an empty string before it is first read reliably
    disables the live path without editing any existing file. Normal mode
    leaves this untouched, so it reproduces the paper's real-LLM setting.
    """
    os.environ["GROQ_API_KEY"] = ""


# Method name -> (seed_offset, factory(seed) -> Policy, pi-or-None).
# `pi` is supplied only for methods with a KNOWN fixed action mapping
# (enables the tighter check_static_policy_ceiling guard); adaptive /
# genuinely randomized / not-known-in-advance methods pass None.
_METHODS: dict[str, tuple[int, Callable[[int], Policy], "np.ndarray | None"]] = {
    "Adaptive (UCB1)": (0, lambda s: UCB1Policy(seed=s), None),
    "Adaptive (Thompson)": (500, lambda s: ThompsonPolicy(seed=s), None),
    "Adaptive (e-greedy)": (600, lambda s: EpsilonGreedyPolicy(epsilon=0.15, seed=s), None),
    "Adaptive (Decaying e)": (700, lambda s: DecayingEpsilonGreedyPolicy(seed=s), None),
    "Contextual Cost-Aware": (
        1000, lambda s: ContextualCostAwarePolicy(lambda_cost=0.3, alpha=0.25, seed=s), None,
    ),
    "Naive Retry": (0, lambda s: NaiveRetryPolicy(), _NAIVE_RETRY_PI),
    "Fixed Rule": (0, lambda s: FixedRulePolicy(), BLIND_RULE),
    "UniformRandom": (0, lambda s: UniformRandomPolicy(seed=s), None),
    "One-Shot LLM": (0, lambda s: OneShotLLMPolicy(seed=s), None),
    "LinUCB": (1500, lambda s: LinUCBPolicy(alpha=1.0, seed=s), None),
    "Oracle": (0, lambda s: OraclePolicy(MATRIX), _ORACLE_PI),
}


def run_table_v(n_episodes: int, n_seeds: int, base_seed: int) -> list[dict]:
    rows: list[dict] = []
    for method_name, (offset, factory, pi) in _METHODS.items():
        for seed_idx in range(n_seeds):
            trial_seed = base_seed + seed_idx * 1000
            policy_seed = trial_seed + offset
            policy = factory(policy_seed)
            result = run_trial(
                MATRIX,
                policy,
                PerfectClassifier(),
                n_episodes=n_episodes,
                seed=trial_seed,
                pi=pi,
            )
            rows.append({"method": method_name, "seed": seed_idx, **result})
        print(f"  {method_name}: done ({n_seeds} seeds)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast validation run (<10s).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        # --smoke must never default to the same path as a full run -- a
        # bare `--smoke` invocation used to silently overwrite the real
        # results/tables/table_v.parquet with smoke-scale data (the Task 14
        # operational incident; see docs/PROJECT_STATUS.md). Restoring the
        # data once didn't close this: the shared default was still live
        # until this fix.
        args.output = (
            DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_smoke.parquet")
            if args.smoke else DEFAULT_OUTPUT
        )

    if args.smoke:
        _disable_live_llm_calls()
        n_episodes, n_seeds = SMOKE_N_EPISODES, SMOKE_N_SEEDS
    else:
        n_episodes, n_seeds = DEFAULT_N_EPISODES, DEFAULT_N_SEEDS

    print(f"Table V: stationary comparison ({'smoke' if args.smoke else 'full'} mode)")
    print(f"  n_episodes={n_episodes}  n_seeds={n_seeds}  methods={len(_METHODS)}")

    t0 = time.time()
    rows = run_table_v(n_episodes, n_seeds, DEFAULT_BASE_SEED)
    elapsed = time.time() - t0

    df = write_tidy_table(rows, args.output)

    pi_by_group = {name: pi for name, (_, _, pi) in _METHODS.items() if pi is not None}
    agg = aggregate_and_check(df, ["method"], MATRIX, pi_by_group=pi_by_group)

    print(f"\nWrote {len(df)} rows to {args.output} in {elapsed:.2f}s")
    print(agg.sort_values("normalized", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
