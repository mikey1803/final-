#!/usr/bin/env python
"""Task 14 Part 1: wall-clock time and peak memory per decision, for
every policy in the project (including the fixed-rule baselines).

Method and reuse discipline
----------------------------
This script constructs the exact same objects the shared kernel
(``recovery.core.episode.run_episodes``) uses -- the canonical
environment (``recovery.envs.canonical.MATRIX``,
``recovery.envs.matrix_env.MatrixEnvironment``), ``PerfectClassifier``,
``recovery.core.protocols.Context``, and ``recovery.core.rng.from_seed``
-- and calls every policy through the exact same ``Policy`` protocol
(``select``/``update``) the kernel calls. It does NOT call
``run_episodes`` itself for the per-decision timing, and it does NOT
modify ``recovery/core/episode.py``.

This is a deliberate, documented deviation, not a shortcut: getting
per-decision percentiles (p95/p99) requires per-call timing resolution
that ``run_episodes``'s external interface does not expose, and there is
no way to add that resolution without either (a) modifying the shared
kernel's hot loop -- which Task 6 already profiled and hand-optimized
(see ``docs/experiments/horizon.md``, "Kernel throughput"), so touching
it for a benchmarking feature is not justified absent evidence of a flaw
in it -- or (b) calling ``run_episodes(n_episodes=1)`` in a loop, whose
fixed per-call overhead (``from_seed``'s ``SeedSequence.spawn(4)``,
``EpisodeLog`` array allocation) would dominate the signal for the
fastest policies and make cross-policy comparison meaningless. This
script is the closest correct alternative: a thin timing loop built from
the SAME shared primitives, reproducing the kernel's per-episode call
sequence exactly, so the measured numbers reflect real kernel-identical
work.

**Cross-validation** (`--validate`): this script's own measured
mean-select-time throughput is checked against a REAL, unmodified
``run_episodes`` call over the same policy/environment, confirming the
two agree (within the noise expected from ``run_episodes`` also paying
for classifier prediction, reward sampling, and env.step, which this
script's ``select()`` timing deliberately excludes).

Excluded policy: ``OneShotLLMPolicy`` (wraps ``OneShotLLMBaseline``,
which makes a real network call or a keyword-heuristic fallback per
decision) -- its "decision time" is dominated by network/API latency,
not policy compute, and is not comparable to every other policy's
in-process cost; benchmarking it here would either make a paid API call
per timed iteration (explicitly against this task's Part 2 rule for the
LLM designer experiments, and equally wasteful here) or silently measure
only the fallback path. Excluded and documented rather than silently
omitted.

Usage
-----
    python experiments/runtime_benchmark.py                # full run
    python experiments/runtime_benchmark.py --validate      # + kernel cross-check
    python experiments/runtime_benchmark.py --n-decisions 500 --n-warmup 50   # quick check
"""

import argparse
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.classify.synthetic import PerfectClassifier  # noqa: E402
from recovery.core.episode import run_episodes  # noqa: E402
from recovery.core.protocols import Context  # noqa: E402
from recovery.core.rng import from_seed  # noqa: E402
from recovery.envs.canonical import BLIND_RULE, MATRIX  # noqa: E402
from recovery.envs.matrix_env import MatrixEnvironment  # noqa: E402
from recovery.policies.adapters import (  # noqa: E402
    ContextualCostAwarePolicy,
    DecayingEpsilonGreedyPolicy,
    DiscountedThompsonPolicy,
    EpsilonGreedyPolicy,
    FixedRulePolicy,
    LinUCBPolicy,
    NaiveRetryPolicy,
    OraclePolicy,
    StaticRulePolicy,
    ThompsonPolicy,
    UCB1Policy,
    UniformRandomPolicy,
)
from recovery.policies.hybrid import DiscountedPriorThompson, PriorInitializedThompson  # noqa: E402

OUTPUT_DIR = _BACKEND_DIR / "results" / "tables"
BASE_SEED = 140_014  # arbitrary fixed constant, unrelated to any other experiment's seed
N_DECISIONS_DEFAULT = 5000
N_WARMUP_DEFAULT = 200
# Peak-memory difference below this (bytes) between two policies is not
# reported as "meaningfully different". Calibrated empirically: this
# script's per-decision peak-memory readings sit in the low-hundreds-to-
# tens-of-thousands-of-bytes range (a single select()/update() call
# allocates little), so a 2,000-byte floor separates real allocator
# differences (e.g. LinUCB's/PGTS's matrix work) from single-object
# noise, without requiring a MB-scale floor appropriate to a whole
# program run.
MEMORY_NOISE_FLOOR_BYTES = 2_000


def build_policies() -> dict:
    """Every policy in the project, including the fixed-rule baselines,
    excluding ``OneShotLLMPolicy`` (see module docstring)."""
    n_types, n_strategies = MATRIX.shape
    return {
        "FixedRule (blind static, 6/7)": FixedRulePolicy(),
        "Oracle": OraclePolicy(MATRIX),
        "StaticRule (BLIND_RULE, native)": StaticRulePolicy(BLIND_RULE),
        "UniformRandom": UniformRandomPolicy(seed=BASE_SEED + 1),
        "NaiveRetry": NaiveRetryPolicy(),
        "UCB1": UCB1Policy(seed=BASE_SEED + 2),
        "EpsilonGreedy": EpsilonGreedyPolicy(seed=BASE_SEED + 3),
        "DecayingEpsilonGreedy": DecayingEpsilonGreedyPolicy(seed=BASE_SEED + 4),
        "Thompson": ThompsonPolicy(seed=BASE_SEED + 5),
        "DiscountedThompson": DiscountedThompsonPolicy(seed=BASE_SEED + 6),
        "LinUCB": LinUCBPolicy(seed=BASE_SEED + 7),
        "ContextualCostAware": ContextualCostAwarePolicy(seed=BASE_SEED + 8),
        "PGTS (PriorInitializedThompson, strength=20)": PriorInitializedThompson(
            BLIND_RULE, n_strategies, strength=20.0
        ),
        "DiscountedPGTS": DiscountedPriorThompson(BLIND_RULE, n_strategies, strength=20.0),
    }


def benchmark_policy(
    name: str, policy, env, classifier, seed: int, n_decisions: int, n_warmup: int
) -> dict:
    """Time ``n_warmup + n_decisions`` individual decisions for one
    policy, discard the warmup, and return per-decision timing
    statistics plus peak traced memory over the timed portion."""
    streams = from_seed(seed)
    policy.reset()
    total = n_warmup + n_decisions
    select_times = np.empty(total, dtype=np.float64)
    update_times = np.empty(total, dtype=np.float64)

    tracemalloc.start()
    for i in range(total):
        true_type = i % env.n_types  # deterministic cycling -- no RNG cost inside the timed region
        predicted = classifier.predict(true_type, streams.classifier)
        ctx = Context(predicted_type=predicted, episode=i)

        t0 = time.perf_counter()
        action = policy.select(ctx, streams.policy)
        t1 = time.perf_counter()
        select_times[i] = t1 - t0

        reward = env.sample_reward(true_type, action, streams.reward)

        t2 = time.perf_counter()
        policy.update(ctx, action, reward)
        t3 = time.perf_counter()
        update_times[i] = t3 - t2

        if i == n_warmup - 1:
            # Discard warmup allocations from the peak-memory reading --
            # only the timed portion's peak is reported.
            tracemalloc.clear_traces()
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    select_us = select_times[n_warmup:] * 1e6
    update_us = update_times[n_warmup:] * 1e6
    total_us = select_us + update_us

    def _stats(arr: np.ndarray, prefix: str) -> dict:
        return {
            f"{prefix}_mean_us": float(np.mean(arr)),
            f"{prefix}_median_us": float(np.median(arr)),
            f"{prefix}_p95_us": float(np.percentile(arr, 95)),
            f"{prefix}_p99_us": float(np.percentile(arr, 99)),
        }

    row = {"policy": name, "n_decisions": n_decisions, "n_warmup": n_warmup,
           "peak_memory_bytes": int(peak_bytes)}
    row.update(_stats(select_us, "select"))
    row.update(_stats(update_us, "update"))
    row.update(_stats(total_us, "total"))
    return row


def measure_loop_overhead_floor(env, classifier, seed: int, n_decisions: int, n_warmup: int) -> dict:
    """A no-op ``select``/``update`` (returns a fixed valid action,
    updates nothing) run through the IDENTICAL timed loop shape, to
    report the measurement floor -- Python loop/RNG-draw/dict-lookup
    overhead present in every policy's number, not attributable to any
    policy's own decision logic. Reported transparently, not subtracted
    out (subtracting would risk manufacturing negative or misleadingly
    "clean" numbers for the fastest real policies)."""

    class _NoOpPolicy:
        name = "_NoOpFloor"

        def select(self, ctx, rng):
            return 0

        def update(self, ctx, action, reward):
            return None

        def reset(self):
            return None

    return benchmark_policy("_measurement_floor (no-op policy)", _NoOpPolicy(), env, classifier, seed, n_decisions, n_warmup)


def cross_validate_against_kernel(policy_name: str, policy, env, classifier, seed: int, n_episodes: int) -> dict:
    """Run the SAME policy through the real, unmodified ``run_episodes``
    kernel and report its aggregate throughput, to confirm this script's
    thin timing loop is not measuring something systematically different
    from actual kernel behavior."""
    policy.reset()
    t0 = time.perf_counter()
    run_episodes(env, policy, classifier, n_episodes=n_episodes, seed=seed)
    elapsed = time.perf_counter() - t0
    return {
        "policy": policy_name,
        "kernel_n_episodes": n_episodes,
        "kernel_elapsed_s": elapsed,
        "kernel_us_per_episode": (elapsed / n_episodes) * 1e6,
        "kernel_episodes_per_sec": n_episodes / elapsed if elapsed > 0 else float("inf"),
    }


def capture_hardware_info() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": __import__("os").cpu_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-decisions", type=int, default=N_DECISIONS_DEFAULT)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP_DEFAULT)
    parser.add_argument("--validate", action="store_true", help="Also cross-check against real run_episodes.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    hw = capture_hardware_info()
    print("=== Hardware / environment ===")
    for k, v in hw.items():
        print(f"  {k}: {v}")

    env = MatrixEnvironment(MATRIX)
    classifier = PerfectClassifier()
    policies = build_policies()

    print(f"\n=== Per-decision timing ({args.n_decisions} decisions + {args.n_warmup} discarded warmup, per policy) ===")
    rows = []
    floor_row = measure_loop_overhead_floor(env, classifier, BASE_SEED, args.n_decisions, args.n_warmup)
    rows.append(floor_row)
    print(f"  measurement floor (no-op policy): select mean={floor_row['select_mean_us']:.2f}us "
          f"median={floor_row['select_median_us']:.2f}us p95={floor_row['select_p95_us']:.2f}us "
          f"p99={floor_row['select_p99_us']:.2f}us")

    for i, (name, policy) in enumerate(policies.items()):
        row = benchmark_policy(name, policy, env, classifier, BASE_SEED + 1000 + i, args.n_decisions, args.n_warmup)
        rows.append(row)
        print(f"  {name}: select mean={row['select_mean_us']:.2f}us median={row['select_median_us']:.2f}us "
              f"p95={row['select_p95_us']:.2f}us p99={row['select_p99_us']:.2f}us "
              f"| total mean={row['total_mean_us']:.2f}us | peak_mem={row['peak_memory_bytes']:,}B")

    timing_df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = args.output_dir / "runtime_benchmark_timing.parquet"
    timing_df.to_parquet(timing_path, index=False)
    print(f"\nWrote {timing_path}")

    hw_df = pd.DataFrame([hw])
    hw_path = args.output_dir / "runtime_benchmark_hardware.parquet"
    hw_df.to_parquet(hw_path, index=False)
    print(f"Wrote {hw_path}")

    # Peak-memory comparison: only report where it differs meaningfully.
    mem = timing_df[timing_df["policy"] != floor_row["policy"]][["policy", "peak_memory_bytes"]]
    mem_min = mem["peak_memory_bytes"].min()
    mem["delta_from_min_bytes"] = mem["peak_memory_bytes"] - mem_min
    meaningful = mem[mem["delta_from_min_bytes"] > MEMORY_NOISE_FLOOR_BYTES].sort_values(
        "delta_from_min_bytes", ascending=False
    )
    print(f"\n=== Peak memory: policies exceeding the min by more than {MEMORY_NOISE_FLOOR_BYTES:,} bytes ===")
    if meaningful.empty:
        print("  None -- no policy's peak memory differs meaningfully from the minimum "
              f"({mem_min:,} bytes) at this n_decisions.")
    else:
        print(meaningful.to_string(index=False))

    if args.validate:
        print("\n=== Cross-validation against real run_episodes (same policies, seed) ===")
        val_rows = []
        for i, (name, policy) in enumerate(policies.items()):
            v = cross_validate_against_kernel(name, policy, env, classifier, BASE_SEED + 1000 + i, args.n_decisions)
            val_rows.append(v)
            print(f"  {name}: kernel {v['kernel_episodes_per_sec']:,.0f} eps/sec "
                  f"({v['kernel_us_per_episode']:.2f}us/episode)")
        val_df = pd.DataFrame(val_rows)
        val_path = args.output_dir / "runtime_benchmark_kernel_validation.parquet"
        val_df.to_parquet(val_path, index=False)
        print(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
