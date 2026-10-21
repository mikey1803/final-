"""Integration test for Part A (OutcomeMemory) and Part B (harness)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.store import OutcomeMemory
from src.recovery.selector import SelectorFactory
from src.failures.taxonomy import FailureType
from src.recovery.strategies import RecoveryStrategy


def test_memory_basic(tmp_path):
    """Record a few rows, read them back, check aggregation."""
    mem = OutcomeMemory(db_path=tmp_path / "test_memory.db")
    mem.reset()

    mem.record("tool_timeout", "retry", True, 1.0, 1)
    mem.record("tool_timeout", "retry", False, 1.0, 2)
    mem.record("tool_error", "switch_tool", True, 1.0, 3)

    stats = mem.get_stats()
    print("\n  [memory] get_stats (all):")
    print(stats.to_string(index=False))

    stats_filtered = mem.get_stats(failure_type="tool_timeout")
    assert len(stats_filtered) == 1
    assert stats_filtered.iloc[0]["attempts"] == 2
    assert stats_filtered.iloc[0]["successes"] == 1
    print("  [memory] filtered stats OK")

    history = mem.get_full_history()
    assert len(history) == 3
    print(f"  [memory] full history: {len(history)} rows OK")

    mem.reset()
    assert len(mem.get_full_history()) == 0
    print("  [memory] reset OK")
    mem.close()
    Path(mem._path).unlink(missing_ok=True)


def test_memory_selector_roundtrip(tmp_path):
    """Record outcomes via selector, then reload into a fresh selector."""
    db_path = tmp_path / "test_roundtrip.db"
    mem = OutcomeMemory(db_path=db_path)
    mem.reset()

    sel1 = SelectorFactory.create("ucb1", seed=42, memory=mem)
    import random
    rng = random.Random(42)
    for i in range(50):
        ft = rng.choice(list(FailureType))
        strat = sel1.select_strategy(ft)
        from src.failures.taxonomy import FailureEvent
        from src.recovery.strategies import execute_strategy
        event = FailureEvent(failure_type=ft, raw_error="test")
        outcome = execute_strategy(strat, event, {"tool_name": "x", "step_index": i}, rng=rng)
        sel1.update(ft, strat, success=outcome["success"], episode_number=i)

    stats1 = sel1.get_stats_table()
    print("\n  [roundtrip] selector1 stats (last 3 rows):")
    print(stats1.iloc[-3:].to_string(float_format="{:.2f}".format))

    # Reload into a brand-new selector
    sel2 = SelectorFactory.create("ucb1", seed=999)  # different seed
    mem.load_into_selector(sel2)
    stats2 = sel2.get_stats_table()

    # Stats should match (same total attempts + successes)
    import pandas as pd
    assert stats1.equals(stats2), "Reloaded selector stats must match original"
    print("  [roundtrip] reload matches — PASS")

    mem.close()
    Path(db_path).unlink(missing_ok=True)


def test_harness_smoke(tmp_path):
    """Run a small comparison (50 episodes x 2 trials) to verify the pipeline.

    Must write to tmp_path, never the real data_dir/results_dir -- this test
    used to default results_dir to the real results/phase1/ (harness.py's
    fallback when results_dir=None), silently overwriting the actual n=500
    x 50 comparison_summary.csv with n=2-trial smoke numbers on every pytest
    run. Caught 2026-08-20 while auditing OneShotLLMBaseline result
    provenance -- see docs/PAPER_PREP.md's provenance-audit note.
    """
    from src.evaluation.harness import run_and_report

    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results_phase1"
    summary = run_and_report(
        num_episodes=50, num_trials=2, data_dir=data_dir, results_dir=results_dir
    )

    print("  [harness] summary table:")
    print(summary.to_string(float_format="{:.4f}".format))

    assert len(summary) == 11, "Must have 11 methods"
    # Rate columns are true proportions and must lie in [0, 1]. The ci95_*
    # columns are normal-approximation bounds, which can dip slightly
    # outside [0, 1] for low-rate methods at small trial counts — so we
    # only check that they exist, not that they're bounded.
    for col in [
        "mean_cumulative_success_rate",
        "std_cumulative_success_rate",
        "mean_late_stage_success_rate",
        "std_late_stage_success_rate",
    ]:
        assert col in summary.columns
        assert (summary[col] >= 0).all()
        assert (summary[col] <= 1.0).all()
    for col in [
        "ci95_lower_cumulative",
        "ci95_upper_cumulative",
        "ci95_lower_late_stage",
        "ci95_upper_late_stage",
    ]:
        assert col in summary.columns

    # Oracle/UniformRandom-normalized score: Oracle == 1.0 (ceiling),
    # UniformRandom == 0.0 (floor) by construction.
    assert summary.loc["Oracle", "normalized_cumulative_score"] == 1.0
    assert summary.loc["Oracle", "normalized_late_stage_score"] == 1.0
    assert summary.loc["UniformRandom", "normalized_cumulative_score"] == 0.0
    assert summary.loc["UniformRandom", "normalized_late_stage_score"] == 0.0
    print("  [harness] basic assertions PASS\n")

    # Verify effect-size table was computed
    assert "thompson_effect_sizes" in summary.attrs, "Cohen's d table must be attached"
    print("  [harness] Cohen's d table present — PASS")

    # Verify files were created
    assert (data_dir / "method_comparison.png").exists(), "Plot must exist"
    assert (data_dir / "evaluation_results.csv").exists(), "CSV must exist"
    print("  [harness] output files exist — PASS\n")


if __name__ == "__main__":
    test_memory_basic()
    test_memory_selector_roundtrip()
    test_harness_smoke()
    print("All tests passed.\n")
