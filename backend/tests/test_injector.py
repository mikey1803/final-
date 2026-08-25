"""Sanity-check test for the FailureInjector.

Runs the injector 100 times with injection_rate=0.5 and prints the
distribution of injected failures by FailureType so you can eyeball whether
the injection logic is working roughly as expected.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.failures.injector import FailureInjector
from src.failures.taxonomy import FailureType


def test_injector_distribution():
    """Inject 100 times at 50 % rate and report the per-type breakdown."""
    injector = FailureInjector(injection_rate=0.5, seed=42)

    total_injected = 0
    type_counts: Counter[str] = Counter()

    for i in range(100):
        step_context = {"step_index": i, "tool_name": "web_search", "goal": "find info"}
        event = injector.inject(step_context)
        if event is not None:
            total_injected += 1
            type_counts[event.failure_type.value] += 1

    print(f"\n{'=' * 50}")
    print(f"  Injection sanity check  (n=100, rate=0.5)")
    print(f"{'=' * 50}")
    print(f"  Total injected : {total_injected} / 100  (expected ~50)")
    print(f"  Unique types   : {len(type_counts)} / {len(FailureType)}")
    print()
    for ftype in FailureType:
        count = type_counts.get(ftype.value, 0)
        bar = "#" * count
        print(f"  {ftype.value:28s} | {count:3d}  {bar}")
    print(f"{'=' * 50}\n")

    log = injector.get_injection_log()
    assert len(log) == total_injected, "Log length must match injected count"
    print("  PASS — injection log length matches count.\n")


if __name__ == "__main__":
    test_injector_distribution()
