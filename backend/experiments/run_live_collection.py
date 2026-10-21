"""Entry point for Phase 1 live-pilot collection.

    python experiments/run_live_collection.py --store results/live_pilot --n-runs 3

Runs the task list below against the live agent loop, writing captured
FailureEvents into a content-addressed EventStore. Re-running with the same
--store is safe -- events are content-addressed by (session_id, step_index,
tool_name, raw_error), so a fresh session_id per run means nothing collides,
and nothing here reruns/overwrites a prior session's events.

The task list here is a STARTER set only (one or two per category), not the
200-300 target -- task authoring, especially the genuinely-ambiguous and
out-of-scope ones meant to elicit the four agent-side failure types, is
domain judgement to be extended by hand, not mass-generated here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.agent_loop import run_task
from recovery.live.event_schema import Task
from recovery.live.event_store import EventStore
from recovery.live.pilot_report import compute_yield_report
from recovery.live.task_manifest import (
    assert_campaign, intended_vs_observed_distribution, load_manifest, require_frozen_manifest,
)
from recovery.live.tools import FlakyWrapper, health_check_all_tools

STARTER_TASKS: list[Task] = [
    Task(
        task_id="wiki_basic_01",
        goal="Look up a short summary of the city of Kolkata using the wikipedia tool.",
        category="tool_call_basic",
    ),
    Task(
        task_id="country_basic_01",
        goal="Find the capital city of India using the countriesnow tool.",
        category="tool_call_basic",
    ),
    Task(
        task_id="weather_basic_01",
        goal="Get the current weather for latitude 22.57, longitude 88.36 using the open_meteo tool.",
        category="tool_call_basic",
    ),
    Task(
        task_id="weather_basic_02",
        goal="Get the current weather for the location 'London' using the wttr tool.",
        category="tool_call_basic",
    ),
    Task(
        task_id="placeholder_basic_01",
        goal="Fetch todo item number 5 using the jsonplaceholder tool.",
        category="tool_call_basic",
    ),
    Task(
        task_id="ambiguous_01",
        goal="Find out about the weather situation there.",
        category="ambiguous",
        clarify_canned_answer="I mean the weather in Kolkata right now.",
    ),
    Task(
        task_id="out_of_scope_01",
        goal="Book a flight from Kolkata to London for tomorrow morning.",
        category="out_of_scope",
    ),
    Task(
        task_id="long_context_01",
        goal=(
            "First look up the capital of India using countriesnow. Then, "
            "using ONLY the city you just found (do not look it up again), "
            "get the current weather for that city using the wttr tool."
        ),
        category="long_context",
        steps=[
            "Find the capital city of India using the countriesnow tool.",
            "Get the current weather for the city found in the previous step, using the wttr tool.",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="EventStore root directory")
    parser.add_argument("--n-runs", type=int, default=1, help="How many times to run the task list")
    parser.add_argument("--injection-rate", type=float, default=0.20)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Frozen task-manifest JSON (recovery.live.task_manifest). Must already be "
             "committed with no uncommitted changes -- collection refuses to start "
             "otherwise. If omitted, uses the STARTER_TASKS list in this file instead "
             "(a code list, not a manifest -- no freeze check applies, and no "
             "intended-vs-observed report is produced).",
    )
    parser.add_argument(
        "--campaign", choices=["A", "B"], default=None,
        help="Post-pilot only: 'A' requires every manifest task to be targeted=True "
             "(confusion-matrix campaign), 'B' requires every task to be targeted=False "
             "(deployment-distribution campaign). Refuses to start if the manifest mixes "
             "the two. Omit for the 28-task pilot, which deliberately mixes both.",
    )
    parser.add_argument(
        "--inline-recovery", action="store_true",
        help="Task 21: on each captured tool-call failure, run one uniformly drawn recovery "
             "strategy (docs/live_pilot/strategy_definitions_v2.md) and store a RecoveryAttempt "
             "under <store>/recoveries/. Off by default: plain Phase 1 collection applies no strategy.",
    )
    parser.add_argument("--definition-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--session-seed", type=int, default=None,
                        help="Seed for inline strategy draws (default: derived from the clock, and printed)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task_ids to run (subset of the list)")
    args = parser.parse_args()

    if args.manifest is not None:
        require_frozen_manifest(args.manifest)
        tasks = load_manifest(args.manifest)
        print(f"Loaded frozen manifest: {args.manifest} ({len(tasks)} tasks)")
        if args.campaign is not None:
            assert_campaign(tasks, args.campaign)
            print(f"Campaign {args.campaign} check passed: manifest is single-campaign.")
    else:
        if args.campaign is not None:
            raise SystemExit("--campaign requires --manifest (STARTER_TASKS is pilot-only)")
        tasks = STARTER_TASKS
        print(f"No --manifest given; using the {len(tasks)}-task STARTER_TASKS list.")

    if args.tasks:
        wanted = args.tasks.split(",")
        tasks = [t for t in tasks if t.task_id in wanted]
        missing = set(wanted) - {t.task_id for t in tasks}
        if missing:
            raise SystemExit(f"unknown task_id(s): {sorted(missing)}")

    store = EventStore(args.store)
    wrapper = FlakyWrapper(injection_rate=args.injection_rate)
    recovery = None
    if args.inline_recovery:
        import time as _time
        import zlib

        from recovery.live.executors.llm import GroqChatClient
        from recovery.live.inline_recovery import InlineRecovery
        from recovery.live.recovery_attempt import RecoveryStore

        seed = args.session_seed if args.session_seed is not None else zlib.crc32(str(_time.time_ns()).encode())
        recovery = InlineRecovery(store=RecoveryStore(args.store), session_seed=seed,
                                  definition_version=args.definition_version, llm=GroqChatClient())
        print(f"Inline recovery ON: definition {args.definition_version}, session_seed={seed}, "
              f"arms={recovery.arms}, budget={recovery.budget.as_dict()}")

    print("Checking tool health (once, session start) ...")
    tool_health = health_check_all_tools()
    for tool_name, health in tool_health.items():
        status = "OK" if health["healthy"] else f"DOWN ({health['error_text']})"
        print(f"  {tool_name:16s} {status}")
    if not all(h["healthy"] for h in tool_health.values()):
        print("  WARNING: at least one tool is down -- failures on its peer via "
              "switch_tool may be confounded with this outage. See event.tool_health.")

    n_completed = 0
    n_failed = 0
    attempts_by_task_id: dict[str, int] = {}
    for run_idx in range(args.n_runs):
        for task in tasks:
            attempts_by_task_id[task.task_id] = attempts_by_task_id.get(task.task_id, 0) + 1
            outcome = run_task(task, store, wrapper, tool_health=tool_health, recovery=recovery)
            status = "OK" if outcome.completed else f"FAILED at step {outcome.steps_completed}"
            if outcome.recovery_attempt is not None:
                a = outcome.recovery_attempt
                status += (f" -> {a.strategy}: success={a.result['success']} "
                           f"({a.result['evidence']['reason']}, attempts={a.result['attempts']}, "
                           f"tokens={a.result['cost_tokens']})")
            elif outcome.recovery_skipped_reason:
                status += f" -> recovery skipped ({outcome.recovery_skipped_reason})"
            print(f"[run {run_idx}] {task.task_id:24s} {status}")
            if outcome.completed:
                n_completed += 1
            else:
                n_failed += 1

    store.rebuild_index()
    print(f"\nDone. completed={n_completed} failed(captured)={n_failed}")
    if recovery is not None:
        print(f"Inline recovery: {recovery.failure_index} attempt(s) written; skipped: {recovery.skipped or 'none'}")
    print("Progress:", store.labelling_progress())

    if args.manifest is not None:
        report = intended_vs_observed_distribution(tasks, store)
        print("\nIntended failure-type distribution (manifest):", report["intended"])
        print("Observed failure-type distribution (labelled events so far):", report["observed"])

        print("\nYield rate per type (n_observed / n_attempts, labelled events only):")
        yield_report = compute_yield_report(tasks, store, attempts_by_task_id)
        for ftype, row in sorted(yield_report.items()):
            if row["yield_rate"] is None:
                print(f"  {ftype:24s} attempts={row['n_attempts']:<4d} observed={row['n_observed']:<4d} "
                      f"yield=n/a (no attempts yet)")
            else:
                needed = (
                    f"need ~{row['attempts_needed_for_low']}-{row['attempts_needed_for_high']} "
                    f"attempts for n=30-45"
                    if row["attempts_needed_for_low"] is not None
                    else "yield=0 so far, cannot project attempts needed"
                )
                print(f"  {ftype:24s} attempts={row['n_attempts']:<4d} observed={row['n_observed']:<4d} "
                      f"yield={row['yield_rate']:.1%}  {needed}")


if __name__ == "__main__":
    main()
