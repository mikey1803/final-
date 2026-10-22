#!/usr/bin/env python
"""Run one live-pilot v3 session (docs/live_pilot/strategy_definitions_v3.md,
docs/live_pilot/v3_preregistration.md §2).

A session runs all six blocks (T, S, O2, O5, O20, D) once, in the seeded
``block_order``; the T block runs 10 passes over the task list and every other
block 5 (configuration choices, pre-reg §2). Session ``s`` uses
``session_seed = 231000 + s``; ``s = 0`` is reserved for smoke runs and is
never data.

    python experiments/run_live_v3.py --store results/live_pilot_v3 --session-index 1
    python experiments/run_live_v3.py --smoke --store <dir>_smoke     # s = 0, adaptive smoke passes

Tasks come from the frozen v3 manifest ``recovery/live/v3/data/v3_tasks.json``
(D-10); a data session refuses to start if it is untracked or modified.
Agent selection uses the v3 Layer-2 selector (pre-reg D-2); every LLM call
shares one client whose rate-limit headers drive the pacer (D-6), which
waits only at checkpoints outside every strategy's budget clock.
--smoke runs each block until it has at least --smoke-min-injected injected
failures (default 3) or --smoke-max-passes passes (default 6): smoke only,
never data.

Writes into ``--store``: events/, recoveries/ (v3 RecoveryAttempts),
v3_resends/, v3_episodes/, v3_injection_logs/ (genuine values -- instrument
only, never shown to the agent), v3_selector_log/ and v3_pacing/ (every
selector call and every pacing wait), v3_sessions/<seed>.json, and on halt H1
v3_quarantine/<seed>.json (exit code 3).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.live.event_store import EventStore  # noqa: E402
from recovery.live.recovery_attempt import RecoveryStore  # noqa: E402
from recovery.live.tools import health_check_all_tools  # noqa: E402
from recovery.live.v3 import DEFINITIONS_COMMIT, PREREGISTRATION_COMMIT  # noqa: E402
from recovery.live.v3.flaky import FlakyWrapperV3  # noqa: E402
from recovery.live.v3.libraries import frozen_libraries  # noqa: E402
from recovery.live.v3.loop import BLOCK_TYPES, BlockSpec, HaltH1, V3Dirs, run_episode_v3  # noqa: E402
from recovery.live.v3.pacing import RESERVE_REQUESTS, RESERVE_TOKENS, Pacer  # noqa: E402
from recovery.live.v3.selector import V3Selector  # noqa: E402
from recovery.live.v3.recovery import InlineRecoveryV3  # noqa: E402
from recovery.live.v3.seeds import session_seed_for, stream_rng  # noqa: E402
from recovery.live.v3.tasks import TASKS_MANIFEST, load_v3_tasks  # noqa: E402

PASSES_T = 10   # configuration choice (pre-reg §2)
PASSES_OTHER = 5  # configuration choice (pre-reg §2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(_BACKEND_DIR), capture_output=True, text=True)
    return r.stdout.strip()


def _append_jsonl(path: Path, rows: list, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, **extra}, default=str, ensure_ascii=False) + "\n")
    rows.clear()


def block_plan(session_seed: int, smoke: bool) -> tuple[list[BlockSpec], dict]:
    rng, rec = stream_rng(session_seed, "block_order")
    order = [int(i) for i in rng.permutation(len(BLOCK_TYPES))]
    blocks = []
    for position, type_index in enumerate(order):
        mode, n = BLOCK_TYPES[type_index]
        passes = 1 if smoke else (PASSES_T if mode == "transient" else PASSES_OTHER)
        blocks.append(BlockSpec(block_index=position, mode=mode, outage_N=n, passes=passes))
    return blocks, rec


def _write_log(dirs: V3Dirs, session_seed: int, block: BlockSpec, wrapper: FlakyWrapperV3) -> None:
    d = dirs.root / "v3_injection_logs"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"s{session_seed}_b{block.block_index}.jsonl", "w", encoding="utf-8") as fh:
        for e in wrapper.log:
            fh.write(json.dumps({**e, "session_seed": session_seed, "block_label": block.label},
                                default=str, ensure_ascii=False) + "\n")


def main() -> int:
    from recovery.live.executors.llm import GroqChatClient

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--session-index", type=int, default=None, help="s >= 1 for data; omitted with --smoke (s = 0)")
    ap.add_argument("--smoke", action="store_true", help="s = 0, one pass per block; store name must end in _smoke")
    ap.add_argument("--tasks", default=None, help="smoke only: comma-separated task_ids")
    ap.add_argument("--smoke-min-injected", type=int, default=3)
    ap.add_argument("--smoke-max-passes", type=int, default=6)
    args = ap.parse_args()

    if args.smoke:
        if args.session_index not in (None, 0):
            raise SystemExit("--smoke always uses session index 0 (never data)")
        if not args.store.name.endswith("_smoke"):
            raise SystemExit("--smoke needs a store directory whose name ends in _smoke (CLAUDE.md rule 5)")
        s = 0
    else:
        if args.session_index is None or args.session_index < 1:
            raise SystemExit("a data session needs --session-index >= 1")
        if args.tasks:
            raise SystemExit("--tasks is smoke-only: data sessions run the full frozen task list")
        s = args.session_index
    session_seed = session_seed_for(s)
    tasks = load_v3_tasks(require_frozen=not args.smoke)   # D-10: frozen manifest for data sessions
    if args.tasks:
        wanted = args.tasks.split(",")
        tasks = [t for t in tasks if t.task_id in wanted]

    dirs = V3Dirs(args.store)
    manifest_path = dirs.root / "v3_sessions" / f"{session_seed}.json"
    if manifest_path.exists():
        raise SystemExit(f"session seed {session_seed} already exists in {dirs.root}: seeds are never re-run")

    libs = frozen_libraries()   # hash-checked against the frozen constants
    blocks, order_rec = block_plan(session_seed, args.smoke)
    head, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain", "--", "recovery", "experiments"))
    manifest = {
        "session_seed": session_seed, "session_index": s, "smoke": args.smoke, "status": "running",
        "started_at": _now(), "code_commit": head, "code_dirty": dirty,
        "definitions_commit": DEFINITIONS_COMMIT, "preregistration_commit": PREREGISTRATION_COMMIT,
        "block_order": [{"block_index": b.block_index, "label": b.label, "mode": b.mode, "outage_N": b.outage_N,
                         "passes": b.passes} for b in blocks],
        "block_order_seed": order_rec, "tasks": [t.task_id for t in tasks], "libraries": libs.hashes,
        "task_manifest": {"path": "backend/recovery/live/v3/data/v3_tasks.json",
                          "blob": _git("rev-parse", "HEAD:./recovery/live/v3/data/v3_tasks.json"),
                          "dirty": bool(_git("status", "--porcelain", "--", str(TASKS_MANIFEST)))},
        "deviations_applied": ["D-2 selector", "D-3 family split", "D-4 judge input", "D-5 read timeout",
                               "D-6 pacing", "D-8 outage k", "D-9 tool_use_failed", "D-10 task manifest"],
        "pacing_reserves": {"tokens": RESERVE_TOKENS, "requests": RESERVE_REQUESTS},
        "smoke_rule": ({"min_injected": args.smoke_min_injected, "max_passes": args.smoke_max_passes}
                       if args.smoke else None),
    }
    dirs.write_json(dirs.root / "v3_sessions", str(session_seed), manifest)

    print("Checking tool health (once, session start) ...")
    tool_health = health_check_all_tools()
    for tool_name, h in tool_health.items():
        print(f"  {tool_name:16s} {'OK' if h['healthy'] else 'DOWN (' + str(h['error_text'])[:80] + ')'}")
    manifest["tool_health"] = tool_health

    events, recoveries = EventStore(dirs.root), RecoveryStore(dirs.root)
    client = GroqChatClient()                      # one client: its headers drive the pacer (D-6)
    selector = V3Selector(client)                  # D-2
    pacer = Pacer(client)
    manifest["selector"] = selector.settings()
    dirs.write_json(dirs.root / "v3_sessions", str(session_seed), manifest)
    recovery = InlineRecoveryV3(store=recoveries, session_seed=session_seed, llm=client)
    session = {"session_seed": session_seed, "session_index": s}
    sel_log = dirs.root / "v3_selector_log" / f"s{session_seed}.jsonl"
    pace_log = dirs.root / "v3_pacing" / f"s{session_seed}.jsonl"
    print(f"v3 session s={s} seed={session_seed} code={head[:7]}{' (dirty)' if dirty else ''} "
          f"blocks={[b.label + 'x' + str(b.passes) for b in blocks]}")
    episode_index = 0
    try:
        for block in blocks:
            wrapper = FlakyWrapperV3(mode=block.mode, session_seed=session_seed, block_index=block.block_index,
                                     outage_N=block.outage_N, libraries=libs)
            injected_in_block = 0
            try:
                p = 0
                while True:
                    if args.smoke:
                        if injected_in_block >= args.smoke_min_injected or p >= args.smoke_max_passes:
                            break
                    elif p >= block.passes:
                        break
                    for task in tasks:
                        rec = run_episode_v3(task, wrapper=wrapper, events=events, recoveries=recoveries,
                                             recovery=recovery, select=selector, block=block, session=session,
                                             episode_index=episode_index, tool_health=tool_health, dirs=dirs,
                                             pacer=pacer)
                        episode_index += 1
                        injected_in_block += sum(f["fault_origin"] == "injected" for f in rec["failures"])
                        tag = "OK" if rec["task_completed"] else ("aborted" if rec["aborted"] else "not completed")
                        fl = "; ".join(f"{f['fault_origin']}->{f['strategy']}:{'Y' if f['primary_outcome'] else 'N'}"
                                       for f in rec["failures"])
                        print(f"[{block.label:3s} pass {p}] {task.task_id:22s} {tag:13s} {fl}", flush=True)
                    p += 1
            finally:
                _write_log(dirs, session_seed, block, wrapper)
                _append_jsonl(sel_log, selector.log, {"block_index": block.block_index, "block_label": block.label})
                _append_jsonl(pace_log, pacer.log, {"block_index": block.block_index})
                if args.smoke:
                    manifest.setdefault("smoke_passes", {})[block.label] = p
    except HaltH1 as exc:
        manifest.update(status="halted_H1", finished_at=_now(), halt=str(exc))
        dirs.write_json(dirs.root / "v3_quarantine", str(session_seed),
                        {"session_seed": session_seed, "reason": "H1", "detail": str(exc), "at": _now()})
        dirs.write_json(dirs.root / "v3_sessions", str(session_seed), manifest)
        print(f"\nHALT H1: {exc}\nSession {session_seed} quarantined.")
        return 3
    manifest.update(status="complete", finished_at=_now(), episodes=episode_index,
                    failures=recovery.failure_index)
    dirs.write_json(dirs.root / "v3_sessions", str(session_seed), manifest)
    print(f"\nDone: {episode_index} episodes, {recovery.failure_index} failures with a drawn strategy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
