#!/usr/bin/env python
"""Selector empty-reply rate before and after D-2 (v3 pre-registration §11),
on the same step goals: every distinct step goal of the v3 task list
(STARTER_TASKS), ``--reps`` times each, through

- **before:** the Layer 1 selector ``src.agent.llm_client.parse_tool_request``
  (gpt-oss-20b, ``max_tokens=200``, no reasoning effort) -- read-only, called as is;
- **after:** the v3 selector ``recovery.live.v3.selector.V3Selector``
  (byte-identical prompt, ``max_tokens=300``, ``reasoning_effort="low"``).

The calls are interleaved goal by goal, so both selectors face the same
moments. Rate-limit pacing uses the v3 client's headers, plus a fixed gap
before each Layer 1 call, whose client does not expose them. A live
measurement: no seeds, and re-running gives new observations.

    python experiments/selector_empty_rate_v3.py --out <file.json> [--reps 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.live.executors.llm import GroqChatClient  # noqa: E402
from recovery.live.tools import TOOL_DESCRIPTIONS  # noqa: E402
from recovery.live.v3.pacing import Pacer  # noqa: E402
from recovery.live.v3.selector import V3Selector  # noqa: E402

LAYER1_GAP_S = 4.0


def classify_layer1(goal: str) -> dict:
    from src.agent.llm_client import parse_tool_request
    try:
        parse_tool_request(goal, TOOL_DESCRIPTIONS)
        return {"outcome": "parsed"}
    except ValueError as exc:
        empty = str(exc) == "LLM did not return valid JSON: ''"
        return {"outcome": "empty" if empty else "unparseable", "detail": str(exc)[:200]}
    except Exception as exc:
        return {"outcome": "provider_error", "detail": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> int:
    from experiments.run_live_collection import STARTER_TASKS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    goals = []
    for t in STARTER_TASKS:
        for g in (t.steps or [t.goal]):
            if g not in goals:
                goals.append(g)
    client = GroqChatClient()
    v3 = V3Selector(client)
    pacer = Pacer(client)
    rows = []
    for rep in range(args.reps):
        for gi, goal in enumerate(goals):
            time.sleep(LAYER1_GAP_S)
            rows.append({"selector": "layer1", "goal_index": gi, "rep": rep, **classify_layer1(goal)})
            pacer.before("selection")
            try:
                v3(goal)
                out = {"outcome": "parsed"}
            except ValueError as exc:
                out = {"outcome": v3.log[-1]["outcome"], "detail": str(exc)[:200]}
            except Exception as exc:
                out = {"outcome": "provider_error", "detail": f"{type(exc).__name__}: {exc}"[:200]}
            rows.append({"selector": "v3", "goal_index": gi, "rep": rep, **out})
            print(f"rep {rep} goal {gi}: layer1={rows[-2]['outcome']:14s} v3={rows[-1]['outcome']}", flush=True)
    summary = {}
    for sel in ("layer1", "v3"):
        r = [x for x in rows if x["selector"] == sel]
        counts = {k: sum(x["outcome"] == k for x in r) for k in ("parsed", "empty", "unparseable", "provider_error")}
        per_goal = {gi: sum(x["outcome"] == "empty" for x in r if x["goal_index"] == gi) for gi in range(len(goals))}
        summary[sel] = {"calls": len(r), **counts, "empty_rate": round(counts["empty"] / len(r), 4) if r else None,
                        "empty_by_goal": per_goal}
    doc = {"measured_at_utc": datetime.now(timezone.utc).isoformat(), "reps": args.reps, "goals": goals,
           "v3_selector_settings": v3.settings(), "layer1": "src.agent.llm_client.parse_tool_request (max_tokens=200)",
           "summary": summary, "rows": rows, "pacing_waits": pacer.log}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
