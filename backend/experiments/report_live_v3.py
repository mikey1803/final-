#!/usr/bin/env python
"""Print the v3 descriptive report for a store (offline; reads no network).

    python experiments/report_live_v3.py --store <dir>

Per block: each captured failure (origin, tool, drawn strategy, strategy
outcome y vs the paired resend outcome b), the per-cell table (n, rates,
delta_itt / delta_m with episode-clustered bootstrap intervals), the
manipulation checks (pre-reg §5), instrument-fault exclusions, VERIFY
detection per field x family, and the count-only stopping status.

Descriptive only: P1-P3 and cell verdicts belong to the registered v3
analysis script (pre-reg §9, §12).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

import pandas as pd  # noqa: E402

from recovery.live.v3 import report  # noqa: E402
from recovery.live.v3.stopping import stopping_status  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=report.N_RESAMPLES)
    args = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)

    df = report.load_attempts_frame(args.store)
    if df.empty:
        print("no v3 attempts in this store")
        return 0
    print(f"v3 attempts: {len(df)}  (injected {int((df.fault_origin == 'injected').sum())}, "
          f"natural {int((df.fault_origin == 'natural').sum())}, agent_side {int((df.fault_origin == 'agent_side').sum())}; "
          f"instrument faults {int(df.instrument_fault.sum())}; quarantined {int(df.quarantined.sum())})\n")

    print("== every captured failure: strategy outcome y vs paired resend outcome b ==")
    cols = ["block_label", "fault_origin", "tool", "strategy", "y", "b", "reason", "applicable", "fresh",
            "detected", "corrected", "task_completed"]
    order = {b: i for i, b in enumerate(report.BLOCK_ORDER)}
    print(df.assign(_o=df.block_label.map(order)).sort_values(["_o", "episode_id"])[cols].to_string(index=False))

    print("\n== per block (all origins shown separately; descriptive) ==")
    agg = (df.groupby(["block_label", "fault_origin"])
           .agg(n=("y", "size"), strategy_y=("y", "mean"), resend_b=("b", "mean"),
                instrument_faults=("instrument_fault", "sum")).reset_index())
    agg["_o"] = agg.block_label.map(order)
    print(agg.sort_values(["_o", "fault_origin"]).drop(columns="_o").round(3).to_string(index=False))

    print("\n== cells (injected, instrument faults excluded) ==")
    cells = report.cell_table(df, n_resamples=args.resamples)
    print(cells[cells.n > 0].to_string(index=False))

    print("\n== manipulation checks ==")
    print(json.dumps(report.manipulation_checks(args.store), indent=1))
    print("\n== instrument-fault exclusions per strategy ==")
    print(report.exclusion_counts(df).to_string())
    det = report.detection_table(args.store, df)
    print("\n== VERIFY detection per field x family (D, injected) ==")
    print(det.to_string(index=False) if not det.empty else "(no VERIFY attempts on degraded failures)")
    print("\n== llm_error rate per cell (injected, before exclusion; flag > 5%, D-6) ==")
    le = report.llm_error_by_cell(df)
    print(le.to_string(index=False) if not le.empty else "(none)")
    print("\n== effective injection rate per tool (fresh draws; T-A1 leak visible in T) ==")
    ei = report.effective_injection_by_tool(args.store)
    print(ei.to_string(index=False) if not ei.empty else "(no injection logs)")
    print("\n== v3 selector calls (D-2) ==")
    ss = report.selector_summary(args.store)
    print(ss.to_string() if not ss.empty else "(no selector log)")
    print("\n== pacing (D-6) ==")
    print(json.dumps(report.pacing_summary(args.store), indent=1))
    st = stopping_status(args.store)
    print(f"\n== stopping rule (count-only) == sessions complete: {st['sessions_complete']}, "
          f"cells below n_min={st['n_min']}: {len(st['cells_below_n_min'])}/36, stop={st['stop']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
