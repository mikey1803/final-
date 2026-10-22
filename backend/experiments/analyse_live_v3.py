#!/usr/bin/env python
"""Run the registered v3 analysis (docs/live_pilot/v3_preregistration.md §9;
``recovery/live/v3/analysis.py``) over a v3 store. Offline.

    python experiments/analyse_live_v3.py --store <store> --out <dir> [--calibration <fa.json>]

Writes ``<out>/v3_analysis.json`` and ``<out>/v3_analysis.md``. A store whose
name ends in ``_smoke`` must write to an ``--out`` ending in ``_smoke``
(CLAUDE.md rule 5): smoke results are never data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from recovery.live.v3.analysis import analyse, to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None, help="output of the offline false-alarm calibration")
    ap.add_argument("--resamples", type=int, default=10_000)
    args = ap.parse_args()
    if args.store.name.endswith("_smoke") and not args.out.name.endswith("_smoke"):
        raise SystemExit("a _smoke store must be analysed into an --out directory ending in _smoke")
    cal = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    res = analyse(args.store, calibration=cal, n_resamples=args.resamples)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v3_analysis.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = to_markdown(res)
    (args.out / "v3_analysis.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
