"""The v3 stopping rule (pre-registration §2): COUNT-ONLY.

After each completed session this reports how many included attempts each
cell (block label x strategy) has, and whether every cell has at least
``N_MIN`` or ``S_MAX`` sessions have run. It reads no outcomes: each record is
projected onto the identity and inclusion fields below immediately after it
is parsed, and nothing else is ever accessed (acceptance test A10 checks the
source statically and at run time).

Included = injected fault origin, not an instrument fault, session not
quarantined (pre-reg §3.3); in tool_outage blocks only failures that opened
their outage, outage index k = 1 (D-8). ``outage_k`` is design state, not an
outcome.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

__all__ = ["N_MIN", "S_MAX", "ALLOWED_TOP", "ALLOWED_V3", "project", "count_cells", "stopping_status"]

N_MIN = 30   # configuration choice (pre-reg §2)
S_MAX = 60   # configuration choice (pre-reg §2)

ALLOWED_TOP = ("strategy", "definition_version", "v3")
ALLOWED_V3 = ("session_seed", "block_index", "block_label", "injection_mode", "outage_N", "fault_origin",
              "instrument_fault", "outage_k")
_BLOCKS = ("T", "S", "O2", "O5", "O20", "D")
_STRATEGIES = ("retry", "switch_tool", "verify", "re_ground", "replan", "clarify")


def _parse(text: str) -> Mapping:
    return json.loads(text)


def project(record: Mapping) -> dict:
    """The only access to a record: allowed identity / inclusion keys, nothing else."""
    v3 = record["v3"]
    out = {"strategy": record["strategy"], "definition_version": record["definition_version"]}
    for k in ALLOWED_V3:
        out[k] = v3.get(k)      # outage_k: design state (D-8); absent on pre-D-8 records
    return out


def count_cells(projected: Iterable[dict], quarantined: set) -> Counter:
    c: Counter = Counter()
    for r in projected:
        if (r["definition_version"] == "v3" and r["fault_origin"] == "injected" and not r["instrument_fault"]
                and int(r["session_seed"]) not in quarantined
                and (r["injection_mode"] != "tool_outage" or r["outage_k"] == 1)):   # D-8: O* cells count k = 1
            c[(r["block_label"], r["strategy"])] += 1
    return c


def _sessions(root: Path) -> tuple[int, set]:
    done, quarantined = 0, set()
    qdir = root / "v3_quarantine"
    if qdir.exists():
        for p in qdir.glob("*.json"):
            quarantined.add(int(_parse(p.read_text(encoding="utf-8"))["session_seed"]))
    sdir = root / "v3_sessions"
    if sdir.exists():
        for p in sdir.glob("*.json"):
            s = _parse(p.read_text(encoding="utf-8"))
            if s["status"] == "complete" and int(s["session_seed"]) not in quarantined and int(s["session_index"]) > 0:
                done += 1
    return done, quarantined


def stopping_status(root: Path, n_min: int = N_MIN, s_max: int = S_MAX) -> dict:
    root = Path(root)
    sessions_done, quarantined = _sessions(root)
    rdir = root / "recoveries"
    projected = []
    if rdir.exists():
        for p in sorted(rdir.glob("*.json")):
            rec = _parse(p.read_text(encoding="utf-8"))
            if rec["definition_version"] != "v3":
                continue
            projected.append(project(rec))
    counts = count_cells(projected, quarantined)
    cells = {f"{b}/{s}": counts.get((b, s), 0) for b in _BLOCKS for s in _STRATEGIES}
    below = [k for k, n in cells.items() if n < n_min]
    return {"sessions_complete": sessions_done, "n_min": n_min, "s_max": s_max, "cells": cells,
            "cells_below_n_min": below, "stop": (not below) or sessions_done >= s_max,
            "stop_reason": "all cells >= n_min" if not below else ("s_max reached" if sessions_done >= s_max else None)}
