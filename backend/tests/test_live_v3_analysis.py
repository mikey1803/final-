"""The registered v3 analysis recovers known effects from a synthetic store,
detects a planted contradiction, marks small cells insufficient, flags
llm_error cells, applies the detection claim rule, and runs on a real-shaped
store. Synthetic outcomes are constructed with exact proportions, so every
expected verdict is determined, not left to chance."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.v3.analysis import PREDICTIONS, analyse, cell_verdict, to_markdown  # noqa: E402

MODES = {"T": ("transient", None), "S": ("sticky_args", None), "O2": ("tool_outage", 2),
         "O5": ("tool_outage", 5), "O20": ("tool_outage", 20), "D": ("degraded_output", None)}
N = 60


class Store:
    def __init__(self, root: Path):
        self.root, self.i = root, 0
        (root / "recoveries").mkdir(parents=True, exist_ok=True)
        (root / "events").mkdir(exist_ok=True)

    def add(self, blk, strategy, y, b, *, applicable=True, fresh=None, origin="injected", reason=None,
            tool="countriesnow", family=None, detected=None, episode=None):
        self.i += 1
        mode, n = MODES[blk]
        eid = f"ev{self.i:05d}"
        v3 = {"block_label": blk, "injection_mode": mode, "outage_N": n, "fault_origin": origin,
              "agent_side_kind": None, "episode_id": episode or f"ep{self.i:05d}", "session_seed": 231001,
              "tool": tool, "instrument_fault": reason in ("llm_error", "executor_error"), "applicable": applicable,
              "primary_outcome": bool(y), "resend_baseline": {"outcome": bool(b)}, "fresh": fresh,
              "detected": detected, "corrected": bool(y) if mode == "degraded_output" else None,
              "task_completed": False, "task_completed_applicable": True, "failure_ordinal": 1,
              "h1_exempt_natural": False}
        a = {"attempt_id": f"a{self.i:05d}", "event_id": eid, "strategy": strategy, "definition_version": "v3",
             "result": {"success": bool(y), "evidence": {"reason": reason or ("success" if y else "exhausted_attempts")}},
             "no_action_taken": False, "v3": v3}
        (self.root / "recoveries" / f"a{self.i:05d}.json").write_text(json.dumps(a))
        (self.root / "events" / f"{eid}.json").write_text(json.dumps(
            {"event_id": eid, "v3": {"call_meta": {"family": family}}}))


def ones(n, k):
    """Exactly k ones out of n, at the front."""
    return [1] * k + [0] * (n - k)


def shifted(n, k, by):
    """Exactly k ones out of n, starting at index ``by`` (wrapping)."""
    base = ones(n, k)
    return base[-by:] + base[:-by] if by else base


def fill_cell(st, blk, strat, ys, bs, **kw):
    for y, b in zip(ys, bs):
        st.add(blk, strat, y, b, **kw)


def build_known(root: Path, contradiction: bool = False) -> Store:
    st = Store(root)
    zero = [0] * N
    for blk in MODES:
        for strat in ("retry", "switch_tool", "verify", "re_ground", "replan", "clarify"):
            pred = PREDICTIONS[(blk, strat)]
            if blk == "T":
                if strat == "retry":
                    fill_cell(st, blk, strat, ones(N, 59), shifted(N, 48, 12))   # Δ ≈ +0.18
                else:
                    fill_cell(st, blk, strat, ones(N, 48), shifted(N, 48, 12))   # Δ = 0 exactly
                continue
            if pred == "+":
                k = 57 if strat == "retry" else 48
                if contradiction and blk == "S" and strat == "switch_tool":
                    k = 0
                fill_cell(st, blk, strat, ones(N, k), zero)
            elif pred == "0":
                if blk == "D" and strat == "verify":
                    for j in range(N):
                        fam = "structural" if j % 2 == 0 else "plausible_wrong"
                        st.add(blk, strat, 0, 0, tool="countriesnow" if fam == "structural" else "open_meteo",
                               family=fam, detected=fam == "structural")
                else:
                    fill_cell(st, blk, strat, zero, zero)
            else:  # "~": 30 fresh at +0.8, 30 not fresh at 0
                fill_cell(st, blk, strat, ones(30, 24), [0] * 30, fresh=True)
                fill_cell(st, blk, strat, [0] * 30, [0] * 30, fresh=False)
    return st


def calibration(rejections_countriesnow=2, rejections_open_meteo=3):
    items = []
    for f, rej in (("countriesnow", rejections_countriesnow), ("open_meteo", rejections_open_meteo)):
        for j in range(50):
            items.append({"field": f, "call_id": j, "episode_id": f"fa-{f}-{j}", "schema_passed": True,
                          "judge": {"satisfies": j >= rej, "reason": "x"}})
    return {"items": items}


def by_cell(res):
    return {(c["block"], c["strategy"]): c for c in res["cells"]}


@pytest.fixture(scope="module")
def known_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("v3_known")
    build_known(root)
    return root


def test_known_effects_are_recovered(known_store):
    res = analyse(known_store, calibration=calibration(), n_resamples=2000)
    cells = by_cell(res)
    for (blk, strat), pred in PREDICTIONS.items():
        c = cells[(blk, strat)]
        if blk == "T" and strat != "retry":
            assert c["verdict"] in ("supported", "inconclusive")       # per-cell n=60 is underpowered by design
        elif pred in "+0":
            assert c["verdict"] == "supported", (blk, strat, c)
        else:
            assert c["fresh"]["verdict"] == "supported" and c["not_fresh"]["verdict"] == "supported", (blk, strat)
    assert res["P1"]["verdict"] == "supported" and res["P1"]["pooled_n"] == 5 * N
    assert res["P2"]["verdict"] == "supported" and res["P2"]["counts"]["contradicted"] == 0
    assert res["P3"]["verdict"] == "supported"
    assert res["P3"]["by_block"]["S"]["D"] == pytest.approx(-0.8)
    det = {(d["field"], d["family"]): d for d in res["detection"]}
    assert det[("countriesnow", "structural")]["detection_claimed"] is True
    assert det[("countriesnow", "structural")]["verdict"] == "supported"
    assert det[("open_meteo", "plausible_wrong")]["verdict"] == "supported"   # "x": no detection, as predicted
    assert "P1" in to_markdown(res)


def test_planted_contradiction_is_found(tmp_path):
    build_known(tmp_path, contradiction=True)
    res = analyse(tmp_path, n_resamples=1000)
    assert by_cell(res)[("S", "switch_tool")]["verdict"] == "contradicted"
    assert res["P2"]["counts"]["contradicted"] == 1 and res["P2"]["verdict"] == "inconclusive"
    assert res["P3"]["verdict"] in ("inconclusive", "contradicted")
    assert res["P3"]["by_block"]["S"]["D"] == pytest.approx(0.0)


def test_small_cells_are_insufficient_and_llm_errors_flagged(tmp_path):
    st = Store(tmp_path)
    fill_cell(st, "S", "switch_tool", ones(10, 8), [0] * 10)
    for _ in range(5):
        st.add("S", "replan", 0, 0, reason="llm_error")
    fill_cell(st, "S", "replan", ones(40, 30), [0] * 40)
    res = analyse(tmp_path, n_resamples=500)
    cells = by_cell(res)
    assert cells[("S", "switch_tool")]["verdict"] == "insufficient data"
    assert cells[("S", "replan")]["n"] == 40                      # llm_error excluded from the cell
    assert {"block_label": "S", "strategy": "replan"} in res["llm_error_flagged_cells"]
    assert res["P1"]["verdict"] == "insufficient data" and res["P3"]["verdict"] == "insufficient data"
    assert all(d["verdict"].startswith("no claim") or d["verdict"].startswith("no verdict") for d in res["detection"])


def test_natural_and_agent_side_never_enter_cells(tmp_path):
    st = Store(tmp_path)
    fill_cell(st, "S", "retry", ones(40, 40), [0] * 40, origin="natural")
    fill_cell(st, "S", "retry", ones(40, 40), [0] * 40, origin="agent_side")
    res = analyse(tmp_path, n_resamples=200)
    assert by_cell(res)[("S", "retry")]["n"] == 0 and res["n_included"] == 0


def test_cell_verdict_rules():
    assert cell_verdict("+", (0.01, 0.5), 30, 30) == "supported"
    assert cell_verdict("+", (-0.2, 0.0), 30, 30) == "contradicted"
    assert cell_verdict("+", (-0.1, 0.2), 30, 30) == "inconclusive"
    assert cell_verdict("0", (-0.1, 0.15), 30, 30) == "supported"
    assert cell_verdict("0", (0.16, 0.4), 30, 30) == "contradicted"
    assert cell_verdict("0", (0.1, 0.3), 30, 30) == "inconclusive"
    assert cell_verdict("+", (0.1, 0.2), 29, 30) == "insufficient data"


def test_deterministic(known_store):
    a = analyse(known_store, n_resamples=500)
    b = analyse(known_store, n_resamples=500)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
