"""A10: the stopping-rule script cannot import or read outcome fields --
checked statically (imports and string constants in its source) and at run
time (records whose outcome fields raise on access)."""

import ast
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery.live.v3 import stopping  # noqa: E402

OUTCOME_FIELDS = {"success", "corrected", "detected", "task_completed", "resend_baseline", "evidence", "result",
                  "primary_outcome", "outcome", "applicable", "fresh", "budget_used", "no_action_taken", "aborted",
                  "final_call_id", "corrected_reference", "task_completed_applicable"}
OUTCOME_MODULES = {"recovery.live.v3.report", "recovery.live.v3.recovery", "recovery.live.v3.answers",
                   "recovery.live.v3.loop", "recovery.live.recovery_attempt", "recovery.live.v3.calibration",
                   "pandas"}


def test_stopping_source_imports_and_names_no_outcomes():
    tree = ast.parse(inspect.getsource(stopping))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported <= {"__future__", "json", "collections", "pathlib", "typing"}, imported
    assert not imported & OUTCOME_MODULES
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not strings & OUTCOME_FIELDS, strings & OUTCOME_FIELDS


class Guarded(dict):
    """A record whose outcome fields raise on any access."""

    def __getitem__(self, k):
        if k in OUTCOME_FIELDS:
            raise AssertionError(f"stopping rule read outcome field {k!r}")
        v = super().__getitem__(k)
        return Guarded(v) if isinstance(v, dict) else v

    def get(self, k, default=None):
        return self[k] if k in self else default

    def items(self):
        raise AssertionError("stopping rule iterated a whole record")

    def values(self):
        raise AssertionError("stopping rule iterated a whole record")


def _attempt(strategy, label, origin="injected", fault=False, seed=231001):
    return {"strategy": strategy, "definition_version": "v3", "result": {"success": True, "evidence": {}},
            "v3": {"session_seed": seed, "block_index": 0, "block_label": label, "injection_mode": "x",
                   "outage_N": None, "fault_origin": origin, "instrument_fault": fault, "success": True,
                   "corrected": True, "primary_outcome": True, "resend_baseline": {"outcome": True},
                   "task_completed": True}}


def test_stopping_runtime_never_touches_outcomes(tmp_path, monkeypatch):
    rd = tmp_path / "recoveries"
    rd.mkdir()
    recs = [_attempt("retry", "T"), _attempt("retry", "T"), _attempt("verify", "D", origin="natural"),
            _attempt("replan", "S", fault=True), _attempt("clarify", "O2", seed=231002)]
    for i, r in enumerate(recs):
        (rd / f"a{i}.json").write_text(json.dumps(r))
    (tmp_path / "v3_quarantine").mkdir()
    (tmp_path / "v3_quarantine" / "231002.json").write_text(json.dumps({"session_seed": 231002}))
    (tmp_path / "v3_sessions").mkdir()
    (tmp_path / "v3_sessions" / "231001.json").write_text(
        json.dumps({"session_seed": 231001, "session_index": 1, "status": "complete"}))
    monkeypatch.setattr(stopping, "_parse", lambda text: Guarded(json.loads(text)))
    st = stopping.stopping_status(tmp_path, n_min=2)
    assert st["cells"]["T/retry"] == 2
    assert st["cells"]["D/verify"] == 0          # natural: not included
    assert st["cells"]["S/replan"] == 0          # instrument fault: excluded
    assert st["cells"]["O2/clarify"] == 0        # quarantined session
    assert st["sessions_complete"] == 1 and not st["stop"] and "T/retry" not in st["cells_below_n_min"]


def test_stop_at_s_max(tmp_path):
    (tmp_path / "v3_sessions").mkdir()
    for s in range(1, 4):
        (tmp_path / "v3_sessions" / f"{231000 + s}.json").write_text(
            json.dumps({"session_seed": 231000 + s, "session_index": s, "status": "complete"}))
    (tmp_path / "v3_sessions" / "231000.json").write_text(       # the smoke session never counts
        json.dumps({"session_seed": 231000, "session_index": 0, "status": "complete"}))
    st = stopping.stopping_status(tmp_path, n_min=30, s_max=3)
    assert st["sessions_complete"] == 3 and st["stop"] and st["stop_reason"] == "s_max reached"
