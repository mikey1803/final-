"""v3 cell estimates, manipulation checks and per-field detection
(pre-registration §3-§8). Offline, over a v3 store; never shown to the agent.

Inclusion (pre-reg §3.3): only ``fault_origin == "injected"`` attempts enter
cells; ``instrument_fault`` attempts (``llm_error`` / ``executor_error``) are
excluded from every rate and counted; quarantined sessions are excluded
entirely. ``natural`` and ``agent_side`` attempts are reported separately.

Per cell (block label x strategy):

- ``delta_itt`` = mean(y - b) over included attempts; ``delta_m`` = the same
  over attempts that sent >= 1 tool call (``applicable``);
- ``applicability`` = share of included attempts that sent a call;
- intervals: cluster bootstrap over ``episode_id`` (resample episodes with
  replacement, ratio estimator sum(y - b) / n), 10,000 resamples, percentile,
  ``V3_BOOTSTRAP_SEED = 23001`` with one ``SeedSequence(23001).spawn`` child per
  cell in the fixed cell order (and one grandchild per measure). No p-values.

The primary claims P1-P3 and cell verdicts are computed by the registered v3
analysis script (pre-reg §12), not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from recovery.live.v3.loop import BLOCK_TYPES, _LABELS

__all__ = [
    "V3_BOOTSTRAP_SEED", "N_RESAMPLES", "CELL_ORDER", "STRATEGY_ORDER", "load_attempts_frame", "included",
    "cluster_bootstrap_ci", "cell_table", "manipulation_checks", "detection_table", "exclusion_counts",
    "load_injection_logs", "quarantined_sessions",
]

V3_BOOTSTRAP_SEED = 23001
N_RESAMPLES = 10_000
BLOCK_ORDER = [_LABELS[b] for b in BLOCK_TYPES]                      # T, S, O2, O5, O20, D
STRATEGY_ORDER = ["retry", "switch_tool", "verify", "re_ground", "replan", "clarify"]
CELL_ORDER = [(blk, s) for blk in BLOCK_ORDER for s in STRATEGY_ORDER]


def _read_dir(d: Path) -> Iterable[dict]:
    if d.exists():
        for p in sorted(d.glob("*.json")):
            yield json.loads(p.read_text(encoding="utf-8"))


def quarantined_sessions(root: Path) -> set:
    return {int(q["session_seed"]) for q in _read_dir(Path(root) / "v3_quarantine")}


def load_attempts_frame(root: Path) -> pd.DataFrame:
    rows = []
    for a in _read_dir(Path(root) / "recoveries"):
        v3 = a.get("v3")
        if a.get("definition_version") != "v3" or not v3:
            continue
        ev = a["result"]["evidence"]
        rb = v3.get("resend_baseline") or {}
        rows.append({
            "attempt_id": a["attempt_id"], "event_id": a["event_id"], "strategy": a["strategy"],
            "block_label": v3["block_label"], "mode": v3["injection_mode"], "outage_N": v3.get("outage_N"),
            "fault_origin": v3["fault_origin"], "agent_side_kind": v3.get("agent_side_kind"),
            "episode_id": v3["episode_id"], "session_seed": v3["session_seed"], "tool": v3.get("tool"),
            "reason": ev.get("reason"), "instrument_fault": bool(v3["instrument_fault"]),
            "applicable": bool(v3["applicable"]), "y": bool(v3["primary_outcome"]),
            "b": bool(rb.get("outcome", False)), "success": bool(a["result"]["success"]),
            "fresh": v3.get("fresh"), "detected": v3.get("detected"), "corrected": v3.get("corrected"),
            "no_action_taken": bool(a.get("no_action_taken")), "untranslatable": ev.get("reason") == "untranslatable_args",
            "task_completed": v3.get("task_completed"), "task_completed_applicable": v3.get("task_completed_applicable"),
            "failure_ordinal": v3.get("failure_ordinal"), "h1_exempt_natural": bool(v3.get("h1_exempt_natural")),
            "outage_k": v3.get("outage_k"), "outage_remaining": v3.get("outage_remaining"),
            "h1_exempt_outage_end": bool(v3.get("h1_exempt_outage_end")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    q = quarantined_sessions(root)
    df["quarantined"] = df["session_seed"].astype(int).isin(q)
    return df


def included(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["fault_origin"] == "injected") & ~df["instrument_fault"] & ~df["quarantined"]]


def exclusion_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Instrument faults per strategy (pre-reg §3.3: excluded from rates, counted)."""
    if df.empty:
        return pd.DataFrame()
    live = df[~df["quarantined"]]
    return (live.assign(llm_error=live["reason"] == "llm_error", executor_error=live["reason"] == "executor_error")
            .groupby("strategy")[["llm_error", "executor_error"]].sum().astype(int))


def cluster_bootstrap_ci(diff: np.ndarray, clusters: np.ndarray, seed_seq: np.random.SeedSequence,
                         n_resamples: int = N_RESAMPLES, level: float = 0.95) -> Optional[tuple[float, float]]:
    """Percentile CI for sum(diff)/n, resampling clusters with replacement. None if < 2 clusters."""
    diff = np.asarray(diff, dtype=float)
    labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(labels)
    if k < 2:
        return None
    sums = np.bincount(inv, weights=diff, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed_seq)
    idx = rng.integers(0, k, size=(n_resamples, k))
    stat = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    a = (1.0 - level) / 2.0
    return float(np.quantile(stat, a)), float(np.quantile(stat, 1.0 - a))


def _fmt_ci(ci: Optional[tuple]) -> Optional[list]:
    return None if ci is None else [round(ci[0], 4), round(ci[1], 4)]


def cell_table(df: pd.DataFrame, n_resamples: int = N_RESAMPLES) -> pd.DataFrame:
    inc = included(df)
    children = np.random.SeedSequence(V3_BOOTSTRAP_SEED).spawn(len(CELL_ORDER))
    rows = []
    for (blk, strat), child in zip(CELL_ORDER, children):
        c = inc[(inc["block_label"] == blk) & (inc["strategy"] == strat)] if not inc.empty else inc
        n = len(c)
        row = {"block": blk, "strategy": strat, "n": n,
               "n_episodes": int(c["episode_id"].nunique()) if n else 0}
        if n:
            d = c["y"].astype(float).to_numpy() - c["b"].astype(float).to_numpy()
            m = c["applicable"].to_numpy()
            g_itt, g_m = child.spawn(2)
            row.update(
                rate_y=round(float(c["y"].mean()), 4), rate_b=round(float(c["b"].mean()), 4),
                applicability=round(float(m.mean()), 4),
                delta_itt=round(float(d.mean()), 4),
                delta_itt_ci=_fmt_ci(cluster_bootstrap_ci(d, c["episode_id"].to_numpy(), g_itt, n_resamples)),
                n_m=int(m.sum()),
                delta_m=round(float(d[m].mean()), 4) if m.any() else None,
                delta_m_ci=_fmt_ci(cluster_bootstrap_ci(d[m], c["episode_id"].to_numpy()[m], g_m, n_resamples))
                if m.sum() else None,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_injection_logs(root: Path) -> pd.DataFrame:
    rows = []
    d = Path(root) / "v3_injection_logs"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    rows.append({k: e.get(k) for k in ("call_id", "mode", "block_index", "outage_N", "probe",
                                                       "actor", "tool", "fresh_draw", "draw_injected", "injected",
                                                       "degraded", "success", "outage_leak", "session_seed",
                                                       "block_label", "transformation", "family")})
    return pd.DataFrame(rows)


def manipulation_checks(root: Path) -> dict:
    """Resend rates on injected failures per block (pre-reg §5) and the observed
    injection rate per block (fresh draws that injected / fresh draws, agent and
    strategy calls only)."""
    out: dict = {"resend_on_injected": {}, "observed_injection_rate": {}, "h1_exempt_natural": 0,
                 "outage_leaks": 0}
    q = quarantined_sessions(root)
    res = [r for r in _read_dir(Path(root) / "v3_resends") if int(r["session_seed"]) not in q]
    for blk in BLOCK_ORDER:
        inj = [r for r in res if r["block_label"] == blk and r["fault_origin"] == "injected"
               and not r.get("h1_exempt_outage_end")]
        out["resend_on_injected"][blk] = {"n": len(inj), "successes": int(sum(bool(r.get("outcome")) for r in inj))}
    out["h1_exempt_natural"] = int(sum(bool(r.get("h1_exempt_natural")) for r in res))
    out["h1_exempt_outage_end"] = int(sum(bool(r.get("h1_exempt_outage_end")) for r in res))
    logs = load_injection_logs(root)
    if not logs.empty:
        logs = logs[~logs["session_seed"].astype(int).isin(q)]
        fresh = logs[(logs["fresh_draw"] == True) & (logs["probe"] == False)]  # noqa: E712
        for blk in BLOCK_ORDER:
            f = fresh[fresh["block_label"] == blk]
            out["observed_injection_rate"][blk] = {"fresh_draws": int(len(f)),
                                                   "injected": int(f["draw_injected"].fillna(False).sum())}
        out["outage_leaks"] = int(logs["outage_leak"].fillna(False).astype(bool).sum())
    return out


def detection_table(root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """VERIFY ``detected`` per answer field (failing tool) x transformation family,
    D blocks, included attempts only. Never pooled across fields (pre-reg §6)."""
    inc = included(df)
    if inc.empty:
        return pd.DataFrame()
    v = inc[(inc["mode"] == "degraded_output") & (inc["strategy"] == "verify")]
    if v.empty:
        return pd.DataFrame()
    fam = {}
    for e in _read_dir(Path(root) / "events"):
        m = (e.get("v3") or {}).get("call_meta") or {}
        fam[e["event_id"]] = m.get("family")
    v = v.assign(family=v["event_id"].map(fam))
    return (v.groupby(["tool", "family"]).agg(n=("detected", "size"), detected=("detected", "sum"),
                                              corrected=("corrected", "sum")).reset_index())


# ---------------------------------------------------------------- operational diagnostics

LLM_ERROR_FLAG_RATE = 0.05   # pre-reg D-6: cells with llm_error > 5% are flagged


def llm_error_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """llm_error rate per (block, strategy) over injected attempts BEFORE the
    exclusion (quarantined sessions removed); flagged above 5% (D-6)."""
    if df.empty:
        return pd.DataFrame()
    c = df[(df["fault_origin"] == "injected") & ~df["quarantined"]]
    if c.empty:
        return pd.DataFrame()
    g = (c.assign(llm_error=c["reason"] == "llm_error").groupby(["block_label", "strategy"])
         .agg(n_all=("llm_error", "size"), llm_error=("llm_error", "sum")).reset_index())
    g["llm_error_rate"] = (g["llm_error"] / g["n_all"]).round(4)
    g["flag_over_5pct"] = g["llm_error_rate"] > LLM_ERROR_FLAG_RATE
    return g


def effective_injection_by_tool(root: Path) -> pd.DataFrame:
    """Per block x tool, over fresh draws on agent/strategy calls: draws, nominal
    injections, and injections that actually took effect (the call failed, or in
    D was degraded). Makes the transient 0.2 s timeout leak visible (T-A1)."""
    logs = load_injection_logs(root)
    if logs.empty:
        return pd.DataFrame()
    q = quarantined_sessions(root)
    f = logs[(logs["fresh_draw"] == True) & (logs["probe"] == False)  # noqa: E712
             & ~logs["session_seed"].astype(int).isin(q)].copy()
    if f.empty:
        return pd.DataFrame()
    f["nominal"] = f["draw_injected"].fillna(False).astype(bool)
    took = (~f["success"].fillna(False).astype(bool)) | f["degraded"].fillna(False).astype(bool)
    f["effective"] = f["nominal"] & took
    g = (f.groupby(["block_label", "tool"]).agg(draws=("nominal", "size"), nominal_injected=("nominal", "sum"),
                                               effective_injected=("effective", "sum")).reset_index())
    g["nominal_rate"] = (g["nominal_injected"] / g["draws"]).round(3)
    g["effective_rate"] = (g["effective_injected"] / g["draws"]).round(3)
    return g


def _read_jsonl(d: Path) -> list[dict]:
    rows = []
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def selector_summary(root: Path) -> pd.DataFrame:
    """Every v3 selector call (D-2): empty-reply, unparseable and provider-error rates by actor."""
    rows = _read_jsonl(Path(root) / "v3_selector_log")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["actor"] = df.get("actor", pd.Series(["agent"] * len(df))).fillna("agent")
    out = df.groupby("actor")["outcome"].value_counts().unstack(fill_value=0)
    out["calls"] = out.sum(axis=1)
    out.loc["ALL"] = out.sum(axis=0)
    for col in ("empty", "unparseable", "tool_use_failed", "provider_error", "parsed"):
        if col not in out:
            out[col] = 0
    out["empty_rate"] = (out["empty"] / out["calls"]).round(4)
    return out[["calls", "parsed", "empty", "unparseable", "tool_use_failed", "provider_error", "empty_rate"]]


def pacing_summary(root: Path) -> dict:
    rows = _read_jsonl(Path(root) / "v3_pacing")
    by = {}
    for r in rows:
        by.setdefault(r["checkpoint"], [0, 0.0])
        by[r["checkpoint"]][0] += 1
        by[r["checkpoint"]][1] += float(r["waited_s"])
    return {"waits": len(rows), "total_wait_s": round(sum(float(r["waited_s"]) for r in rows), 1),
            "by_checkpoint": {k: {"waits": v[0], "wait_s": round(v[1], 1)} for k, v in by.items()}}


def selector_by_task(root: Path) -> pd.DataFrame:
    """Selector calls per task (D-9): HTTP 400 tool_use_failed and empty-reply rates."""
    rows = _read_jsonl(Path(root) / "v3_selector_log")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "task_id" not in df:
        df["task_id"] = None
    df["task_id"] = df["task_id"].fillna("(unknown)")
    g = df.groupby("task_id")["outcome"].value_counts().unstack(fill_value=0)
    for col in ("parsed", "empty", "unparseable", "tool_use_failed", "provider_error"):
        if col not in g:
            g[col] = 0
    g["calls"] = g[["parsed", "empty", "unparseable", "tool_use_failed", "provider_error"]].sum(axis=1)
    g["tool_use_failed_rate"] = (g["tool_use_failed"] / g["calls"]).round(4)
    g["empty_rate"] = (g["empty"] / g["calls"]).round(4)
    return g[["calls", "parsed", "empty", "unparseable", "tool_use_failed", "provider_error",
              "tool_use_failed_rate", "empty_rate"]]
