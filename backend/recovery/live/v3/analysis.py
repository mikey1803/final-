"""The registered v3 analysis (docs/live_pilot/v3_preregistration.md §4-§10).

Offline, over a v3 store; reads no network. It computes exactly what the
pre-registration specifies, and nothing is tuned after seeing data:

- **Cell verdicts (§9.1)** on Δ_M = mean(y - b) over included attempts that
  sent >= 1 tool call, with a 95% episode-clustered percentile bootstrap.
  ``+``: supported if lower > 0, contradicted if upper <= 0.
  ``0``: supported if the interval is inside [-m, +m], contradicted if it lies
  entirely outside, with m = 0.15. ``~`` cells are split into the
  ``fresh`` (predicted ``+``) and ``not fresh`` (predicted ``0``) strata (§4.3).
  Insufficient data: an unstratified cell with < 30 included attempts, or a
  stratum with < 15. Δ_ITT is reported for every cell, with no verdict.
- **P1 (§9.2)**: pooled Δ_M of switch_tool, verify, re_ground, replan and
  clarify in T, 98.33% interval (Bonferroni over 3), the ``0`` rule.
  Insufficient if any of those five T cells has < 30 included attempts.
- **P2**: over the persistent blocks S, O2, O5, O20, D -- the ``+`` / ``0`` cell
  verdicts and the two stratum verdicts of every ``~`` cell. Supported if
  none is contradicted and >= 75% are supported; contradicted if >= 25% are
  contradicted; otherwise inconclusive. Insufficient-data verdicts count in
  the denominator as not supported.
- **P3**: ``D_mode = rate(retry) - rate(switch_tool)`` over ALL included
  attempts (ITT, raw y), 98.33% interval from a joint episode bootstrap of
  both arms. Supported: D_T > 0, D_S < 0, D_O20 < 0. Contradicted: D_S or
  D_O20 entirely above 0, or D_T entirely below 0.
- **Detection (§6)** per answer field x transformation family, VERIFY only;
  claimed when the 95% interval of (detected - false-alarm) has lower bound
  > m_det = 0.15, using the offline calibration output; without it: raw rates,
  no claim.
- Manipulation checks (§5), exclusions (§3.3), llm_error per cell with the
  5% flag (D-6), per-tool effective injection (T-A1), selector and pacing
  summaries (D-2, D-6).

Seeds: ``SeedSequence(23001).spawn(36 + 1 + 6 + 1)`` -- one child per cell in
the fixed cell order (grandchildren: ITT, M, fresh, not fresh), then P1, then
P3 per block in block order, then detection. 10,000 resamples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from recovery.live.v3 import report
from recovery.live.v3.report import BLOCK_ORDER, CELL_ORDER, STRATEGY_ORDER, cluster_bootstrap_ci, included

__all__ = ["PREDICTIONS", "analyse", "cell_verdict", "M", "N_MIN", "N_MIN_STRATUM", "to_markdown",
           "DETECTION_PREDICTIONS"]

M = 0.15
N_MIN = 30
N_MIN_STRATUM = 15
LEVEL_CELL = 0.95
LEVEL_PRIMARY = 1 - 0.05 / 3
P2_SUPPORT_SHARE = 0.75
P2_CONTRA_SHARE = 0.25
M_DET = 0.15
DET_HIGH = 0.50
SEED = 23001
PERSISTENT = ("S", "O2", "O5", "O20", "D")
P1_ARMS = ("switch_tool", "verify", "re_ground", "replan", "clarify")

_P = {  # pre-registration §4.2
    "T":   {"retry": "+", "switch_tool": "0", "verify": "0", "re_ground": "0", "replan": "0", "clarify": "0"},
    "S":   {"retry": "0", "switch_tool": "+", "verify": "0", "re_ground": "~", "replan": "+", "clarify": "~"},
    "O2":  {"retry": "+", "switch_tool": "+", "verify": "0", "re_ground": "~", "replan": "~", "clarify": "~"},
    "O5":  {"retry": "0", "switch_tool": "+", "verify": "0", "re_ground": "~", "replan": "~", "clarify": "~"},
    "O20": {"retry": "0", "switch_tool": "+", "verify": "0", "re_ground": "~", "replan": "~", "clarify": "~"},
    "D":   {"retry": "0", "switch_tool": "+", "verify": "0", "re_ground": "~", "replan": "~", "clarify": "~"},
}
PREDICTIONS = {(b, s): _P[b][s] for b, s in CELL_ORDER}
# §6: structural -> high for every field; plausible-wrong -> "~" (no verdict) for capital / extract, "x" otherwise
DETECTION_PREDICTIONS = {
    (t, "structural"): "high" for t in ("countriesnow", "wikipedia", "open_meteo", "wttr", "jsonplaceholder", "dummyjson")
} | {("countriesnow", "plausible_wrong"): "~", ("wikipedia", "plausible_wrong"): "~",
     ("open_meteo", "plausible_wrong"): "x", ("wttr", "plausible_wrong"): "x",
     ("jsonplaceholder", "plausible_wrong"): "x", ("dummyjson", "plausible_wrong"): "x"}


def cell_verdict(pred: str, ci: Optional[tuple], n: int, n_min: int, m: float = M) -> str:
    if n < n_min or ci is None:
        return "insufficient data"
    lo, hi = ci
    if pred == "+":
        return "supported" if lo > 0 else ("contradicted" if hi <= 0 else "inconclusive")
    if pred == "0":
        if lo >= -m and hi <= m:
            return "supported"
        return "contradicted" if (lo > m or hi < -m) else "inconclusive"
    raise ValueError(f"no verdict rule for prediction {pred!r}")


def _ci(d, ep, seq, n_res, level) -> Optional[tuple]:
    if len(d) == 0:
        return None
    return cluster_bootstrap_ci(np.asarray(d, float), np.asarray(ep), seq, n_res, level)


def _r(x):
    return None if x is None else (round(float(x), 4) if not isinstance(x, (list, tuple)) else [round(float(v), 4) for v in x])


def _arm_diff_ci(sub: pd.DataFrame, seq, n_res, level) -> Optional[tuple]:
    """rate(retry) - rate(switch_tool), joint bootstrap over the block's episodes."""
    eps, inv = np.unique(sub["episode_id"].to_numpy(), return_inverse=True)
    k = len(eps)
    if k < 2:
        return None
    y = sub["y"].astype(float).to_numpy()
    is_r = (sub["strategy"] == "retry").to_numpy()
    sr = np.bincount(inv, weights=y * is_r, minlength=k)
    nr = np.bincount(inv, weights=is_r.astype(float), minlength=k)
    ss = np.bincount(inv, weights=y * ~is_r, minlength=k)
    ns = np.bincount(inv, weights=(~is_r).astype(float), minlength=k)
    idx = np.random.default_rng(seq).integers(0, k, size=(n_res, k))
    Nr, Ns = nr[idx].sum(1), ns[idx].sum(1)
    ok = (Nr > 0) & (Ns > 0)
    if not ok.any():
        return None
    stat = sr[idx].sum(1)[ok] / Nr[ok] - ss[idx].sum(1)[ok] / Ns[ok]
    a = (1 - level) / 2
    return float(np.quantile(stat, a)), float(np.quantile(stat, 1 - a))


def _rate_boot(vals: np.ndarray, ep: np.ndarray, seq, n_res) -> Optional[np.ndarray]:
    if len(vals) == 0:
        return None
    labs, inv = np.unique(ep, return_inverse=True)
    k = len(labs)
    s = np.bincount(inv, weights=vals.astype(float), minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    idx = np.random.default_rng(seq).integers(0, k, size=(n_res, k))
    return s[idx].sum(1) / c[idx].sum(1)


def _detection(store: Path, inc: pd.DataFrame, calibration: Optional[dict], seq, n_res) -> list[dict]:
    fam = {}
    v = inc[(inc["mode"] == "degraded_output") & (inc["strategy"] == "verify")] if not inc.empty else inc
    for eid in (v["event_id"] if not v.empty else []):   # only the events needed, not the whole store
        p = Path(store) / "events" / f"{eid}.json"
        if p.exists():
            e = json.loads(p.read_text(encoding="utf-8"))
            fam[eid] = ((e.get("v3") or {}).get("call_meta") or {}).get("family")
    if not v.empty:
        v = v.assign(family=v["event_id"].map(fam))
    fa_items = {}
    if calibration:
        for it in calibration.get("items", []):
            if it.get("judge") is not None and "llm_error" in (it.get("judge") or {}):
                continue
            rej = (not it["schema_passed"]) or bool(it.get("judge") and it["judge"].get("satisfies") is False)
            fa_items.setdefault(it["field"], []).append((it.get("episode_id") or str(it["call_id"]), rej))
    out = []
    keys = list(DETECTION_PREDICTIONS)
    children = seq.spawn(len(keys))
    for (tool, family), child in zip(keys, children):
        sub = v[(v["tool"] == tool) & (v["family"] == family)] if not v.empty else v
        n = len(sub)
        row = {"field": tool, "family": family, "prediction": DETECTION_PREDICTIONS[(tool, family)], "n": n}
        g_det, g_fa = child.spawn(2)
        det = sub["detected"].fillna(False).astype(bool).to_numpy() if n else np.array([])
        row["detected_rate"] = _r(det.mean()) if n else None
        det_b = _rate_boot(det, sub["episode_id"].to_numpy(), g_det, n_res) if n else None
        row["detected_ci"] = _r(list(np.quantile(det_b, [0.025, 0.975]))) if det_b is not None else None
        fa = fa_items.get(tool, [])
        row["false_alarm_n"] = len(fa)
        if not calibration:
            row.update(false_alarm_rate=None, diff_ci=None, detection_claimed=None,
                       verdict="no claim: calibration not run" if row["prediction"] != "~" else "no verdict (~)")
            out.append(row)
            continue
        fa_vals = np.array([r for _, r in fa], dtype=bool)
        row["false_alarm_rate"] = _r(fa_vals.mean()) if len(fa) else None
        fa_b = _rate_boot(fa_vals, np.array([e for e, _ in fa]), g_fa, n_res) if len(fa) else None
        diff_ci = None
        if det_b is not None and fa_b is not None:
            diff_ci = tuple(np.quantile(det_b - fa_b, [0.025, 0.975]))
        row["diff_ci"] = _r(diff_ci)
        claimed = diff_ci is not None and diff_ci[0] > M_DET
        row["detection_claimed"] = bool(claimed) if diff_ci is not None else None
        pred = row["prediction"]
        if diff_ci is None or n < 2:
            row["verdict"] = "insufficient data"
        elif pred == "~":
            row["verdict"] = "no verdict (~)"
        elif pred == "high":
            if claimed and row["detected_ci"][0] > DET_HIGH:
                row["verdict"] = "supported"
            elif diff_ci[1] < M_DET or row["detected_ci"][1] < DET_HIGH:
                row["verdict"] = "contradicted"
            else:
                row["verdict"] = "inconclusive"
        else:  # "x"
            row["verdict"] = "supported" if diff_ci[1] < M_DET else ("contradicted" if claimed else "inconclusive")
        out.append(row)
    return out


def analyse(store: Path, calibration: Optional[dict] = None, n_resamples: int = 10_000) -> dict:
    store = Path(store)
    df = report.load_attempts_frame(store)
    inc = included(df) if not df.empty else df
    root = np.random.SeedSequence(SEED)
    children = root.spawn(len(CELL_ORDER) + 1 + len(BLOCK_ORDER) + 1)
    cell_seeds = children[: len(CELL_ORDER)]
    p1_seed = children[len(CELL_ORDER)]
    p3_seeds = children[len(CELL_ORDER) + 1: len(CELL_ORDER) + 1 + len(BLOCK_ORDER)]
    det_seed = children[-1]

    cells, verdict_index = [], {}
    for (blk, strat), child in zip(CELL_ORDER, cell_seeds):
        g_itt, g_m, g_fr, g_nf = child.spawn(4)
        c = inc[(inc["block_label"] == blk) & (inc["strategy"] == strat)] if not inc.empty else inc
        n = len(c)
        pred = PREDICTIONS[(blk, strat)]
        row = {"block": blk, "strategy": strat, "prediction": pred, "n": n,
               "n_episodes": int(c["episode_id"].nunique()) if n else 0}
        d = (c["y"].astype(float) - c["b"].astype(float)).to_numpy() if n else np.array([])
        ep = c["episode_id"].to_numpy() if n else np.array([])
        m = c["applicable"].to_numpy().astype(bool) if n else np.array([], bool)
        row["applicability"] = _r(m.mean()) if n else None
        row["delta_itt"] = _r(d.mean()) if n else None
        row["delta_itt_ci"] = _r(_ci(d, ep, g_itt, n_resamples, LEVEL_CELL)) if n else None
        row["n_m"] = int(m.sum())
        row["delta_m"] = _r(d[m].mean()) if m.any() else None
        ci_m = _ci(d[m], ep[m], g_m, n_resamples, LEVEL_CELL) if m.any() else None
        row["delta_m_ci"] = _r(ci_m)
        if pred != "~":
            row["verdict"] = cell_verdict(pred, ci_m, n, N_MIN)
            verdict_index[(blk, strat, None)] = row["verdict"]
        else:
            row["verdict"] = "stratified (see strata)"
            fresh = c["fresh"].to_numpy() if n else np.array([])
            for label, flag, g, spred in (("fresh", True, g_fr, "+"), ("not_fresh", False, g_nf, "0")):
                sel = m & np.array([f is flag or f == flag for f in fresh], bool) if n else np.array([], bool)
                ns = int(sel.sum())
                ci_s = _ci(d[sel], ep[sel], g, n_resamples, LEVEL_CELL) if ns else None
                v = cell_verdict(spred, ci_s, ns, N_MIN_STRATUM)
                row[f"{label}"] = {"prediction": spred, "n": ns, "delta_m": _r(d[sel].mean()) if ns else None,
                                   "delta_m_ci": _r(ci_s), "verdict": v}
                verdict_index[(blk, strat, label)] = v
            row["fresh_share"] = _r(np.mean([f is True for f in fresh[m]])) if m.any() else None
        cells.append(row)

    # P1
    t = inc[(inc["block_label"] == "T") & inc["strategy"].isin(P1_ARMS)] if not inc.empty else inc
    t_counts = {a: int((t["strategy"] == a).sum()) if len(t) else 0 for a in P1_ARMS}
    tm = t[t["applicable"].astype(bool)] if len(t) else t
    p1_ci = _ci((tm["y"].astype(float) - tm["b"].astype(float)).to_numpy(), tm["episode_id"].to_numpy(),
                p1_seed, n_resamples, LEVEL_PRIMARY) if len(tm) else None
    p1_v = ("insufficient data" if min(t_counts.values()) < N_MIN or p1_ci is None
            else cell_verdict("0", p1_ci, len(t), N_MIN))
    p1 = {"pooled_n": len(t), "pooled_n_m": len(tm), "per_arm_n": t_counts, "level": round(LEVEL_PRIMARY, 4),
          "delta_m": _r((tm["y"].astype(float) - tm["b"].astype(float)).mean()) if len(tm) else None,
          "ci": _r(p1_ci), "verdict": p1_v}

    # P2
    p2_items = {k: v for k, v in verdict_index.items() if k[0] in PERSISTENT}
    tot = len(p2_items)
    counts = {s: sum(v == s for v in p2_items.values())
              for s in ("supported", "contradicted", "inconclusive", "insufficient data")}
    if tot and counts["contradicted"] / tot >= P2_CONTRA_SHARE:
        p2_v = "contradicted"
    elif tot and counts["contradicted"] == 0 and counts["supported"] / tot >= P2_SUPPORT_SHARE:
        p2_v = "supported"
    else:
        p2_v = "inconclusive"
    p2 = {"n_verdicts": tot, "counts": counts, "verdict": p2_v,
          "note": "insufficient-data verdicts count in the denominator as not supported"}

    # P3
    p3_rows = {}
    for blk, g in zip(BLOCK_ORDER, p3_seeds):
        sub = inc[(inc["block_label"] == blk) & inc["strategy"].isin(["retry", "switch_tool"])] if not inc.empty else inc
        nr = int((sub["strategy"] == "retry").sum()) if len(sub) else 0
        ns = int((sub["strategy"] == "switch_tool").sum()) if len(sub) else 0
        ci = _arm_diff_ci(sub, g, n_resamples, LEVEL_PRIMARY) if nr and ns else None
        est = (float(sub[sub.strategy == "retry"]["y"].mean()) - float(sub[sub.strategy == "switch_tool"]["y"].mean())
               ) if nr and ns else None
        p3_rows[blk] = {"n_retry": nr, "n_switch_tool": ns, "D": _r(est), "ci": _r(ci),
                        "sufficient": bool(nr >= N_MIN and ns >= N_MIN and ci is not None)}
    T, S, O20 = p3_rows["T"], p3_rows["S"], p3_rows["O20"]
    contra = ((S["sufficient"] and S["ci"][0] > 0) or (O20["sufficient"] and O20["ci"][0] > 0)
              or (T["sufficient"] and T["ci"][1] < 0))
    if contra:
        p3_v = "contradicted"
    elif not (T["sufficient"] and S["sufficient"] and O20["sufficient"]):
        p3_v = "insufficient data"
    elif T["ci"][0] > 0 and S["ci"][1] < 0 and O20["ci"][1] < 0:
        p3_v = "supported"
    else:
        p3_v = "inconclusive"
    p3 = {"level": round(LEVEL_PRIMARY, 4), "by_block": p3_rows, "verdict": p3_v,
          "note": "ITT: all included attempts, raw y; strategies that could not act count as failures"}

    # manipulation checks (§5)
    mc = report.manipulation_checks(store)
    res_rows = []
    rdir = store / "v3_resends"
    q = report.quarantined_sessions(store)
    if rdir.exists():
        for p in rdir.glob("*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if int(r["session_seed"]) not in q and r["fault_origin"] == "injected" and r["block_label"] == "T":
                res_rows.append((r["episode_id"], bool(r.get("outcome"))))
    if res_rows:
        vals = np.array([o for _, o in res_rows])
        boot = _rate_boot(vals, np.array([e for e, _ in res_rows]), np.random.SeedSequence(SEED + 1), n_resamples)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        mc["resend_T"] = {"n": len(vals), "rate": _r(vals.mean()), "ci": _r([lo, hi]),
                          "contains_0.80": bool(lo <= 0.80 <= hi)}
    else:
        mc["resend_T"] = {"n": 0}

    le = report.llm_error_by_cell(df)
    ei = report.effective_injection_by_tool(store)
    sel = report.selector_summary(store)
    return {
        "store": str(store), "n_attempts_total": int(len(df)), "n_included": int(len(inc)),
        "by_origin": df["fault_origin"].value_counts().to_dict() if len(df) else {},
        "parameters": {"m": M, "n_min": N_MIN, "n_min_stratum": N_MIN_STRATUM, "level_cell": LEVEL_CELL,
                       "level_primary": round(LEVEL_PRIMARY, 6), "m_det": M_DET, "det_high": DET_HIGH,
                       "bootstrap_seed": SEED, "n_resamples": n_resamples},
        "cells": cells, "P1": p1, "P2": p2, "P3": p3,
        "detection": _detection(store, inc, calibration, det_seed, n_resamples),
        "manipulation_checks": mc,
        "exclusions": report.exclusion_counts(df).reset_index().to_dict("records") if len(df) else [],
        "llm_error_by_cell": le.to_dict("records") if not le.empty else [],
        "llm_error_flagged_cells": le[le["flag_over_5pct"]][["block_label", "strategy"]].to_dict("records")
        if not le.empty else [],
        "effective_injection_by_tool": ei.to_dict("records") if not ei.empty else [],
        "selector": sel.reset_index().to_dict("records") if not sel.empty else [],
        "pacing": report.pacing_summary(store),
    }


def to_markdown(res: dict) -> str:
    L = [f"# v3 analysis — {res['store']}", "",
         f"Attempts: {res['n_attempts_total']} (included: {res['n_included']}); by origin: {res['by_origin']}", "",
         "## Primary claims", "",
         f"- **P1** (pooled Δ_M, T, {res['P1']['level']:.2%}): n={res['P1']['pooled_n']}, Δ_M={res['P1']['delta_m']}, "
         f"CI={res['P1']['ci']} → **{res['P1']['verdict']}**",
         f"- **P2**: {res['P2']['counts']} of {res['P2']['n_verdicts']} → **{res['P2']['verdict']}**",
         f"- **P3** (ITT): " + "; ".join(f"{b}: D={r['D']} CI={r['ci']}" for b, r in res['P3']['by_block'].items())
         + f" → **{res['P3']['verdict']}**", "", "## Cells (Δ_M verdicts)", "",
         "| block | strategy | pred | n | applic. | Δ_ITT [95%] | Δ_M [95%] | verdict |", "|---|---|---|---|---|---|---|---|"]
    for c in res["cells"]:
        v = c["verdict"]
        if c["prediction"] == "~":
            v = f"fresh: {c['fresh']['verdict']} (n={c['fresh']['n']}); not fresh: {c['not_fresh']['verdict']} (n={c['not_fresh']['n']})"
        L.append(f"| {c['block']} | {c['strategy']} | {c['prediction']} | {c['n']} | {c['applicability']} | "
                 f"{c['delta_itt']} {c['delta_itt_ci']} | {c['delta_m']} {c['delta_m_ci']} | {v} |")
    L += ["", "## Detection (VERIFY, D)", "", "| field | family | pred | n | detected [95%] | FA | claimed | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for d in res["detection"]:
        L.append(f"| {d['field']} | {d['family']} | {d['prediction']} | {d['n']} | {d['detected_rate']} {d['detected_ci']} | "
                 f"{d.get('false_alarm_rate')} | {d.get('detection_claimed')} | {d['verdict']} |")
    L += ["", f"llm_error cells flagged (> 5%): {res['llm_error_flagged_cells'] or 'none'}", ""]
    return "\n".join(L)
