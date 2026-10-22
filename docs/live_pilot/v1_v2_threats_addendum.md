# Live pilot v1 / v2 — threats-to-validity addendum

`strategy_definitions.md` (v1) and `strategy_definitions_v2.md` (v2) are frozen
and are not edited (CLAUDE.md rule 3). This addendum records threats to the
validity of v1/v2 data found after those freezes. Each entry is dated.

## T-A1 (2026-10-22): injected timeouts often do not fire

**What.** v1/v2 `FlakyWrapper` injects a timeout by issuing the real call
with `timeout=0.2` s (`INJECTED_TIMEOUT_S`). `requests` applies that value to
each socket operation separately (connect, then each read), not to the whole
call. A response that arrives in chunks, each within 0.2 s, therefore
succeeds even though the call takes far longer.

**Measured, 2026-10-22,** with 4 calls per tool at `timeout=0.2` (health-check
arguments):

| Tool | Succeeded despite the injected timeout | Latency |
|---|---|---|
| wikipedia | 4/4 | 1.1–1.9 s |
| jsonplaceholder | 4/4 | 1.1–1.3 s |
| open_meteo | 1/4 | |
| countriesnow, wttr, dummyjson | 0/4 | |

**Consequences for v1/v2 data:**

- **The effective injection rate is lower than the frozen 0.20, and it differs by tool.** A timeout injection that does not fire returns a genuine success, which the loop does not record as a failure. Half of all injections are timeouts, so on wikipedia and jsonplaceholder the effective rate is roughly halved. This is a rough reading of 4 calls per tool, not a measured rate.
- **The peer-symmetry assumption is weakened.** v1 §"Tool-alternatives map" requires both members of a peer pair to face identical injection risk. With timeouts that leak by tool (wikipedia leaks, countriesnow does not), the realised risk is not identical within the encyclopedic pair or the placeholder pair.
- **No mislabelled records.** The loop writes a `FailureEvent` only for a failed call, so an injected call that succeeded never became an event. Checked 2026-10-22: the only stored v1/v2 events are in the Task 21 scratch store, which is not data. Those are 4 failed malformed-request calls. The repository fixture is synthetic, a failed call. No Phase 1 live data exists on disk.
- **What remains valid.** Malformed-request injections are unaffected: all six tools returned real errors, 404, 400 or 500, on 2026-10-22.

**How v3 handles it.** Under v3, `transient` keeps the 0.2 s mechanism unchanged, as frozen (v3 §3.1). The v3 report shows the leak per tool, as each tool's effective injection rate in each block. `tool_outage` uses a read timeout that does fire (v3 pre-registration D-5).
