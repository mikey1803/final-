"""Rate-limit pacing for v3 collection (v3 pre-registration D-6, operational).

At three checkpoints -- before an agent selection, before a selection-rerun
resend probe, and before a strategy runs -- the runner waits until the
provider's last reported remaining tokens (per minute) and remaining
requests (per day) cover a fixed reserve for what comes next. The numbers
come from the response headers ``GroqChatClient`` records
(``last_rate_limit``).

Waits happen only at these checkpoints, OUTSIDE every strategy's budget
clock (a strategy's ``BudgetTracker`` starts inside ``execute``), so
strategies are timed exactly as in an unpaced run. Every wait is logged.
A 429 that still happens stays ``llm_error`` (pre-reg §3.3).

Reserves (configuration choices): 1,200 tokens / 1 request before a
selection or a selection-rerun resend; 4,500 tokens / 6 requests before a
strategy (its 1,500-token budget, the provider's per-request charge for
``max_tokens``, and up to three 300-token selector re-runs for an agent-side
retry).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

__all__ = ["Pacer", "RESERVE_TOKENS", "RESERVE_REQUESTS"]

RESERVE_TOKENS = {"selection": 1_200, "resend_selection": 1_200, "strategy": 4_500}
RESERVE_REQUESTS = {"selection": 1, "resend_selection": 1, "strategy": 6}
MARGIN_S = 0.5


@dataclass
class Pacer:
    client: object                                   # has .last_rate_limit (dict or None)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    reserve_tokens: dict = field(default_factory=lambda: dict(RESERVE_TOKENS))
    reserve_requests: dict = field(default_factory=lambda: dict(RESERVE_REQUESTS))
    log: list = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def before(self, kind: str) -> float:
        info: Optional[dict] = getattr(self.client, "last_rate_limit", None)
        if not info:
            return 0.0
        elapsed = self.clock() - info["at"]
        waits, reasons = [], []
        rt, rr = info.get("remaining_tokens"), info.get("remaining_requests")
        if rt is not None and rt < self.reserve_tokens[kind] and info.get("reset_tokens_s") is not None:
            w = info["reset_tokens_s"] - elapsed + MARGIN_S
            if w > 0:
                waits.append(w)
                reasons.append("tokens_per_minute")
        if rr is not None and rr < self.reserve_requests[kind] and info.get("reset_requests_s") is not None:
            w = info["reset_requests_s"] - elapsed + MARGIN_S
            if w > 0:
                waits.append(w)
                reasons.append("requests_per_day")
        if info.get("retry_after_s"):
            w = info["retry_after_s"] - elapsed + MARGIN_S
            if w > 0:
                waits.append(w)
                reasons.append("retry_after")
        wait = max(waits, default=0.0)
        if wait > 0:
            self.sleep(wait)
            self.log.append({"ts_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": kind,
                             "waited_s": round(wait, 3), "reasons": reasons, "remaining_tokens": rt,
                             "remaining_requests": rr, **self.context})
            info["remaining_tokens"] = None   # stale after the wait; refreshed by the next response
            info["remaining_requests"] = None if "requests_per_day" in reasons else rr
            info["retry_after_s"] = None
        return wait
