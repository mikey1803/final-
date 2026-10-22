"""Runtime guardrails for the ``with_recovery`` wrapper (Task 22).

Unlike ``invariants.py`` (scientific assertions over completed simulation
results), these guards act DURING an agent run:

- ``TaskBudget``: hard caps on recovery attempts, LLM tokens and recovery wall
  time per task, using the existing live ``Budget`` type.
- ``CircuitBreaker`` / ``BreakerBoard``: per-tool closed / open / half-open
  breakers. While open, the wrapper routes calls to the registry backup;
  after a cooldown one probe goes to the primary.
- ``escalate`` / ``EscalationResult``: hand-off to a responder (a human
  callback) on low classifier confidence or an exhausted budget. Always
  returns a structured result; never raises silently.
- ``exploit_choice`` / ``choose_arm``: tools with ``side_effect=True`` never
  get an exploratory choice -- the policy's highest-mean arm is returned
  without calling ``policy.select`` (no exploration noise is drawn).

Dependencies: ``Budget`` and ``TOOL_CALL_RESERVE_MS`` from
``recovery.live.executors.base`` only. Policies, classifiers and responders
are used by duck typing, never imported.

Every threshold below is a configuration choice with a documented default,
not a measured value.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

import numpy as np

from recovery.live.executors.base import TOOL_CALL_RESERVE_MS, Budget

__all__ = [
    "TaskBudget", "BreakerState", "BreakerConfig", "CircuitBreaker", "BreakerBoard",
    "EscalationResult", "escalate", "ESCALATION_REASONS", "DEFAULT_CONFIDENCE_THRESHOLD",
    "MeanTable", "policy_means", "exploit_choice", "choose_arm", "thompson_posterior", "thompson_propensity",
    "DEFAULT_PROPENSITY_DRAWS",
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.5   # configuration choice


# ----------------------------------------------------------------------
# per-task budget
# ----------------------------------------------------------------------

class TaskBudget:
    """Cumulative recovery spend for one task against one ``Budget``.

    ``attempts`` and ``tokens`` are recovery tool calls and LLM tokens (the
    agent's own first call is not a recovery attempt). Wall time is the time
    spent inside recovery executors. A strategy is admitted only if one
    attempt and one worst-case tool call (``max_call_ms``, default the
    existing ``TOOL_CALL_RESERVE_MS``) still fit; the executor then receives
    ``remaining()`` as its own ``Budget``, and the existing executors never
    start an action that does not fit it (v2 §3 rule). A tool that runs past
    ``max_call_ms`` can still overrun -- reported by ``exceeded()``, never
    killed (admission control, decided 2026-10-22).
    """

    def __init__(self, budget: Budget, max_call_ms: float = TOOL_CALL_RESERVE_MS) -> None:
        self.budget = budget
        self.max_call_ms = float(max_call_ms)
        self.attempts = 0
        self.tokens = 0
        self.wall_ms = 0.0
        self.strategy_runs = 0   # strategies started this task, whether or not they issued a call

    def remaining(self) -> Budget:
        return Budget(max_attempts=max(0, self.budget.max_attempts - self.attempts),
                      max_wall_ms=max(0.0, self.budget.max_wall_ms - self.wall_ms),
                      max_tokens=max(0, self.budget.max_tokens - self.tokens))

    def admits_strategy(self) -> bool:
        """Also caps strategy RUNS at ``max_attempts``: a strategy that issues
        no call (untranslatable args, an LLM call refused for tokens, an
        unanswered clarify question) spends no attempt, and without this cap
        a policy could pick it forever."""
        r = self.remaining()
        return (r.max_attempts >= 1 and self.strategy_runs < self.budget.max_attempts
                and r.max_wall_ms >= self.max_call_ms)

    def charge(self, attempts: int, tokens: int, wall_ms: float) -> None:
        self.strategy_runs += 1
        self.attempts += int(attempts)
        self.tokens += int(tokens)
        self.wall_ms += float(wall_ms)

    def exceeded(self) -> bool:
        b = self.budget
        return (self.attempts > b.max_attempts or self.tokens > b.max_tokens
                or self.wall_ms > b.max_wall_ms + 1e-6)

    def used(self) -> dict:
        return {"attempts": self.attempts, "tokens": self.tokens, "wall_ms": round(self.wall_ms, 1),
                "strategy_runs": self.strategy_runs}


# ----------------------------------------------------------------------
# circuit breaker
# ----------------------------------------------------------------------

class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerConfig:
    """Defaults are configuration choices: open after 3 consecutive failed
    calls, probe after 30 s, one probe at a time, close on one probe success."""

    failure_threshold: int = 3
    cooldown_s: float = 30.0
    half_open_probes: int = 1

    def __post_init__(self) -> None:
        if self.failure_threshold < 1 or self.half_open_probes < 1 or self.cooldown_s < 0:
            raise ValueError(f"invalid breaker config {self!r}")


class CircuitBreaker:
    """One tool's breaker. Every call issued to the tool -- the agent's call
    and recovery calls alike -- is recorded, since each is evidence of the
    tool's health.

    closed --(failure_threshold consecutive failures)--> open
    open --(cooldown_s elapsed)--> half_open
    half_open --(probe succeeds)--> closed;  --(probe fails)--> open (cooldown restarts)
    """

    def __init__(self, tool: str, config: BreakerConfig = BreakerConfig(),
                 clock: Callable[[], float] = time.monotonic,
                 on_transition: Optional[Callable[[dict], None]] = None) -> None:
        self.tool, self.config, self._clock = tool, config, clock
        self._state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self.opened_at: Optional[float] = None
        self.probes_in_flight = 0
        self.transitions: list[dict] = []
        self._on_transition = on_transition

    def _move(self, to: BreakerState, why: str) -> None:
        rec = {"tool": self.tool, "t": self._clock(), "from": self._state.value, "to": to.value, "why": why}
        self._state = to
        self.transitions.append(rec)
        if self._on_transition:
            self._on_transition(rec)

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and self._clock() - self.opened_at >= self.config.cooldown_s:
            self.probes_in_flight = 0
            self._move(BreakerState.HALF_OPEN, f"cooldown {self.config.cooldown_s} s elapsed")
        return self._state

    def allow(self) -> bool:
        """Whether a call may go to this tool now. In half-open, granting a
        call reserves one probe slot until its outcome is recorded."""
        s = self.state
        if s is BreakerState.CLOSED:
            return True
        if s is BreakerState.HALF_OPEN and self.probes_in_flight < self.config.half_open_probes:
            self.probes_in_flight += 1
            return True
        return False

    def record(self, success: bool) -> None:
        s = self.state
        if s is BreakerState.HALF_OPEN:
            self.probes_in_flight = max(0, self.probes_in_flight - 1)
            if success:
                self.consecutive_failures = 0
                self._move(BreakerState.CLOSED, "probe succeeded")
            else:
                self.opened_at = self._clock()
                self._move(BreakerState.OPEN, "probe failed")
            return
        if success:
            self.consecutive_failures = 0
            return
        self.consecutive_failures += 1
        if s is BreakerState.CLOSED and self.consecutive_failures >= self.config.failure_threshold:
            self.opened_at = self._clock()
            self._move(BreakerState.OPEN, f"{self.consecutive_failures} consecutive failures")


class BreakerBoard:
    """One breaker per tool name, created on first use, sharing one config,
    clock and transition log."""

    def __init__(self, config: BreakerConfig = BreakerConfig(), clock: Callable[[], float] = time.monotonic) -> None:
        self.config, self._clock = config, clock
        self._breakers: dict[str, CircuitBreaker] = {}
        self.transitions: list[dict] = []

    def get(self, tool: str) -> CircuitBreaker:
        if tool not in self._breakers:
            self._breakers[tool] = CircuitBreaker(tool, self.config, self._clock, self.transitions.append)
        return self._breakers[tool]

    def states(self) -> dict[str, str]:
        return {t: b.state.value for t, b in self._breakers.items()}


# ----------------------------------------------------------------------
# escalation
# ----------------------------------------------------------------------

ESCALATION_REASONS = frozenset({"low_confidence", "budget_exhausted"})


@dataclass
class EscalationResult:
    """What a wrapped tool returns when it hands the failure to a responder."""

    reason: str
    tool: str
    args: dict
    error: Optional[str]
    predicted_type: Optional[str]
    confidence: Optional[float]
    attempts_made: int
    budget_used: dict
    attempt_ids: list[str]
    question: str
    responder: Optional[str]
    answer: Optional[str]
    handled: bool
    responder_error: Optional[str] = None
    escalated: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.reason not in ESCALATION_REASONS:
            raise ValueError(f"escalation reason {self.reason!r} not in {sorted(ESCALATION_REASONS)}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_message(self) -> str:
        """A plain-text form for an agent's LLM (LangChain tool output)."""
        head = (f"[escalated: {self.reason}] tool {self.tool} failed after {self.attempts_made} "
                f"recovery attempt(s): {self.error}")
        if self.handled:
            return f"{head}\nHuman responder ({self.responder}) says: {self.answer}"
        return f"{head}\nNo responder answer; the tool result is unavailable."


def escalate(reason: str, *, tool: str, args: dict, error: Optional[str], predicted_type: Optional[str],
             confidence: Optional[float], attempts_made: int, budget_used: dict, attempt_ids: Sequence[str],
             task: Any, responder: Any = None, timeout_s: float = 60.0) -> EscalationResult:
    """Ask ``responder.answer(question, task, timeout_s)`` (the existing
    ``Responder`` protocol). A missing responder, a None answer, a timeout or
    a responder exception all give ``handled=False``, recorded, never raised."""
    question = (f"The agent's tool '{tool}' failed ({reason.replace('_', ' ')}). Error: {(error or '')[:200]}. "
                f"How should the agent proceed?")
    answer, err = None, None
    if responder is not None:
        try:
            answer = responder.answer(question, task, timeout_s)
        except Exception as exc:  # recorded on the result, not swallowed
            err = f"{type(exc).__name__}: {exc}"[:300]
    return EscalationResult(
        reason=reason, tool=tool, args=dict(args), error=error, predicted_type=predicted_type,
        confidence=confidence, attempts_made=attempts_made, budget_used=dict(budget_used),
        attempt_ids=list(attempt_ids), question=question,
        responder=getattr(responder, "kind", None) if responder is not None else None,
        answer=answer, handled=answer is not None, responder_error=err,
    )


# ----------------------------------------------------------------------
# side-effect tools: exploitation only
# ----------------------------------------------------------------------

class MeanTable:
    """Per (type, arm) success counts with a Beta(1, 1) mean -- the fallback
    'highest mean' for policies that expose no ``posterior_mean()``."""

    def __init__(self, n_types: int, n_arms: int) -> None:
        self.successes = np.zeros((n_types, n_arms))
        self.trials = np.zeros((n_types, n_arms))

    def update(self, t: int, arm: int, success: bool) -> None:
        self.trials[t, arm] += 1
        self.successes[t, arm] += bool(success)

    def means(self, t: int) -> np.ndarray:
        return (self.successes[t] + 1.0) / (self.trials[t] + 2.0)


def policy_means(policy: Any, t: int, fallback: MeanTable) -> tuple[np.ndarray, str]:
    pm = getattr(policy, "posterior_mean", None)
    if callable(pm):
        return np.asarray(pm()[t], dtype=float), "policy.posterior_mean"
    return fallback.means(t), "wrapper_mean_table"


def exploit_choice(means: Sequence[float], allowed: Sequence[int]) -> int:
    """Highest-mean arm among ``allowed``; ties go to the lowest index."""
    allowed = sorted(allowed)
    if not allowed:
        raise ValueError("no allowed arms")
    m = np.asarray(means, dtype=float)
    return max(allowed, key=lambda a: (m[a], -a))


DEFAULT_PROPENSITY_DRAWS = 1000   # configuration choice


def thompson_posterior(policy: Any, t: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """The Beta posterior row ``(alpha, beta)`` a Thompson policy samples
    from for type ``t``, or None for any other policy. Uses a
    ``thompson_posterior(t)`` method when the policy has one, else reads
    ``PriorInitializedThompson``'s ``_alpha`` / ``_beta`` arrays (copied,
    read-only; the policy is not modified)."""
    fn = getattr(policy, "thompson_posterior", None)
    if callable(fn):
        a, b = fn(t)
        return np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()
    a, b = getattr(policy, "_alpha", None), getattr(policy, "_beta", None)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.ndim == 2 and a.shape == b.shape:
        return a[t].astype(float).copy(), b[t].astype(float).copy()
    return None


def thompson_propensity(alpha: np.ndarray, beta: np.ndarray, arm: int, allowed: Sequence[int],
                        fallback_arm: int, draws: int, rng: np.random.Generator) -> float:
    """Fraction of ``draws`` posterior samples for which the decision rule
    -- argmax of a Thompson draw, with a draw landing on a masked arm
    replaced by ``fallback_arm`` -- picks ``arm``. With nothing masked this
    is the plain fraction of draws whose argmax is ``arm``."""
    theta = rng.beta(alpha, beta, size=(int(draws), len(alpha)))
    picks = theta.argmax(axis=1)
    executed = np.where(np.isin(picks, np.asarray(list(allowed))), picks, fallback_arm)
    return float((executed == arm).mean())


def choose_arm(policy: Any, ctx: Any, rng: np.random.Generator, *, side_effect: bool,
               allowed: Sequence[int], fallback: MeanTable,
               propensity_rng: Optional[np.random.Generator] = None,
               propensity_draws: int = DEFAULT_PROPENSITY_DRAWS) -> tuple[int, dict]:
    """The strategy for one failure, and how it was chosen.

    - ``side_effect=True``: ``exploit_choice`` over the policy's means;
      ``policy.select`` is never called. ``selection_mode="exploit_only"``,
      propensity 1.0 (deterministic).
    - Otherwise ``policy.select``. If the pick is runnable,
      ``selection_mode="policy"``; for a Thompson policy the propensity is
      estimated at decision time from ``propensity_draws`` posterior draws on
      ``propensity_rng`` (the decision's own seeded stream, separate from the
      one ``select`` used); a policy exposing ``selection_propensity(ctx,
      arm)`` reports its own; any other policy's is unknown.
    - If the pick is not runnable (breaker / applicability rules), the
      highest-mean runnable arm is forced: ``selection_mode="masked"``,
      ``masked_from`` = the policy's pick, propensity 1.0 (the forced arm is
      deterministic given that pick).

    ``masked_out_arms`` always lists the arms that could not run.
    """
    means, source = policy_means(policy, ctx.predicted_type, fallback)
    masked_out = sorted(set(range(len(means))) - set(allowed))
    if side_effect:
        return exploit_choice(means, allowed), {
            "selection_mode": "exploit_only", "means_source": source, "masked_out_arms": masked_out,
            "propensity": 1.0, "propensity_known": True, "propensity_method": "deterministic"}
    post = thompson_posterior(policy, ctx.predicted_type)      # decision-time posterior, before any update
    a = int(policy.select(ctx, rng))
    if a not in allowed:
        return exploit_choice(means, allowed), {
            "selection_mode": "masked", "masked_from": a, "masked_out_arms": masked_out, "means_source": source,
            "propensity": 1.0, "propensity_known": True, "propensity_method": "forced_by_mask"}
    info = {"selection_mode": "policy", "masked_out_arms": masked_out, "propensity": None,
            "propensity_known": False, "propensity_method": None}
    prop_fn = getattr(policy, "selection_propensity", None)
    if callable(prop_fn):
        p = prop_fn(ctx, a)
        info.update(propensity=p, propensity_known=p is not None, propensity_method="policy.selection_propensity")
    elif post is not None and propensity_rng is not None:
        est = thompson_propensity(post[0], post[1], a, allowed, exploit_choice(means, allowed),
                                  propensity_draws, propensity_rng)
        info.update(propensity_method="monte_carlo_posterior", propensity_draws=int(propensity_draws),
                    propensity_estimate=est)
        if est > 0:
            info.update(propensity=est, propensity_known=True)
        else:   # no draw reproduced the pick: not a usable estimate (raise propensity_draws)
            info["propensity_note"] = f"0 of {propensity_draws} draws picked the chosen arm"
    return a, info
