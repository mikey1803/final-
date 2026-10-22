"""``with_recovery``: failure-aware recovery for any agent tool (Task 22).

    from recovery.integrations import with_recovery
    weather = with_recovery(weather_tool)                     # plain callable or LangChain/LangGraph tool

On a failed call the wrapped tool classifies the failure, picks a strategy
with the policy, runs that strategy's executor, logs a ``RecoveryAttempt``,
updates the policy, and returns the recovered output -- or a structured
``EscalationResult`` when it hands off to the responder. A call that succeeds
is returned unchanged, with no extra call and no log record.

Tools wrapped by one ``RecoveryRuntime`` share its breakers, task budget,
policy and log::

    rt = RecoveryRuntime(policy=..., budget=Budget(...), responder=..., store="logs/with_recovery")
    wttr, meteo = rt.wrap(wttr_tool), rt.wrap(meteo_tool)
    with rt.task("q1", goal="weather in Kolkata"):          # one per-task budget
        graph.invoke(...)

Without ``rt.task(...)`` each top-level call is its own task.

Reused unchanged: the live ``Budget``/``FailureContext``/executors
(``build_executors("v2")`` for the six live tool names), ``tool_registry``
(backup + translation for those names), ``RecoveryAttempt``/``RecoveryStore``
(records carry ``context="with_recovery"``), the ``Policy`` protocol and
``Context``, ``PriorInitializedThompson`` (default policy, prior table
``BLIND_RULE``), and the trained Layer-1 classifier as fallback. Other tools
use ``recovery.integrations.executors`` (Layer 2).

Not pilot data: attempts logged here come from a different selection process
from any live-pilot collection and must never be pooled with it; the store
refuses live-pilot / v3 collection paths. ``selection_propensity`` is exact
only when ``executor_config.selection.propensity_known`` is true (exploit-only
choices: 1.0; policies exposing ``selection_propensity``); otherwise the
required field holds 1.0 and the attempt is not usable for off-policy
estimation.

Thread safety: calls through one runtime are serialised by a lock (LangGraph
may run tool calls concurrently). LangChain tools are wrapped synchronously.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from recovery.core.protocols import Context
from recovery.envs.canonical import BLIND_RULE, TYPE_NAMES
from recovery.guards.runtime import (
    DEFAULT_CONFIDENCE_THRESHOLD, BreakerBoard, BreakerConfig, BreakerState, EscalationResult, MeanTable,
    TaskBudget, choose_arm, escalate,
)
from recovery.integrations.classify import CIRCUIT_OPEN_PREFIX, Classification, RuleThenTrainedClassifier
from recovery.integrations.executors import LLM_ARMS, ToolCatalog, build_generic_executors
from recovery.live.event_schema import STRATEGY_NAMES, AgentState, FailureEvent, Task
from recovery.live.executors import build_executors
from recovery.live.executors.base import TOOL_CALL_RESERVE_MS, Budget, FailureContext
from recovery.live.executors.verify import VerifyExecutor
from recovery.live.recovery_attempt import RecoveryAttempt, RecoveryStore
from recovery.live.tool_registry import REGISTRY, ToolSpec, translate_args
from recovery.live.tools import DEFAULT_TIMEOUT_S, TOOL_DESCRIPTIONS, TOOLS, ToolCallResult
from recovery.policies.hybrid import PriorInitializedThompson

__all__ = ["with_recovery", "RecoveryRuntime", "RecoveryLog", "default_policy", "check_store_root"]

_CLIP = 2000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(text: Optional[str], n: int = _CLIP) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= n else text[:n] + "...[truncated]"


def default_policy() -> PriorInitializedThompson:
    """PGTS with the blind rule table and its documented defaults (strength 20, prior mean 0.7)."""
    return PriorInitializedThompson(BLIND_RULE, len(STRATEGY_NAMES), name="PGTS(BLIND_RULE)")


# ----------------------------------------------------------------------
# log (never the live-pilot / v3 collection store)
# ----------------------------------------------------------------------

def check_store_root(root: Union[str, Path]) -> Path:
    p = Path(root).resolve()
    for part in p.parts:
        low = part.lower()
        if "faarf-v3" in low or low.startswith("live_pilot"):
            raise ValueError(f"refusing store {p}: '{part}' looks like a live-pilot / v3 collection path; "
                             "with_recovery attempts are never written there")
    for d in (p, *p.parents):
        if (d / "v3_sessions").exists():
            raise ValueError(f"refusing store {p}: {d} is a v3 collection store (has v3_sessions/)")
    return p


class RecoveryLog:
    """Attempts and escalations, in memory and (with ``root``) on disk:
    ``<root>/recoveries/`` via the existing ``RecoveryStore`` and
    ``<root>/escalations/``. The call timeline (every issued or blocked call,
    for inspection) is kept in memory only, so a successful call writes
    nothing."""

    def __init__(self, root: Optional[Union[str, Path]] = None, timeline_max: int = 10_000) -> None:
        self.root = check_store_root(root) if root is not None else None
        self._store = RecoveryStore(self.root) if self.root is not None else None
        self.attempts: list[RecoveryAttempt] = []
        self.escalations: list[dict] = []
        self.calls: collections.deque = collections.deque(maxlen=timeline_max)

    def write_attempt(self, attempt: RecoveryAttempt) -> None:
        self.attempts.append(attempt)
        if self._store is not None:
            self._store.write(attempt)

    def write_escalation(self, rec: dict) -> None:
        self.escalations.append(rec)
        if self.root is not None:
            d = self.root / "escalations"
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob("*.json")))
            (d / f"{n:06d}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------
# tool adapters
# ----------------------------------------------------------------------

@dataclass
class _ToolEntry:
    name: str
    fn: Callable[[dict], Any]
    description: str
    args_desc: dict
    spec: ToolSpec
    translate: Optional[Callable[[dict], Optional[dict]]]
    is_failure: Optional[Callable[[Any], bool]]
    live: bool


def _langchain_base_tool():
    try:
        from langchain_core.tools import BaseTool
        return BaseTool
    except ImportError:  # pragma: no cover
        return None


def _status_of(exc: BaseException) -> Optional[int]:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _as_text(out: Any) -> str:
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out, default=str)
    except (TypeError, ValueError):
        return str(out)


@dataclass
class _TaskScope:
    task: Task
    budget: TaskBudget
    history: list = field(default_factory=list)
    auto: bool = False


# ----------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------

class RecoveryRuntime:
    def __init__(self, policy: Any = None, budget: Budget = Budget(), responder: Any = None, *,
                 store: Union[None, str, Path, RecoveryLog] = None, llm: Any = None, classifier: Any = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 breaker: BreakerConfig = BreakerConfig(), max_call_ms: float = TOOL_CALL_RESERVE_MS,
                 seed: int = 0, session_id: Optional[str] = None, escalation_timeout_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.policy = policy if policy is not None else default_policy()
        self.budget, self.responder, self.llm = budget, responder, llm
        self.classifier = classifier or RuleThenTrainedClassifier()
        self.confidence_threshold = float(confidence_threshold)
        self.max_call_ms = float(max_call_ms)
        self.seed, self.escalation_timeout_s = int(seed), float(escalation_timeout_s)
        self.session_id = session_id or f"with_recovery-{self.seed}"
        self.clock, self.sleep = clock, sleep
        self.log = store if isinstance(store, RecoveryLog) else RecoveryLog(store)
        self.breakers = BreakerBoard(breaker, clock)
        self.means = MeanTable(len(TYPE_NAMES), len(STRATEGY_NAMES))
        self.tools: dict[str, _ToolEntry] = {}
        self.failure_index = 0
        self._calls_seen = 0
        self._scope: Optional[_TaskScope] = None
        self._lock = threading.RLock()

        self._live_execs = build_executors("v2")
        if llm is None:   # the v2 VERIFY needs its judge; without an LLM only the schema check can run
            self._live_execs["verify"] = VerifyExecutor(schema_check=True, llm_judge=False)
        self._catalog = ToolCatalog(self._describe, self._backup_for)
        self._generic_execs = build_generic_executors(self._catalog)

    # ---------------------------------------------------------- wrapping
    def wrap(self, tool: Any, *, name: Optional[str] = None, description: Optional[str] = None,
             backup: Any = None, translate: Optional[Callable[[dict], Optional[dict]]] = None,
             side_effect: Optional[bool] = None, is_failure: Optional[Callable[[Any], bool]] = None,
             category: Optional[str] = None) -> Any:
        """Wrap a plain callable or a LangChain ``BaseTool``. ``backup`` is a
        tool name or tool object (registered here if new); ``translate`` maps
        args for the backup (None result = untranslatable)."""
        entry, lc = self._register(tool, name=name, description=description, backup=backup, translate=translate,
                                   side_effect=side_effect, is_failure=is_failure, category=category)
        if lc is not None:
            from langchain_core.tools import StructuredTool

            def _call(**kwargs):
                out = self._invoke(entry, kwargs)
                return out.to_message() if isinstance(out, EscalationResult) else out

            return StructuredTool.from_function(func=_call, name=entry.name, description=lc.description,
                                                args_schema=lc.args_schema, return_direct=lc.return_direct)

        sig = inspect.signature(tool)
        if any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
            raise TypeError(f"{entry.name}: tools taking *args cannot be wrapped (args must be named)")

        @functools.wraps(tool)
        def wrapped(*a, **kw):
            return self._invoke(entry, self._bind(sig, a, kw))

        wrapped.__recovery_runtime__ = self
        return wrapped

    @staticmethod
    def _bind(sig: inspect.Signature, a: tuple, kw: dict) -> dict:
        bound = sig.bind(*a, **kw)
        args = {}
        for k, v in bound.arguments.items():
            if sig.parameters[k].kind is inspect.Parameter.VAR_KEYWORD:
                args.update(v)
            else:
                args[k] = v
        return args

    def _register(self, tool, *, name, description, backup, translate, side_effect, is_failure, category):
        BaseTool = _langchain_base_tool()
        lc = tool if BaseTool is not None and isinstance(tool, BaseTool) else None
        if lc is not None:
            name = name or lc.name
            description = description or lc.description
            fn = lambda args, _t=lc: _t.invoke(args)   # noqa: E731
            schema = lc.args_schema
            fields = getattr(schema, "model_fields", None) if schema is not None and not isinstance(schema, dict) else None
            args_desc = ({k: getattr(f.annotation, "__name__", str(f.annotation)) for k, f in fields.items()}
                         if fields else dict(lc.args))
        elif callable(tool):
            name = name or getattr(tool, "__name__", None)
            if not name:
                raise ValueError("a tool without __name__ needs name=")
            description = description or (inspect.getdoc(tool) or name).split("\n")[0]
            fn = lambda args, _f=tool: _f(**args)   # noqa: E731
            args_desc = {k: (p.annotation.__name__ if isinstance(p.annotation, type) else str(p.annotation))
                         for k, p in inspect.signature(tool).parameters.items()}
        else:
            raise TypeError(f"cannot wrap {tool!r}: not callable and not a LangChain tool")

        backup_name = None
        if backup is not None:
            backup_name = backup if isinstance(backup, str) else self._register(
                backup, name=None, description=None, backup=None, translate=None, side_effect=None,
                is_failure=None, category=None)[0].name
        base = REGISTRY.get(name)
        live = name in TOOLS and backup is None and translate is None
        spec = ToolSpec(name=name, category=category or (base.category if base else "user"),
                        backup=backup_name or (base.backup if base and live else None),
                        side_effect=bool(side_effect) if side_effect is not None else (base.side_effect if base else False))
        entry = _ToolEntry(name, fn, description, args_desc, spec, translate, is_failure, live)
        self.tools[name] = entry
        return entry, lc

    # ---------------------------------------------------------- tasks
    @contextlib.contextmanager
    def task(self, task_id: str, goal: str = ""):
        """One per-task budget scope (``Budget`` caps apply to the whole task)."""
        with self._lock:
            if self._scope is not None:
                raise RuntimeError("a task is already open on this runtime")
            self._scope = _TaskScope(Task(task_id=task_id, goal=goal or task_id, category="with_recovery"),
                                     TaskBudget(self.budget, self.max_call_ms))
        try:
            yield self._scope
        finally:
            with self._lock:
                self._scope = None

    # ---------------------------------------------------------- registry
    def _is_live(self, tool: str) -> bool:
        e = self.tools.get(tool)
        return e.live if e is not None else tool in TOOLS

    def _backup_name(self, tool: str) -> Optional[str]:
        e = self.tools.get(tool)
        if e is not None:
            return e.spec.backup
        return REGISTRY[tool].backup if tool in REGISTRY else None

    def _backup_for(self, tool: str, args: dict) -> Optional[tuple[str, Optional[dict]]]:
        b = self._backup_name(tool)
        if not b:
            return None
        e = self.tools.get(tool)
        if e is not None and e.translate is not None:
            return b, e.translate(dict(args))
        if self._is_live(tool):
            tr = translate_args(tool, args)
            return b, (tr.args if tr.ok else None)
        return b, dict(args)

    def _describe(self) -> list[dict]:
        out = [{"name": e.name, "description": e.description, "args": e.args_desc} for e in self.tools.values()]
        known = {e.name for e in self.tools.values()}
        for e in list(self.tools.values()):
            b = e.spec.backup
            if b and b not in known and b in TOOLS:
                out += [d for d in TOOL_DESCRIPTIONS if d["name"] == b]
                known.add(b)
        return out

    def _runnable_arms(self, tool: str) -> list[int]:
        live = self._is_live(tool)
        tool_open = self.breakers.get(tool).state is BreakerState.OPEN
        arms = []
        for i, a in enumerate(STRATEGY_NAMES):
            if a in LLM_ARMS and self.llm is None:
                continue
            if a in ("retry", "verify") and tool_open and self._backup_name(tool):
                continue   # both re-issue the same call, which the open breaker would block
            if a == "verify" and self.llm is None and not live:
                continue
            if a == "verify" and self.llm is None and not live:
                continue
            if a == "switch_tool" and not self._backup_name(tool):
                continue
            arms.append(i)
        return arms or [STRATEGY_NAMES.index("retry")]

    def _executor(self, tool: str, arm: str):
        return (self._live_execs if self._is_live(tool) else self._generic_execs)[arm]

    # ---------------------------------------------------------- calls
    def _raw(self, name: str, args: dict) -> tuple[ToolCallResult, Any, Optional[BaseException]]:
        entry = self.tools.get(name)
        if entry is None:
            if name in TOOLS:   # an unregistered live tool (e.g. a backup): the live function itself
                r = TOOLS[name](dict(args), False, DEFAULT_TIMEOUT_S)
                return r, (r.full_text if r.full_text is not None else r.body), None
            return ToolCallResult(False, None, None, f"UnknownTool: {name!r} is not registered", 0.0), None, None
        t0 = self.clock()
        try:
            out = entry.fn(dict(args))
        except Exception as exc:
            ms = (self.clock() - t0) * 1000.0
            msg = f"{type(exc).__name__}: {exc}"
            status = _status_of(exc)
            if status is not None and not msg.startswith("HTTP "):
                msg = f"HTTP {status}: {msg}"
            return ToolCallResult(False, None, None, _clip(msg), ms), None, exc
        ms = (self.clock() - t0) * 1000.0
        text = _as_text(out)
        if entry.is_failure is not None and entry.is_failure(out):
            return ToolCallResult(False, None, _clip(text), f"FailedOutput: {_clip(text, 500)}", ms,
                                  full_text=text), out, None
        return ToolCallResult(True, None, _clip(text), None, ms, full_text=text), out, None

    def _gated(self, name: str, args: dict, purpose: str, preallowed: bool = False):
        br = self.breakers.get(name)
        if not preallowed and not br.allow():
            r = ToolCallResult(False, None, None,
                               f"{CIRCUIT_OPEN_PREFIX} {name} breaker is {br.state.value}; call not issued", 0.0)
            self._timeline(name, args, purpose, r, issued=False)
            return r, None, None
        r, out, exc = self._raw(name, args)
        br.record(r.success)
        self._timeline(name, args, purpose, r, issued=True)
        return r, out, exc

    def _timeline(self, name, args, purpose, r: ToolCallResult, issued: bool) -> None:
        self.log.calls.append({"t": self.clock(), "tool": name, "args": dict(args), "purpose": purpose,
                               "issued": issued, "success": r.success, "error": _clip(r.error_text, 160),
                               "latency_ms": round(r.latency_ms, 1), "breaker_after": self.breakers.get(name).state.value})

    # ---------------------------------------------------------- the wrapped call
    def _invoke(self, entry: _ToolEntry, args: dict) -> Any:
        with self._lock:
            self._calls_seen += 1
            auto = self._scope is None
            if auto:
                self._scope = _TaskScope(Task(task_id=f"call-{self._calls_seen}", goal=f"Use {entry.name}",
                                              category="with_recovery"),
                                         TaskBudget(self.budget, self.max_call_ms), auto=True)
            try:
                return self._invoke_in_scope(entry, args)
            finally:
                if auto:
                    self._scope = None

    def _invoke_in_scope(self, entry: _ToolEntry, args: dict) -> Any:
        br = self.breakers.get(entry.name)
        if br.allow():
            purpose = "probe" if br.state is BreakerState.HALF_OPEN else "agent"
            r, out, exc = self._gated(entry.name, args, purpose, preallowed=True)
            tool, targs = entry.name, dict(args)
        else:
            routed = self._backup_for(entry.name, args)
            if routed is None or routed[1] is None:
                r, out, exc = self._gated(entry.name, args, "agent")   # blocked: CircuitOpen, nothing issued
                tool, targs = entry.name, dict(args)
            else:
                tool, targs = routed
                r, out, exc = self._gated(tool, targs, f"routed_from:{entry.name}")
        if r.success:
            self._scope.history.append({"role": "tool", "tool": tool, "args": targs, "content": r.body})
            return out
        return self._recover(entry, args, tool, targs, r, exc)

    def _recover(self, entry: _ToolEntry, orig_args: dict, tool: str, targs: dict,
                 r: ToolCallResult, exc: Optional[BaseException]) -> Any:
        scope = self._scope
        attempt_ids: list[str] = []
        k = 0
        while True:
            cls: Classification = self.classifier.classify(r.error_text, tool, _status_of(exc) if exc else None, exc)
            if cls.confidence < self.confidence_threshold:
                return self._escalate("low_confidence", entry, orig_args, r, cls, attempt_ids)
            if not scope.budget.admits_strategy():
                return self._escalate("budget_exhausted", entry, orig_args, r, cls, attempt_ids)

            index = self.failure_index
            self.failure_index += 1
            seq = np.random.SeedSequence(self.seed, spawn_key=(index,))
            derived = int(seq.generate_state(1, dtype=np.uint64)[0])
            t = TYPE_NAMES.index(cls.failure_type)
            ctx = Context(predicted_type=t, tool_id=list(self.tools).index(entry.name), retry_count=k, episode=index)
            arm_i, selection = choose_arm(self.policy, ctx, np.random.default_rng(derived),
                                          side_effect=entry.spec.side_effect, allowed=self._runnable_arms(tool),
                                          fallback=self.means)
            arm = STRATEGY_NAMES[arm_i]
            executor = self._executor(tool, arm)
            event = FailureEvent(
                session_id=self.session_id, step_index=index, tool_name=tool, tool_args=dict(targs),
                raw_error=r.error_text or "", captured_at=_now(), task_id=scope.task.task_id,
                agent_state=AgentState(goal=scope.task.goal, conversation_history=list(scope.history),
                                       tool_call_index=len(scope.history)),
                http_status=r.http_status, latency_ms=r.latency_ms, source="with_recovery")
            calls: list = []

            def call_tool(name: str, a: dict, _arm=arm) -> ToolCallResult:
                res, o, e = self._gated(name, a, f"recovery:{_arm}")
                calls.append((res, o, e, name, dict(a)))
                return res

            fctx = FailureContext(
                event=event, task=scope.task, step_goal=f"Call {entry.name} ({entry.description}) for: {scope.task.goal}",
                tool_name=tool, tool_args=dict(targs), failed_result=r, history=list(scope.history),
                call_tool=call_tool, rng=random.Random(derived), llm=self.llm, responder=self.responder,
                sleep=self.sleep, clock=self.clock)
            remaining = scope.budget.remaining()
            started, t0 = _now(), self.clock()
            try:
                res = executor.execute(fctx, remaining).to_dict()
            except Exception as err:   # a bug in an executor must not crash the agent
                res = {"success": False, "cost_tokens": 0, "latency_ms": round((self.clock() - t0) * 1000, 1),
                       "attempts": sum(1 for c in calls), "evidence": {
                           "calls": [], "llm_calls": [], "answered_by": None, "reason": "executor_error",
                           "executor_error": f"{type(err).__name__}: {err}"[:300]}}
            wall_ms = (self.clock() - t0) * 1000.0
            scope.budget.charge(res["attempts"], res["cost_tokens"], wall_ms)
            ev = res["evidence"]
            prop = selection.get("propensity")
            attempt = RecoveryAttempt(
                event_id=event.event_id, session_id=self.session_id, task_id=scope.task.task_id, strategy=arm,
                definition_version="v2", selection_policy=getattr(self.policy, "name", type(self.policy).__name__),
                selection_propensity=float(prop) if prop else 1.0,
                seed={"runtime_seed": self.seed, "failure_index": index, "derived_seed": derived,
                      "method": "numpy.SeedSequence(runtime_seed, spawn_key=(failure_index,))"},
                model_id=next((c["model_id"] for c in ev.get("llm_calls", [])), None),
                executor_config={"name": executor.name, **executor.config(),
                                 "executors": "live_v2" if self._is_live(tool) else "with_recovery_generic",
                                 "selection": selection, "classification": {
                                     "type": cls.failure_type, "confidence": cls.confidence, "source": cls.source},
                                 "wrapped_tool": entry.name, "side_effect": entry.spec.side_effect,
                                 "breakers": self.breakers.states()},
                budget_limits=remaining.as_dict(),
                budget_used={"attempts": res["attempts"], "wall_ms": round(wall_ms, 1), "tokens": res["cost_tokens"]},
                budget_exceeded=bool(ev.get("budget_exceeded", False)) or scope.budget.exceeded(),
                result=res, started_at=started, finished_at=_now(), aborted=bool(ev.get("aborted", False)),
                no_action_taken=bool(ev.get("no_action_taken", False)), context="with_recovery")
            self.log.write_attempt(attempt)
            attempt_ids.append(attempt.attempt_id)
            success = bool(res["success"])
            self.policy.update(ctx, arm_i, int(success))
            self.means.update(t, arm_i, success)
            if success:
                good = [c for c in calls if c[0].success]
                if good:
                    res_ok, out, _, name, a = good[-1]
                    scope.history.append({"role": "tool", "tool": name, "args": a, "content": res_ok.body})
                    return out
                return r.full_text if r.full_text is not None else r.body   # VERIFY accepted the original result
            failed = [c for c in calls if not c[0].success]
            if failed:
                r, exc = failed[-1][0], failed[-1][2]
            k += 1

    def _escalate(self, reason, entry, args, r, cls, attempt_ids) -> EscalationResult:
        scope = self._scope
        e = escalate(reason, tool=entry.name, args=args, error=r.error_text, predicted_type=cls.failure_type,
                     confidence=cls.confidence, attempts_made=len(attempt_ids), budget_used=scope.budget.used(),
                     attempt_ids=attempt_ids, task=scope.task, responder=self.responder,
                     timeout_s=self.escalation_timeout_s)
        self.log.write_escalation({**e.to_dict(), "task_id": scope.task.task_id, "at": _now(),
                                   "classification_source": cls.source})
        return e


def with_recovery(tool: Any, policy: Any = None, budget: Budget = Budget(), responder: Any = None, *,
                  runtime: Optional[RecoveryRuntime] = None, backup: Any = None,
                  translate: Optional[Callable[[dict], Optional[dict]]] = None, side_effect: Optional[bool] = None,
                  is_failure: Optional[Callable[[Any], bool]] = None, name: Optional[str] = None,
                  **runtime_kwargs: Any) -> Any:
    """Wrap one tool. Without ``runtime`` a private ``RecoveryRuntime`` is
    built from ``policy``, ``budget``, ``responder`` and ``runtime_kwargs``
    (``store``, ``llm``, ``classifier``, ``confidence_threshold``, ``breaker``,
    ``seed``, ...). Pass the same ``runtime`` to share breakers, budget,
    policy and log across tools."""
    if runtime is None:
        runtime = RecoveryRuntime(policy=policy, budget=budget, responder=responder, **runtime_kwargs)
    elif policy is not None or responder is not None or runtime_kwargs:
        raise ValueError("pass policy/responder/runtime options to the RecoveryRuntime, not alongside runtime=")
    return runtime.wrap(tool, backup=backup, translate=translate, side_effect=side_effect, is_failure=is_failure,
                        name=name)
