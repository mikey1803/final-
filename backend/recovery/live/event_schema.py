"""Data model for the live-pilot failure event store.

See ``docs/live_pilot/strategy_definitions.md`` for the frozen instrument
this schema serves. Two record types:

- ``FailureEvent`` -- one captured failure from a live task run. Phase 1
  produces these and the labelling CLI fills in ``true_type``.
- ``ReplayResult`` -- one applied strategy trial against an already-labelled
  event. Schema is frozen now (per sign-off) so Phase 2 can be built against
  a stable contract, but no Phase 1 code path constructs one.

Both are content-addressed the same way ``recovery.core.parallel.RunConfig``
addresses a sweep cell: a deterministic sha256 of the fields that define
*identity*, never of large payload fields (``raw_error``,
``conversation_history``, ...) which can vary in ways that shouldn't split
otherwise-identical records into different addresses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.failures.taxonomy import FailureType

SCHEMA_VERSION = "1"

# The frozen six, in the canonical order used throughout this project
# (matches recovery.envs.canonical.STRATEGY_NAMES) plus the control arm.
STRATEGY_NAMES: list[str] = [
    "retry", "replan", "switch_tool", "re_ground", "verify", "clarify",
]
NULL_CONTROL = "null_control"
ALL_ARMS: list[str] = STRATEGY_NAMES + [NULL_CONTROL]

# tool_timeout/tool_error come from FlakyWrapper injection on ordinary
# task execution, never from task design -- a task cannot legitimately
# "target" either. The other five FailureType values are agent-side and
# can be deliberately elicited by task design (Task.targeted=True).
NOT_TASK_TARGETABLE_TYPES: frozenset[str] = frozenset({"tool_timeout", "tool_error"})
AGENT_SIDE_TARGETABLE_TYPES: frozenset[str] = frozenset(
    ft.value for ft in FailureType
) - NOT_TASK_TARGETABLE_TYPES


@dataclass
class AgentState:
    """Everything needed to reconstruct the agent's state at the point of
    failure, for either a human reviewing the event or a Phase 2 replay.

    ``conversation_history`` is truncated to end exactly at the failing
    call -- it must never include anything after ``tool_call_index``, or a
    replay would leak information the agent didn't actually have yet.
    """

    goal: str
    conversation_history: list[dict] = field(default_factory=list)
    tool_call_index: int = 0


@dataclass
class FailureEvent:
    """One captured failure from a live task run.

    ``event_id`` is computed once at construction (via ``__post_init__``)
    from the identity fields only -- two events with the same session,
    step, tool, and error text are the same event even if captured twice.
    """

    session_id: str
    step_index: int
    tool_name: str
    tool_args: dict
    raw_error: str
    agent_state: AgentState
    captured_at: str
    task_id: Optional[str] = None  # which manifest Task produced this, for intended-vs-observed
    http_status: Optional[int] = None
    raw_response_body: Optional[str] = None
    latency_ms: Optional[float] = None
    injected: bool = False
    injected_kind: Optional[str] = None  # "timeout" | "malformed_request" | None

    # All-6-tools health check taken once at collection-session start,
    # attached unchanged to every event captured that session -- see
    # recovery.live.tools.health_check_all_tools. Not re-derivable later:
    # third-party services flake/recover independently of when we replay.
    tool_health: Optional[dict] = None

    true_type: Optional[str] = None
    labelled_by: Optional[str] = None
    labelled_at: Optional[str] = None
    second_choice_type: Optional[str] = None
    label_confidence: Optional[str] = None  # "confident" | "unsure" | None

    # Second, independent blind label for the same event -- used only by the
    # relabelling-mode Cohen's-kappa check. Never influences true_type.
    relabel_type: Optional[str] = None
    relabelled_by: Optional[str] = None
    relabelled_at: Optional[str] = None

    predicted_type: Optional[str] = None
    predicted_proba: Optional[dict] = None

    schema_version: str = SCHEMA_VERSION
    source: str = "live"

    # v3 only (docs/live_pilot/strategy_definitions_v3.md §2, §8): fault_origin,
    # injection_mode, block_index, injection_params, agent_side_kind,
    # detected_by, episode fields. Not an identity field (event_id unchanged),
    # and omitted from to_dict() when None, so v1/v2 events serialise exactly
    # as before.
    v3: Optional[dict] = None

    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.event_id = self._compute_event_id()

    def _compute_event_id(self) -> str:
        payload = {
            "session_id": self.session_id,
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "raw_error": self.raw_error,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def is_labelled(self) -> bool:
        return self.true_type is not None

    def label(
        self,
        true_type: FailureType,
        labelled_by: str,
        labelled_at: str,
        second_choice_type: Optional[FailureType] = None,
        confidence: Optional[str] = None,
    ) -> None:
        """Apply a human label. Does not re-derive or validate the type
        against anything -- the human's judgement is the ground truth this
        whole pilot exists to collect.

        ``second_choice_type``/``confidence`` are optional secondary signal
        (not the primary label): the human's runner-up type and how sure
        they were, used to bound achievable classifier accuracy, never to
        adjust ``true_type`` itself.
        """
        if confidence is not None and confidence not in ("confident", "unsure"):
            raise ValueError(f"confidence must be 'confident', 'unsure', or None, got {confidence!r}")
        self.true_type = true_type.value
        self.labelled_by = labelled_by
        self.labelled_at = labelled_at
        self.second_choice_type = second_choice_type.value if second_choice_type else None
        self.label_confidence = confidence

    def relabel(self, relabel_type: FailureType, relabelled_by: str, relabelled_at: str) -> None:
        """Apply a second, independent blind label for the agreement check
        (Cohen's kappa). Deliberately separate from ``true_type`` -- this
        must never overwrite the original label."""
        self.relabel_type = relabel_type.value
        self.relabelled_by = relabelled_by
        self.relabelled_at = relabelled_at

    def validate_label_prediction_order(self) -> None:
        """Structural guard for labelling integrity: a classifier prediction
        must never exist on an event before that event has a human label.
        Called by ``EventStore.write`` on every write, so this invalid state
        can never be persisted -- not just caught after the fact by a test."""
        if self.predicted_type is not None and self.true_type is None:
            raise ValueError(
                f"event {self.event_id}: predicted_type is set ({self.predicted_type!r}) "
                "but true_type is None -- prediction must run strictly after labelling"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["v3"] is None:
            del d["v3"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FailureEvent":
        d = dict(d)
        event_id = d.pop("event_id", None)
        agent_state_d = d.pop("agent_state")
        agent_state = AgentState(**agent_state_d)
        obj = cls(agent_state=agent_state, **d)
        if event_id is not None and event_id != obj.event_id:
            raise ValueError(
                f"stored event_id {event_id!r} does not match recomputed "
                f"{obj.event_id!r} -- identity fields were edited after capture"
            )
        return obj

    def index_row(self) -> dict:
        """Flat scalar-only view for the parquet index (no nested fields --
        agent_state.conversation_history lives only in the JSON record)."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "http_status": self.http_status,
            "latency_ms": self.latency_ms,
            "injected": self.injected,
            "injected_kind": self.injected_kind,
            "captured_at": self.captured_at,
            "true_type": self.true_type,
            "labelled_by": self.labelled_by,
            "labelled_at": self.labelled_at,
            "second_choice_type": self.second_choice_type,
            "label_confidence": self.label_confidence,
            "relabel_type": self.relabel_type,
            "relabelled_by": self.relabelled_by,
            "relabelled_at": self.relabelled_at,
            "predicted_type": self.predicted_type,
            "schema_version": self.schema_version,
            "source": self.source,
        }


@dataclass
class ReplayResult:
    """One applied-strategy trial against an already-labelled event.

    Phase 2 schema, frozen now, unused by any Phase 1 code path. See
    ``docs/live_pilot/strategy_definitions.md`` for what each field means
    and why (the ``question_unmatched`` outcome, ``replan_chosen_tool``
    overlap logging, the non-independence-of-same-event-rows note).
    """

    event_id: str
    strategy: str  # one of ALL_ARMS
    trial_index: int  # 0-based actual execution order (post-randomization)
    replayed_at: str
    true_type: str  # copied from the event's label; replay is invalid without one

    success: Optional[bool] = None  # None when aborted (e.g. question_unmatched)
    aborted: bool = False
    abort_reason: Optional[str] = None  # e.g. "question_unmatched"
    outcome_detail: str = ""

    replan_chosen_tool: Optional[str] = None  # only set when strategy == "replan"

    wall_clock_ms: Optional[float] = None
    api_cost_usd: Optional[float] = None

    schema_version: str = SCHEMA_VERSION
    replay_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.strategy not in ALL_ARMS:
            raise ValueError(f"strategy must be one of {ALL_ARMS}, got {self.strategy!r}")
        payload = {
            "event_id": self.event_id,
            "strategy": self.strategy,
            "trial_index": self.trial_index,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.replay_id = hashlib.sha256(encoded).hexdigest()[:24]


@dataclass
class Task:
    """One task the live agent attempts -- the unit of task-design work
    referenced in the frozen instrument doc. Deliberately minimal: task
    authoring (especially the ambiguous/long-context/out-of-scope ones
    meant to elicit the four agent-side failure types) is domain judgement,
    not something to mass-generate.
    """

    task_id: str
    goal: str
    category: str  # free text, e.g. "weather_lookup", "ambiguous", "long_context"
    steps: list[str] = field(default_factory=list)  # multi-step tasks: sub-goals in order
    clarify_canned_answer: Optional[str] = None  # authored before any run, per the freeze rule

    # The FailureType this task was authored to elicit, or None for a task
    # meant to succeed cleanly (a control, not a failure-eliciting task).
    # Declared before collection so intended-vs-observed can be reported
    # afterward -- see recovery.live.task_manifest.
    intended_failure_type: Optional[str] = None

    # True for a task deliberately designed/steered to elicit
    # intended_failure_type (one of the five agent-side types); False for
    # an ordinary/natural task, where tool_timeout/tool_error can still
    # occur via FlakyWrapper injection but nothing about the task design
    # steers toward any particular type. This flag is what keeps the
    # balanced confusion-matrix design counts and the deployment
    # distribution p structurally separate -- see
    # recovery.live.deployment_distribution, which only ever reads
    # targeted=False events.
    targeted: bool = False

    def __post_init__(self) -> None:
        if self.intended_failure_type is not None:
            valid = {ft.value for ft in FailureType}
            if self.intended_failure_type not in valid:
                raise ValueError(
                    f"intended_failure_type {self.intended_failure_type!r} must be one of "
                    f"{sorted(valid)} or None"
                )

        if self.targeted:
            if self.intended_failure_type is None:
                raise ValueError("targeted=True task must declare intended_failure_type")
            if self.intended_failure_type not in AGENT_SIDE_TARGETABLE_TYPES:
                raise ValueError(
                    f"{self.intended_failure_type!r} is not task-targetable -- it comes from "
                    "FlakyWrapper injection on ordinary tasks, not task design. Set "
                    "targeted=False and leave intended_failure_type=None instead."
                )
        elif self.intended_failure_type is not None:
            raise ValueError(
                "targeted=False task must not declare intended_failure_type -- a natural/"
                "untargeted task isn't designed to elicit any particular type"
            )
