"""Deliberate failure injection for testing agent robustness.

The FailureInjector class provides a controlled way to introduce realistic
failures into the agent's execution pipeline.  Each call to ``inject()``
randomly decides (based on the configured rate) whether a failure should
occur, and if so produces a concrete FailureEvent that simulates one of the
recognised failure modes with plausible-looking erroneous output.

Each failure type is rendered through multiple differently-worded variants
with randomised details (error codes, durations, tool names, phrasing) so
that failures of the same type produce realistically diverse text rather
than near-identical templates.  Crucially, many variants deliberately
overlap in vocabulary and structure with other failure types, ensuring that
the downstream classifier cannot rely on simple keyword matching.
"""

import json
import random
import time
from typing import Optional

from src.failures.taxonomy import FailureEvent, FailureType


# Shared pool of tool names reused across several simulators.
_TOOL_POOL: list[str] = [
    "web_search", "calculator", "file_reader", "api_caller",
    "database_query", "code_executor", "translator", "summariser",
]


class FailureInjector:
    """Injects realistic failures into the agent pipeline at a configurable rate.

    Args:
        injection_rate: Probability (0.0–1.0) that any single call to
            ``inject()`` will produce a failure.  For example, 0.3 means a
            30 % chance per call.
        seed: Optional random seed for reproducibility during experiments.
    """

    def __init__(self, injection_rate: float = 0.3, seed: Optional[int] = None) -> None:
        if not 0.0 <= injection_rate <= 1.0:
            raise ValueError("injection_rate must be between 0.0 and 1.0")
        self.injection_rate = injection_rate
        self._rng = random.Random(seed)
        self._injection_log: list[FailureEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, step_context: dict) -> Optional[FailureEvent]:
        """Optionally inject a failure into the current step.

        With probability ``injection_rate`` this method picks a random
        ``FailureType``, builds a realistic erroneous output, wraps it in a
        ``FailureEvent``, appends it to the internal log, and returns it.
        Otherwise it returns ``None``.

        Args:
            step_context: Metadata about what the agent is currently doing.
                The dict is forwarded into the ``FailureEvent.step_context``
                field unchanged.

        Returns:
            A ``FailureEvent`` if a failure was injected, else ``None``.
        """
        if self._rng.random() >= self.injection_rate:
            return None

        failure_type = self._rng.choice(list(FailureType))
        event = self._build_event(failure_type, step_context)
        self._injection_log.append(event)
        return event

    def get_injection_log(self) -> list[FailureEvent]:
        """Return a copy of every failure injected so far.

        Useful for post-hoc analysis, evaluation harnesses, and dashboard
        visualisation.
        """
        return list(self._injection_log)

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    def _build_event(self, failure_type: FailureType, step_context: dict) -> FailureEvent:
        """Dispatch to the type-specific simulator and return a FailureEvent."""
        simulators = {
            FailureType.TOOL_TIMEOUT: self._sim_tool_timeout,
            FailureType.TOOL_ERROR: self._sim_tool_error,
            FailureType.WRONG_TOOL_SELECTED: self._sim_wrong_tool_selected,
            FailureType.HALLUCINATED_OUTPUT: self._sim_hallucinated_output,
            FailureType.MALFORMED_OUTPUT: self._sim_malformed_output,
            FailureType.CONTEXT_LOSS: self._sim_context_loss,
            FailureType.AMBIGUOUS_INSTRUCTION: self._sim_ambiguous_instruction,
        }
        raw_error = simulators[failure_type](step_context)
        return FailureEvent(
            failure_type=failure_type,
            step_context=step_context,
            raw_error=raw_error,
        )

    def _pick_tool(self, step_context: dict) -> str:
        """Return a tool name — sometimes from context, sometimes random."""
        if self._rng.random() < 0.6:
            return step_context.get("tool_name", self._rng.choice(_TOOL_POOL))
        return self._rng.choice(_TOOL_POOL)

    def _other_tool(self, exclude: str) -> str:
        """Pick a tool name that is *not* ``exclude``."""
        return self._rng.choice([t for t in _TOOL_POOL if t != exclude])

    # ------------------------------------------------------------------
    # TOOL_TIMEOUT
    #
    # 7 variants.  ~3 are distinctively timeout-flavoured.
    #  ~2 overlap with TOOL_ERROR (HTTP error + timeout, connection error)
    #  ~1 overlaps with MALFORMED_OUTPUT (partial JSON from interrupted write)
    # ------------------------------------------------------------------

    def _sim_tool_timeout(self, step_context: dict) -> str:
        """Simulate a tool call that takes too long or stops responding."""
        tool = self._pick_tool(step_context)
        dur = self._rng.choice([3, 5, 8, 10, 15, 30, 45, 60, 90, 120])
        style = self._rng.randint(0, 6)

        if style == 0:
            return f"Request to {tool} timed out after {dur}s. No data was received within the deadline."

        elif style == 1:
            return f"TimeoutError: no response from {tool} within {dur} seconds"

        elif style == 2:
            return f"[WARN] {tool} did not respond, connection dropped after {dur}s wait."

        elif style == 3:
            return (
                f"urllib3.exceptions.ReadTimeout: Read timed out. "
                f"(read timeout={dur})\n"
                f"  at {tool}.execute (client.py:{self._rng.randint(200,800)})"
            )

        elif style == 4:
            # Overlap with TOOL_ERROR — phrased as HTTP error
            return f"HTTP 504 Gateway Timeout from {tool} after {dur}s. Upstream service did not respond."

        elif style == 5:
            # Overlap with TOOL_ERROR — generic connection failure
            return json.dumps({"error": f"connection to {tool} timed out", "code": 110, "seconds": dur})

        else:
            # Overlap with MALFORMED_OUTPUT — interrupted mid-write
            return '{"result": "partial'  # truncated because connection dropped

    # ------------------------------------------------------------------
    # TOOL_ERROR
    #
    # 7 variants.  ~3 are distinctively error-flavoured.
    #  ~2 overlap with MALFORMED_OUTPUT (broken JSON, HTML page)
    #  ~1 overlaps with WRONG_TOOL_SELECTED (error caused by wrong tool)
    #  ~1 overlaps with TOOL_TIMEOUT (connection timeout as error)
    # ------------------------------------------------------------------

    def _sim_tool_error(self, step_context: dict) -> str:
        """Simulate a tool returning an error or garbage response."""
        tool = self._pick_tool(step_context)
        style = self._rng.randint(0, 6)

        if style == 0:
            code = self._rng.choice([400, 401, 403, 404, 429, 500, 502, 503])
            msgs = {400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
                    404: "Not Found", 429: "Too Many Requests",
                    500: "Internal Server Error", 502: "Bad Gateway",
                    503: "Service Unavailable"}
            return json.dumps({"error": f"HTTP {code} {msgs[code]}", "tool": tool})

        elif style == 1:
            reason = self._rng.choice([
                "connection reset by peer", "connection refused",
                "SSL handshake failed", "DNS resolution failed",
            ])
            return f"{tool}: {reason}"

        elif style == 2:
            return (
                f"Traceback (most recent call last):\n"
                f"  File \"{tool}_client.py\", line {self._rng.randint(40,300)}\n"
                f"    raise {self._rng.choice(['APIError','ServiceError','ConnectionError'])}\n"
                f"Error: service {tool} unavailable"
            )

        elif style == 3:
            # Overlap with MALFORMED_OUTPUT — error response is broken JSON
            return '{"error": "internal failure", "raw": '

        elif style == 4:
            # Overlap with MALFORMED_OUTPUT — HTML error page
            return f"<html><body><h1>502 Bad Gateway</h1><p>{tool}</p></body></html>"

        elif style == 5:
            # Overlap with WRONG_TOOL_SELECTED — error because wrong tool was used
            wrong = self._other_tool(tool)
            return json.dumps({"error": f"Tool '{wrong}' cannot process this request", "suggestion": f"Use '{tool}' instead", "code": 422})

        else:
            # Overlap with TOOL_TIMEOUT
            secs = self._rng.choice([5, 10, 15, 30])
            return json.dumps({"error": f"ConnectTimeout on {tool}", "detail": f"timed out after {secs}s", "errno": 110})

    # ------------------------------------------------------------------
    # WRONG_TOOL_SELECTED
    #
    # 7 variants.  ~3 are distinctively about tool mismatch.
    #  ~2 overlap with TOOL_ERROR (error caused by wrong tool)
    #  ~1 overlaps with CONTEXT_LOSS (agent lost track of which tool to use)
    #  ~1 overlaps with AMBIGUOUS_INSTRUCTION (misread the task)
    # ------------------------------------------------------------------

    def _sim_wrong_tool_selected(self, step_context: dict) -> str:
        """Simulate the agent calling a plausible-but-inappropriate tool."""
        correct = step_context.get("tool_name", self._rng.choice(_TOOL_POOL))
        wrong = self._other_tool(correct)
        style = self._rng.randint(0, 6)

        if style == 0:
            return json.dumps({
                "tool_called": wrong, "intended_tool": correct,
                "detail": f"Agent dispatched to '{wrong}' but the task requires '{correct}'.",
                "tool_args": step_context.get("tool_args", {}),
            })

        elif style == 1:
            return f"WRONG_TOOL  selected={wrong}  expected={correct}  step={step_context.get('step_index', '?')}"

        elif style == 2:
            return (
                f"Attempted to use {wrong} for this subtask, but the task "
                f"demands {correct}.  {wrong} does not have the capability "
                f"to {step_context.get('goal', 'perform this operation')}."
            )

        elif style == 3:
            # Overlap with TOOL_ERROR — the wrong tool returns an error
            return json.dumps({"error": f"Tool '{wrong}' returned HTTP 422 Unprocessable Entity", "tool": wrong})

        elif style == 4:
            # Overlap with CONTEXT_LOSS — agent forgot which tool to use
            return (
                f"I think we haven't tried {wrong} yet, let me use that. "
                f"(Actually {correct} was already invoked at step "
                f"{self._rng.randint(1,20)} and produced results.)"
            )

        elif style == 5:
            # Overlap with AMBIGUOUS_INSTRUCTION — misread the task requirements
            return (
                f"The user's request was unclear about whether to use "
                f"{wrong} or {correct}.  Proceeding with {wrong}."
            )

        else:
            # Terse diagnostic — no clear marker for what type this is
            return f"dispatch={wrong}  (expected={correct})  status=error"

    # ------------------------------------------------------------------
    # HALLUCINATED_OUTPUT
    #
    # 7 variants.  ~2 clearly marked as hallucination.
    #  ~2 blend naturally into the text (very hard to detect).
    #  ~1 overlaps with CONTEXT_LOSS (fabricated prior step reference)
    #  ~1 overlaps with MALFORMED_OUTPUT (hallucination breaks JSON)
    #  ~1 overlaps with WRONG_TOOL_SELECTED (hallucinated tool output)
    # ------------------------------------------------------------------

    _FACTS = [
        "The capital of Australia is Sydney.",
        "Python was created by James Gosling in 1995.",
        "The boiling point of water at sea level is 90°C.",
        "The Great Wall of China is visible from the Moon.",
        "Diamonds are made of compressed coal.",
        "Sound travels faster in air than in water.",
        "The human body has 206 bones at birth.",
        "The Amazon River flows north to south.",
        "Mercury is the hottest planet in the solar system.",
        "Sharks are mammals, not fish.",
        "Mount Everest grows by approximately 1 metre every year.",
        "Goldfish have a memory span of only three seconds.",
    ]

    def _sim_hallucinated_output(self, step_context: dict) -> str:
        """Inject a fabricated fact into a text response."""
        original = step_context.get("tool_output", "No prior output.")
        fact = self._rng.choice(self._FACTS)
        tool = self._pick_tool(step_context)
        style = self._rng.randint(0, 6)

        if style == 0:
            # Explicitly marked
            return f"{original}\n\n[HALLUCINATED INSERT]: {fact}"

        elif style == 1:
            # Blended into paragraph — no marker
            parts = original.split(". ")
            parts.insert(self._rng.randint(0, max(1, len(parts))), fact)
            return ". ".join(parts)

        elif style == 2:
            # Appended as continuation
            return f"{original}\n\nAdditionally, {fact.lower()}"

        elif style == 3:
            # Overlap with CONTEXT_LOSS — fabricated step reference
            ghost = self._rng.randint(1, 50)
            return (
                f"As confirmed in step {ghost}, {fact.lower()}  "
                f"Building on that result to continue."
            )

        elif style == 4:
            # Overlap with MALFORMED_OUTPUT — hallucination in broken JSON
            return json.dumps({"answer": original[:60], "extra_fact": fact, "confidence": "high"})

        elif style == 5:
            # Overlap with WRONG_TOOL_SELECTED — wrong tool produces hallucinated text
            return f"{tool} output: {fact}  (Note: this information was fabricated by the model.)"

        else:
            # Short confident assertion — very subtle
            return f"Analysis result: {original}  Key fact: {fact}"

    # ------------------------------------------------------------------
    # MALFORMED_OUTPUT
    #
    # 7 variants.  ~3 are clearly structural failures.
    #  ~2 overlap with TOOL_ERROR (error response that's also malformed)
    #  ~1 overlaps with TOOL_TIMEOUT (truncated because connection dropped)
    #  ~1 overlaps with HALLUCINATED_OUTPUT (wrong-type value in JSON)
    # ------------------------------------------------------------------

    def _sim_malformed_output(self, step_context: dict) -> str:
        """Break the expected structure of a tool response."""
        tool = self._pick_tool(step_context)
        style = self._rng.randint(0, 6)

        if style == 0:
            # Truncated mid-write
            partial = '{"result": "ok", "data": ['
            for _ in range(self._rng.randint(1, 3)):
                partial += f'"{self._rng.choice(["a","b","c"])}",'
            return partial

        elif style == 1:
            return self._rng.choice(['42', '"just a string"', '[1, 2, 3]', 'null', 'true'])

        elif style == 2:
            return self._rng.choice([
                "binary garbage \\x00\\x01\\x02 here",
                "\\uFFFD replacement character \uFFFD sequence",
            ])

        elif style == 3:
            # Overlap with TOOL_ERROR — malformed error response
            return json.dumps({"error": "response_decode_failure", "raw": f"unparseable output from {tool}", "http_status": 200})

        elif style == 4:
            # Overlap with TOOL_ERROR — HTML instead of JSON
            return f"<html><body><h1>502 Bad Gateway</h1><p>{tool}</p></body></html>"

        elif style == 5:
            # Overlap with TOOL_TIMEOUT — truncated because connection dropped
            return '{"result": "partial'  # truncated

        else:
            # Overlap with HALLUCINATED_OUTPUT — a string where a number expected
            return json.dumps({"count": "fifteen", "status": "ok", "data": []})

    # ------------------------------------------------------------------
    # CONTEXT_LOSS
    #
    # 7 variants.  ~2 are distinctively about lost context.
    #  ~2 overlap with AMBIGUOUS_INSTRUCTION (vague/confused reasoning)
    #  ~2 overlap with HALLUCINATED_OUTPUT (fabricated step reference)
    #  ~1 overlaps with WRONG_TOOL_SELECTED (forgot which tool was used)
    # ------------------------------------------------------------------

    def _sim_context_loss(self, step_context: dict) -> str:
        """Simulate the agent losing track of earlier steps or the goal."""
        step_index = step_context.get("step_index", 0)
        ghost = self._rng.randint(1, 200)
        while ghost == step_index:
            ghost = self._rng.randint(1, 200)
        tool = self._pick_tool(step_context)
        style = self._rng.randint(0, 6)

        if style == 0:
            return (
                f"Referencing output from step {ghost}: the result was X. "
                f"Proceeding based on that. (Note: step {ghost} never "
                f"actually occurred — context was lost.)"
            )

        elif style == 1:
            goal = step_context.get("goal", "the original task")
            return (
                f"Goal has been updated: now focusing on completing step {ghost} "
                f"of the workflow. (Original goal was: {goal}.)"
            )

        elif style == 2:
            # Overlap with WRONG_TOOL_SELECTED — forgot which tool was used
            return (
                f"Let me try {tool} for this. We haven't used it yet. "
                f"(Actually {tool} was called at step {ghost}.)"
            )

        elif style == 3:
            # Overlap with AMBIGUOUS_INSTRUCTION — vague restatement
            return (
                f"The user asked for something about "
                f"{step_context.get('goal', 'the thing')}.  I think we need "
                f"to process the data first, then...  what was the request?"
            )

        elif style == 4:
            # Overlap with HALLUCINATED_OUTPUT — fabricated prior result
            return (
                f"As we established in step {ghost}, the data shows the "
                f"system is operational. Proceeding with the next action."
            )

        elif style == 5:
            # Overlap with AMBIGUOUS_INSTRUCTION — lost scope
            return (
                f"I believe the scope was '{self._rng.choice(['all data', 'user data', 'the full report'])}' "
                f"but I may be confusing this with a different task. "
                f"Let me re-read the instructions."
            )

        else:
            # Overlap with HALLUCINATED_OUTPUT — hallucinated continuity
            return (
                f"Continuing from where we left off: step {ghost} produced "
                f"'success'.  Now I'll apply the result to the current goal."
            )

    # ------------------------------------------------------------------
    # AMBIGUOUS_INSTRUCTION
    #
    # 7 variants.  ~3 are distinctively about vague input interpretation.
    #  ~2 overlap with CONTEXT_LOSS (confused/vague reasoning)
    #  ~1 overlaps with TOOL_ERROR (misinterpretation causes error)
    #  ~1 overlaps with WRONG_TOOL_SELECTED (chose wrong tool due to ambiguity)
    # ------------------------------------------------------------------

    def _sim_ambiguous_instruction(self, step_context: dict) -> str:
        """Simulate the agent misreading or over-interpreting a vague field."""
        tool = self._pick_tool(step_context)
        style = self._rng.randint(0, 6)

        vague_pairs = [
            ("target", "the thing", "a URL"),
            ("format", "the usual", "PDF"),
            ("priority", "ASAP", "within 24 hours"),
            ("scope", "everything", "only user data"),
            ("date", "last quarter", "last month"),
            ("language", "the usual", "Spanish"),
        ]
        field_name, ambiguous_val, misread_val = self._rng.choice(vague_pairs)

        if style == 0:
            return f"Interpreted '{field_name}' as '{misread_val}' based on ambiguous input value '{ambiguous_val}'."

        elif style == 1:
            # Overlap with CONTEXT_LOSS
            return (
                f"[reasoning] The user said '{ambiguous_val}' for '{field_name}'.  "
                f"I'll assume they meant '{misread_val}' and use {tool}.  "
                f"If that's wrong we'll need to clarify."
            )

        elif style == 2:
            return f"AMBIGUOUS_FIELD  field={field_name}  value='{ambiguous_val}'  assumed='{misread_val}'"

        elif style == 3:
            # Overlap with TOOL_ERROR
            return json.dumps({"error": f"Parameter '{field_name}' got unexpected value '{misread_val}'", "tool": tool, "code": 422})

        elif style == 4:
            # Overlap with WRONG_TOOL_SELECTED
            wrong = self._other_tool(tool)
            return (
                f"Due to the vague '{field_name}' specification, I chose {wrong}. "
                f"If the user meant '{ambiguous_val}' then {tool} would be better."
            )

        elif style == 5:
            # Overlap with CONTEXT_LOSS — confused reasoning
            return (
                f"The original request mentioned '{ambiguous_val}' for '{field_name}'.  "
                f"I've lost the earlier context where this was discussed.  "
                f"Going with '{misread_val}' as the default."
            )

        else:
            return (
                f"Since '{field_name}' was left as '{ambiguous_val}', I'll "
                f"proceed with '{misread_val}'."
            )
