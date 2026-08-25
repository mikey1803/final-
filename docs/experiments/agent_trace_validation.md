# Real Agent Trace Validation (Task 20) — SCOPING ONLY, not implemented

**Status of this document: read-only planning.** No code was written, no
LLM calls were made, no traces were collected. This exists to record the
trace-source decision, the tool/taxonomy mapping, the labelling protocol,
and a spend estimate — so that when Task 20 is actually picked up, the
design questions are already resolved rather than re-litigated. Per
explicit instruction, this document does not touch the running Task 19A
sweep or any of its code paths, and proposes no implementation.

## 1. Why this task exists

Every result in this project to date is on synthetic reward matrices
against a seeded RNG kernel (`docs/PAPER_PREP.md`'s threats-to-validity
list, "Simulation-only"). The 7-type failure taxonomy
(`src/failures/taxonomy.py`) and the hand-authored `SUCCESS_MATRIX` are
both design artifacts, not measurements — nothing in the project so far
establishes that real agent tool-use failures actually distribute across
these 7 types the way the taxonomy assumes, or that a real classifier's
confusion structure resembles what `compute_empirical_confusion_matrix`
measures against synthetically-injected error text
(`src.failures.injector.FailureInjector`). Task 20 closes part of that gap:
collect real LangGraph agent execution traces, observe real tool-use
failures, and check whether the taxonomy/classifier apparatus this
project's entire simulation rests on holds up against them.

## 2. The trace-source decision

**Decision: an instrumented LangGraph agent running real tasks against a
mix of real and deliberately fragile tools — not synthetic failure
injection, and not retrofitting a public benchmark's own trace/label
format onto this taxonomy.**

Two alternatives were considered and rejected, for stated reasons:

- **Pure public benchmark traces** (e.g. existing tool-use/agent-
  benchmark trace datasets) were rejected as the PRIMARY source. Their
  own failure categorizations, where they exist at all, were not designed
  around this taxonomy's 7 hand-authored types — mapping them onto
  `FailureType` would require essentially the same manual/LLM relabelling
  effort this document already scopes for real traces (§4), while giving
  up control over label balance. Real-world tool failures in an arbitrary
  public dataset are very likely skewed toward `TOOL_ERROR`/`TOOL_TIMEOUT`
  and may contain few or zero naturally-occurring examples of
  `CONTEXT_LOSS` or `AMBIGUOUS_INSTRUCTION` specifically, since those two
  types describe agent-side reasoning failures rather than tool-side
  faults, and this taxonomy's granularity is this project's own
  invention, not derived from any existing dataset.
- **Synthetic failure injection into the agent loop** (mirroring how the
  simulator's `FailureInjector` already works) was rejected as
  insufficiently different from what the project already has — it would
  add a real LLM in the loop but keep failures scripted, which does not
  test the thing Task 20 exists to test (whether the taxonomy and
  classifier hold up against failures that were not designed to fit
  them).

**What "instrumented" means here**: build out the currently-stub
`src/agent/graph.py` (a one-line placeholder comment today: *"LangGraph
state machine defining the Plan -> Act -> Observe -> Classify -> Recover
loop"* — not a small tweak, this is close to new construction) into a
working Plan→Act→Observe→Classify→Recover loop using the existing
`src/agent/llm_client.py` (`parse_tool_request`,
`classify_recovery_strategy`, both already wired to Groq) as the
tool-selection and classification steps. The agent runs REAL tasks
against REAL tool implementations (§3); failures are allowed to occur
NATURALLY from real tool behavior (network latency, rate limits, a
genuinely malformed API response, a genuinely ambiguous task prompt) —
task and tool DESIGN is chosen to make each of the 7 types plausible, but
no failure is scripted or forced to happen. This preserves ecological
validity (these are real failures, not injected ones) while still giving
enough control to avoid the "may never see 5 of 7 types" problem a
found/public dataset would carry.

## 3. Tool set and taxonomy mapping

Reuses the 6-tool vocabulary already named in this codebase
(`recovery/classify/adapter.py::_KNOWN_TOOLS`) rather than inventing a new
one, so the real-trace classifier stays comparable to the existing
synthetic-classifier setup.

| Tool | Real implementation | Taxonomy type(s) it's expected to surface | How |
|---|---|---|---|
| `web_search` | Real search API call (rate-limited free tier) | `TOOL_TIMEOUT`, `HALLUCINATED_OUTPUT` | Real network latency/rate-limit errors; LLM summarizing a search result in a way not actually supported by the returned snippet |
| `calculator` | Real (correct) arithmetic evaluator | `WRONG_TOOL_SELECTED` | Deliberately ambiguous math-or-code tasks where `code_executor` is also plausible — the failure is in tool SELECTION, not the calculator itself |
| `file_reader` | Reads real local files, some in unexpected encodings/formats | `MALFORMED_OUTPUT`, `CONTEXT_LOSS` | A file whose content doesn't match the format the agent's prompt assumed; a later step referencing content from a much earlier read |
| `api_caller` | Real HTTP call to a free public API (e.g. weather/reference API) | `TOOL_TIMEOUT`, `TOOL_ERROR`, `MALFORMED_OUTPUT` | Real timeouts/5xx errors; a real API response shape the agent's parser doesn't expect |
| `database_query` | Real local SQLite database with a deliberately partial schema | `TOOL_ERROR` | Real malformed-query/constraint errors from schema mismatches the agent doesn't know about in advance |
| `code_executor` | Real sandboxed Python `exec` against agent-authored code | `TOOL_ERROR`, `HALLUCINATED_OUTPUT` | Real runtime exceptions; code that runs without error but silently produces a wrong-but-plausible result |

Two taxonomy types are intentionally NOT tied to a specific tool, because
they are properties of the TASK/INSTRUCTION design, not of any one tool's
behavior:

- **`AMBIGUOUS_INSTRUCTION`** — surfaced by a subset of task prompts
  written to be genuinely underspecified (multiple valid tool choices or
  interpretations), independent of which tool is eventually called.
- **`CONTEXT_LOSS`** — surfaced by multi-step tasks specifically designed
  to require information from an early step at a much later one, again
  independent of any single tool.

This mapping is a design target for task/tool authoring, not a promise —
real tool-use failures do not sort themselves cleanly, and some observed
failures will not fit their "intended" cell (itself a finding worth
reporting, consistent with this project's practice of reporting
mechanism-level surprises rather than smoothing them into the nearest
expected bucket).

## 4. Labelling protocol

Two tiers, by how the label is obtained — never one tier presented as if
it were the other:

**Tier A — structural/mechanical labels (no rater needed, ground truth is
exact).** Directly observable from the trace object, not inferred:
`TOOL_TIMEOUT` (the tool call raised a timeout exception),
`TOOL_ERROR` (the tool call raised any other exception),
`MALFORMED_OUTPUT` (the agent's own output-schema validation failed).
These three types get their label mechanically at collection time.

**Tier B — semantic labels (requires a rater, blind to any prior label).**
`WRONG_TOOL_SELECTED`, `HALLUCINATED_OUTPUT`, `CONTEXT_LOSS`,
`AMBIGUOUS_INSTRUCTION` cannot be determined mechanically from a raw
exception or a schema check — they require reading the trace and judging
intent/correctness. Protocol, modeled on this project's existing
blind-design-study discipline (`study/protocol.md`,
`docs/experiments/human_study.md`):

- **Two independent raters per Tier-B trace** (LLM-based initially, given
  no institutional human-study approval is in place for this task
  either — same constraint Task 11's human arm hit), each shown ONLY the
  trace (task prompt, tool calls, tool outputs, final agent output) and
  the 7 taxonomy descriptions, blind to the other rater's label and to
  any Tier-A label already assigned to that trace.
- **Agreement measured, not assumed**: Cohen's kappa between the two
  raters, computed and reported before any trace's label is treated as
  final — consistent with this project's standing discipline of never
  fabricating agreement or treating a single unverified label as ground
  truth (parallel to Task 13's statistical audit and Task 11's blindness
  audit).
- **Disagreements**: adjudicated by a third independent rater, not by
  either of the first two, and the disagreement RATE itself is reported
  (a high Tier-B disagreement rate would itself be a finding about
  taxonomy boundary clarity, not swept under a majority vote).
- **Pre-registered before collection, not after**: the exact rater
  prompt, the taxonomy description text shown, and the agreement
  threshold below which a batch is flagged for taxonomy-definition review
  (not silently accepted) would need to be frozen in a
  `study/`-style preregistration document before any real trace is
  labelled, mirroring the discipline already applied to Task 19's kappa
  protocol and Task 11's human study.

**Target sample size**: minimum 30 traces per taxonomy type (this
project's own established `n_seeds≥30` convention, Task 13's statistical
audit) = **210 traces minimum**, with a stretch target of 50/type (350
total) if labelling throughput allows. The binding constraint on scale is
rater/engineering time, not LLM spend (§5).

## 5. Total spend estimate

Grounded in this project's own already-measured pricing
(`results/tables/llm_cost_accounting_summary.parquet`, Task 14 Part 2):
Groq's `llama-3.1-8b-instant` (already the model `src/agent/llm_client.py`
is wired to) cost **$0.000682 across 21 samples / 12,742 tokens** in the
blind-design study = **~$5.35 × 10⁻⁸ per token** (~$0.0535 per million
tokens).

A single agent episode (Plan → Act → Observe → Classify → Recover) is
estimated at ~4 LLM calls (plan, tool-selection via
`parse_tool_request`, an observe/classify step, a recovery-strategy
selection). Using ~600 tokens/call as a working estimate (the blind-design
study's own per-sample average was ~535 prompt + ~72 completion tokens for
a single call with a full taxonomy description in context; agent calls
additionally carry tool descriptions, so rounded up rather than down) →
~2,400 tokens/episode → **~$0.000128/episode**.

| Scale | Episodes | Estimated LLM spend | Rater cost (LLM-based Tier B, 2 raters × ~70% of traces need Tier B) |
|---|---:|---:|---|
| Minimum (30/type) | 210 | **~$0.027** | ~294 extra rating calls, same per-call cost order — negligible, well under $1 total combined |
| Stretch (50/type) | 350 | **~$0.045** | ~490 extra rating calls — still well under $1 total combined |

**LLM dollar spend is not the constraint at any scale considered here** —
consistent with Task 14's own finding that prior study phases' real spend
was effectively ~$0 on free/dev-tier keys. The actual cost driver is
engineering time to build the currently-stub agent loop and the fragile
tool implementations (§2–3), and rater/labelling throughput for Tier B
(§4) — neither of which this document estimates in dollars, since neither
is a paid-API cost.

## 6. What is explicitly out of scope for this document

No code, no traces, no labels, no preregistration file. This document
does not commit to a start date, and revising any decision above (trace
source, tool set, sample size) before implementation begins is expected
and fine — it is a proposal to react to, not a frozen protocol yet
(unlike §4's Tier-B protocol, which — once implementation actually
starts — WOULD need to be frozen before the first real trace is labelled,
per this project's standing discipline).
