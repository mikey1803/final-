"""Live pilot v3 (docs/live_pilot/strategy_definitions_v3.md, FROZEN v3;
docs/live_pilot/v3_preregistration.md).

v3 adds injection persistence modes, a read-only resend baseline, fault
origins, agent-side selection failures, ``continue_after_recovery`` with
``task_completed``, and the instrument-side ``corrected`` check. Everything
lives in this package; the v1/v2 paths (``recovery.live.tools.FlakyWrapper``,
``recovery.live.agent_loop``, ``recovery.live.inline_recovery``) are unchanged,
so v1/v2 runs reproduce exactly. v3 data is never merged with v1/v2 data.
"""

DEFINITION_VERSION = "v3"
DEFINITIONS_DOC = "docs/live_pilot/strategy_definitions_v3.md"
DEFINITIONS_COMMIT = "c1705bf"
PREREGISTRATION_DOC = "docs/live_pilot/v3_preregistration.md"
PREREGISTRATION_COMMIT = "4723b2b"
