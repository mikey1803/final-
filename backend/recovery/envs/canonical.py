"""Single source of truth for the canonical recovery environment.

This module is pure data — zero functions, zero classes. All experiments
and analytic tools in this task must import ``TYPE_NAMES``,
``STRATEGY_NAMES``, ``MATRIX``, and ``BLIND_RULE`` from here rather than
re-declaring them, so there is exactly one place the ground-truth numbers
can be edited.

These values mirror ``backend/src/recovery/strategies.py`` (``SUCCESS_MATRIX``)
and ``backend/src/evaluation/baselines.py`` (``_FIXED_RULES_V2_BLIND``) —
see ``docs/repo_inventory.md`` item 2 for the cross-check. This module does
not import from either; it is an independent restatement for the analytic
layer to validate the simulator against.
"""

import numpy as np

TYPE_NAMES: list[str] = [
    "tool_timeout",
    "tool_error",
    "wrong_tool_selected",
    "hallucinated_output",
    "malformed_output",
    "context_loss",
    "ambiguous_instruction",
]

STRATEGY_NAMES: list[str] = [
    "retry",
    "replan",
    "switch_tool",
    "re_ground",
    "verify",
    "clarify",
]

# MATRIX[t, k] = P(success | true type t, strategy k). Rows follow
# TYPE_NAMES order, columns follow STRATEGY_NAMES order.
MATRIX: np.ndarray = np.array(
    [
        [0.70, 0.50, 0.60, 0.15, 0.30, 0.10],
        [0.65, 0.55, 0.75, 0.20, 0.35, 0.25],
        [0.15, 0.60, 0.75, 0.45, 0.40, 0.55],
        [0.10, 0.45, 0.30, 0.55, 0.75, 0.50],
        [0.40, 0.50, 0.60, 0.35, 0.75, 0.30],
        [0.10, 0.55, 0.25, 0.80, 0.40, 0.50],
        [0.10, 0.45, 0.20, 0.35, 0.30, 0.80],
    ],
    dtype=np.float64,
)

# BLIND_RULE[t] = strategy index played for true type t, built by name
# lookup into STRATEGY_NAMES (never hardcoded ints).
_BLIND_RULE_BY_NAME: list[str] = [
    "retry",       # tool_timeout
    "switch_tool", # tool_error
    "replan",      # wrong_tool_selected
    "verify",      # hallucinated_output
    "verify",      # malformed_output
    "re_ground",   # context_loss
    "clarify",     # ambiguous_instruction
]

BLIND_RULE: np.ndarray = np.array(
    [STRATEGY_NAMES.index(name) for name in _BLIND_RULE_BY_NAME],
    dtype=np.int64,
)
