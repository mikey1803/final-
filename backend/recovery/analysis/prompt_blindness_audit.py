"""Task 15: blindness audit for LLM-designer prompt TEXT (paraphrases and
presentation-order-randomised variants), reusing the exact same
value/key-pattern check ``tests/test_study.py``'s
``test_blindness_no_endpoint_leaks_matrix_values_or_reward_shaped_keys``
applies to the study web app's HTTP responses -- see
``study/blindness_audit.md`` for the full protocol this check enforces.

This module is free to import ``recovery.envs.canonical.MATRIX`` (unlike
``recovery/envs/descriptions.py`` and ``recovery/envs/paraphrases.py``,
which must never import it): it is an OFFLINE audit tool, not part of any
request-serving or prompt-building path, exactly the same trust boundary
``recovery/analysis/human_study.py`` already occupies for offline scoring.
"""

import re

import numpy as np

from recovery.envs.canonical import MATRIX

__all__ = ["audit_prompt_text"]

_FORBIDDEN_KEY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [r"\breward\b", r"\bsuccess_rate\b", r"\bmatrix\b", r"\bscore\b", r"\bp_success\b"]
]


def audit_prompt_text(text: str) -> list[str]:
    """Return a list of blindness violations found in ``text`` (empty list
    = passes). Checks, identically to the study backend's blindness test:

    1. No substring matching any ``MATRIX`` value to 2 decimal places.
    2. No forbidden reward-shaped key/word (``reward``, ``success_rate``,
       ``matrix``, ``score``, ``p_success``, case-insensitive).
    """
    violations = []
    forbidden_values = {f"{v:.2f}" for v in np.asarray(MATRIX).flatten()}
    for forbidden in forbidden_values:
        if forbidden in text:
            violations.append(f"leaked a MATRIX value: {forbidden!r}")
    for pattern in _FORBIDDEN_KEY_PATTERNS:
        if pattern.search(text):
            violations.append(f"contains forbidden key pattern: {pattern.pattern!r}")
    return violations
