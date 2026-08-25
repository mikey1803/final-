"""Deterministic per-token display-order randomiser.

Extracted from ``study/app/backend/main.py`` (Task 11's human instrument),
where it was originally defined as two private, module-local functions.
Moved here, byte-identical in behaviour, so a SECOND caller -- Task 15's
LLM presentation-order-randomisation supplementary experiment
(``experiments/prompt_robustness.py``) -- can reuse the EXACT SAME
randomiser rather than re-deriving its own seeding scheme. ``study/app/backend/main.py``
now imports from here instead of defining its own copy; this is a pure
refactor (identical seed formula, identical RNG calls), verified by
``tests/test_study.py``'s existing display-order tests continuing to pass
unchanged.

No import of ``recovery.envs.canonical.MATRIX``/``BLIND_RULE`` here --
this module is reachable from the study backend's request-serving path
(via ``study/app/backend/main.py``), so it is bound by the same blindness
hard rule as ``recovery/envs/descriptions.py``.
"""

import hashlib

import numpy as np

__all__ = ["seed_from_token", "display_order"]


def seed_from_token(token: str) -> int:
    """Deterministic per-token seed: the SAME token always yields the SAME
    display order. First 8 bytes of SHA-256(token), big-endian."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def display_order(token: str, n_types: int, n_strategies: int) -> tuple[list[int], list[int]]:
    """A deterministic random permutation of ``range(n_types)`` and
    ``range(n_strategies)``, seeded from ``token`` via :func:`seed_from_token`.

    ``token`` need not be a real study participant token -- Task 15 seeds
    it off an LLM sample index (e.g. ``f"llm-sample-{i:04d}"``) instead, to
    get a reproducible-but-varying presentation order per API call.
    """
    rng = np.random.default_rng(seed_from_token(token))
    type_order = rng.permutation(n_types).tolist()
    strategy_order = rng.permutation(n_strategies).tolist()
    return type_order, strategy_order
