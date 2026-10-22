"""Named, independent random streams for one v3 session (v3 §8).

Every stream is derived from the one recorded ``session_seed`` with
``numpy.random.SeedSequence(session_seed, spawn_key=(stream_index, *keys))``
-- never ``hash()``. ``stream_index`` comes from the fixed ``STREAMS`` order
below, so adding a key never shifts another stream. The derived integer seed
is returned with a description, to be recorded on the record that used it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["STREAMS", "stream_seed", "stream_rng", "session_seed_for"]

STREAMS: tuple[str, ...] = (
    "block_order",   # seeded order of mode blocks
    "selection",     # which strategy is drawn, per failure
    "jitter",        # retry backoff, per failure
    "injection",     # per-call / per-key injection (transient, sticky_args), per block
    "resend",        # the resend baseline's own transient draws, per block
    "outage",        # outage starts, per block
    "degradation",   # degraded keys, transformation, wrong-value offsets, per block
)

SESSION_SEED_BASE = 231000  # pre-registration §2: session s uses 231000 + s


def session_seed_for(session_index: int) -> int:
    if session_index < 0:
        raise ValueError("session_index must be >= 0 (0 is reserved for smoke runs, never data)")
    return SESSION_SEED_BASE + session_index


def stream_seed(session_seed: int, stream: str, *keys: int) -> tuple[int, dict]:
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; must be one of {STREAMS}")
    spawn_key = (STREAMS.index(stream), *(int(k) for k in keys))
    seq = np.random.SeedSequence(int(session_seed), spawn_key=spawn_key)
    derived = int(seq.generate_state(1, dtype=np.uint64)[0])
    return derived, {"session_seed": int(session_seed), "stream": stream, "spawn_key": list(spawn_key),
                     "derived_seed": derived,
                     "method": "numpy.SeedSequence(session_seed, spawn_key=(stream_index, *keys))"}


def stream_rng(session_seed: int, stream: str, *keys: int) -> tuple[np.random.Generator, dict]:
    derived, record = stream_seed(session_seed, stream, *keys)
    return np.random.default_rng(derived), record
