"""The v3 session plan: seeded block order, all six block types once, R = 10
passes for T and 5 otherwise (pre-reg §2), one pass each in smoke runs, and
session seeds 231000 + s with s = 0 reserved for smoke."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from experiments.run_live_v3 import block_plan  # noqa: E402
from recovery.live.v3.seeds import session_seed_for  # noqa: E402


def test_block_plan_is_seeded_complete_and_uses_registered_passes():
    blocks, rec = block_plan(231001, smoke=False)
    assert [b.block_index for b in blocks] == list(range(6))
    assert sorted(b.label for b in blocks) == sorted(["T", "S", "O2", "O5", "O20", "D"])
    assert {b.label: b.passes for b in blocks} == {"T": 10, "S": 5, "O2": 5, "O5": 5, "O20": 5, "D": 5}
    assert [b.label for b in block_plan(231001, smoke=False)[0]] == [b.label for b in blocks]
    assert rec["stream"] == "block_order"
    orders = {tuple(b.label for b in block_plan(session_seed_for(s), False)[0]) for s in range(1, 30)}
    assert len(orders) > 1


def test_smoke_plan_is_one_pass_per_block():
    blocks, _ = block_plan(session_seed_for(0), smoke=True)
    assert all(b.passes == 1 for b in blocks) and len(blocks) == 6


def test_session_seeds():
    assert session_seed_for(0) == 231000 and session_seed_for(7) == 231007
    with pytest.raises(ValueError):
        session_seed_for(-1)
