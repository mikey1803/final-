"""A1: v1/v2 behaviour is unchanged by v3.

``fixtures/v1v2_golden.json`` was recorded from the code at commit 4723b2b,
before any v3 change: the v1 ``FlakyWrapper``'s injection decisions for three
seeds x 300 calls, and the v2 inline selection draws (arm, propensity, derived
seed, retry jitter) for three session seeds x 100 failures. Both must replay
bit-for-bit.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recovery.live.tools as tools  # noqa: E402
from recovery.live.event_schema import AgentState, FailureEvent  # noqa: E402
from recovery.live.inline_recovery import InlineRecovery  # noqa: E402
from recovery.live.recovery_attempt import RecoveryStore  # noqa: E402
from recovery.live.tools import FlakyWrapper, ToolCallResult  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "v1v2_golden.json").read_text(encoding="utf-8"))


def test_v1_flaky_wrapper_replays_bit_for_bit(monkeypatch):
    seen = []

    def make(name):
        def fn(args, malformed, timeout):
            seen.append((malformed, timeout))
            return ToolCallResult(True, 200, "{}", None, 1.0, full_text="{}")
        return fn
    order = list(tools.TOOLS)
    monkeypatch.setattr(tools, "TOOLS", {k: make(k) for k in order})
    for seed, expected in GOLDEN["flaky_wrapper_v1"].items():
        seen.clear()
        w = FlakyWrapper(0.20, rng=random.Random(int(seed)))
        got = []
        for i in range(300):
            r = w.call(order[i % 6], {})
            assert r.v3 is None
            got.append([seen[-1][0], seen[-1][1], r.injected, r.injected_kind])
        assert got == expected, f"seed {seed}"


def test_v2_inline_selection_replays_bit_for_bit(tmp_path):
    for seed, gold in GOLDEN["inline_selection_v2"].items():
        rec = InlineRecovery(store=RecoveryStore(tmp_path), session_seed=int(seed))
        assert rec.arms == gold["arms"]
        for i, (derived, arm, prop, delays) in enumerate(gold["draws"]):
            seq = np.random.SeedSequence(int(seed), spawn_key=(i,))
            d = int(seq.generate_state(1, dtype=np.uint64)[0])
            assert d == derived
            assert list(rec.policy.choose(rec.arms, np.random.default_rng(d))) == [arm, prop]
            jit = random.Random(d)
            assert [round(jit.uniform(0.0, min(4.0, 0.5 * 2 ** k)), 6) for k in range(3)] == delays


def test_v2_event_serialisation_has_no_v3_key():
    e = FailureEvent(session_id="s", step_index=0, tool_name="wikipedia", tool_args={}, raw_error="x",
                     agent_state=AgentState(goal="g"), captured_at="t")
    d = e.to_dict()
    assert "v3" not in d
    assert FailureEvent.from_dict(d).event_id == e.event_id
    fixture = json.loads((Path(__file__).parent / "fixtures" / "live_event_pre_task21.json").read_text())
    assert FailureEvent.from_dict(fixture).event_id == fixture["event_id"]
