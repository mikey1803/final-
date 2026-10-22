"""The v3 episode loop with fake services, a fake LLM and a scripted selector:
fault-origin tagging, the resend baseline, halt H1 (A9), continue_after_recovery
and task_completed (A5), agent-side events, the genuine value never reaching
the agent or an LLM (A4), and the instrument-fault exclusion (A7)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import recovery.live.v3.flaky as flaky_mod  # noqa: E402
from recovery.live.event_schema import Task  # noqa: E402
from recovery.live.event_store import EventStore  # noqa: E402
from recovery.live.executors.base import LLMError  # noqa: E402
from recovery.live.recovery_attempt import RecoveryAttempt, RecoveryStore  # noqa: E402
from recovery.live.v3 import report  # noqa: E402
from recovery.live.v3.degrade import DegradeSpec  # noqa: E402
from recovery.live.v3.flaky import FlakyWrapperV3  # noqa: E402
from recovery.live.v3.loop import BlockSpec, HaltH1, V3Dirs, run_episode_v3  # noqa: E402
from recovery.live.v3.recovery import InlineRecoveryV3  # noqa: E402
from tests._v3_fakes import FakeServices, RecordingLLM, ScriptedSelector, tiny_libraries  # noqa: E402

COUNTRY = Task(task_id="country_basic_01", goal="Find the capital city of India using the countriesnow tool.",
               category="tool_call_basic", clarify_canned_answer="India, the country.")
TWO_STEP = Task(task_id="long_context_01", goal="capital then weather", category="long_context",
                steps=["Find the capital city of India using the countriesnow tool.",
                       "Get the current weather for that city using the wttr tool."])
OUT_OF_SCOPE = Task(task_id="out_of_scope_01", goal="Book a flight.", category="out_of_scope")
PICK_COUNTRY = {"tool": "countriesnow", "args": {"country": "India"}}
PICK_WTTR = {"tool": "wttr", "args": {"location": "Genuinopolis"}}


class FakeRng:
    """Stands in for a numpy Generator's ``random()``: scripted values, then 0.99."""

    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0) if self.values else 0.99


class Rig:
    def __init__(self, tmp_path, mode, *, n=None, svc=None, selector=None, llm=None, arm=None, rate=0.2,
                 seed=231000):
        self.svc = svc or FakeServices()
        self.wrapper = FlakyWrapperV3(mode=mode, session_seed=seed, block_index=0, rate=rate, outage_N=n,
                                      libraries=tiny_libraries(), tools=self.svc.tools())
        self.events, self.recs = EventStore(tmp_path), RecoveryStore(tmp_path)
        self.llm = llm or RecordingLLM()
        self.recovery = InlineRecoveryV3(store=self.recs, session_seed=seed, llm=self.llm, sleep=lambda s: None)
        if arm is not None:
            self.recovery.arms = [arm]          # force the drawn strategy
        self.selector = selector or ScriptedSelector([PICK_COUNTRY])
        self.block = BlockSpec(0, mode, n, 1)
        self.dirs = V3Dirs(tmp_path)
        self.seed = seed
        self.i = 0

    def run(self, task):
        rec = run_episode_v3(task, wrapper=self.wrapper, events=self.events, recoveries=self.recs,
                             recovery=self.recovery, select=self.selector, block=self.block,
                             session={"session_seed": self.seed, "session_index": 0}, episode_index=self.i,
                             tool_health=None, dirs=self.dirs)
        self.i += 1
        return rec

    def attempts(self):
        return list(self.recs.iter_attempts())

    def event(self):
        (e,) = list(self.events.iter_events())
        return e


def test_injected_failure_tagged_resend_recorded_and_attempt_fields(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="switch_tool")
    rig.wrapper._rng["injection"] = FakeRng([0.0, 0.0])        # India corrupted; the wikipedia peer key too
    ep = rig.run(COUNTRY)
    e = rig.event()
    assert e.v3["fault_origin"] == "injected" and e.v3["injection_mode"] == "sticky_args"
    assert e.v3["block_label"] == "S" and e.v3["failure_ordinal"] == 1 and e.v3["call_meta"]["fresh_draw"]
    (a,) = rig.attempts()
    assert a.definition_version == "v3" and a.strategy == "switch_tool"
    v = a.v3
    assert v["resend_baseline"]["outcome"] is False and v["resend_baseline"]["sticky_hit"] is True
    assert v["applicable"] and v["fresh"] is True and v["primary_outcome"] is False
    assert v["task_completed"] is False and v["task_completed_applicable"] is True
    assert v["continue_after_recovery"] is True and v["prior_attempt_ids"] == []
    assert not v["instrument_fault"]
    row = json.loads((tmp_path / "v3_resends" / f"{e.event_id}.json").read_text())
    assert row["strategy_drawn"] == "switch_tool" and row["outcome"] is False
    assert ep["failures"][0]["attempt_id"] == a.attempt_id and not ep["task_completed"]


def test_natural_failure_is_tagged_and_exempt_from_h1(tmp_path):
    rig = Rig(tmp_path, "sticky_args", svc=FakeServices(natural_fail={"countriesnow": 1}), arm="retry", rate=0.0)
    rig.run(COUNTRY)
    e = rig.event()
    assert e.v3["fault_origin"] == "natural" and not e.injected
    (a,) = rig.attempts()
    assert a.v3["resend_baseline"]["outcome"] is True and a.v3["h1_exempt_natural"] is True


class MalformedOnceServices(FakeServices):
    """An instrument fault: a malformed request that the 'server' accepts the second time."""

    def _make(self, tool):
        inner = super()._make(tool)
        seen = {"n": 0}

        def fn(args, malformed, timeout):
            if malformed:
                seen["n"] += 1
                self.accept_malformed = seen["n"] > 1
            return inner(args, malformed, timeout)
        return fn


def test_h1_halts_on_an_impossible_resend_success(tmp_path):
    rig = Rig(tmp_path, "sticky_args", svc=MalformedOnceServices(), arm="retry")
    rig.wrapper._rng["injection"] = FakeRng([0.0])
    with pytest.raises(HaltH1):
        rig.run(COUNTRY)
    e = rig.event()
    assert e.v3["fault_origin"] == "injected"
    assert json.loads((tmp_path / "v3_resends" / f"{e.event_id}.json").read_text())["outcome"] is True
    assert rig.attempts() == []                                   # the strategy never ran


def test_h1_does_not_apply_to_transient(tmp_path):
    rig = Rig(tmp_path, "transient", arm="retry")
    rig.wrapper._rng["injection"] = FakeRng([0.0, 0.9])           # malformed request, then the probe's own stream
    rig.run(COUNTRY)                                               # resend may succeed: no halt in T


def test_continue_after_recovery_runs_the_next_step_and_completes(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="switch_tool", selector=ScriptedSelector([PICK_COUNTRY, PICK_WTTR]))
    rig.wrapper._rng["injection"] = FakeRng([0.0])               # only the first key is corrupted
    ep = rig.run(TWO_STEP)
    assert ep["finished_all_steps"] and ep["task_completed"]
    (a,) = rig.attempts()
    assert a.v3["primary_outcome"] and a.v3["task_completed"] is True
    assert [c["tool"] for c in a.result["evidence"]["calls"]] == ["wikipedia"]
    assert rig.selector.goals == TWO_STEP.steps                   # step 2 was reached


def test_unrecovered_failure_stops_the_task(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="retry", selector=ScriptedSelector([PICK_COUNTRY, PICK_WTTR]))
    rig.wrapper._rng["injection"] = FakeRng([0.0])
    ep = rig.run(TWO_STEP)
    assert not ep["finished_all_steps"] and not ep["task_completed"]
    assert rig.selector.goals == TWO_STEP.steps[:1]               # step 2 never ran


# ---------------------------------------------------------------- degraded_output

def _degrade_first_key(monkeypatch, spec):
    """The first drawn key gets ``spec``; every later key is clean."""
    state = {"n": 0}

    def fake_draw(rng, rate):
        state["n"] += 1
        return spec if state["n"] == 1 else DegradeSpec(False)
    monkeypatch.setattr(flaky_mod, "draw_spec", fake_draw)


PLAUSIBLE = DegradeSpec(True, "plausible_wrong", "plausible_wrong", k=1, delta=10.0)


def test_degraded_failure_surfaced_by_oracle_switch_tool_corrects(tmp_path, monkeypatch):
    _degrade_first_key(monkeypatch, PLAUSIBLE)
    rig = Rig(tmp_path, "degraded_output", arm="switch_tool")
    ep = rig.run(COUNTRY)
    e = rig.event()
    assert e.v3["detected_by"] == "oracle" and e.v3["fault_origin"] == "injected"
    assert e.v3["call_meta"]["synthetic_content"] and e.http_status == 200
    (a,) = rig.attempts()
    v = a.v3
    assert v["resend_baseline"]["outcome"] is False and v["resend_baseline"]["degraded"] is True
    assert v["corrected"] is True and v["corrected_reference"] == "same_call" and v["primary_outcome"] is True
    assert v["detected"] is False                                  # switch_tool has no content check
    assert ep["task_completed"] is True                            # the last answer is the genuine wikipedia one


def test_verify_detects_but_cannot_correct_a_sticky_degradation(tmp_path, monkeypatch):
    _degrade_first_key(monkeypatch, DegradeSpec(True, "structural", "remove_field"))
    rig = Rig(tmp_path, "degraded_output", arm="verify")
    rig.run(COUNTRY)
    (a,) = rig.attempts()
    assert a.v3["detected"] is True and a.v3["corrected"] is False and a.v3["primary_outcome"] is False
    assert a.v3["task_completed"] is False


def test_verify_accepting_a_plausible_wrong_value_is_not_corrected(tmp_path, monkeypatch):
    _degrade_first_key(monkeypatch, PLAUSIBLE)
    rig = Rig(tmp_path, "degraded_output", arm="verify", llm=RecordingLLM(verdict=True))
    rig.run(COUNTRY)
    (a,) = rig.attempts()
    assert a.no_action_taken and a.result["success"] is True
    assert a.v3["detected"] is False and a.v3["corrected"] is False and a.v3["final_call_id"] == a.v3["call_id"]


@pytest.mark.parametrize("arm", ["retry", "switch_tool", "verify", "re_ground", "replan", "clarify"])
def test_genuine_value_never_reaches_the_agent_or_an_llm(tmp_path, monkeypatch, arm):
    """A4: the degraded call's genuine capital (Genuinopolis) must not appear in
    any prompt, any selector input, or any record other than the injection log."""
    _degrade_first_key(monkeypatch, PLAUSIBLE)
    llm = RecordingLLM(call=PICK_COUNTRY, verdict=False)
    rig = Rig(tmp_path, "degraded_output", arm=arm, llm=llm)
    rig.run(COUNTRY)
    assert all("Genuinopolis" not in p for p in llm.prompts)
    assert all("Genuinopolis" not in g for g in rig.selector.goals)
    for sub in ("events", "recoveries", "v3_resends", "v3_episodes"):
        for f in (tmp_path / sub).glob("*.json"):
            assert "Genuinopolis" not in f.read_text(encoding="utf-8"), f"{sub}/{f.name}"
    assert any(e.get("genuine_raw") == "Genuinopolis" for e in rig.wrapper.log)   # it IS in the log


# ---------------------------------------------------------------- agent-side failures

def test_malformed_selection_is_an_agent_side_event_and_retry_recovers(tmp_path):
    sel = ScriptedSelector([ValueError("LLM did not return valid JSON: oops"), PICK_COUNTRY])
    rig = Rig(tmp_path, "transient", arm="retry", selector=sel, rate=0.0)
    ep = rig.run(COUNTRY)
    e = rig.event()
    assert e.tool_name == "<llm_tool_selection>" and e.v3["fault_origin"] == "agent_side"
    assert e.v3["agent_side_kind"] == "malformed_output" and e.true_type is None
    (a,) = rig.attempts()
    assert a.v3["resend_baseline"]["kind"] == "selection_rerun" and a.v3["resend_baseline"]["outcome"] is True
    assert a.result["success"] and a.v3["fresh"] is None and ep["task_completed"]
    assert a.executor_config["on"] == "selection_failure"


def test_wrong_tool_selection_switch_tool_is_untranslatable(tmp_path):
    sel = ScriptedSelector([{"tool": "google", "args": {"q": "capital of India"}}])
    rig = Rig(tmp_path, "transient", arm="switch_tool", selector=sel, rate=0.0)
    rig.run(COUNTRY)
    e = rig.event()
    assert e.v3["agent_side_kind"] == "wrong_tool_selected"
    (a,) = rig.attempts()
    assert a.result["evidence"]["reason"] == "untranslatable_args" and not a.v3["applicable"]


def test_wrong_arg_names_are_wrong_tool_selected(tmp_path):
    sel = ScriptedSelector([{"tool": "countriesnow", "args": {"name": "India"}}, PICK_COUNTRY])
    rig = Rig(tmp_path, "transient", arm="replan", selector=sel, rate=0.0, llm=RecordingLLM(call=PICK_COUNTRY))
    rig.run(COUNTRY)
    assert rig.event().v3["agent_side_kind"] == "wrong_tool_selected"
    (a,) = rig.attempts()
    assert a.result["success"] and "previous answer was unusable" in rig.llm.prompts[0]


# ---------------------------------------------------------------- instrument faults (A7), records

class FailingLLM(RecordingLLM):
    def complete(self, prompt, max_tokens, timeout_s):
        raise LLMError("HTTP 503: provider down")


def test_llm_error_attempts_are_excluded_from_rates_and_counted(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="replan", llm=FailingLLM())
    rig.wrapper._rng["injection"] = FakeRng([0.0])
    rig.run(COUNTRY)
    (a,) = rig.attempts()
    assert a.result["evidence"]["reason"] == "llm_error" and a.v3["instrument_fault"] is True
    df = report.load_attempts_frame(tmp_path)
    assert len(df) == 1 and report.included(df).empty
    assert int(report.exclusion_counts(df).loc["replan", "llm_error"]) == 1
    cells = report.cell_table(df, n_resamples=200)
    assert int(cells["n"].sum()) == 0


def test_out_of_scope_task_completed_not_applicable(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="switch_tool", selector=ScriptedSelector([PICK_COUNTRY]))
    rig.wrapper._rng["injection"] = FakeRng([0.0])
    rig.run(OUT_OF_SCOPE)
    (a,) = rig.attempts()
    assert a.v3["task_completed_applicable"] is False


def test_v3_attempt_roundtrip_and_v2_records_have_no_v3_key(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="retry")
    rig.wrapper._rng["injection"] = FakeRng([0.0])
    rig.run(COUNTRY)
    (a,) = rig.attempts()
    assert RecoveryAttempt.from_dict(a.to_dict()).attempt_id == a.attempt_id
    v2 = RecoveryAttempt(event_id="e", session_id="s", task_id="t", strategy="retry", definition_version="v2",
                         selection_policy="uniform", selection_propensity=1 / 6, seed={}, model_id=None,
                         executor_config={}, budget_limits={}, budget_used={}, budget_exceeded=False,
                         result={"success": True}, started_at="x", finished_at="y")
    assert "v3" not in v2.to_dict()
    with pytest.raises(ValueError):
        RecoveryAttempt(**{**{k: v for k, v in v2.to_dict().items() if k != "attempt_id"}, "definition_version": "v3"})
    assert rig.event().to_dict()["v3"]["fault_origin"] == "injected"


def test_cell_table_paired_deltas_and_bootstrap_are_reproducible(tmp_path):
    rig = Rig(tmp_path, "sticky_args", arm="switch_tool")
    for _ in range(4):
        rig.wrapper._rng["injection"] = FakeRng([0.0])
        rig.run(COUNTRY)
    df = report.load_attempts_frame(tmp_path)
    t1 = report.cell_table(df, n_resamples=500)
    t2 = report.cell_table(df, n_resamples=500)
    import pandas as pd
    pd.testing.assert_frame_equal(t1, t2)
    row = t1[(t1["block"] == "S") & (t1["strategy"] == "switch_tool")].iloc[0]
    assert row["n"] == 4 and row["n_episodes"] == 4 and row["rate_b"] == 0.0
    assert row["delta_itt"] == row["rate_y"] - row["rate_b"] and row["applicability"] == 1.0
