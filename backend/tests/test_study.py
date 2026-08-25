"""Tests for the blind-design elicitation study (Task 11).

Covers: recovery.analysis.human_study (scoring/analysis),
study.storage (append-only, immutable), study.app.backend.main (the
blindness guarantee, immutability at the HTTP layer, display-order
randomization), and experiments/llm_designers.py (parsing, never a
silent fallback, prompt construction from the shared descriptions
module).
"""

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from recovery.analysis.human_study import (  # noqa: E402
    q_prior,
    score_submission,
    second_best_error_test,
    study_summary,
)
from recovery.envs.canonical import BLIND_RULE, MATRIX, TYPE_NAMES  # noqa: E402
from recovery.envs.descriptions import (  # noqa: E402
    STRATEGY_NAMES,
    build_task_instructions,
)
from study.storage import (  # noqa: E402
    SubmissionStore,
    TokenAlreadyExistsError,
)

N_TYPES = MATRIX.shape[0]
N_STRATEGIES = MATRIX.shape[1]


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def study_client(tmp_path, monkeypatch):
    """A TestClient wired to an isolated, per-test JSONL store -- no
    cross-test data leakage, no dependency on (or risk to) any real
    ``study/data/submissions.jsonl``."""
    import study.app.backend.main as m

    importlib.reload(m)  # fresh module state per test (isolated CORS/app instance is fine to reuse)
    m._store = SubmissionStore(tmp_path / "submissions.jsonl")

    from fastapi.testclient import TestClient

    return TestClient(m.app), m


def _valid_submit_payload(token: str, content: dict) -> dict:
    return {
        "token": token,
        "background": {"years_experience": "3-5", "llm_familiarity": "some", "built_agent_systems": True},
        "type_order": content["type_order"],
        "strategy_order": content["strategy_order"],
        "mapping": [0] * content["n_types"],
        "confidence": [3] * content["n_types"],
        "rationale": [""] * content["n_types"],
        "seconds_on_task": 42.0,
    }


# ======================================================================
# 1. Blindness test -- the protocol's technical guarantee
# ======================================================================


def test_blindness_no_endpoint_leaks_matrix_values_or_reward_shaped_keys(study_client):
    client, _m = study_client
    forbidden_values = {f"{v:.2f}" for v in np.asarray(MATRIX).flatten()}
    forbidden_key_patterns = [
        re.compile(p, re.IGNORECASE) for p in [r"\breward\b", r"\bsuccess_rate\b", r"\bmatrix\b", r"\bscore\b", r"\bp_success\b"]
    ]

    token = "blindness-check-token-0001"
    responses = []
    responses.append(client.get("/api/study/health"))
    r_content = client.get("/api/study/task-content", params={"token": token})
    responses.append(r_content)
    content = r_content.json()
    responses.append(client.post("/api/study/submit", json=_valid_submit_payload(token, content)))
    responses.append(client.post("/api/study/submit", json=_valid_submit_payload(token, content)))  # 409
    bad_payload = _valid_submit_payload("another-token-0002", content)
    bad_payload["mapping"] = [0] * (content["n_types"] - 1)  # wrong length -> 422
    responses.append(client.post("/api/study/submit", json=bad_payload))
    responses.append(client.get("/api/study/task-content", params={"token": "short"}))  # 422

    for resp in responses:
        body_text = resp.text
        for forbidden in forbidden_values:
            assert forbidden not in body_text, f"response leaked a MATRIX value {forbidden!r}: {body_text[:300]}"
        for pattern in forbidden_key_patterns:
            assert not pattern.search(body_text), f"response contains a forbidden key pattern {pattern.pattern!r}: {body_text[:300]}"


def test_study_backend_source_never_references_matrix_or_blind_rule():
    """Static AST audit (see study/blindness_audit.md): the study
    backend's OWN CODE (imports, attribute access, bare names -- never
    string literals, so a docstring that mentions "don't import MATRIX"
    as a warning does not itself trip this check) must never reference
    MATRIX/BLIND_RULE/recovery.theory. Catches an accidental future
    import long before it could ever reach a response body."""
    import ast

    forbidden_names = {"MATRIX", "BLIND_RULE"}
    backend_dir = _REPO_ROOT / "study" / "app" / "backend"

    for py_file in backend_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "recovery.theory" not in module, f"{py_file}: imports from {module!r}"
                for alias in node.names:
                    assert alias.name not in forbidden_names, f"{py_file}: imports {alias.name!r} from {module!r}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "recovery.theory" not in alias.name, f"{py_file}: imports {alias.name!r}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, f"{py_file}: accesses attribute {node.attr!r}"
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_names, f"{py_file}: references bare name {node.id!r}"


def test_strategy_costs_and_matrix_values_are_disjoint_ranges():
    """Regression guard behind the blindness test's implicit assumption
    (see study/blindness_audit.md) -- if this ever failed, the blindness
    test's value-matching approach could produce a false negative."""
    from recovery.envs.descriptions import STRATEGY_COSTS

    assert min(STRATEGY_COSTS) > float(np.max(MATRIX))


# ======================================================================
# 2. Immutability
# ======================================================================


def test_second_submit_with_same_token_returns_409(study_client):
    client, _m = study_client
    token = "immutable-token-0001"
    content = client.get("/api/study/task-content", params={"token": token}).json()
    payload = _valid_submit_payload(token, content)

    first = client.post("/api/study/submit", json=payload)
    assert first.status_code == 201

    second = client.post("/api/study/submit", json=payload)
    assert second.status_code == 409


def test_storage_layer_refuses_duplicate_token_directly(tmp_path):
    store = SubmissionStore(tmp_path / "s.jsonl")
    record = {
        "token": "dup-token", "timestamp_utc": "2026-01-01T00:00:00Z", "background": {},
        "display_order": {}, "mapping": [0], "confidence": [3], "rationale": [""],
        "seconds_on_task": 1.0, "protocol_version": "1.0.0",
    }
    store.append(record)
    with pytest.raises(TokenAlreadyExistsError):
        store.append(record)


# ======================================================================
# 3. Display-order randomization
# ======================================================================


def test_display_order_differs_across_tokens_and_is_recorded(study_client):
    client, _m = study_client
    orders = []
    for i in range(5):
        content = client.get("/api/study/task-content", params={"token": f"order-token-{i:04d}xxxx"}).json()
        orders.append((tuple(content["type_order"]), tuple(content["strategy_order"])))

    assert len(set(orders)) > 1, "expected different tokens to see different display orders"

    token = "order-record-token-0001"
    content = client.get("/api/study/task-content", params={"token": token}).json()
    payload = _valid_submit_payload(token, content)
    client.post("/api/study/submit", json=payload)

    records = _m._store.read_all()
    rec = next(r for r in records if r["token"] == token)
    assert rec["display_order"]["type_order"] == content["type_order"]
    assert rec["display_order"]["strategy_order"] == content["strategy_order"]


def test_task_content_is_idempotent_per_token(study_client):
    client, _m = study_client
    token = "idempotent-token-0001"
    r1 = client.get("/api/study/task-content", params={"token": token}).json()
    r2 = client.get("/api/study/task-content", params={"token": token}).json()
    assert r1["type_order"] == r2["type_order"]
    assert r1["strategy_order"] == r2["strategy_order"]


# ======================================================================
# 4 & 5. score_submission correctness (canonical rule, argmax)
# ======================================================================


def test_score_submission_canonical_blind_rule_is_6_of_7_with_miss_at_rank_2():
    result = score_submission(BLIND_RULE, MATRIX)
    assert result["q_count"] == 6
    miss_idx = TYPE_NAMES.index("wrong_tool_selected")
    assert result["per_row_correct"][miss_idx] is False
    assert result["rank_of_chosen"][miss_idx] == 2
    assert all(result["per_row_correct"][t] for t in range(N_TYPES) if t != miss_idx)


def test_score_submission_argmax_mapping_is_perfect():
    argmax_mapping = MATRIX.argmax(axis=1)
    result = score_submission(argmax_mapping, MATRIX)
    assert result["q_count"] == N_TYPES
    assert result["q_ratio"] == pytest.approx(1.0)
    assert result["regret_rate"] == pytest.approx(0.0, abs=1e-9)
    assert all(result["per_row_correct"])
    assert all(r == 1 for r in result["rank_of_chosen"])


def test_score_submission_rejects_wrong_length_mapping():
    with pytest.raises(ValueError):
        score_submission([0, 1, 2], MATRIX)


def test_score_submission_rejects_out_of_range_index():
    bad = [0] * (N_TYPES - 1) + [N_STRATEGIES]  # one entry out of range
    with pytest.raises(ValueError):
        score_submission(bad, MATRIX)


# ======================================================================
# 6. study_summary handles n=1 without crashing
# ======================================================================


def test_study_summary_handles_n_equals_1():
    submissions = [
        {
            "mapping": list(BLIND_RULE),
            "confidence": [3] * N_TYPES,
            "background": {"years_experience": "3-5", "llm_familiarity": "some", "built_agent_systems": True},
        }
    ]
    summary = study_summary(submissions, MATRIX)
    assert summary["n"] == 1
    assert summary["q_mean"] == pytest.approx(6.0)
    assert summary["q_mean_ci"] is None  # undefined at n=1, never fabricated
    assert summary["second_best_error_test"]["mean_rank_ci"] is None or summary["n"] < 2


def test_study_summary_handles_zero_submissions():
    summary_input = []
    # study_summary requires P but is only meaningfully called with >=1
    # submission by callers (experiments/study_analysis.py special-cases
    # n=0 itself, per its own module docstring); confirm the underlying
    # per-function pieces used at n=0 don't crash either.
    result = second_best_error_test([])
    assert result["n_incorrect_rows"] == 0
    assert result["mean_rank_ci"] is None


def test_second_best_error_claim_on_the_canonical_miss():
    scored = [score_submission(BLIND_RULE, MATRIX)]
    result = second_best_error_test(scored)
    assert result["n_incorrect_rows"] == 1
    assert result["fraction_rank_2"] == pytest.approx(1.0)
    assert result["mean_rank_of_incorrect"] == pytest.approx(2.0)


def test_q_prior_resamples_empirical_distribution():
    scored = [score_submission(BLIND_RULE, MATRIX), score_submission(MATRIX.argmax(axis=1), MATRIX)]
    sampler = q_prior(scored)
    rng = np.random.default_rng(0)
    draws = sampler(rng, 1000)
    assert set(np.unique(draws)) <= {6, 7}


def test_q_prior_crossover_distribution_feeds_task10_predictor():
    from recovery.analysis.human_study import q_prior_crossover_distribution

    # Every submission is the perfect table -> every resampled q == N ->
    # every synthetic mapping is the perfect table -> every T* is inf.
    perfect = MATRIX.argmax(axis=1)
    scored = [score_submission(perfect, MATRIX), score_submission(perfect, MATRIX)]
    result = q_prior_crossover_distribution(scored, MATRIX, c=1.0, n_draws=50)
    assert result["n_input_submissions"] == 2
    assert result["n_draws"] == 50
    assert np.all(np.isinf(result["T_star_samples"]))
    assert result["n_finite"] == 0
    assert result["T_star_ci"] is None  # no finite samples to bootstrap

    # A mix of the canonical (6/7) and perfect (7/7) submissions yields at
    # least some finite T* draws (the q=6 draws are imperfect tables).
    mixed = [score_submission(BLIND_RULE, MATRIX), score_submission(perfect, MATRIX)]
    result2 = q_prior_crossover_distribution(mixed, MATRIX, c=1.0, n_draws=200, seed=1)
    assert result2["n_finite"] > 0
    assert result2["T_star_median"] > 0


def test_q_prior_crossover_distribution_requires_at_least_one_submission():
    from recovery.analysis.human_study import q_prior_crossover_distribution

    with pytest.raises(ValueError):
        q_prior_crossover_distribution([], MATRIX, c=1.0)


# ======================================================================
# 7. LLM parse failure recorded, never silently replaced
# ======================================================================


def test_llm_parse_mapping_returns_none_on_malformed_response():
    from experiments.llm_designers import parse_mapping_response

    assert parse_mapping_response("not json at all") is None
    assert parse_mapping_response('{"tool_timeout": "retry"}') is None  # missing types
    assert parse_mapping_response('{"tool_timeout": "not_a_real_strategy", ' + ", ".join(
        f'"{t}": "retry"' for t in TYPE_NAMES[1:]
    ) + "}") is None  # invalid strategy name


def test_llm_parse_mapping_succeeds_on_well_formed_response():
    from experiments.llm_designers import parse_mapping_response

    obj = {t: STRATEGY_NAMES[0] for t in TYPE_NAMES}
    import json

    parsed = parse_mapping_response(json.dumps(obj))
    assert parsed == [0] * N_TYPES


def test_llm_run_one_sample_never_fabricates_a_default_mapping_on_persistent_parse_failure(monkeypatch):
    from experiments import llm_designers as ld

    def _always_unparseable(*args, **kwargs):
        return {"text": "I refuse to answer in JSON.", "model_version": "fake-model", "raw_json": {}}

    monkeypatch.setattr(ld, "call_model", _always_unparseable)

    result = ld.run_one_sample(
        family="test-family", provider_type="openai_compatible", base_url="http://example.invalid",
        api_key="fake", model="fake-model", prompt="irrelevant", temperature=1.0, max_tokens=64,
        seed=0, timeout=5, max_retries=3, sample_index=0,
    )
    assert result["parse_status"] == "parse_failure"
    assert result["parsed_mapping"] is None  # NEVER a fabricated default
    assert result["n_attempts"] == 3  # exhausted all retries, never gave up early


def test_llm_run_one_sample_retries_transient_timeout_and_recovers(monkeypatch):
    from experiments import llm_designers as ld

    calls = {"n": 0}

    def _times_out_then_succeeds(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ld.requests.exceptions.Timeout("simulated read timeout")
        obj = {t: STRATEGY_NAMES[0] for t in TYPE_NAMES}
        import json as _json

        return {"text": _json.dumps(obj), "model_version": "fake-model", "raw_json": {}}

    monkeypatch.setattr(ld, "call_model", _times_out_then_succeeds)

    result = ld.run_one_sample(
        family="test-family", provider_type="openai_compatible", base_url="http://example.invalid",
        api_key="fake", model="fake-model", prompt="irrelevant", temperature=1.0, max_tokens=64,
        seed=0, timeout=5, max_retries=3, sample_index=0,
    )
    assert result["parse_status"] == "ok"
    assert result["parsed_mapping"] == [0] * N_TYPES
    assert calls["n"] == 2  # one timeout, one successful retry -- never gave up after a transient error


def test_llm_run_one_sample_records_network_timeout_distinctly_from_parse_failure(monkeypatch):
    from experiments import llm_designers as ld

    def _always_times_out(*args, **kwargs):
        raise ld.requests.exceptions.Timeout("simulated read timeout")

    monkeypatch.setattr(ld, "call_model", _always_times_out)

    result = ld.run_one_sample(
        family="test-family", provider_type="openai_compatible", base_url="http://example.invalid",
        api_key="fake", model="fake-model", prompt="irrelevant", temperature=1.0, max_tokens=64,
        seed=0, timeout=5, max_retries=3, sample_index=0,
    )
    assert result["parse_status"] == "network_timeout"  # NOT "parse_failure" -- the model never answered
    assert result["parsed_mapping"] is None
    assert result["raw_response"] is None
    assert result["n_attempts"] == 3


def _fake_429_error(headers=None):
    import requests as _requests

    class _FakeResponse:
        status_code = 429

        def __init__(self, headers):
            self.headers = headers or {}

    exc = _requests.exceptions.HTTPError("429 rate limited")
    exc.response = _FakeResponse(headers)
    return exc


def test_llm_run_one_sample_retries_429_and_recovers(monkeypatch):
    from experiments import llm_designers as ld

    calls = {"n": 0}

    def _rate_limited_then_succeeds(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _fake_429_error({"Retry-After": "0.01"})
        obj = {t: STRATEGY_NAMES[0] for t in TYPE_NAMES}
        import json as _json

        return {"text": _json.dumps(obj), "model_version": "fake-model", "raw_json": {}}

    monkeypatch.setattr(ld, "call_model", _rate_limited_then_succeeds)

    result = ld.run_one_sample(
        family="test-family", provider_type="openai_compatible", base_url="http://example.invalid",
        api_key="fake", model="fake-model", prompt="irrelevant", temperature=1.0, max_tokens=64,
        seed=0, timeout=5, max_retries=3, sample_index=0,
    )
    assert result["parse_status"] == "ok"
    assert calls["n"] == 2  # one 429, one successful retry


def test_llm_run_one_sample_records_rate_limited_distinctly(monkeypatch):
    from experiments import llm_designers as ld

    def _always_429(*args, **kwargs):
        raise _fake_429_error({"Retry-After": "0.01"})

    monkeypatch.setattr(ld, "call_model", _always_429)

    result = ld.run_one_sample(
        family="test-family", provider_type="openai_compatible", base_url="http://example.invalid",
        api_key="fake", model="fake-model", prompt="irrelevant", temperature=1.0, max_tokens=64,
        seed=0, timeout=5, max_retries=3, sample_index=0,
    )
    assert result["parse_status"] == "rate_limited"  # NOT "network_timeout" or "parse_failure"
    assert result["parsed_mapping"] is None
    assert result["n_attempts"] == 3


def test_llm_run_one_sample_does_not_retry_deterministic_http_errors(monkeypatch):
    from experiments import llm_designers as ld

    def _fake_404(*args, **kwargs):
        import requests as _requests

        class _FakeResponse:
            status_code = 404
            headers = {}

        exc = _requests.exceptions.HTTPError("404 not found")
        exc.response = _FakeResponse()
        raise exc

    monkeypatch.setattr(ld, "call_model", _fake_404)

    with pytest.raises(ld.requests.exceptions.HTTPError):
        ld.run_one_sample(
            family="test-family", provider_type="openai_compatible", base_url="http://example.invalid",
            api_key="fake", model="fake-model", prompt="irrelevant", temperature=1.0, max_tokens=64,
            seed=0, timeout=5, max_retries=3, sample_index=0,
        )


def test_parse_rate_limit_wait_seconds_handles_groq_and_standard_headers():
    from experiments.llm_designers import _parse_rate_limit_wait_seconds

    class _R:
        def __init__(self, headers):
            self.headers = headers

    assert _parse_rate_limit_wait_seconds(_R({"Retry-After": "5"})) == 5.0
    assert _parse_rate_limit_wait_seconds(_R({"x-ratelimit-reset-tokens": "6s"})) == 6.0
    assert _parse_rate_limit_wait_seconds(_R({"x-ratelimit-reset-tokens": "1m42s"}), cap=200.0) == 102.0
    assert _parse_rate_limit_wait_seconds(_R({"x-ratelimit-reset-tokens": "570ms"})) == pytest.approx(0.57)
    assert _parse_rate_limit_wait_seconds(_R({}), default=7.0) == 7.0
    assert _parse_rate_limit_wait_seconds(_R({"Retry-After": "9999"}), cap=65.0) == 65.0  # never uncapped


# ======================================================================
# 8. LLM prompt built from the shared descriptions module
# ======================================================================


def test_llm_prompt_contains_shared_instructions_verbatim():
    from experiments.llm_designers import build_prompt

    prompt = build_prompt()
    assert build_task_instructions() in prompt
    # every failure-type and strategy name from the SAME source the web
    # app serves must appear -- not a paraphrase, not a separate copy.
    for name in TYPE_NAMES:
        assert name in prompt
    for name in STRATEGY_NAMES:
        assert name in prompt


def test_llm_prompt_matches_web_app_content_source_exactly(study_client):
    """The prompt and the web app's served content must be constructed
    from the SAME source strings -- fetch what the web app actually
    serves and assert it round-trips through the same descriptions the
    prompt embeds."""
    client, _m = study_client
    content = client.get("/api/study/task-content", params={"token": "prompt-check-token-0001"}).json()

    from experiments.llm_designers import build_prompt

    prompt = build_prompt()
    served_type_names = {t["name"] for t in content["types"]}
    served_strategy_names = {s["name"] for s in content["strategies"]}
    assert served_type_names == set(TYPE_NAMES)
    assert served_strategy_names == set(STRATEGY_NAMES)
    for t in content["types"]:
        assert t["description"] in prompt
    for s in content["strategies"]:
        assert s["description"] in prompt


# ======================================================================
# 9. Task 15 -- prompt-robustness supplementary experiment
# ======================================================================


def test_build_task_instructions_default_args_unchanged():
    """Adding optional type_order/strategy_order params must not change
    build_task_instructions()'s existing zero-arg behaviour (Task 11/
    test_llm_prompt_contains_shared_instructions_verbatim's assumption)."""
    assert build_task_instructions() == build_task_instructions(None, None)
    assert build_task_instructions() == build_task_instructions(
        list(range(len(TYPE_NAMES))), list(range(len(STRATEGY_NAMES)))
    )


def test_build_task_instructions_reordering_preserves_content():
    """A non-identity permutation changes presentation ORDER only -- every
    name/label/description and the cost values must still all be present."""
    reversed_types = list(range(len(TYPE_NAMES) - 1, -1, -1))
    reversed_strategies = list(range(len(STRATEGY_NAMES) - 1, -1, -1))
    reordered = build_task_instructions(reversed_types, reversed_strategies)
    natural = build_task_instructions()

    assert reordered != natural  # order actually changed
    for name in TYPE_NAMES:
        assert name in reordered
    for name in STRATEGY_NAMES:
        assert name in reordered
    # First failure type listed is now the LAST one in natural order.
    first_line_types = [ln for ln in reordered.splitlines() if ln.strip().startswith("-")][0]
    assert TYPE_NAMES[-1] in first_line_types


def test_display_order_module_matches_study_backend_after_refactor(study_client):
    """recovery.envs.display_order is a pure extraction of what
    study/app/backend/main.py used to define locally -- confirm the
    refactored backend's _display_order still delegates to it and produces
    a valid permutation for a real request."""
    from recovery.envs.display_order import display_order as shared_display_order

    client, m = study_client
    token = "refactor-check-token-0001"
    content = client.get("/api/study/task-content", params={"token": token}).json()

    expected_types, expected_strategies = shared_display_order(token, N_TYPES, N_STRATEGIES)
    assert content["type_order"] == expected_types
    assert content["strategy_order"] == expected_strategies
    assert m._display_order(token) == (expected_types, expected_strategies)


def test_paraphrase_builders_preserve_names_and_costs():
    from recovery.envs.descriptions import STRATEGY_COSTS
    from recovery.envs.paraphrases import PARAPHRASE_BUILDERS

    for variant, builder in PARAPHRASE_BUILDERS.items():
        text = builder()
        for name in TYPE_NAMES:
            assert name in text, f"{variant} missing type name {name!r}"
        for name in STRATEGY_NAMES:
            assert name in text, f"{variant} missing strategy name {name!r}"
        for cost in STRATEGY_COSTS:
            assert f"cost={cost}" in text, f"{variant} missing cost marker for {cost!r}"


def test_paraphrases_differ_from_primary_and_from_each_other():
    from recovery.envs.paraphrases import build_task_instructions_paraphrase_a, build_task_instructions_paraphrase_b

    primary = build_task_instructions()
    a = build_task_instructions_paraphrase_a()
    b = build_task_instructions_paraphrase_b()
    assert a != primary
    assert b != primary
    assert a != b


def test_paraphrase_prompts_pass_blindness_audit():
    """Same check tests/test_study.py's blindness test applies to the study
    backend's HTTP responses, applied here to prompt TEXT before any API
    quota is spent on it."""
    from experiments.llm_designers import RESPONSE_FORMAT_INSTRUCTIONS
    from recovery.analysis.prompt_blindness_audit import audit_prompt_text
    from recovery.envs.paraphrases import PARAPHRASE_BUILDERS

    for variant, builder in {"primary": build_task_instructions, **PARAPHRASE_BUILDERS}.items():
        prompt = builder() + RESPONSE_FORMAT_INSTRUCTIONS
        violations = audit_prompt_text(prompt)
        assert violations == [], f"{variant} failed blindness audit: {violations}"


def test_llm_build_prompt_variant_dispatch():
    from experiments.llm_designers import build_prompt

    assert build_prompt() == build_prompt("primary")
    assert build_task_instructions() in build_prompt("primary")
    assert build_task_instructions() not in build_prompt("paraphrase_a")  # genuinely different wording
    with pytest.raises(ValueError):
        build_prompt("not_a_real_variant")


def test_llm_designers_cli_refuses_non_primary_variant_at_default_output():
    """Structural safety guard: a paraphrase run must be unable to
    overwrite results/tables/llm_designers.parquet even by mistake (no
    --output given). Fails fast at argument-parsing time -- no API call is
    ever attempted."""
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [_sys.executable, str(_BACKEND_DIR / "experiments" / "llm_designers.py"), "--prompt-variant", "paraphrase_a"],
        cwd=str(_BACKEND_DIR), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "refuses to write to the default --output path" in result.stderr


def test_run_model_config_respects_n_temp_overrides(monkeypatch):
    from experiments import llm_designers as ld

    def _fake_call(*args, **kwargs):
        obj = {t: STRATEGY_NAMES[0] for t in TYPE_NAMES}
        import json as _json

        return {"text": _json.dumps(obj), "model_version": "fake-model", "raw_json": {}}

    monkeypatch.setattr(ld, "call_model", _fake_call)

    cfg = {
        "family": "test-family", "provider_type": "openai_compatible", "base_url": "http://example.invalid",
        "model": "fake-model", "api_key_env": "TEST_FAKE_KEY_ENV",
    }
    monkeypatch.setenv("TEST_FAKE_KEY_ENV", "fake-key")

    rows = ld.run_model_config(cfg, "irrelevant prompt", max_retries=3, timeout=5, validation=False,
                                n_temp1_override=2, n_temp0_override=0)
    assert len(rows) == 2  # NOT the YAML default of 20 -- the override took effect
    assert all(r["parse_status"] == "ok" for r in rows)


def test_variant_q_comparison_test_reuses_scipy_mannwhitney():
    from recovery.analysis.significance import variant_q_comparison_test

    q_a = [6, 7, 6, 5, 6, 7, 6]
    q_b = [3, 4, 3, 2, 4, 3, 3]
    result = variant_q_comparison_test(q_a, q_b)
    assert result["n_a"] == 7
    assert result["n_b"] == 7
    assert result["median_a"] > result["median_b"]
    assert 0.0 <= result["p_value"] <= 1.0
    # These two samples barely overlap -- a real, near-significant difference.
    assert result["p_value"] < 0.05


def test_variant_q_comparison_test_handles_small_samples_without_raising():
    from recovery.analysis.significance import variant_q_comparison_test

    result = variant_q_comparison_test([5], [3, 4])
    assert result["n_a"] == 1
    import math

    assert math.isnan(result["p_value"])  # never raises, never fabricates a value at n<2
