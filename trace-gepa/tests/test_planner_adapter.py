from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt import planner_adapter as pa_mod  # noqa: E402
from agent_opt.planner_adapter import PlannerAdapter, _file_set, _jaccard, _score  # noqa: E402

GT = {
    "id": "planner_test_1",
    "user_request": "Add a greet function in greet.py and tests for it.",
    "ground_truth_issues": [
        {"title": "Create greet.py", "expectedFiles": ["greet.py"], "verifierCommands": ["test -f greet.py"]},
        {"title": "Add tests/test_greet.py", "expectedFiles": ["tests/test_greet.py"], "verifierCommands": ["pytest tests/test_greet.py -q"]},
    ],
    "src": "synthetic", "src_path": "n/a",
}


def test_score_perfect_match():
    pred = {"issues": [
        {"title": "Make greet.py", "expectedFiles": ["greet.py"], "verifierCommands": ["test -f greet.py"]},
        {"title": "Test it", "expectedFiles": ["tests/test_greet.py"], "verifierCommands": ["pytest -q"]},
    ]}
    score, bd = _score(GT, pred)
    assert score == pytest.approx(1.0)
    assert bd["json_valid"] == 0.3 and bd["count_match"] == 0.2 and bd["verifier_coverage"] == 0.2


def test_score_no_json():
    score, bd = _score(GT, None)
    assert score == 0.0 and bd["json_valid"] == 0.0


def test_score_missing_issues_key():
    score, _ = _score(GT, {"foo": "bar"})
    assert score == 0.0


def test_score_count_off_no_files():
    pred = {"issues": [{"title": f"i{i}", "expectedFiles": [], "verifierCommands": []} for i in range(8)]}
    score, bd = _score(GT, pred)
    assert bd["json_valid"] == 0.3 and bd["count_match"] == 0.0 and score == pytest.approx(0.3)


def test_score_partial_file_overlap_no_verifier():
    pred = {"issues": [{"title": "a", "expectedFiles": ["greet.py"], "verifierCommands": []}]}
    score, bd = _score(GT, pred)
    assert bd["json_valid"] == 0.3 and bd["count_match"] == 0.2
    assert bd["file_jaccard"] > 0.0 and bd["verifier_coverage"] == 0.0


def test_jaccard_basics():
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({"a"}, {"a"}) == 1.0
    assert _jaccard({"a", "b"}, {"a"}) == pytest.approx(0.5)


def test_file_set_normalizes():
    s = _file_set([{"expectedFiles": ["/Users/joe/proj/src/foo.ts"]}])
    assert "foo.ts" in s and any("src/foo.ts" in p for p in s)


def test_evaluate_with_patched_chat(monkeypatch):
    def fake(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return json.dumps({"issues": [
            {"title": "make greet.py", "expectedFiles": ["greet.py"], "verifierCommands": ["test -f greet.py"]},
            {"title": "test it", "expectedFiles": ["tests/test_greet.py"], "verifierCommands": ["pytest -q"]},
        ]})
    monkeypatch.setattr(pa_mod._llm, "chat", fake)
    r = PlannerAdapter(max_tokens=512).evaluate([GT], {"system": "x"}, capture_traces=True)
    assert r.scores[0] == pytest.approx(1.0)
    assert r.trajectories and "breakdown" in r.trajectories[0]


def test_evaluate_invalid_json_returns_zero(monkeypatch):
    monkeypatch.setattr(pa_mod._llm, "chat", lambda **k: "I refuse to output JSON, sorry.")
    r = PlannerAdapter().evaluate([GT], {"system": "x"}, capture_traces=True)
    assert r.scores == [0.0]


def test_make_reflective_dataset_emits_feedback(monkeypatch):
    monkeypatch.setattr(pa_mod._llm, "chat", lambda **k: json.dumps(
        {"issues": [{"title": "wrong", "expectedFiles": ["other.py"], "verifierCommands": []}]}))
    a = PlannerAdapter()
    eb = a.evaluate([GT], {"system": "x"}, capture_traces=True)
    refl = a.make_reflective_dataset({"system": "x"}, eb, ["system"])
    fb = list(refl["system"])[0]["Feedback"]
    assert "MISSED_FILES" in fb or "EXTRA_FILES" in fb
