"""Minimal unit test for the counterfactual annotator (mocked LM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt import counterfactual as cf  # noqa: E402


_RECORD = {
    "id": "bad_smoke_1",
    "label": "bad",
    "context": {
        "user_request": "Run the formatter and verify the workspace is clean.",
        "recent_actions": [
            {"name": "Bash", "input": {"command": "pnpm exec vp check"}},
        ],
        "available_tools": ["Read", "Edit", "Bash", "Grep"],
    },
    "observed_action": {
        "kind": "tool_use",
        "name": "Bash",
        "input": {"command": "pnpm exec vp check"},
        "result_is_error": True,
        "result_excerpt": "Exit code 1\nFormatting issues found.",
    },
    "failure_category": "bash_exit_nonzero",
    "next_user_message": "just run --fix and commit",
}


def test_annotate_parses_valid_response(monkeypatch):
    fake_raw = json.dumps(
        {
            "counterfactual_action": {
                "name": "Bash",
                "input": {"command": "pnpm exec vp check --fix && pnpm exec vp check"},
            },
            "rationale": "Auto-fix formatting then re-verify, do not surface fixable lint errors.",
            "delta_kind": "input_fix",
            "confidence": 0.85,
        }
    )

    def fake_chat_with_cache(user_msg, model, max_tokens):
        assert "observed_action_that_failed" in user_msg
        assert model == "claude-opus-4-7"
        return fake_raw

    monkeypatch.setattr(cf, "_chat_with_cache", fake_chat_with_cache)
    out = cf.annotate(_RECORD)
    assert out is not None
    assert out["record_id"] == "bad_smoke_1"
    assert out["delta_kind"] == "input_fix"
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["counterfactual_action"]["name"] == "Bash"
    assert out["observed_action"]["name"] == "Bash"


def test_annotate_rejects_copout(monkeypatch):
    fake_raw = json.dumps(
        {
            "counterfactual_action": {"name": "", "input": None},
            "rationale": "I cannot determine without more context what to do here.",
            "delta_kind": "abort",
            "confidence": 0.1,
        }
    )
    monkeypatch.setattr(cf, "_chat_with_cache", lambda *a, **k: fake_raw)
    assert cf.annotate(_RECORD) is None


def test_annotate_rejects_bad_delta_kind(monkeypatch):
    fake_raw = json.dumps(
        {
            "counterfactual_action": {"name": "Bash", "input": {"command": "ls"}},
            "rationale": "Listing the directory first is safer.",
            "delta_kind": "look_around",
            "confidence": 0.5,
        }
    )
    monkeypatch.setattr(cf, "_chat_with_cache", lambda *a, **k: fake_raw)
    assert cf.annotate(_RECORD) is None


def test_annotate_handles_api_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(cf, "_chat_with_cache", boom)
    assert cf.annotate(_RECORD) is None


def test_annotate_recovers_malformed_json_with_prose(monkeypatch):
    raw = (
        "Here is the answer:\n"
        '{"counterfactual_action":{"name":"Read","input":{"file_path":"/x"}},'
        '"rationale":"Inspect the file before editing it.",'
        '"delta_kind":"verify_first","confidence":0.7}\n'
        "(end)"
    )
    monkeypatch.setattr(cf, "_chat_with_cache", lambda *a, **k: raw)
    out = cf.annotate(_RECORD)
    assert out is not None
    assert out["delta_kind"] == "verify_first"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
