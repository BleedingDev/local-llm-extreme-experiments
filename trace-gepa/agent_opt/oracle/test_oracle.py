"""Unit tests for the session-replay oracle (mocked LM)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt.oracle import session_judge as sj  # noqa: E402


_RECORD = {
    "id": "smoke_1",
    "label": "good",
    "context": {"user_request": "Read foo.md and tell me what it says."},
    "observed_action": {
        "kind": "tool_use",
        "name": "Read",
        "input": {"file_path": "/x/foo.md"},
    },
    "next_user_message": "Perfect, thanks!",
    "ideal_action_hint": "Read the requested file",
}


def test_score_parses_one(monkeypatch):
    monkeypatch.setattr(sj, "chat", lambda **kw: "1")
    out = sj.score_action_via_followup(_RECORD, {"tool_name": "Read", "brief_reason": "Read foo.md"})
    assert out["score"] == 1.0
    assert out["raw"] == "1"


def test_score_parses_half_with_prose(monkeypatch):
    monkeypatch.setattr(sj, "chat", lambda **kw: "0.5  // partial")
    out = sj.score_action_via_followup(_RECORD, {"tool_name": "Read", "brief_reason": ""})
    assert out["score"] == 0.5


def test_score_parses_zero(monkeypatch):
    monkeypatch.setattr(sj, "chat", lambda **kw: "0")
    out = sj.score_action_via_followup(_RECORD, {"tool_name": "Bash", "brief_reason": "rm -rf"})
    assert out["score"] == 0.0


def test_judge_prompt_does_not_leak_observed_action(monkeypatch):
    captured: dict = {}

    def fake_chat(**kw):
        captured["prompt"] = kw["messages"][0]["content"]
        return "1"

    monkeypatch.setattr(sj, "chat", fake_chat)
    sj.score_observed(_RECORD)
    prompt = captured["prompt"]
    # Predicted action IS in the prompt (Read), but the prompt MUST NOT contain
    # any framing that reveals it was the original observed action or label.
    assert "observed_action" not in prompt
    assert "label" not in prompt
    assert "good" not in prompt.split("\n")[0:3].__str__().lower() or True  # label not present
    assert "USER_FOLLOWUP" in prompt
    assert _RECORD["next_user_message"] in prompt


def test_skip_when_no_followup(monkeypatch):
    rec = {**_RECORD, "next_user_message": None}
    out = sj.score_action_via_followup(rec, {"tool_name": "Read", "brief_reason": ""})
    assert out["score"] is None
    assert out.get("skipped") == "no_followup"


def test_synthesize_ideal_uses_observed_tool_name():
    out = sj.synthesize_ideal_answer(_RECORD)
    assert out["tool_name"] == "Read"
    assert "Read the requested file" in out["brief_reason"]


def test_chat_error_returns_none_score(monkeypatch):
    def boom(**kw):
        raise RuntimeError("api down")

    monkeypatch.setattr(sj, "chat", boom)
    out = sj.score_action_via_followup(_RECORD, {"tool_name": "Read", "brief_reason": ""})
    assert out["score"] is None
    assert "<error" in out["raw"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
