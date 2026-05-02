from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt import adapter as adapter_mod  # noqa: E402
from agent_opt.adapter import TraceAdapter  # noqa: E402
from agent_opt.reflection import REFLECTION_PROMPT_TEMPLATE  # noqa: E402


GOOD_RECORD = {
    "id": "good_1",
    "src": "cc",
    "context": {
        "user_request": "Read the config file at config/app.yaml",
        "recent_actions": [{"name": "LS", "input": "config/"}],
        "available_tools": ["Read", "Edit", "Bash", "Grep"],
    },
    "observed_action": {"kind": "tool_use", "name": "Read", "input": "config/app.yaml"},
    "label": "good",
    "failure_category": None,
}

BAD_RECORD = {
    "id": "bad_1",
    "src": "cc",
    "context": {
        "user_request": "Run the migration",
        "recent_actions": [{"name": "Bash", "input": "alembic upgrade head"}],
        "available_tools": ["Read", "Edit", "Bash", "Grep"],
    },
    "observed_action": {"kind": "tool_use", "name": "Bash", "input": "alembic upgrade head"},
    "label": "bad",
    "failure_category": "bash_exit_nonzero",
}


@pytest.fixture
def patched_chat(monkeypatch):
    calls = []

    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        calls.append({"messages": messages, "system": system, "model": model})
        return '{"tool_name":"Read","brief_reason":"inspect file"}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    return calls


def test_evaluate_scores_and_trajectories(patched_chat):
    adapter = TraceAdapter(task_model="claude-haiku-4-5", max_tokens=64)
    batch = [GOOD_RECORD, BAD_RECORD]
    candidate = {"system": "Be careful and verify before acting."}

    result = adapter.evaluate(batch, candidate, capture_traces=True)

    assert len(result.outputs) == 2
    assert len(result.scores) == 2
    assert result.trajectories is not None and len(result.trajectories) == 2

    assert result.scores[0] == 1.0
    assert result.scores[1] == 1.0

    assert all(call["system"] == candidate["system"] for call in patched_chat)
    assert all(call["model"] == "claude-haiku-4-5" for call in patched_chat)

    for traj, rec in zip(result.trajectories, batch):
        assert traj["input_record"] is rec
        assert traj["predicted"] == {"tool_name": "Read", "brief_reason": "inspect file"}
        assert traj["raw_output"].startswith("{")


def test_evaluate_no_traces(patched_chat):
    adapter = TraceAdapter()
    result = adapter.evaluate([GOOD_RECORD], {"system": "x"}, capture_traces=False)
    assert result.trajectories is None
    assert result.scores == [1.0]


def test_score_bad_when_chooses_failed_action(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return '{"tool_name":"Bash","brief_reason":"run it"}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    result = adapter.evaluate([BAD_RECORD], {"system": "x"}, capture_traces=True)
    assert result.scores == [0.0]


def test_score_parse_failure_returns_zero(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return "I will not output JSON."

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    result = adapter.evaluate([GOOD_RECORD], {"system": "x"}, capture_traces=True)
    assert result.scores == [0.0]
    assert result.outputs == [None]


def test_make_reflective_dataset_feedback(patched_chat):
    adapter = TraceAdapter()
    batch = [GOOD_RECORD, BAD_RECORD]
    eval_batch = adapter.evaluate(batch, {"system": "x"}, capture_traces=True)

    reflective = adapter.make_reflective_dataset(
        candidate={"system": "x"},
        eval_batch=eval_batch,
        components_to_update=["system"],
    )

    assert "system" in reflective
    recs = list(reflective["system"])
    assert len(recs) == 2

    good_fb = recs[0]["Feedback"]
    bad_fb = recs[1]["Feedback"]

    assert good_fb.startswith("GOOD: reproduced known-good action `Read`")
    assert "GOOD: avoided previously-failed action `Bash`" in bad_fb
    assert "bash_exit_nonzero" in bad_fb

    for rec in recs:
        assert "Inputs" in rec and "Generated Outputs" in rec and "Feedback" in rec
        assert "user_request" in rec["Inputs"]


def test_make_reflective_dataset_miss_feedback(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return '{"tool_name":"Bash","brief_reason":"guessing"}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    eval_batch = adapter.evaluate([GOOD_RECORD], {"system": "x"}, capture_traces=True)
    reflective = adapter.make_reflective_dataset({"system": "x"}, eval_batch, ["system"])
    fb = list(reflective["system"])[0]["Feedback"]
    assert fb.startswith("MISS:")
    assert "Bash" in fb and "Read" in fb


def test_reflection_template_has_required_placeholders():
    assert "<curr_instructions>" in REFLECTION_PROMPT_TEMPLATE
    assert "<inputs_outputs_feedback>" in REFLECTION_PROMPT_TEMPLATE
    assert "hallucinated_skill" in REFLECTION_PROMPT_TEMPLATE


GOOD_BASH_GREP_RECORD = {
    "id": "good_bash_grep",
    "src": "cc",
    "context": {
        "user_request": "Find TODOs in src/",
        "recent_actions": [],
        "available_tools": ["Read", "Edit", "Bash", "Grep"],
    },
    "observed_action": {"kind": "tool_use", "name": "Bash", "input": "rg -n TODO src/"},
    "label": "good",
    "failure_category": None,
}

USER_CONFIRMED_RECORD = {
    "id": "uc_1",
    "src": "cc",
    "context": {
        "user_request": "Read the config",
        "recent_actions": [],
        "available_tools": ["Read", "Edit", "Bash", "Grep"],
    },
    "observed_action": {"kind": "tool_use", "name": "Read", "input": "config/app.yaml"},
    "label": "user_confirmed",
    "failure_category": None,
}


def test_score_family_partial_credit_good(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return '{"tool_name":"Grep","brief_reason":"search code"}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    result = adapter.evaluate([GOOD_BASH_GREP_RECORD], {"system": "x"}, capture_traces=False)
    assert result.scores == [0.5]


def test_score_parse_fallback_recovers_tool_name(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return "I think we should use Read here to inspect."

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    result = adapter.evaluate([GOOD_RECORD], {"system": "x"}, capture_traces=True)
    # observed=Read, fallback=Read, weight=0.5, base=1.0 -> 0.5
    assert result.scores == [0.5]
    assert result.outputs == [None]


def test_score_user_confirmed_credits_match(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return '{"tool_name":"Read","brief_reason":"open the config"}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    result = adapter.evaluate([USER_CONFIRMED_RECORD], {"system": "x"}, capture_traces=False)
    assert result.scores == [1.0]


def test_score_user_confirmed_family_partial(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return '{"tool_name":"Bash","input":"cat config/app.yaml","brief_reason":"inspect file"}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    # Bash with cat -> read family, observed Read -> read family -> 0.5
    result = adapter.evaluate([USER_CONFIRMED_RECORD], {"system": "x"}, capture_traces=False)
    assert result.scores == [0.5]


def test_score_brief_reason_penalty(monkeypatch):
    def fake_chat(messages, model, max_tokens=1024, temperature=0.0, system=None):
        return '{"tool_name":"Read","brief_reason":""}'

    monkeypatch.setattr(adapter_mod._llm, "chat", fake_chat)
    adapter = TraceAdapter()
    result = adapter.evaluate([GOOD_RECORD], {"system": "x"}, capture_traces=False)
    assert result.scores == [pytest.approx(0.95)]
