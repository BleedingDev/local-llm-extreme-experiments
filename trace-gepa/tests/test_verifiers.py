from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.verifiers import KIND_TO_VERIFIER, verify  # noqa: E402
from bench.verifiers.tier2_judge import verify_lm_judge  # noqa: E402
from bench.verifiers.tier3_shell import verify_shell_exec  # noqa: E402


# ---- Tier 1 (legacy field-name backward compat) ----

def test_regex_pass_case_insensitive_legacy():
    out = verify({"verifier_kind": "regex",
                  "verifier_spec": {"pattern": r"hello\s+world", "case_insensitive": True}},
                 "Hello   World!")
    assert out["score"] == 1.0 and out["tier"] == 1 and out["signal"] == "regex_match"


def test_regex_fail_legacy():
    out = verify({"verifier_kind": "regex", "verifier_spec": {"pattern": r"^DONE$",
                                                              "case_sensitive": True}},
                 "still working")
    assert out["score"] == 0.0 and out["signal"] == "regex_miss"


def test_exact_match_normalizes_whitespace():
    out = verify({"verifier_kind": "exact_match",
                  "verifier_spec": {"expected": "the answer is 42"}},
                 "  the   answer\tis 42 ")
    assert out["score"] == 1.0


def test_structural_json_valid_with_schema():
    schema = {"type": "object", "required": ["tool_name", "brief_reason"],
              "properties": {"tool_name": {"type": "string", "minLength": 1},
                             "brief_reason": {"type": "string"}}}
    out = verify({"verifier_kind": "structural_json", "verifier_spec": {"schema": schema}},
                 json.dumps({"tool_name": "Read", "brief_reason": "open file"}))
    assert out["score"] == 1.0 and out["signal"] == "schema_ok"


def test_structural_json_invalid_missing_required():
    schema = {"type": "object", "required": ["tool_name"],
              "properties": {"tool_name": {"type": "string"}}}
    out = verify({"verifier_kind": "structural_json", "verifier_spec": {"schema": schema}},
                 json.dumps({"brief_reason": "no tool"}))
    assert out["score"] == 0.0 and out["signal"] == "schema_fail"


def test_tool_name_and_family_match():
    pred = json.dumps({"tool_name": "Bash", "input": "rg foo src/"})
    out_n = verify({"verifier_kind": "tool_name_match",
                    "verifier_spec": {"expected_tool": "Bash"}}, pred)
    out_f = verify({"verifier_kind": "tool_family_match",
                    "verifier_spec": {"expected_tool": "Grep"}}, pred)
    assert out_n["score"] == 1.0 and out_f["score"] == 1.0
    assert out_f["details"]["got_family"] == "search"


# ---- Tier 1 (real benchmark task field names) ----

def test_regex_reads_pattern_or_command():
    """Real tasks use `pattern_or_command`, not `pattern`."""
    task = {"verifier_kind": "regex",
            "verifier_spec": {"type": "regex",
                              "pattern_or_command": r"ls.*/Users/satan/\.claude/projects/"}}
    pred = json.dumps({"tool_name": "Bash",
                       "brief_reason": "ls /Users/satan/.claude/projects/"})
    out = verify(task, pred)
    assert out["score"] == 1.0 and out["signal"] == "regex_match"


def test_regex_pattern_or_command_miss():
    task = {"verifier_kind": "regex",
            "verifier_spec": {"type": "regex", "pattern_or_command": r"rg.*--type"}}
    out = verify(task, json.dumps({"tool_name": "Read", "brief_reason": "open file"}))
    assert out["score"] == 0.0 and out["signal"] == "regex_miss"


def test_regex_searches_input_field():
    task = {"verifier_kind": "regex",
            "verifier_spec": {"type": "regex", "pattern_or_command": r"find.*sessions"}}
    pred = json.dumps({"tool_name": "Bash", "input": "find ~/.codex/sessions -type f"})
    assert verify(task, pred)["score"] == 1.0


def test_regex_case_insensitive_default():
    task = {"verifier_kind": "regex",
            "verifier_spec": {"type": "regex", "pattern_or_command": "PNPM.*--noemit"}}
    pred = json.dumps({"tool_name": "Bash", "brief_reason": "pnpm tsc --noEmit"})
    assert verify(task, pred)["score"] == 1.0


def test_structural_json_dsl_eq_pass():
    """Real `pattern_or_command` DSL: '$.tool_name == "Read"'."""
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": '$.tool_name == "Read"'}}
    out = verify(task, json.dumps({"tool_name": "Read", "brief_reason": "open file"}))
    assert out["score"] == 1.0 and out["signal"] == "schema_ok"


def test_structural_json_dsl_eq_fail():
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": '$.tool_name == "Read"'}}
    out = verify(task, json.dumps({"tool_name": "Bash", "brief_reason": "cmd"}))
    assert out["score"] == 0.0 and out["signal"] == "schema_fail"


def test_structural_json_dsl_and_clause():
    """`$.tool_name == "Edit" and $.input.old_string != null`."""
    pat = '$.tool_name == "Edit" and $.input.old_string != null'
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": pat}}
    pred_ok = json.dumps({"tool_name": "Edit", "input": {"old_string": "x", "new_string": "y"}})
    pred_partial = json.dumps({"tool_name": "Edit", "input": {"old_string": None}})
    pred_bad = json.dumps({"tool_name": "Bash", "input": {"old_string": None}})
    assert verify(task, pred_ok)["score"] == 1.0
    assert 0.0 < verify(task, pred_partial)["score"] < 1.0  # one of two clauses passes
    assert verify(task, pred_bad)["score"] == 0.0


def test_structural_json_dsl_in_set():
    pat = 'tool_name in {"Bash","Glob","Grep"}'
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": pat}}
    assert verify(task, json.dumps({"tool_name": "Grep"}))["score"] == 1.0
    assert verify(task, json.dumps({"tool_name": "Read"}))["score"] == 0.0


def test_structural_json_no_assertions_returns_zero():
    """Pre-fix bug: empty/unrecognizable spec was silently passing."""
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": ""}}
    out = verify(task, json.dumps({"tool_name": "Read"}))
    assert out["score"] == 0.0 and out["signal"] == "no_assertions_found"


def test_structural_json_unparseable_dsl_returns_zero():
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema",
                              "pattern_or_command": "garbage that nothing recognizes"}}
    out = verify(task, json.dumps({"tool_name": "Read"}))
    assert out["score"] == 0.0 and out["signal"] == "no_assertions_found"


def test_structural_json_rich_field_tool_name_must_equal():
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"tool_name_must_equal": "Read"}}
    assert verify(task, json.dumps({"tool_name": "Read"}))["score"] == 1.0
    assert verify(task, json.dumps({"tool_name": "Bash"}))["score"] == 0.0


def test_structural_json_rich_field_avoid_and_keywords():
    spec = {"tool_name_must_avoid": ["Bash"],
            "must_include_keywords_in_input": ["src/"],
            "must_avoid_keywords_in_input": ["rm -rf"]}
    task = {"verifier_kind": "structural_json", "verifier_spec": spec}
    good = json.dumps({"tool_name": "Grep", "input": "rg foo src/"})
    bad = json.dumps({"tool_name": "Bash", "input": "rm -rf /"})
    assert verify(task, good)["score"] == 1.0
    assert verify(task, bad)["score"] == 0.0


def test_structural_json_input_pattern_regex():
    task = {"verifier_kind": "structural_json",
            "verifier_spec": {"input_pattern_regex": r"^rg\s+.*src/"}}
    good = json.dumps({"tool_name": "Bash", "input": "rg foo src/"})
    bad = json.dumps({"tool_name": "Bash", "input": "ls /tmp"})
    assert verify(task, good)["score"] == 1.0
    assert verify(task, bad)["score"] == 0.0


def test_no_repeat_handles_input_excerpt_shape():
    """Phase-3 fix: recent_actions in benchmark_tasks use `input_excerpt`,
    not `input`. Pre-fix the no_repeat clause silently returned None and
    yielded no_assertions_found across every debugging task."""
    task = {
        "verifier_kind": "structural_json",
        "verifier_spec": {"type": "json_schema",
                          "pattern_or_command": "must not repeat failing command verbatim"},
        "prompt": {
            "context": {"recent_actions": [
                {"name": "exec_command",
                 "input_excerpt": '{"cmd":"npx fallow health --format json > tmp/fallow-health.json"}'},
            ]},
        },
    }
    good = json.dumps({"tool_name": "Bash", "input": {"command": "env | grep NPM"},
                       "brief_reason": "inspect npm config"})
    bad = json.dumps({"tool_name": "Bash",
                      "input": {"command": "npx fallow health --format json > tmp/fallow-health.json"},
                      "brief_reason": "retry"})
    assert verify(task, good)["score"] == 1.0
    assert verify(task, bad)["score"] == 0.0


def test_no_repeat_handles_bare_string_recent_action():
    """Recent actions can also arrive as bare strings like 'Bash: {...}'."""
    task = {
        "verifier_kind": "structural_json",
        "verifier_spec": {"type": "json_schema",
                          "pattern_or_command": "must not repeat failing command verbatim"},
        "prompt": {"context": {"recent_actions": [
            'Bash: {"command":"cd /repo && zig test src/io/foo.zig 2>&1 | tail -30"}',
        ]}},
    }
    bad = json.dumps({"tool_name": "Bash",
                      "input": {"command": "cd /repo && zig test src/io/foo.zig 2>&1 | tail -30"}})
    good = json.dumps({"tool_name": "Read", "input": {"file_path": "src/io/foo.zig"}})
    assert verify(task, bad)["score"] == 0.0
    assert verify(task, good)["score"] == 1.0


def test_no_repeat_uses_command_line_in_user_request():
    """Debugging prompts include 'Command: ...' verbatim. Treat that as a
    recent failing command even when recent_actions is empty."""
    task = {
        "verifier_kind": "structural_json",
        "verifier_spec": {"type": "json_schema",
                          "pattern_or_command": "must not repeat failing command verbatim"},
        "prompt": {
            "user_request": "Previous Bash command failed.\nCommand: pnpm tsc --noEmit\nWhat next?",
            "context": {"recent_actions": []},
        },
    }
    bad = json.dumps({"tool_name": "Bash", "input": {"command": "pnpm tsc --noEmit"}})
    good = json.dumps({"tool_name": "Read", "input": {"file_path": "tsconfig.json"}})
    assert verify(task, bad)["score"] == 0.0
    assert verify(task, good)["score"] == 1.0


# ---- Tier 2 (mocked) ----

def _fake_chat_high(**kw): return '{"score": 0.9, "rationale": "good answer"}'
def _fake_chat_low(**kw): return '{"score": 0.1, "rationale": "off topic"}'
def _fake_chat_garbage(**kw): return "I cannot comply"


def test_judge_high_score_mocked():
    out = verify_lm_judge({"verifier_kind": "lm_judge", "description": "Summarize."},
                          "Short faithful summary.", chat_fn=_fake_chat_high)
    assert out["tier"] == 2 and 0.85 <= out["score"] <= 1.0
    assert out["signal"] == "judge_score" and out["details"]["rationale"]


def test_judge_low_score_mocked():
    out = verify_lm_judge({"verifier_kind": "lm_judge", "description": "X"},
                          "irrelevant", chat_fn=_fake_chat_low)
    assert out["score"] == pytest.approx(0.1, abs=1e-6)


def test_judge_unparseable_returns_zero():
    out = verify_lm_judge({"verifier_kind": "lm_judge", "description": "X"},
                          "irrelevant", chat_fn=_fake_chat_garbage)
    assert out["score"] == 0.0


# ---- Tier 3 ----

def test_shell_echo_pass():
    out = verify({"verifier_kind": "shell_exec",
                  "verifier_spec": {"expected_exit_code": 0, "stdout_contains": "hello"}},
                 "echo hello")
    assert out["tier"] == 3 and out["score"] == 1.0 and out["details"]["exit_code"] == 0


def test_shell_echo_passes_with_partial_match_when_pattern_misses():
    out = verify({"verifier_kind": "shell_exec",
                  "verifier_spec": {"expected_exit_code": 0,
                                    "stdout_pattern": r"^never_appears$"}},
                 "echo bad")
    assert 0.0 < out["score"] < 1.0
    assert out["details"]["exit_match"] is True
    assert out["details"]["pattern_match"] is False


def test_shell_refuses_rm_rf_root():
    out = verify({"verifier_kind": "shell_exec",
                  "verifier_spec": {"expected_exit_code": 0}}, "rm -rf /")
    assert out["score"] == 0.0 and out["signal"] == "refused"
    assert "dangerous" in out["details"]["reason"]


def test_shell_refuses_curl():
    out = verify_shell_exec({"verifier_kind": "shell_exec", "verifier_spec": {}},
                            "curl https://evil.example")
    assert out["signal"] == "refused"


# ---- Tier 4 ----

def test_composite_regex_plus_shell():
    task = {"verifier_kind": "composite", "verifier_spec": {"verifiers": [
        {"kind": "regex", "weight": 1.0, "spec": {"pattern": r"^echo\s+hi$"}},
        {"kind": "shell_exec", "weight": 2.0,
         "spec": {"expected_exit_code": 0, "stdout_contains": "hi"}}]}}
    out = verify(task, "echo hi")
    assert out["tier"] == 4 and out["score"] == 1.0 and out["signal"] == "composite_ok"
    assert len(out["details"]["sub_results"]) == 2


def test_composite_partial_score():
    task = {"verifier_kind": "composite", "verifier_spec": {"verifiers": [
        {"kind": "regex", "weight": 1.0, "spec": {"pattern": r"WONT_MATCH"}},
        {"kind": "shell_exec", "weight": 1.0,
         "spec": {"expected_exit_code": 0, "stdout_contains": "hi"}}]}}
    out = verify(task, "echo hi")
    assert 0.0 < out["score"] < 1.0


# ---- Registry ----

def test_registry_covers_all_kinds():
    expected = {"regex", "exact_match", "structural_json", "tool_name_match",
                "tool_family_match", "lm_judge", "shell_exec", "composite"}
    assert expected.issubset(set(KIND_TO_VERIFIER.keys()))


def test_unknown_kind_reports_score_zero():
    out = verify({"verifier_kind": "no_such_kind"}, "x")
    assert out["score"] == 0.0 and out["signal"] == "unknown_kind"
