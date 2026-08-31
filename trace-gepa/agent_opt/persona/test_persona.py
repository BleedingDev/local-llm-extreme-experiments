"""Smoke tests for persona fingerprint + prefix wiring.

Run with: python -m pytest agent_opt/persona/test_persona.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_opt.persona import fingerprint as fp_mod
from agent_opt.persona.prefix import build_persona_prefix, inject_persona

PROFILE = Path(__file__).resolve().parent / "profile.json"


def test_profile_artifact_present():
    assert PROFILE.exists(), "run `python -m agent_opt.persona.fingerprint` first"
    p = json.loads(PROFILE.read_text())
    for k in (
        "summary",
        "tool_histogram_top10",
        "bash_verb_top20",
        "path_histogram",
        "language_signals",
        "repo_top5",
        "recovery_top5",
    ):
        assert k in p, f"missing key {k}"
    assert p["summary"]["n_records"] > 1000


def test_bash_verb_extraction():
    # `cd X && git status` -> verb should be `git`, not `cd`.
    assert fp_mod._leading_verb("cd /tmp && git status") == "git"
    # Pipe -> first verb wins.
    assert fp_mod._leading_verb("rg foo | head") == "rg"
    # No chain.
    assert fp_mod._leading_verb("bun test --watch") == "bun"


def test_path_bucket():
    inp = '{"file_path":"/Users/satan/side/experiments/ir-expo/src/x.ts"}'
    bucket = None
    for pb in fp_mod.PATH_PREFIXES:
        if pb in inp:
            bucket = pb
            break
    assert bucket == "/Users/satan/side/experiments/"


def test_prefix_shape_and_length():
    profile = json.loads(PROFILE.read_text())
    prefix = build_persona_prefix(profile)
    # Acceptable band per spec: <= 500 chars, with a non-trivial floor.
    assert 200 <= len(prefix) <= 500, f"unexpected length {len(prefix)}"
    # Must mention the user's signature path-prefix and at least one Czech token.
    assert "/Users/satan/side/experiments/" in prefix
    czech_tokens = [t for t, _ in profile["language_signals"]["czech_token_counts"][:3]]
    assert any(tok in prefix for tok in czech_tokens), "no Czech token surfaced"


def test_prefix_matches_actual_data_not_invented():
    """Regression: prefix must reflect the real histogram (pnpm/grep dominate),
    NOT invented preferences for bun/rg.
    """
    profile = json.loads(PROFILE.read_text())
    prefix = build_persona_prefix(profile)
    # grep dominates rg ~20:1 in the data; prefix must surface grep.
    assert "grep" in prefix, "prefix should mention grep (top-3 verb)"
    # bun does not appear in top-20 verbs; rg only ranks 12th.
    # The prefix must NOT claim a preference for bun or rg over alternatives.
    assert "bun" not in prefix, "prefix invented a 'bun' preference not in data"
    assert "rg over" not in prefix, "prefix invented an 'rg over grep' preference"
    # Top-3 bash verbs from the data: git, zig, grep.
    for verb in ("git", "zig", "grep"):
        assert verb in prefix, f"missing top-3 verb {verb!r}"


def test_prefix_max_chars_param():
    profile = json.loads(PROFILE.read_text())
    short = build_persona_prefix(profile, max_chars=200)
    assert len(short) <= 200


def test_inject_persona_prepends():
    profile = json.loads(PROFILE.read_text())
    base = "BASE SYSTEM PROMPT BODY"
    out = inject_persona(base, profile)
    assert out.startswith("PERSONA NOTES")
    assert out.endswith(base)
    # Empty base should still produce a prefix-only string.
    assert inject_persona("", profile).startswith("PERSONA NOTES")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
