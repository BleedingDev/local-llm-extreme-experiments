"""Unit tests for the sanitisation pipeline."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from agent_opt.sanitise import (
    DEFAULT_RULES,
    CZECH_PROFANITY,
    sanitise_record,
    sanitise_file,
    _shannon_bits,
)


def _sanitise(s: str, **kw) -> str:
    audit: list = []
    counts: Counter = Counter()
    return sanitise_record(s, DEFAULT_RULES, counts, audit, **kw)


def test_path_users_satan_replaced() -> None:
    out = _sanitise("see /Users/satan/projects/foo/bar.py")
    assert out == "see /Users/USER/projects/foo/bar.py"


def test_tilde_satan_replaced() -> None:
    out = _sanitise("cd ~/satan/work")
    assert out == "cd ~/USER/work"


def test_anthropic_key_redacted() -> None:
    out = _sanitise("ANTHROPIC=sk-ant-api03-AbCdEf_xyz-123ABC")
    assert "sk-ant-" not in out
    assert "<REDACTED_KEY>" in out


def test_openai_key_redacted() -> None:
    out = _sanitise("OPENAI=sk-proj-AbCdEfGhIjKlMnOpQrStUv1234")
    assert "<REDACTED_KEY>" in out
    assert "sk-proj-AbCdEfGhIjKlMnOpQrStUv" not in out


def test_github_token_redacted() -> None:
    out = _sanitise("export GH=ghp_ABCDEFGHIJ1234567890abcd")
    assert "<REDACTED_GH_TOKEN>" in out
    assert "ghp_AB" not in out


def test_private_repo_renamed() -> None:
    out = _sanitise("clone github.com/BleedingDev/ferndesk-connector")
    assert "<PRIVATE_PROJECT_1>" in out
    assert "ferndesk-connector" not in out


def test_public_repo_preserved() -> None:
    # Public repos must NOT be redacted.
    s = "see ir-multivector-retrieval and effect-copilotx and local-llm-extreme-experiments"
    out = _sanitise(s)
    assert out == s


def test_recursive_dict_walk() -> None:
    rec = {
        "id": "abc",
        "context": {
            "user_request": "I work on /Users/satan/code/ferndesk-connector",
            "tags": ["sk-ant-AAAAbbbbCCCCdddd1234567890", "ok"],
        },
    }
    audit: list = []
    counts: Counter = Counter()
    out = sanitise_record(rec, DEFAULT_RULES, counts, audit)
    payload = json.dumps(out)
    assert "/Users/satan" not in payload
    assert "ferndesk-connector" not in payload
    assert "sk-ant-" not in payload
    assert counts["path_users_satan"] >= 1
    assert counts["private_repo_1"] >= 1
    assert counts["anthropic_key"] >= 1


def test_profanity_default_off() -> None:
    out = _sanitise("kurva to je teda věc")
    assert "kurva" in out


def test_profanity_scrub_on() -> None:
    out = _sanitise("kurva to je teda věc", scrub_profanity=True)
    assert "kurva" not in out
    assert "<EMPHATIC>" in out


def test_streaming_file(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    rows = [
        {"id": "r1", "text": "/Users/satan/foo"},
        {"id": "r2", "text": "ghp_ABCDEFGHIJKLMN0123456789xy"},
        {"id": "r3", "nested": {"v": "ferndesk-connector"}},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    audit: list = []
    n, counts, ib, ob = sanitise_file(src, dst, DEFAULT_RULES, audit)
    assert n == 3
    text = dst.read_text(encoding="utf-8")
    assert "/Users/satan" not in text
    assert "ghp_" not in text
    assert "ferndesk-connector" not in text


def test_entropy_helper() -> None:
    assert _shannon_bits("aaaaaa") < 1.0
    assert _shannon_bits("abcdefghijklmnopqrstuvwx") > 4.0


def test_czech_pattern_detects_common_word() -> None:
    assert CZECH_PROFANITY.search("kurva")
    assert CZECH_PROFANITY.search("Kurva")
    assert not CZECH_PROFANITY.search("hello world")
