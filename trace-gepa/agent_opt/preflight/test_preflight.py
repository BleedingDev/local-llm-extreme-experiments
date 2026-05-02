"""Unit tests for preflight predicates. Run: pytest -q"""
from __future__ import annotations

import json
import os
import tempfile

from agent_opt.preflight.checker import (
    PreflightChecker,
    check_cmd_args,
    check_cmd_exists,
    check_edit_unique,
    check_file_was_read,
    check_parallel_safety,
    check_path_exists,
    check_skill_listed,
)


def _act(name: str, inp: dict) -> dict:
    return {"name": name, "input": json.dumps(inp)}


# ---- check_cmd_exists ----

def test_cmd_exists_ok():
    ok, _ = check_cmd_exists(_act("Bash", {"command": "ls -la /tmp"}))
    assert ok


def test_cmd_exists_fail():
    ok, msg = check_cmd_exists(_act("Bash", {"command": "definitely_not_a_binary_xyz --foo"}))
    assert not ok and "not in PATH" in msg


def test_cmd_exists_skips_cd_prefix():
    ok, _ = check_cmd_exists(_act("Bash", {"command": "cd /tmp && ls"}))
    assert ok


def test_cmd_exists_ignores_builtin():
    ok, _ = check_cmd_exists(_act("Bash", {"command": "echo hi"}))
    assert ok


# ---- check_path_exists ----

def test_path_exists_ok():
    ok, _ = check_path_exists(_act("Read", {"file_path": "/tmp"}))
    assert ok


def test_path_exists_fail():
    ok, msg = check_path_exists(_act("Read", {"file_path": "/no/such/path/here_xyz_123"}))
    assert not ok and "does not exist" in msg


# ---- check_edit_unique ----

def test_edit_unique_ok():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("alpha\nbeta\ngamma\n")
        path = f.name
    try:
        ok, _ = check_edit_unique(_act("Edit", {"file_path": path, "old_string": "beta"}))
        assert ok
    finally:
        os.unlink(path)


def test_edit_unique_multi():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("dup\ndup\ndup\n")
        path = f.name
    try:
        ok, msg = check_edit_unique(_act("Edit", {"file_path": path, "old_string": "dup"}))
        assert not ok and "occurs 3" in msg
    finally:
        os.unlink(path)


def test_edit_unique_zero():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write("only this\n")
        path = f.name
    try:
        ok, msg = check_edit_unique(_act("Edit", {"file_path": path, "old_string": "missing"}))
        assert not ok and "not found" in msg
    finally:
        os.unlink(path)


# ---- check_file_was_read ----

def test_file_was_read_ok():
    ctx = {"recent_actions": ['Read: {"file_path":"/tmp/x.txt"}']}
    ok, _ = check_file_was_read(_act("Edit", {"file_path": "/tmp/x.txt", "old_string": "y"}), ctx)
    assert ok


def test_file_was_read_missing():
    ctx = {"recent_actions": ['Read: {"file_path":"/tmp/other.txt"}']}
    ok, msg = check_file_was_read(_act("Edit", {"file_path": "/tmp/x.txt", "old_string": "y"}), ctx)
    assert not ok and "not read" in msg


# ---- check_skill_listed ----

def test_skill_listed_ok():
    ok, _ = check_skill_listed(_act("Skill", {"skill": "loop"}), {"available_skills": ["loop", "review"]})
    assert ok


def test_skill_listed_unknown():
    ok, msg = check_skill_listed(_act("Skill", {"skill": "warpdrive"}), {"available_skills": ["loop"]})
    assert not ok and "not in inventory" in msg


# ---- check_parallel_safety ----

def test_parallel_safety_blocks_find_root():
    # bare `find /` with no scoping flags
    ok, msg = check_parallel_safety(_act("Bash", {"command": "find / -type f"}))
    assert not ok and "unbounded find" in msg


def test_parallel_safety_allows_scoped_path():
    ok, _ = check_parallel_safety(_act("Bash", {"command": "find /tmp -name foo"}))
    assert ok


def test_parallel_safety_allows_find_root_with_name_filter():
    # `-name`/`-path` scoping makes a root-level find tractable
    ok, _ = check_parallel_safety(_act("Bash", {"command": "find / -name 'pylate*.py' 2>/dev/null | head"}))
    assert ok


# ---- check_cmd_args ----

def test_cmd_args_blocks_ls_colon():
    ok, msg = check_cmd_args(_act("Bash", {"command": "ls :la /tmp"}))
    assert not ok and "ls flag" in msg


def test_cmd_args_allows_normal_ls():
    ok, _ = check_cmd_args(_act("Bash", {"command": "ls -la /tmp"}))
    assert ok


# ---- aggregator ----

def test_checker_passes_on_clean_action():
    result = PreflightChecker().check(_act("Bash", {"command": "echo hi"}), {})
    assert result["passed"] is True
    assert result["blocked_by"] == []


def test_checker_blocks_with_reason():
    result = PreflightChecker().check(_act("Bash", {"command": "find /"}), {})
    assert result["passed"] is False
    assert any("unbounded find" in m for m in result["blocked_by"])
    assert "parallel_safety" in result["fired"]
