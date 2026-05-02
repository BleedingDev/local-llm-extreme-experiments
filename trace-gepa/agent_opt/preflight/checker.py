"""Deterministic preflight predicates. Each fn returns (ok: bool, msg: str).

`action` is a dict shaped like dataset_v2 `observed_action`:
    {"name": "Bash"|"Read"|"Edit"|"Skill"|..., "input": "<json-string>" | dict, ...}

`context` carries (optional): recent_actions: list[str], available_skills: list[str].
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from typing import Any


# ---- helpers ---------------------------------------------------------------

def _parse_input(action: dict) -> dict:
    """Action.input may be a JSON string or already-parsed dict."""
    raw = action.get("input")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


_BUILTINS = {
    # shell builtins / loop keywords / common utilities; skip PATH check
    "cd", "echo", "set", "unset", "export", "true", "false", ":", "[", "test",
    "exit", "return", "source", ".", "eval", "exec", "trap", "alias", "unalias",
    "pwd", "read", "shift", "wait", "kill", "type", "command", "let", "local",
    "printf", "for", "while", "until", "if", "case", "do", "done", "fi", "esac",
    "then", "else", "elif", "function", "in", "select", "time",
}


def _first_token(cmd: str) -> str:
    """First executable token (skip `cd dir &&`, env assignments, shebang-pipes)."""
    s = (cmd or "").strip()
    if not s:
        return ""
    # peel `cd <dir> && ...`
    m = re.match(r"^cd\s+\S+\s*&&\s*(.*)$", s)
    if m:
        s = m.group(1)
    # split off the first command in a pipeline / sequence
    # Cheap split — we only need the leading bin name.
    s = re.split(r"[|&;\n]", s, maxsplit=1)[0].strip()
    if not s:
        return ""
    try:
        toks = shlex.split(s, posix=True)
    except ValueError:
        toks = s.split()
    # skip leading VAR=val tokens
    for t in toks:
        if "=" in t and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            continue
        return t
    return ""


# ---- predicates ------------------------------------------------------------

def check_cmd_exists(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Bash: leading command must resolve via shutil.which()."""
    if action.get("name") != "Bash":
        return True, ""
    cmd = _parse_input(action).get("command", "")
    bin_ = _first_token(cmd)
    if not bin_ or bin_ in _BUILTINS:
        return True, ""
    # project-relative tooling (resolved against the caller's cwd, which we
    # don't know in replay). Defer judgement to runtime.
    if bin_.startswith("node_modules/") or "/node_modules/" in bin_:
        return True, ""
    # absolute / explicit-relative path: trust if it exists
    if bin_.startswith("/") or bin_.startswith("./") or bin_.startswith("~"):
        path = os.path.expanduser(bin_)
        if os.path.exists(path):
            return True, ""
        return False, f"command not in PATH: {bin_}"
    if shutil.which(bin_) is None:
        return False, f"command not in PATH: {bin_}"
    return True, ""


def check_path_exists(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Read/Edit: file_path must exist on disk."""
    if action.get("name") not in ("Read", "Edit"):
        return True, ""
    p = _parse_input(action).get("file_path")
    if not p:
        return True, ""
    if not os.path.exists(os.path.expanduser(p)):
        return False, f"path does not exist: {p}"
    return True, ""


def check_edit_unique(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Edit: old_string must occur exactly once in file (skip if file unreadable)."""
    if action.get("name") != "Edit":
        return True, ""
    inp = _parse_input(action)
    p = inp.get("file_path")
    old = inp.get("old_string")
    if not p or not old or inp.get("replace_all"):
        return True, ""
    path = os.path.expanduser(p)
    if not os.path.exists(path):
        return True, ""  # path-exists predicate handles this
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception:
        return True, ""  # don't double-block on IO errors
    n = src.count(old)
    if n == 0:
        return False, "old_string not found in file — Edit will fail"
    if n > 1:
        return False, f"old_string occurs {n}>1 times — Edit will fail"
    return True, ""


def check_file_was_read(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Edit: a Read of file_path must appear in recent_actions."""
    if action.get("name") != "Edit":
        return True, ""
    p = _parse_input(action).get("file_path")
    if not p:
        return True, ""
    recent = (context or {}).get("recent_actions") or []
    needle = f'"file_path":"{p}"'
    needle_sp = f'"file_path": "{p}"'
    for entry in recent:
        s = entry if isinstance(entry, str) else json.dumps(entry)
        if not s.startswith("Read"):
            # also accept dict-form {"name":"Read",...}
            if not (isinstance(entry, dict) and entry.get("name") == "Read"):
                continue
        if needle in s or needle_sp in s:
            return True, ""
    return False, "file not read in this session — Edit will fail"


def check_skill_listed(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Skill: skill name must appear in available_skills inventory."""
    if action.get("name") != "Skill":
        return True, ""
    name = _parse_input(action).get("skill")
    if not name:
        return True, ""
    inv = (context or {}).get("available_skills") or []
    # tolerate "plugin:skill" form on either side of the comparison
    if name in inv:
        return True, ""
    if ":" in name and name.split(":", 1)[1] in inv:
        return True, ""
    for entry in inv:
        if ":" in entry and (entry.split(":", 1)[1] == name or entry.endswith(":" + name)):
            return True, ""
    return False, f"skill not in inventory: {name}"


_FIND_UNBOUNDED = re.compile(r"(^|[\s|;&(])find\s+(/|~)(\s|$)")
_FIND_SCOPING_FLAGS = re.compile(r"\s-(path|name|iname|regex|prune|maxdepth|mindepth)\b")


def check_parallel_safety(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Bash: unbounded `find /` or `find ~` (no scoping flags) — likely Exit 141."""
    if action.get("name") != "Bash":
        return True, ""
    cmd = _parse_input(action).get("command", "") or ""
    if not _FIND_UNBOUNDED.search(cmd):
        return True, ""
    # if any scoping flag is present (-path, -name, -maxdepth, ...) treat as bounded
    if _FIND_SCOPING_FLAGS.search(cmd):
        return True, ""
    return False, "unbounded find on root — likely Exit 141"


_LS_BAD = re.compile(r"(^|[\s|;&(])ls\s+:")


def check_cmd_args(action: dict, context: dict | None = None) -> tuple[bool, str]:
    """Bash: `ls :flag` — colon-prefixed token treated as path. Conservative typo check."""
    if action.get("name") != "Bash":
        return True, ""
    cmd = _parse_input(action).get("command", "") or ""
    if _LS_BAD.search(cmd):
        return False, f"ls flag treated as path: {cmd[:80]}"
    return True, ""


# ---- aggregator ------------------------------------------------------------

PREDICATES = (
    ("cmd_exists", check_cmd_exists),
    ("path_exists", check_path_exists),
    ("edit_unique", check_edit_unique),
    ("file_was_read", check_file_was_read),
    ("skill_listed", check_skill_listed),
    ("parallel_safety", check_parallel_safety),
    ("cmd_args", check_cmd_args),
)


class PreflightChecker:
    """Aggregates predicates and returns a structured veto result."""

    def __init__(self, predicates=PREDICATES) -> None:
        self.predicates = predicates

    def check(self, action: dict, context: dict | None = None) -> dict[str, Any]:
        """Returns {passed: bool, blocked_by: list[str], warnings: list[str], fired: list[str]}."""
        context = context or {}
        blocked: list[str] = []
        warns: list[str] = []
        fired: list[str] = []
        for label, pred in self.predicates:
            try:
                ok, msg = pred(action, context)
            except Exception as e:  # never let a predicate take down the dispatcher
                warns.append(f"{label}: predicate error {e!r}")
                continue
            if not ok:
                fired.append(label)
                blocked.append(msg)
        return {
            "passed": not blocked,
            "blocked_by": blocked,
            "warnings": warns,
            "fired": fired,
        }
