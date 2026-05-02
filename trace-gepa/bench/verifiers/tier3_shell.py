"""Tier 3: sandboxed shell exec. Whitelist + dangerous-pattern refusal + 30s timeout."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from typing import Any

_TIMEOUT_S = 30
_OUT_TRUNC = 4000

_WHITELIST = {"ls", "wc", "cat", "head", "tail", "grep", "rg", "find", "echo",
              "printf", "test", "[", "mkdir", "touch", "awk", "sed", "true", "false"}

_DANGEROUS_RE = re.compile(
    r"rm\s+-rf\s+/|:\(\)\s*\{\s*:\|:&\s*\};:|dd\s+[^|]*of=/dev/|>\s*/dev/sd[a-z]|"
    r"\bmkfs\b|\bcurl\b|\bwget\b|\bnc\b|\bnetcat\b|\bssh\b|\bscp\b",
    re.IGNORECASE,
)
_ABS_PATH_RE = re.compile(r"(?<!\w)(/(?:etc|root|home|var|usr|bin|sbin|dev|proc|sys)\b)")


def _refuse(reason: str) -> dict:
    return {"score": 0.0, "tier": 3, "signal": "refused", "details": {"reason": reason}}


def _scan_command(cmd: str) -> tuple[bool, str]:
    if not cmd or not cmd.strip():
        return False, "empty command"
    if _DANGEROUS_RE.search(cmd):
        return False, "dangerous pattern"
    if _ABS_PATH_RE.search(cmd):
        return False, "absolute system path reference"
    if ".." in cmd.split():
        return False, "parent-dir traversal"
    for seg in re.split(r"\|\||&&|;|\||&", cmd):
        seg = seg.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError as e:
            return False, f"shlex error: {e}"
        if not tokens:
            continue
        head = tokens[0]
        if "=" in head and not head.startswith("="):
            return False, "env assignment at command head"
        if head not in _WHITELIST:
            return False, f"non-whitelisted command: {head}"
    return True, ""


def _run(cmd: str, cwd: str) -> tuple[int, str, str, bool]:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": cwd, "LANG": "C", "LC_ALL": "C"}
    try:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=_TIMEOUT_S)
        return proc.returncode, proc.stdout or "", proc.stderr or "", False
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        return 124, out or "", "timeout", True


def verify_shell_exec(task: dict, predicted: Any) -> dict:
    spec = task.get("verifier_spec") or {}
    cmd = predicted if isinstance(predicted, str) else (
        (predicted or {}).get("input") if isinstance(predicted, dict) else "")
    cmd = (cmd or "").strip()
    ok, why = _scan_command(cmd)
    if not ok:
        return _refuse(why)

    expected_exit = int(spec.get("expected_exit_code", 0))
    stdout_pattern = spec.get("stdout_pattern")
    stdout_contains = spec.get("stdout_contains")

    with tempfile.TemporaryDirectory(prefix="bench_t3_") as tmp:
        for fname, content in (spec.get("seed_files") or {}).items():
            with open(os.path.join(tmp, os.path.basename(fname)), "w") as fh:
                fh.write(content)
        rc, out, err, timed_out = _run(cmd, cwd=tmp)

    exit_ok = rc == expected_exit
    pattern_ok = re.search(stdout_pattern, out) is not None if stdout_pattern else True
    contains_ok = (stdout_contains in out) if stdout_contains is not None else True
    components = [exit_ok, pattern_ok, contains_ok]
    score = sum(1.0 for c in components if c) / len(components)
    signal = "shell_ok" if score == 1.0 else ("shell_partial" if score > 0 else "shell_fail")

    return {"score": score, "tier": 3, "signal": signal,
            "details": {"exit_code": rc, "expected_exit_code": expected_exit,
                        "exit_match": exit_ok, "pattern_match": pattern_ok,
                        "contains_match": contains_ok, "timed_out": timed_out,
                        "stdout": out[:_OUT_TRUNC], "stderr": err[:_OUT_TRUNC]}}
