"""Multi-tier verifier suite. Exports `verify` and `KIND_TO_VERIFIER`."""
from __future__ import annotations

from typing import Any, Callable

from .composite import verify_composite
from .tier1_regex import (verify_exact_match, verify_regex, verify_structural_json,
                          verify_tool_family_match, verify_tool_name_match)
from .tier2_judge import verify_lm_judge
from .tier3_shell import verify_shell_exec

KIND_TO_VERIFIER: dict[str, Callable[..., dict]] = {
    "regex": verify_regex,
    "exact_match": verify_exact_match,
    "structural_json": verify_structural_json,
    "tool_name_match": verify_tool_name_match,
    "tool_family_match": verify_tool_family_match,
    "lm_judge": verify_lm_judge,
    "shell_exec": verify_shell_exec,
    "composite": verify_composite,
}


def _dispatch(kind: str, task: dict, predicted: Any) -> dict:
    fn = KIND_TO_VERIFIER.get(kind)
    if fn is None:
        return {"score": 0.0, "tier": 0, "signal": "unknown_kind", "details": {"kind": kind}}
    return fn(task, predicted)


def verify(task: dict, predicted: Any) -> dict:
    """Top-level verifier. Reads task['verifier_kind'] and dispatches to a tier."""
    return _dispatch(task.get("verifier_kind") or "exact_match", task, predicted)


__all__ = ["verify", "KIND_TO_VERIFIER"]
