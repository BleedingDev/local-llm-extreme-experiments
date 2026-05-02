"""Preflight deterministic predicate checker (proposal J).

Pure-Python pre-tool-call veto layer. No LM calls.
"""
from .checker import (
    PreflightChecker,
    check_cmd_exists,
    check_path_exists,
    check_edit_unique,
    check_file_was_read,
    check_skill_listed,
    check_parallel_safety,
    check_cmd_args,
)

__all__ = [
    "PreflightChecker",
    "check_cmd_exists",
    "check_path_exists",
    "check_edit_unique",
    "check_file_was_read",
    "check_skill_listed",
    "check_parallel_safety",
    "check_cmd_args",
]
