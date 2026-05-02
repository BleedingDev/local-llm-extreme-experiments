from __future__ import annotations


def codex_only(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("src") == "codex"]


def bag_only(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("src") == "cc"]
