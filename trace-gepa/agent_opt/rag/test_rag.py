"""Minimal tests for trace-rag.

Tests build_query_text deterministically (no model load) and a tiny end-to-end
build+query cycle when sentence-transformers is available.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from . import embed
from .embed import build_query_text


def test_build_query_text_basic():
    rec = {
        "context": {
            "user_request": "find broken tests",
            "recent_actions": [
                {"kind": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                {"kind": "tool_use", "name": "Read", "input": {"file_path": "/a/b"}},
                {"kind": "tool_use", "name": "Edit", "input": {"file_path": "/c"}},
            ],
        }
    }
    text = build_query_text(rec)
    assert text.startswith("find broken tests")
    # Only last 2 actions kept.
    assert "Bash" not in text
    assert "Read" in text
    assert "Edit" in text


def test_build_query_text_empty():
    assert build_query_text({}) == "(empty)"
    assert build_query_text({"context": {}}) == "(empty)"


def test_build_and_query_smoke(tmp_path):
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        pytest.skip("sentence-transformers not installed")

    ds = tmp_path / "tiny.jsonl"
    rows = [
        {"id": "r1", "src": "test", "context": {"user_request": "edit a typescript test", "recent_actions": []}, "observed_action": {"kind": "tool_use", "name": "Edit"}, "label": "good"},
        {"id": "r2", "src": "test", "context": {"user_request": "run bash command", "recent_actions": []}, "observed_action": {"kind": "tool_use", "name": "Bash"}, "label": "good"},
        {"id": "r3", "src": "test", "context": {"user_request": "write a python file", "recent_actions": []}, "observed_action": {"kind": "tool_use", "name": "Write"}, "label": "good"},
    ]
    with open(ds, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    out = tmp_path / "idx"
    info = embed.build_index([str(ds)], str(out))
    assert info["n_records"] == 3
    assert info["dim"] > 0
    assert (out / "embeddings.npz").exists()
    assert (out / "metadata.jsonl").exists()

    from .index import TraceIndex
    idx = TraceIndex(str(out))
    hits = idx.query("typescript test files", k=3)
    assert len(hits) == 3
    # First hit should be the typescript record.
    assert hits[0]["record"]["id"] == "r1"
