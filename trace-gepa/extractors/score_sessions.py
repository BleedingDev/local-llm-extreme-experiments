#!/usr/bin/env python3
"""Score CC + Codex sessions by error/correction signal density.

Writes ``data/v2_manifest_proposal.json`` with the top-N sessions per source.
Score heuristic mirrors v1: tool errors + user corrections + nonzero exit codes
+ subagent dispatches. Cheap pass-once over the JSONL files; no LM calls.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2_manifest_proposal.json"

CC_DIR = Path("/Users/satan/.claude/projects")
CODEX_DIR = Path("/Users/satan/.codex/sessions")

CC_TOP = 199
CODEX_TOP = 1
EXIT_RE = re.compile(r"Process exited with code\s+(-?\d+)")
CORRECT_PREFIXES = ("no", "stop", "don't", "dont", "wrong", "actually")


def score_cc(path: Path) -> tuple[int, int]:
    score = 0
    events = 0
    try:
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events += 1
                try:
                    ev = orjson.loads(line)
                except Exception:
                    continue
                et = ev.get("type")
                if et == "user":
                    msg = ev.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, list):
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "tool_result":
                                if b.get("is_error"):
                                    score += 1
                    elif isinstance(content, str):
                        low = content.strip().lower()
                        if any(low.startswith(p) for p in CORRECT_PREFIXES):
                            score += 2
                elif et == "assistant":
                    msg = ev.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, list):
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "tool_use":
                                if b.get("name") in ("Agent", "Task"):
                                    score += 1
    except Exception:
        return 0, 0
    return score, events


def score_codex(path: Path) -> tuple[int, int]:
    score = 0
    events = 0
    try:
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events += 1
                try:
                    ev = orjson.loads(line)
                except Exception:
                    continue
                etype = ev.get("type")
                payload = ev.get("payload") or {}
                ptype = payload.get("type")
                if etype == "response_item" and ptype == "function_call_output":
                    out = payload.get("output") or ""
                    m = EXIT_RE.search(out)
                    if m:
                        try:
                            if int(m.group(1)) != 0:
                                score += 1
                        except ValueError:
                            pass
                elif etype == "event_msg" and ptype == "user_message":
                    msg = (payload.get("message") or "").strip().lower()
                    if any(msg.startswith(p) for p in CORRECT_PREFIXES):
                        score += 2
    except Exception:
        return 0, 0
    return score, events


def collect(root: Path, scorer, label: str, top_n: int) -> list[dict]:
    candidates: list[dict] = []
    files = list(root.rglob("*.jsonl"))
    print(f"[{label}] scoring {len(files)} files", file=sys.stderr)
    for i, fp in enumerate(files):
        if i % 500 == 0 and i:
            print(f"[{label}] {i}/{len(files)}", file=sys.stderr)
        s, ev = scorer(fp)
        if s > 0 and ev >= 20:
            candidates.append({"path": str(fp), "events": ev, "score": s})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def main() -> int:
    cc_top = collect(CC_DIR, score_cc, "cc", CC_TOP)
    codex_top = collect(CODEX_DIR, score_codex, "codex", CODEX_TOP)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "$schema_version": 2,
        "_note": "Top sessions ranked by error+correction+dispatch signals.",
        "cc_sessions": cc_top,
        "codex_sessions": codex_top,
    }, indent=2))
    print(f"wrote {OUT} cc={len(cc_top)} codex={len(codex_top)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
