#!/usr/bin/env python3
"""Extract a synthetic planner dataset from CC + Codex seed sessions.

For each session: take the first non-command user message as `user_request`,
then bucket the next ~200 tool-use events into ~3-10 issue-like units. Buckets
split on TaskCreate when present; else fixed-size time slices.
Output: trace-gepa/data/planner_dataset.jsonl
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
SEED = ROOT / "data" / "seed_sessions.json"
OUT = ROOT / "data" / "planner_dataset.jsonl"

EVENT_WINDOW, MAX_ISSUES, TIME_BUCKET = 200, 10, 25
NOISY = {"TodoWrite", "TaskUpdate", "TaskList", "TaskGet"}
REDACT = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")
VERIFIER_KW = ("test", "pytest", "tsc", "build", "lint", "check")
PATH_EXTS = (".ts", ".py", ".rs", ".js", ".tsx", ".md", ".json")


def _trunc(s: str, n: int) -> str:
    return s if not s or len(s) <= n else s[:n] + "..."


def _redact(s: str, n: int = 1500) -> str:
    return _trunc(REDACT.sub("<REDACTED_KEY>", s or ""), n)


def _is_cmd(t: str) -> bool:
    t = (t or "").strip()
    return not t or t.startswith(("<command-", "<local-command-", "[Request interrupted"))


def _cc_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _bucket(actions: list[dict]) -> list[list[dict]]:
    if not actions:
        return []
    bnd = [i for i, a in enumerate(actions) if a["name"] == "TaskCreate"]
    if bnd:
        out, last = [], 0
        for b in bnd:
            if b > last:
                out.append(actions[last:b])
            last = b
        if last < len(actions):
            out.append(actions[last:])
        return [b for b in out if b][:MAX_ISSUES]
    return [actions[i:i + TIME_BUCKET] for i in range(0, len(actions), TIME_BUCKET)][:MAX_ISSUES]


def _bash_str(inp: Any) -> str:
    if isinstance(inp, str):
        return inp
    if isinstance(inp, dict):
        return inp.get("command") or json.dumps(inp)[:200]
    return str(inp)[:200]


def _to_issue(bucket: list[dict], idx: int) -> dict:
    files: list[str] = []; seen: set[str] = set(); bash: list[str] = []; title = ""
    for a in bucket:
        f = a.get("file") or ""
        if f and f not in seen:
            seen.add(f); files.append(f)
        if a["name"] == "Bash":
            cmd = _trunc(_bash_str(a.get("input")), 200)
            if cmd:
                bash.append(cmd)
        if a["name"] == "TaskCreate" and isinstance(a.get("input"), dict):
            title = a["input"].get("description") or a["input"].get("title") or title
    if not title:
        title = f"Modify {Path(files[0]).name}" if files else (f"Run {_trunc(bash[0],60)}" if bash else f"Issue {idx+1}")
    verifier = [c for c in bash if any(k in c.lower() for k in VERIFIER_KW)] or bash[:2]
    return {"title": _trunc(title, 120), "expectedFiles": files[:16], "verifierCommands": verifier[:4]}


def _read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_bytes().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(orjson.loads(line))
        except Exception:
            pass
    return out


def _cc_first_user(events: list[dict]) -> tuple[int, str]:
    for i, ev in enumerate(events):
        if ev.get("type") != "user":
            continue
        msg = ev.get("message") or {}
        if msg.get("role") != "user":
            continue
        c = msg.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            continue
        txt = _cc_text(c) if isinstance(c, list) else (c if isinstance(c, str) else "")
        if not _is_cmd(txt):
            return i, txt
    return -1, ""


def _codex_first_user(events: list[dict]) -> tuple[int, str]:
    for i, ev in enumerate(events):
        if ev.get("type") != "event_msg":
            continue
        p = ev.get("payload") or {}
        if p.get("type") != "user_message":
            continue
        msg = p.get("message", "")
        if not _is_cmd(msg):
            return i, msg
    return -1, ""


def _build_record(path: str, src: str, sid: str, req: str, actions: list[dict]) -> dict | None:
    buckets = _bucket(actions)
    if not buckets:
        return None
    return {"id": f"planner_{src}_{sid}", "user_request": _redact(req),
            "ground_truth_issues": [_to_issue(b, i) for i, b in enumerate(buckets)],
            "src": src, "src_path": path}


def process_cc(path: str) -> dict | None:
    events = _read_jsonl(path)
    if not events:
        return None
    start, req = _cc_first_user(events)
    if start < 0:
        return None
    actions: list[dict] = []
    for ev in events[start: start + EVENT_WINDOW]:
        if ev.get("type") != "assistant":
            continue
        c = (ev.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for blk in c:
            if not isinstance(blk, dict) or blk.get("type") != "tool_use":
                continue
            name = blk.get("name", "")
            if name in NOISY - {"TaskCreate"}:
                continue
            inp = blk.get("input") or {}
            fp = (inp.get("file_path") or inp.get("path") or "") if isinstance(inp, dict) else ""
            actions.append({"name": name, "input": inp, "file": fp})
    sid = (re.search(r"([0-9a-f]{8})-", path) or [None, "unknown0"])[1][:8]
    return _build_record(path, "cc", sid, req, actions)


def process_codex(path: str) -> dict | None:
    events = _read_jsonl(path)
    if not events:
        return None
    start, req = _codex_first_user(events)
    if start < 0:
        return None
    actions: list[dict] = []
    for ev in events[start: start + EVENT_WINDOW]:
        if ev.get("type") != "response_item":
            continue
        p = ev.get("payload") or {}
        if p.get("type") != "function_call":
            continue
        args = p.get("arguments") or ""
        fp = ""
        try:
            d = json.loads(args)
            if isinstance(d, dict):
                fp = d.get("path") or d.get("file_path") or ""
                if not fp and isinstance(d.get("command"), list):
                    for tok in d["command"]:
                        if isinstance(tok, str) and ("/" in tok or tok.endswith(PATH_EXTS)):
                            fp = tok; break
        except Exception:
            pass
        actions.append({"name": p.get("name") or "", "input": args, "file": fp})
    sid = Path(path).stem.split("-")[-1][:8]
    return _build_record(path, "codex", sid, req, actions)


def main() -> int:
    seed = orjson.loads(SEED.read_bytes())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cnt: Counter = Counter(); counts: list[int] = []
    with OUT.open("wb") as f:
        for src, key, fn in (("cc", "cc_sessions", process_cc), ("codex", "codex_sessions", process_codex)):
            for sess in seed.get(key, []):
                r = fn(sess["path"])
                if r is None:
                    cnt[f"skipped_{src}"] += 1; continue
                f.write(orjson.dumps(r)); f.write(b"\n")
                cnt[src] += 1; counts.append(len(r["ground_truth_issues"]))
    total = cnt["cc"] + cnt["codex"]
    avg = sum(counts) / len(counts) if counts else 0.0
    print(f"Total planner records: {total}  (cc={cnt['cc']} codex={cnt['codex']})")
    print(f"  skipped: cc={cnt['skipped_cc']} codex={cnt['skipped_codex']}")
    print(f"  avg issues / record: {avg:.2f}")
    print(f"Output: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
