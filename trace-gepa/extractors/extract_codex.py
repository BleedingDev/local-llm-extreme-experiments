#!/usr/bin/env python3
"""Extract a labelled GEPA training dataset from Codex JSONL seed sessions."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, deque
from pathlib import Path

import orjson

REPO_ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx")
SEED_PATH = REPO_ROOT / "trace-gepa" / "data" / "seed_sessions.json"
OUT_PATH = REPO_ROOT / "trace-gepa" / "data" / "codex_dataset.jsonl"

MAX_FIELD = 2048
MAX_RESULT_EXCERPT = 500
MAX_REASON = 500
MAX_PER_SESSION = 200
WINDOW = 5

REDACT_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
    re.compile(r"ghp_[A-Za-z0-9]+"),
    re.compile(r"hf_[A-Za-z0-9]+"),
]

CORRECT_PREFIXES = ("no", "stop", "don't", "dont", "wrong", "actually", "ne ", "počkej", "espera")
CONFIRM_TOKENS = ("great", "perfect", "thanks", "ok", "exactly")

EXIT_RE = re.compile(r"Process exited with code\s+(-?\d+)")


def redact(text: str) -> str:
    for pat in REDACT_PATTERNS:
        text = pat.sub("<REDACTED_KEY>", text)
    return text


def truncate(text: str, limit: int = MAX_FIELD) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def parse_exit_code(output: str) -> int | None:
    m = EXIT_RE.search(output or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def label_for(exit_code: int | None, output: str, next_user: str | None) -> tuple[str, str | None]:
    nu = (next_user or "").strip().lower()
    failure_category: str | None = None
    if exit_code is not None and exit_code != 0:
        if exit_code == 127:
            failure_category = "cmd_not_found_127"
        elif exit_code == 141:
            failure_category = "bash_timeout_141"
        else:
            failure_category = "bash_exit_nonzero"
    elif "patch_apply" in (output or "").lower() and ("failed" in (output or "").lower() or "error" in (output or "").lower()):
        failure_category = "edit_string_not_unique"

    if exit_code is not None and exit_code != 0:
        return "bad", failure_category
    if nu:
        if any(nu.startswith(p) for p in CORRECT_PREFIXES):
            return "user_corrected", failure_category or "user_correction"
        if any(tok in nu for tok in CONFIRM_TOKENS):
            return "user_confirmed", failure_category
    return "good", failure_category


def short_action(call: dict) -> dict:
    args = call.get("arguments") or ""
    return {
        "name": call.get("name") or "",
        "input_excerpt": truncate(redact(args), 600),
    }


def short_result(out: dict) -> dict:
    text = out.get("output") or ""
    return {
        "excerpt": truncate(redact(text), 600),
    }


def extract_session(path: str, src_idx: int) -> list[dict]:
    records: list[dict] = []
    user_msgs: deque[str] = deque(maxlen=WINDOW)
    recent_calls: deque[dict] = deque(maxlen=WINDOW)
    recent_outputs: deque[dict] = deque(maxlen=WINDOW)
    pending_calls: dict[str, dict] = {}
    pending_order: list[str] = []
    available_tools: set[str] = set()
    last_user_request = ""

    def emit(call: dict, output: dict, next_user: str | None, evt_idx: int) -> None:
        args_str = call.get("arguments") or ""
        out_text = output.get("output") or ""
        exit_code = parse_exit_code(out_text)
        label, failure_category = label_for(exit_code, out_text, next_user)
        sid = Path(path).stem.split("-")[-1][:8] if "-" in Path(path).stem else "session"
        rec = {
            "id": f"codex_{sid}_evt{evt_idx:06d}",
            "src": "codex",
            "src_path": path,
            "src_event_idx": evt_idx,
            "context": {
                "user_request": truncate(redact(last_user_request), 1024),
                "recent_actions": [short_action(c) for c in list(recent_calls)[-WINDOW:]],
                "recent_tool_results": [short_result(r) for r in list(recent_outputs)[-WINDOW:]],
                "available_tools": sorted(available_tools),
                "available_skills": [],
            },
            "observed_action": {
                "kind": "tool_use",
                "name": call.get("name") or "",
                "input": truncate(redact(args_str), MAX_FIELD),
                "result_is_error": exit_code is not None and exit_code != 0,
                "result_excerpt": truncate(redact(out_text), MAX_RESULT_EXCERPT),
            },
            "label": label,
            "failure_category": failure_category,
            "ideal_action_hint": truncate(redact(next_user or ""), 600) if label == "user_corrected" else "",
            "next_user_message": truncate(redact(next_user or ""), 600),
        }
        records.append(rec)

    with open(path, "rb") as fh:
        evt_idx = -1
        for line in fh:
            evt_idx += 1
            if not line.strip():
                continue
            try:
                evt = orjson.loads(line)
            except Exception:
                continue
            etype = evt.get("type")
            payload = evt.get("payload") or {}
            ptype = payload.get("type")

            if etype == "event_msg" and ptype == "user_message":
                msg = payload.get("message") or ""
                last_user_request = msg
                user_msgs.append(msg)
                # Resolve outstanding calls now that a follow-up user message has arrived.
                if pending_order:
                    for cid in list(pending_order):
                        info = pending_calls.get(cid)
                        if info and info.get("output") is not None:
                            emit(info["call"], info["output"], msg, info["evt_idx"])
                            pending_calls.pop(cid, None)
                            pending_order.remove(cid)
                            if len(records) >= MAX_PER_SESSION:
                                return records
                continue

            if etype == "event_msg" and ptype == "agent_reasoning":
                continue

            if etype == "response_item" and ptype == "function_call":
                name = payload.get("name") or ""
                if name:
                    available_tools.add(name)
                call_id = payload.get("call_id") or f"_anon_{evt_idx}"
                call_obj = {
                    "name": name,
                    "arguments": payload.get("arguments") or "",
                    "call_id": call_id,
                }
                recent_calls.append(call_obj)
                pending_calls[call_id] = {"call": call_obj, "output": None, "evt_idx": evt_idx}
                pending_order.append(call_id)
                continue

            if etype == "response_item" and ptype == "function_call_output":
                call_id = payload.get("call_id") or ""
                out_obj = {"output": payload.get("output") or "", "call_id": call_id}
                recent_outputs.append(out_obj)
                info = pending_calls.get(call_id)
                if info is not None:
                    info["output"] = out_obj
                    out_text = out_obj["output"]
                    exit_code = parse_exit_code(out_text)
                    if exit_code is not None and exit_code != 0:
                        emit(info["call"], out_obj, None, info["evt_idx"])
                        pending_calls.pop(call_id, None)
                        if call_id in pending_order:
                            pending_order.remove(call_id)
                        if len(records) >= MAX_PER_SESSION:
                            return records
                continue

    # Flush any remaining successful pending calls without a follow-up user message.
    for cid in pending_order:
        info = pending_calls.get(cid)
        if info and info.get("output") is not None:
            emit(info["call"], info["output"], None, info["evt_idx"])
            if len(records) >= MAX_PER_SESSION:
                break
    return records


def main() -> int:
    seed = json.loads(SEED_PATH.read_text())
    sessions = seed.get("codex_sessions") or []
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    total = 0
    per_session: list[tuple[str, int]] = []

    with open(OUT_PATH, "wb") as out_fh:
        for idx, sess in enumerate(sessions):
            path = sess["path"]
            if not os.path.exists(path):
                continue
            records = extract_session(path, idx)
            per_session.append((path, len(records)))
            for r in records:
                out_fh.write(orjson.dumps(r) + b"\n")
                total += 1
                label_counts[r["label"]] += 1
                if r["failure_category"]:
                    fail_counts[r["failure_category"]] += 1

    print(f"Total records: {total}")
    print("Label distribution:")
    for k in sorted(label_counts):
        print(f"  {k}: {label_counts[k]}")
    print("Failure category counts:")
    for k in sorted(fail_counts):
        print(f"  {k}: {fail_counts[k]}")
    print("Per-session counts:")
    for p, n in per_session:
        print(f"  {n:5d}  {p}")
    print(f"Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
