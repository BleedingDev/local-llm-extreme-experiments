from __future__ import annotations

import os
import re
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

import orjson

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed_sessions.json"
OUT = ROOT / "data" / "cc_dataset.jsonl"

NOISY_TOOLS = {"TodoWrite", "TaskUpdate", "TaskList", "TaskGet"}
PER_SESSION_CAP = 200
WINDOW_K = 5
EVENT_TRUNC = 1500
FIELD_TRUNC = 2000

CORRECTION_PREFIXES = (
    "no", "stop", "don't", "wrong", "actually",
    "espera", "ne ", "počkej", "pozor",
)
CONFIRM_TOKENS = (
    "great", "perfect", "nice", "thanks", "ok continue",
    "exactly", "super", "skvělé", "díky",
)

REDACT_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]+"), "<REDACTED_KEY>"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "<REDACTED_KEY>"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "<REDACTED_KEY>"),
]


def redact(s: str) -> str:
    for pat, repl in REDACT_PATTERNS:
        s = pat.sub(repl, s)
    return s


def trunc(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "...<truncated>"


def stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return orjson.dumps(v).decode("utf-8", errors="replace")
    except Exception:
        return str(v)


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("type")
                if t == "text":
                    parts.append(blk.get("text", ""))
                elif t == "tool_result":
                    inner = blk.get("content", "")
                    parts.append(extract_text(inner))
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", "") or content.get("content", "") or ""
    return ""


def classify_failure(tool_name: str, is_error: bool, result_text: str) -> str | None:
    if not is_error:
        return None
    rt = result_text.lower()
    if tool_name == "Bash":
        if "command not found" in rt or "exit code 127" in rt or "exit 127" in rt:
            return "cmd_not_found_127"
        if "exit 141" in rt or "exit code 141" in rt:
            return "bash_timeout_141"
        if "cancelled: parallel" in rt or "cancelled because a parallel" in rt:
            return "cancelled_parallel_batch"
        return "bash_exit_nonzero"
    if tool_name == "Edit":
        if "old_string" in rt and ("not unique" in rt or "not found" in rt or "no match" in rt):
            return "edit_string_not_unique"
        if "must use read tool" in rt or "must read" in rt:
            return "edit_file_not_read"
    if tool_name == "Skill":
        if "unknown skill" in rt:
            return "hallucinated_skill"
    return None


def determine_label(is_error: bool, next_user_text: str | None) -> str:
    if is_error:
        return "bad"
    if next_user_text:
        low = next_user_text.strip().lower()
        for pref in CORRECTION_PREFIXES:
            if low.startswith(pref):
                return "user_corrected"
        for tok in CONFIRM_TOKENS:
            if tok in low:
                return "user_confirmed"
    return "good"


def first_user_text_after(events: list[dict], start_idx: int, max_scan: int = 30) -> str | None:
    end = min(len(events), start_idx + max_scan)
    for i in range(start_idx + 1, end):
        ev = events[i]
        if ev.get("type") != "user":
            continue
        msg = ev.get("message") or {}
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
            if has_tool_result:
                continue
            txt = extract_text(content)
        elif isinstance(content, str):
            txt = content
        else:
            continue
        if not txt:
            continue
        if txt.startswith("<command-") or txt.startswith("<local-command-"):
            continue
        return txt
    return None


def find_tool_result(events: list[dict], start_idx: int, tool_use_id: str) -> tuple[bool, str] | None:
    end = min(len(events), start_idx + 50)
    for i in range(start_idx + 1, end):
        ev = events[i]
        if ev.get("type") != "user":
            continue
        msg = ev.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "tool_result" and blk.get("tool_use_id") == tool_use_id:
                txt = extract_text(blk.get("content", ""))
                return bool(blk.get("is_error", False)), txt
    return None


def latest_user_request(prev_users: deque) -> str:
    for txt in reversed(prev_users):
        if txt and not txt.startswith("<command-") and not txt.startswith("<local-command-"):
            return txt
    return prev_users[-1] if prev_users else ""


def process_session(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    events: list[dict] = []
    with p.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(orjson.loads(line))
            except Exception:
                continue

    sid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", path)
    sid_short = sid_match.group(1)[:8] if sid_match else "unknown0"

    available_tools: list[str] = []
    available_skills: list[str] = []
    prev_users: deque = deque(maxlen=WINDOW_K)
    prev_actions: deque = deque(maxlen=WINDOW_K)
    prev_results: deque = deque(maxlen=WINDOW_K)

    out: list[dict] = []

    for idx, ev in enumerate(events):
        et = ev.get("type")

        if et == "attachment":
            att = ev.get("attachment") or {}
            atype = att.get("type")
            if atype == "deferred_tools_delta":
                added = att.get("addedNames") or []
                for n in added:
                    if n not in available_tools:
                        available_tools.append(n)
            elif atype == "skill_listing":
                listing = att.get("content", "") or ""
                for line in listing.splitlines():
                    m = re.match(r"\s*-\s*([A-Za-z0-9_\-:]+):", line)
                    if m:
                        sk = m.group(1)
                        if sk not in available_skills:
                            available_skills.append(sk)
            continue

        if et == "user":
            msg = ev.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                )
                if has_tool_result:
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            rt = extract_text(b.get("content", ""))
                            prev_results.append(trunc(redact(rt), EVENT_TRUNC))
                else:
                    txt = extract_text(content)
                    if txt:
                        prev_users.append(trunc(redact(txt), EVENT_TRUNC))
            elif isinstance(content, str):
                prev_users.append(trunc(redact(content), EVENT_TRUNC))
            continue

        if et != "assistant":
            continue

        msg = ev.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for blk in content:
            if not isinstance(blk, dict) or blk.get("type") != "tool_use":
                continue
            tool_name = blk.get("name", "")
            if tool_name in NOISY_TOOLS:
                continue
            tool_input = blk.get("input", {})
            tool_use_id = blk.get("id", "")

            tr = find_tool_result(events, idx, tool_use_id)
            if tr is None:
                is_error, result_text = False, ""
            else:
                is_error, result_text = tr

            next_user_txt = first_user_text_after(events, idx)
            label = determine_label(is_error, next_user_txt)
            fc = classify_failure(tool_name, is_error, result_text)

            input_str = redact(stringify(tool_input))
            input_str = trunc(input_str, FIELD_TRUNC)
            result_excerpt = trunc(redact(result_text), 500)

            rec = {
                "id": f"cc_{sid_short}_evt{idx:05d}",
                "src": "cc",
                "src_path": path,
                "src_event_idx": idx,
                "context": {
                    "user_request": latest_user_request(prev_users),
                    "recent_actions": list(prev_actions),
                    "recent_tool_results": list(prev_results),
                    "available_tools": list(available_tools),
                    "available_skills": list(available_skills),
                },
                "observed_action": {
                    "kind": "tool_use",
                    "name": tool_name,
                    "input": input_str,
                    "result_is_error": is_error,
                    "result_excerpt": result_excerpt,
                },
                "label": label,
                "failure_category": fc,
                "ideal_action_hint": (
                    trunc(redact(next_user_txt), 500)
                    if (label == "user_corrected" and next_user_txt)
                    else None
                ),
                "next_user_message": (
                    trunc(redact(next_user_txt), 500) if next_user_txt else None
                ),
            }

            action_summary = f"{tool_name}: {trunc(input_str, 400)}"
            prev_actions.append(action_summary)

            out.append(rec)
            if len(out) >= PER_SESSION_CAP:
                return out
    return out


def main() -> int:
    seed = orjson.loads(SEED.read_bytes())
    sessions = seed.get("cc_sessions", [])
    OUT.parent.mkdir(parents=True, exist_ok=True)

    label_counts: Counter = Counter()
    fc_counts: Counter = Counter()
    total = 0

    with OUT.open("wb") as f:
        for s in sessions:
            recs = process_session(s["path"])
            for r in recs:
                f.write(orjson.dumps(r))
                f.write(b"\n")
                label_counts[r["label"]] += 1
                if r["failure_category"]:
                    fc_counts[r["failure_category"]] += 1
                total += 1
            print(f"  {Path(s['path']).name}: {len(recs)} records", file=sys.stderr)

    print(f"\nTotal records: {total}")
    print("Label distribution:")
    for lab, n in label_counts.most_common():
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {lab}: {n} ({pct:.1f}%)")
    non_good = total - label_counts.get("good", 0)
    pct_ng = (100.0 * non_good / total) if total else 0.0
    print(f"Non-good fraction: {non_good}/{total} ({pct_ng:.1f}%)")
    print("Top failure categories:")
    for fc, n in fc_counts.most_common(5):
        print(f"  {fc}: {n}")
    print(f"Output: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
