#!/usr/bin/env python3
"""Build a failure-recovery pairs dataset.

Pairs (R1, R2) within the same session where R1 failed and a later record R2
(within 30 events) succeeded against a similar goal. Output:
    data/dataset_recovery.jsonl
    data/splits_recovery.json
    data/dataset_recovery_summary.md

Heuristic only - no LM calls.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
SOURCES = [("v1", ROOT / "data" / "dataset.jsonl"),
           ("v2", ROOT / "data" / "dataset_v2.jsonl")]
OUT_PAIRS = ROOT / "data" / "dataset_recovery.jsonl"
OUT_SPLITS = ROOT / "data" / "splits_recovery.json"
OUT_SUMMARY = ROOT / "data" / "dataset_recovery_summary.md"

MAX_DISTANCE = 30
STRONG_LEVENSHTEIN_THRESHOLD = 0.50  # ratio above which inputs are "similar"
PATH_RE = re.compile(r"(/[^\s\"\']{2,})|([A-Za-z0-9_./-]+\.[a-zA-Z]{1,5})")

try:  # pragma: no cover
    import Levenshtein  # type: ignore

    def lev_ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        return Levenshtein.ratio(a, b)
except Exception:  # noqa: BLE001
    def lev_ratio(a: str, b: str) -> float:
        # Fallback: ratio derived from common prefix + common suffix.
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        j = 0
        while j < (n - i) and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
            j += 1
        common = i + j
        return (2.0 * common) / (len(a) + len(b))


def _load_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag, path in SOURCES:
        if not path.exists():
            continue
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_dataset"] = tag
            rows.append(r)
    return rows


def _is_failure(r: dict[str, Any]) -> bool:
    if r.get("label") == "bad":
        return True
    if r.get("failure_category"):
        return True
    act = r.get("observed_action") or {}
    if act.get("result_is_error") is True:
        return True
    return False


def _is_success(r: dict[str, Any]) -> bool:
    act = r.get("observed_action") or {}
    if act.get("result_is_error") is True:
        return False
    if r.get("label") == "good":
        return True
    if r.get("failure_category"):
        return False
    return True


def _action_input(r: dict[str, Any]) -> str:
    act = r.get("observed_action") or {}
    inp = act.get("input")
    if isinstance(inp, str):
        return inp
    if inp is None:
        return ""
    return json.dumps(inp, sort_keys=True)


def _tool(r: dict[str, Any]) -> str:
    act = r.get("observed_action") or {}
    return act.get("name") or act.get("kind") or ""


def _affected_paths(text: str) -> set[str]:
    if not text:
        return set()
    return {m.group(0) for m in PATH_RE.finditer(text[:4000])}


def _user_request(r: dict[str, Any]) -> str:
    return ((r.get("context") or {}).get("user_request") or "")[:400]


def _request_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aw = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
    bw = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def _goal_similar(r1: dict[str, Any], r2: dict[str, Any]) -> bool:
    if _user_request(r1) == _user_request(r2):
        return True
    if _request_overlap(_user_request(r1), _user_request(r2)) >= 0.6:
        return True
    p1 = _affected_paths(_action_input(r1))
    p2 = _affected_paths(_action_input(r2))
    if p1 and p2 and (p1 & p2):
        return True
    return False


def _bash_command(inp: str) -> str:
    try:
        obj = json.loads(inp)
        if isinstance(obj, dict) and "command" in obj:
            return str(obj["command"])
    except Exception:  # noqa: BLE001
        pass
    return inp


def _lesson(r1: dict[str, Any], r2: dict[str, Any], tool_changed: bool,
            input_lev: float) -> str:
    t1, t2 = _tool(r1), _tool(r2)
    cat = r1.get("failure_category")
    in1, in2 = _action_input(r1), _action_input(r2)
    if tool_changed:
        return f"switched from {t1} to {t2} after {cat or 'failure'}"
    if t1 == "Bash" and t2 == "Bash":
        c1, c2 = _bash_command(in1), _bash_command(in2)
        h1 = c1.split()[0] if c1.split() else ""
        h2 = c2.split()[0] if c2.split() else ""
        if h1 and h2 and h1 != h2:
            return f"changed bash command head from '{h1}' to '{h2}'"
        if "|" in c2 and "|" not in c1:
            return "added pipeline filter to bash command"
        if " | head" in c2 and " | head" not in c1:
            return "narrowed bash output via head"
        if "2>&1" in c2 and "2>&1" not in c1:
            return "redirected stderr to capture full error"
        if cat == "bash_timeout_141":
            return "shortened bash command after timeout"
        if cat == "cmd_not_found_127":
            return "corrected command name after 127"
        return "tweaked bash arguments after failure"
    if t1 in {"Read", "Edit", "Write"} and t2 in {"Read", "Edit", "Write"}:
        try:
            o1 = json.loads(in1)
            o2 = json.loads(in2)
            p1 = (o1 or {}).get("file_path") or ""
            p2 = (o2 or {}).get("file_path") or ""
            if p1 != p2:
                if not p1.startswith("/") and p2.startswith("/"):
                    return "fixed path to use absolute"
                return "switched to a different file path"
        except Exception:  # noqa: BLE001
            pass
        if cat == "hallucinated_path":
            return "corrected path after hallucinated_path"
        return f"adjusted {t1} input after failure"
    if t1 == "Grep" and t2 == "Grep":
        return "narrowed grep scope"
    if input_lev < 0.3:
        return f"rewrote {t1} input substantially"
    return f"retried {t1} with minor edits"


def _strength(tool_changed: bool, input_lev: float, identical: bool) -> str:
    if identical:
        return "transient"
    if tool_changed or input_lev < (1.0 - STRONG_LEVENSHTEIN_THRESHOLD):
        return "strong"
    return "weak"


def _build_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_session[r["src_path"]].append(r)
    for k in by_session:
        by_session[k].sort(key=lambda x: x.get("src_event_idx", 0))

    pairs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for sid, evts in by_session.items():
        for i, r1 in enumerate(evts):
            if not _is_failure(r1):
                continue
            for j in range(i + 1, len(evts)):
                r2 = evts[j]
                dist = r2.get("src_event_idx", 0) - r1.get("src_event_idx", 0)
                if dist <= 0:
                    continue
                if dist > MAX_DISTANCE:
                    break
                if not _is_success(r2):
                    continue
                if (r2.get("observed_action") or {}).get("kind") != "tool_use":
                    continue
                if (r1.get("observed_action") or {}).get("kind") != "tool_use":
                    continue
                # next assistant action OR same goal
                next_action = (j == i + 1)
                if not (next_action or _goal_similar(r1, r2)):
                    continue
                t1, t2 = _tool(r1), _tool(r2)
                in1, in2 = _action_input(r1), _action_input(r2)
                tool_changed = (t1 != t2)
                identical = (not tool_changed) and (in1 == in2)
                lev = lev_ratio(in1, in2)
                input_changed = (in1 != in2)
                cat_changed = (r1.get("failure_category")
                               != r2.get("failure_category"))
                strength = _strength(tool_changed, lev, identical)
                pid = (f"recovery_{r1['src']}_"
                       f"{Path(r1['src_path']).stem}_"
                       f"{r1.get('src_event_idx')}_"
                       f"{r2.get('src_event_idx')}")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                pair = {
                    "id": pid,
                    "src": r1["src"],
                    "session_id": Path(r1["src_path"]).stem,
                    "failed_record": {k: v for k, v in r1.items()
                                      if not k.startswith("_")},
                    "recovery_record": {k: v for k, v in r2.items()
                                        if not k.startswith("_")},
                    "distance_events": int(dist),
                    "pair_strength": strength,
                    "transformation": {
                        "tool_changed": bool(tool_changed),
                        "input_changed": bool(input_changed),
                        "input_levenshtein_ratio": round(lev, 4),
                        "category_changed": bool(cat_changed),
                    },
                    "lesson": _lesson(r1, r2, tool_changed, lev),
                }
                pairs.append(pair)
                break  # one recovery per failure
    return pairs


def _stratified_split(pairs: list[dict[str, Any]], seed: int = 42
                      ) -> dict[str, list[str]]:
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for p in pairs:
        cat = p["failed_record"].get("failure_category") or "uncategorized"
        key = f"{cat}|{p['pair_strength']}"
        buckets[key].append(p["id"])
    splits = {"train": [], "val": [], "test": []}
    for ids in buckets.values():
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(round(n * 0.10))) if n >= 5 else 0
        n_test = max(1, int(round(n * 0.10))) if n >= 5 else 0
        n_train = n - n_val - n_test
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train:n_train + n_val])
        splits["test"].extend(ids[n_train + n_val:])
    return splits


def _summary(pairs: list[dict[str, Any]]) -> str:
    total = len(pairs)
    strength = Counter(p["pair_strength"] for p in pairs)
    lessons = Counter(p["lesson"] for p in pairs)
    cats = Counter(
        (p["failed_record"].get("failure_category") or "uncategorized")
        for p in pairs)
    mean_dist = (sum(p["distance_events"] for p in pairs) / total) if total else 0.0
    lines: list[str] = []
    lines.append("# Failure-Recovery Pairs Summary")
    lines.append("")
    lines.append(f"- total pairs: {total}")
    lines.append(f"- strong: {strength.get('strong', 0)}")
    lines.append(f"- weak: {strength.get('weak', 0)}")
    lines.append(f"- transient: {strength.get('transient', 0)}")
    lines.append(f"- mean distance_events: {mean_dist:.2f}")
    lines.append("")
    lines.append("## Top 10 transformation lessons")
    for lesson, n in lessons.most_common(10):
        lines.append(f"- {n}: {lesson}")
    lines.append("")
    lines.append("## Failure categories (R1)")
    for cat, n in cats.most_common():
        lines.append(f"- {n}: {cat}")
    return "\n".join(lines) + "\n"


def main() -> None:
    records = _load_records()
    pairs = _build_pairs(records)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PAIRS.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False))
            f.write("\n")
    splits = _stratified_split(pairs)
    OUT_SPLITS.write_text(json.dumps(splits, indent=2))
    OUT_SUMMARY.write_text(_summary(pairs))

    # Console report
    strength = Counter(p["pair_strength"] for p in pairs)
    lessons = Counter(p["lesson"] for p in pairs)
    print(f"pairs: {len(pairs)}")
    print(f"strong/weak/transient: "
          f"{strength.get('strong', 0)}/{strength.get('weak', 0)}/"
          f"{strength.get('transient', 0)}")
    print("top lessons:")
    for lesson, n in lessons.most_common(5):
        print(f"  {n}\t{lesson}")
    if pairs:
        mean_d = sum(p["distance_events"] for p in pairs) / len(pairs)
        print(f"mean distance_events: {mean_d:.2f}")


if __name__ == "__main__":
    main()
