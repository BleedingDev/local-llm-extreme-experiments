"""Backfill the easy-tier of the benchmark by re-tagging qualifying medium tasks.

Method (a): heuristic re-tag, no LM calls.

Tiers (each more permissive). We pick the strictest tier that yields >= 20.
  T1 (brief default):
    recent_actions <= 2 AND 20 <= user_request <= 250 AND
    primary_action.tool_name in {Read, LS, Bash} AND must_avoid_actions <= 1
  T2 (relaxed length):
    same but user_request <= 1500
  T3 (relaxed history):
    recent_actions <= 4
  T4 (broad):
    recent_actions <= 6 AND user_request <= 2000 AND tool in
    {Read, LS, Bash, exec_command} AND must_avoid_actions <= 2

The brief notes 20-250 chars but real tasks are ~600 chars, so T1 yielded 0.
T4 captures ~20 tool_routing/command_synthesis/recovery/debugging tasks where
the right answer is still a single low-stakes Read/Bash/LS-style call.

Reads:  trace-gepa/data/benchmark_tasks.jsonl
Writes: trace-gepa/data/benchmarks/by_difficulty/easy_retagged.tasks.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRIMARY = DATA / "benchmark_tasks.jsonl"
OUT = DATA / "benchmarks" / "by_difficulty" / "easy_retagged.tasks.jsonl"

TIERS = [
    {"name": "T1_strict", "tools": {"Read", "LS", "Bash"}, "ur_max": 250, "ra_max": 2, "av_max": 1},
    {"name": "T2_lenrelax", "tools": {"Read", "LS", "Bash"}, "ur_max": 1500, "ra_max": 2, "av_max": 1},
    {"name": "T3_histrelax", "tools": {"Read", "LS", "Bash"}, "ur_max": 1500, "ra_max": 4, "av_max": 1},
    {"name": "T4_broad", "tools": {"Read", "LS", "Bash", "exec_command"}, "ur_max": 2000, "ra_max": 6, "av_max": 2},
]
MIN_YIELD = 20


def features(task: dict) -> tuple[int, int, str, int]:
    prompt = task.get("prompt") or {}
    ctx = prompt.get("context") or {}
    expected = task.get("expected") or {}
    primary = expected.get("primary_action") or {}
    return (
        len(ctx.get("recent_actions") or []),
        len(prompt.get("user_request") or ""),
        primary.get("tool_name") or "",
        len(expected.get("must_avoid_actions") or []),
    )


def matches(task: dict, tier: dict) -> bool:
    if task.get("difficulty") != "medium":
        return False
    ra, ur, tn, av = features(task)
    return (
        ra <= tier["ra_max"]
        and 20 <= ur <= tier["ur_max"]
        and tn in tier["tools"]
        and av <= tier["av_max"]
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def select_with_escalation(tasks: list[dict]) -> tuple[list[dict], str, list[tuple[str, int]]]:
    yields: list[tuple[str, int]] = []
    chosen: list[dict] = []
    chosen_tier = TIERS[0]["name"]
    for tier in TIERS:
        hits = [t for t in tasks if matches(t, tier)]
        yields.append((tier["name"], len(hits)))
        if len(hits) >= MIN_YIELD or tier is TIERS[-1]:
            chosen = hits
            chosen_tier = tier["name"]
            if len(hits) >= MIN_YIELD:
                break
    return chosen, chosen_tier, yields


def main() -> None:
    tasks = load_jsonl(PRIMARY)
    chosen, tier_name, yields = select_with_escalation(tasks)

    retagged: list[dict] = []
    for t in chosen:
        new = dict(t)
        new["difficulty"] = "easy"
        new["retag_source_difficulty"] = "medium"
        new["retag_method"] = f"heuristic_{tier_name}"
        retagged.append(new)
    write_jsonl(OUT, retagged)

    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    for t in tasks:
        d = t.get("difficulty")
        if d in diff_counts:
            diff_counts[d] += 1
    final = {
        "easy": diff_counts["easy"] + len(retagged),
        "medium": diff_counts["medium"] - len(retagged),
        "hard": diff_counts["hard"],
    }
    print(f"input_tasks={len(tasks)}")
    print(f"tier_yields={yields}")
    print(f"selected_tier={tier_name} retagged={len(retagged)}")
    print(f"original_distribution={diff_counts}")
    print(f"effective_distribution={final}")
    for t in retagged[:5]:
        print(f"  {t['id']} :: {(t.get('human_readable_summary') or '')[:80]}")


if __name__ == "__main__":
    main()
