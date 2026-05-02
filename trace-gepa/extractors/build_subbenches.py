"""Slice the unified benchmark task file into category/difficulty sub-benchmarks.

Reads:
  trace-gepa/data/benchmark_tasks.jsonl              (required)
  trace-gepa/data/benchmark_tasks_synthetic.jsonl    (optional)

Writes:
  trace-gepa/data/benchmarks/<category>.tasks.jsonl
  trace-gepa/data/benchmarks/by_difficulty/<level>.tasks.jsonl
  trace-gepa/data/benchmarks/minibench.jsonl   (~30 stratified tasks, seed=42)
  trace-gepa/data/benchmarks/stress.jsonl      (top-20 hardest)
  trace-gepa/data/benchmarks/INDEX.md          (documentation)

Pure Python, no LM calls.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "benchmarks"
OUT_DIFF = OUT / "by_difficulty"

CATEGORIES = [
    "tool_routing",
    "planning",
    "debugging",
    "recovery",
    "path_grounding",
    "command_synthesis",
    "edit_safety",
]
DIFFICULTIES = ["easy", "medium", "hard"]
MINIBENCH_TARGET = 30
STRESS_TARGET = 20
SEED = 42


def load_tasks() -> list[dict]:
    tasks: list[dict] = []
    primary = DATA / "benchmark_tasks.jsonl"
    synth = DATA / "benchmark_tasks_synthetic.jsonl"
    seen: set[str] = set()
    for path in (primary, synth):
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                tid = t.get("id")
                if tid in seen:
                    continue
                if tid:
                    seen.add(tid)
                tasks.append(t)
    return tasks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def slice_by_category(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cat in CATEGORIES:
        rows = [t for t in tasks if t.get("category") == cat]
        write_jsonl(OUT / f"{cat}.tasks.jsonl", rows)
        counts[cat] = len(rows)
    return counts


def slice_by_difficulty(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for level in DIFFICULTIES:
        rows = [t for t in tasks if t.get("difficulty") == level]
        write_jsonl(OUT_DIFF / f"{level}.tasks.jsonl", rows)
        counts[level] = len(rows)
    return counts


def build_minibench(tasks: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_cat_diff: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in tasks:
        by_cat_diff[(t.get("category"), t.get("difficulty"))].append(t)
    for v in by_cat_diff.values():
        v.sort(key=lambda r: r.get("id", ""))
        rng.shuffle(v)

    # Target ~4 per category, balanced across available difficulties.
    per_cat = max(3, MINIBENCH_TARGET // max(1, len(CATEGORIES)))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    for cat in CATEGORIES:
        avail_diffs = [d for d in DIFFICULTIES if by_cat_diff.get((cat, d))]
        if not avail_diffs:
            continue
        picks: list[dict] = []
        # Round-robin difficulties for balance.
        idx = 0
        cursors = {d: 0 for d in avail_diffs}
        while len(picks) < per_cat:
            d = avail_diffs[idx % len(avail_diffs)]
            idx += 1
            pool = by_cat_diff.get((cat, d), [])
            c = cursors[d]
            if c < len(pool):
                picks.append(pool[c])
                cursors[d] = c + 1
            elif all(cursors[dd] >= len(by_cat_diff.get((cat, dd), [])) for dd in avail_diffs):
                break
        for p in picks:
            if p["id"] not in selected_ids:
                selected.append(p)
                selected_ids.add(p["id"])

    # Top up to MINIBENCH_TARGET if short.
    if len(selected) < MINIBENCH_TARGET:
        remaining = [t for t in tasks if t["id"] not in selected_ids]
        rng.shuffle(remaining)
        for t in remaining:
            if len(selected) >= MINIBENCH_TARGET:
                break
            selected.append(t)
            selected_ids.add(t["id"])
    elif len(selected) > MINIBENCH_TARGET:
        selected = selected[:MINIBENCH_TARGET]

    selected.sort(key=lambda r: r.get("id", ""))
    write_jsonl(OUT / "minibench.jsonl", selected)
    return selected


def build_stress(tasks: list[dict]) -> list[dict]:
    adversarial_cats = {"path_grounding", "edit_safety", "command_synthesis", "recovery"}

    def stress_score(t: dict) -> tuple[int, int, str]:
        diff = t.get("difficulty")
        d_rank = {"hard": 2, "medium": 1, "easy": 0}.get(diff, 0)
        adv = 1 if t.get("category") in adversarial_cats else 0
        return (d_rank, adv, t.get("id", ""))

    sorted_tasks = sorted(tasks, key=stress_score, reverse=True)
    stress = sorted_tasks[:STRESS_TARGET]
    write_jsonl(OUT / "stress.jsonl", stress)
    return stress


def render_index(
    total: int,
    cat_counts: dict[str, int],
    diff_counts: dict[str, int],
    minibench: list[dict],
    stress: list[dict],
    have_synth: bool,
) -> None:
    mini_matrix: dict[tuple[str, str], int] = Counter(
        (t.get("category"), t.get("difficulty")) for t in minibench
    )
    lines: list[str] = []
    lines.append("# Sub-Benchmarks Index")
    lines.append("")
    lines.append(f"Source: `data/benchmark_tasks.jsonl`" + (" + `_synthetic.jsonl`" if have_synth else ""))
    lines.append(f"Total tasks: {total}")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Count | File | When to run |")
    lines.append("| --- | ---: | --- | --- |")
    when = {
        "tool_routing": "Diagnosing tool-selection regressions.",
        "planning": "Validating plan structure / decomposition.",
        "debugging": "Stress-test reasoning over failure traces.",
        "recovery": "Post-error recovery and retry logic.",
        "path_grounding": "Filesystem-grounded path correctness.",
        "command_synthesis": "Shell / CLI argument synthesis.",
        "edit_safety": "Pre-edit invariants (Read-before-Edit, etc).",
    }
    for cat in CATEGORIES:
        lines.append(f"| {cat} | {cat_counts.get(cat, 0)} | `benchmarks/{cat}.tasks.jsonl` | {when[cat]} |")
    lines.append("")
    lines.append("## By difficulty")
    lines.append("")
    lines.append("| Difficulty | Count | File |")
    lines.append("| --- | ---: | --- |")
    for d in DIFFICULTIES:
        lines.append(f"| {d} | {diff_counts.get(d, 0)} | `benchmarks/by_difficulty/{d}.tasks.jsonl` |")
    lines.append("")
    lines.append("## minibench.jsonl")
    lines.append("")
    lines.append(f"{len(minibench)} stratified tasks (seed={SEED}). Use for cheap CI-style runs.")
    lines.append("")
    lines.append("Composition (category x difficulty):")
    lines.append("")
    header = "| category | " + " | ".join(DIFFICULTIES) + " | total |"
    sep = "| --- | " + " | ".join(["---:"] * len(DIFFICULTIES)) + " | ---: |"
    lines.append(header)
    lines.append(sep)
    for cat in CATEGORIES:
        row_counts = [mini_matrix.get((cat, d), 0) for d in DIFFICULTIES]
        total_row = sum(row_counts)
        if total_row == 0:
            continue
        lines.append("| " + cat + " | " + " | ".join(str(x) for x in row_counts) + f" | {total_row} |")
    lines.append("")
    lines.append("## stress.jsonl")
    lines.append("")
    lines.append(f"{len(stress)} hardest tasks (hard-first, then adversarial categories). Use for regression hunts.")
    lines.append("")
    for t in stress:
        summary = (t.get("human_readable_summary") or "")[:80]
        lines.append(f"- `{t['id']}` ({t.get('category')}/{t.get('difficulty')}) {summary}")
    lines.append("")
    (OUT / "INDEX.md").write_text("\n".join(lines))


def main() -> None:
    tasks = load_tasks()
    have_synth = (DATA / "benchmark_tasks_synthetic.jsonl").exists()
    cat_counts = slice_by_category(tasks)
    diff_counts = slice_by_difficulty(tasks)
    minibench = build_minibench(tasks)
    stress = build_stress(tasks)
    render_index(len(tasks), cat_counts, diff_counts, minibench, stress, have_synth)
    print(f"total={len(tasks)} synth={have_synth}")
    print(f"categories={cat_counts}")
    print(f"difficulties={diff_counts}")
    print(f"minibench={len(minibench)} stress={len(stress)}")


if __name__ == "__main__":
    main()
