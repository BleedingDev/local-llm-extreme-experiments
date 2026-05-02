#!/usr/bin/env python3
"""Build benchmark_tasks.jsonl from trace-gepa datasets."""
from __future__ import annotations
import json, re, random
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
DATA = ROOT / "data"
OUT = DATA / "benchmark_tasks.jsonl"
SUMMARY = DATA / "benchmark_tasks_summary.md"

random.seed(42)


def jload(p: Path):
    if not p.exists(): return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue


def parse_input(oa: dict) -> dict:
    inp = (oa or {}).get("input")
    if isinstance(inp, dict): return inp
    if isinstance(inp, str):
        try: return json.loads(inp)
        except Exception: return {}
    return {}


def difficulty_for(ctx: dict) -> str:
    nr = len(ctx.get("recent_actions") or [])
    rl = len((ctx.get("user_request") or ""))
    if nr <= 1 and rl < 250: return "easy"
    if nr >= 3 or rl > 600: return "hard"
    return "medium"


def trim_request(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    return s[:n].rsplit(" ", 1)[0] + " ..." if len(s) > n else s


def make_prompt(ctx: dict) -> dict:
    return {"user_request": trim_request(ctx.get("user_request") or ""),
            "context": {"available_tools": (ctx.get("available_tools") or [])[:40],
                        "recent_actions": (ctx.get("recent_actions") or [])[:5],
                        "available_skills": (ctx.get("available_skills") or [])[:25]}}


def build_tool_routing(records, target=25):
    out = []
    seen_tools = Counter()
    # Prefer short-context records for diversity in difficulty
    pool = []
    for r in records:
        if r.get("label") != "good" or r.get("failure_category"):
            continue
        oa = r.get("observed_action") or {}
        if oa.get("kind") != "tool_use":
            continue
        ctx = r.get("context") or {}
        if len(ctx.get("available_tools") or []) < 3:
            continue
        if len(ctx.get("user_request") or "") < 30:
            continue
        pool.append(r)
    pool.sort(key=lambda r: len((r["context"].get("recent_actions") or [])))
    for r in pool:
        oa = r["observed_action"]
        tool = oa.get("name")
        if not tool or seen_tools[tool] >= max(3, target // 6):
            continue
        seen_tools[tool] += 1
        out.append((r, tool))
        if len(out) >= target:
            break
    tasks = []
    for i, (r, tool) in enumerate(out):
        ctx = r["context"]
        ur = ctx.get("user_request") or ""
        nr = len(ctx.get("recent_actions") or [])
        d = "easy" if nr <= 1 and len(ur) < 500 else ("hard" if nr >= 5 else "medium")
        tasks.append({
            "id": f"task_tool_routing_{i:03d}",
            "category": "tool_routing",
            "difficulty": d,
            "source_record_ids": [r["id"]],
            "prompt": make_prompt(ctx),
            "expected": {
                "primary_action": {"tool_name": tool, "input_pattern_regex": ".*"},
                "must_avoid_actions": [],
                "must_include_keywords_in_reason": [],
                "must_avoid_keywords_in_reason": ["maybe", "i think"],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": f'$.tool_name == "{tool}"'},
            "rubric_weight": 1.0,
            "human_readable_summary": f"Pick {tool} given {len(ctx.get('available_tools') or [])} available tools.",
        })
    return tasks


def build_command_synthesis(records, target=10):
    out = []
    seen_cmd_kinds = Counter()
    pool = []
    for r in records:
        if r.get("label") != "good" or r.get("failure_category"):
            continue
        oa = r.get("observed_action") or {}
        if oa.get("name") != "Bash":
            continue
        inp = parse_input(oa)
        cmd = (inp.get("command") or "").strip()
        if not cmd or len(cmd) > 300:
            continue
        head = cmd.split()[0] if cmd.split() else ""
        if head not in {"git", "ls", "rg", "wc", "head", "tail", "cat", "find", "pnpm", "npm", "node", "python3"}:
            continue
        ctx = r.get("context") or {}
        if len(ctx.get("user_request") or "") < 20:
            continue
        pool.append((r, cmd, head))
    pool.sort(key=lambda x: len((x[0]["context"].get("recent_actions") or [])))
    for r, cmd, head in pool:
        if seen_cmd_kinds[head] >= 2:
            continue
        seen_cmd_kinds[head] += 1
        out.append((r, cmd, head))
        if len(out) >= target:
            break
    tasks = []
    for i, (r, cmd, head) in enumerate(out):
        ctx = r["context"]
        # Extract characteristic substrings as keyword anchors
        anchors = []
        for tok in cmd.split():
            if "/" in tok or tok.startswith("-") or tok in {"|", "&&"}:
                anchors.append(tok)
            if len(anchors) >= 3:
                break
        # Build a tolerant regex: must contain head and the most distinctive flag
        pat = re.escape(head)
        if anchors:
            distinct = max(anchors, key=len)
            pat += ".*" + re.escape(distinct[:30])
        nr = len(ctx.get("recent_actions") or [])
        d = "easy" if nr <= 1 else ("hard" if nr >= 5 else "medium")
        tasks.append({
            "id": f"task_command_synthesis_{i:03d}",
            "category": "command_synthesis",
            "difficulty": d,
            "source_record_ids": [r["id"]],
            "prompt": make_prompt(ctx),
            "expected": {
                "primary_action": {"tool_name": "Bash", "input_pattern_regex": pat},
                "must_avoid_actions": [{"tool_name": "Bash", "input_pattern_regex": r".*\brm\s+-rf\s+/.*"}],
                "must_include_keywords_in_reason": [],
                "must_avoid_keywords_in_reason": [],
            },
            "verifier_kind": "regex",
            "verifier_spec": {"type": "regex", "pattern_or_command": pat},
            "rubric_weight": 1.0,
            "human_readable_summary": f"Synthesize a {head} command for the stated goal.",
        })
    return tasks


def build_edit_safety(records, target=10):
    pos, neg = [], []
    for r in records:
        oa = r.get("observed_action") or {}
        if r.get("failure_category") == "edit_string_not_unique":
            if len(neg) < target // 2:
                neg.append(r)
            continue
        if oa.get("name") == "Edit" and r.get("label") == "good":
            recent = (r.get("context") or {}).get("recent_actions") or []
            inp = parse_input(oa)
            fp = inp.get("file_path") or ""
            fn = fp.split("/")[-1] if fp else ""
            read_first = bool(fn) and any(("Read" in a) and (fn in a) for a in recent[-3:])
            if read_first and len(pos) < target - len(neg):
                pos.append(r)
        if len(pos) + len(neg) >= target:
            break
    tasks = []
    for i, r in enumerate(pos):
        ctx = r["context"] or {}
        inp = parse_input(r["observed_action"])
        fp = inp.get("file_path") or ""
        recent = ctx.get("recent_actions") or []
        read_first = any("Read" in a and fp.split("/")[-1] in a for a in recent[-5:]) if fp else False
        tasks.append({
            "id": f"task_edit_safety_{i:03d}",
            "category": "edit_safety",
            "difficulty": "medium" if read_first else "easy",
            "source_record_ids": [r["id"]],
            "prompt": make_prompt(ctx),
            "expected": {
                "primary_action": {"tool_name": "Edit", "input_pattern_regex": ".*"},
                "must_avoid_actions": [{"tool_name": "Edit", "input_pattern_regex": r"^(?!.*old_string).*$"}],
                "must_include_keywords_in_reason": ["read", "unique"] if read_first else [],
                "must_avoid_keywords_in_reason": [],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema",
                              "pattern_or_command": '$.tool_name == "Edit" and $.input.old_string != null'},
            "rubric_weight": 1.0,
            "human_readable_summary": "Edit safely: ensure Read precedes Edit and old_string is unique.",
        })
    for j, r in enumerate(neg):
        ctx = r["context"] or {}
        idx = len(pos) + j
        tasks.append({
            "id": f"task_edit_safety_{idx:03d}",
            "category": "edit_safety",
            "difficulty": "hard",
            "source_record_ids": [r["id"]],
            "prompt": make_prompt(ctx),
            "expected": {
                "primary_action": {"tool_name": "Read", "input_pattern_regex": ".*"},
                "must_avoid_actions": [{"tool_name": "Edit", "input_pattern_regex": ".*"}],
                "must_include_keywords_in_reason": ["context", "unique"],
                "must_avoid_keywords_in_reason": [],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": '$.tool_name == "Read"'},
            "rubric_weight": 1.0,
            "human_readable_summary": "Recover from edit_string_not_unique: re-read and widen old_string.",
        })
    return tasks[:target]


def build_path_grounding(records, target=10):
    halls = [r for r in records if r.get("failure_category") == "hallucinated_path"][:target]
    tasks = []
    for i, r in enumerate(halls):
        ctx = r["context"] or {}
        oa = r.get("observed_action") or {}
        inp = parse_input(oa)
        bad_path = inp.get("file_path") or inp.get("path") or ""
        nr = len(ctx.get("recent_actions") or [])
        d = "easy" if nr <= 1 else ("medium" if nr <= 3 else "hard")
        tasks.append({
            "id": f"task_path_grounding_{i:03d}",
            "category": "path_grounding",
            "difficulty": d,
            "source_record_ids": [r["id"]],
            "prompt": make_prompt(ctx),
            "expected": {
                "primary_action": {"tool_name": "Bash", "input_pattern_regex": r"(ls|rg|find)\s.*"},
                "must_avoid_actions": [
                    {"tool_name": oa.get("name") or "Read",
                     "input_pattern_regex": re.escape(bad_path)[:100] if bad_path else r"^\Z"}
                ],
                "must_include_keywords_in_reason": ["verify", "exists"],
                "must_avoid_keywords_in_reason": ["assume"],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema",
                              "pattern_or_command": 'tool_name in {"Bash","Glob","Grep"}'},
            "rubric_weight": 1.5,
            "human_readable_summary": "Avoid hallucinated path; verify existence before reading.",
        })
    return tasks


def build_debugging(records, target=15):
    bugs = [r for r in records
            if r.get("failure_category") == "bash_exit_nonzero"
            and r.get("observed_action", {}).get("result_excerpt")]
    random.shuffle(bugs)
    bugs = bugs[:target]
    tasks = []
    for i, r in enumerate(bugs):
        ctx = r["context"] or {}
        oa = r["observed_action"]
        excerpt = (oa.get("result_excerpt") or "")[:400]
        # Inject the failing command + error into the user_request for self-containment
        inp = parse_input(oa)
        cmd = (inp.get("command") or "")[:200]
        prompt = make_prompt(ctx)
        prompt["context"]["recent_actions"] = (prompt["context"]["recent_actions"] or []) + [
            f"Bash: {cmd}",
        ]
        prompt["user_request"] = (
            (prompt["user_request"] + "\n\n" if prompt["user_request"] else "") +
            f"The previous Bash command failed with:\n{excerpt}\nWhat is the next correct action?"
        )
        d = "easy" if len(excerpt) < 120 else ("hard" if len(excerpt) > 320 else "medium")
        tasks.append({
            "id": f"task_debugging_{i:03d}",
            "category": "debugging",
            "difficulty": d,
            "source_record_ids": [r["id"]],
            "prompt": prompt,
            "expected": {
                "primary_action": {"tool_name": "Bash", "input_pattern_regex": ".*"},
                "must_avoid_actions": [
                    {"tool_name": "Bash", "input_pattern_regex": re.escape(cmd[:60]) if cmd else r"^\Z"}
                ],
                "must_include_keywords_in_reason": ["fail", "error"],
                "must_avoid_keywords_in_reason": ["retry"],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema",
                              "pattern_or_command": 'must not repeat failing command verbatim'},
            "rubric_weight": 1.0,
            "human_readable_summary": "Diagnose a non-zero exit and choose a corrective next action.",
        })
    return tasks


def build_recovery(target=15):
    pairs = list(jload(DATA / "dataset_recovery.jsonl"))
    strong = [p for p in pairs if p.get("pair_strength") == "strong"]
    random.shuffle(strong)
    chosen = strong[:target]
    tasks = []
    for i, p in enumerate(chosen):
        failed = p["failed_record"]
        recov = p.get("recovery_record") or {}
        ctx = failed.get("context") or {}
        f_oa = failed.get("observed_action") or {}
        r_oa = recov.get("observed_action") or {}
        f_inp = parse_input(f_oa)
        r_tool = r_oa.get("name") or "Bash"
        excerpt = (f_oa.get("result_excerpt") or "")[:300]
        prompt = make_prompt(ctx)
        prompt["context"]["recent_actions"] = (prompt["context"]["recent_actions"] or []) + [
            f"{f_oa.get('name')}: {json.dumps(f_inp)[:200]}"
        ]
        prompt["user_request"] = (
            (prompt["user_request"] + "\n\n" if prompt["user_request"] else "") +
            f"Your last action failed: {excerpt}\nRecover correctly."
        )
        tasks.append({
            "id": f"task_recovery_{i:03d}",
            "category": "recovery",
            "difficulty": "hard" if p.get("distance_events", 0) > 2 else "medium",
            "source_record_ids": [failed.get("id"), recov.get("id")],
            "prompt": prompt,
            "expected": {
                "primary_action": {"tool_name": r_tool, "input_pattern_regex": ".*"},
                "must_avoid_actions": [
                    {"tool_name": f_oa.get("name") or "Bash",
                     "input_pattern_regex": re.escape(json.dumps(f_inp)[:60])}
                ],
                "must_include_keywords_in_reason": ["recover"],
                "must_avoid_keywords_in_reason": [],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {"type": "json_schema", "pattern_or_command": f'$.tool_name == "{r_tool}"'},
            "rubric_weight": 1.5,
            "human_readable_summary": (p.get("lesson") or "Recover from a failed action.")[:99],
        })
    return tasks


def build_planning(records, target=15):
    plans = list(jload(DATA / "planner_dataset.jsonl"))
    tasks = []
    for i, p in enumerate(plans[:target]):
        issues = p.get("ground_truth_issues") or []
        n = len(issues)
        files = [f for iss in issues[:3] for f in (iss.get("expectedFiles") or [])[:3]]
        prompt = {"user_request": trim_request(p.get("user_request") or "Plan the next steps."),
                  "context": {"available_tools": ["TaskCreate", "Read", "Bash", "Edit", "Write"],
                              "recent_actions": [], "available_skills": []}}
        d = "easy" if n <= 2 else ("hard" if n >= 5 else "medium")
        tasks.append({
            "id": f"task_planning_{i:03d}",
            "category": "planning",
            "difficulty": d,
            "source_record_ids": [p.get("id")],
            "prompt": prompt,
            "expected": {
                "primary_action": {"tool_name": "TaskCreate", "input_pattern_regex": ".*"},
                "must_avoid_actions": [],
                "must_include_keywords_in_reason": ["plan", "decompose"],
                "must_avoid_keywords_in_reason": [],
            },
            "verifier_kind": "structural_json",
            "verifier_spec": {
                "type": "json_schema",
                "pattern_or_command": f"plan must mention >= {min(n,3)} of: " + ", ".join(files[:6]),
            },
            "rubric_weight": 1.0,
            "human_readable_summary": f"Decompose the request into ~{n} concrete sub-issues with file refs.",
        })
    if len(tasks) < target:
        idx = len(tasks)
        for r in records:
            if (r.get("observed_action") or {}).get("name") != "TaskCreate" or r.get("label") != "good":
                continue
            ctx = r["context"] or {}
            if len(ctx.get("user_request") or "") < 40: continue
            tasks.append({"id": f"task_planning_{idx:03d}", "category": "planning",
                          "difficulty": "medium", "source_record_ids": [r["id"]],
                          "prompt": make_prompt(ctx),
                          "expected": {"primary_action": {"tool_name": "TaskCreate", "input_pattern_regex": ".*"},
                                       "must_avoid_actions": [],
                                       "must_include_keywords_in_reason": ["plan"],
                                       "must_avoid_keywords_in_reason": []},
                          "verifier_kind": "structural_json",
                          "verifier_spec": {"type": "json_schema", "pattern_or_command": '$.tool_name == "TaskCreate"'},
                          "rubric_weight": 1.0,
                          "human_readable_summary": "Open a TaskCreate to track multi-step work."})
            idx += 1
            if len(tasks) >= target: break
    return tasks[:target]


def validate_task(t: dict) -> tuple[bool, str]:
    try:
        json.dumps(t)
    except Exception as e:
        return False, f"non-serializable: {e}"
    if not t.get("expected", {}).get("primary_action", {}).get("tool_name"):
        return False, "primary_action empty"
    vk = t.get("verifier_kind")
    spec = t.get("verifier_spec") or {}
    if vk == "regex" and not spec.get("pattern_or_command"):
        return False, "regex spec missing pattern"
    if vk == "structural_json" and spec.get("type") != "json_schema":
        return False, "structural_json mismatch"
    summary = t.get("human_readable_summary") or ""
    if len(summary) > 100:
        return False, f"summary too long ({len(summary)})"
    return True, "ok"


def write_summary(tasks: list[dict]):
    by_cat = Counter(t["category"] for t in tasks)
    by_diff = Counter(t["difficulty"] for t in tasks)
    by_ver = Counter(t["verifier_kind"] for t in tasks)
    samples = []
    seen = set()
    for t in tasks:
        if t["category"] not in seen:
            samples.append(t)
            seen.add(t["category"])
        if len(samples) >= 7:
            break
    lines = ["# Benchmark Tasks Summary", "",
             f"Total tasks: {len(tasks)}", "",
             "## Per-category counts", ""]
    for c, n in by_cat.most_common():
        lines.append(f"- {c}: {n}")
    lines += ["", "## Difficulty distribution", ""]
    for d, n in by_diff.most_common():
        lines.append(f"- {d}: {n}")
    lines += ["", "## Verifier kind distribution", ""]
    for v, n in by_ver.most_common():
        lines.append(f"- {v}: {n}")
    lines += ["", "## Sample task per category", "",
              "| id | category | difficulty | summary |",
              "| --- | --- | --- | --- |"]
    for t in samples:
        lines.append(f"| {t['id']} | {t['category']} | {t['difficulty']} | "
                     f"{t['human_readable_summary']} |")
    SUMMARY.write_text("\n".join(lines) + "\n")


def main():
    print("loading dataset_v2 ...")
    v2 = list(jload(DATA / "dataset_v2.jsonl"))
    print(f"  {len(v2)} records")

    tasks: list[dict] = []
    tasks += build_tool_routing(v2, 28)
    tasks += build_command_synthesis(v2, 12)
    tasks += build_edit_safety(v2, 10)
    tasks += build_path_grounding(v2, 10)
    tasks += build_debugging(v2, 15)
    tasks += build_recovery(15)
    tasks += build_planning(v2, 15)

    valid, bad = [], []
    for t in tasks:
        ok, msg = validate_task(t)
        (valid if ok else bad).append((t, msg))
    print(f"  valid={len(valid)} bad={len(bad)}")
    for t, msg in bad[:5]:
        print("    bad:", t.get("id"), msg)

    OUT.write_text("\n".join(json.dumps(v[0]) for v in valid) + "\n")
    write_summary([v[0] for v in valid])
    print(f"wrote {OUT} ({len(valid)} tasks)")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
