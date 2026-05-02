"""Synthetic benchmark task generator (Phase-3 Fix Agent #FIX2).

Reads `data/benchmark_tasks.jsonl` (real-trace tasks from Bench Agent #2) and
uses claude-opus-4-7 to backfill the easy-tier gap and underrepresented failure
modes. Writes `benchmark_tasks_synthetic.jsonl`, `benchmark_tasks_full.jsonl`
(real + synthetic, deduplicated by id), and an audit markdown.

Schema field names match the existing real tasks exactly: pattern_or_command,
must_avoid_actions{tool_name,input_pattern_regex}, etc.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt import llm  # noqa: E402

REAL = _ROOT / "data" / "benchmark_tasks.jsonl"
SYNTH = _ROOT / "data" / "benchmark_tasks_synthetic.jsonl"
FULL = _ROOT / "data" / "benchmark_tasks_full.jsonl"
AUDIT = _ROOT / "data" / "benchmark_tasks_synthetic_audit.md"

OPUS = "claude-opus-4-7"
MAX_TOKENS = 8192
MAX_CALLS = 15

REQ = {"id", "category", "difficulty", "source_record_ids", "prompt", "expected",
       "verifier_kind", "verifier_spec", "rubric_weight", "human_readable_summary"}
PCTX = {"available_tools", "recent_actions", "available_skills"}

# Targets requested by the FIX2 directive:
#   - 30 easy tasks distributed across 7 categories
#   - 8 edit_string_not_unique + 6 edit_file_not_read
#   - 6 hallucinated_skill (no skill_listing context => fall back, do NOT invent skill name)
#   - 10 adversarial path_grounding (paths look plausible but do not exist)
GAPS = [
    {"prefix": "synth_easy_tool_routing", "category": "tool_routing", "difficulty": "easy", "n": 5,
     "focus": "Single-shot tool routing with an OBVIOUS correct tool. Available_tools includes Read, "
              "Write, Edit, Bash, Glob, Grep plus a few MCP names. recent_actions empty. user_request "
              "names a concrete file path or directory and asks to read/list/search it. Expected "
              "primary_action.tool_name in {Read, Glob, Grep, Bash}. verifier_spec.pattern_or_command "
              "is a structural_json check like '$.tool_name == \"Read\"'."},
    {"prefix": "synth_easy_planning", "category": "planning", "difficulty": "easy", "n": 4,
     "focus": "User asks for a plan / breakdown of a small task. available_tools includes TaskCreate "
              "(or ExitPlanMode), Read, Bash. The CORRECT primary_action is TaskCreate. Keep the "
              "request short (under 200 chars). Verifier: '$.tool_name == \"TaskCreate\"'."},
    {"prefix": "synth_easy_debugging", "category": "debugging", "difficulty": "easy", "n": 5,
     "focus": "A short error message is included; one obvious diagnostic command will reveal cause. "
              "Recent_actions show ONE failed Bash with a clear error (e.g. 'ENOENT', 'syntax error', "
              "'tsc TS2307'). Expected primary_action: Bash with a narrow follow-up (cat package.json, "
              "pnpm tsc --noEmit on one file, ls). verifier regex matches that command shape."},
    {"prefix": "synth_easy_recovery", "category": "recovery", "difficulty": "easy", "n": 4,
     "focus": "One previous action failed; the recovery step is obvious (re-run with corrected flag, "
              "create missing dir, install missing package). Expected primary_action.tool_name = Bash. "
              "verifier_spec.pattern_or_command is a regex on the corrected command shape."},
    {"prefix": "synth_easy_path_grounding", "category": "path_grounding", "difficulty": "easy", "n": 4,
     "focus": "User mentions a file by basename or partial path. Correct first action is Glob or Bash "
              "'ls'/'find' to locate it before reading. recent_actions empty. Expected primary_action "
              "tool_name in {Glob, Bash}; verifier regex like '(ls|find|rg)\\\\s.*' or Glob pattern."},
    {"prefix": "synth_easy_command_synthesis", "category": "command_synthesis", "difficulty": "easy", "n": 4,
     "focus": "User asks for a one-liner Bash command (count lines, list dir, grep for word, git status, "
              "git diff). Expected primary_action.tool_name = Bash. verifier_spec.pattern_or_command is "
              "a regex pinning the command head (e.g. '^wc -l\\\\s', '^git status', '^grep\\\\b')."},
    {"prefix": "synth_easy_edit_safety", "category": "edit_safety", "difficulty": "easy", "n": 4,
     "focus": "User asks to edit a file that WAS already Read in recent_actions and the snippet to "
              "change is unique. Expected primary_action.tool_name = Edit. Keep must_avoid_actions = []. "
              "verifier: '$.tool_name == \"Edit\"'."},

    {"prefix": "synth_edit_strnotuniq", "category": "edit_safety", "difficulty": "hard", "n": 8,
     "focus": "Failure mode edit_string_not_unique. The user supplies (or implies) an old_string that "
              "appears multiple times in the target file (repeated 'import x', 'const result =', empty "
              "function bodies, boilerplate header). Expected primary_action: Read or Grep first to "
              "disambiguate. must_avoid_actions includes {tool_name:'Edit', input_pattern_regex:'.*'}. "
              "must_include_keywords_in_reason includes 'unique' or 'context'."},
    {"prefix": "synth_edit_notread", "category": "edit_safety", "difficulty": "medium", "n": 6,
     "focus": "Failure mode edit_file_not_read. recent_actions does NOT contain a Read of the target "
              "file. Expected primary_action: Read of the exact target path. must_avoid_actions: "
              "{tool_name:'Edit', input_pattern_regex:'.*'} and {tool_name:'Write', input_pattern_regex:'.*'}. "
              "must_include_keywords_in_reason includes 'read' and 'before'."},
    {"prefix": "synth_hallucinated_skill", "category": "tool_routing", "difficulty": "hard", "n": 6,
     "focus": "Hallucinated skill avoidance. user_request asks for a skill BY NAME (e.g. 'use the "
              "deploy-rocket skill', 'invoke graph-render skill'). available_skills is EMPTY. The "
              "correct fallback is Read or Bash, NOT Skill with an invented name. Expected "
              "primary_action.tool_name in {Read, Bash, Glob}. must_avoid_actions includes "
              "{tool_name:'Skill', input_pattern_regex:'.*'}. must_avoid_keywords_in_reason includes "
              "the invented skill name."},
    {"prefix": "synth_path_adversarial", "category": "path_grounding", "difficulty": "hard", "n": 10,
     "focus": "Adversarial path grounding. user_request mentions a path that LOOKS plausible but is "
              "subtly wrong (e.g. 'src/utils/helper.ts' when only 'src/util/helper.ts' exists; "
              "'lib/effect/runtime.ts' vs 'libs/effect-runtime/index.ts'; trailing 's', wrong dir name, "
              "wrong extension). Expected primary_action: Glob or Bash 'ls'/'find' to verify. "
              "must_avoid_actions includes {tool_name:'Read', input_pattern_regex:'.*<the bad path>.*'}. "
              "Use varied repo roots (/Users/x/work/proj, /Users/y/side/repo, /home/dev/app)."},
]


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _samples(real: list[dict], cat: str, k: int = 3) -> list[dict]:
    pool = [r for r in real if r.get("category") == cat]
    return list(pool) if len(pool) <= k else random.sample(pool, k)


def _user_prompt(g: dict, examples: list[dict]) -> str:
    ex = "\n\n".join(json.dumps(e, ensure_ascii=False) for e in examples)
    return (
        "You are designing a benchmark for coding-agent action selection. Each task is one JSON "
        "object with this exact schema (keep field names verbatim):\n"
        "{\"id\": str, \"category\": str, \"difficulty\": \"easy|medium|hard\", "
        "\"source_record_ids\": [\"synthetic\"], "
        "\"prompt\": {\"user_request\": str, \"context\": {\"available_tools\": [str], "
        "\"recent_actions\": [str], \"available_skills\": [str]}}, "
        "\"expected\": {\"primary_action\": {\"tool_name\": str, \"input_pattern_regex\": str}, "
        "\"must_avoid_actions\": [{\"tool_name\": str, \"input_pattern_regex\": str}], "
        "\"must_include_keywords_in_reason\": [str], "
        "\"must_avoid_keywords_in_reason\": [str]}, "
        "\"verifier_kind\": \"structural_json|regex\", "
        "\"verifier_spec\": {\"type\": str, \"pattern_or_command\": str}, "
        "\"rubric_weight\": 1.0, \"human_readable_summary\": str}\n\n"
        f"Real-trace examples for category={g['category']} (for grounding only, do not copy):\n{ex}\n\n"
        f"Generate {g['n']} NEW benchmark tasks for category={g['category']}, difficulty={g['difficulty']}.\n"
        f"Focus: {g['focus']}\n\n"
        "Requirements: realistic and diverse (TS, Python, Rust, Go, Zig, sh); available_tools = a "
        "realistic Claude Code inventory (Read, Write, Edit, Bash, Glob, Grep, Skill, Task plus a few "
        "MCP names); available_skills = 0-8 plausibly named entries (use [] when the focus says so); "
        f"IDs prefixed with {g['prefix']}_ then 3-digit counter starting at 000; "
        "source_record_ids=[\"synthetic\"]; rubric_weight=1.0. "
        f"Return ONLY a JSON array of length exactly {g['n']}, no markdown, no commentary."
    )


def _parse(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j <= i:
        raise ValueError("no JSON array")
    return json.loads(text[i:j + 1])


def _validate(t: dict) -> tuple[bool, str]:
    if not isinstance(t, dict):
        return False, "not dict"
    miss = REQ - set(t.keys())
    if miss:
        return False, f"missing {sorted(miss)}"
    if t["difficulty"] not in {"easy", "medium", "hard"}:
        return False, "bad difficulty"
    p = t.get("prompt") or {}
    if not isinstance(p, dict) or "user_request" not in p or "context" not in p:
        return False, "bad prompt"
    ctx = p["context"]
    if not isinstance(ctx, dict) or PCTX - set(ctx.keys()):
        return False, "bad context"
    for k in PCTX:
        if not isinstance(ctx[k], list):
            return False, f"bad context.{k}"
    exp = t.get("expected") or {}
    pa = exp.get("primary_action") or {}
    if not isinstance(pa, dict) or "tool_name" not in pa:
        return False, "bad primary_action"
    if t["verifier_kind"] not in {"structural_json", "regex"}:
        return False, "bad verifier_kind"
    vs = t.get("verifier_spec") or {}
    if "type" not in vs or "pattern_or_command" not in vs:
        return False, "bad verifier_spec"
    return True, "ok"


def _normalize(t: dict, prefix: str, idx: int, used_ids: set[str]) -> dict:
    t["source_record_ids"] = ["synthetic"]
    t["rubric_weight"] = 1.0
    exp = t["expected"]
    exp.setdefault("must_avoid_actions", [])
    exp.setdefault("must_include_keywords_in_reason", [])
    exp.setdefault("must_avoid_keywords_in_reason", [])
    pa = exp.get("primary_action") or {}
    pa.setdefault("input_pattern_regex", ".*")
    cur = str(t.get("id", ""))
    if not cur.startswith(prefix) or cur in used_ids:
        # find next free counter for this prefix
        n = idx
        while f"{prefix}_{n:03d}" in used_ids:
            n += 1
        t["id"] = f"{prefix}_{n:03d}"
    return t


def _audit(per: Counter, accepted: list[dict], calls: int, wall: float, rej: int) -> str:
    diff_counts = Counter(t["difficulty"] for t in accepted)
    cat_counts = Counter(t["category"] for t in accepted)
    lines = ["# Synthetic benchmark tasks audit (FIX2)", "",
             f"- generated: {len(accepted)} tasks", f"- opus calls: {calls}",
             f"- wallclock_s: {wall:.1f}", f"- rejected (malformed): {rej}", "",
             "## Per-bucket distribution", ""]
    for k, v in per.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## By category", ""]
    for k, v in cat_counts.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## By difficulty", ""]
    for k, v in diff_counts.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Sample tasks (first 5)", ""]
    for s in accepted[:5]:
        lines += [f"### {s['id']}", "", "```json",
                  json.dumps(s, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    random.seed(0)
    real = _load(REAL)
    print(f"loaded {len(real)} real-trace tasks")

    # Preserve any prior synthetic output that is still well-formed.
    prior = _load(SYNTH)
    used_ids = {r["id"] for r in real} | {p["id"] for p in prior}
    accepted: list[dict] = list(prior)
    per: Counter = Counter()
    for t in prior:
        # crude bucket attribution from id prefix
        for g in GAPS:
            if t["id"].startswith(g["prefix"]):
                per[g["prefix"]] += 1
                break
    print(f"preserved {len(prior)} prior synthetic tasks")

    SYNTH.parent.mkdir(parents=True, exist_ok=True)
    out_fh = SYNTH.open("w")
    for r in prior:
        out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_fh.flush()

    t0 = time.time()
    calls = 0
    rej = 0
    sys_msg = "You generate benchmark tasks as compact JSON arrays. Output JSON only."

    for g in GAPS:
        if calls >= MAX_CALLS:
            print("max calls reached")
            break
        # Skip if this bucket already filled by prior run.
        if per.get(g["prefix"], 0) >= g["n"]:
            print(f"[{g['prefix']}] already full ({per[g['prefix']]}); skipping")
            continue
        examples = _samples(real, g["category"], 3)
        try:
            raw = llm.chat(messages=[{"role": "user", "content": _user_prompt(g, examples)}],
                           model=OPUS, max_tokens=MAX_TOKENS, system=sys_msg)
            calls += 1
        except Exception as e:
            print(f"[{g['prefix']}] call failed: {e}")
            continue
        try:
            arr = _parse(raw)
        except Exception as e:
            print(f"[{g['prefix']}] parse failed: {e}")
            rej += g["n"]
            continue
        kept = 0
        for idx, t in enumerate(arr):
            ok, why = _validate(t)
            if not ok:
                rej += 1
                continue
            t = _normalize(t, g["prefix"], idx, used_ids)
            if t["id"] in used_ids:
                rej += 1
                continue
            used_ids.add(t["id"])
            accepted.append(t)
            per[g["prefix"]] += 1
            kept += 1
            out_fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        out_fh.flush()
        print(f"[{g['prefix']}] kept {kept}/{len(arr)} (call#{calls}, total={per[g['prefix']]})")
    out_fh.close()
    wall = time.time() - t0

    # Concatenate real + synthetic, dedup by id (real wins).
    seen: set[str] = set()
    with FULL.open("w") as f:
        for r in real:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for s in accepted:
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    AUDIT.write_text(_audit(per, accepted, calls, wall, rej))
    print(f"\nDONE: total_synth={len(accepted)} new_calls={calls} rej={rej} wall={wall:.1f}s")
    print(f"distribution: {dict(per)}")
    print(f"wrote {SYNTH}\nwrote {FULL}\nwrote {AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
