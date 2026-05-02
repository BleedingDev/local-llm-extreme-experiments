"""Heuristic persona fingerprint extractor.

Streams trace-gepa datasets, aggregates per-session signals, emits a
structured profile of *how this user codes* — tools, bash verbs +
flag combos, path priors, Czech corrective tokens, skills/subagents,
repo signals, and failure-recovery transformations.

Outputs:
  - agent_opt/persona/profile.json  (structured)
  - agent_opt/persona/profile.md    (human-readable)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "profile.json"
OUT_MD = OUT_DIR / "profile.md"
DATASETS = [DATA / "dataset.jsonl", DATA / "dataset_v2.jsonl"]
RECOVERY = DATA / "dataset_recovery.jsonl"
CZECH_TOKENS = ["nene", "počkej", "pozor", "ne tak", "spíš", "lepší", "fakt", "vůbec"]
PATH_PREFIXES = [
    "/Users/satan/side/experiments/", "/Users/satan/.codex/sessions/",
    "/Users/satan/.claude/", "~/.codex/sessions/", "~/.claude/",
    "Schaltwerk/", "/Users/satan/.config/",
]
_VERB_RE = re.compile(r"[A-Za-z][\w./-]*")
_REPO_RE = re.compile(r"/Users/satan/side/experiments/([A-Za-z0-9_.\-]+)")


def _load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _try_json(s: str) -> dict:
    try:
        return json.loads(s) if s and s[0:1] == "{" else {}
    except Exception:
        return {}


def _bash_cmd(raw) -> str:
    if not raw:
        return ""
    d = raw if isinstance(raw, dict) else _try_json(raw) if isinstance(raw, str) else {}
    if d:
        return (d.get("command") or d.get("cmd") or "").strip()
    return raw.strip() if isinstance(raw, str) else ""


def _leading_verb(cmd: str) -> str:
    c = cmd.strip()
    m = re.match(r"cd\s+\S+\s*(?:&&|;)\s*(.+)", c)
    if m:
        c = m.group(1).strip()
    for sep in ("&&", "||", ";", "|"):
        if sep in c:
            parts = [p.strip() for p in c.split(sep) if p.strip()]
            if parts:
                c = parts[0]
            break
    m = _VERB_RE.match(c)
    return m.group(0).split("/")[-1] if m else ""


def _flag_combo(cmd: str, max_args: int = 3) -> str:
    parts = cmd.split()
    if not parts:
        return ""
    out, n = [parts[0].split("/")[-1]], 0
    for p in parts[1:]:
        if n >= max_args:
            break
        if p.startswith("-") or ("/" not in p and "." not in p and len(p) <= 12):
            out.append(p)
            n += 1
    return " ".join(out)


def extract_profile():
    sessions = defaultdict(list)
    for ds in DATASETS:
        for rec in _load_jsonl(ds):
            sessions[rec.get("src_path") or "?"].append(rec)
    tool_hist, bash_verbs, path_hist = Counter(), Counter(), Counter()
    skill_hist, subagent_hist, repo_hist = Counter(), Counter(), Counter()
    czech_count, czech_hits = Counter(), []
    bash_combos = defaultdict(Counter)
    for sid, recs in sessions.items():
        m = _REPO_RE.search(sid)
        if m:
            repo_hist[m.group(1)] += 1
        for rec in recs:
            ctx = rec.get("context") or {}
            ur = ctx.get("user_request") or ""
            ur_lc = ur.lower()
            for tok in CZECH_TOKENS:
                if tok in ur_lc:
                    czech_count[tok] += 1
                    if len(czech_hits) < 12:
                        i = ur_lc.find(tok)
                        czech_hits.append({"token": tok,
                                           "snippet": ur[max(0, i - 40): i + 60].replace("\n", " ")})
            act = rec.get("observed_action") or {}
            name = act.get("name") or ""
            if not name:
                continue
            tool_hist[name] += 1
            inp = act.get("input")
            inp_s = inp if isinstance(inp, str) else (json.dumps(inp) if inp else "")
            for pb in PATH_PREFIXES:
                if pb in inp_s:
                    path_hist[pb] += 1
            rm = _REPO_RE.search(inp_s)
            if rm:
                repo_hist[rm.group(1)] += 1
            if name == "Bash":
                cmd = _bash_cmd(inp)
                v = _leading_verb(cmd)
                if v:
                    bash_verbs[v] += 1
                    bash_combos[v][_flag_combo(cmd)] += 1
            elif name in ("Task", "spawn_agent"):
                pj = _try_json(inp_s)
                st = (pj.get("subagent_type") or pj.get("agent_type") or "").strip()
                if st:
                    subagent_hist[st] += 1
            elif name == "Skill":
                sk = (_try_json(inp_s).get("skill") or "").strip()
                if sk:
                    skill_hist[sk] += 1
    recovery_pairs = Counter()
    for rec in _load_jsonl(RECOVERY):
        f = ((rec.get("failed_record") or {}).get("observed_action") or {}).get("name") or ""
        r = ((rec.get("recovery_record") or rec.get("recovered_record") or {}).get("observed_action") or {}).get("name") or ""
        if f and r:
            recovery_pairs[(f, r)] += 1
    profile = {
        "summary": {
            "n_sessions": len(sessions),
            "n_records": sum(len(v) for v in sessions.values()),
            "datasets": [str(d.relative_to(ROOT)) for d in DATASETS if d.exists()],
        },
        "tool_histogram_top10": tool_hist.most_common(10),
        "bash_verb_top20": bash_verbs.most_common(20),
        "bash_flag_combos_top5_per_verb": {
            v: bash_combos[v].most_common(5) for v, _ in bash_verbs.most_common(8)
        },
        "path_histogram": path_hist.most_common(),
        "language_signals": {"czech_token_counts": czech_count.most_common(),
                             "samples": czech_hits},
        "skill_top10": skill_hist.most_common(10),
        "subagent_top10": subagent_hist.most_common(10),
        "repo_top5": repo_hist.most_common(5),
        "recovery_top5": [{"failed": k[0], "recovered": k[1], "count": v}
                          for k, v in recovery_pairs.most_common(5)],
    }
    OUT_JSON.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    OUT_MD.write_text(_render_md(profile))
    return profile


def _render_md(p: dict) -> str:
    s = p["summary"]
    lines = [
        "# Persona Profile", "",
        f"- sessions: {s['n_sessions']}  records: {s['n_records']}",
        f"- datasets: {', '.join(s['datasets'])}", "",
    ]
    for title, items in [
        ("Tools (top 10)", p["tool_histogram_top10"]),
        ("Bash verbs (top 20)", p["bash_verb_top20"]),
        ("Path prefixes", p["path_histogram"]),
        ("Czech corrective tokens", p["language_signals"]["czech_token_counts"]),
        ("Skills (top 10)", p["skill_top10"]),
        ("Subagents (top 10)", p["subagent_top10"]),
        ("Repos (top 5)", p["repo_top5"]),
    ]:
        lines += [f"## {title}"] + [f"- {n}: {c}" for n, c in items] + [""]
    lines.append("## Bash flag combos (top 5 per top verb)")
    for verb, combos in p["bash_flag_combos_top5_per_verb"].items():
        lines.append(f"- **{verb}**: " + ", ".join(f"`{c}` x{n}" for c, n in combos))
    lines += ["", "## Failure-recovery (top 5)"] + [
        f"- `{r['failed']}` -> `{r['recovered']}` x{r['count']}" for r in p["recovery_top5"]
    ]
    if p["language_signals"]["samples"]:
        lines += ["", "## Czech sample contexts"] + [
            f"- `{x['token']}`: {x['snippet']!r}" for x in p["language_signals"]["samples"][:6]
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    pf = extract_profile()
    print(f"wrote {OUT_JSON} (sessions={pf['summary']['n_sessions']}, records={pf['summary']['n_records']})")
