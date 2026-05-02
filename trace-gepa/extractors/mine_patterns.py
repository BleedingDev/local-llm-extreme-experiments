"""Frequent-subsequence miner for trace-gepa session traces.

Groups records by `src_path` (one session per file), tokenises each
`observed_action` into a stable alphabet, then mines length-3..10
frequent contiguous subsequences with a minimum support of 10
distinct sessions.

Token alphabet:
  - Non-Bash:   "<Tool>"                 e.g. "Read", "Edit"
  - Bash:       "Bash:<verb>"            verb = first token of cmd
  - Filtered:   noisy book-keeping tools (TodoWrite, TaskList, ...)
                are dropped before mining so they cannot dominate.

Output: trace-gepa/data/mined_patterns_top30.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [ROOT / "data" / "dataset.jsonl", ROOT / "data" / "dataset_v2.jsonl"]
OUT = ROOT / "data" / "mined_patterns_top30.json"

# Drop pure-bookkeeping actions: they form trivial repeated triples.
NOISE = {"TodoWrite", "TaskList", "TaskGet", "TaskUpdate"}
# Keep at most this many *distinct* sessions sampled per dataset to bound runtime.
SESSION_CAP = 5000
MIN_SUPPORT = 10
MIN_LEN = 3
MAX_LEN = 10
TOP_LEN3 = 80                  # seeds for greedy extension
TOP_PATTERNS_OUT = 30
EXAMPLES_PER_PATTERN = 3

VERB_RE = re.compile(r"[A-Za-z][\w./-]*")


def bash_verb(raw_input: str) -> str:
    """Extract a stable verb token from a Bash/exec_command input string."""
    try:
        parsed = json.loads(raw_input) if raw_input and raw_input[0] == "{" else {}
    except Exception:
        parsed = {}
    cmd = None
    if isinstance(parsed, dict):
        cmd = parsed.get("command") or parsed.get("cmd")
    if not cmd:
        cmd = raw_input or ""
    cmd = cmd.strip()
    # Strip a leading `cd <dir> && ...` (very common) so we count the real verb.
    cd_chain = re.match(r"cd\s+\S+\s*(?:&&|;)\s*(.+)", cmd)
    if cd_chain:
        cmd = cd_chain.group(1).strip()
    # For other chains, take the *last* meaningful command (closer to real intent).
    for sep in ("&&", "||", ";"):
        if sep in cmd:
            parts = [p.strip() for p in cmd.split(sep) if p.strip()]
            if parts:
                cmd = parts[-1]
    if "|" in cmd:
        cmd = cmd.split("|", 1)[0].strip()
    m = VERB_RE.match(cmd)
    return m.group(0).split("/")[-1] if m else "_unknown"


def tokenise(record: dict) -> str | None:
    act = record.get("observed_action") or {}
    name = act.get("name")
    if not name or name in NOISE:
        return None
    if name == "Bash":
        return f"Bash:{bash_verb(act.get('input') or '')}"
    if name == "exec_command":
        return f"Exec:{bash_verb(act.get('input') or '')}"
    return name


def load_sessions(paths: list[Path]) -> dict[str, list[str]]:
    sessions: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for p in paths:
        if not p.exists():
            continue
        with p.open("r") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                tok = tokenise(rec)
                if tok is None:
                    continue
                sid = rec.get("src_path") or "?"
                idx = rec.get("src_event_idx") or 0
                sessions[sid].append((idx, tok))
    out: dict[str, list[str]] = {}
    for sid, pairs in sessions.items():
        pairs.sort(key=lambda x: x[0])
        seq = [t for _, t in pairs]
        if len(seq) >= MIN_LEN:
            out[sid] = seq
    return out


def count_ngrams(sessions: dict[str, list[str]], n: int) -> Counter:
    """Distinct-session support counter for contiguous n-grams."""
    counter: Counter = Counter()
    for sid, seq in sessions.items():
        seen = set()
        for i in range(len(seq) - n + 1):
            seen.add(tuple(seq[i : i + n]))
        for k in seen:
            counter[k] += 1
    return counter


def example_sessions(sessions: dict[str, list[str]], pat: tuple[str, ...], k: int) -> list[str]:
    out = []
    for sid, seq in sessions.items():
        n = len(pat)
        for i in range(len(seq) - n + 1):
            if tuple(seq[i : i + n]) == pat:
                out.append(sid)
                break
        if len(out) >= k:
            break
    return out


def is_meaningful(pat: tuple[str, ...]) -> bool:
    """Workflow-quality gate: enough distinct concrete actions."""
    distinct = len(set(pat))
    if distinct < max(3, len(pat) // 2):
        return False
    # Drop patterns where >60% of tokens are bookkeeping bash verbs.
    bk = {"Bash:cd", "Bash:ls", "Bash:pwd", "Bash:echo"}
    if sum(1 for t in pat if t in bk) > 0.6 * len(pat):
        return False
    return True


def density(pat: tuple[str, ...], support: int) -> float:
    """Score that prefers meaningful, longer, well-supported patterns."""
    return support * len(set(pat)) * (len(pat) ** 0.5)


def mine() -> dict:
    random.seed(7)
    sessions = load_sessions(INPUTS)
    if len(sessions) > SESSION_CAP:
        keys = random.sample(list(sessions.keys()), SESSION_CAP)
        sessions = {k: sessions[k] for k in keys}
    print(f"loaded {len(sessions)} sessions", file=sys.stderr)

    # Length-3 seed mining.
    c3 = count_ngrams(sessions, 3)
    seeds = [pat for pat, sup in c3.most_common() if sup >= MIN_SUPPORT][:TOP_LEN3]

    found: dict[tuple[str, ...], int] = {}
    # Add length-3 seeds themselves.
    for s in seeds:
        found[s] = c3[s]

    # Greedy extension: for each seed, try to extend right by counting
    # n+1-grams that start with the seed (or include it as a prefix).
    frontier = list(seeds)
    for n in range(4, MAX_LEN + 1):
        cn = count_ngrams(sessions, n)
        kept_any = False
        for pat, sup in cn.most_common(2000):
            if sup < MIN_SUPPORT:
                break
            # Must extend something we already kept (prefix of length n-1).
            if pat[:-1] in found or pat[1:] in found:
                found[pat] = sup
                kept_any = True
        if not kept_any:
            break
        frontier = [p for p in cn if cn[p] >= MIN_SUPPORT]

    # Rank by information-density score.
    ranked = sorted(found.items(), key=lambda kv: density(kv[0], kv[1]), reverse=True)

    out_patterns = []
    for pat, sup in ranked:
        if not is_meaningful(pat):
            continue
        out_patterns.append(
            {
                "pattern": list(pat),
                "length": len(pat),
                "support": sup,
                "examples": example_sessions(sessions, pat, EXAMPLES_PER_PATTERN),
            }
        )
        if len(out_patterns) >= TOP_PATTERNS_OUT:
            break

    summary = {
        "n_sessions": len(sessions),
        "min_support": MIN_SUPPORT,
        "len_range": [MIN_LEN, MAX_LEN],
        "noise_filtered": sorted(NOISE),
        "patterns": out_patterns,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT} ({len(out_patterns)} patterns)", file=sys.stderr)
    return summary


if __name__ == "__main__":
    mine()
