"""Trace dataset sanitisation pipeline.

Streams JSONL records, recursively walks every string field, and applies a
configurable list of regex-based redactions. Designed for ~200 MB inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


# -- High-entropy heuristic ---------------------------------------------------

_HIGH_ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9_+/=\-]{40,}")


def _shannon_bits(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# -- Redaction rule -----------------------------------------------------------

@dataclass
class Redaction:
    name: str
    pattern: re.Pattern
    replacement: str


# Order matters: path prefixes first, then api keys, then private repos.
DEFAULT_RULES: list[Redaction] = [
    Redaction("path_users_satan", re.compile(r"/Users/satan(?=/|\b)"), "/Users/USER"),
    Redaction("path_tilde_satan", re.compile(r"~/satan(?=/|\b)"), "~/USER"),
    Redaction("env_home_satan", re.compile(r"\$HOME/satan(?=/|\b)"), "$HOME"),
    Redaction("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]+"), "<REDACTED_KEY>"),
    Redaction("openai_key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "<REDACTED_KEY>"),
    Redaction("github_token", re.compile(r"ghp_[A-Za-z0-9]{20,}"), "<REDACTED_GH_TOKEN>"),
    Redaction("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}"), "<REDACTED_HF_TOKEN>"),
    # Private repo names — match as whole-ish tokens (allow path/url contexts).
    Redaction("private_repo_1", re.compile(r"\bferndesk-connector\b"), "<PRIVATE_PROJECT_1>"),
    Redaction("private_repo_2", re.compile(r"\bcodex-trading\b"), "<PRIVATE_PROJECT_2>"),
    Redaction("private_repo_3", re.compile(r"\bkopac-do-zadku\b"), "<PRIVATE_PROJECT_3>"),
    Redaction("private_repo_4", re.compile(r"\bkrajta-strihac\b"), "<PRIVATE_PROJECT_4>"),
    Redaction("private_repo_5", re.compile(r"\bax-optimise-anything\b"), "<PRIVATE_PROJECT_5>"),
    Redaction("private_repo_6", re.compile(r"\bir-expo\b"), "<PRIVATE_PROJECT_6>"),
    Redaction("private_repo_7", re.compile(r"\bcodex-native\b"), "<PRIVATE_PROJECT_7>"),
]

CZECH_PROFANITY = re.compile(
    r"\b(kurva|do prdele|piča|pica|kokot|debil|hovno|sracka|svině|svine|zmrd|mrdat|jebat|jebnu|vole|kurvit)\b",
    re.IGNORECASE,
)


# -- Core sanitisation --------------------------------------------------------

def _apply_rules_to_string(
    s: str,
    rules: list[Redaction],
    counts: Counter,
    long_strings_audit: list[dict],
    src_id: str | None,
    scrub_profanity: bool,
) -> str:
    out = s
    for r in rules:
        new, n = r.pattern.subn(r.replacement, out)
        if n:
            counts[r.name] += n
            out = new
    if scrub_profanity:
        new, n = CZECH_PROFANITY.subn("<EMPHATIC>", out)
        if n:
            counts["czech_profanity"] += n
            out = new
    # High-entropy heuristic — flag, redact only if entropy > 4.5 bits/char.
    for m in _HIGH_ENTROPY_TOKEN.finditer(out):
        tok = m.group(0)
        if tok.startswith(("<REDACTED", "<PRIVATE")):
            continue
        ent = _shannon_bits(tok)
        long_strings_audit.append({
            "src_id": src_id,
            "len": len(tok),
            "entropy_bits": round(ent, 3),
            "preview": tok[:12] + "..." + tok[-4:],
            "redacted": ent > 4.5,
        })
    if any(a["redacted"] and a["src_id"] == src_id for a in long_strings_audit[-8:]):
        def _maybe_redact(m: re.Match) -> str:
            tok = m.group(0)
            if tok.startswith(("<REDACTED", "<PRIVATE")):
                return tok
            if _shannon_bits(tok) > 4.5:
                counts["high_entropy"] += 1
                return "<REDACTED_HIGH_ENTROPY>"
            return tok
        out = _HIGH_ENTROPY_TOKEN.sub(_maybe_redact, out)
    return out


def sanitise_record(
    rec: Any,
    rules: list[Redaction],
    counts: Counter,
    long_strings_audit: list[dict],
    src_id: str | None = None,
    scrub_profanity: bool = False,
) -> Any:
    if isinstance(rec, str):
        return _apply_rules_to_string(rec, rules, counts, long_strings_audit, src_id, scrub_profanity)
    if isinstance(rec, list):
        return [sanitise_record(x, rules, counts, long_strings_audit, src_id, scrub_profanity) for x in rec]
    if isinstance(rec, dict):
        sid = rec.get("id", src_id) if src_id is None else src_id
        return {
            k: sanitise_record(v, rules, counts, long_strings_audit, sid, scrub_profanity)
            for k, v in rec.items()
        }
    return rec


# -- File streaming -----------------------------------------------------------

def sanitise_file(
    in_path: Path,
    out_path: Path,
    rules: list[Redaction],
    long_strings_audit: list[dict],
    proper_noun_collector: Counter | None = None,
    scrub_profanity: bool = False,
) -> tuple[int, Counter, int, int]:
    counts: Counter = Counter()
    n_records = 0
    in_bytes = 0
    out_bytes = 0
    proper_pat = re.compile(
        r"\b([A-Z][a-z]{2,15})\s+(said|wrote|asked|told|sent|replied|mentioned|says|writes|noted|added|requested|emailed|messaged|called|reported|suggested)\b"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            in_bytes += len(line.encode("utf-8"))
            line = line.rstrip("\n")
            if not line:
                fout.write("\n")
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                continue
            if proper_noun_collector is not None:
                # Walk strings cheaply for proper-noun audit.
                stack = [rec]
                while stack:
                    v = stack.pop()
                    if isinstance(v, str):
                        for m in proper_pat.finditer(v):
                            proper_noun_collector[m.group(1)] += 1
                    elif isinstance(v, dict):
                        stack.extend(v.values())
                    elif isinstance(v, list):
                        stack.extend(v)
            sanitised = sanitise_record(rec, rules, counts, long_strings_audit, scrub_profanity=scrub_profanity)
            ser = json.dumps(sanitised, ensure_ascii=False)
            out_bytes += len(ser.encode("utf-8")) + 1
            fout.write(ser + "\n")
            n_records += 1
    return n_records, counts, in_bytes, out_bytes


# -- CLI ----------------------------------------------------------------------

def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scrub-profanity", action="store_true")
    args = ap.parse_args()
    audit: list[dict] = []
    n, counts, ib, ob = sanitise_file(
        Path(args.input), Path(args.output), DEFAULT_RULES, audit, scrub_profanity=args.scrub_profanity
    )
    print(json.dumps({"records": n, "counts": dict(counts), "in_bytes": ib, "out_bytes": ob}))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
