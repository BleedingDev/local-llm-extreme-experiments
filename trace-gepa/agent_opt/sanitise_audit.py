"""Audit sanitised JSONL outputs — assert no leaked secrets, count violations."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


CHECKS = [
    ("path_users_satan", re.compile(r"/Users/satan(?=/|\b)")),
    ("path_flat_users_satan", re.compile(r"-Users-satan(?=-|\b)")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]+")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("github_token", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("repo_ferndesk", re.compile(r"\bferndesk-connector\b")),
    ("repo_codex_trading", re.compile(r"\bcodex-trading\b")),
    ("repo_kopac", re.compile(r"\bkopac-do-zadku\b")),
    ("repo_krajta", re.compile(r"\bkrajta-strihac\b")),
    ("repo_ax_optimise", re.compile(r"\bax-optimise-anything\b")),
    ("repo_ir_expo", re.compile(r"\bir-expo\b")),
    ("repo_codex_native", re.compile(r"\bcodex-native\b")),
]


def audit_file(path: Path) -> Counter:
    c: Counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            for name, pat in CHECKS:
                if pat.search(line):
                    c[name] += len(pat.findall(line))
    return c


def audit_dir(d: Path) -> dict:
    out = {}
    overall: Counter = Counter()
    for p in sorted(d.glob("*.jsonl")):
        c = audit_file(p)
        out[p.name] = dict(c)
        overall.update(c)
    out["__total__"] = dict(overall)
    return out


if __name__ == "__main__":
    d = Path(sys.argv[1])
    res = audit_dir(d)
    print(json.dumps(res, indent=2))
    if res["__total__"]:
        sys.exit(1)
