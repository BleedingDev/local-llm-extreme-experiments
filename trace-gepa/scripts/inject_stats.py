#!/usr/bin/env python3
"""Inject computed dataset stats between the AUTO-FILLED markers in README.md.

Idempotent: re-running replaces the previous block, never appends.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
README = ROOT / "README.md"

# categorize.py-tagged categories (worth calling out in a footnote)
TAGGED_BY_CATEGORIZE = {
    "hallucinated_path", "hallucinated_skill",
    "retry_loop", "user_correction", "subagent_terse_prompt",
}

LABEL_ORDER = ["good", "bad", "user_confirmed", "user_corrected"]


def load_dataset(path: Path):
    src = Counter(); labels = Counter(); cats = Counter(); total = 0
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            total += 1
            src[r.get("src", "?")] += 1
            labels[r.get("label", "?")] += 1
            cats[r.get("failure_category")] += 1
    return total, src, labels, cats


def date_range(seed_path: Path):
    with seed_path.open() as f:
        seed = json.load(f)
    days = set()
    for s in seed.get("codex_sessions", []):
        m = re.search(r"rollout-(\d{4}-\d{2}-\d{2})", s.get("path", ""))
        if m:
            days.add(m.group(1))
    for s in seed.get("cc_sessions", []):
        p = s.get("path")
        if p and os.path.exists(p):
            days.add(datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc).strftime("%Y-%m-%d"))
    return (min(days), max(days)) if days else (None, None)


def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.1f}%" if d else "0.0%"


def build_block(total, src, labels, cats, splits, dmin, dmax) -> str:
    lines = []
    lines.append("**Source counts** (total: {})".format(total))
    lines.append("")
    lines.append("| source | count |")
    lines.append("|---|---|")
    for k in sorted(src):
        lines.append(f"| {k} | {src[k]} |")
    lines.append("")
    lines.append("**Label distribution**")
    lines.append("")
    lines.append("| label | count | pct |")
    lines.append("|---|---|---|")
    for k in LABEL_ORDER:
        if k in labels:
            lines.append(f"| {k} | {labels[k]} | {pct(labels[k], total)} |")
    lines.append("")
    lines.append("**Top 8 failure categories** (excluding `null`)")
    lines.append("")
    lines.append("| category | count |")
    lines.append("|---|---|")
    non_null = [(k, v) for k, v in cats.items() if k is not None]
    top = sorted(non_null, key=lambda kv: -kv[1])[:8]
    for k, v in top:
        suffix = " *(tagged by categorize.py)*" if k in TAGGED_BY_CATEGORIZE else ""
        lines.append(f"| `{k}`{suffix} | {v} |")
    lines.append("")
    lines.append("**Splits**")
    lines.append("")
    counts = splits["counts"]
    ld = splits["label_distribution"]
    header_labels = LABEL_ORDER
    lines.append("| split | total | " + " | ".join(header_labels) + " |")
    lines.append("|" + "---|" * (2 + len(header_labels)))
    for sp in ("train", "val", "test"):
        row = [sp, str(counts[sp])] + [str(ld[sp].get(lbl, 0)) for lbl in header_labels]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if dmin and dmax:
        lines.append(f"**Date range (approx, from seed source files):** {dmin} to {dmax}. See `data/seed_sessions.json` for per-session detail.")
    else:
        lines.append("**Date range:** see `data/seed_sessions.json`.")
    return "\n".join(lines)


# Marker pattern: replace everything between the first <!-- AUTO-FILLED --> line
# and the next <!-- AUTO-FILLED --> line (inclusive of helper comments between).
MARKER = "<!-- AUTO-FILLED -->"


def inject(readme_text: str, block: str) -> str:
    lines = readme_text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == MARKER]
    if len(starts) >= 2:
        i, j = starts[0], starts[1]
        new = lines[: i + 1] + [""] + block.splitlines() + [""] + lines[j:]
        return "\n".join(new) + ("\n" if readme_text.endswith("\n") else "")
    if len(starts) == 1:
        i = starts[0]
        # find next blank-then-heading or just next ## heading
        j = i + 1
        while j < len(lines) and not lines[j].startswith("## "):
            j += 1
        new = lines[: i + 1] + [""] + block.splitlines() + ["", MARKER, ""] + lines[j:]
        return "\n".join(new) + ("\n" if readme_text.endswith("\n") else "")
    raise SystemExit("no AUTO-FILLED marker found in README.md")


def main():
    total, src, labels, cats = load_dataset(DATA / "dataset.jsonl")
    with (DATA / "splits.json").open() as f:
        splits = json.load(f)
    dmin, dmax = date_range(DATA / "seed_sessions.json")
    block = build_block(total, src, labels, cats, splits, dmin, dmax)
    text = README.read_text()
    new = inject(text, block)
    if new != text:
        README.write_text(new)
        print(f"updated {README}")
    else:
        print(f"no change to {README}")


if __name__ == "__main__":
    main()
