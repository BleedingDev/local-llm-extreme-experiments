"""Run sanitise pipeline on all listed datasets and emit audit artefacts."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from agent_opt.sanitise import DEFAULT_RULES, sanitise_file
from agent_opt.sanitise_audit import audit_dir


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "sanitised"

TARGETS = [
    "dataset.jsonl",
    "dataset_v2.jsonl",
    "cc_dataset.jsonl",
    "cc_dataset_v2_new.jsonl",
    "codex_dataset.jsonl",
    "codex_dataset_v2_new.jsonl",
    "codex_gpt55_dataset.jsonl",
    "dataset_recovery.jsonl",
    "dataset_toolcalling.jsonl",
    "counterfactuals.jsonl",
    "benchmark_tasks_full.jsonl",
    "benchmark_tasks.jsonl",
    "benchmark_tasks_synthetic.jsonl",
    "planner_dataset.jsonl",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_in = total_out = 0
    grand_counts: Counter = Counter()
    proper_nouns: Counter = Counter()
    long_strings_audit: list[dict] = []
    per_file: dict[str, dict] = {}

    for name in TARGETS:
        src = DATA / name
        if not src.exists():
            per_file[name] = {"status": "missing"}
            continue
        dst = OUT / name
        n, counts, ib, ob = sanitise_file(
            src, dst, DEFAULT_RULES, long_strings_audit, proper_noun_collector=proper_nouns
        )
        per_file[name] = {
            "records": n,
            "in_bytes": ib,
            "out_bytes": ob,
            "delta_bytes": ob - ib,
            "counts": dict(counts),
        }
        grand_counts.update(counts)
        total_in += ib
        total_out += ob
        print(f"[done] {name}: {n} records, {ib}→{ob} bytes, counts={dict(counts)}")

    # Proper-noun audit — drop common English filler words & code-y tokens.
    DROP = {"Message", "File", "Permission", "Paper", "Codex", "Marketing"}
    candidates = sorted(
        ((w, c) for w, c in proper_nouns.items() if w not in DROP and c >= 2),
        key=lambda x: -x[1],
    )
    (OUT / "_proper_nouns_audit.json").write_text(
        json.dumps(
            {
                "_note": "Candidate proper nouns detected via 'Capitalised + verb' heuristic. "
                         "Human review required before adding to redaction list.",
                "candidates": [{"name": w, "count": c} for w, c in candidates],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUT / "_redacted_long_strings.json").write_text(
        json.dumps({"entries": long_strings_audit[:500], "total": len(long_strings_audit)}, indent=2),
        encoding="utf-8",
    )

    # Audit pass.
    audit_result = audit_dir(OUT)
    audit_pass = not audit_result["__total__"]
    elapsed = time.time() - started

    summary = {
        "wallclock_s": round(elapsed, 2),
        "files_processed": sum(1 for v in per_file.values() if "records" in v),
        "total_in_bytes": total_in,
        "total_out_bytes": total_out,
        "delta_bytes": total_out - total_in,
        "grand_counts": dict(grand_counts),
        "audit_pass": audit_pass,
        "audit_residuals": audit_result["__total__"],
        "proper_noun_candidates": len(candidates),
        "per_file": per_file,
    }
    (OUT / "_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# Sanitisation Audit Summary", ""]
    md.append(f"- Wallclock: {summary['wallclock_s']}s")
    md.append(f"- Files processed: {summary['files_processed']} / {len(TARGETS)}")
    md.append(f"- Total bytes: {total_in:,} -> {total_out:,} (delta {total_out-total_in:+,})")
    md.append(f"- Audit pass: {audit_pass}")
    md.append(f"- Proper-noun candidates: {len(candidates)}")
    md.append("")
    md.append("## Replacement counts")
    for k, v in sorted(grand_counts.items(), key=lambda x: -x[1]):
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Per-file")
    for name, info in per_file.items():
        if "records" not in info:
            md.append(f"- {name}: MISSING")
            continue
        md.append(f"- {name}: {info['records']} records, {info['delta_bytes']:+,} B delta")
    md.append("")
    md.append("## Audit residuals (must be all zero)")
    md.append("```json")
    md.append(json.dumps(audit_result["__total__"], indent=2))
    md.append("```")
    (OUT / "_audit_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=2))


if __name__ == "__main__":
    main()
