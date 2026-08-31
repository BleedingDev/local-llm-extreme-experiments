#!/usr/bin/env python3
"""bag_audit_run.py — Vacuous-pass audit CLI for harbor-style job directories.

Usage:
    python scripts/bag_audit_run.py bench/jobs/2026-05-02__09-45-38/
    python scripts/bag_audit_run.py --json bench/jobs/2026-05-02__09-45-38/
    python scripts/bag_audit_run.py --jsonl bench/jobs/2026-*

The pretty-printed default lists every vacuous pass and the headline-vs-effective
delta. `--json` emits the full rollup + per-trial array for machine parsing.
`--jsonl` emits one trial audit per line, suitable for piping into jq.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `bench/` importable when the script is run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT / "bench"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from audit.vacuous_pass import audit_job, effective_score  # noqa: E402


def _format_pretty(rollup: dict, audits: list[dict]) -> str:
    job = rollup["job_dir"]
    lines: list[str] = []
    lines.append(f"=== audit: {job} ===")
    lines.append(
        f"trials: {rollup['n_trials']}  scored: {rollup['n_scored']}  "
        f"real_wins: {rollup['n_real_wins']}  "
        f"vacuous_passes: {rollup['n_vacuous_passes']}  "
        f"real_losses: {rollup['n_real_losses']}"
    )
    lines.append(
        f"headline_score: {rollup['headline_score']:.4f}  "
        f"effective_score: {rollup['effective_score']:.4f}  "
        f"delta: {rollup['delta']:+.4f}"
    )
    if rollup["vacuous_trials"]:
        lines.append("vacuous trials (reward=1, no real agent work):")
        # Index audits by trial for stop-reason context.
        index = {a["trial"]: a for a in audits}
        for trial in rollup["vacuous_trials"]:
            entry = index.get(trial, {})
            stop = entry.get("stop_reason") or "<no stop reason recorded>"
            ev = entry.get("evidence") or {}
            files = ", ".join(ev.get("trace_files_present") or []) or "<none>"
            lines.append(
                f"  - {trial}  stop={stop}  turns={ev.get('turns_used')} "
                f"tools={ev.get('tool_calls')}  trace_files=[{files}]"
            )
    else:
        lines.append("no vacuous passes detected.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a harbor-style benchmark job for vacuous passes.",
    )
    parser.add_argument(
        "job_dirs",
        nargs="+",
        type=Path,
        help="One or more job directories (e.g. bench/jobs/2026-05-02__09-45-38).",
    )
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object with rollup + per-trial audits.",
    )
    fmt.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one JSON line per trial audit (no rollup line).",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    aggregate: list[dict] = []

    for job_dir in args.job_dirs:
        if not job_dir.is_dir():
            print(f"warning: {job_dir} is not a directory; skipping", file=sys.stderr)
            exit_code = 2
            continue

        audits = audit_job(job_dir)
        rollup = effective_score(job_dir)

        if args.jsonl:
            for entry in audits:
                print(json.dumps(entry, sort_keys=True))
        elif args.json:
            aggregate.append({"rollup": rollup, "audits": audits})
        else:
            print(_format_pretty(rollup, audits))
            print()

    if args.json:
        if len(aggregate) == 1:
            print(json.dumps(aggregate[0], indent=2, sort_keys=True))
        else:
            print(json.dumps(aggregate, indent=2, sort_keys=True))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
