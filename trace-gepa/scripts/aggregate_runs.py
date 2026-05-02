"""Aggregate every GEPA optimisation run into a single canonical report.

Walks `artifacts/optimized-prompts/` (relative to repo root), reads
`run_meta.json` from each completed run dir, captures prompt size + excerpt,
optionally attaches benchmark results from `trace-gepa/bench/results_*.json`,
and emits a Markdown table to `trace-gepa/REPORT.md`.

Bench matching: a bench JSON's `prompt_b_source` typically points at
`latest/best_candidate.system.md`, which silently re-targets whenever the
`latest` symlink is bumped. Following that symlink at report-time produces
false matches (the run that was actually benched is no longer `latest`).
We instead match on **prompt content equality**: the bench JSON's
`prompt_b_chars` (length proxy for content; existing bench files don't embed
the content) must equal the byte-length of the run's `best_candidate.system.md`.
If multiple runs collide on length, prefer the run whose prompt mtime is closest
to (and not after) the bench JSON's mtime.

Usage:
    python aggregate_runs.py                 # refresh REPORT.md
    python aggregate_runs.py --json out.json # also dump machine-readable summary

Read-only on run dirs and dataset.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "artifacts" / "optimized-prompts"
BENCH_DIR = REPO_ROOT / "trace-gepa" / "bench"
REPORT_PATH = REPO_ROOT / "trace-gepa" / "REPORT.md"
RUN_DIR_PATTERN = re.compile(r"^([a-z0-9]+_)?run_\d{8}T\d{6}Z$")
SEED_PREFIXES = (("bag_run_", "bag"), ("codex_run_", "codex"), ("v2_run_", "v2"), ("xl_run_", "xl"))


def _seed_module(name: str) -> str:
    for prefix, label in SEED_PREFIXES:
        if name.startswith(prefix):
            return label
    return "default" if name.startswith("run_") else "unknown"


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and bool(RUN_DIR_PATTERN.match(p.name))


def _load_bench_index() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not BENCH_DIR.exists():
        return out
    for path in sorted(BENCH_DIR.glob("results_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append({"path": path, "mtime": mtime, "summary": data.get("summary") or data})
    return out


def _assign_benches(
    rows: list[dict[str, Any]],
    bench_index: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assign each bench JSON to AT MOST ONE run, by content-length match.

    For every bench file with a usable `prompt_b_chars`, find every run whose
    `best_candidate.system.md` has the same byte-length AND whose mtime is
    not after the bench's mtime (the run had to exist before being benched).
    Pick the run with the smallest non-negative `bench_mtime - run_mtime` gap.
    Ties broken by run_id ordering. Each bench file maps to one run; each
    run can collect multiple benches.
    """
    assignments: dict[str, dict[str, Any]] = {}  # run_id -> bench entry
    for entry in bench_index:
        bench_chars = entry["summary"].get("prompt_b_chars")
        if not isinstance(bench_chars, int) or bench_chars <= 0:
            continue
        bench_mtime = entry.get("mtime") or 0.0
        best: tuple[float, str] | None = None
        for r in rows:
            if r.get("prompt_chars") != bench_chars:
                continue
            run_mtime = r.get("_prompt_mtime") or 0.0
            if run_mtime <= 0.0:
                continue
            if run_mtime > bench_mtime + 1.0:
                continue
            gap = bench_mtime - run_mtime
            key = (gap, r["run_id"])
            if best is None or key < best:
                best = key
        if best is not None:
            run_id = best[1]
            # First bench wins per run; if a later (older-or-equal-gap) bench
            # collides, keep the earliest-assigned for stable output.
            assignments.setdefault(run_id, entry)
    return assignments


def collect_runs() -> tuple[list[dict[str, Any]], list[str]]:
    if not RUN_ROOT.exists():
        return [], []
    bench_index = _load_bench_index()
    rows: list[dict[str, Any]] = []
    in_flight: list[str] = []
    for child in sorted(RUN_ROOT.iterdir()):
        if not _is_run_dir(child):
            continue
        meta_path = child / "run_meta.json"
        if not meta_path.exists():
            in_flight.append(child.name)
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        prompt_path = child / "best_candidate.system.md"
        prompt_chars, excerpt = 0, ""
        prompt_mtime = 0.0
        if prompt_path.exists():
            text = prompt_path.read_text()
            prompt_chars = len(prompt_path.read_bytes())
            excerpt = text[:200].replace("\n", " ").strip()
            try:
                prompt_mtime = prompt_path.stat().st_mtime
            except OSError:
                prompt_mtime = 0.0
        before, after = meta.get("val_score_before"), meta.get("val_score_after")
        delta = meta.get("delta")
        if delta is None and before is not None and after is not None:
            delta = after - before
        rows.append({
            "run_id": child.name,
            "seed_module": _seed_module(child.name),
            "budget": meta.get("budget"),
            "train": meta.get("train_size"),
            "val": meta.get("val_size"),
            "val_before": before,
            "val_after": after,
            "delta": delta,
            "prompt_chars": prompt_chars,
            "_prompt_mtime": prompt_mtime,  # internal: dropped before serialisation
            "wallclock_s": meta.get("elapsed_seconds"),
            "bench_overall_pass": None,
            "bench_source": None,
            "prompt_excerpt": excerpt,
            "task_model": meta.get("task_model"),
            "reflection_model": meta.get("reflection_model"),
            "error": meta.get("error"),
        })

    assignments = _assign_benches(rows, bench_index)
    for r in rows:
        bench = assignments.get(r["run_id"])
        if bench is not None:
            r["bench_overall_pass"] = bench["summary"].get("pass_rate_b")
            r["bench_source"] = bench["path"].name
        r.pop("_prompt_mtime", None)

    rows.sort(key=lambda r: (r["val_after"] if r["val_after"] is not None else -1.0), reverse=True)
    return rows, in_flight


def _fmt(value: Any, fmt: str = ".4f") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{'+' if value >= 0 else ''}{value:.4f}"


def render_markdown(rows: list[dict[str, Any]], in_flight: list[str]) -> str:
    out = [
        "# GEPA Optimisation Run Comparison\n\n",
        "Auto-generated by `trace-gepa/scripts/aggregate_runs.py`. Re-run that script to refresh.\n\n",
    ]
    if not rows:
        out.append("_No completed runs found in `artifacts/optimized-prompts/`._\n")
    else:
        out.append(
            "| run_id | seed_module | budget | train | val | val_before | val_after | delta |"
            " prompt_chars | wallclock_s | bench_overall_pass | notes |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
        for r in rows:
            note_bits: list[str] = []
            if r.get("error"):
                note_bits.append(f"error: {r['error']}")
            if r.get("bench_source"):
                note_bits.append(f"bench={r['bench_source']}")
            if r.get("prompt_chars") == 0:
                note_bits.append("no prompt artefact")
            note = "; ".join(note_bits) if note_bits else "-"
            out.append(
                f"| {r['run_id']} | {r['seed_module']} | {_fmt(r['budget'])} | {_fmt(r['train'])} |"
                f" {_fmt(r['val'])} | {_fmt(r['val_before'])} | {_fmt(r['val_after'])} |"
                f" {_fmt_delta(r['delta'])} | {_fmt(r['prompt_chars'])} |"
                f" {_fmt(r['wallclock_s'], '.1f')} | {_fmt(r['bench_overall_pass'])} | {note} |\n"
            )
        best = rows[0]
        out.append(
            "\n## Winning configuration so far\n\n"
            f"`{best['run_id']}` (seed_module=`{best['seed_module']}`) leads with "
            f"val_score_after={_fmt(best['val_after'])} (delta={_fmt_delta(best['delta'])}, "
            f"budget={_fmt(best['budget'])}, train={_fmt(best['train'])}, val={_fmt(best['val'])}, "
            f"prompt_chars={best['prompt_chars']}, wallclock={_fmt(best['wallclock_s'], '.1f')}s).\n"
        )
        if best.get("prompt_excerpt"):
            out.append(f"\nPrompt excerpt (first 200 chars):\n\n> {best['prompt_excerpt']}\n")
    if in_flight:
        out.append("\n## In-flight runs (no `run_meta.json` yet)\n\n")
        out.append("\n".join(f"- {name}" for name in in_flight))
        out.append("\n\nRe-run this aggregator after they finish to fold them into the table.\n")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out", default=None, help="Optional path for machine-readable JSON summary.")
    parser.add_argument("--report", dest="report_path", default=str(REPORT_PATH), help="Override REPORT.md path.")
    args = parser.parse_args()
    rows, in_flight = collect_runs()
    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(rows, in_flight))
    print(f"[aggregate_runs] wrote {report} ({len(rows)} completed, {len(in_flight)} in-flight)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows, "in_flight": in_flight}, indent=2, default=str))
        print(f"[aggregate_runs] wrote {args.json_out}")


if __name__ == "__main__":
    main()
