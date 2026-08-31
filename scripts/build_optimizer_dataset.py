#!/usr/bin/env python3
"""Walk BAG trials under bench/jobs/ and emit a JSONL dataset for GEPA.

One record per trial. Read-only over jobs/. Stdlib only (json, pathlib,
argparse, re, sys). Skips non-BAG runs (Opus-direct / Claude-Code / oracle)
by detecting the absence of agent/bag-acp-summary.json. Tolerant of missing
fields: every optional value is null when not found. Idempotent: records
are sorted by trial_id before writing.

Usage:
    python3 scripts/build_optimizer_dataset.py \
        --jobs-dir bench/jobs \
        --output bench/.bag/optimizer/dataset.jsonl \
        [--include-job <substring>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

VERIFIER_TAIL_BYTES = 1000


def warn(msg: str) -> None:
    print(f"warn: {msg}", file=sys.stderr)


def load_json(path: Path) -> Any | None:
    """Load a JSON file or return None on error / missing."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"failed to parse {path}: {exc}")
        return None


def read_text_tail(path: Path, n_bytes: int) -> str | None:
    """Read the last n_bytes chars (utf-8 decoded, errors replaced)."""
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > n_bytes:
                fh.seek(size - n_bytes)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError as exc:
        warn(f"failed to read {path}: {exc}")
        return None


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warn(f"failed to read {path}: {exc}")
        return None


def parse_iso_seconds(start: str | None, end: str | None) -> float | None:
    """Compute end-start in seconds for ISO-8601 'Z' timestamps."""
    if not start or not end:
        return None
    try:
        # Python's fromisoformat handles 'Z' since 3.11.
        from datetime import datetime
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return (e - s).total_seconds()
    except (TypeError, ValueError):
        return None


def get_path(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def find_traces(trial_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate traces tarball and/or traces dir, if present.

    Tarballs: agent/bag-traces*.tar(.gz|.zst|.xz) or trial-root.
    Dirs: agent/bag-traces/ or trial-root/bag-traces/.
    """
    tarball: Path | None = None
    traces_dir: Path | None = None

    candidates_dir = [trial_dir / "agent" / "bag-traces", trial_dir / "bag-traces"]
    for c in candidates_dir:
        if c.is_dir():
            traces_dir = c
            break

    tar_re = re.compile(r"^bag-traces.*\.(tar(\.(gz|zst|xz|bz2))?|tgz|tzst)$")
    for parent in (trial_dir / "agent", trial_dir):
        if not parent.is_dir():
            continue
        try:
            for entry in parent.iterdir():
                if entry.is_file() and tar_re.match(entry.name):
                    tarball = entry
                    break
        except OSError:
            continue
        if tarball is not None:
            break

    return tarball, traces_dir


def load_routing(traces_dir: Path | None) -> dict[str, Any] | None:
    """Find a routing-decision.json under bag-traces/.bag/runs/<id>/.

    Returns the most recent (lexicographically last) one if multiple exist.
    """
    if traces_dir is None:
        return None
    runs_dir = traces_dir / ".bag" / "runs"
    if not runs_dir.is_dir():
        return None
    candidates: list[Path] = []
    try:
        for run_dir in runs_dir.iterdir():
            f = run_dir / "routing-decision.json"
            if f.is_file():
                candidates.append(f)
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.parent.name)
    data = load_json(candidates[-1])
    if not isinstance(data, dict):
        return None
    return {
        "shape": data.get("shape"),
        "mode": data.get("mode"),
        "confidence": data.get("confidence"),
        "reasoning": data.get("reasoning"),
    }


def extract_instruction(summary: dict[str, Any] | None, stdout_log: Path) -> str | None:
    """Prefer the ACP summary's 'task' field; fall back to bag-stdout.log JSON."""
    if isinstance(summary, dict):
        task = summary.get("task")
        if isinstance(task, str) and task.strip():
            return task
    text = read_text(stdout_log)
    if text:
        # First top-level JSON object (the structured replay header) often
        # contains "task" as the first field.
        m = re.search(r'"task"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            try:
                return json.loads('"' + m.group(1) + '"')
            except json.JSONDecodeError:
                return m.group(1)
    return None


def sum_tokens(llm_metrics: Any) -> tuple[int, int, int]:
    """Return (call_count, tokens_in, tokens_out)."""
    if not isinstance(llm_metrics, list):
        return 0, 0, 0
    n = 0
    pin = 0
    pout = 0
    for item in llm_metrics:
        if not isinstance(item, dict):
            continue
        n += 1
        pin += int(item.get("promptTokens") or 0)
        pout += int(item.get("completionTokens") or 0)
    return n, pin, pout


def build_record(trial_dir: Path, job_id: str) -> dict[str, Any] | None:
    result_path = trial_dir / "result.json"
    summary_path = trial_dir / "agent" / "bag-acp-summary.json"
    manifest_path = trial_dir / "agent" / "bag-manifest.json"
    config_path = trial_dir / "config.json"
    stdout_log = trial_dir / "agent" / "bag-stdout.log"
    verifier_stdout = trial_dir / "verifier" / "test-stdout.txt"
    verifier_reward = trial_dir / "verifier" / "reward.txt"

    # Spec: skip trials with no result.json.
    if not result_path.is_file():
        return None
    # Spec: skip Opus-direct runs (no bag-acp-summary.json). This also
    # filters Claude-Code, oracle, and any non-BAG agent.
    if not summary_path.is_file():
        return None

    result = load_json(result_path) or {}
    summary = load_json(summary_path) or {}
    manifest = load_json(manifest_path)  # may be None
    config = load_json(config_path) or {}

    trial_id = result.get("trial_name") or trial_dir.name
    task_name = result.get("task_name")

    agent_cfg = config.get("agent") if isinstance(config, dict) else {}
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
    kwargs = agent_cfg.get("kwargs") if isinstance(agent_cfg.get("kwargs"), dict) else {}
    bag_mode = kwargs.get("bag_mode")
    model = agent_cfg.get("model_name")

    reward = get_path(result, "verifier_result", "rewards", "reward")
    exception_info = result.get("exception_info")
    exception_type = None
    if isinstance(exception_info, dict):
        exception_type = exception_info.get("type") or exception_info.get(
            "exception_type"
        )

    wall_seconds = parse_iso_seconds(
        get_path(result, "agent_setup", "started_at"),
        get_path(result, "agent_execution", "finished_at"),
    )
    if wall_seconds is None:
        wall_seconds = parse_iso_seconds(
            result.get("started_at"), result.get("finished_at")
        )

    counts = summary.get("counts") if isinstance(summary, dict) else {}
    if not isinstance(counts, dict):
        counts = {}
    agent_summary = {
        "stop_reason": summary.get("stopReason") if isinstance(summary, dict) else None,
        "session_updates": counts.get("sessionUpdates"),
        "fs_read": counts.get("fsRead"),
        "fs_write": counts.get("fsWrite"),
        "terminal_create": counts.get("terminalCreate"),
        "tools_submitted": (
            summary.get("stopReason") == "end_turn"
            if isinstance(summary, dict)
            else None
        ),
    }

    manifest_block: dict[str, Any] | None = None
    if isinstance(manifest, dict):
        call_count, tokens_in, tokens_out = sum_tokens(manifest.get("llmMetrics"))
        manifest_block = {
            "run_id": manifest.get("runId"),
            "self_eval_score": get_path(manifest, "selfEvaluation", "score"),
            "llm_call_count": call_count,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    tarball, traces_dir = find_traces(trial_dir)
    routing = load_routing(traces_dir)

    verifier_block = {
        "stdout_tail": read_text_tail(verifier_stdout, VERIFIER_TAIL_BYTES),
        "reward_raw": (read_text(verifier_reward) or "").strip() or None,
    }

    instruction_text = extract_instruction(summary, stdout_log)

    record: dict[str, Any] = {
        "trial_id": trial_id,
        "task_name": task_name,
        "job_id": job_id,
        "bag_mode": bag_mode,
        "model": model,
        "reward": reward,
        "exception_type": exception_type,
        "wall_seconds": wall_seconds,
        "agent_summary": agent_summary,
        "manifest": manifest_block,
        "routing": routing,
        "verifier": verifier_block,
        "instruction_text": instruction_text,
        "source_paths": {
            "result": str(result_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
            "traces_tarball": str(tarball) if tarball else None,
            "traces_dir": str(traces_dir) if traces_dir else None,
        },
    }
    return record


def iter_trial_dirs(jobs_dir: Path, include: str | None) -> list[tuple[str, Path]]:
    """Yield (job_id, trial_dir) pairs."""
    out: list[tuple[str, Path]] = []
    if not jobs_dir.is_dir():
        return out
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        if include and include not in job_dir.name:
            continue
        for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
            out.append((job_dir.name, trial_dir))
    return out


def summarize(records: list[dict[str, Any]]) -> None:
    bag_mode_counts: Counter[str] = Counter()
    reward_counts: Counter[str] = Counter()
    routing_count = 0
    manifest_count = 0
    manifest_with_tokens = 0
    traces_dir_count = 0
    for r in records:
        bag_mode_counts[str(r.get("bag_mode"))] += 1
        rew = r.get("reward")
        if rew is None:
            reward_counts["None"] += 1
        elif isinstance(rew, (int, float)) and float(rew) >= 1.0:
            reward_counts["1"] += 1
        elif isinstance(rew, (int, float)) and float(rew) <= 0.0:
            reward_counts["0"] += 1
        else:
            reward_counts[f"other:{rew}"] += 1
        if r.get("routing"):
            routing_count += 1
        if r.get("manifest"):
            manifest_count += 1
            if (r["manifest"].get("tokens_in") or 0) > 0:
                manifest_with_tokens += 1
        if r.get("source_paths", {}).get("traces_dir"):
            traces_dir_count += 1

    print(f"total records: {len(records)}")
    print("by bag_mode:")
    for k, v in sorted(bag_mode_counts.items()):
        print(f"  {k}: {v}")
    print("by reward:")
    for k, v in sorted(reward_counts.items()):
        print(f"  {k}: {v}")
    print(f"routing populated: {routing_count}")
    print(f"manifest populated: {manifest_count} (tokens_in>0: {manifest_with_tokens})")
    print(f"traces_dir present: {traces_dir_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include-job",
        default=None,
        help="optional substring filter on job dir name",
    )
    args = parser.parse_args()

    jobs_dir: Path = args.jobs_dir.resolve()
    if not jobs_dir.is_dir():
        print(f"error: --jobs-dir does not exist: {jobs_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for job_id, trial_dir in iter_trial_dirs(jobs_dir, args.include_job):
        rec = build_record(trial_dir, job_id)
        if rec is None:
            continue
        records.append(rec)

    # Idempotent ordering.
    records.sort(key=lambda r: (r.get("trial_id") or "", r.get("job_id") or ""))

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=False))
            fh.write("\n")

    print(f"wrote {out_path} ({len(records)} records)")
    summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
