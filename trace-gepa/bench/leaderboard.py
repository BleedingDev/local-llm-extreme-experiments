#!/usr/bin/env python3
"""Leaderboard: ingest harness result files, rank, write Markdown + JSON.
Tolerates partial / malformed files; emits 'incomplete' rows instead of crashing.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

KNOWN_HARNESSES = ("anthropic", "codex", "mlx")
HARNESS_ALIASES = {"run_mlx": "mlx", "run_codex": "codex", "run_anthropic": "anthropic"}

def _safe_float(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None

def _safe_int(x: Any) -> int | None:
    try:
        return None if x is None else int(x)
    except (TypeError, ValueError):
        return None

def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def _infer_harness(raw: dict[str, Any], path: Path) -> str:
    h = HARNESS_ALIASES.get(str(raw.get("harness") or "").lower(),
                            str(raw.get("harness") or "").lower())
    if h in KNOWN_HARNESSES:
        return h
    name = path.name.lower()
    for cand in KNOWN_HARNESSES:
        if cand in name:
            return cand
    m = str(raw.get("model") or (raw.get("config") or {}).get("model") or "").lower()
    if "claude" in m:
        return "anthropic"
    if "gpt" in m or "codex" in m:
        return "codex"
    if any(k in m for k in ("mlx", "qwen", "llama", "gemma")):
        return "mlx"
    return h or "unknown"

def _bucket_synonyms(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, stats in (d or {}).items():
        if not isinstance(stats, dict):
            continue
        ms = stats.get("mean_score") if "mean_score" in stats else stats.get("pass_rate")
        cnt = stats.get("count") if "count" in stats else stats.get("n")
        out[name] = {"mean_score": _safe_float(ms), "count": _safe_int(cnt)}
    return out

def _build_overall(raw: dict[str, Any], summary: dict[str, Any],
                   config: dict[str, Any]) -> dict[str, Any] | None:
    """Build normalised overall block from union of harness shapes."""
    for ov in ((raw.get("aggregate") or {}).get("overall"), summary.get("overall")):
        if isinstance(ov, dict) and (ov.get("mean_score") is not None
                                      or ov.get("pass_rate") is not None):
            return {"mean_score": _safe_float(ov.get("mean_score") or ov.get("pass_rate")),
                    "count": _safe_int(ov.get("count") or ov.get("n")
                                       or ov.get("total_tasks")),
                    "wallclock_s": _safe_float(ov.get("wallclock_s")),
                    "lm_calls": _safe_int(ov.get("lm_calls")),
                    "input_tokens": _safe_int(ov.get("input_tokens")
                                              or ov.get("prompt_tokens_est")),
                    "output_tokens": _safe_int(ov.get("output_tokens")
                                               or ov.get("output_tokens_est"))}
    ms = _safe_float(summary.get("pass_rate") or raw.get("pass_rate"))
    if ms is None:
        return None
    cnt = _safe_int(summary.get("n") or raw.get("n_tasks") or raw.get("n_pass"))
    wall = _safe_float(config.get("elapsed_total_s") or raw.get("total_gen_seconds")
                       or raw.get("wallclock_s"))
    if wall is None and summary.get("mean_elapsed_s") and cnt:
        wall = float(summary["mean_elapsed_s"]) * cnt
    return {"mean_score": ms, "count": cnt, "wallclock_s": wall,
            "lm_calls": _safe_int(raw.get("lm_calls")),
            "input_tokens": _safe_int(raw.get("input_tokens")),
            "output_tokens": _safe_int(raw.get("total_gen_tokens")
                                       or raw.get("output_tokens"))}

def _normalize(path: Path, raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {"source_file": str(path), "tasks_file": None, "limit": None,
                "schema_version": None, "aggregate": {}, "n_results": 0,
                "harness": "unknown", "model": "unknown",
                "system_prompt_id": "unknown", "status": "incomplete",
                "reason": "unreadable_or_invalid_json"}
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    harness = _infer_harness(raw, path)
    overall = _build_overall(raw, summary, config)
    spec_agg = raw.get("aggregate") or {}
    by_cat = _bucket_synonyms(spec_agg.get("by_category") or summary.get("by_category")
                              or summary.get("per_category") or raw.get("by_category") or {})
    by_diff = _bucket_synonyms(spec_agg.get("by_difficulty") or summary.get("by_difficulty")
                               or summary.get("per_difficulty")
                               or raw.get("by_difficulty") or {})
    agg: dict[str, Any] = {}
    if overall is not None:
        agg["overall"] = overall
    if by_cat:
        agg["by_category"] = by_cat
    if by_diff:
        agg["by_difficulty"] = by_diff
    results = raw.get("results") or raw.get("per_task")
    if overall is None:
        status, reason = "incomplete", "missing_overall_score"
    elif harness not in KNOWN_HARNESSES:
        status, reason = "incomplete", f"unknown_harness:{harness}"
    else:
        status, reason = "ok", None
    return {"source_file": str(path), "harness": harness,
            "model": str(raw.get("model") or config.get("model")
                         or raw.get("task_model") or "unknown"),
            "system_prompt_id": str(raw.get("system_prompt_id") or raw.get("prompt_id")
                                    or config.get("system_prompt_id") or "seed"),
            "tasks_file": raw.get("tasks_file") or raw.get("tasks_path")
                          or config.get("tasks_path"),
            "limit": _safe_int(raw.get("limit") or config.get("limit")),
            "schema_version": raw.get("schema_version"),
            "status": status, "reason": reason, "aggregate": agg,
            "n_results": len(results) if isinstance(results, list) else 0}

def _ingest(results_dir: Path) -> list[dict[str, Any]]:
    if not results_dir.exists():
        return []
    return [_normalize(p, _read_json(p)) for p in sorted(results_dir.glob("*.json"))]

def _rank_overall(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        ov = (run.get("aggregate") or {}).get("overall") or {}
        ms = _safe_float(ov.get("mean_score"))
        wc = _safe_float(ov.get("wallclock_s"))
        cnt = _safe_int(ov.get("count")) or run["n_results"]
        cost = (ms / (wc / max(cnt, 1))
                if (ms is not None and wc and cnt and wc > 0) else None)
        rows.append({"harness": run["harness"], "model": run["model"],
                     "system_prompt_id": run["system_prompt_id"], "mean_score": ms,
                     "count": cnt, "wallclock_s": wc,
                     "lm_calls": _safe_int(ov.get("lm_calls")),
                     "input_tokens": _safe_int(ov.get("input_tokens")),
                     "output_tokens": _safe_int(ov.get("output_tokens")),
                     "cost_effectiveness": cost, "status": run["status"],
                     "source_file": run["source_file"]})
    rows.sort(key=lambda r: (r["mean_score"] is None, -(r["mean_score"] or 0.0)))
    return rows

def _rank_by_bucket(runs: list[dict[str, Any]], bucket: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        agg = (run.get("aggregate") or {}).get(bucket) or {}
        if not isinstance(agg, dict):
            continue
        for name, stats in agg.items():
            if not isinstance(stats, dict):
                continue
            out.setdefault(name, []).append({
                "harness": run["harness"], "model": run["model"],
                "system_prompt_id": run["system_prompt_id"],
                "mean_score": _safe_float(stats.get("mean_score") or stats.get("pass_rate")),
                "count": _safe_int(stats.get("count") or stats.get("n")),
                "source_file": run["source_file"]})
    for name in out:
        out[name].sort(key=lambda r: (r["mean_score"] is None, -(r["mean_score"] or 0.0)))
    return out

def _fmt_score(x: float | None) -> str: return "n/a" if x is None else f"{x:.4f}"
def _fmt_int(x: int | None) -> str: return "n/a" if x is None else str(x)
def _fmt_float(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |\n",
           "|" + "|".join("---" for _ in headers) + "|\n"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |\n")
    return "".join(out)

def _md_overall_table(rows: list[dict[str, Any]]) -> str:
    body = [[str(i), r["harness"], r["model"], r["system_prompt_id"],
             _fmt_score(r["mean_score"]), _fmt_int(r["count"]), _fmt_float(r["wallclock_s"]),
             _fmt_int(r["lm_calls"]), _fmt_int(r["input_tokens"]),
             _fmt_int(r["output_tokens"]), r["status"]]
            for i, r in enumerate(rows, 1)]
    return _md_table(["rank", "harness", "model", "system_prompt_id", "mean_score",
                      "count", "wallclock_s", "lm_calls", "in_tok", "out_tok", "status"], body)

def _md_bucket_section(title: str, buckets: dict[str, list[dict[str, Any]]]) -> str:
    if not buckets:
        return f"## {title}\n\n_No data._\n"
    parts = [f"## {title}\n"]
    for name in sorted(buckets):
        body = [[str(i), r["harness"], r["model"], r["system_prompt_id"],
                 _fmt_score(r["mean_score"]), _fmt_int(r["count"])]
                for i, r in enumerate(buckets[name], 1)]
        parts.append(f"### {name}\n\n" + _md_table(
            ["rank", "harness", "model", "system_prompt_id", "mean_score", "count"], body)
            + "\n")
    return "".join(parts)

def _md_cost_table(rows: list[dict[str, Any]]) -> str:
    eligible = sorted([r for r in rows if r["cost_effectiveness"] is not None],
                      key=lambda r: -(r["cost_effectiveness"] or 0.0))
    if not eligible:
        return "_No cost-effectiveness data (need wallclock_s + count + mean_score)._\n"
    body = [[str(i), r["harness"], r["model"], r["system_prompt_id"],
             _fmt_float(r["cost_effectiveness"], 6), _fmt_score(r["mean_score"]),
             _fmt_float(r["wallclock_s"]), _fmt_int(r["count"])]
            for i, r in enumerate(eligible, 1)]
    return _md_table(["rank", "harness", "model", "system_prompt_id",
                      "score_per_sec_per_task", "mean_score", "wallclock_s", "count"], body)

def _build_markdown(runs: list[dict[str, Any]], overall: list[dict[str, Any]],
                    by_cat: dict[str, list[dict[str, Any]]],
                    by_diff: dict[str, list[dict[str, Any]]]) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    incomplete = [r for r in runs if r["status"] != "ok"]
    seen = {r["harness"] for r in runs if r["status"] == "ok"}
    missing = [h for h in KNOWN_HARNESSES if h not in seen]
    parts = [f"# Trace-GEPA Bench Leaderboard\n_Generated: {ts}_\n",
             f"_Total runs ingested: {len(runs)} (ok: {len(runs) - len(incomplete)},"
             f" incomplete: {len(incomplete)})_\n\n## Overall ranking\n",
             _md_overall_table(overall),
             _md_bucket_section("Per-category rankings", by_cat),
             _md_bucket_section("Per-difficulty rankings", by_diff),
             "\n## Cost-effectiveness (mean_score / mean_wallclock_per_task)\n",
             _md_cost_table(overall), "\n## Footer\n"]
    parts.append(f"- Missing harnesses (no ok runs): {', '.join(missing)}\n"
                 if missing else "- All known harnesses represented.\n")
    if incomplete:
        parts.append("- Truncated / incomplete runs:\n")
        for r in incomplete:
            parts.append(f"  - `{r['source_file']}` harness=`{r['harness']}`"
                         f" model=`{r['model']}` reason=`{r['reason']}`\n")
    else:
        parts.append("- No truncated runs.\n")
    return "".join(parts)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trace-GEPA bench leaderboard builder.")
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    args = ap.parse_args(argv)
    runs = _ingest(args.results_dir)
    overall = _rank_overall(runs)
    by_cat = _rank_by_bucket(runs, "by_category")
    by_diff = _rank_by_bucket(runs, "by_difficulty")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_build_markdown(runs, overall, by_cat, by_diff), encoding="utf-8")
    seen_ok = {r["harness"] for r in runs if r["status"] == "ok"}
    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "results_dir": str(args.results_dir), "total_runs": len(runs),
        "ok_runs": sum(1 for r in runs if r["status"] == "ok"),
        "incomplete_runs": [r for r in runs if r["status"] != "ok"],
        "missing_harnesses": [h for h in KNOWN_HARNESSES if h not in seen_ok],
        "runs": runs,
        "rankings": {"overall": overall, "by_category": by_cat, "by_difficulty": by_diff}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"leaderboard: ingested {len(runs)} run(s) from {args.results_dir} "
          f"-> {args.output_md} + {args.output_json}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
