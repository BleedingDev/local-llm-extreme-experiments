"""Codex CLI benchmark harness.

Runs Phase-3 benchmark tasks through the local `codex` CLI binary in
non-interactive (`codex exec --json`) mode. Mirrors the Anthropic harness
shape: load tasks, fan out with a thread pool, capture per-task latency +
exit-code + parsed prediction, score with `bench.verifiers.verify`.

Usage:
    .venv-gepa/bin/python trace-gepa/bench/run_codex.py \
        --tasks trace-gepa/data/benchmark_tasks.jsonl \
        --model gpt-5.5 --reasoning xhigh \
        --max-workers 4 \
        --output trace-gepa/bench/results/codex_smoke.json \
        --limit 5

If the `codex` CLI is missing, or if the first task fails because of an auth
problem, every remaining task is marked `error="codex_unavailable"` and given
a score of 0 instead of attempting the rest.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.verifiers import verify  # noqa: E402


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[\s\S]*\}")


_DEFAULT_INSTR = (
    "You are evaluating a single agent step.\n"
    "Pick the SINGLE best next action for the user request below.\n"
    "Respond with ONE JSON OBJECT ONLY, no prose, no fences.\n"
    "Schema: {\"tool_name\": <string>, \"input\": <object>, \"reason\": <string>}.\n"
    "- `tool_name` MUST be one of `available_tools` if provided.\n"
    "- `input` is the JSON arguments you would pass to that tool.\n"
    "- `reason` is one short sentence.\n"
    "Do not invoke or execute the tool yourself. Output JSON only."
)


def _build_prompt(task: dict) -> str:
    """Render a task into a single text prompt for codex exec."""
    prompt_obj = task.get("prompt") or {}
    user_request = prompt_obj.get("user_request") or task.get("user_request") or ""
    ctx = prompt_obj.get("context") or {}
    available_tools = ctx.get("available_tools") or []
    recent_actions = ctx.get("recent_actions") or []
    recent_results = ctx.get("recent_tool_results") or []

    lines: list[str] = [_DEFAULT_INSTR, ""]
    lines.append(f"Task category: {task.get('category', 'unknown')}")
    if available_tools:
        lines.append("Available tools: " + ", ".join(map(str, available_tools[:60])))
    if recent_actions:
        lines.append("Recent actions:")
        for a in recent_actions[-6:]:
            lines.append(f"  - {a}")
    if recent_results:
        lines.append("Recent tool results (truncated):")
        for r in recent_results[-3:]:
            lines.append(f"  - {str(r)[:300]}")
    lines.append("")
    lines.append("User request:")
    lines.append(str(user_request)[:6000])
    lines.append("")
    lines.append("Return ONE JSON object only.")
    return "\n".join(lines)


def _extract_final_message(stdout: str) -> tuple[str, str]:
    """From a codex --json stream, return (final_text, parser_status)."""
    final_text = ""
    saw_completed = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type") or ""
        if t == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message":
                txt = item.get("text") or ""
                if txt:
                    final_text = txt
        if t == "turn.completed":
            saw_completed = True
    if final_text:
        return final_text, "ok"
    if saw_completed:
        return "", "no_final_message"
    return "", "no_turn_completed"


def _parse_predicted(final_text: str) -> tuple[Any, str]:
    """Try to extract a JSON object from the agent's final message."""
    if not final_text:
        return None, "empty"
    s = final_text.strip()
    # 1) bare object?
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s), "json_direct"
        except Exception:
            pass
    # 2) fenced ```json ... ```
    m = _FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1)), "json_fenced"
        except Exception:
            pass
    # 3) any {...} substring (greedy, may over-grab; try smallest valid)
    candidates = _BARE_JSON_RE.findall(s)
    for cand in candidates:
        try:
            return json.loads(cand), "json_substring"
        except Exception:
            continue
    # 4) fall back to raw text
    return s, "raw_text"


def _run_codex(
    prompt: str,
    *,
    codex_bin: str,
    model: str,
    reasoning: str | None,
    timeout_s: float,
    cwd: Path,
) -> dict:
    """Invoke codex exec --json non-interactively. Returns metadata + stdout."""
    cmd = [
        codex_bin, "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c", "approval_policy=never",
        "-m", model,
    ]
    if reasoning:
        cmd += ["-c", f"model_reasoning_effort={reasoning}"]
    cmd += ["-"]  # read prompt from stdin

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            text=True,
            cwd=str(cwd),
            env={**os.environ},
        )
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = -signal.SIGKILL
        stdout = (e.stdout or "") if hasattr(e, "stdout") else ""
        stderr = (e.stderr or "") if hasattr(e, "stderr") else ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
    except FileNotFoundError as e:
        return {
            "exit_code": -1,
            "elapsed_s": time.time() - t0,
            "stdout": "",
            "stderr": str(e),
            "timed_out": False,
            "missing_binary": True,
        }
    return {
        "exit_code": rc,
        "elapsed_s": time.time() - t0,
        "stdout": stdout,
        "stderr": stderr[-4000:],
        "timed_out": timed_out,
        "missing_binary": False,
    }


def _looks_like_auth_failure(stderr: str, stdout: str) -> bool:
    blob = (stderr or "") + "\n" + (stdout or "")
    blob_l = blob.lower()
    needles = (
        "not authenticated",
        "please run `codex login`",
        "missing api key",
        "unauthorized",
        "401",
        "auth.json",
        "no credentials",
    )
    return any(n in blob_l for n in needles)


def _score_task(task: dict, predicted: Any) -> dict:
    try:
        return verify(task, predicted)
    except Exception as e:
        return {"score": 0.0, "tier": 0, "signal": "verify_error", "details": {"error": str(e)[:200]}}


def _process_one(
    task: dict,
    *,
    codex_bin: str,
    model: str,
    reasoning: str | None,
    timeout_s: float,
    cwd: Path,
) -> dict:
    prompt = _build_prompt(task)
    run = _run_codex(
        prompt,
        codex_bin=codex_bin,
        model=model,
        reasoning=reasoning,
        timeout_s=timeout_s,
        cwd=cwd,
    )
    if run["missing_binary"]:
        return {
            "id": task.get("id"),
            "category": task.get("category"),
            "score": 0.0,
            "error": "codex_unavailable",
            "elapsed_s": 0.0,
            "exit_code": -1,
            "parser_status": "n/a",
            "predicted_tool": None,
            "verifier": {"score": 0.0, "signal": "skipped"},
        }
    final_text, msg_status = _extract_final_message(run["stdout"])
    predicted, parser_status = _parse_predicted(final_text)
    verifier = _score_task(task, predicted)
    pred_tool = None
    if isinstance(predicted, dict):
        pred_tool = predicted.get("tool_name")
    return {
        "id": task.get("id"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "score": float(verifier.get("score", 0.0)),
        "verifier": verifier,
        "elapsed_s": run["elapsed_s"],
        "exit_code": run["exit_code"],
        "timed_out": run["timed_out"],
        "stdout_chars": len(run["stdout"]),
        "stderr_tail": run["stderr"][-400:],
        "final_message_chars": len(final_text or ""),
        "parser_status": parser_status,
        "message_status": msg_status,
        "predicted_tool": pred_tool,
        "predicted_preview": (json.dumps(predicted) if not isinstance(predicted, str) else predicted)[:240]
        if predicted is not None
        else None,
    }


def _summarise(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "pass_rate": 0.0}
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_diff: dict[str, list[float]] = defaultdict(list)
    parser_counts: Counter = Counter()
    error_counts: Counter = Counter()
    timeouts = 0
    elapsed_total = 0.0
    for r in results:
        by_cat[str(r.get("category"))].append(float(r.get("score") or 0.0))
        by_diff[str(r.get("difficulty"))].append(float(r.get("score") or 0.0))
        parser_counts[r.get("parser_status") or "n/a"] += 1
        if r.get("error"):
            error_counts[r["error"]] += 1
        if r.get("timed_out"):
            timeouts += 1
        elapsed_total += float(r.get("elapsed_s") or 0.0)
    pass_rate = sum(float(r.get("score") or 0.0) for r in results) / n
    by_cat_summary = {
        k: {"n": len(v), "pass_rate": (sum(v) / len(v)) if v else 0.0} for k, v in by_cat.items()
    }
    by_diff_summary = {
        k: {"n": len(v), "pass_rate": (sum(v) / len(v)) if v else 0.0} for k, v in by_diff.items()
    }
    return {
        "n": n,
        "pass_rate": pass_rate,
        "by_category": by_cat_summary,
        "by_difficulty": by_diff_summary,
        "parser_counts": dict(parser_counts),
        "error_counts": dict(error_counts),
        "timeouts": timeouts,
        "mean_elapsed_s": elapsed_total / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning", default="high",
                    help="model_reasoning_effort: low|medium|high|xhigh (passed via -c)")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total number of tasks; 0 means no cap")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-task subprocess timeout in seconds")
    ap.add_argument("--output", required=True)
    ap.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex"))
    ap.add_argument("--cwd", default=str(_ROOT),
                    help="cwd for the codex subprocess (we use a repo dir + skip-git-repo-check)")
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"ERROR: tasks file not found: {tasks_path}", file=sys.stderr)
        return 2

    codex_bin = args.codex_bin
    resolved = shutil.which(codex_bin) or codex_bin
    if not Path(resolved).exists() and shutil.which(codex_bin) is None:
        print(f"WARN: codex binary not on PATH ({codex_bin}); will record codex_unavailable.", file=sys.stderr)

    tasks: list[dict] = []
    with tasks_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except Exception as e:
                print(f"  skip malformed task line: {e}", file=sys.stderr)
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks after filtering", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"codex_bin = {resolved}")
    print(f"model     = {args.model}   reasoning = {args.reasoning}")
    print(f"tasks     = {len(tasks)}   workers = {args.max_workers}   timeout/task = {args.timeout}s")
    print(f"output    = {output_path}")

    # Probe with the first task to detect auth failure / binary missing.
    abort_remaining = False
    abort_reason: str | None = None
    cwd = Path(args.cwd).resolve()
    if not cwd.exists():
        cwd = _ROOT

    t_start = time.time()
    results: list[dict | None] = [None] * len(tasks)

    # Run task 0 alone first.
    print("probing task 0 ...")
    r0 = _process_one(
        tasks[0],
        codex_bin=codex_bin,
        model=args.model,
        reasoning=args.reasoning,
        timeout_s=args.timeout,
        cwd=cwd,
    )
    results[0] = r0
    print(f"  task 0 -> score={r0.get('score'):.2f} parser={r0.get('parser_status')} "
          f"exit={r0.get('exit_code')} elapsed={r0.get('elapsed_s', 0.0):.1f}s")
    if r0.get("error") == "codex_unavailable":
        abort_remaining = True
        abort_reason = "codex_unavailable"
    elif r0.get("exit_code") not in (0, None) and _looks_like_auth_failure(r0.get("stderr_tail") or "", ""):
        abort_remaining = True
        abort_reason = "codex_auth_failure"

    if abort_remaining:
        print(f"  aborting remaining {len(tasks)-1} tasks: {abort_reason}", file=sys.stderr)
        for i in range(1, len(tasks)):
            results[i] = {
                "id": tasks[i].get("id"),
                "category": tasks[i].get("category"),
                "difficulty": tasks[i].get("difficulty"),
                "score": 0.0,
                "error": abort_reason,
                "elapsed_s": 0.0,
                "exit_code": -1,
                "parser_status": "skipped",
                "verifier": {"score": 0.0, "signal": "skipped"},
            }
    else:
        idxs = list(range(1, len(tasks)))
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            fut_to_idx = {
                ex.submit(
                    _process_one,
                    tasks[i],
                    codex_bin=codex_bin,
                    model=args.model,
                    reasoning=args.reasoning,
                    timeout_s=args.timeout,
                    cwd=cwd,
                ): i
                for i in idxs
            }
            done = 0
            total = len(fut_to_idx)
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = {
                        "id": tasks[i].get("id"),
                        "category": tasks[i].get("category"),
                        "difficulty": tasks[i].get("difficulty"),
                        "score": 0.0,
                        "error": f"future_exception:{e}",
                        "elapsed_s": 0.0,
                        "exit_code": -1,
                        "parser_status": "exception",
                        "verifier": {"score": 0.0, "signal": "exception"},
                    }
                done += 1
                if done % max(1, total // 10) == 0 or done == total:
                    print(f"  progress: {done}/{total}  elapsed={time.time()-t_start:.1f}s")

    final = [r for r in results if r is not None]
    summary = _summarise(final)
    out = {
        "summary": summary,
        "config": {
            "model": args.model,
            "reasoning": args.reasoning,
            "max_workers": args.max_workers,
            "timeout_s": args.timeout,
            "limit": args.limit,
            "tasks_path": str(tasks_path),
            "codex_bin": resolved,
            "elapsed_total_s": time.time() - t_start,
            "tasks_n": len(tasks),
        },
        "results": final,
    }
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {output_path}")
    print(f"pass_rate = {summary['pass_rate']:.3f}  n = {summary['n']}  "
          f"timeouts = {summary['timeouts']}  parser = {summary['parser_counts']}")
    print(f"errors    = {summary['error_counts']}")
    print(f"by_category:")
    for k in sorted(summary["by_category"]):
        v = summary["by_category"][k]
        print(f"  {k:<22}n={v['n']:<3}  pass={v['pass_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
