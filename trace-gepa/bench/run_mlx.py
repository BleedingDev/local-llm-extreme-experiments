"""MLX local-model benchmark harness. Single-threaded with per-task timeout."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.verifiers.tier1_regex import (  # noqa: E402
    verify_exact_match,
    verify_regex,
    verify_structural_json,
    verify_tool_family_match,
    verify_tool_name_match,
)


class _TaskTimeout(Exception):
    pass


@contextmanager
def _alarm(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle(_s, _f):
        raise _TaskTimeout(f"timeout after {seconds}s")

    prev = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


_VERIFIERS = {
    "regex": verify_regex,
    "exact_match": verify_exact_match,
    "structural_json": verify_structural_json,
    "tool_name_match": verify_tool_name_match,
    "tool_family_match": verify_tool_family_match,
}


def verify(task: dict, predicted: Any) -> dict:
    fn = _VERIFIERS.get(task.get("verifier_kind") or "structural_json", verify_structural_json)
    try:
        return fn(task, predicted)
    except Exception as e:
        return {"score": 0.0, "tier": 1, "signal": "verifier_error", "details": {"error": str(e)}}


_SYS_MSG = (
    "You are an autonomous coding agent. Decide the next single action.\n"
    'Respond with ONLY a JSON object: {"tool_name": <str>, "input": <obj>, "reason": <str>}.\n'
    "Pick from the listed available tools. No prose outside the JSON."
)


def _build_prompt(tokenizer, task: dict) -> str:
    prompt = task.get("prompt") or {}
    ctx = prompt.get("context") or {}
    user_request = (prompt.get("user_request") or task.get("user_request") or "").strip()
    user_lines = [user_request, ""]
    if ctx.get("available_tools"):
        user_lines.append("Available tools: " + ", ".join(ctx["available_tools"]))
    if ctx.get("available_skills"):
        user_lines.append("Available skills: " + ", ".join(ctx["available_skills"]))
    if ctx.get("recent_actions"):
        user_lines.append("Recent actions: " + " | ".join(map(str, ctx["recent_actions"][-4:])))
    msgs = [{"role": "system", "content": _SYS_MSG},
            {"role": "user", "content": "\n".join(user_lines)}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"<|system|>\n{_SYS_MSG}\n<|user|>\n{msgs[1]['content']}\n<|assistant|>\n"


def _load_tasks(path: Path, limit: int | None) -> list[dict]:
    out: list[dict] = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
        if limit and len(out) >= limit:
            break
    return out


def _peak_memory_bytes() -> int | None:
    try:
        import mlx.core as mx  # type: ignore

        if hasattr(mx, "get_peak_memory"):
            return int(mx.get_peak_memory())
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return int(mx.metal.get_peak_memory())
    except Exception:
        return None
    return None


def _parse_predicted(text: str) -> Any:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass
    return text


def _run(args) -> int:
    tasks_path = Path(args.tasks).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm  # type: ignore
    except Exception as e:
        output_path.write_text(json.dumps({
            "status": "mlx_not_installed", "error": str(e), "model": args.model,
            "tasks_path": str(tasks_path), "harness": "run_mlx",
            "install_hint": "pip install mlx-lm  (Apple Silicon required)",
        }, indent=2))
        print(f"[run_mlx] mlx_lm not importable: {e}")
        return 0

    tasks = _load_tasks(tasks_path, args.limit)
    if not tasks:
        output_path.write_text(json.dumps({"status": "no_tasks", "model": args.model}, indent=2))
        return 0

    print(f"[run_mlx] loading {args.model} ...", flush=True)
    t0 = time.time()
    try:
        model, tokenizer = mlx_lm.load(args.model)
    except Exception as e:
        output_path.write_text(json.dumps({
            "status": "model_load_failed", "model": args.model,
            "error": str(e), "load_seconds": time.time() - t0,
        }, indent=2))
        print(f"[run_mlx] load failed: {e}")
        return 1
    load_seconds = time.time() - t0
    print(f"[run_mlx] loaded in {load_seconds:.1f}s", flush=True)

    per_task: list[dict] = []
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    signals: Counter = Counter()
    total_tokens = 0
    total_seconds = 0.0
    skipped_oom = 0
    skipped_timeout = 0

    for i, task in enumerate(tasks):
        prompt = _build_prompt(tokenizer, task)
        rec_start = time.time()
        gen_text = ""
        err: str | None = None
        status = "ok"
        try:
            with _alarm(args.task_timeout):
                gen_text = mlx_lm.generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=args.max_tokens, verbose=False,
                )
        except _TaskTimeout as e:
            status, err = "timeout", str(e); skipped_timeout += 1
        except MemoryError as e:
            status, err = "oom", str(e); skipped_oom += 1
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("memory", "alloc", "metal")):
                status = "oom"; skipped_oom += 1
            else:
                status = "error"
            err = str(e)

        latency = time.time() - rec_start
        try:
            gen_tokens = len(tokenizer.encode(gen_text)) if gen_text else 0
        except Exception:
            gen_tokens = 0
        if status == "ok" and latency > 0:
            total_tokens += gen_tokens
            total_seconds += latency

        verdict = (verify(task, _parse_predicted(gen_text))
                   if status == "ok"
                   else {"score": 0.0, "tier": 1, "signal": status, "details": {"error": err}})
        score = float(verdict.get("score") or 0.0)
        signals[verdict.get("signal") or status] += 1
        cat = str(task.get("category"))
        kind = str(task.get("verifier_kind"))
        by_cat[cat].append(score)
        by_kind[kind].append(score)
        per_task.append({
            "id": task.get("id"), "category": cat, "verifier_kind": kind,
            "score": score, "signal": verdict.get("signal"), "status": status,
            "latency_seconds": round(latency, 3), "gen_tokens": gen_tokens,
            "tokens_per_sec": round(gen_tokens / latency, 2) if latency > 0 else 0.0,
            "error": err, "predicted_preview": (gen_text or "")[:200],
        })
        print(
            f"[{i + 1}/{len(tasks)}] {task.get('id')} score={score:.0f} "
            f"toks={gen_tokens} {latency:.1f}s status={status}",
            flush=True,
        )

    n = len(per_task)
    n_pass = sum(1 for r in per_task if r["score"] >= 1.0)
    mean_tps = (total_tokens / total_seconds) if total_seconds > 0 else 0.0
    peak_mem = _peak_memory_bytes()
    result = {
        "status": "ok", "harness": "run_mlx", "model": args.model,
        "tasks_path": str(tasks_path),
        "n_tasks": n, "n_pass": n_pass, "pass_rate": (n_pass / n) if n else 0.0,
        "load_seconds": round(load_seconds, 2),
        "total_gen_tokens": total_tokens, "total_gen_seconds": round(total_seconds, 2),
        "mean_tokens_per_sec": round(mean_tps, 2),
        "peak_memory_bytes": peak_mem,
        "peak_memory_gb": round(peak_mem / (1024 ** 3), 3) if peak_mem else None,
        "skipped_timeout": skipped_timeout, "skipped_oom": skipped_oom,
        "by_category": {k: {"n": len(v), "pass_rate": sum(v) / len(v)} for k, v in by_cat.items()},
        "by_verifier_kind": {k: {"n": len(v), "pass_rate": sum(v) / len(v)} for k, v in by_kind.items()},
        "signal_counts": dict(signals), "per_task": per_task,
        "config": {"max_tokens": args.max_tokens, "temp": args.temp,
                   "top_p": args.top_p, "task_timeout": args.task_timeout},
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"[run_mlx] done: {n_pass}/{n} ({result['pass_rate']:.2%}) "
        f"mean_tps={mean_tps:.1f} peak_mem_gb={result['peak_memory_gb']}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--task-timeout", type=int, default=60)
    return _run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
