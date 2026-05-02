"""Run Phase-3 benchmark tasks against a single Claude model.

Each task carries its own verifier spec. Per task: build prompt, call
`agent_opt.llm.chat`, score via `bench.verifiers.verify` (with an in-file
fallback). Per-task failures are captured as score=0 + error, never fatal.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt import llm as _llm  # noqa: E402

_DEFAULT_SYSTEM = (
    "You are a coding agent. Given a user request, recent actions, and the available "
    "tool list, choose the next action. Output STRICTLY one compact JSON line: "
    '{"tool_name": "<one of available_tools or empty>", "brief_reason": "<<=20 words>"}'
)

_USER_TEMPLATE = (
    "Given the context below, decide on the next action.\n\n"
    "User request:\n{user_request}\n\n"
    "Recent assistant actions (most recent last):\n{recent_actions}\n\n"
    "Available tools: {available_tools}\n\n"
    "Output STRICTLY as compact JSON on a single line:\n"
    '{{"tool_name": "<one of available_tools or empty>", "brief_reason": "<<=20 words>"}}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_PRED_RE = re.compile(r"\$\.(\w+)\s*==\s*\"([^\"]+)\"")


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _compact_actions(actions: list[Any] | None) -> str:
    if not actions:
        return "(none)"
    out: list[str] = []
    for a in actions[-5:]:
        if isinstance(a, dict):
            name = a.get("name") or a.get("tool_name") or a.get("kind") or "?"
            inp = a.get("input")
            inp_s = inp[:120] if isinstance(inp, str) else ("" if inp is None else json.dumps(inp)[:120])
            out.append(f"- {name}: {inp_s}")
        else:
            out.append(f"- {str(a)[:160]}")
    return "\n".join(out)


def _build_user_prompt(task: dict) -> str:
    prompt = task.get("prompt") or {}
    ctx = prompt.get("context") or {}
    tools = ctx.get("available_tools") or []
    tools_s = ", ".join(str(t) for t in tools[:25]) if isinstance(tools, list) else str(tools)
    return _USER_TEMPLATE.format(user_request=str(prompt.get("user_request") or "")[:1500],
                                  recent_actions=_compact_actions(ctx.get("recent_actions")),
                                  available_tools=tools_s or "(unknown)")


def _try_import_verify():
    try:
        from bench.verifiers import verify  # type: ignore[attr-defined]
    except Exception:
        return None
    return verify


def _fallback_verify(task: dict, predicted: Any) -> dict:
    """In-file dispatcher: handles `regex` and `structural_json` (`$.field == "v"`)."""
    kind = task.get("verifier_kind") or ""
    pat = (task.get("verifier_spec") or {}).get("pattern_or_command") or ""
    if kind == "regex":
        text = predicted if isinstance(predicted, str) else json.dumps(predicted or {}, sort_keys=True)
        try:
            ok = re.search(pat, text) is not None
        except re.error as e:
            return {"score": 0.0, "signal": "regex_error", "details": {"error": str(e)}}
        return {"score": 1.0 if ok else 0.0, "signal": "regex_match" if ok else "regex_miss"}
    if kind == "structural_json":
        obj = predicted if isinstance(predicted, dict) else _parse_json(str(predicted or ""))
        if obj is None:
            return {"score": 0.0, "signal": "json_parse_fail"}
        m = _PRED_RE.match(pat.strip())
        if not m:
            return {"score": 0.0, "signal": "spec_unparsed", "details": {"pat": pat[:120]}}
        field, want = m.group(1), m.group(2)
        got = obj.get(field) if isinstance(obj, dict) else None
        ok = got == want
        return {"score": 1.0 if ok else 0.0,
                "signal": "tool_match" if ok else "tool_miss",
                "details": {"field": field, "expected": want, "got": got}}
    return {"score": 0.0, "signal": "unknown_kind", "details": {"kind": kind}}


def _run_one(task: dict, model: str, system_prompt: str, max_tokens: int, verify_fn) -> dict:
    t0 = time.time()
    user_prompt = _build_user_prompt(task)
    prompt_tokens_est = (len(system_prompt) + len(user_prompt)) // 4
    raw, parsed, err, verdict = "", None, None, {"score": 0.0, "signal": "no_verdict"}
    try:
        raw = _llm.chat(messages=[{"role": "user", "content": user_prompt}], model=model,
                        max_tokens=max_tokens, temperature=0.0, system=system_prompt)
    except Exception as e:
        err = f"llm_error: {e}"
    if raw and err is None:
        parsed = _parse_json(raw)
    if err is None:
        try:
            verdict = verify_fn(task, parsed if parsed is not None else raw)
        except Exception as e:
            err, verdict = f"verify_error: {e}", {"score": 0.0, "signal": "verify_error"}
    score = float(verdict.get("score", 0.0)) if isinstance(verdict, dict) else 0.0
    return {
        "id": task.get("id"), "category": task.get("category"), "difficulty": task.get("difficulty"),
        "score": 0.0 if err else score,
        "verifier_signal": verdict.get("signal") if isinstance(verdict, dict) else None,
        "verifier_details": verdict.get("details") if isinstance(verdict, dict) else None,
        "raw_output": (raw or "")[:1200], "parsed_output": parsed,
        "prompt_tokens_est": prompt_tokens_est, "output_tokens_est": max(1, len(raw or "") // 4),
        "latency_ms": int((time.time() - t0) * 1000), "error": err,
    }


def _aggregate(rows: list[dict], elapsed: float) -> dict:
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_diff: dict[str, list[float]] = defaultdict(list)
    sig_by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_in = total_out = n_err = 0
    for r in rows:
        s = float(r.get("score") or 0.0)
        by_cat[str(r.get("category"))].append(s)
        by_diff[str(r.get("difficulty"))].append(s)
        sig_by_cat[str(r.get("category"))][str(r.get("verifier_signal") or "")] += 1
        total_in += int(r.get("prompt_tokens_est") or 0)
        total_out += int(r.get("output_tokens_est") or 0)
        if r.get("error"):
            n_err += 1
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    return {
        "overall": {
            "total_tasks": len(rows), "mean_score": round(mean([float(r.get("score") or 0.0) for r in rows]), 4),
            "wallclock_s": round(elapsed, 2), "lm_calls": len(rows), "lm_errors": n_err,
            "prompt_tokens_est": total_in, "output_tokens_est": total_out,
            "total_tokens_est": total_in + total_out,
        },
        "per_category": {k: {"count": len(v), "mean_score": round(mean(v), 4),
                             "signal_breakdown": dict(sig_by_cat[k])} for k, v in by_cat.items()},
        "per_difficulty": {k: {"count": len(v), "mean_score": round(mean(v), 4)} for k, v in by_diff.items()},
    }


def _print_summary(agg: dict, model: str, system_chars: int) -> None:
    o = agg["overall"]
    print(f"\nmodel={model}  system_chars={system_chars}")
    print(f"tasks={o['total_tasks']}  mean_score={o['mean_score']:.3f}  errors={o['lm_errors']}  wallclock={o['wallclock_s']}s")
    print(f"tokens (est): in={o['prompt_tokens_est']}  out={o['output_tokens_est']}  total={o['total_tokens_est']}")
    for label, key in (("per-category", "per_category"), ("per-difficulty", "per_difficulty")):
        print(f"\n{label}:\n  {'name':<22}{'n':>5}{'mean':>10}")
        for k in sorted(agg[key]):
            v = agg[key][k]
            print(f"  {k:<22}{v['count']:>5}{v['mean_score']:>10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(_ROOT / "data" / "benchmark_tasks.jsonl"))
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = run all")
    ap.add_argument("--output", default=None)
    ap.add_argument("--system-prompt-file", default=None)
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"no tasks: {tasks_path} does not exist (Agent #2 not finished?)", file=sys.stderr)
        return 2

    tasks: list[dict] = []
    for line in tasks_path.open():
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except Exception as e:
            print(f"WARN: skipping malformed task line: {e}", file=sys.stderr)
    if not tasks:
        print(f"no tasks loaded from {tasks_path}", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    if args.system_prompt_file:
        sp_path = Path(args.system_prompt_file)
        if not sp_path.is_absolute():
            sp_path = _ROOT.parent / args.system_prompt_file
        try:
            system_prompt = sp_path.read_text()
        except Exception as e:
            print(f"could not read system prompt {sp_path}: {e}", file=sys.stderr)
            return 2
    else:
        system_prompt = _DEFAULT_SYSTEM

    verify_fn = _try_import_verify() or _fallback_verify
    used_fallback = verify_fn is _fallback_verify
    print(f"loaded {len(tasks)} tasks  model={args.model}  workers={args.max_workers}")
    print(f"system_prompt: chars={len(system_prompt)}  src={args.system_prompt_file or '(default)'}")
    print(f"verifier: {'fallback (in-file)' if used_fallback else 'bench.verifiers.verify'}")

    t0 = time.time()
    rows: list[dict] = [None] * len(tasks)  # type: ignore[assignment]
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fut_to_idx = {ex.submit(_run_one, t, args.model, system_prompt, args.max_tokens, verify_fn): i
                      for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                rows[i] = fut.result()
            except Exception as e:
                rows[i] = {"id": tasks[i].get("id"), "category": tasks[i].get("category"),
                           "difficulty": tasks[i].get("difficulty"), "score": 0.0,
                           "error": f"runner_error: {e}", "verifier_signal": None,
                           "raw_output": "", "parsed_output": None,
                           "prompt_tokens_est": 0, "output_tokens_est": 0, "latency_ms": 0}
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  progress: {done}/{len(tasks)} ({100.0*done/len(tasks):.0f}%) elapsed={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    agg = _aggregate(rows, elapsed)

    out = {
        "model": args.model, "tasks_path": str(tasks_path),
        "system_prompt_file": args.system_prompt_file,
        "system_prompt_chars": len(system_prompt), "limit": args.limit,
        "verifier": "bench.verifiers.verify" if not used_fallback else "fallback",
        "summary": agg, "per_task": rows,
    }
    if args.output:
        out_path = Path(args.output)
    else:
        ts = int(time.time())
        out_path = _HERE / "results" / f"anthropic_{args.model}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")
    _print_summary(agg, args.model, len(system_prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
