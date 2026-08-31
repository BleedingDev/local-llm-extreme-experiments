#!/usr/bin/env python3
"""Benchmark OpenAI-compatible MLX server aggregate concurrent throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


PROMPT = (
    "Generate a long, dense technical explanation of local LLM inference batching. "
    "Keep writing until the token limit is reached."
)


def post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    elapsed = time.perf_counter() - start
    return json.loads(body), elapsed


def run_one(url: str, model: str, max_tokens: int, idx: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{PROMPT}\nRequest id: {idx}",
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    try:
        response, elapsed = post_json(url, payload, timeout)
        usage = response.get("usage") or {}
        choice = (response.get("choices") or [{}])[0]
        return {
            "ok": True,
            "elapsed_s": elapsed,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "finish_reason": choice.get("finish_reason"),
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": None,
            "error": repr(exc),
        }


def benchmark_level(url: str, model: str, concurrency: int, max_tokens: int, timeout: float) -> dict:
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_one, url, model, max_tokens, idx, timeout)
            for idx in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - start
    ok = [r for r in results if r["ok"]]
    completion_tokens = sum(r["completion_tokens"] for r in ok)
    prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    per_request_elapsed = [r["elapsed_s"] for r in ok]
    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": concurrency - len(ok),
        "wall_s": wall,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "aggregate_completion_tps": completion_tokens / wall if wall else 0,
        "aggregate_total_tps": (completion_tokens + prompt_tokens) / wall if wall else 0,
        "mean_request_s": statistics.mean(per_request_elapsed) if per_request_elapsed else 0,
        "p50_request_s": statistics.median(per_request_elapsed) if per_request_elapsed else 0,
        "max_request_s": max(per_request_elapsed) if per_request_elapsed else 0,
        "finish_reasons": {
            reason: sum(1 for r in ok if r["finish_reason"] == reason)
            for reason in sorted({r["finish_reason"] for r in ok}, key=str)
        },
        "errors": [r["error"] for r in results if not r["ok"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--model", default="majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--levels", default="1,2,4,8,10")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/chat/completions"
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    all_results = []
    for level in levels:
        result = benchmark_level(url, args.model, level, args.max_tokens, args.timeout)
        all_results.append(result)
        print(
            f"concurrency={level:2d} ok={result['ok']:2d}/{level:<2d} "
            f"wall={result['wall_s']:.2f}s completion={result['completion_tokens']} "
            f"agg_decode={result['aggregate_completion_tps']:.2f} tok/s "
            f"mean_req={result['mean_request_s']:.2f}s"
        )
    payload = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "levels": all_results,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if all(r["failed"] == 0 for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
