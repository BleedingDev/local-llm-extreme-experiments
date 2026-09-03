#!/usr/bin/env python3
"""Measure useful aggregate throughput from concurrent MTPLX requests."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

from benchmark_mtplx_qwen38_context import (
    DEFAULT_MODEL_PATH,
    SYSTEM_PROMPT,
    build_filler_tokens,
    json_request,
    parse_int_list,
    thermal_snapshot,
    wait_for_start_temperature,
)


ROOT = Path(__file__).resolve().parents[1]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_worker_prompt(
    tokenizer: Any,
    filler_tokens: list[int],
    target_context: int,
    label: str,
    width: int,
    worker: int,
) -> tuple[str, str]:
    marker = f"{label.upper()}-B{width}-W{worker}"
    instruction = (
        f"You are overnight benchmark worker {marker}.\n"
        f"First line exactly: JOB={marker}\n"
        "Then write a detailed numbered reliability checklist for unattended local-LLM "
        "automation. Continue for at least 1,200 words, do not summarize, and do not stop early.\n\n"
    )
    fixed = len(tokenizer.encode(instruction, add_special_tokens=False))
    filler_count = max(64, target_context - fixed)
    filler = tokenizer.decode(filler_tokens[:filler_count], skip_special_tokens=True)
    return instruction + filler, marker


def run_request(
    *,
    start_event: threading.Event,
    endpoint: str,
    model: str,
    content: str,
    marker: str,
    max_tokens: int,
    generation_mode: str,
    depth: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "stream": False,
        "generation_mode": generation_mode,
        "reasoning": "off",
    }
    if generation_mode == "mtp":
        payload["depth"] = depth

    start_event.wait()
    started = time.monotonic()
    try:
        response = json_request(f"{endpoint}/v1/chat/completions", payload, timeout)
        finished = time.monotonic()
        choice = response.get("choices", [{}])[0]
        output = (choice.get("message") or {}).get("content") or ""
        timings = response.get("timings") or {}
        stats = response.get("mtplx_stats") or {}
        return {
            "status": "ok",
            "marker": marker,
            "quality_pass": f"JOB={marker}" in output,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": timings.get("prompt_n", stats.get("prompt_tokens")),
            "completion_tokens": timings.get(
                "predicted_n", stats.get("completion_tokens")
            ),
            "server_decode_tps": timings.get(
                "predicted_per_second", stats.get("decode_tps")
            ),
            "server_ttft_s": stats.get("ttft_s"),
            "server_queue_wait_s": stats.get("queue_wait_s"),
            "client_started_s": started,
            "client_finished_s": finished,
            "client_latency_s": finished - started,
            "peak_memory_bytes": stats.get("peak_memory_bytes"),
            "response": response,
        }
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = time.monotonic()
    return {
        "status": "failed",
        "marker": marker,
        "quality_pass": False,
        "client_started_s": started,
        "client_finished_s": finished,
        "client_latency_s": finished - started,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11500")
    parser.add_argument("--model", default="mtplx-qwen38-27b-optimized-speed")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scheduler-label", required=True)
    parser.add_argument("--generation-mode", choices=("mtp", "ar"), required=True)
    parser.add_argument("--concurrency", type=parse_int_list, default=(1, 2, 4))
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-start-temp", type=float, default=65)
    parser.add_argument("--cooldown-timeout", type=float, default=1200)
    parser.add_argument("--cooldown-seconds", type=float, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.context <= 0 or args.max_tokens <= 0 or args.depth <= 0:
        parser.error("context, max-tokens, and depth must be positive")
    return args


def main() -> int:
    args = parse_args()
    health_before = json_request(f"{args.endpoint}/health", None, 30)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    filler_tokens = build_filler_tokens(tokenizer, args.context + 256)
    output_dir = args.output_dir or (
        ROOT / "artifacts" / "benchmarks" / f"qwen38-mtplx-batch-{utc_stamp()}"
    )
    raw_dir = output_dir / "raw" / args.scheduler_label
    raw_dir.mkdir(parents=True, exist_ok=False)

    cohorts: list[dict[str, Any]] = []
    for width in args.concurrency:
        before = wait_for_start_temperature(args.max_start_temp, args.cooldown_timeout)
        start_event = threading.Event()
        with ThreadPoolExecutor(max_workers=width) as executor:
            futures = []
            for worker in range(1, width + 1):
                content, marker = build_worker_prompt(
                    tokenizer,
                    filler_tokens,
                    args.context,
                    args.scheduler_label,
                    width,
                    worker,
                )
                futures.append(
                    executor.submit(
                        run_request,
                        start_event=start_event,
                        endpoint=args.endpoint,
                        model=args.model,
                        content=content,
                        marker=marker,
                        max_tokens=args.max_tokens,
                        generation_mode=args.generation_mode,
                        depth=args.depth,
                        seed=width * 1000 + worker,
                        timeout=args.timeout,
                    )
                )
            released = time.monotonic()
            start_event.set()
            results = [future.result() for future in as_completed(futures)]
        completed = time.monotonic()
        after = thermal_snapshot()

        results.sort(key=lambda item: item["marker"])
        for result in results:
            response = result.pop("response", None)
            if response is not None:
                (raw_dir / f"b{width}-{result['marker']}.response.json").write_text(
                    json.dumps(response, indent=2)
                )
        good = [result for result in results if result["status"] == "ok"]
        total_tokens = sum(int(result.get("completion_tokens") or 0) for result in good)
        wall_s = completed - released
        latencies = [float(result["client_latency_s"]) for result in good]
        peak_values = [
            int(result["peak_memory_bytes"])
            for result in good
            if result.get("peak_memory_bytes") is not None
        ]
        cohort = {
            "scheduler_label": args.scheduler_label,
            "generation_mode": args.generation_mode,
            "concurrency": width,
            "wall_s": wall_s,
            "completed_requests": len(good),
            "quality_passes": sum(bool(result["quality_pass"]) for result in good),
            "completion_tokens": total_tokens,
            "aggregate_completion_tps": total_tokens / wall_s if wall_s else None,
            "mean_client_latency_s": statistics.mean(latencies) if latencies else None,
            "max_client_latency_s": max(latencies) if latencies else None,
            "mean_server_decode_tps": statistics.mean(
                float(result["server_decode_tps"])
                for result in good
                if result.get("server_decode_tps") is not None
            )
            if any(result.get("server_decode_tps") is not None for result in good)
            else None,
            "peak_memory_gb": max(peak_values) / (1024**3) if peak_values else None,
            "thermal_before": before,
            "thermal_after": after,
            "requests": results,
        }
        cohorts.append(cohort)
        print(
            f"[{args.scheduler_label} B{width}] aggregate={cohort['aggregate_completion_tps']:.2f} "
            f"tok/s wall={wall_s:.2f}s quality={cohort['quality_passes']}/{width} "
            f"hottest={after['hottest_c']}C",
            flush=True,
        )
        (output_dir / f"{args.scheduler_label}.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "arguments": vars(args)
                    | {
                        "model_path": str(args.model_path),
                        "output_dir": str(output_dir),
                    },
                    "server_health_before": health_before,
                    "cohorts": cohorts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)

    return (
        0
        if all(
            cohort["completed_requests"] == cohort["concurrency"]
            and cohort["quality_passes"] == cohort["concurrency"]
            for cohort in cohorts
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
