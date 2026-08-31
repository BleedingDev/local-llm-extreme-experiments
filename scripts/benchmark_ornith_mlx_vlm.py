#!/usr/bin/env python3
"""Benchmark Ornith through mlx-vlm without running agent tasks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template


DEFAULT_MODEL = "mlx-community/Ornith-1.0-35B-4bit"
DEFAULT_PROMPT = (
    "Write a dense technical explanation of Apple Silicon local LLM inference. "
    "Discuss Metal kernels, quantized weights, KV cache behavior, batching, and "
    "why keeping expert layers on GPU matters. Continue until the token limit."
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected non-negative integer, got {value!r}")
    return parsed


def reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.item()
    except Exception:
        pass
    try:
        return value.tolist()
    except Exception:
        return str(value)


def build_prompt(processor: Any, config: Any, args: argparse.Namespace) -> str:
    if args.mode == "direct":
        return args.prompt

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    return apply_chat_template(
        processor,
        config,
        messages,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_generation(model: Any, processor: Any, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    reset_peak_memory()
    started = time.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        image=None,
        audio=None,
        verbose=False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        kv_quant_scheme=args.kv_quant_scheme,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
        skip_special_tokens=args.skip_special_tokens,
        enable_thinking=args.enable_thinking,
        thinking_budget=args.thinking_budget,
    )
    wall_s = time.perf_counter() - started
    payload = asdict(result)
    payload["token"] = scalar(payload.get("token"))
    payload["logprobs"] = None
    payload["wall_s"] = wall_s
    payload["text_preview"] = result.text[:300]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("chat", "direct"), default="chat")
    parser.add_argument("--system", default="")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=positive_int, default=256)
    parser.add_argument("--warmup-tokens", type=non_negative_int, default=16)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=positive_int, default=2048)
    parser.add_argument("--max-kv-size", type=positive_int, default=None)
    parser.add_argument("--kv-bits", type=float, default=None)
    parser.add_argument("--kv-quant-scheme", choices=("uniform", "turboquant"), default="uniform")
    parser.add_argument("--kv-group-size", type=positive_int, default=64)
    parser.add_argument("--quantized-kv-start", type=non_negative_int, default=5000)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=non_negative_int, default=None)
    parser.add_argument("--skip-special-tokens", action="store_true", default=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    load_started = time.perf_counter()
    model, processor = load(args.model)
    load_s = time.perf_counter() - load_started
    prompt = build_prompt(processor, model.config, args)

    warmup: dict[str, Any] | None = None
    if args.warmup_tokens:
        warmup_args = argparse.Namespace(**{**vars(args), "max_tokens": args.warmup_tokens})
        warmup = run_generation(model, processor, prompt, warmup_args)
        print(
            "warmup "
            f"gen_tokens={warmup['generation_tokens']} "
            f"gen_tps={warmup['generation_tps']:.2f} "
            f"peak_gb={warmup['peak_memory']:.2f}"
        )

    runs = []
    for idx in range(args.repeats):
        result = run_generation(model, processor, prompt, args)
        runs.append(result)
        print(
            f"run={idx + 1} "
            f"prompt_tokens={result['prompt_tokens']} "
            f"prompt_tps={result['prompt_tps']:.2f} "
            f"gen_tokens={result['generation_tokens']} "
            f"gen_tps={result['generation_tps']:.2f} "
            f"wall_s={result['wall_s']:.2f} "
            f"peak_gb={result['peak_memory']:.2f}"
        )

    summary = {
        "model": args.model,
        "mode": args.mode,
        "load_s": load_s,
        "params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "prefill_step_size": args.prefill_step_size,
            "max_kv_size": args.max_kv_size,
            "kv_bits": args.kv_bits,
            "kv_quant_scheme": args.kv_quant_scheme,
            "kv_group_size": args.kv_group_size,
            "quantized_kv_start": args.quantized_kv_start,
            "enable_thinking": args.enable_thinking,
            "thinking_budget": args.thinking_budget,
        },
        "prompt_chars": len(prompt),
        "warmup": warmup,
        "runs": runs,
        "generation_tps": summarize([float(r["generation_tps"]) for r in runs]),
        "prompt_tps": summarize([float(r["prompt_tps"]) for r in runs]),
        "peak_gb": summarize([float(r["peak_memory"]) for r in runs]),
        "min_generation_tokens": min(int(r["generation_tokens"]) for r in runs),
    }

    print(
        "summary "
        f"gen_tps_mean={summary['generation_tps']['mean']:.2f} "
        f"gen_tps_min={summary['generation_tps']['min']:.2f} "
        f"prompt_tps_mean={summary['prompt_tps']['mean']:.2f} "
        f"peak_gb_max={summary['peak_gb']['max']:.2f} "
        f"load_s={load_s:.2f}"
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
