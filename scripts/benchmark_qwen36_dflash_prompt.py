#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "Count from 1 to 40, one number per line, and stop."
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "benchmarks" / "qwen36-dflash-prompt"


@dataclass
class RunMetrics:
    mode: str
    repeat_index: int
    block_size: int | None
    wall_time_sec: float
    prompt_tps: float | None
    generation_tps: float | None
    peak_memory_gb: float | None
    prompt_tokens: int | None
    generation_tokens: int | None
    finish_reason: str | None
    acceptance_lengths: list[int]
    acceptance_mean: float | None
    output_text: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-only throughput runner for Qwen 3.6 MLX baseline vs DFlash."
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "dflash"),
        required=True,
        help="Generation path to benchmark",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3.6-35B-A3B-4bit",
        help="Target model id/path",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="z-lab/Qwen3.6-35B-A3B-DFlash",
        help="Draft model id/path for DFlash mode",
    )
    parser.add_argument(
        "--block-size",
        type=_positive_int,
        default=None,
        help="Optional DFlash block size override",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--prompt-file", type=str, default="", help="Read prompt text from file")
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Use the raw prompt string instead of applying the tokenizer chat template",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=64, help="Generation token limit")
    parser.add_argument("--temperature", type=_non_negative_float, default=0.0, help="Sampling temperature")
    parser.add_argument("--repeats", type=_positive_int, default=1, help="Number of measured repeats")
    parser.add_argument("--warmup-passes", type=_non_negative_int, default=1, help="Warmup passes before timing")
    parser.add_argument("--warmup-tokens", type=_positive_int, default=8, help="Warmup token limit")
    parser.add_argument(
        "--prime-baseline-tokens",
        type=_non_negative_int,
        default=0,
        help="Run a one-off baseline generate before DFlash to prime the target model",
    )
    parser.add_argument("--max-kv-size", type=_positive_int, default=2048, help="KV cache cap")
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=512, help="DFlash prefill chunk size")
    parser.add_argument(
        "--draft-sliding-window-size",
        type=_positive_int,
        default=4096,
        help="Optional DFlash draft sliding window size",
    )
    parser.add_argument("--seed", type=int, default=42, help="MLX random seed")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Artifacts root directory")
    parser.add_argument("--run-name", type=str, default="", help="Optional run directory name")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit without loading models")
    parser.add_argument("--debug", action="store_true", help="Print traceback on failure")
    return parser.parse_args()


def _resolve_prompt(args: argparse.Namespace) -> str:
    prompt = args.prompt
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompt = prompt_path.read_text()
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is empty. Provide --prompt or --prompt-file.")
    return prompt


def _prepare_run_dir(artifacts_dir: str, run_name: str) -> Path:
    root = Path(artifacts_dir).expanduser()
    if not root.is_absolute():
        root = ROOT_DIR / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"qwen36-dflash-prompt-{stamp}"
    run_dir = root / name
    if run_dir.exists():
        run_dir = root / f"{name}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_runtime() -> dict[str, Any]:
    try:
        import mlx.core as mx  # noqa: WPS433
        from mlx_lm.sample_utils import make_sampler  # noqa: WPS433
        from mlx_lm import stream_generate as baseline_stream_generate  # noqa: WPS433
        from dflash.model_mlx import load as load_target  # noqa: WPS433
        from dflash.model_mlx import load_draft, stream_generate as dflash_stream_generate  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(
            "Missing MLX/DFlash runtime pieces. Ensure .venv has mlx, mlx-lm, and dflash installed."
        ) from exc

    return {
        "mx": mx,
        "make_sampler": make_sampler,
        "baseline_stream_generate": baseline_stream_generate,
        "load_target": load_target,
        "load_draft": load_draft,
        "dflash_stream_generate": dflash_stream_generate,
    }


def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _maybe_encode_prompt(tokenizer: Any, prompt: str) -> int | None:
    try:
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        try:
            return len(tokenizer.encode(prompt))
        except Exception:
            return None
    except Exception:
        return None


def _seed_mx(runtime: dict[str, Any], seed: int) -> None:
    try:
        runtime["mx"].random.seed(seed)
    except Exception:
        pass


def _clear_cache(runtime: dict[str, Any]) -> None:
    try:
        runtime["mx"].clear_cache()
    except Exception:
        pass


def _run_generation(
    *,
    mode: str,
    block_size: int | None,
    iterator_factory,
    tokenizer: Any,
    prompt: str,
) -> RunMetrics:
    wall_start = time.perf_counter()
    output_parts: list[str] = []
    prompt_tps = None
    generation_tps = None
    peak_memory_gb = None
    prompt_tokens = None
    generation_tokens = None
    finish_reason = None
    acceptance_lengths: list[int] = []
    counted_generation_tokens = 0

    for response in iterator_factory():
        text = getattr(response, "text", "")
        if text:
            output_parts.append(text)
        if getattr(response, "prompt_tps", None) is not None:
            prompt_tps = float(response.prompt_tps)
        if getattr(response, "generation_tps", None) is not None:
            generation_tps = float(response.generation_tps)
        if getattr(response, "peak_memory", None) is not None:
            peak_memory_gb = float(response.peak_memory)
        if getattr(response, "prompt_tokens", None) is not None:
            prompt_tokens = int(response.prompt_tokens)
        if getattr(response, "generation_tokens", None) is not None:
            generation_tokens = int(response.generation_tokens)
        if getattr(response, "finish_reason", None) is not None:
            finish_reason = str(response.finish_reason)
        if getattr(response, "accepted", None) is not None:
            acceptance_lengths.append(int(response.accepted))

        tokens = getattr(response, "tokens", None)
        if tokens is not None:
            counted_generation_tokens += len(tokens)
        elif getattr(response, "token", None) is not None:
            counted_generation_tokens += 1

    wall_time_sec = time.perf_counter() - wall_start
    if prompt_tokens is None:
        prompt_tokens = _maybe_encode_prompt(tokenizer, prompt)
    if generation_tokens is None and counted_generation_tokens > 0:
        generation_tokens = counted_generation_tokens
    acceptance_mean = None
    if acceptance_lengths:
        acceptance_mean = float(statistics.mean(acceptance_lengths))
    return RunMetrics(
        mode=mode,
        repeat_index=0,
        block_size=block_size,
        wall_time_sec=wall_time_sec,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        peak_memory_gb=peak_memory_gb,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        finish_reason=finish_reason,
        acceptance_lengths=acceptance_lengths,
        acceptance_mean=acceptance_mean,
        output_text="".join(output_parts),
    )


def _aggregate_metrics(metrics: list[RunMetrics]) -> dict[str, Any]:
    def mean_of(name: str) -> float | None:
        values = [getattr(item, name) for item in metrics if getattr(item, name) is not None]
        if not values:
            return None
        return float(statistics.mean(values))

    accept_lengths = [value for item in metrics for value in item.acceptance_lengths]
    return {
        "repeats": len(metrics),
        "mean_wall_time_sec": mean_of("wall_time_sec"),
        "mean_prompt_tps": mean_of("prompt_tps"),
        "mean_generation_tps": mean_of("generation_tps"),
        "mean_peak_memory_gb": mean_of("peak_memory_gb"),
        "mean_prompt_tokens": mean_of("prompt_tokens"),
        "mean_generation_tokens": mean_of("generation_tokens"),
        "mean_acceptance_length": (
            float(statistics.mean(accept_lengths)) if accept_lengths else None
        ),
    }


def run(args: argparse.Namespace) -> int:
    prompt = _resolve_prompt(args)
    run_dir = _prepare_run_dir(args.artifacts_dir, args.run_name)

    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt)

    config_path = run_dir / "config.json"
    config_data = vars(args).copy()
    config_data["prompt_chars"] = len(prompt)
    config_path.write_text(json.dumps(config_data, indent=2))

    resolved = {
        "mode": args.mode,
        "model": args.model,
        "draft_model": args.draft_model if args.mode == "dflash" else None,
        "block_size": args.block_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "warmup_passes": args.warmup_passes,
        "warmup_tokens": args.warmup_tokens,
        "prime_baseline_tokens": args.prime_baseline_tokens,
        "max_kv_size": args.max_kv_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "draft_sliding_window_size": args.draft_sliding_window_size,
        "run_dir": str(run_dir),
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0

    runtime = _load_runtime()
    _seed_mx(runtime, args.seed)
    sampler = runtime["make_sampler"](temp=args.temperature)

    target_model, tokenizer = runtime["load_target"](args.model)
    rendered_prompt = prompt if args.disable_chat_template else _apply_chat_template(tokenizer, prompt)

    draft_model = None
    effective_block_size = args.block_size
    if args.mode == "dflash":
        draft_model = runtime["load_draft"](
            args.draft_model,
            sliding_window_size=args.draft_sliding_window_size,
        )
        if effective_block_size is None:
            effective_block_size = int(draft_model.config.block_size)
        if args.prime_baseline_tokens > 0:
            _clear_cache(runtime)
            list(
                runtime["baseline_stream_generate"](
                    target_model,
                    tokenizer,
                    rendered_prompt,
                    args.prime_baseline_tokens,
                    sampler=sampler,
                    max_kv_size=args.max_kv_size,
                )
            )

    results: list[RunMetrics] = []

    for _ in range(args.warmup_passes):
        _clear_cache(runtime)
        if args.mode == "baseline":
            list(
                runtime["baseline_stream_generate"](
                    target_model,
                    tokenizer,
                    rendered_prompt,
                    args.warmup_tokens,
                    sampler=sampler,
                    max_kv_size=args.max_kv_size,
                )
            )
        else:
            list(
                runtime["dflash_stream_generate"](
                    target_model,
                    draft_model,
                    tokenizer,
                    rendered_prompt,
                    block_size=effective_block_size,
                    max_tokens=args.warmup_tokens,
                    sampler=sampler,
                    max_kv_size=args.max_kv_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                )
            )

    for repeat_index in range(args.repeats):
        _clear_cache(runtime)
        if args.mode == "baseline":
            run_metrics = _run_generation(
                mode=args.mode,
                block_size=None,
                iterator_factory=lambda: runtime["baseline_stream_generate"](
                    target_model,
                    tokenizer,
                    rendered_prompt,
                    args.max_new_tokens,
                    sampler=sampler,
                    max_kv_size=args.max_kv_size,
                ),
                tokenizer=tokenizer,
                prompt=rendered_prompt,
            )
        else:
            run_metrics = _run_generation(
                mode=args.mode,
                block_size=effective_block_size,
                iterator_factory=lambda: runtime["dflash_stream_generate"](
                    target_model,
                    draft_model,
                    tokenizer,
                    rendered_prompt,
                    block_size=effective_block_size,
                    max_tokens=args.max_new_tokens,
                    sampler=sampler,
                    max_kv_size=args.max_kv_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                ),
                tokenizer=tokenizer,
                prompt=rendered_prompt,
            )
        run_metrics.repeat_index = repeat_index
        results.append(run_metrics)

    payload = {
        "config": {
            **resolved,
            "block_size": effective_block_size,
            "prompt": prompt,
            "disable_chat_template": args.disable_chat_template,
            "prompt_tokens": _maybe_encode_prompt(tokenizer, rendered_prompt),
        },
        "summary": _aggregate_metrics(results),
        "runs": [asdict(metric) for metric in results],
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Result JSON: {result_path}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise
