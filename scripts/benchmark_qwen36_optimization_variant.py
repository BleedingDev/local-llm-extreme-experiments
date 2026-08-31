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
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.6-35B-A3B-DFlash"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "benchmarks" / "qwen36-optimization-matrix"
DEFAULT_PROMPT = (
    "Write a concise engineering note about what matters when benchmarking a local "
    "Apple Silicon language model for agent workloads."
)
CONTEXT_FILLER = (
    "Local inference benchmark record. The workload includes code editing, tool calls, "
    "long repository context, deterministic summaries, and careful memory accounting. "
)

VARIANTS = (
    "baseline",
    "mlx-kv4",
    "dflash",
    "dflash-kv4",
    "dflash-turboquant-lean",
    "dflash-turboquant-rot",
    "dflash-triattention",
)


@dataclass
class RunMetrics:
    variant: str
    phase: str
    label: str
    status: str
    error: str | None
    prompt_tokens_target: int | None
    prompt_tokens: int | None
    max_tokens: int
    max_kv_size: int | None
    wall_time_sec: float
    time_to_first_token_sec: float | None
    prompt_tps: float | None
    generation_tps: float | None
    peak_memory_gb: float | None
    generation_tokens: int | None
    finish_reason: str | None
    acceptance_mean: float | None
    acceptance_min: int | None
    acceptance_max: int | None
    triattention_cache_len: int | None
    triattention_compressions: int
    output_preview: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one Qwen3.6 RotorQuant MLX optimization variant."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument(
        "--mode",
        choices=("throughput", "context", "generation", "all"),
        default="all",
    )
    parser.add_argument("--throughput-tokens", type=_positive_int, default=256)
    parser.add_argument("--throughput-repeats", type=_positive_int, default=2)
    parser.add_argument("--throughput-kv-sizes", type=_int_list, default=[2048, 8192])
    parser.add_argument("--context-targets", type=_int_list, default=[4096, 16384, 32768, 65536])
    parser.add_argument("--context-new-tokens", type=_positive_int, default=32)
    parser.add_argument(
        "--unsafe-allow-large-context",
        action="store_true",
        help="Allow context targets above 65536 tokens. This can hard-crash memory-constrained Macs.",
    )
    parser.add_argument("--generation-targets", type=_int_list, default=[8192])
    parser.add_argument("--generation-kv-size", type=_positive_int, default=12288)
    parser.add_argument("--disable-eos-stop", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=512)
    parser.add_argument("--draft-sliding-window-size", type=_positive_int, default=4096)
    parser.add_argument("--triattention-kv-budget", type=_positive_int, default=2048)
    parser.add_argument("--triattention-divide-length", type=_positive_int, default=8)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _ensure_vendor_paths() -> None:
    for rel in ("vendor/triattention", "vendor/turboquant-mlx"):
        path = str(ROOT_DIR / rel)
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)


def _load_runtime(variant: str) -> dict[str, Any]:
    _ensure_vendor_paths()
    import mlx.core as mx  # noqa: WPS433
    from mlx_lm import load as mlx_load, stream_generate as mlx_stream_generate  # noqa: WPS433
    from mlx_lm.sample_utils import make_sampler  # noqa: WPS433

    runtime: dict[str, Any] = {
        "mx": mx,
        "mlx_load": mlx_load,
        "mlx_stream_generate": mlx_stream_generate,
        "make_sampler": make_sampler,
    }
    if variant.startswith("dflash"):
        from dflash.model_mlx import load as dflash_load  # noqa: WPS433
        from dflash.model_mlx import load_draft, stream_generate as dflash_stream_generate  # noqa: WPS433

        runtime.update(
            {
                "dflash_load": dflash_load,
                "load_draft": load_draft,
                "dflash_stream_generate": dflash_stream_generate,
            }
        )
    return runtime


def _resolve_prompt(args: argparse.Namespace) -> str:
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty")
    return prompt


def _prepare_run_dir(args: argparse.Namespace) -> Path:
    root = Path(args.artifacts_dir).expanduser()
    if not root.is_absolute():
        root = ROOT_DIR / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.run_name.strip() or f"{args.variant}-{stamp}"
    run_dir = root / name
    if run_dir.exists():
        run_dir = root / f"{name}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _count_tokens(tokenizer: Any, prompt: str | list[int]) -> int:
    if isinstance(prompt, list):
        return len(prompt)
    return len(_encode(tokenizer, prompt))


def _make_context_prompt(tokenizer: Any, target_tokens: int) -> str:
    instruction = (
        "\n\nAfter reading the repeated benchmark records above, reply with exactly "
        "four short bullet points about the workload. Do not mention repetition.\n"
    )
    low = 1
    high = 1
    while _count_tokens(tokenizer, _apply_chat_template(tokenizer, CONTEXT_FILLER * high + instruction)) < target_tokens:
        high *= 2

    while low < high:
        mid = (low + high) // 2
        token_count = _count_tokens(tokenizer, _apply_chat_template(tokenizer, CONTEXT_FILLER * mid + instruction))
        if token_count >= target_tokens:
            high = mid
        else:
            low = mid + 1

    return _apply_chat_template(tokenizer, CONTEXT_FILLER * low + instruction)


def _seed(runtime: dict[str, Any], seed: int) -> None:
    try:
        runtime["mx"].random.seed(seed)
    except Exception:
        pass


def _clear_cache(runtime: dict[str, Any]) -> None:
    try:
        runtime["mx"].clear_cache()
    except Exception:
        pass


def _mean(values: list[int]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _variant_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant == "baseline":
        return {}
    if args.variant == "mlx-kv4":
        return {"kv_bits": 4, "kv_group_size": 64, "quantized_kv_start": 0}
    if args.variant == "dflash":
        return {}
    if args.variant == "dflash-kv4":
        return {
            "cache_optimization": "kv-quant",
            "kv_bits": 4,
            "kv_group_size": 64,
            "quantized_kv_start": 0,
        }
    if args.variant == "dflash-turboquant-lean":
        return {
            "cache_optimization": "turboquant",
            "turboquant_strategy": "tqv2_4bit_lean",
            "quantized_kv_start": 0,
        }
    if args.variant == "dflash-turboquant-rot":
        return {
            "cache_optimization": "turboquant",
            "turboquant_strategy": "tqv2_4bit_rot",
            "quantized_kv_start": 0,
        }
    if args.variant == "dflash-triattention":
        return {
            "triattention_enable": True,
            "triattention_kv_budget": args.triattention_kv_budget,
            "triattention_divide_length": args.triattention_divide_length,
        }
    raise ValueError(f"unsupported variant: {args.variant}")


def _run_one(
    *,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    model: Any,
    draft: Any,
    tokenizer: Any,
    sampler: Any,
    phase: str,
    label: str,
    prompt: str,
    prompt_tokens_target: int | None,
    max_tokens: int,
    max_kv_size: int | None,
) -> RunMetrics:
    _clear_cache(runtime)
    start = time.perf_counter()
    first_token_at = None
    output_parts: list[str] = []
    prompt_tps = None
    generation_tps = None
    peak_memory_gb = None
    prompt_tokens = None
    generation_tokens = None
    finish_reason = None
    acceptance_lengths: list[int] = []
    triattention_cache_len = None
    triattention_compressions = 0
    counted_tokens = 0

    try:
        if args.variant.startswith("dflash"):
            kwargs: dict[str, Any] = {
                "max_tokens": max_tokens,
                "sampler": sampler,
                "max_kv_size": max_kv_size,
                "prefill_chunk_size": args.prefill_chunk_size,
                **_variant_kwargs(args),
            }
            iterator = runtime["dflash_stream_generate"](model, draft, tokenizer, prompt, **kwargs)
        else:
            kwargs = {"sampler": sampler, "max_kv_size": max_kv_size, **_variant_kwargs(args)}
            iterator = runtime["mlx_stream_generate"](model, tokenizer, prompt, max_tokens, **kwargs)

        for response in iterator:
            if first_token_at is None:
                first_token_at = time.perf_counter()
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
            if getattr(response, "triattention_cache_len", None) is not None:
                triattention_cache_len = int(response.triattention_cache_len)
            if getattr(response, "triattention_compressions", None) is not None:
                triattention_compressions = int(response.triattention_compressions)

            tokens = getattr(response, "tokens", None)
            if tokens is not None:
                counted_tokens += len(tokens)
            elif getattr(response, "token", None) is not None:
                counted_tokens += 1

        wall = time.perf_counter() - start
        if prompt_tokens is None:
            prompt_tokens = _count_tokens(tokenizer, prompt)
        if generation_tokens is None and counted_tokens:
            generation_tokens = counted_tokens
        return RunMetrics(
            variant=args.variant,
            phase=phase,
            label=label,
            status="ok",
            error=None,
            prompt_tokens_target=prompt_tokens_target,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            max_kv_size=max_kv_size,
            wall_time_sec=wall,
            time_to_first_token_sec=(first_token_at - start) if first_token_at is not None else None,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            peak_memory_gb=peak_memory_gb,
            generation_tokens=generation_tokens,
            finish_reason=finish_reason,
            acceptance_mean=_mean(acceptance_lengths),
            acceptance_min=min(acceptance_lengths) if acceptance_lengths else None,
            acceptance_max=max(acceptance_lengths) if acceptance_lengths else None,
            triattention_cache_len=triattention_cache_len,
            triattention_compressions=triattention_compressions,
            output_preview="".join(output_parts)[:500],
        )
    except Exception as exc:
        wall = time.perf_counter() - start
        return RunMetrics(
            variant=args.variant,
            phase=phase,
            label=label,
            status="failed",
            error=repr(exc),
            prompt_tokens_target=prompt_tokens_target,
            prompt_tokens=prompt_tokens or _count_tokens(tokenizer, prompt),
            max_tokens=max_tokens,
            max_kv_size=max_kv_size,
            wall_time_sec=wall,
            time_to_first_token_sec=(first_token_at - start) if first_token_at is not None else None,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            peak_memory_gb=peak_memory_gb,
            generation_tokens=generation_tokens,
            finish_reason=finish_reason,
            acceptance_mean=_mean(acceptance_lengths),
            acceptance_min=min(acceptance_lengths) if acceptance_lengths else None,
            acceptance_max=max(acceptance_lengths) if acceptance_lengths else None,
            triattention_cache_len=triattention_cache_len,
            triattention_compressions=triattention_compressions,
            output_preview="".join(output_parts)[:500],
        )


def _summarize(runs: list[RunMetrics]) -> dict[str, Any]:
    ok = [run for run in runs if run.status == "ok"]

    def mean_attr(name: str) -> float | None:
        values = [getattr(run, name) for run in ok if getattr(run, name) is not None]
        return float(statistics.mean(values)) if values else None

    return {
        "runs": len(runs),
        "ok": len(ok),
        "failed": len(runs) - len(ok),
        "mean_time_to_first_token_sec": mean_attr("time_to_first_token_sec"),
        "mean_prompt_tps": mean_attr("prompt_tps"),
        "mean_generation_tps": mean_attr("generation_tps"),
        "mean_peak_memory_gb": mean_attr("peak_memory_gb"),
        "max_successful_context_tokens": max(
            [run.prompt_tokens or 0 for run in ok if run.phase == "context"],
            default=None,
        ),
        "max_successful_generation_tokens": max(
            [run.generation_tokens or 0 for run in ok if run.phase == "generation"],
            default=None,
        ),
    }


def _write_payload(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    load_time_sec: float | None,
    runs: list[RunMetrics],
) -> dict[str, Any]:
    payload = {
        "variant": args.variant,
        "model": args.model,
        "draft_model": args.draft_model if args.variant.startswith("dflash") else None,
        "load_time_sec": load_time_sec,
        "run_dir": str(run_dir),
        "summary": _summarize(runs),
        "runs": [asdict(run) for run in runs],
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> int:
    args = parse_args()
    if (
        args.mode in {"context", "all"}
        and not args.unsafe_allow_large_context
        and any(target > 65536 for target in args.context_targets)
    ):
        raise ValueError("context targets above 65536 require --unsafe-allow-large-context")

    prompt = _resolve_prompt(args)
    run_dir = _prepare_run_dir(args)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "prompt": prompt}, indent=2))

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "config": {**vars(args), "prompt": prompt}}, indent=2))
        return 0

    runtime = _load_runtime(args.variant)
    _seed(runtime, args.seed)
    sampler = runtime["make_sampler"](temp=args.temperature, top_p=args.top_p)

    load_start = time.perf_counter()
    if args.variant.startswith("dflash"):
        model, tokenizer = runtime["dflash_load"](args.model)
        draft = runtime["load_draft"](
            args.draft_model,
            sliding_window_size=args.draft_sliding_window_size,
        )
    else:
        model, tokenizer = runtime["mlx_load"](args.model)
        draft = None
    load_time_sec = time.perf_counter() - load_start

    if args.disable_eos_stop:
        tokenizer.eos_token_ids = set()

    rendered_prompt = _apply_chat_template(tokenizer, prompt)
    runs: list[RunMetrics] = []

    if args.mode in {"throughput", "all"}:
        for kv_size in args.throughput_kv_sizes:
            for repeat in range(args.throughput_repeats):
                run = _run_one(
                    args=args,
                    runtime=runtime,
                    model=model,
                    draft=draft,
                    tokenizer=tokenizer,
                    sampler=sampler,
                    phase="throughput",
                    label=f"kv{kv_size}-rep{repeat + 1}",
                    prompt=rendered_prompt,
                    prompt_tokens_target=None,
                    max_tokens=args.throughput_tokens,
                    max_kv_size=kv_size,
                )
                runs.append(run)
                _write_payload(run_dir=run_dir, args=args, load_time_sec=load_time_sec, runs=runs)
                if run.status != "ok":
                    break

    if args.mode in {"context", "all"}:
        for target in args.context_targets:
            context_prompt = _make_context_prompt(tokenizer, target)
            context_tokens = _count_tokens(tokenizer, context_prompt)
            run = _run_one(
                args=args,
                runtime=runtime,
                model=model,
                draft=draft,
                tokenizer=tokenizer,
                sampler=sampler,
                phase="context",
                label=f"ctx{target}",
                prompt=context_prompt,
                prompt_tokens_target=target,
                max_tokens=args.context_new_tokens,
                max_kv_size=context_tokens + args.context_new_tokens + 16,
            )
            runs.append(run)
            _write_payload(run_dir=run_dir, args=args, load_time_sec=load_time_sec, runs=runs)
            if run.status != "ok":
                break

    if args.mode in {"generation", "all"}:
        for target in args.generation_targets:
            run = _run_one(
                args=args,
                runtime=runtime,
                model=model,
                draft=draft,
                tokenizer=tokenizer,
                sampler=sampler,
                phase="generation",
                label=f"gen{target}",
                prompt=rendered_prompt,
                prompt_tokens_target=None,
                max_tokens=target,
                max_kv_size=args.generation_kv_size,
            )
            runs.append(run)
            _write_payload(run_dir=run_dir, args=args, load_time_sec=load_time_sec, runs=runs)
            if run.status != "ok":
                break

    payload = _write_payload(run_dir=run_dir, args=args, load_time_sec=load_time_sec, runs=runs)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise
