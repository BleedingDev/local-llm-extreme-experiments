#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "majentik/Qwen3.6-35B-A3B-RotorQuant-MLX-3bit"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "benchmarks" / "qwen36-rotorquant-mlx"
DEFAULT_PROMPT = (
    "Write a concise engineering note about what matters when benchmarking a local "
    "Apple Silicon language model for agent workloads."
)
CONTEXT_FILLER = (
    "Local inference benchmark record. The workload includes code editing, tool calls, "
    "long repository context, deterministic summaries, and careful memory accounting. "
)


@dataclass
class RunMetrics:
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
        description="Benchmark Qwen3.6-35B-A3B RotorQuant MLX 3-bit with TTFT, context, and long-generation probes."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX model id or local path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Short throughput prompt")
    parser.add_argument("--prompt-file", default="", help="Read short throughput prompt from file")
    parser.add_argument(
        "--mode",
        choices=("throughput", "context", "generation", "all"),
        default="all",
        help="Benchmark phase to run",
    )
    parser.add_argument("--throughput-tokens", type=_positive_int, default=256)
    parser.add_argument("--throughput-repeats", type=_positive_int, default=3)
    parser.add_argument("--throughput-kv-sizes", type=_int_list, default=[2048, 4096, 8192])
    parser.add_argument("--context-targets", type=_int_list, default=[4096, 16384, 32768, 65536])
    parser.add_argument("--context-new-tokens", type=_positive_int, default=32)
    parser.add_argument(
        "--unsafe-allow-large-context",
        action="store_true",
        help="Allow context targets above 65536 tokens. This can hard-crash memory-constrained Macs.",
    )
    parser.add_argument("--generation-targets", type=_int_list, default=[512, 1024, 2048, 4096])
    parser.add_argument("--generation-kv-size", type=_positive_int, default=16384)
    parser.add_argument(
        "--disable-eos-stop",
        action="store_true",
        help="Clear tokenizer EOS stop ids so generation stress runs stop only at max_tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _load_runtime() -> dict[str, Any]:
    import mlx.core as mx  # noqa: WPS433
    from mlx_lm import load, stream_generate  # noqa: WPS433
    from mlx_lm.sample_utils import make_sampler  # noqa: WPS433

    return {
        "mx": mx,
        "load": load,
        "stream_generate": stream_generate,
        "make_sampler": make_sampler,
    }


def _prepare_run_dir(artifacts_dir: str, run_name: str) -> Path:
    root = Path(artifacts_dir).expanduser()
    if not root.is_absolute():
        root = ROOT_DIR / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"qwen36-rotorquant-mlx-{stamp}"
    run_dir = root / name
    if run_dir.exists():
        run_dir = root / f"{name}-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    else:
        prompt = args.prompt
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty")
    return prompt


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


def _run_one(
    *,
    runtime: dict[str, Any],
    model: Any,
    tokenizer: Any,
    sampler: Any,
    phase: str,
    label: str,
    prompt: str | list[int],
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
    counted_tokens = 0

    try:
        kwargs: dict[str, Any] = {
            "sampler": sampler,
        }
        if max_kv_size is not None:
            kwargs["max_kv_size"] = max_kv_size

        for response in runtime["stream_generate"](model, tokenizer, prompt, max_tokens, **kwargs):
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
            output_preview="".join(output_parts)[:500],
        )
    except Exception as exc:
        wall = time.perf_counter() - start
        return RunMetrics(
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
            output_preview="".join(output_parts)[:500],
        )


def _summarize(runs: list[RunMetrics]) -> dict[str, Any]:
    ok = [run for run in runs if run.status == "ok"]

    def mean(name: str) -> float | None:
        values = [getattr(run, name) for run in ok if getattr(run, name) is not None]
        if not values:
            return None
        return float(statistics.mean(values))

    return {
        "runs": len(runs),
        "ok": len(ok),
        "failed": len(runs) - len(ok),
        "mean_time_to_first_token_sec": mean("time_to_first_token_sec"),
        "mean_generation_tps": mean("generation_tps"),
        "mean_prompt_tps": mean("prompt_tps"),
        "mean_peak_memory_gb": mean("peak_memory_gb"),
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
    model_id: str,
    load_time_sec: float | None,
    runs: list[RunMetrics],
) -> dict[str, Any]:
    payload = {
        "model": model_id,
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
        raise ValueError(
            "context targets above 65536 require --unsafe-allow-large-context; "
            "131072 hard-crashed this 34 GB M5 machine during benchmarking"
        )
    prompt = _resolve_prompt(args)
    run_dir = _prepare_run_dir(args.artifacts_dir, args.run_name)
    config = vars(args).copy()
    config["prompt"] = prompt
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "config": config}, indent=2))
        return 0

    runtime = _load_runtime()
    _seed(runtime, args.seed)
    sampler = runtime["make_sampler"](temp=args.temperature, top_p=args.top_p)

    load_start = time.perf_counter()
    model, tokenizer = runtime["load"](args.model)
    load_time_sec = time.perf_counter() - load_start
    if args.disable_eos_stop:
        tokenizer.eos_token_ids = set()

    rendered_prompt = _apply_chat_template(tokenizer, prompt)
    runs: list[RunMetrics] = []

    if args.mode in {"throughput", "all"}:
        for kv_size in args.throughput_kv_sizes:
            for repeat in range(args.throughput_repeats):
                runs.append(
                    _run_one(
                        runtime=runtime,
                        model=model,
                        tokenizer=tokenizer,
                        sampler=sampler,
                        phase="throughput",
                        label=f"kv{kv_size}-rep{repeat + 1}",
                        prompt=rendered_prompt,
                        prompt_tokens_target=None,
                        max_tokens=args.throughput_tokens,
                        max_kv_size=kv_size,
                    )
                )
                _write_payload(
                    run_dir=run_dir,
                    model_id=args.model,
                    load_time_sec=load_time_sec,
                    runs=runs,
                )

    if args.mode in {"context", "all"}:
        for target in args.context_targets:
            context_prompt = _make_context_prompt(tokenizer, target)
            context_prompt_tokens = _count_tokens(tokenizer, context_prompt)
            max_kv_size = context_prompt_tokens + args.context_new_tokens + 16
            run = _run_one(
                runtime=runtime,
                model=model,
                tokenizer=tokenizer,
                sampler=sampler,
                phase="context",
                label=f"ctx{target}",
                prompt=context_prompt,
                prompt_tokens_target=target,
                max_tokens=args.context_new_tokens,
                max_kv_size=max_kv_size,
            )
            runs.append(run)
            _write_payload(
                run_dir=run_dir,
                model_id=args.model,
                load_time_sec=load_time_sec,
                runs=runs,
            )
            if run.status != "ok":
                break

    if args.mode in {"generation", "all"}:
        for target in args.generation_targets:
            run = _run_one(
                runtime=runtime,
                model=model,
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
            _write_payload(
                run_dir=run_dir,
                model_id=args.model,
                load_time_sec=load_time_sec,
                runs=runs,
            )
            if run.status != "ok":
                break

    payload = _write_payload(
        run_dir=run_dir,
        model_id=args.model,
        load_time_sec=load_time_sec,
        runs=runs,
    )
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
