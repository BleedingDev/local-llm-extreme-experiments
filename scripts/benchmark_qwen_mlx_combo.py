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
DEFAULT_PROMPT = (
    "Explain in one short paragraph how speculative decoding and quantization "
    "can complement each other on Apple Silicon."
)
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "benchmarks" / "qwen-mlx-combo"
DEFAULT_MODES = ("dflash", "paro", "both")


@dataclass
class RunMetrics:
    mode: str
    repeat_index: int
    wall_time_sec: float
    prompt_tps: float | None
    generation_tps: float | None
    peak_memory_gb: float | None
    prompt_tokens: int | None
    generation_tokens: int | None
    finish_reason: str | None
    output_text: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLX combinations for Qwen-family targets: DFlash-only, ParoQuant-only, and ParoQuant+DFlash."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3.6-35B-A3B-4bit",
        help="Baseline/DFlash target model id/path",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="z-lab/Qwen3.6-35B-A3B-DFlash",
        help="DFlash draft model id/path",
    )
    parser.add_argument(
        "--paro-model",
        type=str,
        default="",
        help="ParoQuant target model id/path for 'paro' and 'both' modes",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated subset of: baseline,dflash,paro,both",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--prompt-file", type=str, default="", help="Read prompt text from file")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=64, help="Generation token limit")
    parser.add_argument("--temperature", type=_non_negative_float, default=0.0, help="Sampling temperature")
    parser.add_argument("--repeats", type=_positive_int, default=1, help="Number of repeats per mode")
    parser.add_argument("--warmup-passes", type=_positive_int, default=1, help="Warmup passes per mode before timing")
    parser.add_argument("--warmup-tokens", type=_positive_int, default=3, help="Warmup token limit per pass")
    parser.add_argument("--max-kv-size", type=_positive_int, default=2048, help="KV cache cap")
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=512, help="DFlash prefill chunk size")
    parser.add_argument(
        "--draft-sliding-window-size",
        type=_positive_int,
        default=4096,
        help="Optional DFlash draft sliding window size",
    )
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


def _parse_modes(raw_modes: str) -> list[str]:
    modes = [part.strip().lower() for part in raw_modes.split(",") if part.strip()]
    allowed = {"baseline", "dflash", "paro", "both"}
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"Unsupported modes: {', '.join(invalid)}")
    if not modes:
        raise ValueError("At least one mode is required.")
    return modes


def _prepare_run_dir(artifacts_dir: str, run_name: str) -> Path:
    root = Path(artifacts_dir).expanduser()
    if not root.is_absolute():
        root = ROOT_DIR / root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name.strip() or f"qwen-mlx-combo-{stamp}"
    run_dir = root / name
    if run_dir.exists():
        run_dir = root / f"{name}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_runtime() -> dict[str, Any]:
    try:
        import mlx.core as mx  # noqa: WPS433
        from mlx_lm import stream_generate as mlx_stream_generate  # noqa: WPS433
        from mlx_lm.sample_utils import make_sampler  # noqa: WPS433
        from dflash.model_mlx import load as load_target  # noqa: WPS433
        from dflash.model_mlx import load_draft, stream_generate as dflash_stream_generate  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(
            "Missing MLX/DFlash runtime pieces. Ensure .venv has mlx, mlx-lm, and dflash installed."
        ) from exc

    try:
        from paroquant.inference.backends.mlx.load import load as load_paro  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(
            "ParoQuant MLX backend is unavailable. Install vendor/paroquant or set PAROQUANT_INSTALL_SPEC and rerun setup."
        ) from exc

    return {
        "mx": mx,
        "mlx_stream_generate": mlx_stream_generate,
        "make_sampler": make_sampler,
        "load_target": load_target,
        "load_draft": load_draft,
        "dflash_stream_generate": dflash_stream_generate,
        "load_paro": load_paro,
    }


def _normalize_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


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


def _run_generation(*, iterator_factory, tokenizer: Any, prompt: str) -> RunMetrics:
    wall_start = time.perf_counter()
    output_parts: list[str] = []
    prompt_tps = None
    generation_tps = None
    peak_memory_gb = None
    prompt_tokens = None
    generation_tokens = None
    finish_reason = None

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

    wall_time_sec = time.perf_counter() - wall_start
    if prompt_tokens is None:
        prompt_tokens = _maybe_encode_prompt(tokenizer, prompt)
    return RunMetrics(
        mode="",
        repeat_index=0,
        wall_time_sec=wall_time_sec,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        peak_memory_gb=peak_memory_gb,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        finish_reason=finish_reason,
        output_text="".join(output_parts),
    )


def _aggregate_metrics(metrics: list[RunMetrics]) -> dict[str, Any]:
    def mean_of(name: str) -> float | None:
        values = [getattr(item, name) for item in metrics if getattr(item, name) is not None]
        if not values:
            return None
        return float(statistics.mean(values))

    return {
        "repeats": len(metrics),
        "mean_wall_time_sec": mean_of("wall_time_sec"),
        "mean_prompt_tps": mean_of("prompt_tps"),
        "mean_generation_tps": mean_of("generation_tps"),
        "mean_peak_memory_gb": mean_of("peak_memory_gb"),
        "mean_prompt_tokens": mean_of("prompt_tokens"),
        "mean_generation_tokens": mean_of("generation_tokens"),
        "finish_reasons": sorted({item.finish_reason for item in metrics if item.finish_reason}),
    }


def main() -> int:
    args = parse_args()
    prompt = _resolve_prompt(args)
    modes = _parse_modes(args.modes)
    if any(mode in {"paro", "both"} for mode in modes) and not args.paro_model:
        raise ValueError(
            "--paro-model is required for 'paro' and 'both' modes. "
            "As of 2026-04-21, a public z-lab Qwen3.6-35B-A3B PARO checkpoint was not found via the Hugging Face API."
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "draft_model": args.draft_model,
                    "paro_model": args.paro_model,
                    "modes": modes,
                    "prompt": prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "repeats": args.repeats,
                    "warmup_passes": args.warmup_passes,
                    "warmup_tokens": args.warmup_tokens,
                    "max_kv_size": args.max_kv_size,
                    "prefill_chunk_size": args.prefill_chunk_size,
                    "draft_sliding_window_size": args.draft_sliding_window_size,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    run_dir = _prepare_run_dir(args.artifacts_dir, args.run_name)
    runtime = _load_runtime()
    mx = runtime["mx"]
    make_sampler = runtime["make_sampler"]
    sampler = make_sampler(temp=args.temperature)

    results: dict[str, Any] = {
        "config": {
            "model": args.model,
            "draft_model": args.draft_model,
            "paro_model": args.paro_model,
            "modes": modes,
            "prompt": prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "repeats": args.repeats,
            "warmup_passes": args.warmup_passes,
            "warmup_tokens": args.warmup_tokens,
            "max_kv_size": args.max_kv_size,
            "prefill_chunk_size": args.prefill_chunk_size,
            "draft_sliding_window_size": args.draft_sliding_window_size,
        },
        "run_dir": str(run_dir),
        "per_mode": {},
    }

    base_prompt_token_count = None
    if "baseline" in modes or "dflash" in modes:
        base_target_model, base_tokenizer = runtime["load_target"](args.model)
        base_prompt = _apply_chat_template(base_tokenizer, prompt)
        base_prompt_token_count = _maybe_encode_prompt(base_tokenizer, base_prompt)
    else:
        base_target_model, base_tokenizer, base_prompt = None, None, None

    draft_model = None
    if "dflash" in modes or "both" in modes:
        draft_model = runtime["load_draft"](
            args.draft_model,
            sliding_window_size=args.draft_sliding_window_size,
        )

    paro_target_model, paro_processor, _ = (None, None, None)
    paro_tokenizer = None
    paro_prompt = None
    if "paro" in modes or "both" in modes:
        paro_target_model, paro_processor, _ = runtime["load_paro"](args.paro_model, force_text=True)
        paro_tokenizer = _normalize_tokenizer(paro_processor)
        paro_prompt = _apply_chat_template(paro_tokenizer, prompt)

    for mode in modes:
        mode_metrics: list[RunMetrics] = []
        for _ in range(args.warmup_passes):
            mx.clear_cache()
            if mode == "baseline":
                list(
                    runtime["mlx_stream_generate"](
                        base_target_model,
                        base_tokenizer,
                        base_prompt,
                        args.warmup_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                    )
                )
            elif mode == "dflash":
                list(
                    runtime["dflash_stream_generate"](
                        base_target_model,
                        draft_model,
                        base_tokenizer,
                        base_prompt,
                        max_tokens=args.warmup_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                    )
                )
            elif mode == "paro":
                list(
                    runtime["mlx_stream_generate"](
                        paro_target_model,
                        paro_tokenizer,
                        paro_prompt,
                        args.warmup_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                    )
                )
            else:
                list(
                    runtime["dflash_stream_generate"](
                        paro_target_model,
                        draft_model,
                        paro_tokenizer,
                        paro_prompt,
                        max_tokens=args.warmup_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                    )
                )

        for repeat_index in range(args.repeats):
            mx.clear_cache()
            if mode == "baseline":
                run = _run_generation(
                    iterator_factory=lambda: runtime["mlx_stream_generate"](
                        base_target_model,
                        base_tokenizer,
                        base_prompt,
                        args.max_new_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                    ),
                    tokenizer=base_tokenizer,
                    prompt=base_prompt,
                )
            elif mode == "dflash":
                run = _run_generation(
                    iterator_factory=lambda: runtime["dflash_stream_generate"](
                        base_target_model,
                        draft_model,
                        base_tokenizer,
                        base_prompt,
                        max_tokens=args.max_new_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                    ),
                    tokenizer=base_tokenizer,
                    prompt=base_prompt,
                )
            elif mode == "paro":
                run = _run_generation(
                    iterator_factory=lambda: runtime["mlx_stream_generate"](
                        paro_target_model,
                        paro_tokenizer,
                        paro_prompt,
                        args.max_new_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                    ),
                    tokenizer=paro_tokenizer,
                    prompt=paro_prompt,
                )
            else:
                run = _run_generation(
                    iterator_factory=lambda: runtime["dflash_stream_generate"](
                        paro_target_model,
                        draft_model,
                        paro_tokenizer,
                        paro_prompt,
                        max_tokens=args.max_new_tokens,
                        sampler=sampler,
                        max_kv_size=args.max_kv_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                    ),
                    tokenizer=paro_tokenizer,
                    prompt=paro_prompt,
                )

            run.mode = mode
            run.repeat_index = repeat_index
            if run.prompt_tokens is None and mode in {"baseline", "dflash"}:
                run.prompt_tokens = base_prompt_token_count
            mode_metrics.append(run)

        results["per_mode"][mode] = {
            "summary": _aggregate_metrics(mode_metrics),
            "runs": [asdict(metric) for metric in mode_metrics],
        }

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results["per_mode"], indent=2, ensure_ascii=False))
    print(f"Result JSON: {result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise
