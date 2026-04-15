#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_lm


DEFAULT_MODEL = "Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
DEFAULT_ROTOR_MODEL = "majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit"
DEFAULT_DRAFT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_PROMPT = (
    "Write a concise engineering note describing how to benchmark model latency, "
    "throughput, and context limits."
)

VARIANTS = (
    "baseline",
    "speculative",
    "kv4",
    "triattention",
    "turboquant-v2-lean",
    "turboquant-v2-rot",
    "turboquant-v3-3.5",
    "rotorquant",
    "speculative-rotorquant",
)


@dataclass
class TriRuntime:
    compressor: Any
    step_fn: Any
    kv_indices: tuple[int, ...]
    compressions: int = 0


def _is_gemma26_a4b(model_id: str) -> bool:
    normalized = model_id.lower()
    return (
        ("26b" in normalized and "a4b" in normalized and "gemma" in normalized)
        or "supergemma4-26b" in normalized
    )


def _ensure_model_family(model_id: str, label: str) -> None:
    if not _is_gemma26_a4b(model_id):
        raise ValueError(
            f"{label} must be Gemma 4 26B A4B / SuperGemma 26B A4B family, got: {model_id}"
        )


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    if args.prompt:
        return args.prompt
    return DEFAULT_PROMPT


def _resolve_target_model(args: argparse.Namespace) -> str:
    if args.variant in {"rotorquant", "speculative-rotorquant"}:
        return args.rotor_model
    return args.model


def _needs_speculative(variant: str) -> bool:
    return variant in {"speculative", "speculative-rotorquant"}


def _needs_triattention(variant: str) -> bool:
    return variant == "triattention"


def _needs_turboquant(variant: str) -> bool:
    return variant.startswith("turboquant-")


def _json_output(payload: dict, output_json: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


def _preflight(args: argparse.Namespace) -> dict:
    checks: list[dict] = []

    try:
        import mlx_lm  # noqa: F401

        checks.append({"name": "mlx_lm", "status": "ok"})
    except Exception as exc:  # pragma: no cover - script path
        checks.append({"name": "mlx_lm", "status": "error", "error": str(exc)})

    if _needs_triattention(args.variant):
        try:
            from triattention.mlx import apply_triattention_mlx  # noqa: F401
            from triattention.mlx.triattention_mlx import (  # noqa: F401
                triattention_generate_step,
            )

            checks.append({"name": "triattention.mlx", "status": "ok"})
        except Exception as exc:  # pragma: no cover - script path
            checks.append(
                {"name": "triattention.mlx", "status": "error", "error": str(exc)}
            )

    if _needs_turboquant(args.variant):
        try:
            import turboquant.patch as tq_patch  # noqa: F401
            from turboquant.cache_v2 import TurboQuantKVCacheV2  # noqa: F401
            from turboquant.cache_v3 import TurboQuantKVCacheV3  # noqa: F401

            checks.append({"name": "turboquant", "status": "ok"})
            _ = tq_patch
        except Exception as exc:  # pragma: no cover - script path
            checks.append({"name": "turboquant", "status": "error", "error": str(exc)})

    ok = all(c["status"] == "ok" for c in checks)
    return {
        "status": "ok" if ok else "error",
        "mode": "preflight",
        "variant": args.variant,
        "target_model": _resolve_target_model(args),
        "draft_model": args.draft_model if _needs_speculative(args.variant) else None,
        "offline_only": bool(args.offline_only),
        "checks": checks,
    }


def _encode_prompt(tokenizer, prompt: str) -> mx.array:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                kwargs["enable_thinking"] = False
        except (TypeError, ValueError):
            pass
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **kwargs
        )
    else:
        formatted = prompt
    return mx.array(tokenizer.encode(formatted, add_special_tokens=False))


def _get_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers
    raise RuntimeError("Unable to resolve transformer layers from model object.")


def _make_turbo_cache(model, variant: str, seed: int):
    import turboquant.patch as tq_patch
    from mlx_lm.models.cache import KVCache, make_prompt_cache
    from turboquant.cache_v2 import TurboQuantKVCacheV2
    from turboquant.cache_v3 import TurboQuantKVCacheV3

    tq_patch.apply()

    layers = _get_layers(model)
    cache = make_prompt_cache(model)
    replaced = 0

    for idx, layer_cache in enumerate(cache):
        if not isinstance(layer_cache, KVCache):
            continue
        if idx >= len(layers):
            continue

        attention = getattr(layers[idx], "self_attn", None)
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            continue

        if variant == "turboquant-v2-lean":
            cache[idx] = TurboQuantKVCacheV2(
                head_dim=int(head_dim),
                bits=4,
                group_size=64,
                use_rotation=False,
                use_normalization=False,
                use_qjl=False,
                seed=seed + idx,
            )
        elif variant == "turboquant-v2-rot":
            cache[idx] = TurboQuantKVCacheV2(
                head_dim=int(head_dim),
                bits=4,
                group_size=64,
                use_rotation=True,
                use_normalization=True,
                use_qjl=False,
                seed=seed + idx,
            )
        elif variant == "turboquant-v3-3.5":
            v3_cache = TurboQuantKVCacheV3(
                head_dim=int(head_dim),
                bits=3,
                n_outlier=int(head_dim) // 2,
                outlier_bits=4,
                use_qjl=False,
                seed=seed + idx,
            )
            # mlx_lm attention path expects cache.group_size on quantized caches.
            v3_cache.group_size = 64
            cache[idx] = v3_cache
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported turboquant variant: {variant}")
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "TurboQuant variant requested, but no KVCache layers were replaceable."
        )
    return cache, replaced


def _cache_offset(cache_entry) -> int:
    return int(getattr(cache_entry, "offset", 0))


def _apply_triattention_step(
    runtime: TriRuntime,
    cache,
    *,
    is_prefill: bool,
    current_position: int,
) -> None:
    kv_states = [cache[idx].state for idx in runtime.kv_indices]
    if not is_prefill and kv_states:
        seq_len = int(kv_states[0][0].shape[2])
        cache_positions = getattr(runtime.compressor, "cache_positions", [])
        if seq_len <= 0:
            runtime.compressor.cache_positions = []
        elif len(cache_positions) != seq_len:
            start = max(current_position - seq_len + 1, 0)
            runtime.compressor.cache_positions = list(range(start, start + seq_len))
        runtime.compressor.absolute_position = current_position + 1

    new_states = runtime.step_fn(
        runtime.compressor,
        kv_states,
        is_prefill=is_prefill,
        current_position=current_position,
    )
    for idx, state in zip(runtime.kv_indices, new_states):
        cache[idx].state = state


def _init_triattention_runtime(args: argparse.Namespace, model, cache) -> TriRuntime:
    from mlx_lm.models.cache import KVCache
    from triattention.mlx import apply_triattention_mlx
    from triattention.mlx.triattention_mlx import triattention_generate_step

    compressor = apply_triattention_mlx(
        model,
        stats_path=args.triattention_stats_path,
        kv_budget=args.triattention_kv_budget,
        divide_length=args.triattention_divide_length,
        prefill_pin=args.triattention_prefill_pin,
        disable_trig=args.triattention_disable_trig,
        disable_mlr=args.triattention_disable_mlr,
    )
    kv_indices = tuple(i for i, c in enumerate(cache) if isinstance(c, KVCache))
    if not kv_indices:
        raise RuntimeError("TriAttention needs KVCache layers, but none were found.")
    return TriRuntime(compressor=compressor, step_fn=triattention_generate_step, kv_indices=kv_indices)


def _enforce_max_kv_size(cache, max_kv_size: int) -> None:
    if max_kv_size <= 0:
        return
    for entry in cache:
        offset = int(getattr(entry, "offset", 0))
        if offset <= max_kv_size:
            continue
        if hasattr(entry, "trim"):
            entry.trim(offset - max_kv_size)


def _run_stream_variant(args: argparse.Namespace, model, tokenizer, prompt: str) -> dict:
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)
    target_kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "sampler": sampler,
    }
    if not _needs_speculative(args.variant):
        target_kwargs["max_kv_size"] = args.max_kv_size
    if args.variant == "kv4":
        target_kwargs.update(
            {
                "kv_bits": args.kv_bits,
                "kv_group_size": args.kv_group_size,
                "quantized_kv_start": args.quantized_kv_start,
            }
        )

    draft_model = None
    if _needs_speculative(args.variant):
        draft_model, _ = mlx_lm.load(args.draft_model)
        target_kwargs["draft_model"] = draft_model
        target_kwargs["num_draft_tokens"] = args.num_draft_tokens

    started = time.perf_counter()
    segments: list[str] = []
    last = None
    for response in mlx_lm.stream_generate(model, tokenizer, prompt, **target_kwargs):
        last = response
        segments.append(response.text)
    elapsed = time.perf_counter() - started

    text = "".join(segments)
    if last is None:
        raise RuntimeError("No generation output was produced.")

    return {
        "status": "ok",
        "mode": "run",
        "variant": args.variant,
        "target_model": _resolve_target_model(args),
        "draft_model": args.draft_model if _needs_speculative(args.variant) else None,
        "max_tokens": args.max_tokens,
        "max_kv_size": args.max_kv_size,
        "prompt_tokens": int(last.prompt_tokens),
        "prompt_tps": float(last.prompt_tps),
        "generation_tokens": int(last.generation_tokens),
        "generation_tps": float(last.generation_tps),
        "peak_memory_gb": float(last.peak_memory),
        "finish_reason": last.finish_reason,
        "elapsed_seconds": elapsed,
        "text_preview": text[:280],
    }


def _run_manual_cache_variant(args: argparse.Namespace, model, tokenizer, prompt: str) -> dict:
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(args.seed)
    prompt_ids = _encode_prompt(tokenizer, prompt)
    cache = make_prompt_cache(model)
    turbo_replaced = 0

    if _needs_turboquant(args.variant):
        cache, turbo_replaced = _make_turbo_cache(model, args.variant, args.seed)

    tri_runtime = None
    if _needs_triattention(args.variant):
        tri_runtime = _init_triattention_runtime(args, model, cache)

    sampler = make_sampler(
        temp=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        eos_ids.add(int(eos_id))

    mx.clear_cache()
    prefill_started = time.perf_counter()
    with mx.stream(generation_stream):
        logits = model(prompt_ids[None], cache)
    mx.eval(logits)
    prefill_elapsed = max(time.perf_counter() - prefill_started, 1e-6)
    prompt_tps = float(prompt_ids.size) / prefill_elapsed if prompt_ids.size > 0 else 0.0

    if tri_runtime is not None and prompt_ids.size > 0:
        _apply_triattention_step(
            tri_runtime,
            cache,
            is_prefill=True,
            current_position=int(prompt_ids.size) - 1,
        )
    _enforce_max_kv_size(cache, args.max_kv_size)

    token = int(sampler(logits[:, -1:])[0, 0].item())
    generated_tokens: list[int] = []
    finish_reason = "length"

    decode_started = time.perf_counter()
    while len(generated_tokens) < args.max_tokens:
        if token in eos_ids:
            finish_reason = "stop"
            break

        generated_tokens.append(token)
        with mx.stream(generation_stream):
            logits = model(mx.array([[token]]), cache)
        mx.eval(logits)

        if tri_runtime is not None:
            first_idx = tri_runtime.kv_indices[0]
            before_len = _cache_offset(cache[first_idx])
            _apply_triattention_step(
                tri_runtime,
                cache,
                is_prefill=False,
                current_position=int(prompt_ids.size + len(generated_tokens) - 1),
            )
            after_len = _cache_offset(cache[first_idx])
            if after_len < before_len:
                tri_runtime.compressions += 1

        _enforce_max_kv_size(cache, args.max_kv_size)
        token = int(sampler(logits[:, -1:])[0, 0].item())

    decode_elapsed = max(time.perf_counter() - decode_started, 1e-6)
    generated = len(generated_tokens)
    generation_tps = float(generated) / decode_elapsed if generated > 0 else 0.0
    text = tokenizer.decode(generated_tokens)

    tri_cache_len = None
    tri_compressions = 0
    if tri_runtime is not None:
        tri_cache_len = _cache_offset(cache[tri_runtime.kv_indices[0]])
        tri_compressions = tri_runtime.compressions

    return {
        "status": "ok",
        "mode": "run",
        "variant": args.variant,
        "target_model": _resolve_target_model(args),
        "draft_model": None,
        "max_tokens": args.max_tokens,
        "max_kv_size": args.max_kv_size,
        "prompt_tokens": int(prompt_ids.size),
        "prompt_tps": prompt_tps,
        "generation_tokens": generated,
        "generation_tps": generation_tps,
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9,
        "finish_reason": finish_reason,
        "elapsed_seconds": prefill_elapsed + decode_elapsed,
        "text_preview": text[:280],
        "turboquant_replaced_layers": turbo_replaced if _needs_turboquant(args.variant) else None,
        "triattention_cache_len": tri_cache_len,
        "triattention_compressions": tri_compressions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 4 26B A4B variant runner (prep + benchmark runtime)")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rotor-model", default=DEFAULT_ROTOR_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)

    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-kv-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)

    parser.add_argument("--num-draft-tokens", type=int, default=4)

    parser.add_argument("--triattention-stats-path", default="")
    parser.add_argument("--triattention-kv-budget", type=int, default=2048)
    parser.add_argument("--triattention-divide-length", type=int, default=8)
    parser.add_argument("--triattention-prefill-pin", action="store_true")
    parser.add_argument("--triattention-disable-trig", action="store_true")
    parser.add_argument("--triattention-disable-mlr", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0")
    if args.max_kv_size <= 0:
        parser.error("--max-kv-size must be > 0")
    if args.num_draft_tokens <= 0:
        parser.error("--num-draft-tokens must be > 0")
    if args.kv_bits <= 0:
        parser.error("--kv-bits must be > 0")
    if args.kv_group_size <= 0:
        parser.error("--kv-group-size must be > 0")
    if args.triattention_kv_budget <= 0:
        parser.error("--triattention-kv-budget must be > 0")
    if args.triattention_divide_length <= 0:
        parser.error("--triattention-divide-length must be > 0")

    if args.offline_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    target_model = _resolve_target_model(args)
    _ensure_model_family(target_model, "Target model")
    if _needs_speculative(args.variant) and not args.draft_model:
        parser.error("--draft-model is required for speculative variants")

    if args.preflight:
        payload = _preflight(args)
        _json_output(payload, args.output_json or None)
        return 0 if payload["status"] == "ok" else 1

    prompt = _load_prompt(args)
    payload = {
        "status": "ok",
        "mode": "dry-run",
        "variant": args.variant,
        "target_model": target_model,
        "draft_model": args.draft_model if _needs_speculative(args.variant) else None,
        "max_tokens": args.max_tokens,
        "max_kv_size": args.max_kv_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "offline_only": bool(args.offline_only),
        "prompt_preview": prompt[:220],
        "triattention": {
            "enabled": _needs_triattention(args.variant),
            "stats_path": args.triattention_stats_path or None,
            "kv_budget": args.triattention_kv_budget,
            "divide_length": args.triattention_divide_length,
            "prefill_pin": args.triattention_prefill_pin,
            "disable_trig": args.triattention_disable_trig,
            "disable_mlr": args.triattention_disable_mlr,
        },
        "turboquant": {
            "enabled": _needs_turboquant(args.variant),
            "strategy": args.variant if _needs_turboquant(args.variant) else None,
        },
        "speculative": {
            "enabled": _needs_speculative(args.variant),
            "num_draft_tokens": args.num_draft_tokens if _needs_speculative(args.variant) else None,
        },
    }

    if args.dry_run:
        _json_output(payload, args.output_json or None)
        return 0

    model, tokenizer = mlx_lm.load(target_model)
    if _needs_triattention(args.variant) or _needs_turboquant(args.variant):
        result = _run_manual_cache_variant(args, model, tokenizer, prompt)
    else:
        result = _run_stream_variant(args, model, tokenizer, prompt)

    _json_output(result, args.output_json or None)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
