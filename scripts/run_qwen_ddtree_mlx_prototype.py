#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-4B-DFlash"
DEFAULT_PROMPT = "Write one short sentence confirming this DDTree-style MLX prototype run completed."
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "ddtree-mlx-prototype" / "runs"


@dataclass(frozen=True)
class CandidatePath:
    tokens: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class VerificationResult:
    accepted: int
    fallback_token: int
    target_calls: int
    dropped_tokens: int


@dataclass
class RoundTrace:
    round_index: int
    depth: int
    tree_budget: int
    context_tokens: int
    context_tokens_dropped: int
    candidate_count: int
    verify_calls: int
    accepted_tokens: int
    committed_tokens: list[int]
    fallback_token: int | None
    fallback_event: bool
    fallback_reason: str
    best_candidate_tokens: list[int]
    best_candidate_score: float | None
    draft_error: str | None
    elapsed_sec: float


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
        description="Experimental DDTree-style speculative decoding prototype on MLX for Qwen3.5."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_TARGET_MODEL, help="Target model id/path")
    parser.add_argument("--draft-model", type=str, default=DEFAULT_DRAFT_MODEL, help="DFlash draft model id/path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--prompt-file", type=str, default="", help="Read prompt text from file")
    parser.add_argument(
        "--tree-budget",
        type=_positive_int,
        default=128,
        help="Max candidate paths retained per round (tuned default for higher acceptance)",
    )
    parser.add_argument(
        "--depth",
        type=_positive_int,
        default=1,
        help="Tree expansion depth per round (tuned default for better throughput)",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=64, help="Generation token limit")
    parser.add_argument("--max-kv-size", type=_positive_int, default=1024, help="Context window cap used by cache")
    parser.add_argument("--temperature", type=_non_negative_float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to MLX RNG")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Artifacts root directory")
    parser.add_argument("--run-name", type=str, default="", help="Optional run directory name")
    parser.add_argument("--verbose-rounds", action="store_true", help="Print per-round traces to stdout")
    parser.add_argument("--debug", action="store_true", help="Print traceback on failure")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.tree_budget > 128:
        raise ValueError("--tree-budget must be <= 128 for this prototype path.")
    if args.depth > 16:
        raise ValueError("--depth must be <= 16 for this prototype path.")
    if args.max_new_tokens > 4096:
        raise ValueError("--max-new-tokens must be <= 4096 for this prototype path.")
    if args.max_kv_size < 16:
        raise ValueError("--max-kv-size must be >= 16.")


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
    name = run_name.strip() or f"qwen35-ddtree-mlx-{stamp}"
    run_dir = root / name
    if run_dir.exists():
        run_dir = root / f"{name}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_runtime():
    try:
        import mlx.core as mx  # noqa: WPS433
        from mlx_lm.generate import generation_stream  # noqa: WPS433
        from mlx_lm.models.cache import make_prompt_cache  # noqa: WPS433
        from mlx_lm.sample_utils import make_sampler  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - runtime path
        raise RuntimeError(
            "Failed to import MLX runtime dependencies. Ensure .venv has mlx and mlx-lm installed."
        ) from exc

    try:
        from dflash.model_mlx import _patch_model, load, load_draft  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - runtime path
        raise RuntimeError(
            "Failed to import dflash.model_mlx. Install DFlash in this repo env (scripts/setup_env.sh)."
        ) from exc

    return {
        "mx": mx,
        "generation_stream": generation_stream,
        "make_prompt_cache": make_prompt_cache,
        "make_sampler": make_sampler,
        "patch_model": _patch_model,
        "load_target": load,
        "load_draft": load_draft,
    }


def _reset_peak_memory(mx_mod: Any) -> None:
    if hasattr(mx_mod, "reset_peak_memory"):
        try:
            mx_mod.reset_peak_memory()
            return
        except Exception:
            pass

    metal = getattr(mx_mod, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        try:
            metal.reset_peak_memory()
        except Exception:
            pass


def _get_peak_memory_bytes(mx_mod: Any) -> float:
    if hasattr(mx_mod, "get_peak_memory"):
        try:
            return float(mx_mod.get_peak_memory())
        except Exception:
            pass

    metal = getattr(mx_mod, "metal", None)
    if metal is not None and hasattr(metal, "get_peak_memory"):
        try:
            return float(metal.get_peak_memory())
        except Exception:
            pass

    return float("nan")


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    bos_token = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos_token is None or not prompt.startswith(str(bos_token))

    try:
        encoded = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TypeError:
        encoded = tokenizer.encode(prompt)

    return [int(token) for token in encoded]


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""

    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _collect_eos_token_ids(tokenizer: Any) -> set[int]:
    eos_ids: set[int] = set()

    raw_ids = getattr(tokenizer, "eos_token_ids", None)
    if raw_ids is not None:
        try:
            for token_id in raw_ids:
                eos_ids.add(int(token_id))
        except TypeError:
            eos_ids.add(int(raw_ids))

    raw_id = getattr(tokenizer, "eos_token_id", None)
    if raw_id is not None:
        eos_ids.add(int(raw_id))

    return eos_ids


def _sample_token(sampler: Any, logits: Any) -> int:
    sampled = sampler(logits[:, -1:])
    token = sampled[0, 0]
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def _forward_target(
    mx_mod: Any,
    generation_stream: Any,
    make_prompt_cache: Any,
    target_model: Any,
    context_tokens: list[int],
    max_kv_size: int,
) -> tuple[Any, Any, int]:
    if not context_tokens:
        raise RuntimeError("Target forward received an empty context.")

    if len(context_tokens) > max_kv_size:
        effective_tokens = context_tokens[-max_kv_size:]
        dropped = len(context_tokens) - max_kv_size
    else:
        effective_tokens = context_tokens
        dropped = 0

    inputs = mx_mod.array([effective_tokens], dtype=mx_mod.int32)
    cache = make_prompt_cache(target_model, max_kv_size=max_kv_size)

    with mx_mod.stream(generation_stream):
        logits = target_model(inputs, cache)
        hidden = mx_mod.concatenate(target_model._hidden_states, axis=-1)

    mx_mod.eval(logits, hidden)
    return logits, hidden, dropped


def _forward_draft(
    mx_mod: Any,
    generation_stream: Any,
    make_prompt_cache: Any,
    draft_model: Any,
    hidden_context: Any,
    root_token: int,
    depth: int,
    max_kv_size: int,
    mask_token_id: int,
) -> Any:
    block_tokens = [int(root_token)] + [int(mask_token_id)] * depth
    block_inputs = mx_mod.array([block_tokens], dtype=mx_mod.int32)
    cache = make_prompt_cache(draft_model, max_kv_size=max_kv_size)

    with mx_mod.stream(generation_stream):
        draft_logits = draft_model(block_inputs, hidden_context, cache)

    mx_mod.eval(draft_logits)
    return draft_logits


def _top_tokens_with_logprobs(logits_row: np.ndarray, k: int) -> list[tuple[int, float]]:
    if logits_row.ndim != 1:
        raise ValueError("Expected 1D logits row.")

    k = min(k, int(logits_row.shape[0]))
    if k <= 0:
        return []

    if k == logits_row.shape[0]:
        indices = np.argsort(logits_row)[::-1]
    else:
        partial = np.argpartition(logits_row, -k)[-k:]
        indices = partial[np.argsort(logits_row[partial])[::-1]]

    max_logit = float(np.max(logits_row))
    log_z = max_logit + math.log(float(np.exp(logits_row - max_logit).sum()))
    return [(int(index), float(logits_row[index] - log_z)) for index in indices]


def _build_tree_candidates(step_logits: np.ndarray, tree_budget: int) -> list[CandidatePath]:
    if step_logits.ndim != 2:
        raise ValueError("Expected draft logits with shape [depth, vocab].")
    if step_logits.shape[0] == 0:
        return []

    # Keep first-step coverage aligned with tree_budget; capping at 8 artificially
    # suppresses acceptance when the target token is outside the top-8 draft logits.
    branch_width = max(1, tree_budget)
    top_by_depth = [
        _top_tokens_with_logprobs(step_logits[depth_idx], branch_width)
        for depth_idx in range(step_logits.shape[0])
    ]

    beams: list[tuple[tuple[int, ...], float]] = [(tuple(), 0.0)]
    scored_paths: dict[tuple[int, ...], float] = {}

    for depth_options in top_by_depth:
        if not depth_options:
            break

        expanded: list[tuple[tuple[int, ...], float]] = []
        for path_tokens, path_score in beams:
            for token_id, token_logp in depth_options:
                new_path = path_tokens + (token_id,)
                new_score = path_score + token_logp
                expanded.append((new_path, new_score))

                prev = scored_paths.get(new_path)
                if prev is None or new_score > prev:
                    scored_paths[new_path] = new_score

        expanded.sort(key=lambda item: item[1], reverse=True)
        beams = expanded[:tree_budget]
        if not beams:
            break

    ranked_paths = sorted(scored_paths.items(), key=lambda item: item[1], reverse=True)[:tree_budget]
    return [CandidatePath(tokens=path, score=score) for path, score in ranked_paths]


def _verify_candidate(
    mx_mod: Any,
    generation_stream: Any,
    make_prompt_cache: Any,
    target_model: Any,
    sampler: Any,
    base_context_tokens: list[int],
    candidate_tokens: tuple[int, ...],
    first_target_token: int,
    max_kv_size: int,
) -> VerificationResult:
    if not candidate_tokens:
        return VerificationResult(accepted=0, fallback_token=int(first_target_token), target_calls=0, dropped_tokens=0)

    if int(candidate_tokens[0]) != int(first_target_token):
        return VerificationResult(accepted=0, fallback_token=int(first_target_token), target_calls=0, dropped_tokens=0)

    accepted = 1
    working_context = list(base_context_tokens)
    working_context.append(int(candidate_tokens[0]))
    target_calls = 0
    dropped_tokens = 0

    for token_id in candidate_tokens[1:]:
        logits, _, dropped = _forward_target(
            mx_mod=mx_mod,
            generation_stream=generation_stream,
            make_prompt_cache=make_prompt_cache,
            target_model=target_model,
            context_tokens=working_context,
            max_kv_size=max_kv_size,
        )
        target_calls += 1
        dropped_tokens += dropped
        sampled = _sample_token(sampler, logits)

        if int(token_id) != int(sampled):
            return VerificationResult(
                accepted=accepted,
                fallback_token=int(sampled),
                target_calls=target_calls,
                dropped_tokens=dropped_tokens,
            )

        accepted += 1
        working_context.append(int(token_id))

    logits, _, dropped = _forward_target(
        mx_mod=mx_mod,
        generation_stream=generation_stream,
        make_prompt_cache=make_prompt_cache,
        target_model=target_model,
        context_tokens=working_context,
        max_kv_size=max_kv_size,
    )
    target_calls += 1
    dropped_tokens += dropped
    fallback = _sample_token(sampler, logits)

    return VerificationResult(
        accepted=accepted,
        fallback_token=int(fallback),
        target_calls=target_calls,
        dropped_tokens=dropped_tokens,
    )


def _summary_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "p50": 0,
            "p95": 0,
        }

    sorted_values = sorted(values)
    p50 = sorted_values[len(sorted_values) // 2]
    p95 = sorted_values[min(len(sorted_values) - 1, int(math.ceil(len(sorted_values) * 0.95)) - 1)]
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "min": int(sorted_values[0]),
        "max": int(sorted_values[-1]),
        "p50": int(p50),
        "p95": int(p95),
    }


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    prompt = _resolve_prompt(args)
    run_dir = _prepare_run_dir(args.artifacts_dir, args.run_name)

    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt)

    config_path = run_dir / "config.json"
    config_data = vars(args).copy()
    config_data["prompt_chars"] = len(prompt)
    config_path.write_text(json.dumps(config_data, indent=2))

    runtime = _load_runtime()
    mx_mod = runtime["mx"]
    generation_stream = runtime["generation_stream"]
    make_prompt_cache = runtime["make_prompt_cache"]
    make_sampler = runtime["make_sampler"]
    patch_model = runtime["patch_model"]
    load_target = runtime["load_target"]
    load_draft = runtime["load_draft"]

    print(f"Loading target model: {args.model}")
    target_model, tokenizer = load_target(args.model)

    print(f"Loading draft model: {args.draft_model}")
    draft_model = load_draft(args.draft_model)

    try:
        patch_model(target_model, draft_model.config.target_layer_ids)
    except Exception as exc:
        raise RuntimeError(f"Failed to patch target model for hidden-state capture: {exc}") from exc

    try:
        draft_model.bind(target_model)
    except Exception as exc:
        raise RuntimeError(f"Failed to bind draft model to target embeddings/head: {exc}") from exc

    sampler = make_sampler(temp=args.temperature)
    mask_token_id = int(getattr(draft_model.config, "mask_token_id", 0))

    if args.seed >= 0:
        try:
            mx_mod.random.seed(args.seed)
        except Exception:
            pass

    prompt_tokens = _encode_prompt(tokenizer, prompt)
    if not prompt_tokens:
        raise RuntimeError("Prompt encoding returned zero tokens.")

    eos_token_ids = _collect_eos_token_ids(tokenizer)

    _reset_peak_memory(mx_mod)

    context_tokens = list(prompt_tokens)
    generated_tokens: list[int] = []
    acceptance_lengths: list[int] = []
    candidate_counts: list[int] = []
    fallback_events = 0
    fallback_reasons: dict[str, int] = {}
    round_traces: list[RoundTrace] = []
    target_forward_calls = 0
    draft_forward_calls = 0
    total_context_tokens_dropped = 0
    stop_reason = "length"

    decode_start = time.perf_counter()

    while len(generated_tokens) < args.max_new_tokens:
        round_index = len(round_traces) + 1
        round_start = time.perf_counter()
        remaining = args.max_new_tokens - len(generated_tokens)
        step_depth = min(args.depth, remaining)

        logits, hidden_context, dropped = _forward_target(
            mx_mod=mx_mod,
            generation_stream=generation_stream,
            make_prompt_cache=make_prompt_cache,
            target_model=target_model,
            context_tokens=context_tokens,
            max_kv_size=args.max_kv_size,
        )
        target_forward_calls += 1
        total_context_tokens_dropped += dropped

        first_target_token = _sample_token(sampler, logits)
        root_token = int(context_tokens[-1])

        draft_error: str | None = None
        candidates: list[CandidatePath] = []
        try:
            draft_logits = _forward_draft(
                mx_mod=mx_mod,
                generation_stream=generation_stream,
                make_prompt_cache=make_prompt_cache,
                draft_model=draft_model,
                hidden_context=hidden_context,
                root_token=root_token,
                depth=step_depth,
                max_kv_size=args.max_kv_size,
                mask_token_id=mask_token_id,
            )
            draft_forward_calls += 1
            depth_logits_mx = draft_logits[:, 1 : step_depth + 1, :].astype(mx_mod.float32)
            mx_mod.eval(depth_logits_mx)
            depth_logits = np.asarray(depth_logits_mx)[0]
            candidates = _build_tree_candidates(depth_logits, args.tree_budget)
        except Exception as exc:
            draft_error = f"{type(exc).__name__}: {exc}"

        candidate_counts.append(len(candidates))

        verify_calls_round = 0
        dropped_this_round = dropped
        accepted = 0
        fallback_token = int(first_target_token)
        fallback_reason = "tree_miss"
        best_candidate_tokens: list[int] = []
        best_candidate_score: float | None = None

        if candidates:
            best_key = (-1, float("-inf"))
            best_candidate: CandidatePath | None = None
            best_result: VerificationResult | None = None

            for candidate in candidates:
                verification = _verify_candidate(
                    mx_mod=mx_mod,
                    generation_stream=generation_stream,
                    make_prompt_cache=make_prompt_cache,
                    target_model=target_model,
                    sampler=sampler,
                    base_context_tokens=context_tokens,
                    candidate_tokens=candidate.tokens,
                    first_target_token=first_target_token,
                    max_kv_size=args.max_kv_size,
                )
                verify_calls_round += verification.target_calls
                target_forward_calls += verification.target_calls
                dropped_this_round += verification.dropped_tokens
                total_context_tokens_dropped += verification.dropped_tokens

                key = (verification.accepted, candidate.score)
                if key > best_key:
                    best_key = key
                    best_candidate = candidate
                    best_result = verification

            if best_candidate is not None and best_result is not None:
                accepted = int(best_result.accepted)
                fallback_token = int(best_result.fallback_token)
                best_candidate_tokens = [int(token) for token in best_candidate.tokens]
                best_candidate_score = float(best_candidate.score)
                fallback_reason = "tree_miss" if accepted == 0 else "verified_branch"
            else:
                fallback_reason = "candidate_selection_failed"
        else:
            fallback_reason = "draft_error" if draft_error else "no_candidates"

        planned_commit = best_candidate_tokens[:accepted] + [fallback_token]
        commit_tokens = planned_commit[:remaining]
        if not commit_tokens:
            stop_reason = "length"
            break

        eos_hit = False
        if eos_token_ids:
            for idx, token in enumerate(commit_tokens):
                if token in eos_token_ids:
                    commit_tokens = commit_tokens[: idx + 1]
                    eos_hit = True
                    break

        accepted_committed = min(accepted, len(commit_tokens))
        fallback_committed = len(commit_tokens) > accepted_committed
        fallback_event = accepted_committed == 0 and fallback_committed
        if fallback_event:
            fallback_events += 1

        acceptance_lengths.append(accepted_committed)
        context_tokens.extend(commit_tokens)
        generated_tokens.extend(commit_tokens)

        fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1

        round_trace = RoundTrace(
            round_index=round_index,
            depth=step_depth,
            tree_budget=args.tree_budget,
            context_tokens=len(context_tokens),
            context_tokens_dropped=dropped_this_round,
            candidate_count=len(candidates),
            verify_calls=verify_calls_round,
            accepted_tokens=accepted_committed,
            committed_tokens=[int(token) for token in commit_tokens],
            fallback_token=int(fallback_token) if fallback_committed else None,
            fallback_event=fallback_event,
            fallback_reason=fallback_reason,
            best_candidate_tokens=best_candidate_tokens,
            best_candidate_score=best_candidate_score,
            draft_error=draft_error,
            elapsed_sec=time.perf_counter() - round_start,
        )
        round_traces.append(round_trace)

        if args.verbose_rounds:
            print(
                "round="
                f"{round_index} accepted={accepted_committed} committed={len(commit_tokens)} "
                f"candidates={len(candidates)} fallback_reason={fallback_reason}"
            )

        if eos_hit:
            stop_reason = "eos"
            break

        if len(generated_tokens) >= args.max_new_tokens:
            stop_reason = "length"
            break

        if len(generated_tokens) % 128 == 0:
            try:
                mx_mod.clear_cache()
            except Exception:
                pass

    decode_elapsed = max(time.perf_counter() - decode_start, 1e-9)

    generated_text = _decode_tokens(tokenizer, generated_tokens)
    peak_memory_bytes = _get_peak_memory_bytes(mx_mod)
    peak_memory_gb = None if math.isnan(peak_memory_bytes) else peak_memory_bytes / 1e9

    traces_path = run_dir / "round_traces.jsonl"
    with traces_path.open("w") as handle:
        for trace in round_traces:
            handle.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    generation_path = run_dir / "generation.txt"
    generation_path.write_text(generated_text)

    result = {
        "model": args.model,
        "draft_model": args.draft_model,
        "prompt_tokens": len(prompt_tokens),
        "generation_tokens": len(generated_tokens),
        "generation_tps": len(generated_tokens) / decode_elapsed,
        "rounds": len(round_traces),
        "tree_budget": args.tree_budget,
        "depth": args.depth,
        "max_new_tokens": args.max_new_tokens,
        "max_kv_size": args.max_kv_size,
        "temperature": args.temperature,
        "accepted_length_stats": _summary_stats(acceptance_lengths),
        "candidate_count_stats": _summary_stats(candidate_counts),
        "fallback_events": fallback_events,
        "fallback_rate": fallback_events / max(len(round_traces), 1),
        "fallback_reasons": fallback_reasons,
        "target_forward_calls": target_forward_calls,
        "draft_forward_calls": draft_forward_calls,
        "context_tokens_dropped_total": total_context_tokens_dropped,
        "stop_reason": stop_reason,
        "peak_memory_gb": peak_memory_gb,
        "run_dir": str(run_dir),
        "artifacts": {
            "config": str(config_path),
            "prompt": str(prompt_path),
            "generation": str(generation_path),
            "round_traces": str(traces_path),
        },
    }

    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(f"Prototype artifacts: {run_dir}")
    print(f"Generated tokens: {result['generation_tokens']}")
    print(f"Generation throughput: {result['generation_tps']:.2f} tok/s")
    print(f"Accepted length mean: {result['accepted_length_stats']['mean']:.3f}")
    print(f"Fallback events: {fallback_events}")
    if peak_memory_gb is not None:
        print(f"Peak memory: {peak_memory_gb:.3f} GB")
    print(f"Result JSON: {result_path}")

    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:  # pragma: no cover - runtime path
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
