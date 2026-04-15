#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
VENDOR_DIR="${ROOT_DIR}/vendor"
MODEL="Qwen/Qwen3.5-4B"
STRATEGY="tqv2_4bit_lean"
PROMPT="Napiš jednu stručnou větu o kvantizaci KV cache."
MAX_TOKENS=24
REPEATS=1
SEED=42
COMPARE_FP16=0
OUTPUT_JSON=""
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_qwen_turboquant_mlx.sh [options]

Runs Qwen3.5 with turboquant-mlx on local MLX runtime.
For Qwen3.5 hybrid attention, TurboQuant is applied only to KV-cache layers.

Options:
  --venv-path PATH        Virtualenv path (default: ./.venv)
  --vendor-dir PATH       Vendor directory (default: ./vendor)
  --model NAME            Model id/path (default: Qwen/Qwen3.5-4B)
  --strategy NAME         fp16 | tqv2_4bit_lean | tqv2_4bit_rot | tqv2_3bit_rot_qjl | tqv3_3.5bit_mixed | tqv3_3bit
  --prompt TEXT           Prompt text
  --max-tokens N          Max generated tokens (default: 24)
  --repeats N             Number of repeated runs (default: 1)
  --seed N                Base seed for cache init (default: 42)
  --compare-fp16          Also run fp16 baseline and print deltas
  --output-json PATH      Write machine-readable result JSON
  --dry-run               Print resolved config and exit
  -h, --help              Show this help text
USAGE
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  is_non_negative_int "$1" && [[ "$1" -gt 0 ]]
}

normalize_path() {
  local input_path="$1"
  if [[ "${input_path}" == /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s\n' "${ROOT_DIR}/${input_path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --venv-path" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --vendor-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --vendor-dir" >&2; exit 1; }
      VENDOR_DIR="$2"
      shift 2
      ;;
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --strategy)
      [[ $# -lt 2 ]] && { echo "Missing value for --strategy" >&2; exit 1; }
      STRATEGY="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt" >&2; exit 1; }
      PROMPT="$2"
      shift 2
      ;;
    --max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-tokens" >&2; exit 1; }
      MAX_TOKENS="$2"
      shift 2
      ;;
    --repeats)
      [[ $# -lt 2 ]] && { echo "Missing value for --repeats" >&2; exit 1; }
      REPEATS="$2"
      shift 2
      ;;
    --seed)
      [[ $# -lt 2 ]] && { echo "Missing value for --seed" >&2; exit 1; }
      SEED="$2"
      shift 2
      ;;
    --compare-fp16)
      COMPARE_FP16=1
      shift
      ;;
    --output-json)
      [[ $# -lt 2 ]] && { echo "Missing value for --output-json" >&2; exit 1; }
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${STRATEGY}" in
  fp16|tqv2_4bit_lean|tqv2_4bit_rot|tqv2_3bit_rot_qjl|tqv3_3.5bit_mixed|tqv3_3bit)
    ;;
  *)
    echo "Unsupported --strategy: ${STRATEGY}" >&2
    usage
    exit 1
    ;;
esac

if ! is_positive_int "${MAX_TOKENS}"; then
  echo "--max-tokens must be a positive integer: ${MAX_TOKENS}" >&2
  exit 1
fi

if ! is_positive_int "${REPEATS}"; then
  echo "--repeats must be a positive integer: ${REPEATS}" >&2
  exit 1
fi

if ! is_non_negative_int "${SEED}"; then
  echo "--seed must be a non-negative integer: ${SEED}" >&2
  exit 1
fi

if [[ -n "${OUTPUT_JSON}" ]]; then
  OUTPUT_JSON="$(normalize_path "${OUTPUT_JSON}")"
  mkdir -p "$(dirname "${OUTPUT_JSON}")"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}. Run scripts/setup_turboquant_mlx.sh first." >&2
  exit 1
fi

TURBOQUANT_DIR="${VENDOR_DIR}/turboquant-mlx"
if [[ ! -d "${TURBOQUANT_DIR}" ]]; then
  echo "turboquant-mlx source checkout not found at ${TURBOQUANT_DIR}." >&2
  echo "Run: scripts/setup_turboquant_mlx.sh" >&2
  exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Model: ${MODEL}"
  echo "Strategy: ${STRATEGY}"
  echo "Prompt: ${PROMPT}"
  echo "Max tokens: ${MAX_TOKENS}"
  echo "Repeats: ${REPEATS}"
  echo "Compare fp16: ${COMPARE_FP16}"
  echo "Venv: ${VENV_DIR}"
  echo "TurboQuant source: ${TURBOQUANT_DIR}"
  if [[ -n "${OUTPUT_JSON}" ]]; then
    echo "Output JSON: ${OUTPUT_JSON}"
  fi
  exit 0
fi

PYTHONPATH="${TURBOQUANT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
TQ_MODEL="${MODEL}" \
TQ_STRATEGY="${STRATEGY}" \
TQ_PROMPT="${PROMPT}" \
TQ_MAX_TOKENS="${MAX_TOKENS}" \
TQ_REPEATS="${REPEATS}" \
TQ_SEED="${SEED}" \
TQ_COMPARE_FP16="${COMPARE_FP16}" \
TQ_OUTPUT_JSON="${OUTPUT_JSON}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache

import turboquant.patch as tq_patch
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3


MODEL = os.environ["TQ_MODEL"]
STRATEGY = os.environ["TQ_STRATEGY"]
PROMPT = os.environ["TQ_PROMPT"]
MAX_TOKENS = int(os.environ["TQ_MAX_TOKENS"])
REPEATS = int(os.environ["TQ_REPEATS"])
SEED = int(os.environ["TQ_SEED"])
COMPARE_FP16 = os.environ["TQ_COMPARE_FP16"] == "1"
OUTPUT_JSON = os.environ.get("TQ_OUTPUT_JSON", "")


def maybe_write_output(payload: dict):
    if OUTPUT_JSON:
        out = Path(OUTPUT_JSON)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def fail(message: str):
    payload = {"status": "error", "error": message}
    maybe_write_output(payload)
    print(message, file=sys.stderr)
    raise SystemExit(1)


def get_layers(model):
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
    fail("Could not resolve model layers for cache construction.")


def make_cache_for_strategy(strategy: str, head_dim: int, seed: int):
    if strategy == "tqv2_4bit_lean":
        return TurboQuantKVCacheV2(
            head_dim=head_dim,
            bits=4,
            group_size=64,
            use_rotation=False,
            use_normalization=False,
            use_qjl=False,
            seed=seed,
        )
    if strategy == "tqv2_4bit_rot":
        return TurboQuantKVCacheV2(
            head_dim=head_dim,
            bits=4,
            group_size=64,
            use_rotation=True,
            use_normalization=True,
            use_qjl=False,
            seed=seed,
        )
    if strategy == "tqv2_3bit_rot_qjl":
        return TurboQuantKVCacheV2(
            head_dim=head_dim,
            bits=3,
            group_size=64,
            use_rotation=True,
            use_normalization=True,
            use_qjl=True,
            seed=seed,
        )
    if strategy == "tqv3_3.5bit_mixed":
        return TurboQuantKVCacheV3(
            head_dim=head_dim,
            bits=3,
            n_outlier=head_dim // 2,
            outlier_bits=4,
            use_qjl=False,
            seed=seed,
        )
    if strategy == "tqv3_3bit":
        return TurboQuantKVCacheV3(
            head_dim=head_dim,
            bits=3,
            use_qjl=False,
            seed=seed,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def build_prompt_cache(model, layers, strategy: str, seed: int):
    cache = make_prompt_cache(model)
    replaced_indices = []

    if strategy == "fp16":
        return cache, replaced_indices

    for idx, layer_cache in enumerate(cache):
        if not isinstance(layer_cache, KVCache):
            continue
        if idx >= len(layers):
            fail(f"Cache index {idx} is out of range for model layers.")

        layer = layers[idx]
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            fail(
                f"Layer {idx} uses KVCache but has no self_attn attribute. "
                "This model path is not currently supported by turboquant-mlx."
            )

        head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            fail(f"Layer {idx} self_attn.head_dim is missing; cannot construct TurboQuant cache.")

        cache[idx] = make_cache_for_strategy(strategy, int(head_dim), seed + idx)
        replaced_indices.append(idx)

    if not replaced_indices:
        fail("No KVCache layers found to replace for TurboQuant strategy.")

    return cache, replaced_indices


def encode_prompt(tokenizer, prompt: str):
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt
    return mx.array(tokenizer.encode(formatted))


def sum_bytes(values):
    return int(sum(values))


def run_once(model, tokenizer, layers, strategy: str, run_seed: int):
    cache, replaced_indices = build_prompt_cache(model, layers, strategy, run_seed)
    input_ids = encode_prompt(tokenizer, PROMPT)

    tokens = []
    started = time.perf_counter()
    for token, _ in generate_step(
        prompt=input_ids,
        model=model,
        max_tokens=MAX_TOKENS,
        prompt_cache=cache,
    ):
        token_id = token.item() if hasattr(token, "item") else int(token)
        if token_id == tokenizer.eos_token_id:
            break
        tokens.append(token_id)
    elapsed = time.perf_counter() - started

    total_cache_bytes = sum_bytes(
        c.nbytes for c in cache if hasattr(c, "nbytes")
    )
    kv_layer_bytes = sum_bytes(
        cache[i].nbytes for i in replaced_indices if hasattr(cache[i], "nbytes")
    )
    kv_layer_fp16_equivalent = sum_bytes(
        cache[i].nbytes_equivalent_fp16
        for i in replaced_indices
        if hasattr(cache[i], "nbytes_equivalent_fp16")
    )

    return {
        "strategy": strategy,
        "generated_tokens": len(tokens),
        "elapsed_seconds": elapsed,
        "tokens_per_second": (len(tokens) / elapsed) if elapsed > 0 else 0.0,
        "response_preview": tokenizer.decode(tokens)[:240],
        "kv_layer_indices": replaced_indices,
        "kv_layer_count": len(replaced_indices),
        "total_cache_bytes": total_cache_bytes,
        "kv_layer_cache_bytes": kv_layer_bytes,
        "kv_layer_fp16_equivalent_bytes": kv_layer_fp16_equivalent,
        "kv_layer_compression_vs_fp16": (
            (kv_layer_fp16_equivalent / kv_layer_bytes) if kv_layer_bytes > 0 else 1.0
        ),
    }


def aggregate_runs(strategy: str, runs):
    tok_s = [r["tokens_per_second"] for r in runs]
    elapsed = [r["elapsed_seconds"] for r in runs]
    tokens = [r["generated_tokens"] for r in runs]
    last = runs[-1]
    return {
        "strategy": strategy,
        "runs": len(runs),
        "tokens_per_second_mean": sum(tok_s) / len(tok_s),
        "tokens_per_second_min": min(tok_s),
        "tokens_per_second_max": max(tok_s),
        "elapsed_seconds_mean": sum(elapsed) / len(elapsed),
        "generated_tokens_mean": sum(tokens) / len(tokens),
        "total_cache_bytes": last["total_cache_bytes"],
        "kv_layer_cache_bytes": last["kv_layer_cache_bytes"],
        "kv_layer_fp16_equivalent_bytes": last["kv_layer_fp16_equivalent_bytes"],
        "kv_layer_compression_vs_fp16": last["kv_layer_compression_vs_fp16"],
        "kv_layer_indices": last["kv_layer_indices"],
        "kv_layer_count": last["kv_layer_count"],
        "response_preview": runs[0]["response_preview"],
        "per_run": runs,
    }


def run_strategy(model, tokenizer, layers, strategy: str):
    rows = []
    for repeat_idx in range(REPEATS):
        rows.append(run_once(model, tokenizer, layers, strategy, SEED + repeat_idx))
    return aggregate_runs(strategy, rows)


def main():
    tq_patch.apply()
    model, tokenizer = mlx_lm.load(MODEL)
    layers = get_layers(model)

    primary = run_strategy(model, tokenizer, layers, STRATEGY)

    baseline = None
    comparison = None
    if COMPARE_FP16 and STRATEGY != "fp16":
        baseline = run_strategy(model, tokenizer, layers, "fp16")
        bps = baseline["tokens_per_second_mean"]
        pps = primary["tokens_per_second_mean"]
        bcache = baseline["total_cache_bytes"]
        pcache = primary["total_cache_bytes"]
        comparison = {
            "tokens_per_second_ratio_vs_fp16": (pps / bps) if bps > 0 else 0.0,
            "tokens_per_second_delta_vs_fp16": pps - bps,
            "total_cache_bytes_delta_vs_fp16": pcache - bcache,
            "total_cache_bytes_reduction_pct_vs_fp16": (
                ((bcache - pcache) / bcache * 100.0) if bcache > 0 else 0.0
            ),
        }

    payload = {
        "status": "ok",
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "repeats": REPEATS,
        "primary_result": primary,
        "fp16_baseline": baseline,
        "comparison_vs_fp16": comparison,
        "notes": (
            "For Qwen3.5, TurboQuant is applied only to KVCache layers; "
            "linear-attention ArraysCache layers remain unchanged."
        ),
    }

    maybe_write_output(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if OUTPUT_JSON:
        print(f"Wrote {OUTPUT_JSON}")


try:
    main()
except Exception as exc:  # pragma: no cover - integration script
    fail(f"{type(exc).__name__}: {exc}")
PY
