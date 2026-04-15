#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_SCRIPT="${ROOT_DIR}/scripts/benchmark_supergemma_mlx.sh"

MODEL="Qwen/Qwen3.5-4B"
PROMPT_FILE="${ROOT_DIR}/artifacts/benchmarks/qwen-kv-eval-prompt.txt"
MAX_TOKENS=256
REPEATS=2
MAX_KV_VALUES=(1024 2048 4096)
KV_BITS=4
KV_GROUP_SIZE=64
QUANTIZED_KV_START=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_qwen_kv_eval.sh [options]

Runs repeatable Qwen3.5 KV-cache evaluation on MLX using benchmark_supergemma_mlx.sh.

Defaults:
  model: Qwen/Qwen3.5-4B
  max-tokens: 256
  repeats: 2
  max-kv-size sweep: 1024, 2048, 4096
  quantized KV config: --kv-bits 4 --kv-group-size 64 --quantized-kv-start 0

Options:
  --model NAME                Override model id/path
  --prompt-file PATH          Prompt file path (default: artifacts/benchmarks/qwen-kv-eval-prompt.txt)
  --max-tokens N              Generation token limit per run (default: 256)
  --repeats N                 Repeats per max-kv-size (default: 2)
  --max-kv-size N             Add KV size to sweep (repeatable; overrides default list if used)
  --kv-bits N                 KV quantization bits for quantized pass (default: 4)
  --kv-group-size N           KV quantization group size (default: 64)
  --quantized-kv-start N      KV quantization start step (default: 0)
  --dry-run                   Plan runs only (delegates dry-run to benchmark script)
  -h, --help                  Show this help text
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

USER_SET_KV_LIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --prompt-file)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt-file" >&2; exit 1; }
      PROMPT_FILE="$2"
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
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      if [[ "${USER_SET_KV_LIST}" -eq 0 ]]; then
        MAX_KV_VALUES=()
        USER_SET_KV_LIST=1
      fi
      MAX_KV_VALUES+=("$2")
      shift 2
      ;;
    --kv-bits)
      [[ $# -lt 2 ]] && { echo "Missing value for --kv-bits" >&2; exit 1; }
      KV_BITS="$2"
      shift 2
      ;;
    --kv-group-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --kv-group-size" >&2; exit 1; }
      KV_GROUP_SIZE="$2"
      shift 2
      ;;
    --quantized-kv-start)
      [[ $# -lt 2 ]] && { echo "Missing value for --quantized-kv-start" >&2; exit 1; }
      QUANTIZED_KV_START="$2"
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

if [[ ! -x "${BENCH_SCRIPT}" ]]; then
  echo "Missing executable benchmark script: ${BENCH_SCRIPT}" >&2
  exit 1
fi

if ! is_positive_int "${MAX_TOKENS}"; then
  echo "--max-tokens must be a positive integer: ${MAX_TOKENS}" >&2
  exit 1
fi

if ! is_positive_int "${REPEATS}"; then
  echo "--repeats must be a positive integer: ${REPEATS}" >&2
  exit 1
fi

if ! is_positive_int "${KV_BITS}"; then
  echo "--kv-bits must be a positive integer: ${KV_BITS}" >&2
  exit 1
fi

if ! is_positive_int "${KV_GROUP_SIZE}"; then
  echo "--kv-group-size must be a positive integer: ${KV_GROUP_SIZE}" >&2
  exit 1
fi

if ! is_non_negative_int "${QUANTIZED_KV_START}"; then
  echo "--quantized-kv-start must be a non-negative integer: ${QUANTIZED_KV_START}" >&2
  exit 1
fi

if [[ "${#MAX_KV_VALUES[@]}" -eq 0 ]]; then
  echo "At least one --max-kv-size value is required." >&2
  exit 1
fi

for kv in "${MAX_KV_VALUES[@]}"; do
  if ! is_positive_int "${kv}"; then
    echo "--max-kv-size values must be positive integers: ${kv}" >&2
    exit 1
  fi
done

PROMPT_FILE="$(normalize_path "${PROMPT_FILE}")"
mkdir -p "$(dirname "${PROMPT_FILE}")"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  PROMPT_FILE_FOR_GEN="${PROMPT_FILE}" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["PROMPT_FILE_FOR_GEN"])
path.parent.mkdir(parents=True, exist_ok=True)
chunk = (
    "This is a synthetic long-context benchmark prompt for Qwen 3.5 on MLX. "
    "It repeats structured technical text about caching, throughput, and latency to fill context windows while staying deterministic. "
    "Please continue reading carefully and preserve coherence across repeated sections. "
)
text = "\n".join([f"Section {i:03d}: {chunk}" for i in range(1, 181)])
path.write_text(text)
print(path)
PY
fi

OUT_DIR="${ROOT_DIR}/artifacts/benchmarks/qwen-kv-eval-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT_DIR}"
RUN_DIRS_FILE="${OUT_DIR}/run_dirs.env"

build_base_cmd() {
  local -n _result=$1
  _result=("${BENCH_SCRIPT}" --model "${MODEL}" --prompt-file "${PROMPT_FILE}" --max-tokens "${MAX_TOKENS}" --repeats "${REPEATS}")
  for kv in "${MAX_KV_VALUES[@]}"; do
    _result+=(--max-kv-size "${kv}")
  done
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    _result+=(--dry-run)
  fi
}

run_case() {
  local label="$1"
  shift
  local -a extra_args=("$@")
  local -a cmd=()
  build_base_cmd cmd
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=(-- "${extra_args[@]}")
  fi

  local log_file="${OUT_DIR}/${label}.log"
  {
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } | tee "${log_file}"

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"

  local run_dir
  run_dir="$(grep -E '^Benchmark artifacts directory: ' "${log_file}" | tail -n 1 | sed 's/^Benchmark artifacts directory: //')"
  if [[ -z "${run_dir}" ]]; then
    echo "Could not parse benchmark run directory for case: ${label}" >&2
    exit 1
  fi

  printf '%s=%s\n' "${label}" "${run_dir}" >> "${RUN_DIRS_FILE}"
}

run_case "fp16"
run_case "kv4" --kv-bits "${KV_BITS}" --kv-group-size "${KV_GROUP_SIZE}" --quantized-kv-start "${QUANTIZED_KV_START}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run complete."
  echo "Run logs: ${OUT_DIR}"
  echo "Run dirs file: ${RUN_DIRS_FILE}"
  exit 0
fi

FP16_RUN_DIR="$(grep '^fp16=' "${RUN_DIRS_FILE}" | tail -n 1 | cut -d'=' -f2-)"
KV4_RUN_DIR="$(grep '^kv4=' "${RUN_DIRS_FILE}" | tail -n 1 | cut -d'=' -f2-)"

SUMMARY_TSV="${OUT_DIR}/summary.tsv"
SUMMARY_JSON="${OUT_DIR}/summary.json"

python3 - <<PY
import csv
import json
import statistics
from pathlib import Path

runs = {
    "fp16": Path("${FP16_RUN_DIR}") / "results.csv",
    "kv4": Path("${KV4_RUN_DIR}") / "results.csv",
}

rows = []
for mode, path in runs.items():
    if not path.exists():
        raise SystemExit(f"Missing results file: {path}")
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            rows.append({
                "mode": mode,
                "max_kv_size": int(row["max_kv_size"]),
                "generation_tps": float(row["generation_tps"]),
                "peak_memory_gb": float(row["peak_memory_gb"]),
                "prompt_tokens": int(row["prompt_tokens"]),
            })

grouped = {}
for row in rows:
    grouped.setdefault((row["mode"], row["max_kv_size"]), []).append(row)

summary_rows = []
for (mode, max_kv_size), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
    gen = [x["generation_tps"] for x in items]
    mem = [x["peak_memory_gb"] for x in items]
    prompts = sorted({x["prompt_tokens"] for x in items})
    summary_rows.append({
        "mode": mode,
        "max_kv_size": max_kv_size,
        "n": len(items),
        "generation_tps_mean": statistics.mean(gen),
        "generation_tps_min": min(gen),
        "generation_tps_max": max(gen),
        "peak_memory_gb_mean": statistics.mean(mem),
        "peak_memory_gb_min": min(mem),
        "peak_memory_gb_max": max(mem),
        "prompt_tokens": prompts,
    })

with Path("${SUMMARY_TSV}").open("w") as f:
    f.write("mode\\tmax_kv_size\\tn\\tgeneration_tps_mean\\tgeneration_tps_min\\tgeneration_tps_max\\tpeak_memory_gb_mean\\tpeak_memory_gb_min\\tpeak_memory_gb_max\\tprompt_tokens\\n")
    for r in summary_rows:
        f.write(
            f"{r['mode']}\\t{r['max_kv_size']}\\t{r['n']}\\t{r['generation_tps_mean']:.3f}\\t{r['generation_tps_min']:.3f}\\t{r['generation_tps_max']:.3f}\\t"
            f"{r['peak_memory_gb_mean']:.3f}\\t{r['peak_memory_gb_min']:.3f}\\t{r['peak_memory_gb_max']:.3f}\\t"
            f"{','.join(map(str, r['prompt_tokens']))}\\n"
        )

Path("${SUMMARY_JSON}").write_text(json.dumps(summary_rows, indent=2))
print(f"Wrote ${SUMMARY_TSV}")
print(f"Wrote ${SUMMARY_JSON}")
PY

echo "Evaluation artifacts: ${OUT_DIR}"
echo "Raw run directories:"
cat "${RUN_DIRS_FILE}"
