#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN=""
MODEL="${QWEN36_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
DRAFT_MODEL="${QWEN36_DRAFT_MODEL:-z-lab/Qwen3.6-35B-A3B-DFlash}"
PROMPT="Count from 1 to 40, one number per line, and stop."
PROMPT_FILE=""
TOKENS_LIST="64,128"
BLOCK_SIZES="8,12,16"
REPEATS=1
WARMUP_PASSES=1
WARMUP_TOKENS=8
MAX_KV_SIZE=2048
PREFILL_CHUNK_SIZE=512
DRAFT_SLIDING_WINDOW_SIZE=4096
ARTIFACTS_ROOT="${ROOT_DIR}/artifacts/benchmarks/qwen36-dflash-prompt-sweep"
RUN_NAME=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_qwen36_dflash_prompt_sweep.sh [options] [-- extra benchmark_qwen36_dflash_prompt.py args]

Runs a prompt-only Qwen 3.6 4-bit baseline + DFlash sweep with one config per Python process.

Options:
  --venv-path PATH                 Override venv path (default: ./.venv)
  --python-bin PATH                Override python binary (default: <venv>/bin/python)
  --model NAME                     Target model (default: mlx-community/Qwen3.6-35B-A3B-4bit)
  --draft-model NAME               Draft model (default: z-lab/Qwen3.6-35B-A3B-DFlash)
  --prompt TEXT                    Prompt text
  --prompt-file PATH               Read prompt from file
  --tokens-list CSV                Max-new-token sweep (default: 64,128)
  --block-sizes CSV                DFlash block-size sweep (default: 8,12,16)
  --repeats N                      Measured repeats per config (default: 1)
  --warmup-passes N                Warmup passes per config (default: 1)
  --warmup-tokens N                Warmup token limit (default: 8)
  --max-kv-size N                  KV cache cap (default: 2048)
  --prefill-chunk-size N           DFlash prefill chunk size (default: 512)
  --draft-sliding-window-size N    Draft sliding window size (default: 4096)
  --artifacts-root PATH            Sweep artifacts root
  --run-name NAME                  Optional sweep directory name
  --dry-run                        Print resolved commands and exit
  -h, --help                       Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --venv-path" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -lt 2 ]] && { echo "Missing value for --python-bin" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --draft-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --draft-model" >&2; exit 1; }
      DRAFT_MODEL="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt" >&2; exit 1; }
      PROMPT="$2"
      PROMPT_FILE=""
      shift 2
      ;;
    --prompt-file)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt-file" >&2; exit 1; }
      PROMPT_FILE="$2"
      shift 2
      ;;
    --tokens-list)
      [[ $# -lt 2 ]] && { echo "Missing value for --tokens-list" >&2; exit 1; }
      TOKENS_LIST="$2"
      shift 2
      ;;
    --block-sizes)
      [[ $# -lt 2 ]] && { echo "Missing value for --block-sizes" >&2; exit 1; }
      BLOCK_SIZES="$2"
      shift 2
      ;;
    --repeats)
      [[ $# -lt 2 ]] && { echo "Missing value for --repeats" >&2; exit 1; }
      REPEATS="$2"
      shift 2
      ;;
    --warmup-passes)
      [[ $# -lt 2 ]] && { echo "Missing value for --warmup-passes" >&2; exit 1; }
      WARMUP_PASSES="$2"
      shift 2
      ;;
    --warmup-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --warmup-tokens" >&2; exit 1; }
      WARMUP_TOKENS="$2"
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      MAX_KV_SIZE="$2"
      shift 2
      ;;
    --prefill-chunk-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --prefill-chunk-size" >&2; exit 1; }
      PREFILL_CHUNK_SIZE="$2"
      shift 2
      ;;
    --draft-sliding-window-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --draft-sliding-window-size" >&2; exit 1; }
      DRAFT_SLIDING_WINDOW_SIZE="$2"
      shift 2
      ;;
    --artifacts-root)
      [[ $# -lt 2 ]] && { echo "Missing value for --artifacts-root" >&2; exit 1; }
      ARTIFACTS_ROOT="$2"
      shift 2
      ;;
    --run-name)
      [[ $# -lt 2 ]] && { echo "Missing value for --run-name" >&2; exit 1; }
      RUN_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="qwen36-dflash-prompt-sweep-${timestamp}"
fi
RUN_ROOT="${ARTIFACTS_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_ROOT}"

common_args=(
  --model "${MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --repeats "${REPEATS}"
  --warmup-passes "${WARMUP_PASSES}"
  --warmup-tokens "${WARMUP_TOKENS}"
  --max-kv-size "${MAX_KV_SIZE}"
  --prefill-chunk-size "${PREFILL_CHUNK_SIZE}"
  --draft-sliding-window-size "${DRAFT_SLIDING_WINDOW_SIZE}"
  --artifacts-dir "${RUN_ROOT}"
)

if [[ -n "${PROMPT_FILE}" ]]; then
  common_args+=(--prompt-file "${PROMPT_FILE}")
else
  common_args+=(--prompt "${PROMPT}")
fi

IFS=',' read -r -a token_grid <<< "${TOKENS_LIST}"
IFS=',' read -r -a block_grid <<< "${BLOCK_SIZES}"

run_case() {
  local label="$1"
  shift
  local cmd=("${PYTHON_BIN}" "${ROOT_DIR}/scripts/benchmark_qwen36_dflash_prompt.py" "$@" "${EXTRA_ARGS[@]}")
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  echo "== ${label}"
  "${cmd[@]}"
}

for token_count in "${token_grid[@]}"; do
  token_count="${token_count//[[:space:]]/}"
  [[ -z "${token_count}" ]] && continue

  run_case \
    "baseline t${token_count}" \
    --mode baseline \
    --max-new-tokens "${token_count}" \
    --run-name "t${token_count}-baseline" \
    "${common_args[@]}"

  for block_size in "${block_grid[@]}"; do
    block_size="${block_size//[[:space:]]/}"
    [[ -z "${block_size}" ]] && continue
    run_case \
      "dflash t${token_count} b${block_size}" \
      --mode dflash \
      --block-size "${block_size}" \
      --max-new-tokens "${token_count}" \
      --run-name "t${token_count}-dflash-b${block_size}" \
      "${common_args[@]}"
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

python3 - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
for result_path in sorted(run_root.glob("*/result.json")):
    payload = json.loads(result_path.read_text())
    config = payload["config"]
    summary = payload["summary"]
    rows.append(
        {
            "run_dir": str(result_path.parent),
            "mode": config["mode"],
            "max_new_tokens": config["max_new_tokens"],
            "block_size": config["block_size"],
            "generation_tps": summary["mean_generation_tps"],
            "prompt_tps": summary["mean_prompt_tps"],
            "peak_memory_gb": summary["mean_peak_memory_gb"],
            "acceptance_length": summary["mean_acceptance_length"],
        }
    )

rows.sort(key=lambda item: (item["max_new_tokens"], item["mode"], item["block_size"] or 0))
summary_path = run_root / "summary.json"
summary_path.write_text(json.dumps(rows, indent=2))

print("")
print("Summary")
print("tokens\tmode\tblock\tgen_tps\tprompt_tps\tpeak_gb\taccept")
for row in rows:
    gen = "n/a" if row["generation_tps"] is None else f"{row['generation_tps']:.2f}"
    prompt = "n/a" if row["prompt_tps"] is None else f"{row['prompt_tps']:.2f}"
    peak = "n/a" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f}"
    accept = "n/a" if row["acceptance_length"] is None else f"{row['acceptance_length']:.2f}"
    block = "-" if row["block_size"] is None else str(row["block_size"])
    print(f"{row['max_new_tokens']}\t{row['mode']}\t{block}\t{gen}\t{prompt}\t{peak}\t{accept}")

print("")
print(f"Summary JSON: {summary_path}")
PY
