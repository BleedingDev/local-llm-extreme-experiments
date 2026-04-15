#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${SUPERGEMMA_MODEL:-Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2}"
PROMPT=""
PROMPT_FILE=""
MAX_TOKEN_VALUES=()
MAX_KV_VALUES=()
REPEATS="${SUPERGEMMA_BENCH_REPEATS:-2}"
MODE="benchmark"
PROBE_START_KV="${SUPERGEMMA_PROBE_START_KV:-512}"
PROBE_MAX_KV="${SUPERGEMMA_PROBE_MAX_KV:-4096}"
PROBE_MAX_TOKENS="${SUPERGEMMA_PROBE_MAX_TOKENS:-32}"
ARTIFACTS_BASE="${ROOT_DIR}/artifacts/benchmarks"
DRY_RUN=0
EXTRA_ARGS=()

RUN_COUNTER=0
BENCHMARK_FAILURES=0
PROBE_STATUS="not-run"
PROBE_BEST_KV=""
PROBE_FIRST_FAILURE_KV=""

PARSED_PROMPT_TOKENS=""
PARSED_PROMPT_TPS=""
PARSED_GENERATION_TOKENS=""
PARSED_GENERATION_TPS=""
PARSED_PEAK_MEMORY_GB=""

LAST_RUN_EXIT_CODE=0
LAST_RUN_STATUS="pending"

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark_supergemma_mlx.sh [options] [-- extra mlx_lm.generate args]

Reproducible benchmark harness for SuperGemma on MLX using project .venv tools.

Options:
  --venv-path PATH          Override venv path (default: ./.venv)
  --model NAME              Model id/path (default: Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2)
  --prompt TEXT             Prompt text to benchmark
  --prompt-file PATH        Read prompt text from file
  --max-tokens N            Generation token limit (repeatable; default: 64)
  --max-kv-size N           KV cache cap (repeatable; default: 512)
  --repeats N               Repeats per max-tokens/max-kv-size combo (default: 2)
  --mode MODE               benchmark|probe|all (default: benchmark)
  --probe-context-limit     Alias for --mode probe
  --benchmark-and-probe     Alias for --mode all
  --probe-start-kv N        Probe starting KV size (default: 512)
  --probe-max-kv N          Probe upper bound KV size (default: 4096)
  --probe-max-tokens N      Max tokens per probe attempt (default: 32)
  --artifacts-dir PATH      Artifact base dir (default: ./artifacts/benchmarks)
  --dry-run                 Do not execute model; write planned runs/logs only
  -h, --help                Show this help text
USAGE
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

normalize_path() {
  local input_path="$1"
  if [[ "${input_path}" == /* ]]; then
    printf '%s\n' "${input_path}"
  else
    printf '%s\n' "${ROOT_DIR}/${input_path}"
  fi
}

parse_metrics() {
  local log_file="$1"

  PARSED_PROMPT_TOKENS=""
  PARSED_PROMPT_TPS=""
  PARSED_GENERATION_TOKENS=""
  PARSED_GENERATION_TPS=""
  PARSED_PEAK_MEMORY_GB=""

  local prompt_line
  local generation_line
  local peak_line

  prompt_line="$(grep -E 'Prompt: [0-9]+ tokens, [0-9]+([.][0-9]+)? tokens-per-sec' "${log_file}" | tail -n 1 || true)"
  generation_line="$(grep -E 'Generation: [0-9]+ tokens, [0-9]+([.][0-9]+)? tokens-per-sec' "${log_file}" | tail -n 1 || true)"
  peak_line="$(grep -E 'Peak memory: [0-9]+([.][0-9]+)? GB' "${log_file}" | tail -n 1 || true)"

  if [[ "${prompt_line}" =~ Prompt:[[:space:]]([0-9]+)[[:space:]]tokens,[[:space:]]([0-9]+([.][0-9]+)?)[[:space:]]tokens-per-sec ]]; then
    PARSED_PROMPT_TOKENS="${BASH_REMATCH[1]}"
    PARSED_PROMPT_TPS="${BASH_REMATCH[2]}"
  fi

  if [[ "${generation_line}" =~ Generation:[[:space:]]([0-9]+)[[:space:]]tokens,[[:space:]]([0-9]+([.][0-9]+)?)[[:space:]]tokens-per-sec ]]; then
    PARSED_GENERATION_TOKENS="${BASH_REMATCH[1]}"
    PARSED_GENERATION_TPS="${BASH_REMATCH[2]}"
  fi

  if [[ "${peak_line}" =~ Peak[[:space:]]memory:[[:space:]]([0-9]+([.][0-9]+)?)[[:space:]]GB ]]; then
    PARSED_PEAK_MEMORY_GB="${BASH_REMATCH[1]}"
  fi
}

append_result_row() {
  local run_id="$1"
  local phase="$2"
  local status="$3"
  local exit_code="$4"
  local max_tokens="$5"
  local max_kv_size="$6"
  local repeat_index="$7"
  local duration_sec="$8"
  local start_utc="$9"
  local end_utc="${10}"
  local stdout_rel="${11}"
  local stderr_rel="${12}"
  local combined_rel="${13}"

  local prompt_tokens="${PARSED_PROMPT_TOKENS:-NA}"
  local prompt_tps="${PARSED_PROMPT_TPS:-NA}"
  local generation_tokens="${PARSED_GENERATION_TOKENS:-NA}"
  local generation_tps="${PARSED_GENERATION_TPS:-NA}"
  local peak_memory_gb="${PARSED_PEAK_MEMORY_GB:-NA}"
  local model_csv="${MODEL//,/;}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${run_id}" \
    "${phase}" \
    "${status}" \
    "${exit_code}" \
    "${model_csv}" \
    "${max_tokens}" \
    "${max_kv_size}" \
    "${repeat_index}" \
    "${prompt_tokens}" \
    "${prompt_tps}" \
    "${generation_tokens}" \
    "${generation_tps}" \
    "${peak_memory_gb}" \
    "${duration_sec}" \
    "${start_utc}" \
    "${end_utc}" \
    "${stdout_rel}" \
    "${stderr_rel}" \
    "${combined_rel}" >> "${RESULTS_CSV}"
}

run_single_generate() {
  local phase="$1"
  local max_tokens="$2"
  local max_kv_size="$3"
  local repeat_index="$4"
  local prompt_text="$5"

  RUN_COUNTER=$((RUN_COUNTER + 1))
  local run_id
  run_id="$(printf '%04d' "${RUN_COUNTER}")"

  local prefix
  prefix="${run_id}_${phase}_kv${max_kv_size}_tok${max_tokens}_rep${repeat_index}"

  local stdout_log="${RAW_DIR}/${prefix}.stdout.log"
  local stderr_log="${RAW_DIR}/${prefix}.stderr.log"
  local combined_log="${RAW_DIR}/${prefix}.combined.log"
  local cmd_log="${RAW_DIR}/${prefix}.cmd.txt"

  local -a cmd=(
    "${MLX_GENERATE_BIN}"
    --model "${MODEL}"
    --prompt "${prompt_text}"
    --max-tokens "${max_tokens}"
    --max-kv-size "${max_kv_size}"
  )
  cmd+=("${EXTRA_ARGS[@]}")

  {
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } > "${cmd_log}"

  echo "[${phase}] run ${run_id}: max_tokens=${max_tokens} max_kv_size=${max_kv_size} repeat=${repeat_index}"

  local start_epoch
  local end_epoch
  local duration_sec
  local start_utc
  local end_utc
  local exit_code=0
  local status="ok"

  start_epoch="$(date +%s)"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    {
      echo "[dry-run] command not executed"
      cat "${cmd_log}"
    } > "${stdout_log}"
    : > "${stderr_log}"
    status="dry-run"
    exit_code=0
  else
    set +e
    "${cmd[@]}" > "${stdout_log}" 2> "${stderr_log}"
    exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
      status="failed"
    fi
  fi

  cat "${stdout_log}" "${stderr_log}" > "${combined_log}"
  parse_metrics "${combined_log}"

  end_epoch="$(date +%s)"
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  duration_sec=$((end_epoch - start_epoch))

  local stdout_rel="raw/${prefix}.stdout.log"
  local stderr_rel="raw/${prefix}.stderr.log"
  local combined_rel="raw/${prefix}.combined.log"

  append_result_row \
    "${run_id}" \
    "${phase}" \
    "${status}" \
    "${exit_code}" \
    "${max_tokens}" \
    "${max_kv_size}" \
    "${repeat_index}" \
    "${duration_sec}" \
    "${start_utc}" \
    "${end_utc}" \
    "${stdout_rel}" \
    "${stderr_rel}" \
    "${combined_rel}"

  LAST_RUN_EXIT_CODE="${exit_code}"
  LAST_RUN_STATUS="${status}"

  return "${exit_code}"
}

run_benchmark_matrix() {
  local max_tokens
  local max_kv_size
  local repeat_index

  for max_tokens in "${MAX_TOKEN_VALUES[@]}"; do
    for max_kv_size in "${MAX_KV_VALUES[@]}"; do
      repeat_index=1
      while [[ "${repeat_index}" -le "${REPEATS}" ]]; do
        if run_single_generate "benchmark" "${max_tokens}" "${max_kv_size}" "${repeat_index}" "${PROMPT}"; then
          :
        else
          BENCHMARK_FAILURES=$((BENCHMARK_FAILURES + 1))
        fi
        repeat_index=$((repeat_index + 1))
      done
    done
  done
}

run_context_probe() {
  local kv
  local last_success=0
  local first_failure=0
  local attempt_index=1

  kv="${PROBE_START_KV}"
  while [[ "${kv}" -le "${PROBE_MAX_KV}" ]]; do
    if run_single_generate "probe-scan" "${PROBE_MAX_TOKENS}" "${kv}" "${attempt_index}" "${PROMPT}"; then
      last_success="${kv}"
      kv=$((kv * 2))
    else
      first_failure="${kv}"
      break
    fi
    attempt_index=$((attempt_index + 1))
  done

  if [[ "${first_failure}" -eq 0 ]]; then
    PROBE_STATUS="no-failure-observed"
    if [[ "${last_success}" -gt 0 ]]; then
      PROBE_BEST_KV="${last_success}"
    else
      PROBE_BEST_KV="none"
    fi
    PROBE_FIRST_FAILURE_KV="none"
    return 0
  fi

  local low="${last_success}"
  local high="${first_failure}"
  local mid

  while (( high - low > 1 )); do
    mid=$(((low + high) / 2))
    attempt_index=$((attempt_index + 1))
    if run_single_generate "probe-refine" "${PROBE_MAX_TOKENS}" "${mid}" "${attempt_index}" "${PROMPT}"; then
      low="${mid}"
    else
      high="${mid}"
    fi
  done

  PROBE_FIRST_FAILURE_KV="${high}"
  if [[ "${low}" -gt 0 ]]; then
    PROBE_STATUS="bounded"
    PROBE_BEST_KV="${low}"
    return 0
  fi

  PROBE_STATUS="no-working-kv"
  PROBE_BEST_KV="none"
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --venv-path" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt" >&2; exit 1; }
      PROMPT="$2"
      shift 2
      ;;
    --prompt-file)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt-file" >&2; exit 1; }
      PROMPT_FILE="$2"
      shift 2
      ;;
    --max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-tokens" >&2; exit 1; }
      MAX_TOKEN_VALUES+=("$2")
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      MAX_KV_VALUES+=("$2")
      shift 2
      ;;
    --repeats)
      [[ $# -lt 2 ]] && { echo "Missing value for --repeats" >&2; exit 1; }
      REPEATS="$2"
      shift 2
      ;;
    --mode)
      [[ $# -lt 2 ]] && { echo "Missing value for --mode" >&2; exit 1; }
      MODE="$2"
      shift 2
      ;;
    --probe-context-limit)
      MODE="probe"
      shift
      ;;
    --benchmark-and-probe)
      MODE="all"
      shift
      ;;
    --probe-start-kv)
      [[ $# -lt 2 ]] && { echo "Missing value for --probe-start-kv" >&2; exit 1; }
      PROBE_START_KV="$2"
      shift 2
      ;;
    --probe-max-kv)
      [[ $# -lt 2 ]] && { echo "Missing value for --probe-max-kv" >&2; exit 1; }
      PROBE_MAX_KV="$2"
      shift 2
      ;;
    --probe-max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --probe-max-tokens" >&2; exit 1; }
      PROBE_MAX_TOKENS="$2"
      shift 2
      ;;
    --artifacts-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --artifacts-dir" >&2; exit 1; }
      ARTIFACTS_BASE="$2"
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
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "benchmark" && "${MODE}" != "probe" && "${MODE}" != "all" ]]; then
  echo "Invalid --mode value: ${MODE}. Expected benchmark, probe, or all." >&2
  exit 1
fi

if [[ "${#MAX_TOKEN_VALUES[@]}" -eq 0 ]]; then
  MAX_TOKEN_VALUES=("64")
fi

if [[ "${#MAX_KV_VALUES[@]}" -eq 0 ]]; then
  MAX_KV_VALUES=("512")
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Prompt file not found: ${PROMPT_FILE}" >&2
    exit 1
  fi
  PROMPT="$(cat "${PROMPT_FILE}")"
fi

if [[ -z "${PROMPT}" ]]; then
  PROMPT="Write one concise Czech sentence confirming this benchmark run completed."
fi

if ! is_positive_int "${REPEATS}"; then
  echo "--repeats must be a positive integer: ${REPEATS}" >&2
  exit 1
fi

if ! is_positive_int "${PROBE_START_KV}"; then
  echo "--probe-start-kv must be a positive integer: ${PROBE_START_KV}" >&2
  exit 1
fi

if ! is_positive_int "${PROBE_MAX_KV}"; then
  echo "--probe-max-kv must be a positive integer: ${PROBE_MAX_KV}" >&2
  exit 1
fi

if ! is_positive_int "${PROBE_MAX_TOKENS}"; then
  echo "--probe-max-tokens must be a positive integer: ${PROBE_MAX_TOKENS}" >&2
  exit 1
fi

if [[ "${PROBE_START_KV}" -gt "${PROBE_MAX_KV}" ]]; then
  echo "--probe-start-kv (${PROBE_START_KV}) must be <= --probe-max-kv (${PROBE_MAX_KV})" >&2
  exit 1
fi

for value in "${MAX_TOKEN_VALUES[@]}"; do
  if ! is_positive_int "${value}"; then
    echo "--max-tokens values must be positive integers: ${value}" >&2
    exit 1
  fi
done

for value in "${MAX_KV_VALUES[@]}"; do
  if ! is_positive_int "${value}"; then
    echo "--max-kv-size values must be positive integers: ${value}" >&2
    exit 1
  fi
done

ARTIFACTS_BASE="$(normalize_path "${ARTIFACTS_BASE}")"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ARTIFACTS_BASE}/${RUN_STAMP}"
if [[ -e "${RUN_DIR}" ]]; then
  RUN_DIR="${RUN_DIR}_$$"
fi
RAW_DIR="${RUN_DIR}/raw"

mkdir -p "${RAW_DIR}"

MLX_GENERATE_BIN="${VENV_DIR}/bin/mlx_lm.generate"
if [[ ! -x "${MLX_GENERATE_BIN}" ]]; then
  echo "mlx_lm.generate not found at ${MLX_GENERATE_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

RESULTS_CSV="${RUN_DIR}/results.csv"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
PROMPT_FILE_OUT="${RUN_DIR}/prompt.txt"
CONFIG_FILE="${RUN_DIR}/config.env"

printf 'run_id,phase,status,exit_code,model,max_tokens,max_kv_size,repeat_index,prompt_tokens,prompt_tps,generation_tokens,generation_tps,peak_memory_gb,duration_sec,start_utc,end_utc,stdout_log,stderr_log,combined_log\n' > "${RESULTS_CSV}"
printf '%s\n' "${PROMPT}" > "${PROMPT_FILE_OUT}"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "mode=${MODE}"
  echo "model=${MODEL}"
  echo "venv_dir=${VENV_DIR}"
  echo "repeats=${REPEATS}"
  echo "max_tokens_values=${MAX_TOKEN_VALUES[*]}"
  echo "max_kv_size_values=${MAX_KV_VALUES[*]}"
  echo "probe_start_kv=${PROBE_START_KV}"
  echo "probe_max_kv=${PROBE_MAX_KV}"
  echo "probe_max_tokens=${PROBE_MAX_TOKENS}"
  echo "dry_run=${DRY_RUN}"
} > "${CONFIG_FILE}"

echo "Benchmark artifacts directory: ${RUN_DIR}"

exit_code=0

if [[ "${MODE}" == "benchmark" || "${MODE}" == "all" ]]; then
  run_benchmark_matrix
fi

if [[ "${MODE}" == "probe" || "${MODE}" == "all" ]]; then
  if ! run_context_probe; then
    exit_code=1
  fi
fi

if [[ "${BENCHMARK_FAILURES}" -gt 0 ]]; then
  exit_code=1
fi

{
  echo "run_dir=${RUN_DIR}"
  echo "results_csv=${RESULTS_CSV}"
  echo "prompt_file=${PROMPT_FILE_OUT}"
  echo "mode=${MODE}"
  echo "dry_run=${DRY_RUN}"
  echo "benchmark_failures=${BENCHMARK_FAILURES}"
  echo "probe_status=${PROBE_STATUS}"
  if [[ -n "${PROBE_BEST_KV}" ]]; then
    echo "probe_best_kv=${PROBE_BEST_KV}"
  fi
  if [[ -n "${PROBE_FIRST_FAILURE_KV}" ]]; then
    echo "probe_first_failure_kv=${PROBE_FIRST_FAILURE_KV}"
  fi
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${SUMMARY_FILE}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${SUMMARY_FILE}"

if [[ "${exit_code}" -ne 0 ]]; then
  echo "Benchmark harness completed with failures." >&2
else
  echo "Benchmark harness completed successfully."
fi

exit "${exit_code}"
