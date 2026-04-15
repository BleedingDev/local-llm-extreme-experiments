#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_gemma4_a4b_variant.sh"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${SUPERGEMMA_MODEL:-Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2}"
ROTOR_MODEL="${GEMMA4_ROTOR_MODEL:-majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit}"
DRAFT_MODEL="${GEMMA4_DRAFT_MODEL:-mlx-community/gemma-4-e2b-it-4bit}"
PROMPT=""
PROMPT_FILE=""
ARTIFACTS_BASE="${ROOT_DIR}/artifacts/benchmarks/gemma4-a4b"
REPEATS=1
EXECUTE=0
PREFLIGHT=1
OFFLINE_ONLY=0

VARIANTS=(
  baseline
  speculative
  kv4
  triattention
  turboquant-v2-lean
  turboquant-v2-rot
  turboquant-v3-3.5
  rotorquant
  speculative-rotorquant
)
MAX_TOKEN_VALUES=(64)
MAX_KV_VALUES=(512 1024)

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark_gemma4_a4b_variants.sh [options]

Prepares and optionally runs Gemma 4 26B A4B benchmark variant matrix.
Safe by default: runs preflight checks and command dry-runs only.

Options:
  --venv-path PATH         Virtualenv path (default: ./.venv)
  --model NAME             Base 26B A4B model (default: SuperGemma 26B A4B)
  --rotor-model NAME       RotorQuant 26B A4B model id
  --draft-model NAME       Draft model for speculative decoding (can be smaller)
  --prompt TEXT            Prompt text for benchmark runs
  --prompt-file PATH       Prompt loaded from file
  --variant NAME           Add variant (repeatable)
  --max-tokens N           Add token budget (repeatable)
  --max-kv-size N          Add KV cap (repeatable)
  --repeats N              Repeats per matrix point (default: 1)
  --artifacts-dir PATH     Output base dir (default: ./artifacts/benchmarks/gemma4-a4b)
  --skip-preflight         Skip preflight checks
  --allow-network          Allow network model fetches (default behavior)
  --execute                Execute runs (default: dry-run planning only)
  -h, --help               Show this help text

Examples:
  scripts/benchmark_gemma4_a4b_variants.sh
  scripts/benchmark_gemma4_a4b_variants.sh --variant baseline --variant triattention --max-kv-size 2048
  scripts/benchmark_gemma4_a4b_variants.sh --execute --repeats 2 --max-kv-size 512 --max-kv-size 1024
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

USER_SET_VARIANTS=0
USER_SET_TOKENS=0
USER_SET_KV=0

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
    --rotor-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --rotor-model" >&2; exit 1; }
      ROTOR_MODEL="$2"
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
      shift 2
      ;;
    --prompt-file)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt-file" >&2; exit 1; }
      PROMPT_FILE="$2"
      shift 2
      ;;
    --variant)
      [[ $# -lt 2 ]] && { echo "Missing value for --variant" >&2; exit 1; }
      if [[ "${USER_SET_VARIANTS}" -eq 0 ]]; then
        VARIANTS=()
        USER_SET_VARIANTS=1
      fi
      VARIANTS+=("$2")
      shift 2
      ;;
    --max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-tokens" >&2; exit 1; }
      if [[ "${USER_SET_TOKENS}" -eq 0 ]]; then
        MAX_TOKEN_VALUES=()
        USER_SET_TOKENS=1
      fi
      MAX_TOKEN_VALUES+=("$2")
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      if [[ "${USER_SET_KV}" -eq 0 ]]; then
        MAX_KV_VALUES=()
        USER_SET_KV=1
      fi
      MAX_KV_VALUES+=("$2")
      shift 2
      ;;
    --repeats)
      [[ $# -lt 2 ]] && { echo "Missing value for --repeats" >&2; exit 1; }
      REPEATS="$2"
      shift 2
      ;;
    --artifacts-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --artifacts-dir" >&2; exit 1; }
      ARTIFACTS_BASE="$2"
      shift 2
      ;;
    --skip-preflight)
      PREFLIGHT=0
      shift
      ;;
    --allow-network)
      OFFLINE_ONLY=0
      shift
      ;;
    --execute)
      EXECUTE=1
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

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner script missing or not executable: ${RUNNER}" >&2
  exit 1
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Prompt file not found: ${PROMPT_FILE}" >&2
    exit 1
  fi
  PROMPT="$(cat "${PROMPT_FILE}")"
fi

if [[ -z "${PROMPT}" ]]; then
  PROMPT="Write one short note describing benchmark setup for Gemma 4 26B A4B."
fi

if ! is_positive_int "${REPEATS}"; then
  echo "--repeats must be a positive integer: ${REPEATS}" >&2
  exit 1
fi

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "At least one --variant is required." >&2
  exit 1
fi

for v in "${VARIANTS[@]}"; do
  case "${v}" in
    baseline|speculative|kv4|triattention|turboquant-v2-lean|turboquant-v2-rot|turboquant-v3-3.5|rotorquant|speculative-rotorquant)
      ;;
    *)
      echo "Unsupported variant: ${v}" >&2
      exit 1
      ;;
  esac
done

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

RESULTS_CSV="${RUN_DIR}/results.csv"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
CONFIG_FILE="${RUN_DIR}/config.env"

printf 'phase,variant,max_tokens,max_kv_size,repeat_index,status,exit_code,log_file\n' > "${RESULTS_CSV}"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "execute=${EXECUTE}"
  echo "preflight=${PREFLIGHT}"
  echo "offline_only=${OFFLINE_ONLY}"
  echo "model=${MODEL}"
  echo "rotor_model=${ROTOR_MODEL}"
  echo "draft_model=${DRAFT_MODEL}"
  echo "variants=${VARIANTS[*]}"
  echo "max_tokens=${MAX_TOKEN_VALUES[*]}"
  echo "max_kv_sizes=${MAX_KV_VALUES[*]}"
  echo "repeats=${REPEATS}"
} > "${CONFIG_FILE}"

echo "Gemma 4 A4B benchmark prep artifacts: ${RUN_DIR}"

failures=0

if [[ "${PREFLIGHT}" -eq 1 ]]; then
  runner_flags=()
  if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
    runner_flags+=(--offline-only)
  fi
  for variant in "${VARIANTS[@]}"; do
    log_file="${RAW_DIR}/preflight_${variant}.log"
    set +e
    "${RUNNER}" \
      --venv-path "${VENV_DIR}" \
      "${runner_flags[@]}" \
      --variant "${variant}" \
      --model "${MODEL}" \
      --rotor-model "${ROTOR_MODEL}" \
      --draft-model "${DRAFT_MODEL}" \
      --preflight > "${log_file}" 2>&1
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      failures=$((failures + 1))
    fi
    status="ok"
    if [[ "${rc}" -ne 0 ]]; then
      status="failed"
    fi
    printf 'preflight,%s,NA,NA,NA,%s,%s,%s\n' \
      "${variant}" "${status}" "${rc}" "raw/$(basename "${log_file}")" >> "${RESULTS_CSV}"
  done
fi

for variant in "${VARIANTS[@]}"; do
  for max_tokens in "${MAX_TOKEN_VALUES[@]}"; do
    for max_kv in "${MAX_KV_VALUES[@]}"; do
      repeat_idx=1
      while [[ "${repeat_idx}" -le "${REPEATS}" ]]; do
        log_file="${RAW_DIR}/matrix_${variant}_kv${max_kv}_tok${max_tokens}_rep${repeat_idx}.log"
        cmd=(
          "${RUNNER}"
          --venv-path "${VENV_DIR}"
          --variant "${variant}"
          --model "${MODEL}"
          --rotor-model "${ROTOR_MODEL}"
          --draft-model "${DRAFT_MODEL}"
          --max-tokens "${max_tokens}"
          --max-kv-size "${max_kv}"
          --prompt "${PROMPT}"
        )
        if [[ "${EXECUTE}" -eq 0 ]]; then
          cmd+=(--dry-run)
        fi
        if [[ "${OFFLINE_ONLY}" -eq 1 ]]; then
          cmd+=(--offline-only)
        fi

        set +e
        "${cmd[@]}" > "${log_file}" 2>&1
        rc=$?
        set -e

        status="ok"
        if [[ "${rc}" -ne 0 ]]; then
          status="failed"
          failures=$((failures + 1))
        fi

        printf 'matrix,%s,%s,%s,%s,%s,%s,%s\n' \
          "${variant}" \
          "${max_tokens}" \
          "${max_kv}" \
          "${repeat_idx}" \
          "${status}" \
          "${rc}" \
          "raw/$(basename "${log_file}")" >> "${RESULTS_CSV}"

        repeat_idx=$((repeat_idx + 1))
      done
    done
  done
done

{
  echo "run_dir=${RUN_DIR}"
  echo "results_csv=${RESULTS_CSV}"
  echo "config_file=${CONFIG_FILE}"
  echo "execute=${EXECUTE}"
  echo "preflight=${PREFLIGHT}"
  echo "offline_only=${OFFLINE_ONLY}"
  echo "failures=${failures}"
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${SUMMARY_FILE}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${SUMMARY_FILE}"

if [[ "${failures}" -gt 0 ]]; then
  echo "Gemma A4B prep completed with failures. Inspect logs in ${RAW_DIR}." >&2
  exit 1
fi

echo "Gemma A4B prep completed successfully."
