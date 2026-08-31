#!/usr/bin/env bash
set -euo pipefail

DATASET="${TB_DATASET:-terminal-bench/terminal-bench-2-1}"
DATASET_PATH="${TB_DATASET_PATH:-}"
LITTLE_CODER_REPO="${LITTLE_CODER_REPO:-bench/vendor/little-coder}"
MODEL="${TB_MODEL:-omlx/mlx-community/Ornith-1.0-35B-4bit}"
N_CONCURRENT="${TB_N_CONCURRENT:-1}"
K="${TB_K:-1}"
JOB_NAME="${TB_JOB_NAME:-tb21_ornith_little_coder_$(date +%Y%m%d_%H%M%S)}"
MAX_TURNS="${TB_MAX_TURNS:-20}"
MAX_OUTPUT_TOKENS="${TB_MAX_OUTPUT_TOKENS:-768}"
WARMUP="${TB_WARMUP:-1}"

if [[ ! -d "${LITTLE_CODER_REPO}" ]]; then
  echo "Missing ${LITTLE_CODER_REPO}. Run scripts/setup_little_coder_tb_harness.sh first." >&2
  exit 1
fi

if ! command -v harbor >/dev/null 2>&1; then
  if [[ -x "bench/.venv/bin/harbor" ]]; then
    # shellcheck disable=SC1091
    source bench/.venv/bin/activate
  else
    echo "harbor is not on PATH and bench/.venv/bin/harbor does not exist." >&2
    exit 1
  fi
fi

export OMLX_API_KEY="${OMLX_API_KEY:-noop}"
export ORNITHCPP_API_KEY="${ORNITHCPP_API_KEY:-noop}"
export LITTLE_CODER_TEMPERATURE_PROVIDERS="${LITTLE_CODER_TEMPERATURE_PROVIDERS:-llamacpp,ollama,lmstudio,ornithcpp,omlx}"
export LITTLE_CODER_BENCHMARK=terminal_bench
export LITTLE_CODER_PERMISSION_MODE="${LITTLE_CODER_PERMISSION_MODE:-accept-all}"
export LITTLE_CODER_MAX_TURNS="${LITTLE_CODER_MAX_TURNS:-${MAX_TURNS}}"
export LITTLE_CODER_MAX_TOKENS="${LITTLE_CODER_MAX_TOKENS:-${MAX_OUTPUT_TOKENS}}"
if [[ "${TB_THINK:-0}" == "1" && -z "${LITTLE_CODER_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  export LITTLE_CODER_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
elif [[ -z "${LITTLE_CODER_CHAT_TEMPLATE_KWARGS:-}" ]]; then
  export LITTLE_CODER_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi

if [[ "${WARMUP}" == "1" ]]; then
  WARMUP_PY="scripts/warmup_ornith_openai_toolcall.py"
  if [[ -f "${WARMUP_PY}" ]]; then
    "${PYTHON:-python3}" "${WARMUP_PY}" \
      --base-url "${ORNITH_MLX_BASE_URL:-http://127.0.0.1:8001/v1}" \
      --model "${MODEL#omlx/}" \
      --timeout "${TB_WARMUP_TIMEOUT:-120}" >&2 || {
        echo "warning: Ornith warmup failed; continuing to benchmark" >&2
      }
  fi
fi

DATASET_ARGS=()
if [[ -n "${DATASET_PATH}" ]]; then
  DATASET_ARGS=(--path "${DATASET_PATH}")
else
  DATASET_ARGS=(-d "${DATASET}")
fi

PYTHONPATH="${LITTLE_CODER_REPO}:${PYTHONPATH:-}" harbor run \
  "${DATASET_ARGS[@]}" \
  --agent-import-path "benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent" \
  -m "${MODEL}" \
  -n "${N_CONCURRENT}" \
  -k "${K}" \
  --job-name "${JOB_NAME}" \
  "$@"
