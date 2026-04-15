#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
MLX_GENERATE_BIN="${VENV_DIR}/bin/mlx_lm.generate"
PIPELINE_SCRIPT="${ROOT_DIR}/scripts/long_context_map_reduce.py"

MODEL="${SUPERGEMMA_MODEL:-Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2}"
TOKENIZER="${LONG_CONTEXT_TOKENIZER:-${MODEL}}"
CHUNK_SIZE="${LONG_CONTEXT_CHUNK_SIZE:-3072}"
CHUNK_OVERLAP="${LONG_CONTEXT_CHUNK_OVERLAP:-256}"
MAP_MAX_TOKENS="${LONG_CONTEXT_MAP_MAX_TOKENS:-192}"
REDUCE_MAX_TOKENS="${LONG_CONTEXT_REDUCE_MAX_TOKENS:-256}"
FINAL_MAX_TOKENS="${LONG_CONTEXT_FINAL_MAX_TOKENS:-384}"
REDUCE_FAN_IN="${LONG_CONTEXT_REDUCE_FAN_IN:-8}"
MAX_KV_SIZE="${LONG_CONTEXT_MAX_KV_SIZE:-4096}"
ARTIFACTS_ROOT="${LONG_CONTEXT_ARTIFACTS_ROOT:-${ROOT_DIR}/artifacts/long-context}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_long_context_map_reduce.sh [pipeline args]

Wrapper for scripts/long_context_map_reduce.py with conservative local defaults.
You still must pass at minimum:
  --input-file PATH
  --question "your objective"

Defaults can be overridden with CLI flags or env vars:
  SUPERGEMMA_MODEL
  LONG_CONTEXT_TOKENIZER
  LONG_CONTEXT_CHUNK_SIZE
  LONG_CONTEXT_CHUNK_OVERLAP
  LONG_CONTEXT_MAP_MAX_TOKENS
  LONG_CONTEXT_REDUCE_MAX_TOKENS
  LONG_CONTEXT_FINAL_MAX_TOKENS
  LONG_CONTEXT_REDUCE_FAN_IN
  LONG_CONTEXT_MAX_KV_SIZE
  LONG_CONTEXT_ARTIFACTS_ROOT

Examples:
  scripts/run_long_context_map_reduce.sh \
    --input-file docs/large.txt \
    --question "List pricing risks and mitigations."

  scripts/run_long_context_map_reduce.sh \
    --input-file docs/large.txt \
    --question "Summarize architecture decisions." \
    --chunk-size 4096 \
    --max-kv-size 6144
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

if [[ ! -x "${MLX_GENERATE_BIN}" ]]; then
  echo "mlx_lm.generate not found at ${MLX_GENERATE_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "Pipeline script missing at ${PIPELINE_SCRIPT}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import transformers" >/dev/null 2>&1; then
  echo "transformers is not available in ${VENV_DIR}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${PIPELINE_SCRIPT}" \
  --mlx-generate-bin "${MLX_GENERATE_BIN}" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --map-max-tokens "${MAP_MAX_TOKENS}" \
  --reduce-max-tokens "${REDUCE_MAX_TOKENS}" \
  --final-max-tokens "${FINAL_MAX_TOKENS}" \
  --reduce-fan-in "${REDUCE_FAN_IN}" \
  --max-kv-size "${MAX_KV_SIZE}" \
  --artifacts-root "${ARTIFACTS_ROOT}" \
  "$@"
