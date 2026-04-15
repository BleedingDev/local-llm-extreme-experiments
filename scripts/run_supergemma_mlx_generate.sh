#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${SUPERGEMMA_MODEL:-Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2}"
PROMPT=""
MAX_TOKENS="${SUPERGEMMA_MAX_TOKENS:-256}"
MAX_KV_SIZE="${SUPERGEMMA_MAX_KV_SIZE:-512}"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_supergemma_mlx_generate.sh [options] [-- extra mlx_lm.generate args]

Runs local SuperGemma generation from the project .venv.

Options:
  --venv-path PATH    Override venv path (default: ./.venv)
  --model NAME        Model id/path (default: Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2)
  --prompt TEXT       Prompt text (default: stdin content or built-in smoke prompt)
  --max-tokens N      Generation token limit (default: 256)
  --max-kv-size N     KV cache cap (default: 512; raise gradually if memory allows)
  -h, --help          Show this help text
USAGE
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
    --max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-tokens" >&2; exit 1; }
      MAX_TOKENS="$2"
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      MAX_KV_SIZE="$2"
      shift 2
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

MLX_GENERATE_BIN="${VENV_DIR}/bin/mlx_lm.generate"
if [[ ! -x "${MLX_GENERATE_BIN}" ]]; then
  echo "mlx_lm.generate not found at ${MLX_GENERATE_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

if [[ -z "${PROMPT}" && ! -t 0 ]]; then
  PROMPT="$(cat)"
fi

if [[ -z "${PROMPT}" ]]; then
  PROMPT="Write one short Czech sentence confirming this local MLX runtime works."
fi

exec "${MLX_GENERATE_BIN}" \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-kv-size "${MAX_KV_SIZE}" \
  "${EXTRA_ARGS[@]}"
