#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${SUPERGEMMA_MODEL:-Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2}"
HOST="${SUPERGEMMA_HOST:-127.0.0.1}"
PORT="${SUPERGEMMA_PORT:-8080}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_supergemma_mlx_server.sh [options] [-- extra mlx_lm.server args]

Runs a local OpenAI-compatible MLX-LM server from project .venv.

Options:
  --venv-path PATH    Override venv path (default: ./.venv)
  --model NAME        Model id/path (default: Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2)
  --host HOST         Bind host (default: 127.0.0.1)
  --port PORT         Bind port (default: 8080)
  --dry-run           Print command and exit
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
    --host)
      [[ $# -lt 2 ]] && { echo "Missing value for --host" >&2; exit 1; }
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -lt 2 ]] && { echo "Missing value for --port" >&2; exit 1; }
      PORT="$2"
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

MLX_SERVER_BIN="${VENV_DIR}/bin/mlx_lm.server"
if [[ ! -x "${MLX_SERVER_BIN}" ]]; then
  echo "mlx_lm.server not found at ${MLX_SERVER_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

cmd=(
  "${MLX_SERVER_BIN}"
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
)
cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
