#!/usr/bin/env bash
set -euo pipefail

MODEL="${ORNITH_MLX_MODEL:-mlx-community/Ornith-1.0-35B-4bit}"
HOST="${ORNITH_MLX_HOST:-127.0.0.1}"
PORT="${ORNITH_MLX_PORT:-8000}"
MAX_TOKENS="${ORNITH_MLX_MAX_TOKENS:-4096}"
TEMP="${ORNITH_MLX_TEMP:-0.3}"
TOP_P="${ORNITH_MLX_TOP_P:-0.95}"
TOP_K="${ORNITH_MLX_TOP_K:-20}"
PREFILL_STEP_SIZE="${ORNITH_MLX_PREFILL_STEP_SIZE:-1024}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_ornith_mlx_server.sh [options] [-- extra mlx_lm.server args]

Runs an OpenAI-compatible MLX-LM server for Ornith-1.0-35B.
The default model is mlx-community/Ornith-1.0-35B-4bit (~19 GiB).

Options:
  --model NAME        Model id/path (default: mlx-community/Ornith-1.0-35B-4bit)
  --host HOST         Bind host (default: 127.0.0.1)
  --port PORT         Bind port (default: 8000)
  --max-tokens N      Default max generation tokens (default: 4096)
  --temp N            Default temperature when the client omits one (default: 0.3)
  --top-p N           Default top-p when the client omits one (default: 0.95)
  --top-k N           Default top-k when the client omits one (default: 20)
  --prefill-step N    MLX prefill step size (default: 1024)
  --dry-run           Print command and exit
  -h, --help          Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-tokens" >&2; exit 1; }
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temp)
      [[ $# -lt 2 ]] && { echo "Missing value for --temp" >&2; exit 1; }
      TEMP="$2"
      shift 2
      ;;
    --top-p)
      [[ $# -lt 2 ]] && { echo "Missing value for --top-p" >&2; exit 1; }
      TOP_P="$2"
      shift 2
      ;;
    --top-k)
      [[ $# -lt 2 ]] && { echo "Missing value for --top-k" >&2; exit 1; }
      TOP_K="$2"
      shift 2
      ;;
    --prefill-step)
      [[ $# -lt 2 ]] && { echo "Missing value for --prefill-step" >&2; exit 1; }
      PREFILL_STEP_SIZE="$2"
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

if ! command -v mlx_lm.server >/dev/null 2>&1; then
  echo "mlx_lm.server is not on PATH. Install/update MLX-LM first." >&2
  exit 1
fi

cmd=(
  mlx_lm.server
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --max-tokens "${MAX_TOKENS}"
  --temp "${TEMP}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --prefill-step-size "${PREFILL_STEP_SIZE}"
)
cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

echo "Serving ${MODEL} at http://${HOST}:${PORT}/v1" >&2
exec "${cmd[@]}"
