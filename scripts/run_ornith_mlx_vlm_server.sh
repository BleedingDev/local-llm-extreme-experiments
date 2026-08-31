#!/usr/bin/env bash
set -euo pipefail

MODEL="${ORNITH_MLX_VLM_MODEL:-mlx-community/Ornith-1.0-35B-4bit}"
HOST="${ORNITH_MLX_VLM_HOST:-127.0.0.1}"
PORT="${ORNITH_MLX_VLM_PORT:-8000}"
PREFILL_STEP_SIZE="${ORNITH_MLX_VLM_PREFILL_STEP_SIZE:-4096}"
MAX_KV_SIZE="${ORNITH_MLX_VLM_MAX_KV_SIZE:-0}"
KV_BITS="${ORNITH_MLX_VLM_KV_BITS:-0}"
KV_QUANT_SCHEME="${ORNITH_MLX_VLM_KV_QUANT_SCHEME:-uniform}"
KV_GROUP_SIZE="${ORNITH_MLX_VLM_KV_GROUP_SIZE:-64}"
QUANTIZED_KV_START="${ORNITH_MLX_VLM_QUANTIZED_KV_START:-5000}"
APC_ENABLED="${ORNITH_MLX_VLM_APC_ENABLED:-0}"
APC_BLOCK_SIZE="${ORNITH_MLX_VLM_APC_BLOCK_SIZE:-16}"
APC_NUM_BLOCKS="${ORNITH_MLX_VLM_APC_NUM_BLOCKS:-8192}"
APC_EXACT_CACHE_ENTRIES="${ORNITH_MLX_VLM_APC_EXACT_CACHE_ENTRIES:-2}"
APC_DISK_PATH="${ORNITH_MLX_VLM_APC_DISK_PATH:-}"
APC_DISK_MAX_GB="${ORNITH_MLX_VLM_APC_DISK_MAX_GB:-0}"
PRELOAD="${ORNITH_MLX_VLM_PRELOAD:-1}"
SERVER_BIN="${ORNITH_MLX_VLM_SERVER_BIN:-}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_ornith_mlx_vlm_server.sh [options] [-- extra mlx_vlm.server args]

Runs the OpenAI-compatible mlx-vlm server for Ornith-1.0-35B.
This is the preferred Apple Silicon path for the MLX VLM checkpoint.

Options:
  --model NAME          Model id/path (default: mlx-community/Ornith-1.0-35B-4bit)
  --host HOST           Bind host (default: 127.0.0.1)
  --port PORT           Bind port (default: 8000)
  --prefill-step N      Prefill step size (default: 4096)
  --max-kv-size N       KV cache token cap; 0 disables (default: 0)
  --kv-bits N           KV cache quant bits; 0 disables (default: 0)
  --kv-scheme NAME      KV quant scheme: uniform or turboquant (default: uniform)
  --kv-group-size N     KV uniform group size (default: 64)
  --kv-start N          Token index to start KV quantization (default: 5000)
  --apc                 Enable mlx-vlm automatic prefix caching
  --apc-block-size N    APC block size in tokens (default: 16)
  --apc-num-blocks N    APC in-memory block pool size (default: 8192)
  --apc-exact-entries N APC exact prompt-cache snapshots (default: 2)
  --apc-disk PATH       Optional APC disk cache path
  --apc-disk-max-gb N   Optional APC disk cache cap in GB (default: unbounded)
  --server-bin PATH     mlx_vlm.server executable (default: .venv/bin/mlx_vlm.server when present)
  --no-preload          Do not load the model at server startup
  --dry-run             Print command and exit
  -h, --help            Show this help text
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
    --prefill-step)
      [[ $# -lt 2 ]] && { echo "Missing value for --prefill-step" >&2; exit 1; }
      PREFILL_STEP_SIZE="$2"
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      MAX_KV_SIZE="$2"
      shift 2
      ;;
    --kv-bits)
      [[ $# -lt 2 ]] && { echo "Missing value for --kv-bits" >&2; exit 1; }
      KV_BITS="$2"
      shift 2
      ;;
    --kv-scheme)
      [[ $# -lt 2 ]] && { echo "Missing value for --kv-scheme" >&2; exit 1; }
      KV_QUANT_SCHEME="$2"
      shift 2
      ;;
    --kv-group-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --kv-group-size" >&2; exit 1; }
      KV_GROUP_SIZE="$2"
      shift 2
      ;;
    --kv-start)
      [[ $# -lt 2 ]] && { echo "Missing value for --kv-start" >&2; exit 1; }
      QUANTIZED_KV_START="$2"
      shift 2
      ;;
    --apc)
      APC_ENABLED=1
      shift
      ;;
    --apc-block-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --apc-block-size" >&2; exit 1; }
      APC_BLOCK_SIZE="$2"
      shift 2
      ;;
    --apc-num-blocks)
      [[ $# -lt 2 ]] && { echo "Missing value for --apc-num-blocks" >&2; exit 1; }
      APC_NUM_BLOCKS="$2"
      shift 2
      ;;
    --apc-exact-entries)
      [[ $# -lt 2 ]] && { echo "Missing value for --apc-exact-entries" >&2; exit 1; }
      APC_EXACT_CACHE_ENTRIES="$2"
      shift 2
      ;;
    --apc-disk)
      [[ $# -lt 2 ]] && { echo "Missing value for --apc-disk" >&2; exit 1; }
      APC_DISK_PATH="$2"
      shift 2
      ;;
    --apc-disk-max-gb)
      [[ $# -lt 2 ]] && { echo "Missing value for --apc-disk-max-gb" >&2; exit 1; }
      APC_DISK_MAX_GB="$2"
      shift 2
      ;;
    --server-bin)
      [[ $# -lt 2 ]] && { echo "Missing value for --server-bin" >&2; exit 1; }
      SERVER_BIN="$2"
      shift 2
      ;;
    --no-preload)
      PRELOAD=0
      shift
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

if [[ -z "${SERVER_BIN}" ]]; then
  if [[ -x ".venv/bin/mlx_vlm.server" ]]; then
    SERVER_BIN=".venv/bin/mlx_vlm.server"
  else
    SERVER_BIN="mlx_vlm.server"
  fi
fi

if ! command -v "${SERVER_BIN}" >/dev/null 2>&1; then
  echo "mlx_vlm.server is not on PATH. Install/update mlx-vlm first." >&2
  exit 1
fi

export PREFILL_STEP_SIZE
export MAX_KV_SIZE
export KV_BITS
export KV_QUANT_SCHEME
export KV_GROUP_SIZE
export QUANTIZED_KV_START
export APC_ENABLED
export APC_BLOCK_SIZE
export APC_NUM_BLOCKS
export APC_EXACT_CACHE_ENTRIES
if [[ -n "${APC_DISK_PATH}" ]]; then
  export APC_DISK_PATH
fi
if [[ "${APC_DISK_MAX_GB}" != "0" ]]; then
  export APC_DISK_MAX_GB
fi
if [[ "${PRELOAD}" -eq 1 ]]; then
  export PRELOAD_MODEL="${MODEL}"
else
  unset PRELOAD_MODEL || true
fi

cmd=(
  "${SERVER_BIN}"
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --prefill-step-size "${PREFILL_STEP_SIZE}"
)
if [[ "${MAX_KV_SIZE}" != "0" ]]; then
  cmd+=(--max-kv-size "${MAX_KV_SIZE}")
fi
if [[ "${KV_BITS}" != "0" ]]; then
  cmd+=(--kv-bits "${KV_BITS}" --kv-quant-scheme "${KV_QUANT_SCHEME}" --kv-group-size "${KV_GROUP_SIZE}" --quantized-kv-start "${QUANTIZED_KV_START}")
fi
cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

echo "Serving ${MODEL} with mlx-vlm at http://${HOST}:${PORT}/v1" >&2
exec "${cmd[@]}"
