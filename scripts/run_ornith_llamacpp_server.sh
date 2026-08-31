#!/usr/bin/env bash
set -euo pipefail

HF_REPO="${ORNITH_GGUF_REPO:-deepreinforce-ai/Ornith-1.0-35B-GGUF}"
HF_FILE="${ORNITH_GGUF_FILE:-ornith-1.0-35b-Q4_K_M.gguf}"
ALIAS="${ORNITH_LLAMA_ALIAS:-ornith-1.0-35b-q4-k-m}"
HOST="${ORNITH_LLAMA_HOST:-127.0.0.1}"
PORT="${ORNITH_LLAMA_PORT:-8888}"
CTX="${ORNITH_LLAMA_CTX:-16384}"
N_GPU_LAYERS="${ORNITH_LLAMA_N_GPU_LAYERS:-all}"
N_CPU_MOE="${ORNITH_LLAMA_N_CPU_MOE:-999}"
CACHE_TYPE_K="${ORNITH_LLAMA_CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${ORNITH_LLAMA_CACHE_TYPE_V:-q8_0}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_ornith_llamacpp_server.sh [options] [-- extra llama-server args]

Runs Ornith-1.0-35B via llama.cpp's OpenAI-compatible server.
Default checkpoint: deepreinforce-ai/Ornith-1.0-35B-GGUF / Q4_K_M.

Options:
  --repo REPO         Hugging Face GGUF repo (default: deepreinforce-ai/Ornith-1.0-35B-GGUF)
  --file FILE         GGUF filename (default: ornith-1.0-35b-Q4_K_M.gguf)
  --alias NAME        Served model id (default: ornith-1.0-35b-q4-k-m)
  --host HOST         Bind host (default: 127.0.0.1)
  --port PORT         Bind port (default: 8888)
  --ctx N             Context window (default: 16384)
  --dry-run           Print command and exit
  -h, --help          Show this help text

Environment:
  ORNITH_LLAMA_N_GPU_LAYERS=all|auto|N  GPU layer offload (default: all)
  ORNITH_LLAMA_N_CPU_MOE=N              MoE expert layers kept on CPU (default: 999)
  ORNITH_LLAMA_CACHE_TYPE_K=q8_0        KV K cache type (default: q8_0)
  ORNITH_LLAMA_CACHE_TYPE_V=q8_0        KV V cache type (default: q8_0)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -lt 2 ]] && { echo "Missing value for --repo" >&2; exit 1; }
      HF_REPO="$2"
      shift 2
      ;;
    --file)
      [[ $# -lt 2 ]] && { echo "Missing value for --file" >&2; exit 1; }
      HF_FILE="$2"
      shift 2
      ;;
    --alias)
      [[ $# -lt 2 ]] && { echo "Missing value for --alias" >&2; exit 1; }
      ALIAS="$2"
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
    --ctx)
      [[ $# -lt 2 ]] && { echo "Missing value for --ctx" >&2; exit 1; }
      CTX="$2"
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

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server is not on PATH. Install llama.cpp first." >&2
  exit 1
fi

cmd=(
  llama-server
  --hf-repo "${HF_REPO}"
  --hf-file "${HF_FILE}"
  --alias "${ALIAS}"
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CTX}"
  --n-gpu-layers "${N_GPU_LAYERS}"
  --flash-attn on
  --cache-type-k "${CACHE_TYPE_K}"
  --cache-type-v "${CACHE_TYPE_V}"
  --jinja
  --reasoning auto
  --reasoning-format deepseek
  --cache-prompt
  --no-webui
)

if [[ -n "${N_CPU_MOE}" ]]; then
  cmd+=(--n-cpu-moe "${N_CPU_MOE}")
fi

cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

echo "Serving ${HF_REPO}/${HF_FILE} as ${ALIAS} at http://${HOST}:${PORT}/v1" >&2
exec "${cmd[@]}"
