#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${OSAURUS_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
PORT="${OSAURUS_PORT:-1337}"
HOST="${OSAURUS_HOST:-127.0.0.1}"
MAX_TOKENS="${OSAURUS_MAX_TOKENS:-64}"
PROMPT="${OSAURUS_PROMPT:-Reply in one short sentence: osaurus qwen 3.6 is running locally.}"
START_SERVER=1
PULL_MODEL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_osaurus_qwen36_smoke.sh [options]

Starts or reuses a local Osaurus server, optionally pulls Qwen 3.6, and sends an
OpenAI-compatible chat completion request to the local endpoint.

Options:
  --model ID          Model id (default: mlx-community/Qwen3.6-35B-A3B-4bit)
  --port N            Server port (default: 1337)
  --host HOST         Server host (default: 127.0.0.1)
  --max-tokens N      Completion cap (default: 64)
  --prompt TEXT       Prompt text
  --pull              Pull the model before running the request
  --no-start          Do not auto-start Osaurus if stopped
  --dry-run           Print resolved commands and exit
  -h, --help          Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --port)
      [[ $# -lt 2 ]] && { echo "Missing value for --port" >&2; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --host)
      [[ $# -lt 2 ]] && { echo "Missing value for --host" >&2; exit 1; }
      HOST="$2"
      shift 2
      ;;
    --max-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-tokens" >&2; exit 1; }
      MAX_TOKENS="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt" >&2; exit 1; }
      PROMPT="$2"
      shift 2
      ;;
    --pull)
      PULL_MODEL=1
      shift
      ;;
    --no-start)
      START_SERVER=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

health_url="http://${HOST}:${PORT}/health"
models_url="http://${HOST}:${PORT}/v1/models"
chat_url="http://${HOST}:${PORT}/v1/chat/completions"

serve_cmd=(osaurus serve --port "${PORT}" --yes)
pull_cmd=(osaurus pull "${MODEL}")
chat_cmd=(
  curl -sS "${chat_url}"
  -H "Content-Type: application/json"
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":${MAX_TOKENS}}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Serve:'
  printf ' %q' "${serve_cmd[@]}"
  printf '\n'
  printf 'Pull:'
  printf ' %q' "${pull_cmd[@]}"
  printf '\n'
  printf 'Chat:'
  printf ' %q' "${chat_cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ "${START_SERVER}" -eq 1 ]]; then
  if ! osaurus status >/dev/null 2>&1; then
    "${serve_cmd[@]}" >/dev/null
  fi
fi

for _ in {1..30}; do
  if curl -sf "${health_url}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -sf "${health_url}" >/dev/null 2>&1; then
  echo "Osaurus health check failed at ${health_url}" >&2
  exit 1
fi

if [[ "${PULL_MODEL}" -eq 1 ]]; then
  "${pull_cmd[@]}"
fi

model_present="$(curl -sf "${models_url}" | python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print("\n".join(item.get("id","") for item in data))' | rg -x "${MODEL}" || true)"
if [[ -z "${model_present}" ]]; then
  echo "Model not listed by Osaurus: ${MODEL}" >&2
  echo "Pull it first with: osaurus pull ${MODEL}" >&2
  exit 1
fi

exec "${chat_cmd[@]}"
