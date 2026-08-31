#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${QWEN36_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
DRAFT_MODEL="${QWEN36_DRAFT_MODEL:-z-lab/Qwen3.6-35B-A3B-DFlash}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_qwen36_dflash_mlx_benchmark.sh [options] [-- extra dflash.benchmark args]

Qwen 3.6-specific wrapper around scripts/run_dflash_mlx_benchmark.sh.

Options:
  --venv-path PATH      Override venv path (default: ./.venv)
  --model NAME          Target model (default: mlx-community/Qwen3.6-35B-A3B-4bit)
  --draft-model NAME    Draft model (default: z-lab/Qwen3.6-35B-A3B-DFlash)
  --dry-run             Print resolved command and exit
  -h, --help            Show this help text
EOF
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
    --draft-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --draft-model" >&2; exit 1; }
      DRAFT_MODEL="$2"
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

cmd=(
  "${ROOT_DIR}/scripts/run_dflash_mlx_benchmark.sh"
  --venv-path "${VENV_DIR}"
  --model "${MODEL}"
  --draft-model "${DRAFT_MODEL}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  cmd+=(--dry-run)
fi
cmd+=("${EXTRA_ARGS[@]}")

exec "${cmd[@]}"
