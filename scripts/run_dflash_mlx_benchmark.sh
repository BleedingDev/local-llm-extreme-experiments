#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${DFLASH_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_MODEL="${DFLASH_DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}"
DATASET="${DFLASH_DATASET:-gsm8k}"
MAX_SAMPLES="${DFLASH_MAX_SAMPLES:-16}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_dflash_mlx_benchmark.sh [options] [-- extra dflash.benchmark args]

Runs the DFlash benchmark in MLX mode (Qwen-focused supported path).
This path is separate from SuperGemma runtime.

Options:
  --venv-path PATH      Override venv path (default: ./.venv)
  --model NAME          Target model (default: Qwen/Qwen3.5-4B)
  --draft-model NAME    DFlash draft model (default: z-lab/Qwen3.5-4B-DFlash)
  --dataset NAME        Dataset name (default: gsm8k)
  --max-samples N       Sample limit (default: 16)
  --dry-run             Print command and exit
  -h, --help            Show this help text

TriAttention merge mode:
  Pass TriAttention flags after --, for example:
    -- --triattention-enable --triattention-kv-budget 2048 --triattention-divide-length 8
  Requires vendor/triattention (run: scripts/fetch_vendor_sources.sh --component triattention).
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
    --draft-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --draft-model" >&2; exit 1; }
      DRAFT_MODEL="$2"
      shift 2
      ;;
    --dataset)
      [[ $# -lt 2 ]] && { echo "Missing value for --dataset" >&2; exit 1; }
      DATASET="$2"
      shift 2
      ;;
    --max-samples)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-samples" >&2; exit 1; }
      MAX_SAMPLES="$2"
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

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  -m dflash.benchmark
  --backend mlx
  --model "${MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --dataset "${DATASET}"
  --max-samples "${MAX_SAMPLES}"
)
cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if ! "${PYTHON_BIN}" -c "import dflash" >/dev/null 2>&1; then
  echo "DFlash is not installed in ${VENV_DIR}." >&2
  echo "Fetch and install locally with:" >&2
  echo "  scripts/fetch_vendor_sources.sh --component dflash" >&2
  echo "  DFLASH_INSTALL_SPEC=${ROOT_DIR}/vendor/dflash scripts/setup_env.sh --skip-smoke-test" >&2
  exit 1
fi

if [[ -d "${ROOT_DIR}/vendor/triattention" ]]; then
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${ROOT_DIR}/vendor/triattention:${PYTHONPATH}"
  else
    export PYTHONPATH="${ROOT_DIR}/vendor/triattention"
  fi
fi

exec "${cmd[@]}"
