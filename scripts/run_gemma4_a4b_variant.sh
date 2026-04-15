#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_gemma4_a4b_variant.sh [options] --variant NAME [-- extra runner args]

Gemma 4 26B A4B runtime wrapper for benchmark variants.
Heavy model execution is opt-in in the Python runner; use --dry-run / --preflight for safe prep.

Options:
  --venv-path PATH      Virtualenv path (default: ./.venv)
  --variant NAME        Variant name (required)
  --dry-run             Print resolved execution plan only
  --preflight           Import/config checks only (no model load)
  -h, --help            Show this help text

Examples:
  scripts/run_gemma4_a4b_variant.sh --variant baseline --dry-run
  scripts/run_gemma4_a4b_variant.sh --variant triattention --preflight
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --venv-path" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --dry-run|--preflight|--variant|--model|--rotor-model|--draft-model|--prompt|--prompt-file|--max-tokens|--max-kv-size|--temperature|--top-p|--top-k|--seed|--kv-bits|--kv-group-size|--quantized-kv-start|--num-draft-tokens|--triattention-stats-path|--triattention-kv-budget|--triattention-divide-length|--output-json)
      EXTRA_ARGS+=("$1")
      if [[ "$1" == "--dry-run" || "$1" == "--preflight" ]]; then
        shift
      else
        [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; exit 1; }
        EXTRA_ARGS+=("$2")
        shift 2
      fi
      ;;
    --triattention-prefill-pin|--triattention-disable-trig|--triattention-disable-mlr|--offline-only)
      EXTRA_ARGS+=("$1")
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

PYTHONPATH_ENTRIES=()
if [[ -d "${ROOT_DIR}/vendor/triattention" ]]; then
  PYTHONPATH_ENTRIES+=("${ROOT_DIR}/vendor/triattention")
fi
if [[ -d "${ROOT_DIR}/vendor/turboquant-mlx" ]]; then
  PYTHONPATH_ENTRIES+=("${ROOT_DIR}/vendor/turboquant-mlx")
fi

if [[ "${#PYTHONPATH_ENTRIES[@]}" -gt 0 ]]; then
  local_path="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${local_path}:${PYTHONPATH}"
  else
    export PYTHONPATH="${local_path}"
  fi
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/gemma4_a4b_variant_runner.py" "${EXTRA_ARGS[@]}"
