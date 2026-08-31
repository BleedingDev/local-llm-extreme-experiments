#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${QWEN36_MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
DRAFT_MODEL="${QWEN36_DRAFT_MODEL:-z-lab/Qwen3.6-35B-A3B-DFlash}"
PARO_MODEL="${QWEN36_PARO_MODEL:-}"
MODES="${QWEN36_COMBO_MODES:-dflash,paro,both}"
PROMPT="${QWEN36_COMBO_PROMPT:-Explain in one short paragraph how speculative decoding and quantization can complement each other on Apple Silicon.}"
MAX_NEW_TOKENS="${QWEN36_COMBO_MAX_NEW_TOKENS:-64}"
REPEATS="${QWEN36_COMBO_REPEATS:-1}"
MAX_KV_SIZE="${QWEN36_COMBO_MAX_KV_SIZE:-2048}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_qwen36_mlx_combo_benchmark.sh [options] [-- extra benchmark args]

Benchmarks MLX Qwen-family combinations:
  - dflash: target + DFlash draft
  - paro: ParoQuant target only
  - both: ParoQuant target + DFlash draft

Important:
  A public Qwen3.6-35B-A3B ParoQuant checkpoint was not found via the Hugging Face
  API on 2026-04-21, so --paro-model is required for 'paro' or 'both' modes.

Options:
  --venv-path PATH      Override venv path (default: ./.venv)
  --model NAME          Baseline/DFlash target model (default: mlx-community/Qwen3.6-35B-A3B-4bit)
  --draft-model NAME    DFlash draft model (default: z-lab/Qwen3.6-35B-A3B-DFlash)
  --paro-model NAME     ParoQuant target model for 'paro' and 'both' modes
  --modes LIST          Comma-separated subset of baseline,dflash,paro,both
  --prompt TEXT         Prompt text
  --max-new-tokens N    Generation token limit (default: 64)
  --repeats N           Number of repeats per mode (default: 1)
  --max-kv-size N       KV cache cap (default: 2048)
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
    --paro-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --paro-model" >&2; exit 1; }
      PARO_MODEL="$2"
      shift 2
      ;;
    --modes)
      [[ $# -lt 2 ]] && { echo "Missing value for --modes" >&2; exit 1; }
      MODES="$2"
      shift 2
      ;;
    --prompt)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt" >&2; exit 1; }
      PROMPT="$2"
      shift 2
      ;;
    --max-new-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-new-tokens" >&2; exit 1; }
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --repeats)
      [[ $# -lt 2 ]] && { echo "Missing value for --repeats" >&2; exit 1; }
      REPEATS="$2"
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      MAX_KV_SIZE="$2"
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

case ",${MODES}," in
  *,paro,*|*,both,*)
    if [[ -z "${PARO_MODEL}" ]]; then
      echo "--paro-model is required when --modes includes 'paro' or 'both'." >&2
      echo "No public Qwen3.6-35B-A3B ParoQuant checkpoint was found via the Hugging Face API on 2026-04-21." >&2
      exit 1
    fi
    ;;
esac

cmd=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/benchmark_qwen_mlx_combo.py"
  --model "${MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --modes "${MODES}"
  --prompt "${PROMPT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --repeats "${REPEATS}"
  --max-kv-size "${MAX_KV_SIZE}"
)
if [[ -n "${PARO_MODEL}" ]]; then
  cmd+=(--paro-model "${PARO_MODEL}")
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  cmd+=(--dry-run)
fi
cmd+=("${EXTRA_ARGS[@]}")

exec "${cmd[@]}"
