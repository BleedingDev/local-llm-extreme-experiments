#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN=""
PROTO_SCRIPT="${ROOT_DIR}/scripts/run_qwen_ddtree_mlx_prototype.py"

MODEL="${DDTREE_MLX_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_MODEL="${DDTREE_MLX_DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}"
PROMPT="${DDTREE_MLX_PROMPT:-Write one short sentence confirming this DDTree-style MLX prototype run completed.}"
TREE_BUDGET="${DDTREE_MLX_TREE_BUDGET:-128}"
DEPTH="${DDTREE_MLX_DEPTH:-1}"
MAX_NEW_TOKENS="${DDTREE_MLX_MAX_NEW_TOKENS:-64}"
MAX_KV_SIZE="${DDTREE_MLX_MAX_KV_SIZE:-1024}"
TEMPERATURE="${DDTREE_MLX_TEMPERATURE:-0.0}"
ARTIFACTS_DIR="${DDTREE_MLX_ARTIFACTS_DIR:-}"
RUN_NAME="${DDTREE_MLX_RUN_NAME:-}"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/run_qwen_ddtree_mlx_prototype.sh [options] [-- extra python args]

Runs scripts/run_qwen_ddtree_mlx_prototype.py with Qwen3.5 + DFlash defaults.

Options:
  --venv-path PATH        Python virtualenv path (default: ./.venv)
  --model NAME            Target model (default: Qwen/Qwen3.5-4B)
  --draft-model NAME      Draft model (default: z-lab/Qwen3.5-4B-DFlash)
  --prompt TEXT           Prompt text override
  --tree-budget N         Candidate budget per round (default: 128)
  --depth N               Tree expansion depth (default: 1)
  --max-new-tokens N      Max generated tokens (default: 64)
  --max-kv-size N         Context/KV cap (default: 1024)
  --temperature FLOAT     Sampling temperature (default: 0.0)
  --artifacts-dir PATH    Artifacts root override
  --run-name NAME         Run directory name override
  -h, --help              Show this help text

Environment overrides:
  DDTREE_MLX_MODEL
  DDTREE_MLX_DRAFT_MODEL
  DDTREE_MLX_PROMPT
  DDTREE_MLX_TREE_BUDGET
  DDTREE_MLX_DEPTH
  DDTREE_MLX_MAX_NEW_TOKENS
  DDTREE_MLX_MAX_KV_SIZE
  DDTREE_MLX_TEMPERATURE
  DDTREE_MLX_ARTIFACTS_DIR
  DDTREE_MLX_RUN_NAME
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
    --prompt)
      [[ $# -lt 2 ]] && { echo "Missing value for --prompt" >&2; exit 1; }
      PROMPT="$2"
      shift 2
      ;;
    --tree-budget)
      [[ $# -lt 2 ]] && { echo "Missing value for --tree-budget" >&2; exit 1; }
      TREE_BUDGET="$2"
      shift 2
      ;;
    --depth)
      [[ $# -lt 2 ]] && { echo "Missing value for --depth" >&2; exit 1; }
      DEPTH="$2"
      shift 2
      ;;
    --max-new-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-new-tokens" >&2; exit 1; }
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-kv-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-kv-size" >&2; exit 1; }
      MAX_KV_SIZE="$2"
      shift 2
      ;;
    --temperature)
      [[ $# -lt 2 ]] && { echo "Missing value for --temperature" >&2; exit 1; }
      TEMPERATURE="$2"
      shift 2
      ;;
    --artifacts-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --artifacts-dir" >&2; exit 1; }
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --run-name)
      [[ $# -lt 2 ]] && { echo "Missing value for --run-name" >&2; exit 1; }
      RUN_NAME="$2"
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
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

if [[ ! -f "${PROTO_SCRIPT}" ]]; then
  echo "Prototype script missing: ${PROTO_SCRIPT}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import mlx, mlx_lm, dflash" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Required dependencies are missing in the selected virtualenv.
Run:
  scripts/setup_env.sh
or install DFlash explicitly:
  DFLASH_INSTALL_SPEC=vendor/dflash scripts/setup_env.sh --skip-smoke-test
EOF
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${PROTO_SCRIPT}"
  --model "${MODEL}"
  --draft-model "${DRAFT_MODEL}"
  --prompt "${PROMPT}"
  --tree-budget "${TREE_BUDGET}"
  --depth "${DEPTH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-kv-size "${MAX_KV_SIZE}"
  --temperature "${TEMPERATURE}"
)

if [[ -n "${ARTIFACTS_DIR}" ]]; then
  cmd+=(--artifacts-dir "${ARTIFACTS_DIR}")
fi

if [[ -n "${RUN_NAME}" ]]; then
  cmd+=(--run-name "${RUN_NAME}")
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

exec "${cmd[@]}"
