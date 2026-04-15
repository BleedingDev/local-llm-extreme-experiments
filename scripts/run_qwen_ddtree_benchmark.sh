#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PINNED_MODEL="Qwen/Qwen3.5-4B"
PINNED_DRAFT_MODEL="z-lab/Qwen3.5-4B-DFlash"

VENV_DIR="${ROOT_DIR}/.venv"
MODEL="${DDTREE_MODEL:-${PINNED_MODEL}}"
DRAFT_MODEL="${DDTREE_DRAFT_MODEL:-${PINNED_DRAFT_MODEL}}"
DATASET="${DDTREE_DATASET:-gsm8k}"
MAX_SAMPLES="${DDTREE_MAX_SAMPLES:-8}"
MAX_NEW_TOKENS="${DDTREE_MAX_NEW_TOKENS:-256}"
TREE_BUDGET="${DDTREE_TREE_BUDGET:-16,32,64,128}"
BLOCK_SIZE="${DDTREE_BLOCK_SIZE:-}"
TEMPERATURE="${DDTREE_TEMPERATURE:-0.0}"
RUN_DIR="${DDTREE_RUN_DIR:-${ROOT_DIR}/artifacts/ddtree/runs}"
LOG_DIR="${DDTREE_LOG_DIR:-${ROOT_DIR}/artifacts/ddtree/logs}"
SAVE_PATH=""
LOG_PATH=""
DRY_RUN=0
EXTRA_ARGS=()

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29600}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_qwen_ddtree_benchmark.sh [options] [-- extra benchmark args]

Runs vendor/ddtree reference benchmark with the pinned Qwen3.5 + DFlash draft pair:
  target: Qwen/Qwen3.5-4B
  draft:  z-lab/Qwen3.5-4B-DFlash

The script performs hard prerequisite checks first:
  - PyTorch is installed with CUDA enabled
  - torch.cuda is available and has enough devices for NPROC_PER_NODE
  - NCCL backend is available (vendor/ddtree uses it via torchrun)
  - flash_attn is importable
  - model configs can be resolved and draft model_type is qwen3

Options:
  --venv-path PATH        Python virtualenv path (default: ./.venv)
  --model NAME            Override target model (default pinned Qwen3.5)
  --draft-model NAME      Override draft model (default pinned DFlash draft)
  --dataset NAME          Dataset name for vendor/ddtree benchmark.py
  --max-samples N         Max dataset samples (default: 8)
  --max-new-tokens N      Max new tokens (default: 256)
  --tree-budget CSV       Tree budgets (default: 16,32,64,128)
  --block-size N          Optional explicit block size
  --temperature FLOAT     Sampling temperature (default: 0.0)
  --run-dir PATH          Output .pt directory (default: artifacts/ddtree/runs)
  --log-dir PATH          Log directory (default: artifacts/ddtree/logs)
  --save-path PATH        Explicit .pt output path
  --log-path PATH         Explicit log output path
  --dry-run               Print command after checks and exit
  -h, --help              Show this help text

Environment:
  NPROC_PER_NODE          torchrun nproc per node (default: 1)
  MASTER_PORT             torchrun master port (default: 29600)
USAGE
}

slugify() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "$value"
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
    --max-new-tokens)
      [[ $# -lt 2 ]] && { echo "Missing value for --max-new-tokens" >&2; exit 1; }
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --tree-budget)
      [[ $# -lt 2 ]] && { echo "Missing value for --tree-budget" >&2; exit 1; }
      TREE_BUDGET="$2"
      shift 2
      ;;
    --block-size)
      [[ $# -lt 2 ]] && { echo "Missing value for --block-size" >&2; exit 1; }
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --temperature)
      [[ $# -lt 2 ]] && { echo "Missing value for --temperature" >&2; exit 1; }
      TEMPERATURE="$2"
      shift 2
      ;;
    --run-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --run-dir" >&2; exit 1; }
      RUN_DIR="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --log-dir" >&2; exit 1; }
      LOG_DIR="$2"
      shift 2
      ;;
    --save-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --save-path" >&2; exit 1; }
      SAVE_PATH="$2"
      shift 2
      ;;
    --log-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --log-path" >&2; exit 1; }
      LOG_PATH="$2"
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
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${VENV_DIR}/bin/python"
TORCHRUN_BIN="${VENV_DIR}/bin/torchrun"
DDTREE_BENCHMARK="${ROOT_DIR}/vendor/ddtree/benchmark.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}." >&2
  echo "Create env with scripts/setup_env.sh or pass --venv-path." >&2
  exit 1
fi

if [[ ! -f "${DDTREE_BENCHMARK}" ]]; then
  echo "DDTree benchmark entrypoint missing: ${DDTREE_BENCHMARK}" >&2
  echo "Run scripts/fetch_vendor_sources.sh --component ddtree" >&2
  exit 1
fi

if [[ "${MODEL}" != "${PINNED_MODEL}" || "${DRAFT_MODEL}" != "${PINNED_DRAFT_MODEL}" ]]; then
  echo "WARN: using non-pinned model pair:" >&2
  echo "  model=${MODEL}" >&2
  echo "  draft=${DRAFT_MODEL}" >&2
fi

for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--flash-attn" ]]; then
    echo "WARN: --flash-attn disables DDTree methods in vendor/ddtree/benchmark.py (DFlash-only run)." >&2
  fi
done

echo "Checking DDTree prerequisites for model=${MODEL} draft=${DRAFT_MODEL}"
if ! DDTREE_TARGET_MODEL="${MODEL}" DDTREE_DRAFT_MODEL="${DRAFT_MODEL}" NPROC_PER_NODE="${NPROC_PER_NODE}" "${PYTHON_BIN}" - <<'PY'
import os
import platform
import sys

errors = []
warnings = []

target_model = os.environ["DDTREE_TARGET_MODEL"]
draft_model = os.environ["DDTREE_DRAFT_MODEL"]
requested_nproc = int(os.environ.get("NPROC_PER_NODE", "1"))

print(f"Host platform: {platform.platform()}")
print(f"Python version: {sys.version.split()[0]}")

if sys.version_info >= (3, 13):
    warnings.append(
        f"Python {sys.version.split()[0]} may not have CUDA PyTorch wheels yet. "
        "Use Python 3.10-3.12 on CUDA hosts if install fails."
    )

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    device_count = torch.cuda.device_count() if cuda_available else 0
    print(f"torch.cuda.device_count(): {device_count}")

    if getattr(torch.version, "cuda", None) is None:
        errors.append("Installed PyTorch build does not include CUDA support.")

    if not cuda_available:
        errors.append("torch.cuda.is_available() is False.")

    try:
        import torch.distributed as dist

        if not dist.is_nccl_available():
            errors.append("torch.distributed NCCL backend is unavailable.")
    except Exception as exc:  # pragma: no cover - script path
        errors.append(f"Failed to query torch.distributed NCCL support: {exc}")

    if device_count < requested_nproc:
        errors.append(
            f"NPROC_PER_NODE={requested_nproc} but only {device_count} CUDA device(s) detected."
        )

except Exception as exc:  # pragma: no cover - script path
    errors.append(f"PyTorch import failed: {exc}")

try:
    import flash_attn  # noqa: F401
    print("flash_attn import: OK")
except Exception as exc:  # pragma: no cover - script path
    errors.append(f"flash_attn import failed: {exc}")

try:
    from transformers import AutoConfig

    target_cfg = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    draft_cfg = AutoConfig.from_pretrained(draft_model, trust_remote_code=True)

    target_type = getattr(target_cfg, "model_type", None)
    draft_type = getattr(draft_cfg, "model_type", None)

    print(f"Target model_type: {target_type}")
    print(f"Draft model_type: {draft_type}")

    if draft_type != "qwen3":
        errors.append(
            f"Draft model_type '{draft_type}' is incompatible with vendor/ddtree/model/dflash.py (expects qwen3)."
        )

    if target_type not in {"qwen3", "qwen3_5", "qwen3_5_text"}:
        warnings.append(
            f"Target model_type '{target_type}' is untested in vendor/ddtree benchmark path."
        )

except Exception as exc:  # pragma: no cover - script path
    errors.append(f"Model config probe failed: {exc}")

if warnings:
    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    sys.exit(1)

print("DDTree prerequisite checks passed.")
PY
then
  cat >&2 <<'EOF'
DDTree benchmark cannot run in the current environment.

vendor/ddtree requires a CUDA-enabled PyTorch stack with flash-attn and NCCL.
On Apple Silicon / MLX-only hosts, use scripts/run_dflash_mlx_benchmark.sh for local Qwen benchmarking,
and run this DDTree script on a separate CUDA-capable Linux machine.

See docs/qwen-ddtree-eval.md for a full CUDA setup and run recipe.
EOF
  exit 1
fi

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found at ${TORCHRUN_BIN}. Install PyTorch in ${VENV_DIR}." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

run_name="qwen35-ddtree-$(slugify "${DATASET}")-$(date +%Y%m%d-%H%M%S)"
if [[ -z "${SAVE_PATH}" ]]; then
  SAVE_PATH="${RUN_DIR}/${run_name}.pt"
fi
if [[ -z "${LOG_PATH}" ]]; then
  LOG_PATH="${LOG_DIR}/${run_name}.log"
fi
mkdir -p "$(dirname "${SAVE_PATH}")" "$(dirname "${LOG_PATH}")"

cmd=(
  "${TORCHRUN_BIN}"
  "--nproc_per_node=${NPROC_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${DDTREE_BENCHMARK}"
  --dataset "${DATASET}"
  --max-samples "${MAX_SAMPLES}"
  --model-name-or-path "${MODEL}"
  --draft-name-or-path "${DRAFT_MODEL}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --tree-budget "${TREE_BUDGET}"
  --temperature "${TEMPERATURE}"
  --save-path "${SAVE_PATH}"
)

if [[ -n "${BLOCK_SIZE}" ]]; then
  cmd+=(--block-size "${BLOCK_SIZE}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "DDTree benchmark log path: ${LOG_PATH}"
echo "DDTree benchmark save path: ${SAVE_PATH}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

(
  cd "${ROOT_DIR}"
  "${cmd[@]}" 2>&1 | tee "${LOG_PATH}"
)
