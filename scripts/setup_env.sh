#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
RUN_SMOKE_TEST=1
STRICT_OPTIONAL=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_env.sh [options]

Creates/updates a local Python virtual environment and installs dependencies
without requiring global package installs.

Options:
  --venv-path PATH      Override venv path (default: ./.venv)
  --python CMD          Python executable to use (default: python3)
  --skip-smoke-test     Skip running scripts/smoke_test.sh
  --strict-optional     Fail when DDTree/DFlash are not installed
  -h, --help            Show this help text

Optional dependency install sources:
  DDTREE_INSTALL_SPEC   pip install spec/path for DDTree (explicit opt-in)
  DFLASH_INSTALL_SPEC   pip install spec/path for DFlash

If DFLASH_INSTALL_SPEC is not set, setup will try editable installs from:
  ./vendor/dflash, ./third_party/dflash, ./dflash

DDTree is not auto-installed from local checkout by default because upstream
targets CUDA/PyTorch stacks. To force local DDTree install, set:
  DDTREE_INSTALL_SPEC=./vendor/ddtree
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -lt 2 ]] && { echo "Missing value for --venv-path" >&2; exit 1; }
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -lt 2 ]] && { echo "Missing value for --python" >&2; exit 1; }
      PYTHON_CMD="$2"
      shift 2
      ;;
    --skip-smoke-test)
      RUN_SMOKE_TEST=0
      shift
      ;;
    --strict-optional)
      STRICT_OPTIONAL=1
      shift
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

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_CMD}" >&2
  exit 1
fi

if ! "${PYTHON_CMD}" -c "import venv" >/dev/null 2>&1; then
  echo "Python venv module is unavailable for ${PYTHON_CMD}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo "requirements.txt is missing in ${ROOT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${VENV_DIR}")"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"

install_optional_dependency() {
  local name="$1"
  local env_var_name="$2"
  shift 2
  local candidates=("$@")
  local spec="${!env_var_name:-}"

  if [[ -n "${spec}" ]]; then
    echo "Installing ${name} from ${env_var_name}"
    python -m pip install "${spec}"
    return 0
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/pyproject.toml" || -f "${candidate}/setup.py" ]]; then
      echo "Installing ${name} from local path: ${candidate}"
      python -m pip install -e "${candidate}"
      return 0
    fi
  done

  if [[ "${STRICT_OPTIONAL}" -eq 1 ]]; then
    echo "Missing ${name}. Set ${env_var_name} or add a local source checkout." >&2
    return 1
  fi

  echo "Skipping ${name} (no install source provided)."
}

install_optional_dependency "DDTree" "DDTREE_INSTALL_SPEC"

install_optional_dependency "DFlash" "DFLASH_INSTALL_SPEC" \
  "${ROOT_DIR}/vendor/dflash" \
  "${ROOT_DIR}/third_party/dflash" \
  "${ROOT_DIR}/dflash"

if [[ "${RUN_SMOKE_TEST}" -eq 1 ]]; then
  smoke_args=(--venv-path "${VENV_DIR}")
  if [[ "${STRICT_OPTIONAL}" -eq 1 ]]; then
    smoke_args+=(--strict-optional)
  fi
  "${ROOT_DIR}/scripts/smoke_test.sh" "${smoke_args[@]}"
fi

echo "Bootstrap complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
