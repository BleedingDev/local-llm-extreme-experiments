#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FETCH_SCRIPT="${ROOT_DIR}/scripts/fetch_vendor_sources.sh"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
VENDOR_DIR="${ROOT_DIR}/vendor"
REFRESH=0
RUN_VALIDATION=1
INSTALL_ROOT_REQUIREMENTS=1

usage() {
  cat <<'USAGE'
Usage: scripts/setup_turboquant_mlx.sh [options]

Creates/updates a local .venv for turboquant-mlx integration and wires vendor checkout
into the environment without global installs.

Options:
  --venv-path PATH            Virtualenv path (default: ./.venv)
  --python CMD                Python executable (default: python3)
  --vendor-dir PATH           Vendor directory (default: ./vendor)
  --refresh                   Refresh git metadata before pin checkout
  --skip-root-requirements    Skip installing root requirements.txt
  --skip-validation           Skip import validation step
  -h, --help                  Show this help text
USAGE
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
    --vendor-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --vendor-dir" >&2; exit 1; }
      VENDOR_DIR="$2"
      shift 2
      ;;
    --refresh)
      REFRESH=1
      shift
      ;;
    --skip-root-requirements)
      INSTALL_ROOT_REQUIREMENTS=0
      shift
      ;;
    --skip-validation)
      RUN_VALIDATION=0
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

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found in PATH." >&2
  exit 1
fi

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_CMD}" >&2
  exit 1
fi

if ! "${PYTHON_CMD}" -c "import venv" >/dev/null 2>&1; then
  echo "Python venv module is unavailable for ${PYTHON_CMD}" >&2
  exit 1
fi

if [[ ! -f "${FETCH_SCRIPT}" ]]; then
  echo "Missing fetch script: ${FETCH_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${VENV_DIR}")"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
fi

fetch_args=(--component turboquant-mlx --vendor-dir "${VENDOR_DIR}")
if [[ "${REFRESH}" -eq 1 ]]; then
  fetch_args+=(--refresh)
fi
bash "${FETCH_SCRIPT}" "${fetch_args[@]}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_ROOT_REQUIREMENTS}" -eq 1 ]]; then
  if [[ ! -f "${ROOT_DIR}/requirements.txt" ]]; then
    echo "requirements.txt not found in ${ROOT_DIR}" >&2
    exit 1
  fi
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

TURBOQUANT_DIR="${VENDOR_DIR}/turboquant-mlx"
if [[ ! -f "${TURBOQUANT_DIR}/requirements.txt" ]]; then
  echo "turboquant-mlx requirements not found in ${TURBOQUANT_DIR}" >&2
  exit 1
fi

python -m pip install -r "${TURBOQUANT_DIR}/requirements.txt"

SITE_PACKAGES="$(python - <<'PY'
import site
import sysconfig

for path in site.getsitepackages():
    if path.endswith("site-packages"):
        print(path)
        raise SystemExit(0)

fallback = sysconfig.get_paths().get("purelib")
if fallback:
    print(fallback)
    raise SystemExit(0)

raise SystemExit("Could not resolve site-packages path.")
PY
)"

PTH_FILE="${SITE_PACKAGES}/turboquant_mlx_local.pth"
printf '%s\n' "${TURBOQUANT_DIR}" > "${PTH_FILE}"

if [[ "${RUN_VALIDATION}" -eq 1 ]]; then
  PYTHONPATH="${TURBOQUANT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import importlib

import turboquant.patch as tq_patch
from benchmark_common import _STRATEGIES

for module_name in (
    "turboquant.cache_v2",
    "turboquant.cache_v3",
    "turboquant.attention_v2",
    "turboquant.attention_v3",
    "benchmark_common",
):
    importlib.import_module(module_name)

tq_patch.apply()

print("TurboQuant import validation passed.")
print(f"Registered TurboQuant strategies: {len(_STRATEGIES)}")
PY
fi

PINNED_COMMIT="$(git -C "${TURBOQUANT_DIR}" rev-parse HEAD)"
echo "TurboQuant MLX setup complete."
echo "Pinned source: ${TURBOQUANT_DIR} @ ${PINNED_COMMIT}"
echo "Local venv: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
