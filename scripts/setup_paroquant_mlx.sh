#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
FETCH_VENDOR=1

usage() {
  cat <<'EOF'
Usage: scripts/setup_paroquant_mlx.sh [options]

Fetches pinned ParoQuant source and installs it into the project virtualenv.
This is the MLX text-only path; the repo's existing mlx/mlx-lm install is reused.

Options:
  --venv-path PATH      Override venv path (default: ./.venv)
  --python CMD          Python executable used to create the venv if missing
  --skip-fetch          Do not fetch/update vendor/paroquant before install
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
    --python)
      [[ $# -lt 2 ]] && { echo "Missing value for --python" >&2; exit 1; }
      PYTHON_CMD="$2"
      shift 2
      ;;
    --skip-fetch)
      FETCH_VENDOR=0
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

if [[ "${FETCH_VENDOR}" -eq 1 ]]; then
  "${ROOT_DIR}/scripts/fetch_vendor_sources.sh" --component paroquant
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${ROOT_DIR}/scripts/setup_env.sh" --venv-path "${VENV_DIR}" --python "${PYTHON_CMD}" --skip-smoke-test
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}/vendor/paroquant"
"${ROOT_DIR}/scripts/smoke_test.sh" --venv-path "${VENV_DIR}"

echo "ParoQuant MLX setup complete."
