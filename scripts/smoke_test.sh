#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${ROOT_DIR}/.venv"
STRICT_OPTIONAL=0

usage() {
  cat <<'EOF'
Usage: scripts/smoke_test.sh [options]

Runs a minimal local validation for the bootstrap environment.

Options:
  --venv-path PATH      Override venv path (default: ./.venv)
  --strict-optional     Fail if DDTree/DFlash imports are missing
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

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtualenv Python not found at ${PYTHON_BIN}. Run scripts/setup_env.sh first." >&2
  exit 1
fi

STRICT_OPTIONAL="${STRICT_OPTIONAL}" "${PYTHON_BIN}" - <<'PY'
import importlib
import os
import platform
import sys

required = {
    "mlx": "mlx",
    "mlx_lm": "mlx-lm",
    "huggingface_hub": "huggingface_hub",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
}
optional = {
    "ddtree": "DDTree",
    "dflash": "DFlash",
}
strict_optional = os.environ.get("STRICT_OPTIONAL", "0") == "1"

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

failures = []
for module_name, package_name in required.items():
    try:
        importlib.import_module(module_name)
        print(f"OK required import: {package_name}")
    except Exception as exc:  # pragma: no cover - script path
        failures.append(f"missing required dependency {package_name}: {exc}")

optional_missing = []
for module_name, package_name in optional.items():
    try:
        importlib.import_module(module_name)
        print(f"OK optional import: {package_name}")
    except Exception:
        optional_missing.append(package_name)

if optional_missing:
    message = "missing optional dependencies: " + ", ".join(optional_missing)
    if strict_optional:
        failures.append(message)
    else:
        print(f"WARN {message}")

if failures:
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    sys.exit(1)

print("Smoke test passed.")
PY
