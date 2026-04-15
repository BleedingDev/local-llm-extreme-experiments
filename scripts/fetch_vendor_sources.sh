#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=./pinned_sources.sh
source "${SCRIPT_DIR}/pinned_sources.sh"

COMPONENT="all"
VENDOR_DIR="${ROOT_DIR}/vendor"
REFRESH=0
APPLY_LOCAL_PATCHES=1

usage() {
  cat <<'USAGE'
Usage: scripts/fetch_vendor_sources.sh [options]

Clones/upgrades pinned upstream sources into local vendor/ directories.
No global installs are performed.

Options:
  --component NAME     all | dflash | ddtree | turboquant-mlx | triattention (default: all)
  --vendor-dir PATH    Destination directory (default: ./vendor)
  --refresh            Always fetch latest remote metadata before checkout
  --skip-local-patches Do not apply local patches from ./patches
  -h, --help           Show this help text
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --component)
      [[ $# -lt 2 ]] && { echo "Missing value for --component" >&2; exit 1; }
      COMPONENT="$2"
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
    --skip-local-patches)
      APPLY_LOCAL_PATCHES=0
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

mkdir -p "${VENDOR_DIR}"

checkout_ref() {
  local name="$1"
  local repo_url="$2"
  local commit="$3"
  local destination="$4"

  if [[ -d "${destination}/.git" ]]; then
    local origin_url
    origin_url="$(git -C "${destination}" remote get-url origin)"
    if [[ "${origin_url}" != "${repo_url}" ]]; then
      echo "Refusing to reuse ${destination}: origin is ${origin_url}, expected ${repo_url}" >&2
      exit 1
    fi

    if [[ "${REFRESH}" -eq 1 ]]; then
      git -C "${destination}" fetch --quiet --tags origin
    fi
  else
    if [[ -e "${destination}" ]]; then
      echo "Path exists and is not a git clone: ${destination}" >&2
      exit 1
    fi

    echo "Cloning ${name} into ${destination}"
    git clone --quiet "${repo_url}" "${destination}"
  fi

  if ! git -C "${destination}" cat-file -e "${commit}"^"{commit}" 2>/dev/null; then
    git -C "${destination}" fetch --quiet origin "${commit}"
  fi

  git -C "${destination}" checkout --quiet --detach "${commit}"

  local actual_commit
  actual_commit="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${commit}" ]]; then
    echo "Failed to pin ${name}: got ${actual_commit}, expected ${commit}" >&2
    exit 1
  fi

  echo "OK ${name}: ${actual_commit}"
}

case "${COMPONENT}" in
  all)
    checkout_ref "dflash" "${DFLASH_REPO_URL}" "${DFLASH_COMMIT}" "${VENDOR_DIR}/dflash"
    checkout_ref "ddtree" "${DDTREE_REPO_URL}" "${DDTREE_COMMIT}" "${VENDOR_DIR}/ddtree"
    checkout_ref "turboquant-mlx" "${TURBOQUANT_MLX_REPO_URL}" "${TURBOQUANT_MLX_COMMIT}" "${VENDOR_DIR}/turboquant-mlx"
    checkout_ref "triattention" "${TRIATTENTION_REPO_URL}" "${TRIATTENTION_COMMIT}" "${VENDOR_DIR}/triattention"
    ;;
  dflash)
    checkout_ref "dflash" "${DFLASH_REPO_URL}" "${DFLASH_COMMIT}" "${VENDOR_DIR}/dflash"
    ;;
  ddtree)
    checkout_ref "ddtree" "${DDTREE_REPO_URL}" "${DDTREE_COMMIT}" "${VENDOR_DIR}/ddtree"
    ;;
  turboquant-mlx|turboquant_mlx)
    checkout_ref "turboquant-mlx" "${TURBOQUANT_MLX_REPO_URL}" "${TURBOQUANT_MLX_COMMIT}" "${VENDOR_DIR}/turboquant-mlx"
    ;;
  triattention)
    checkout_ref "triattention" "${TRIATTENTION_REPO_URL}" "${TRIATTENTION_COMMIT}" "${VENDOR_DIR}/triattention"
    ;;
  *)
    echo "Unsupported component: ${COMPONENT}" >&2
    usage
    exit 1
    ;;
esac

if [[ "${APPLY_LOCAL_PATCHES}" -eq 1 ]]; then
  "${SCRIPT_DIR}/apply_vendor_patches.sh" --component "${COMPONENT}" --vendor-dir "${VENDOR_DIR}"
fi

echo "Pinned source fetch complete."
