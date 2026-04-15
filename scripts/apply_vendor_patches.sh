#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENDOR_DIR="${ROOT_DIR}/vendor"
PATCHES_DIR="${ROOT_DIR}/patches"
COMPONENT="all"

usage() {
  cat <<'USAGE'
Usage: scripts/apply_vendor_patches.sh [options]

Applies local reproducibility patches on top of pinned vendor sources.

Options:
  --component NAME   all | dflash | ddtree | turboquant-mlx | triattention (default: all)
  --vendor-dir PATH  Vendor directory (default: ./vendor)
  --patches-dir PATH Patches directory (default: ./patches)
  -h, --help         Show this help text
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
    --patches-dir)
      [[ $# -lt 2 ]] && { echo "Missing value for --patches-dir" >&2; exit 1; }
      PATCHES_DIR="$2"
      shift 2
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

apply_patch_file() {
  local repo_dir="$1"
  local patch_file="$2"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "Missing git repo: ${repo_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${patch_file}" ]]; then
    echo "Missing patch file: ${patch_file}" >&2
    exit 1
  fi

  if git -C "${repo_dir}" apply --check "${patch_file}" >/dev/null 2>&1; then
    git -C "${repo_dir}" apply "${patch_file}"
    echo "Applied $(basename "${patch_file}") to ${repo_dir}"
    return 0
  fi

  if git -C "${repo_dir}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "Already applied $(basename "${patch_file}") in ${repo_dir}"
    return 0
  fi

  echo "Patch cannot be applied cleanly: ${patch_file}" >&2
  exit 1
}

case "${COMPONENT}" in
  all|dflash)
    apply_patch_file "${VENDOR_DIR}/dflash" "${PATCHES_DIR}/dflash-triattention-mlx.patch"
    ;;
  ddtree|turboquant-mlx|turboquant_mlx|triattention)
    echo "No local patches defined for component: ${COMPONENT}"
    ;;
  *)
    echo "Unsupported component: ${COMPONENT}" >&2
    usage
    exit 1
    ;;
esac

echo "Vendor patch application complete."
