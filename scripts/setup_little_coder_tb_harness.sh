#!/usr/bin/env bash
set -euo pipefail

DEST="${LITTLE_CODER_REPO:-bench/vendor/little-coder}"

if [[ -d "${DEST}/.git" ]]; then
  git -C "${DEST}" pull --ff-only
else
  git clone --depth 1 https://github.com/itayinbarr/little-coder.git "${DEST}"
fi

npm install --prefix "${DEST}"
node "${DEST}/scripts/patch-pi.mjs" >/dev/null 2>&1 || true

echo "Little Coder benchmark harness ready at ${DEST}" >&2
