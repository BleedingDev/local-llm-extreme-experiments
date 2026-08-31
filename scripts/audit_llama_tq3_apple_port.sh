#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/audit_llama_tq3_apple_port.sh /path/to/llama.cpp-tq3

Scans a local llama.cpp-tq3 checkout and reports where TQ3 support is still
missing from the Metal backend.
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

ROOT="$1"

if [[ ! -d "${ROOT}" ]]; then
  echo "Missing checkout: ${ROOT}" >&2
  exit 1
fi

require_file() {
  local path="$1"
  if [[ ! -f "${ROOT}/${path}" ]]; then
    echo "Missing expected file: ${ROOT}/${path}" >&2
    exit 1
  fi
}

require_file "ggml/src/ggml-metal/ggml-metal-device.cpp"
require_file "ggml/src/ggml-metal/ggml-metal-device.m"
require_file "ggml/src/ggml-metal/ggml-metal.metal"

report_section() {
  printf '\n== %s ==\n' "$1"
}

report_count() {
  local label="$1"
  local pattern="$2"
  local file="$3"
  local count
  count="$(rg -c "${pattern}" "${ROOT}/${file}" || true)"
  printf '%-28s %s\n' "${label}" "${count}"
}

report_section "Metal Pipeline Switches"
report_count "mul_mv TQ3 cases" "GGML_TYPE_TQ3_" "ggml/src/ggml-metal/ggml-metal-device.cpp"
report_count "mul_mv_id TQ3 cases" "GGML_TYPE_TQ3_" "ggml/src/ggml-metal/ggml-metal-device.cpp"

report_section "Metal Capability Gates"
report_count "device.m TQ3 mentions" "GGML_TYPE_TQ3_" "ggml/src/ggml-metal/ggml-metal-device.m"

report_section "Kernel Instantiations"
report_count "mul_mv tq3 kernels" "kernel_mul_mv_.*tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "mul_mm tq3 kernels" "kernel_mul_mm_.*tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "mul_mm_id tq3 kernels" "kernel_mul_mm_id_.*tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "get_rows tq3 kernels" "kernel_get_rows_tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "cpy tq3 kernels" "kernel_cpy_.*tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "set_rows tq3 kernels" "kernel_set_rows_tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "flash_attn tq3 kernels" "kernel_flash_attn_ext.*tq3" "ggml/src/ggml-metal/ggml-metal.metal"

report_section "Helper Symbols"
report_count "tq3 dequant helpers" "dequantize_tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "tq3 quant helpers" "quantize_tq3" "ggml/src/ggml-metal/ggml-metal.metal"
report_count "tq3 unpack helpers" "tq3_.*unpack|tq3_.*decode|tq3_.*sign" "ggml/src/ggml-metal/ggml-metal.metal"

report_section "Raw Matches"
rg -n "GGML_TYPE_TQ3_|kernel_.*tq3|dequantize_tq3|quantize_tq3" \
  "${ROOT}/ggml/src/ggml-metal/ggml-metal-device.cpp" \
  "${ROOT}/ggml/src/ggml-metal/ggml-metal-device.m" \
  "${ROOT}/ggml/src/ggml-metal/ggml-metal.metal" || true
