#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_SCRIPT="${ROOT_DIR}/scripts/benchmark_supergemma_mlx.sh"

MODEL="${QWEN_MODEL:-Qwen/Qwen3.5-4B}"
MODE="${1:-both}" # throughput | probe | both

if [[ ! -x "${BENCH_SCRIPT}" && ! -f "${BENCH_SCRIPT}" ]]; then
  echo "Missing benchmark script: ${BENCH_SCRIPT}" >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
Usage: scripts/run_qwen_mlx_kv_sweep.sh [mode]

Modes:
  throughput   Run kv throughput matrix only.
  probe        Run kv probe only.
  both         Run both throughput matrix and kv probe (default).

Environment overrides:
  QWEN_MODEL              Model id/path (default: Qwen/Qwen3.5-4B)
  QWEN_SWEEP_PROMPT       Prompt for throughput run
  QWEN_PROBE_PROMPT       Prompt for probe run
  QWEN_SWEEP_MAX_TOKENS   Throughput max tokens (default: 64)
  QWEN_PROBE_MAX_TOKENS   Probe max tokens (default: 32)
USAGE
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

SWEEP_PROMPT="${QWEN_SWEEP_PROMPT:-Napiš stručné shrnutí optimalizace KV cache.}"
PROBE_PROMPT="${QWEN_PROBE_PROMPT:-Ahoj}"
SWEEP_MAX_TOKENS="${QWEN_SWEEP_MAX_TOKENS:-64}"
PROBE_MAX_TOKENS="${QWEN_PROBE_MAX_TOKENS:-32}"

case "${MODE}" in
  throughput)
    scripts/benchmark_supergemma_mlx.sh \
      --model "${MODEL}" \
      --mode benchmark \
      --prompt "${SWEEP_PROMPT}" \
      --max-tokens "${SWEEP_MAX_TOKENS}" \
      --max-kv-size 512 \
      --max-kv-size 1024 \
      --max-kv-size 2048 \
      --max-kv-size 4096 \
      --repeats 2
    ;;
  probe)
    scripts/benchmark_supergemma_mlx.sh \
      --model "${MODEL}" \
      --mode probe \
      --prompt "${PROBE_PROMPT}" \
      --probe-start-kv 1024 \
      --probe-max-kv 32768 \
      --probe-max-tokens "${PROBE_MAX_TOKENS}"
    ;;
  both)
    scripts/benchmark_supergemma_mlx.sh \
      --model "${MODEL}" \
      --mode benchmark \
      --prompt "${SWEEP_PROMPT}" \
      --max-tokens "${SWEEP_MAX_TOKENS}" \
      --max-kv-size 512 \
      --max-kv-size 1024 \
      --max-kv-size 2048 \
      --max-kv-size 4096 \
      --repeats 2
    scripts/benchmark_supergemma_mlx.sh \
      --model "${MODEL}" \
      --mode probe \
      --prompt "${PROBE_PROMPT}" \
      --probe-start-kv 1024 \
      --probe-max-kv 32768 \
      --probe-max-tokens "${PROBE_MAX_TOKENS}"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
