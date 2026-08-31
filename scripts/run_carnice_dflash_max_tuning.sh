#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/scripts/run_dflash_mlx_benchmark.sh"

MODEL="${CARNICE_MODEL:-jason-schulz/Carnice-9b-MLX}"
MODEL_REVISION="${CARNICE_MODEL_REVISION:-}"
DRAFT_MODELS=()
DRAFT_REVISION="${CARNICE_DRAFT_REVISION:-}"
DATASET="${CARNICE_DATASET:-gsm8k}"
MAX_SAMPLES="${CARNICE_MAX_SAMPLES:-3}"
MAX_NEW_TOKENS="${CARNICE_MAX_NEW_TOKENS:-64}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/run_carnice_dflash_max_tuning.sh [options]

Runs a focused Carnice 9B MLX + DFlash tuning sweep and reports best-performing profile.

Profiles evaluated per draft model:
  baseline
  kv4
  kv4_chunk256
  tqlean
  tqlean_chunk256

Options:
  --model NAME             Target model id/path (default: jason-schulz/Carnice-9b-MLX)
  --model-revision REF     Optional target model revision
  --draft-model NAME       Draft model id/path (repeatable; default: 9B and 4B DFlash drafts)
  --draft-revision REF     Optional draft model revision
  --dataset NAME           Dataset name for dflash.benchmark (default: gsm8k)
  --max-samples N          Max samples per run (default: 3)
  --max-new-tokens N       Max new tokens per sample (default: 64)
  --dry-run                Print planned commands only
  -h, --help               Show this help text
USAGE
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  is_non_negative_int "$1" && [[ "$1" -gt 0 ]]
}

append_profile_args() {
  local profile="$1"
  local -n out_ref="$2"
  case "${profile}" in
    baseline)
      ;;
    kv4)
      out_ref+=(--cache-optimization kv-quant --kv-bits 4 --kv-group-size 64 --quantized-kv-start 0)
      ;;
    kv4_chunk256)
      out_ref+=(--cache-optimization kv-quant --kv-bits 4 --kv-group-size 64 --quantized-kv-start 0 --prefill-chunk-size 256)
      ;;
    tqlean)
      out_ref+=(--cache-optimization turboquant --turboquant-strategy tqv2_4bit_lean --quantized-kv-start 0)
      ;;
    tqlean_chunk256)
      out_ref+=(--cache-optimization turboquant --turboquant-strategy tqv2_4bit_lean --quantized-kv-start 0 --prefill-chunk-size 256)
      ;;
    *)
      echo "Unsupported profile: ${profile}" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -lt 2 ]] && { echo "Missing value for --model" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --model-revision)
      [[ $# -lt 2 ]] && { echo "Missing value for --model-revision" >&2; exit 1; }
      MODEL_REVISION="$2"
      shift 2
      ;;
    --draft-model)
      [[ $# -lt 2 ]] && { echo "Missing value for --draft-model" >&2; exit 1; }
      DRAFT_MODELS+=("$2")
      shift 2
      ;;
    --draft-revision)
      [[ $# -lt 2 ]] && { echo "Missing value for --draft-revision" >&2; exit 1; }
      DRAFT_REVISION="$2"
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
    --dry-run)
      DRY_RUN=1
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

if [[ "${#DRAFT_MODELS[@]}" -eq 0 ]]; then
  DRAFT_MODELS=("z-lab/Qwen3.5-9B-DFlash" "z-lab/Qwen3.5-4B-DFlash")
fi

if ! is_positive_int "${MAX_SAMPLES}"; then
  echo "--max-samples must be a positive integer: ${MAX_SAMPLES}" >&2
  exit 1
fi

if ! is_positive_int "${MAX_NEW_TOKENS}"; then
  echo "--max-new-tokens must be a positive integer: ${MAX_NEW_TOKENS}" >&2
  exit 1
fi

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "Missing executable benchmark wrapper: ${RUN_SCRIPT}" >&2
  exit 1
fi

OUT_DIR="${ROOT_DIR}/artifacts/benchmarks/carnice-max-tune-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUT_DIR}"
RESULTS_CSV="${OUT_DIR}/results.csv"
SUMMARY_JSON="${OUT_DIR}/summary.json"
SUMMARY_TXT="${OUT_DIR}/summary.txt"

printf "draft_model,profile,baseline_tps,dflash_tps,speedup,avg_acceptance,log_file,status\n" > "${RESULTS_CSV}"

profiles=(baseline kv4 kv4_chunk256 tqlean tqlean_chunk256)

run_case() {
  local draft_model="$1"
  local profile="$2"
  local -a cmd=(
    "${RUN_SCRIPT}"
    --model "${MODEL}"
    --draft-model "${draft_model}"
    --dataset "${DATASET}"
    --max-samples "${MAX_SAMPLES}"
  )
  if [[ -n "${MODEL_REVISION}" ]]; then
    cmd+=(--model-revision "${MODEL_REVISION}")
  fi
  if [[ -n "${DRAFT_REVISION}" ]]; then
    cmd+=(--draft-revision "${DRAFT_REVISION}")
  fi

  local -a extra_args=(--max-new-tokens "${MAX_NEW_TOKENS}")
  append_profile_args "${profile}" extra_args
  cmd+=(-- "${extra_args[@]}")

  local safe_draft profile_log log_file
  safe_draft="$(echo "${draft_model}" | tr '/:@' '___')"
  profile_log="${safe_draft}_${profile}"
  log_file="${OUT_DIR}/${profile_log}.log"

  {
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } | tee "${log_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf "%s,%s,,,,,%s,dry-run\n" "${draft_model}" "${profile}" "${log_file}" >> "${RESULTS_CSV}"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  local exit_code=$?
  set -e
  if [[ "${exit_code}" -ne 0 ]]; then
    printf "%s,%s,,,,,%s,failed\n" "${draft_model}" "${profile}" "${log_file}" >> "${RESULTS_CSV}"
    return 0
  fi

  LOG_FILE="${log_file}" python3 - <<'PY' >> "${RESULTS_CSV}"
import os
import re
from pathlib import Path

log_path = Path(os.environ["LOG_FILE"])
text = log_path.read_text(encoding="utf-8", errors="replace")

def parse(pattern: str):
    match = re.search(pattern, text)
    return match.group(1) if match else ""

baseline = parse(r"Baseline throughput:\s+([0-9.]+)\s+tok/s")
dflash = parse(r"DFlash throughput:\s+([0-9.]+)\s+tok/s")
speedup = parse(r"Decoding speedup:\s+([0-9.]+)")
acceptance = parse(r"Average Acceptance length:\s+([0-9.]+)")

draft_model = os.environ["DRAFT_MODEL"]
profile = os.environ["PROFILE"]
status = "ok" if baseline and dflash and speedup else "parse-missing"
print(f"{draft_model},{profile},{baseline},{dflash},{speedup},{acceptance},{log_path},{status}")
PY
}

for draft_model in "${DRAFT_MODELS[@]}"; do
  for profile in "${profiles[@]}"; do
    echo "== Running draft=${draft_model} profile=${profile} =="
    DRAFT_MODEL="${draft_model}" PROFILE="${profile}" run_case "${draft_model}" "${profile}"
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run complete."
  echo "Artifacts: ${OUT_DIR}"
  exit 0
fi

RESULTS_CSV_PATH="${RESULTS_CSV}" SUMMARY_JSON_PATH="${SUMMARY_JSON}" SUMMARY_TXT_PATH="${SUMMARY_TXT}" python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

results_path = Path(os.environ["RESULTS_CSV_PATH"])
summary_json_path = Path(os.environ["SUMMARY_JSON_PATH"])
summary_txt_path = Path(os.environ["SUMMARY_TXT_PATH"])

rows = []
for row in csv.DictReader(results_path.open()):
    if row["status"] != "ok":
        continue
    row["baseline_tps"] = float(row["baseline_tps"])
    row["dflash_tps"] = float(row["dflash_tps"])
    row["speedup"] = float(row["speedup"])
    row["avg_acceptance"] = float(row["avg_acceptance"]) if row["avg_acceptance"] else 0.0
    rows.append(row)

payload = {
    "status": "ok" if rows else "no-successful-runs",
    "successful_runs": len(rows),
    "best_speedup": max(rows, key=lambda r: r["speedup"]) if rows else None,
    "best_dflash_tps": max(rows, key=lambda r: r["dflash_tps"]) if rows else None,
}
summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

with summary_txt_path.open("w") as f:
    if not rows:
        f.write("No successful runs.\n")
    else:
        best_speedup = payload["best_speedup"]
        best_tps = payload["best_dflash_tps"]
        f.write(
            "Best speedup profile:\n"
            f"  draft={best_speedup['draft_model']}\n"
            f"  profile={best_speedup['profile']}\n"
            f"  speedup={best_speedup['speedup']:.3f}x\n"
            f"  baseline_tps={best_speedup['baseline_tps']:.3f}\n"
            f"  dflash_tps={best_speedup['dflash_tps']:.3f}\n"
            f"  avg_acceptance={best_speedup['avg_acceptance']:.3f}\n\n"
        )
        f.write(
            "Best absolute DFlash throughput:\n"
            f"  draft={best_tps['draft_model']}\n"
            f"  profile={best_tps['profile']}\n"
            f"  dflash_tps={best_tps['dflash_tps']:.3f}\n"
            f"  speedup={best_tps['speedup']:.3f}x\n"
        )
print(f"Wrote {summary_json_path}")
print(f"Wrote {summary_txt_path}")
PY

echo "Tuning artifacts: ${OUT_DIR}"
