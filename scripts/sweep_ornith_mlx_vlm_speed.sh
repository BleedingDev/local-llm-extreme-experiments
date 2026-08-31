#!/usr/bin/env bash
set -euo pipefail

MODEL="${ORNITH_MLX_VLM_MODEL:-mlx-community/Ornith-1.0-35B-4bit}"
PYTHON_BIN="${ORNITH_MLX_VLM_PYTHON:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ORNITH_MLX_VLM_SWEEP_DIR:-bench/runs/ornith_mlx_vlm_speed_${STAMP}}"
MAX_TOKENS="${ORNITH_MLX_VLM_SWEEP_MAX_TOKENS:-256}"
REPEATS="${ORNITH_MLX_VLM_SWEEP_REPEATS:-2}"
WARMUP_TOKENS="${ORNITH_MLX_VLM_SWEEP_WARMUP_TOKENS:-16}"
PROMPT="${ORNITH_MLX_VLM_SWEEP_PROMPT:-}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

run_case() {
  local name="$1"
  shift

  local output="${OUT_DIR}/${name}.json"
  echo "== ${name} =="
  cmd=(
    "${PYTHON_BIN}"
    scripts/benchmark_ornith_mlx_vlm.py
    --model "${MODEL}"
    --max-tokens "${MAX_TOKENS}"
    --warmup-tokens "${WARMUP_TOKENS}"
    --repeats "${REPEATS}"
    --output "${output}"
  )
  if [[ -n "${PROMPT}" ]]; then
    cmd+=(--prompt "${PROMPT}")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

run_case p2048_no_thinking \
  --prefill-step-size 2048

run_case p4096_no_thinking \
  --prefill-step-size 4096

run_case p2048_kv4_no_thinking \
  --prefill-step-size 2048 \
  --kv-bits 4 \
  --quantized-kv-start 0

run_case p2048_thinking_budget256 \
  --prefill-step-size 2048 \
  --enable-thinking \
  --thinking-budget 256

python3 - "${OUT_DIR}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    data = json.loads(path.read_text())
    rows.append(
        {
            "case": path.stem,
            "gen_tps_mean": data["generation_tps"]["mean"],
            "gen_tps_min": data["generation_tps"]["min"],
            "prompt_tps_mean": data["prompt_tps"]["mean"],
            "peak_gb_max": data["peak_gb"]["max"],
            "min_generation_tokens": data["min_generation_tokens"],
        }
    )

print("case\tgen_mean\tgen_min\tprompt_mean\tpeak_gb\tmin_gen_tokens")
for row in sorted(rows, key=lambda item: item["gen_tps_mean"], reverse=True):
    print(
        f"{row['case']}\t"
        f"{row['gen_tps_mean']:.2f}\t"
        f"{row['gen_tps_min']:.2f}\t"
        f"{row['prompt_tps_mean']:.2f}\t"
        f"{row['peak_gb_max']:.2f}\t"
        f"{row['min_generation_tokens']}"
    )
PY

echo "Results: ${OUT_DIR}"
