#!/usr/bin/env bash
# Phase-3 Fix Agent #FIX5: comprehensive multi-model multi-harness sweep.
# Runs 8 combinations against the 30-task minibench (post-FIX1 verifiers).
# Each run is time-boxed to 5 minutes; the global budget is ~30 minutes.
#
# Usage:  bash trace-gepa/bench/run_sweep.sh
# Outputs: trace-gepa/bench/results/sweep/<harness>_<model>_<prompt_id>.json
#
set -u  # NOTE: deliberately NOT set -e: we want to keep going if a run fails.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACE_GEPA="${ROOT}/trace-gepa"
PY="${ROOT}/.venv-gepa/bin/python"
MINIBENCH="${TRACE_GEPA}/data/benchmarks/minibench.jsonl"
OPT_PROMPT="${TRACE_GEPA}/artifacts/optimized-prompts/latest/best_candidate.system.md"
SWEEP_DIR="${TRACE_GEPA}/bench/results/sweep"
mkdir -p "${SWEEP_DIR}"

PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-300}"  # 5 minute wall-clock per run

note() { printf '[sweep] %s\n' "$*" >&2; }

# ----------------------------- pre-flight ----------------------------------
if [[ ! -f "${MINIBENCH}" ]]; then
  note "FATAL: minibench not found at ${MINIBENCH}"; exit 2
fi
if [[ ! -x "${PY}" ]]; then
  note "FATAL: venv python not at ${PY}"; exit 2
fi
if [[ ! -f "${OPT_PROMPT}" ]]; then
  note "WARN: optimised prompt not at ${OPT_PROMPT}; opt-runs will fall back to seed"
fi

# tiny helper: run a command with a per-run wall-clock cap; emits a status line.
run_one() {
  local label="$1"; shift
  local out_path="$1"; shift
  local t0=$(date +%s)
  note "START ${label} -> ${out_path}"
  # `timeout` may be `gtimeout` on macOS; fall back if absent.
  local TIMEOUT_BIN
  if command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN="gtimeout"
  elif command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN="timeout"
  else TIMEOUT_BIN=""
  fi
  if [[ -n "${TIMEOUT_BIN}" ]]; then
    "${TIMEOUT_BIN}" --kill-after=10 "${PER_RUN_TIMEOUT}" "$@" \
      >"${out_path}.stdout" 2>"${out_path}.stderr"
    local rc=$?
  else
    "$@" >"${out_path}.stdout" 2>"${out_path}.stderr" &
    local pid=$!
    ( sleep "${PER_RUN_TIMEOUT}" && kill -9 "${pid}" 2>/dev/null ) &
    local watchdog=$!
    wait "${pid}"; local rc=$?
    kill -9 "${watchdog}" 2>/dev/null || true
  fi
  local elapsed=$(( $(date +%s) - t0 ))
  if [[ ! -f "${out_path}" ]]; then
    note "FAIL ${label} (rc=${rc}, ${elapsed}s, no output file)"
  else
    note "DONE ${label} (rc=${rc}, ${elapsed}s)"
  fi
}

# ------------------------ Anthropic harness -------------------------------
run_anthropic() {
  local model="$1"; local prompt_id="$2"; local prompt_arg="$3"
  local out="${SWEEP_DIR}/anthropic_${model}_${prompt_id}.json"
  local args=(
    "${PY}" "${TRACE_GEPA}/bench/run_anthropic.py"
    --tasks "${MINIBENCH}"
    --model "${model}"
    --limit 30
    --max-workers 8
    --output "${out}"
  )
  if [[ -n "${prompt_arg}" ]]; then
    args+=( --system-prompt-file "${prompt_arg}" )
  fi
  run_one "anthropic/${model}/${prompt_id}" "${out}" "${args[@]}"
}

# ------------------------ Codex harness -----------------------------------
run_codex() {
  local model="$1"; local reasoning="$2"; local label_id="$3"
  local out="${SWEEP_DIR}/codex_${model}_${label_id}.json"
  local args=(
    "${PY}" "${TRACE_GEPA}/bench/run_codex.py"
    --tasks "${MINIBENCH}"
    --model "${model}"
    --reasoning "${reasoning}"
    --limit 30
    --max-workers 4
    --timeout 60
    --output "${out}"
  )
  run_one "codex/${model}/${reasoning}" "${out}" "${args[@]}"
}

# ------------------------ MLX harness -------------------------------------
run_mlx() {
  local model="$1"; local label_id="$2"
  # sanitise model id for filename
  local safe="${model//\//__}"
  local out="${SWEEP_DIR}/mlx_${safe}_${label_id}.json"
  local args=(
    "${PY}" "${TRACE_GEPA}/bench/run_mlx.py"
    --tasks "${MINIBENCH}"
    --model "${model}"
    --limit 30
    --max-tokens 256
    --task-timeout 30
    --output "${out}"
  )
  run_one "mlx/${model}/${label_id}" "${out}" "${args[@]}"
}

# ============================== Sweep matrix =================================
# 1-2: Opus, optimised + seed
# 3-4: Haiku, optimised + seed
# 5-7: Codex variants
# 8:   MLX (loadable model)

# Seed prompt: we materialise it from the python module to a temp file.
SEED_PATH="${SWEEP_DIR}/_seed_prompt.txt"
"${PY}" -c "from agent_opt.seed import SEED_PROMPT; import sys; sys.stdout.write(SEED_PROMPT)" \
  > "${SEED_PATH}" 2>/dev/null \
  || (cd "${TRACE_GEPA}" && "${PY}" -c "from agent_opt.seed import SEED_PROMPT; import sys; sys.stdout.write(SEED_PROMPT)" > "${SEED_PATH}")

note "==== Anthropic sweeps ===="
run_anthropic "claude-opus-4-7"   "opt"  "${OPT_PROMPT}"
run_anthropic "claude-opus-4-7"   "seed" "${SEED_PATH}"
run_anthropic "claude-haiku-4-5"  "opt"  "${OPT_PROMPT}"
run_anthropic "claude-haiku-4-5"  "seed" "${SEED_PATH}"

note "==== Codex sweeps ===="
run_codex "gpt-5.5" "xhigh" "xhigh"
run_codex "gpt-5.5" "high"  "high"
run_codex "gpt-5.4" "high"  "high"

note "==== MLX sweep ===="
run_mlx "mlx-community/Llama-3.2-1B-Instruct-4bit" "default"

note "==== sweep complete ===="
ls -la "${SWEEP_DIR}"
