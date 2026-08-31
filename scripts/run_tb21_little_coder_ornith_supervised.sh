#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAMP="${TB_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${TB_RUN_ID:-ornith_harbor_lc_temp02_8192_full_supervised_${STAMP}}"
LOG_DIR="${TB_LOG_DIR:-bench/runs}"
JOBS_DIR="${TB_JOBS_DIR:-bench/jobs}"
JOB_NAME="${TB_JOB_NAME:-${RUN_ID}}"

MODEL_ID="${ORNITH_LITTLE_CODER_MODEL_ID:-mlx-community/Ornith-1.0-35B-4bit}"
HARBOR_MODEL="${TB_MODEL:-omlx/${MODEL_ID}}"
DATASET="${TB_DATASET:-terminal-bench/terminal-bench-2-1}"
DATASET_PATH="${TB_DATASET_PATH:-}"

SERVER_HOST="${ORNITH_MLX_VLM_HOST:-127.0.0.1}"
SERVER_PORT="${ORNITH_MLX_VLM_PORT:-8000}"
PROXY_HOST="${ORNITH_PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${ORNITH_PROXY_PORT:-8001}"
PREFILL_STEP_SIZE="${ORNITH_MLX_VLM_PREFILL_STEP_SIZE:-4096}"
MAX_KV_SIZE="${ORNITH_MLX_VLM_MAX_KV_SIZE:-65536}"

MAX_TURNS="${TB_MAX_TURNS:-40}"
MAX_OUTPUT_TOKENS="${TB_MAX_OUTPUT_TOKENS:-8192}"
N_CONCURRENT="${TB_N_CONCURRENT:-1}"
K="${TB_K:-1}"
TEMPERATURE="${LITTLE_CODER_TEMPERATURE:-0.2}"
CHAT_TEMPLATE_KWARGS="${LITTLE_CODER_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false,\"preserve_thinking\":false}}"

SERVER_LOG="${LOG_DIR}/${RUN_ID}.mlx_vlm.log"
PROXY_JSONL="${LOG_DIR}/${RUN_ID}.proxy.jsonl"
PROXY_LOG="${LOG_DIR}/${RUN_ID}.proxy.stdout.log"
HARBOR_LOG="${LOG_DIR}/${RUN_ID}.harbor.log"
MANIFEST="${LOG_DIR}/${RUN_ID}.manifest.json"

mkdir -p "${LOG_DIR}"
mkdir -p "${JOBS_DIR}"

SERVER_PID=""
PROXY_PID=""
WATCHER_PID=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

kill_if_running() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  kill_if_running "${WATCHER_PID}"
  if [[ "${TB_KEEP_SERVERS:-0}" != "1" ]]; then
    kill_if_running "${PROXY_PID}"
    kill_if_running "${SERVER_PID}"
  fi
}
trap cleanup EXIT

http_code() {
  curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || true
}

free_or_fail_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return
  fi
  if [[ "${TB_KILL_PORTS:-0}" == "1" ]]; then
    log "Killing existing ${label} listener(s) on port ${port}: ${pids}"
    kill ${pids} >/dev/null 2>&1 || true
    sleep 1
    return
  fi
  echo "Port ${port} is already in use by PID(s): ${pids}. Set TB_KILL_PORTS=1 to kill them." >&2
  exit 2
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local timeout_s="$3"
  local pid="$4"
  local start
  start="$(date +%s)"
  while true; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "${label} process exited before becoming healthy. Recent log:" >&2
      tail -n 120 "${SERVER_LOG}" "${PROXY_LOG}" 2>/dev/null >&2 || true
      exit 3
    fi
    local code
    code="$(http_code "${url}")"
    if [[ "${code}" == "200" ]]; then
      return
    fi
    if (( $(date +%s) - start > timeout_s )); then
      echo "Timed out waiting for ${label} at ${url}. Last HTTP code: ${code:-none}" >&2
      tail -n 120 "${SERVER_LOG}" "${PROXY_LOG}" 2>/dev/null >&2 || true
      exit 4
    fi
    sleep 2
  done
}

write_manifest() {
  "${PYTHON:-python3}" - "$MANIFEST" <<PY
import json, os, sys
path = sys.argv[1]
data = {
    "run_id": "${RUN_ID}",
    "job_name": "${JOB_NAME}",
    "dataset": "${DATASET}",
    "dataset_path": "${DATASET_PATH}",
    "harbor_model": "${HARBOR_MODEL}",
    "model_id": "${MODEL_ID}",
    "server_url": "http://${SERVER_HOST}:${SERVER_PORT}/v1",
    "proxy_url": "http://${PROXY_HOST}:${PROXY_PORT}/v1",
    "prefill_step_size": int("${PREFILL_STEP_SIZE}"),
    "max_kv_size": int("${MAX_KV_SIZE}"),
    "max_turns": int("${MAX_TURNS}"),
    "max_output_tokens": int("${MAX_OUTPUT_TOKENS}"),
    "n_concurrent": int("${N_CONCURRENT}"),
    "k": int("${K}"),
    "temperature": float("${TEMPERATURE}"),
    "chat_template_kwargs": json.loads(os.environ["CHAT_TEMPLATE_KWARGS_FOR_MANIFEST"]),
    "logs": {
        "server": "${SERVER_LOG}",
        "proxy_jsonl": "${PROXY_JSONL}",
        "proxy_stdout": "${PROXY_LOG}",
        "harbor": "${HARBOR_LOG}",
    },
    "jobs_dir": "${JOBS_DIR}",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
}

watch_backend() {
  local harbor_pid="$1"
  local server_url="http://${SERVER_HOST}:${SERVER_PORT}/v1/models"
  while kill -0 "${harbor_pid}" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      log "MLX server died; aborting Harbor PID ${harbor_pid}"
      kill "${harbor_pid}" >/dev/null 2>&1 || true
      return 75
    fi
    if ! kill -0 "${PROXY_PID}" >/dev/null 2>&1; then
      log "Proxy died; aborting Harbor PID ${harbor_pid}"
      kill "${harbor_pid}" >/dev/null 2>&1 || true
      return 75
    fi
    local code
    code="$(http_code "${server_url}")"
    if [[ "${code}" != "200" ]]; then
      log "MLX health check failed with HTTP ${code:-none}; aborting Harbor PID ${harbor_pid}"
      kill "${harbor_pid}" >/dev/null 2>&1 || true
      return 75
    fi
    if [[ -s "${PROXY_JSONL}" ]] && "${PYTHON:-python3}" - "${PROXY_JSONL}" <<'PY'
import json, sys
path = sys.argv[1]
bad = False
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row.get("status")
        if isinstance(status, int) and status >= 500:
            bad = True
            break
sys.exit(0 if bad else 1)
PY
    then
      log "Proxy recorded a 5xx response; aborting Harbor PID ${harbor_pid}"
      kill "${harbor_pid}" >/dev/null 2>&1 || true
      return 75
    fi
    sleep 5
  done
}

summarize_proxy() {
  "${PYTHON:-python3}" - "${PROXY_JSONL}" <<'PY'
import collections, json, pathlib, statistics, sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("proxy_summary missing")
    sys.exit(0)
rows = []
for line in path.read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))
statuses = collections.Counter(row.get("status") for row in rows)
finishes = collections.Counter(row.get("finish_reason") for row in rows)
completion = sum(row.get("completion_tokens") or row.get("estimated_completion_tokens") or 0 for row in rows)
elapsed = sum(row.get("elapsed_s") or 0 for row in rows)
tps_values = [
    (row.get("completion_tokens") or row.get("estimated_completion_tokens") or 0) / row["elapsed_s"]
    for row in rows
    if row.get("elapsed_s") and (row.get("completion_tokens") or row.get("estimated_completion_tokens"))
]
errors = [row.get("error") for row in rows if row.get("error")]
print("proxy_summary rows", len(rows))
print("proxy_summary statuses", dict(statuses))
print("proxy_summary finish_reasons", dict(finishes))
print("proxy_summary completion_tokens", completion)
print("proxy_summary elapsed_s", round(elapsed, 2))
print("proxy_summary aggregate_tps", round(completion / elapsed, 2) if elapsed else None)
print("proxy_summary median_call_tps", round(statistics.median(tps_values), 2) if tps_values else None)
print("proxy_summary errors", len(errors), errors[:3])
PY
}

free_or_fail_port "${SERVER_PORT}" "MLX server"
free_or_fail_port "${PROXY_PORT}" "proxy"

export CHAT_TEMPLATE_KWARGS_FOR_MANIFEST="${CHAT_TEMPLATE_KWARGS}"
write_manifest

log "Starting MLX server: ${MODEL_ID}, MAX_KV_SIZE=${MAX_KV_SIZE}, prefill=${PREFILL_STEP_SIZE}"
scripts/run_ornith_mlx_vlm_server.sh \
  --model "${MODEL_ID}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --prefill-step "${PREFILL_STEP_SIZE}" \
  --max-kv-size "${MAX_KV_SIZE}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
wait_for_http "MLX server" "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" 900 "${SERVER_PID}"

PROXY_PY="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PROXY_PY=".venv/bin/python"
fi

log "Starting OpenAI proxy on ${PROXY_HOST}:${PROXY_PORT}"
"${PROXY_PY}" scripts/ornith_openai_proxy.py \
  --upstream "http://${SERVER_HOST}:${SERVER_PORT}" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --model "${MODEL_ID}" \
  --log "${PROXY_JSONL}" \
  --exit-on-upstream-error \
  >"${PROXY_LOG}" 2>&1 &
PROXY_PID="$!"
wait_for_http "proxy" "http://${PROXY_HOST}:${PROXY_PORT}/v1/models" 60 "${PROXY_PID}"

log "Running strict warmup through proxy"
"${PYTHON:-python3}" scripts/warmup_ornith_openai_toolcall.py \
  --base-url "http://${PROXY_HOST}:${PROXY_PORT}/v1" \
  --model "${MODEL_ID}" \
  --timeout "${TB_WARMUP_TIMEOUT:-180}" >&2

log "Starting Harbor job ${JOB_NAME}"
(
  export OPENAI_API_KEY="${OPENAI_API_KEY:-noop}"
  export OMLX_API_KEY="${OMLX_API_KEY:-noop}"
  export ORNITHCPP_API_KEY="${ORNITHCPP_API_KEY:-noop}"
  export ORNITH_MLX_BASE_URL="http://${PROXY_HOST}:${PROXY_PORT}/v1"
  export LITTLE_CODER_CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS}"
  export LITTLE_CODER_TEMPERATURE="${TEMPERATURE}"
  export LITTLE_CODER_TEMPERATURE_PROVIDERS="${LITTLE_CODER_TEMPERATURE_PROVIDERS:-llamacpp,ollama,lmstudio,ornithcpp,omlx}"
  export LITTLE_CODER_BENCHMARK=terminal_bench
  export LITTLE_CODER_PERMISSION_MODE="${LITTLE_CODER_PERMISSION_MODE:-accept-all}"
  export LITTLE_CODER_MAX_TURNS="${MAX_TURNS}"
  export LITTLE_CODER_MAX_TOKENS="${MAX_OUTPUT_TOKENS}"
  export TB_DATASET="${DATASET}"
  export TB_DATASET_PATH="${DATASET_PATH}"
  export TB_MODEL="${HARBOR_MODEL}"
  export TB_N_CONCURRENT="${N_CONCURRENT}"
  export TB_K="${K}"
  export TB_JOB_NAME="${JOB_NAME}"
  export TB_MAX_TURNS="${MAX_TURNS}"
  export TB_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS}"
  export TB_WARMUP=0
  scripts/run_tb21_little_coder_ornith.sh --jobs-dir "${JOBS_DIR}" "$@"
) >"${HARBOR_LOG}" 2>&1 &
HARBOR_PID="$!"

watch_backend "${HARBOR_PID}" &
WATCHER_PID="$!"

set +e
wait "${HARBOR_PID}"
HARBOR_STATUS="$?"
set -e

kill_if_running "${WATCHER_PID}"
WATCHER_PID=""

log "Harbor exited with status ${HARBOR_STATUS}"
summarize_proxy

if [[ "${HARBOR_STATUS}" != "0" ]]; then
  tail -n 120 "${HARBOR_LOG}" >&2 || true
  exit "${HARBOR_STATUS}"
fi

log "Completed. Result: ${JOBS_DIR}/${JOB_NAME}/result.json"
log "Logs: ${SERVER_LOG}, ${PROXY_JSONL}, ${HARBOR_LOG}"
