#!/usr/bin/env bash
# Full eval driver: 6 (model x prompt) combinations on 175 tasks each.
# Writes JSON outputs into bench/results/full_eval/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TASKS="$ROOT/data/benchmark_tasks_full.jsonl"
OUTDIR="$HERE/results/full_eval"
mkdir -p "$OUTDIR"

PY="${PY:-/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.venv-gepa/bin/python}"

SEED_FILE="$ROOT/artifacts/optimized-prompts/_seed_for_full_eval.system.md"
BAG_EXEC_OPUS="$ROOT/artifacts/optimized-prompts/bag_exec_opus_v2_run_20260501T233901Z/best_candidate.system.md"
CODEX="$ROOT/artifacts/optimized-prompts/codex_run_20260501T224340Z/best_candidate.system.md"
BAG="$ROOT/artifacts/optimized-prompts/bag_run_20260501T224339Z/best_candidate.system.md"

# Refresh seed file (idempotent)
"$PY" -c "import sys; sys.path.insert(0, '$ROOT'); from agent_opt.seed import SEED_PROMPT; open('$SEED_FILE', 'w').write(SEED_PROMPT)"

run_one() {
  local model="$1" prompt_id="$2" prompt_file="$3"
  local out="$OUTDIR/${model}_${prompt_id}.json"
  echo "=== $model | $prompt_id -> $out ==="
  "$PY" "$HERE/run_anthropic.py" \
    --tasks "$TASKS" \
    --model "$model" \
    --max-workers 8 \
    --max-tokens 256 \
    --system-prompt-file "$prompt_file" \
    --output "$out"
  # tag the output with system_prompt_id and harness for leaderboard ingest
  "$PY" - <<PY
import json, pathlib
p = pathlib.Path("$out")
if p.exists():
    obj = json.loads(p.read_text())
    obj["system_prompt_id"] = "$prompt_id"
    obj["harness"] = "anthropic"
    p.write_text(json.dumps(obj, indent=2, default=str))
PY
}

# 6 combos: 2 models x 3 prompts (seed, bag_exec_opus_v2 = current latest, codex).
# Set BENCH_INCLUDE_BAG=1 to also run the original BAG candidate.
for model in claude-haiku-4-5 claude-opus-4-7; do
  run_one "$model" "seed"             "$SEED_FILE"
  run_one "$model" "bag_exec_opus_v2" "$BAG_EXEC_OPUS"
  run_one "$model" "codex"            "$CODEX"
  if [[ "${BENCH_INCLUDE_BAG:-0}" == "1" ]]; then
    run_one "$model" "bag"            "$BAG"
  fi
done

echo "DONE: $(ls -1 "$OUTDIR" | wc -l) result files"
