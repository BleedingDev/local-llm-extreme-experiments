#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

# shellcheck source=./pinned_sources.sh
source "${SCRIPT_DIR}/pinned_sources.sh"

status_ref() {
  local label="$1"
  local repo_dir="$2"
  local pinned="$3"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "- ${label}: missing (${repo_dir})"
    return
  fi

  local current
  current="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if [[ "${current}" == "${pinned}" ]]; then
    echo "- ${label}: pinned at ${current}"
  else
    echo "- ${label}: present at ${current} (expected ${pinned})"
  fi
}

echo "== DDTree + DFlash + MLX + SuperGemma combo reality check =="
echo "Repo: ${ROOT_DIR}"
echo

echo "Runnable now in this repo:"
echo "- SuperGemma inference on MLX using scripts/run_supergemma_mlx_generate.sh"
echo "- Optional local OpenAI-compatible server via scripts/run_supergemma_mlx_server.sh"
echo "- Separate DFlash MLX benchmark path (Qwen models) via scripts/run_dflash_mlx_benchmark.sh"
echo

echo "Current local status:"
if [[ -x "${VENV_DIR}/bin/mlx_lm.generate" ]]; then
  echo "- MLX generation wrapper prerequisites: ready (${VENV_DIR})"
else
  echo "- MLX generation wrapper prerequisites: missing .venv packages (run scripts/setup_env.sh)"
fi
status_ref "dflash" "${ROOT_DIR}/vendor/dflash" "${DFLASH_COMMIT}"
status_ref "ddtree" "${ROOT_DIR}/vendor/ddtree" "${DDTREE_COMMIT}"
echo

echo "Not currently supported upstream:"
echo "- DDTree + DFlash + SuperGemma 26B A4B as one integrated MLX runtime stack."
echo "- DDTree upstream remains CUDA/PyTorch-oriented rather than Apple-MLX-native."
echo

echo "Recommended fallback path supported today:"
cat <<'FALLBACK'
1) scripts/setup_env.sh
2) scripts/run_supergemma_mlx_generate.sh --prompt "Napiš stručný český test lokální inference." --max-kv-size 512
3) scripts/run_supergemma_mlx_server.sh --port 8080
4) curl -s http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Say hello in Czech."}],"max_tokens":64}'
5) Optional separate DFlash check:
   scripts/run_dflash_mlx_benchmark.sh --dry-run
FALLBACK
