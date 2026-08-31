#!/usr/bin/env bash
set -euo pipefail

MODEL="${LITTLE_CODER_ORNITH_MODEL:-omlx/mlx-community/Ornith-1.0-35B-4bit}"

export OMLX_API_KEY="${OMLX_API_KEY:-noop}"
export ORNITHCPP_API_KEY="${ORNITHCPP_API_KEY:-noop}"
export LITTLE_CODER_TEMPERATURE_PROVIDERS="${LITTLE_CODER_TEMPERATURE_PROVIDERS:-llamacpp,ollama,lmstudio,ornithcpp,omlx}"
export LITTLE_CODER_CHAT_TEMPLATE_KWARGS="${LITTLE_CODER_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
export LITTLE_CODER_MAX_TOKENS="${LITTLE_CODER_MAX_TOKENS:-768}"

exec little-coder --model "${MODEL}" "$@"
