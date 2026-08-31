#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${ORNITH_LLAMA_BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${ORNITH_LLAMA_ALIAS:-ornith-1.0-35b-q4-k-m}"
MAX_TOKENS="${ORNITH_SMOKE_MAX_TOKENS:-32}"

python3 - "$BASE_URL" "$MODEL" "$MAX_TOKENS" <<'PY'
import json
import sys
import urllib.request

base_url, model, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])

def get(path, timeout=10):
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post(path, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"content-type": "application/json", "authorization": "Bearer noop"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

print(json.dumps(get("/models"), indent=2)[:2000])

resp = post(
    "/chat/completions",
    {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply exactly: ornith ready"}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(json.dumps(resp, indent=2, ensure_ascii=False)[:4000])
PY
