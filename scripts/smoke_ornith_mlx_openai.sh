#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${ORNITH_MLX_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${ORNITH_LITTLE_CODER_MODEL_ID:-mlx-community/Ornith-1.0-35B-4bit}"

python3 - "$BASE_URL" "$MODEL" <<'PY'
import json
import sys
import urllib.request

base_url, model = sys.argv[1], sys.argv[2]

def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"content-type": "application/json", "authorization": "Bearer noop"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))

with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as resp:
    models = json.loads(resp.read().decode("utf-8"))
print(json.dumps(models, indent=2)[:2000])

resp = post(
    "/chat/completions",
    {
        "model": model,
        "messages": [
            {"role": "user", "content": "Write a Python is_even(n) function in one line."}
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.95,
    },
)
print(json.dumps(resp, indent=2, ensure_ascii=False)[:4000])
PY
