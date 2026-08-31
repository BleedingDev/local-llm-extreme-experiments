#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Warm up Ornith's OpenAI-compatible tool-call path before a benchmark run.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="mlx-community/Ornith-1.0-35B-4bit")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "developer",
                "content": "You are warming up a local tool-calling model. Call the provided shell tool.",
            },
            {"role": "user", "content": "Call ShellSession with command: true"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ShellSession",
                    "description": "Run a command in a persistent bash session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    url = args.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", "authorization": "Bearer noop"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        body = resp.read()
    elapsed = time.perf_counter() - started
    if resp.status >= 400:
        raise RuntimeError(body.decode("utf-8", "replace")[:1000])
    print(f"ornith warmup complete in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
