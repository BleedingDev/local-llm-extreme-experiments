from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.env")
_ENV_LOADED = False


def _ensure_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    _ENV_LOADED = True


@lru_cache(maxsize=4)
def _client():
    _ensure_env()
    import anthropic

    key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_AUTH_TOKEN missing from environment")
    return anthropic.Anthropic(api_key=key)


def chat(
    messages: list[dict],
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    system: str | None = None,
) -> str:
    client = _client()
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.messages.create(**kwargs)
            parts = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        except Exception as e:
            last_err = e
            # Some models (e.g. claude-opus-4-7) reject `temperature`. Retry once without it.
            if "temperature" in str(e) and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                continue
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
    raise last_err  # unreachable
