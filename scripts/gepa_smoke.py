"""Smoke test for the GEPA/DSPy environment.

Verifies that:
  * `python-dotenv` can load `.env` from the repo root.
  * `dspy.LM("anthropic/claude-haiku-4-5", ...)` can be constructed
    using ANTHROPIC_AUTH_TOKEN (non-standard env var) passed explicitly.
  * `dspy.Predict("question -> answer")` actually returns a response.
  * `gepa` (or `dspy.teleprompt.GEPA`) imports cleanly.

Usage:
    ./.venv-gepa/bin/python scripts/gepa_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print(f"FAIL: ANTHROPIC_AUTH_TOKEN missing from {env_path}", file=sys.stderr)
        return 1
    print(f"ANTHROPIC_AUTH_TOKEN: present (len={len(api_key)})")

    import dspy

    lm = dspy.LM("anthropic/claude-haiku-4-5", api_key=api_key)
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="What is 2+2?")
    print(f"DSPy Predict response: {result.answer!r}")

    # Verify GEPA is importable.
    try:
        import gepa  # noqa: F401

        print(f"gepa import: OK (module={gepa.__name__})")
    except Exception as e:  # pragma: no cover
        print(f"gepa import FAILED: {e!r}", file=sys.stderr)
        try:
            from dspy.teleprompt import GEPA  # noqa: F401

            print("dspy.teleprompt.GEPA import: OK (fallback)")
        except Exception as e2:
            print(f"dspy.teleprompt.GEPA import FAILED: {e2!r}", file=sys.stderr)
            return 2

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
