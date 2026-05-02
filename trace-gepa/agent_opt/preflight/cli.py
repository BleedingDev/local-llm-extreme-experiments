"""Tiny CLI wrapper:

    python -m agent_opt.preflight.cli --action '{"name":"Bash","input":"find /"}' --context '{}'
"""
from __future__ import annotations

import argparse
import json
import sys

from .checker import PreflightChecker


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="preflight")
    p.add_argument("--action", required=True, help="JSON action dict")
    p.add_argument("--context", default="{}", help="JSON context dict")
    args = p.parse_args(argv)

    try:
        action = json.loads(args.action)
        context = json.loads(args.context)
    except json.JSONDecodeError as e:
        print(json.dumps({"passed": False, "blocked_by": [f"bad json: {e}"], "warnings": [], "fired": []}))
        return 2

    result = PreflightChecker().check(action, context)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
