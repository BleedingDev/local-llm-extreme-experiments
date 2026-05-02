#!/usr/bin/env python3
"""trace-gepa workspace and venv sanity check."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx")
TRACE_GEPA = ROOT / "trace-gepa"
VENV_PY = ROOT / ".venv-gepa" / "bin" / "python"
ENV_FILE = ROOT / ".env"

EXPECTED_FILES = [
    TRACE_GEPA / "SHARED_BRIEFING.md",
    TRACE_GEPA / "README.md",
    TRACE_GEPA / "data" / "seed_sessions.json",
    TRACE_GEPA / "extractors" / "extract_cc.py",
    TRACE_GEPA / "extractors" / "extract_codex.py",
    TRACE_GEPA / "extractors" / "categorize.py",
    TRACE_GEPA / "agent_opt" / "adapter.py",
    TRACE_GEPA / "agent_opt" / "reflection.py",
    TRACE_GEPA / "agent_opt" / "optimize.py",
    TRACE_GEPA / "bench" / "eval_baseline.py",
]

REQUIRED_IMPORTS = ["dspy", "gepa", "anthropic", "litellm", "orjson", "dotenv"]


def main() -> int:
    missing: list[str] = []

    if not VENV_PY.exists():
        missing.append(f"venv python: {VENV_PY}")

    if sys.executable != str(VENV_PY):
        running_under_venv = Path(sys.executable).resolve() == VENV_PY.resolve()
        if not running_under_venv:
            print(f"WARN: running under {sys.executable}, expected {VENV_PY}", file=sys.stderr)

    for fp in EXPECTED_FILES:
        if not fp.exists():
            missing.append(f"file: {fp}")

    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
        except Exception as e:
            missing.append(f"import {mod}: {type(e).__name__}: {e}")

    if not ENV_FILE.exists():
        missing.append(f".env: {ENV_FILE}")
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv(ENV_FILE)
        except Exception as e:
            missing.append(f"dotenv load: {e}")
        if not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            missing.append("env var ANTHROPIC_AUTH_TOKEN not set after load_dotenv")

    seed_path = TRACE_GEPA / "data" / "seed_sessions.json"
    if seed_path.exists():
        try:
            with open(seed_path) as f:
                seed = json.load(f)
            seed_paths: list[str] = []
            for key in ("cc_sessions", "codex_sessions"):
                for entry in seed.get(key, []):
                    seed_paths.append(entry["path"])
            if len(seed_paths) != 30:
                missing.append(f"seed_sessions.json has {len(seed_paths)} paths, expected 30")
            for p in seed_paths:
                if not Path(p).exists():
                    missing.append(f"seed path missing: {p}")
        except Exception as e:
            missing.append(f"seed_sessions.json parse: {e}")
    else:
        missing.append(f"file: {seed_path}")

    dataset_path = TRACE_GEPA / "data" / "dataset.jsonl"
    if dataset_path.exists():
        try:
            out = subprocess.check_output(["wc", "-l", str(dataset_path)], text=True)
            count = out.strip().split()[0]
            print(f"dataset.jsonl: {count} lines")
        except Exception as e:
            print(f"dataset.jsonl: wc failed: {e}")
    else:
        print("dataset.jsonl: not yet written")

    if missing:
        print("FAIL", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
