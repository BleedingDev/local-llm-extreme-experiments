"""Scan ~/.codex/sessions/ rollouts and map each session path to its model.

Reads only the first ~80 lines per file (turn_context appears near the top).
Writes JSON: {session_path: model}. Records 'unknown' if no turn_context found.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.path.expanduser("~/.codex/sessions"))
OUT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/data/codex_session_models.json")
HEAD_LINES = 80


def model_of(path: Path) -> str:
    try:
        with path.open("rb") as f:
            for i, raw in enumerate(f):
                if i > HEAD_LINES:
                    break
                if b'"turn_context"' not in raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if obj.get("type") != "turn_context":
                    continue
                m = (obj.get("payload") or {}).get("model")
                if isinstance(m, str):
                    return m
    except Exception:
        return "error"
    return "unknown"


def main() -> int:
    files = [p for p in ROOT.rglob("rollout-*.jsonl") if p.is_file()]
    print(f"[scan] found {len(files)} session files")
    t0 = time.time()
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(model_of, p): p for p in files}
        for fut in as_completed(futs):
            out[str(futs[fut])] = fut.result()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    counts: dict[str, int] = {}
    for v in out.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"[scan] done in {time.time()-t0:.1f}s; wrote {OUT}")
    print(f"[scan] dist: {sorted(counts.items(), key=lambda kv: -kv[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
