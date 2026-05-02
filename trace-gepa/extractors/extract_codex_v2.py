#!/usr/bin/env python3
"""Run the v1 Codex per-record extractor against the v2 manifest paths."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extractors"))

import extract_codex  # noqa: E402

V1_SEED = ROOT / "data" / "seed_sessions.json"
V2_MANIFEST = ROOT / "data" / "v2_manifest_proposal.json"
OUT = ROOT / "data" / "codex_dataset_v2_new.jsonl"


def main(argv: list[str]) -> int:
    cap = None
    if len(argv) > 1:
        cap = int(argv[1])

    v1_paths = {s["path"] for s in json.loads(V1_SEED.read_text()).get("codex_sessions", [])}
    manifest = json.loads(V2_MANIFEST.read_text())
    sessions = manifest.get("codex_sessions", [])
    if cap is not None:
        sessions = sessions[:cap]

    new_paths = [s for s in sessions if s["path"] not in v1_paths]
    print(f"v2 codex sessions: {len(sessions)} ({len(new_paths)} new)", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter = Counter()
    fc_counts: Counter = Counter()
    total = 0
    with OUT.open("wb") as f:
        for idx, s in enumerate(new_paths):
            path = s["path"]
            if not os.path.exists(path):
                continue
            recs = extract_codex.extract_session(path, idx)
            ph = hashlib.sha1(path.encode()).hexdigest()[:10]
            for r in recs:
                old = r["id"]
                tail = old.rsplit("_evt", 1)[-1] if "_evt" in old else old
                r["id"] = f"codex_v2_{ph}_evt{tail}"
                f.write(orjson.dumps(r) + b"\n")
                label_counts[r["label"]] += 1
                if r["failure_category"]:
                    fc_counts[r["failure_category"]] += 1
                total += 1
            print(f"  {Path(path).name}: {len(recs)}", file=sys.stderr)

    print(f"codex_v2_new total: {total}")
    print(f"label_dist: {dict(label_counts.most_common())}")
    print(f"fc_dist: {dict(fc_counts.most_common(10))}")
    print(f"output: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
