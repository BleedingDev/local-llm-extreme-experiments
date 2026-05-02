#!/usr/bin/env python3
"""Orchestrate v2 dataset build: extract -> merge with v1 -> categorise -> split.

Usage::

    python extractors/build_v2.py            # full manifest (200 sessions)
    python extractors/build_v2.py --cap 80   # cap sessions per source
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "extractors"))

import categorize  # noqa: E402

V1_DATASET = DATA / "dataset.jsonl"
CC_NEW = DATA / "cc_dataset_v2_new.jsonl"
CODEX_NEW = DATA / "codex_dataset_v2_new.jsonl"
OUT_DATASET = DATA / "dataset_v2.jsonl"
OUT_SPLITS = DATA / "splits_v2.json"


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(orjson.loads(line))
            except Exception:
                continue
    return out


def run_extractors(cap: int | None, py: str) -> None:
    args_cc = [py, str(ROOT / "extractors" / "extract_cc_v2.py")]
    args_codex = [py, str(ROOT / "extractors" / "extract_codex_v2.py")]
    if cap is not None:
        args_cc.append(str(cap))
        args_codex.append(str(cap))
    p_cc = subprocess.Popen(args_cc)
    p_codex = subprocess.Popen(args_codex)
    rc_cc = p_cc.wait()
    rc_codex = p_codex.wait()
    if rc_cc != 0:
        raise SystemExit(f"extract_cc_v2 exited {rc_cc}")
    if rc_codex != 0:
        raise SystemExit(f"extract_codex_v2 exited {rc_codex}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=None, help="cap sessions per source")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--py", default=sys.executable)
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_extract:
        run_extractors(args.cap, args.py)

    # Merge: v1 + v2 new, dedup by id (later wins are equivalent so first wins).
    merged: dict[str, dict] = {}
    for path in (V1_DATASET, CC_NEW, CODEX_NEW):
        for r in load_jsonl(path):
            rid = r.get("id")
            if not rid:
                continue
            if rid not in merged:
                merged[rid] = r

    records = list(merged.values())
    pre_cat = Counter((r.get("failure_category") or "null") for r in records)
    categorize.recategorize(records)
    post_cat = Counter((r.get("failure_category") or "null") for r in records)

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATASET.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")

    splits = categorize.stratified_split(records, seed=42)
    OUT_SPLITS.write_bytes(orjson.dumps(splits, option=orjson.OPT_INDENT_2))

    v1_ids = {r.get("id") for r in load_jsonl(V1_DATASET)}
    retained = sum(1 for r in records if r.get("id") in v1_ids)

    label_dist = Counter((r.get("label") or "null") for r in records)
    elapsed = time.time() - t0
    print(f"v2 records: {len(records)}")
    print(f"v1 retained: {retained}/{len(v1_ids)}")
    print(f"label dist: {dict(label_dist.most_common())}")
    print(f"top-10 failure categories (post): {dict(post_cat.most_common(10))}")
    print(f"split counts: {splits['counts']}")
    print(f"wrote {OUT_DATASET}")
    print(f"wrote {OUT_SPLITS}")
    print(f"elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
