"""Hybrid prompt merger.

Combines the BAG-track winner and Codex-track winner into a single hybrid
system prompt via one call to the reflection LM (claude-opus-4-7).

Usage:
    python -m agent_opt.merge_prompts \
        --bag <path-to-bag/best_candidate.system.md> \
        --codex <path-to-codex/best_candidate.system.md> \
        --out-root <artifacts/optimized-prompts>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt.llm import _client  # noqa: E402


MERGE_INSTRUCTIONS = """You are an expert at combining LLM system prompts. Below are two prompts that each won an evaluation on different facets of a tool-selection task.

PROMPT A (won overall on held-out test split, 0.767):
{bag}

PROMPT B (won validation set, scored 0.750 on val):
{codex}

Produce ONE merged prompt that:
- Preserves the strict JSON output contract from PROMPT A (this is non-negotiable: BAG calls planJSON.parse).
- Adopts the more specific failure-category-named rules from whichever prompt has them.
- Removes redundancy.
- Is at most 2,800 chars.

Output ONLY the merged prompt, no commentary, no markdown fences."""


def _strip_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # remove first fence line
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s.strip()


def merge(bag_text: str, codex_text: str, model: str = "claude-opus-4-7") -> str:
    prompt = MERGE_INSTRUCTIONS.format(bag=bag_text, codex=codex_text)
    # Some Anthropic models (e.g. opus-4-7) reject the temperature kwarg; try
    # without it on retry. agent_opt.llm.chat hard-codes temperature, so call
    # the SDK directly here.
    client = _client()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
    parts = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return _strip_fences("".join(parts))


def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to BAG winner system.md")
    ap.add_argument("--codex", required=True, help="Path to Codex winner system.md")
    ap.add_argument(
        "--out-root",
        default=str(_ROOT / "artifacts" / "optimized-prompts"),
        help="Root dir for hybrid_run_<TS>/",
    )
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--max-chars", type=int, default=2800)
    args = ap.parse_args()

    bag_path = Path(args.bag).resolve()
    codex_path = Path(args.codex).resolve()
    bag_text = bag_path.read_text()
    codex_text = codex_path.read_text()

    print(f"BAG source:   {bag_path} ({len(bag_text)} chars)")
    print(f"Codex source: {codex_path} ({len(codex_text)} chars)")

    t0 = time.time()
    merged = merge(bag_text, codex_text, model=args.model)
    elapsed = time.time() - t0
    print(f"merge produced {len(merged)} chars in {elapsed:.1f}s")
    if len(merged) > args.max_chars:
        print(
            f"WARN: merged exceeds {args.max_chars} chars (got {len(merged)}). Persisting anyway.",
            file=sys.stderr,
        )

    ts = _ts_utc()
    run_dir = Path(args.out_root) / f"hybrid_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    sys_md = run_dir / "best_candidate.system.md"
    sys_md.write_text(merged)

    cand_json = run_dir / "best_candidate.json"
    cand_json.write_text(json.dumps({"system": merged}, indent=2))

    meta = {
        "timestamp": ts,
        "method": "hybrid_merge",
        "sources": [bag_path.parent.name, codex_path.parent.name],
        "source_paths": [str(bag_path), str(codex_path)],
        "elapsed_seconds": round(elapsed, 2),
        "merger_model": args.model,
        "chars": len(merged),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote: {sys_md}")
    print(f"wrote: {cand_json}")
    print(f"wrote: {run_dir / 'run_meta.json'}")
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
