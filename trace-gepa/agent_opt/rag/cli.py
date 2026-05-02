"""CLI: python -m agent_opt.rag.cli --query "..." --k 5

Emit JSON to stdout (default) or human-readable text with --pretty.
Resolves index path via TRACE_RAG_INDEX_DIR env first, then --index-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .index import TraceIndex


_HERE = Path(__file__).resolve()
DEFAULT_INDEX = str(_HERE.parents[2] / "artifacts" / "rag_index_v2")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--index-dir",
        default=os.environ.get("TRACE_RAG_INDEX_DIR") or DEFAULT_INDEX,
    )
    p.add_argument("--pretty", action="store_true", help="Emit human-readable text instead of JSON")
    args = p.parse_args(argv)

    idx = TraceIndex(args.index_dir)
    hits = idx.query(args.query, k=args.k)

    if not args.pretty:
        print(json.dumps({"query": args.query, "k": args.k, "results": hits}, indent=2, ensure_ascii=False))
        return 0

    print(f"Query: {args.query}")
    print(f"Index: {args.index_dir} (n={len(idx.metadata)})")
    print(f"Top-{args.k}:")
    for h in hits:
        excerpt = (h.get("user_request_excerpt") or "").replace("\n", " ")
        print(
            f"  rank={h['rank']:>2}  sim={h['similarity']:+.4f}  "
            f"id={h.get('id')}  label={h.get('label')}  cat={h.get('failure_category')}  "
            f"tool={h.get('observed_tool')}\n"
            f"      excerpt: {excerpt[:140]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
