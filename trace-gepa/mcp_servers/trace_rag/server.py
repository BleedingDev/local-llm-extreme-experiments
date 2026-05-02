"""MCP server: trace-rag — retrieve similar past trace records by context text.

Exposes one tool:
  lookup_similar_situation(query: str, k: int = 5)
    -> list of {rank, similarity, observed_tool, label, failure_category,
                user_request_excerpt, next_user_message_excerpt, src_path, id}

Backed by the TF-IDF index at trace-gepa/artifacts/rag_index_v2/.

Run as a Claude Code MCP server (stdio transport):
  python -m mcp_servers.trace_rag.server
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Resolve the repo root so this script works regardless of cwd.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # mcp_servers/trace_rag/server.py -> trace-gepa/ -> repo root
TRACE_GEPA = REPO_ROOT / "trace-gepa"
DEFAULT_INDEX = TRACE_GEPA / "artifacts" / "rag_index_v2"

# Allow importing the index module without installing the package.
sys.path.insert(0, str(TRACE_GEPA))

from agent_opt.rag.index import TraceIndex  # noqa: E402

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402


_INDEX: TraceIndex | None = None


def _index() -> TraceIndex:
    global _INDEX
    if _INDEX is None:
        index_dir = os.environ.get("TRACE_RAG_INDEX_DIR") or str(DEFAULT_INDEX)
        _INDEX = TraceIndex(index_dir)
    return _INDEX


server: Server = Server("trace-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="lookup_similar_situation",
            description=(
                "Retrieve similar past trace records by free-text context. "
                "Returns up to k records describing what action was taken and what the "
                "user said next. Useful when the agent is uncertain and wants to see how "
                "this user has handled similar situations before."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text description of the current situation, request, or stuck point.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results (1-20, default 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "lookup_similar_situation":
        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool {name}"}))]
    query = (arguments or {}).get("query") or ""
    k = int((arguments or {}).get("k") or 5)
    if not query:
        return [TextContent(type="text", text=json.dumps({"error": "query is required"}))]
    results = _index().query(query, k=max(1, min(k, 20)))
    return [TextContent(type="text", text=json.dumps({"results": results}, ensure_ascii=False, indent=2))]


async def _amain() -> int:
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
