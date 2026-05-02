# trace-rag MCP server

Exposes one tool, `lookup_similar_situation`, that retrieves up to 20 trace
records similar (TF-IDF cosine + MMR diversity reranker) to a free-text query.
Backed by `trace-gepa/artifacts/rag_index_v2/` (8,264 records after orchestration filter).

## Manual stdio smoke

```bash
cd /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' | \
  PYTHONPATH=trace-gepa TRACE_RAG_INDEX_DIR=trace-gepa/artifacts/rag_index_v2 \
  .venv-gepa/bin/python -m mcp_servers.trace_rag.server
```

## Wire into Claude Code (project-scoped)

`.mcp.json` is committed at repo root with the server already registered. Restart Claude Code in this repo and the `trace-rag.lookup_similar_situation` tool should appear.

## Wire into Claude Code (user-scoped)

Append to `~/.claude.json` under `mcpServers`:

```json
"trace-rag": {
  "command": "/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.venv-gepa/bin/python",
  "args": ["-m", "mcp_servers.trace_rag.server"],
  "env": {
    "PYTHONPATH": "/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa",
    "TRACE_RAG_INDEX_DIR": "/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/artifacts/rag_index_v2"
  }
}
```

## Tool spec

`lookup_similar_situation(query: str, k: int = 5) -> {results: [...]}`

Each result includes:
- `rank`, `similarity` (cosine vs query)
- `observed_tool` — the tool the agent called in that record
- `label` — `good | bad | user_corrected | user_confirmed`
- `failure_category` — e.g., `bash_exit_nonzero`, `hallucinated_path`, `null`
- `user_request_excerpt` — first 200 chars of the user's request at that point
- `next_user_message_excerpt` — first 200 chars of what the user said next (often the corrective signal)
- `src_path`, `id` — provenance

## Refresh the index

```bash
cd /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx
PYTHONPATH=trace-gepa .venv-gepa/bin/python -m agent_opt.rag.embed \
    --datasets trace-gepa/data/dataset.jsonl trace-gepa/data/dataset_v2.jsonl \
    --output trace-gepa/artifacts/rag_index_v2
```

Re-run after large new trace ingestion (>1000 new records).

## Disable

Comment-out or remove the `trace-rag` entry in `.mcp.json` (or the user-scope equivalent), then restart Claude Code.
