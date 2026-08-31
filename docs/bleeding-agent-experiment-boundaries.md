# BleedingAgent Experiment Boundaries

Generated for execution graph `bleeding-agent-quality-execution-v1`.

## Runtime Rule

Production ACP runtime must not silently depend on machine-local files, absolute user paths, generated optimizer artifacts, or trace indexes. Experimental integrations are allowed only when explicitly opted in or mediated through optimizer policy.

## MCP Local Config

`.mcp.json` is local-only configuration and is ignored by Git. The reusable server shape lives in `.mcp.example.json`.

The current example covers the optional `trace-rag` MCP server with placeholders:

- Python command under the local GEPA environment.
- `PYTHONPATH` pointing at `trace-gepa`.
- `TRACE_RAG_INDEX_DIR` pointing at a local generated RAG index.

This keeps useful setup information available without committing absolute `/Users/...` paths as product source-of-truth.

## Optimized Prompts

Optimized executor prompt loading is opt-in:

- Enable with `BAG_USE_OPTIMIZED_PROMPT=1`.
- Disable or override with `BAG_DISABLE_OPTIMIZED_PROMPT=1`.

An optimized prompt artifact under `trace-gepa/artifacts/optimized-prompts/latest` no longer affects runtime by existing on disk. Promotion must be explicit.

## Trace-RAG Shim

Trace-RAG lookup is opt-in:

- Enable with `BAG_USE_TRACE_RAG=1`.
- Optional overrides: `BAG_REPO_ROOT`, `TRACE_RAG_PY`, and `TRACE_RAG_INDEX_DIR`.

When the opt-in flag is absent, `lookupSimilarSituation()` returns no hits and does not spawn Python. This keeps the ACP runtime fail-closed when local trace indexes or Python dependencies are missing.

## Verification

Relevant tests:

```bash
bun test tests/optimized-prompt-loader.test.ts
bun test tests/prompts-loader.test.ts
bun test tests/prompt-artifact-bridge.test.ts
```

The latest full suite also passed with:

```bash
bun test tests
```

Result: `547 pass`, `0 fail`.
