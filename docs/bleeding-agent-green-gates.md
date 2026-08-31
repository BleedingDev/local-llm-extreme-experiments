# BleedingAgent Green Gates

Generated for execution graph `bleeding-agent-quality-execution-v1`.

## Scope Decision

The product green gate is the TypeScript ACP/runtime/optimizer/replay workspace plus the Bun test suite under `tests/**`.

`trace-gepa/**` remains experimental optimizer research, but its scoped RAG tests are useful because the TypeScript runtime has an optional trace-RAG shim. The RAG test path is therefore a scoped experiment gate, not a production release blocker unless trace-RAG is explicitly enabled.

## Current Green Command Set

Run these from the repository root:

```bash
npm run typecheck
bun test tests
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest trace-gepa/agent_opt/rag/test_rag.py
```

## Latest Results

- `npm run typecheck`: passed.
- `bun test tests`: passed, `547 pass`, `0 fail`, across `87` files.
- `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest trace-gepa/agent_opt/rag/test_rag.py`: passed with `2 passed`, `1 skipped`.

The trace-RAG skip is expected in this workspace Python when `scipy` and `sklearn` are unavailable. The deterministic `build_query_text` tests still run without indexing dependencies, and the dependency-bound TF-IDF build/query smoke is explicitly skipped instead of failing during import.

## Fixes Included

- `src/sdk/agent-session.ts`: SDK event mapping now covers `retry_hint` and `edit_dispatch` autonomous trace variants and retains an exhaustive check.
- `trace-gepa/agent_opt/rag/embed.py`: added `build_query_text` and deferred indexing imports so deterministic tests can run without optional indexing dependencies.
- `trace-gepa/agent_opt/rag/test_rag.py`: updated expectations from old dense embedding artifacts to the current TF-IDF index contract.

## Gate Ownership

- TypeScript typecheck and Bun tests are product gates.
- Trace-GEPA RAG pytest is scoped to the trace-RAG experiment boundary.
- Full trace-GEPA pytest is not currently a product gate until the Python experiment tree is explicitly promoted.
