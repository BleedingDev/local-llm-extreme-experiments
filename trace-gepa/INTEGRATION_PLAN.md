# Trace-RAG → BAG runtime integration plan

## What's already wired

- **MCP route** (Claude Code): `.mcp.json` registers `trace-rag.lookup_similar_situation`. Restart Claude Code in the repo and it appears as a callable tool.
- **TS subprocess route** (BAG): `src/trace-rag-shim.ts` exports `lookupSimilarSituation(query, {k, timeoutMs}) -> TraceRagHit[]` and `summariseHitsForPrompt(hits) -> string`. Compiles, smokes clean via Bun.

## Next wiring step (deferred — needs eval pass)

In `src/autonomous-coding-turn.ts:235` (where `loadOptimizedExecutorPrompt` already injects), add:

```ts
import { lookupSimilarSituation, summariseHitsForPrompt } from "./trace-rag-shim";

// just before the LM call:
const hits = process.env.BAG_DISABLE_TRACE_RAG === "1"
  ? []
  : await lookupSimilarSituation(user_request, { k: 3 });
const rag_block = hits.length ? `\n\n${summariseHitsForPrompt(hits)}\n` : "";
const exec_system = (optimized?.system ?? SYSTEM_PROMPT_DEFAULT) + rag_block;
```

Effect: the executor LM sees up to 3 similar past records before each tool decision, with their failure_category and the user's follow-up message — strongest available "what did this user do last time?" signal.

## Eval gate before enabling by default

Run on the 175-task bench with and without the RAG block:

```bash
.venv-gepa/bin/python trace-gepa/bench/run_anthropic.py \
    --tasks trace-gepa/data/benchmark_tasks_full.jsonl \
    --model claude-opus-4-7 --max-workers 8 \
    --output trace-gepa/bench/results/full_eval/opus_seed_no_rag.json

# (with shim wired into the harness's prompt assembler)
.venv-gepa/bin/python trace-gepa/bench/run_anthropic.py \
    --tasks trace-gepa/data/benchmark_tasks_full.jsonl \
    --model claude-opus-4-7 --max-workers 8 \
    --output trace-gepa/bench/results/full_eval/opus_seed_with_rag.json
```

**Promote** if RAG arm beats no-RAG by ≥ +0.020 on overall pass rate (above noise floor at this n).

**Caveat**: the bench tasks were partly extracted from the same trace records the index contains. Need to dedup by `src_path` at eval time so the index can't trivially "answer" by retrieving its own training row.

## Failsafe

`BAG_DISABLE_TRACE_RAG=1` env var disables retrieval (the shim already returns `[]` on error). MCP server can be commented out of `.mcp.json` to disable that route.

## Refresh cadence

Re-run `python -m agent_opt.rag.embed --datasets ... --output artifacts/rag_index_v2` whenever > 1000 new records land. Index build is < 5s.
