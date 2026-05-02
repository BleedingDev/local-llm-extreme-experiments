# Trace-RAG: Runtime ANN Lookup Over Personal Trace History

**Author:** Brainstorm Round-2 #G   **Owner file:** this doc only

## TLDR

- Embed all ~30K `context` fields once; at runtime call `lookup_similar_situation(current_context, k=8)` to fetch what the user/agent actually did in similar past spots — RAG over **personal** history.
- Reuse the existing `ir-expo/services/warp-service` ColBERT/WARP stack (`lightonai/GTE-ModernColBERT-v1`, 128-dim, CPU `xtr-warp-rs`) — no new infra.
- Expose as MCP tool *and* trace-gepa internal API, so Claude Code and BAG can ask "what did I do last time?" before acting.
- Distinct from Round-1 (SFT, cross-agent, persona, MCP-from-patterns, open release): this is **runtime retrieval**, complementary to all five.

## Path
`trace-gepa/proposals/trace_rag.md`

## Hypothesis
Prompt engineering compresses *general* policy. RAG over traces answers "in a near-identical situation, what did I actually do, and did it work?" Many BAG/CC failures (re-asking clarifiers, re-discovering repo layout, re-running flaky commands) are episodic, not policy gaps. Predicted lift: 5-15% on long-tail bench tasks with near-neighbours in the corpus.

## Proposal
**Embedding model:** `lightonai/GTE-ModernColBERT-v1` — already smoke-tested in `ir-expo/services/warp-service`, 128-dim token matrices, late-interaction beats single-vector on short jargon-heavy `user_request`. Fallback: `nomic-embed-text-v1.5` MLX-quantized for a single-vector baseline.

**Disk:** 30K × ~80 tokens × 128 dims × fp16 = **~600 MB raw**; WARP-compressed ~150-250 MB. MiniLM baseline ~45 MB.

**Index:** `xtr-warp-rs` (vendored in `ir-expo`); sqlite-vss for fallback.

**API:** `lookup_similar_situation(user_request, recent_actions, k=8, filters={src,label,failure_category}) -> [{score, context, observed_action, label, next_user_message, failure_category}]`.

**Exposure:** MCP tool `trace_lookup`; Python helper for BAG planner; CLI for debugging.

## Implementation
1. `scripts/build_trace_index.py` streams `dataset_v2.jsonl` + `cc_dataset.jsonl` + `codex_dataset.jsonl` through warp-service `/encode_document`.
2. `XTRWarp.create(...)` → `artifacts/trace_index/`.
3. `rag/lookup.py` wraps `XTRWarp.search`.
4. Stdio MCP server in `trace-gepa/mcp/trace_rag_server.py`.
5. A/B eval: BAG with vs. without `trace_lookup`; metrics pass-rate, tokens-to-solution, clarifier-rate.

## Effort / ROI / Critique
~3 days. ROI high: reuses `ir-expo` infra, composes with every Round-1 idea (SFT can call retrieval; persona becomes a filter), updatable in seconds where SFT needs a retrain.

**Self-critique:** Trace-RAG is the non-parametric sibling of SFT; if SFT wins decisively at 30K we're redundant — but we must dedupe bench tasks against the index by source-path or we're just retrieving the answer.
