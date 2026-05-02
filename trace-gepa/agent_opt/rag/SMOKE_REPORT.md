# Trace-RAG smoke report

**Path chosen:** Path B (Python-only, sklearn TF-IDF) — abandoned Path A (sentence-transformers + ir-expo WARP) because the install + embedding pass would have exceeded the iteration window (~10 min). TF-IDF runs in seconds and is often as good or better than dense embeddings for retrieving similar trace contexts.

## Index stats

### `artifacts/rag_index/` (full corpus, unfiltered)
- 30,313 records (v1 + v2 datasets concatenated, deduped by id)
- 50,000 vocab entries (TF-IDF, ngram (1,2), sublinear_tf, max_features=50K)
- 9.5M non-zero entries
- 70 MB on disk: 52 MB matrix + 17 MB metadata + 2 MB vectorizer

### `artifacts/rag_index_filtered/` (recommended for retrieval)
- 8,264 records (after dropping 22,049 orchestration-boilerplate entries — `<teammate-message`, `<task-notification`, `<system-reminder`, `<command-name>`, `<command-message>` prefixes)
- 25,521 vocab
- 1.4M non-zeros
- ~22 MB on disk

**73% of v1+v2 records were orchestration boilerplate.** Real-task signal is in the remaining 27%. Use the filtered index for any retrieval-quality work; use the full index only if you specifically need orchestration history.

## 5 sample queries (filtered index)

```
Q: find typescript test files that don't have a corresponding source file
   #1 sim=0.097 tool=Bash | "Phase-2 Build Agent #5 of 6. Read ..."  ← related (source-adapter audit)
   #2 sim=0.092 (variant of #1, subagent fanout)

Q: the user asked me to optimize the GEPA prompt
   #1 sim=0.109 tool=Read | "Read `.codex/plans/bleeding-agent-v1-autonomous-gepa-operations.plan.md` ..."  ← directly relevant

Q: I need to write to artifacts/missing_adapter_tests.txt
   #1 sim=0.107 tool=Read | "I need to understand the architecture for implementing multi-account..."  ← weak match

Q: what should I do when bash command fails with exit code 127
   #1 sim=0.142 tool=Bash | "Team Member 3 of a 10-agent team... mine FAILURE PATTERNS..."  ← meta-relevant

Q: edit a config file with multiple matching old_strings
   #1 sim=0.105 tool=exec_command cat=bash_exit_nonzero | "Read-only root-cause lane. Investigate why libs/platform..."  ← relevant
```

Similarity scores are 0.09-0.14. Low absolute (TF-IDF over short noisy snippets), but relative ranking distinguishes near-relevant from clearly-irrelevant.

## Quality assessment

Retrieval is **mechanically correct** but **moderately useful**:
- Real signal surfaces (3/5 queries had genuinely relevant top-1).
- Top-K results often contain near-duplicates from subagent fan-outs — needs MMR reranking for diversity.
- Generic English queries match orchestration-template phrases better than narrow technical content (TF-IDF limits with short queries against verbose contexts).

## Verdict: **PROCEED with TWEAK**

Improvements queued before BAG-runtime integration:
1. **MMR diversity reranker** — drop top-K duplicates from same `src_path` or with > 0.95 similarity to higher-ranked.
2. **Augment doc text with `next_user_message` and `failure_category`** — captures the corrective feedback signal that's currently weakly weighted.
3. **Fold orchestration filter into `embed.py` as default** (currently it's a one-off in this smoke).

Future upgrade path: dense embeddings via ir-expo's WARP/ColBERT sidecar (when they're ready). Same metadata format means the index can swap underneath without changing the `TraceIndex` API.

## Files

- `agent_opt/rag/embed.py` — TF-IDF index builder (~100 LoC).
- `agent_opt/rag/index.py` — `TraceIndex.query(text, k)` (~30 LoC).
- `agent_opt/rag/cli.py` — `python -m agent_opt.rag.cli --query "..." --k 5` (existing scaffold from prior agent).
- `artifacts/rag_index/` — full unfiltered index.
- `artifacts/rag_index_filtered/` — orchestration-filtered index (recommended).
