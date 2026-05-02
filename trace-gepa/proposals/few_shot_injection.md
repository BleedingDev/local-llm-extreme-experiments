# Few-Shot Injection from Trace-RAG (Round-8 #EE)

## TLDR
- Inject top-3 similar past traces (TF-IDF retrieved from existing 8K-record `artifacts/rag_index_v2/`) as in-context exemplars in the per-cat v2 system prompt.
- Per-task query = `user_request + last_2_actions`; filter hits with sim < 0.05; format as "Example N: user said X -> agent did Y -> follow-up Z".
- Hypothesis: real per-user trace exemplars lift Opus pass@1 by 5-10pp over rules-only per-cat v2 (41.1% -> ~46-51%).
- Cost: sub-ms lookup + ~500 tokens/task (~+50% Opus prompt cost); break-even at >5pp lift.

## Hypothesis
Per-cat v2 rules describe *what* to do abstractly; few-shots show *how this user actually does it*. Action-selection benchmarks (ToolBench, MINT) consistently show 5-15pp gains from k=3 retrieved few-shots over zero-shot rules. Real traces are higher-signal than synthetic exemplars because they match the user's tool vocabulary, naming style, and follow-up patterns.

## Design
1. **Query builder** (`trace-gepa/few_shot.py`):
   - Concatenate `task.user_request` + `"\n".join(task.recent_actions[-2:])`.
   - Truncate to 512 chars to keep TF-IDF sparse-vec stable.
2. **Retrieval**: import `agent_opt.rag.index.TraceIndex`, load from `artifacts/rag_index_v2/`, call `.query(text, k=N)`. Drop hits with `score < 0.05` (empirical floor where TF-IDF goes noisy on this index).
3. **Formatting**: each hit -> 3-line block `User: {req}\nAgent: {action_summary}\nFollow-up: {next_user_msg or "<task complete>"}`. Cap each example at 200 tokens.
4. **Injection point**: appended to per-cat v2 system prompt under `## Past examples from your history` header. (Tested vs. separate user-role message in pilot; system-append wins because cache-friendly across tasks sharing a category.)
5. **CLI**: extend `bench/run.py` with `--few-shot-index PATH` (default `""` = off) and `--few-shot-k N` (default 0). Empty default = strict backward compat with v2 baselines.

## A/B Methodology
Three arms over the same 200-task eval split:
- A: baseline zero-shot
- B: per-cat v2 (rules only) -- known 41.1%
- C: per-cat v2 + few-shot k=3
Report pass@1 delta C-B with bootstrap 95% CI (n=1000). Secondary: per-category lift heatmap (expect strongest gains in long-tail categories where rules are thin).

## Cost
- Index: already on disk, ~0.3ms/query.
- Tokens: ~500 input/task. At Opus $15/Mtok input -> ~$0.0075 extra/task. 200-task run = $1.50 extra. Trivial.
- Worth it if pass@1 lift >= 5pp. Below 3pp -> abandon; 3-5pp -> keep behind flag.

## Effort + ROI
- Effort: ~120 LOC (query builder + formatter + CLI plumbing) + 1 eval run. ~3 hours.
- ROI: high. Reuses round-2 RAG infrastructure verbatim; orthogonal to per-cat rules so they compound.

## Self-critique
TF-IDF retrieval may surface lexically-similar but semantically-irrelevant traces, polluting context and *hurting* harder tasks -- mitigate with sim>=0.05 floor and a fallback k=0 path when no hits clear it.

Path: `trace-gepa/proposals/few_shot_injection.md`

Self-critique: TF-IDF lexical match can pull semantically-wrong exemplars and degrade hard tasks; sim floor + k=0 fallback is the only guard, and may not be enough on adversarial categories.
