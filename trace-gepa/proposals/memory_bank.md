# Proposal: Structured Memory Bank from Traces

## TLDR
- Replace flat CLAUDE.md with a **queryable sqlite memory bank** + Markdown index, populated by an LLM extractor over trace records.
- Five typed tables (`memory_user_role`, `memory_feedback`, `memory_project`, `memory_reference`, `memory_workflow`) with provenance + confidence.
- Agents fetch only **relevant rows** at session start (`user_role` + active `project`) and on-demand via `query_memory(scope, type)` MCP tool — beats reading 5KB of static doc.
- Nightly re-extraction once >=50 new traces accumulate; dedupe + supersede prior rows.

## Hypothesis
Retrieval-conditioned, structured memories yield higher precision guidance than monolithic prose because (a) the agent loads only what is relevant, (b) rows carry provenance for verification, and (c) confidence allows decay/override.

## Schema
```sql
CREATE TABLE memory (
  id TEXT PRIMARY KEY,
  type TEXT CHECK(type IN ('user_role','feedback','project','reference','workflow')),
  scope TEXT,                -- 'global' or '<repo-slug>'
  title TEXT,
  body TEXT,                 -- short, imperative
  source_records JSON,       -- ['rec_018a..', ...]
  confidence REAL,           -- 0..1, derived from frequency + recency
  created_at TIMESTAMP, updated_at TIMESTAMP,
  superseded_by TEXT
);
CREATE INDEX idx_mem_scope_type ON memory(scope, type);
```

## Pipeline
1. Iterate trace records grouped by session.
2. Per session, extractor LLM emits candidate `(type, scope, title, body)` tuples with cited record IDs.
3. Embed `title`; cluster with cosine>=0.85 to dedupe; merge `source_records`, recompute confidence = `min(1, log1p(n_sources)/3) * recency_decay`.
4. Validator pass: drop rows lacking >=2 supporting records OR contradicted by newer high-confidence row (auto-supersede).

## Access
- **Session start**: `SELECT * FROM memory WHERE (scope='global' AND type='user_role') OR (scope=$repo)` — typically <30 rows, <1KB.
- **On demand**: MCP `query_memory(scope, type, q?)` with optional FTS over `body`.

## Sample Memories (one per type)
1. **user_role** — *"Researcher prototyping MLX/Gemma kernels; prefers terse code, no emojis, absolute paths in agent threads."* (src: rec_0142, rec_0188, rec_0203)
2. **feedback** — *"Do NOT create README/summary .md files unless explicitly requested; return findings inline."* (src: rec_0091, rec_0157)
3. **project** — *"`supergemma-dflash-ddtree-mlx`: tests via `pytest -x tests/`; uses MLX, not torch; entrypoint `src/super_gemma/run.py`."* (src: rec_0211)
4. **reference** — *"Library IDs cached in `<repo>/library.md`; check before issuing Context7 search."* (src: rec_0044, rec_0079)
5. **workflow** — *"Brainstorm rounds: read prior `proposals/*.md`, write own under <350 words, end with self-critique."* (src: rec_0220, rec_0221)

## Effort + ROI
- ~3 days: schema, extractor prompt, dedupe, MCP tool, nightly cron.
- ROI: O(1) lookup beats O(N) doc scan; per-project scoping prevents cross-contamination across the user's many repos.

## Self-Critique
Extractor hallucinations could pollute the bank; mitigation (>=2 sources + supersede) is necessary but not sufficient — needs a periodic human audit ritual.

## Path
`trace-gepa/proposals/memory_bank.md`
