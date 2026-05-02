# Proposal: MCP **Resources** Server for the Trace Dataset

**Author:** Brainstorm Round-6 Member #W   **Owner file:** this doc only

## TLDR

- Expose the trace corpus as **MCP resources** (read-only, schema-bound URIs) — distinct from Round-2's `trace-rag` MCP **tool** (TF-IDF retrieval on intent). Resources are listed up-front so the model can pull pre-shaped slices into context at session start; tools fire mid-task.
- Catalogue is parameterised URI templates (`trace://recent/{N}`, `trace://failures/{category}`, `trace://corrections/{token}`, `trace://by-repo/{name}`, `trace://by-tool/{name}`, `trace://workflow-archetype/{label}`) backed by one sqlite metadata index over `dataset_v2.jsonl + cc_dataset.jsonl + codex_dataset.jsonl`.
- Reuses the anomaly-detector index pattern (sqlite + corpus-hash versioning under `artifacts/`) and adds a thin per-resource SQL view; no embeddings, no ANN, no GPU.
- Configured next to existing `trace-rag` entry in `.mcp.json` — both servers coexist, different protocol primitives, different access patterns.

## Path
`trace-gepa/proposals/mcp_resources.md`

## 1. Hypothesis

Tools answer "I'm stuck, retrieve something." Resources answer "before we start, here's relevant prior context." Many CC failures are not stuck-state lookups — they're missing-priors (re-asking what repo conventions are, re-discovering a tool failed last week, re-typing `pnpm` after the user said `bun`). Resources let the host (Claude Code) enumerate available slices and selectively load them into the system prompt — **deterministic, indexed, no similarity threshold to tune**.

## 2. Resource Catalogue

| URI template                          | Returns (JSONL)                                                  | SQL backing                                                 |
| ------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| `trace://recent/{N}`                  | Last N records by `ts` desc                                      | `SELECT … ORDER BY ts DESC LIMIT ?`                         |
| `trace://failures/{category}`         | All records with `failure_category = ?`                          | indexed on `failure_category`                               |
| `trace://corrections/{token}`         | Records where `next_user_message` contains corrective token       | FTS5 over `next_user_message`                               |
| `trace://by-repo/{name}`              | Records where `src_path` or `cwd` matches repo                   | LIKE on `src_path`                                          |
| `trace://by-tool/{name}`              | Records whose `observed_action` invoked tool `?`                 | indexed on extracted `tool_name`                            |
| `trace://workflow-archetype/{label}`  | Sessions matching a mined archetype (e.g. `lint-then-commit`)    | join with `mined_patterns_top30.json` archetype assignments |

Each resource also exposes `mimeType: application/jsonl` and a JSON Schema describing one record (id, src, context, observed_action, label, failure_category, next_user_message, ts).

## 3. Backing

- `scripts/build_resources_index.py` streams the three JSONL datasets into `artifacts/resources_index.sqlite` with FTS5 + B-tree indexes; same corpus-hash versioning as the anomaly detector.
- `mcp_servers/trace_resources/server.py` (stdio) implements `resources/list`, `resources/templates/list`, `resources/read` per the MCP spec; ~200 LOC.

## 4. `.mcp.json`

```json
"trace-resources": {
  "command": "/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.venv-gepa/bin/python",
  "args": ["-m", "mcp_servers.trace_resources.server"],
  "env": {
    "PYTHONPATH": "/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa",
    "TRACE_RESOURCES_DB": "/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/artifacts/resources_index.sqlite"
  }
}
```

## 5. Effort + ROI + Self-Critique

**Effort:** ~1 day. Index build ~80 LOC; MCP server ~200 LOC; six SQL views.
**ROI:** High at low cost — no embeddings, no model. Complementary to (not duplicative of) `trace-rag` tool; together they cover cold-start priors and mid-task retrieval. Workflow-archetype resource alone is novel.
**Self-critique:** Resources are pulled by the host's heuristics; if Claude Code's resource-selection policy is naive it may load `trace://recent/50` every turn, blowing context budget — needs token-budgeted pagination (`?limit=` and `?after_id=`) and per-resource size caps from day one.
