# MCP Tools Mined from Observed Workflow Patterns

**Author:** Brainstorm Team Member #D
**Date:** 2026-05-01

## 1. Hypothesis

The 26 GB of trace data does not just teach the agent *which* tool to pick — it reveals **multi-step compound workflows** the user repeats across sessions. These compounds are MCP-tool-shaped: each is a deterministic chain with a small input surface, a known happy path, and observed recovery branches when steps fail. **Mining them yields personalized MCP servers no off-the-shelf vendor can ship**, because the patterns are specific to this user's repos, scripts, and habits. Optimising action-selection prompts (the prior workstream) makes the agent pick existing tools better; this proposal makes **new high-level tools** so there is less to pick from in the first place.

## 2. Concrete Proposal

### Mining algorithm
- Input: `dataset.jsonl`, filtered to `Bash` records with `label == "good"`, grouped by `session_id`, ordered by timestamp.
- Tokenise commands into a stable canonical form: strip args that are paths/hashes/timestamps, keep verbs and flags (`bun test`, `git commit -m <MSG>`, `ruff check --fix`).
- Run **PrefixSpan** (sequence pattern mining) for sequences of length 3–10 with `min_support = 10 distinct sessions`.
- Secondary pass: collapse near-duplicates via edit distance ≤ 1 over the canonical token sequence; merge their support sets.
- For each pattern, mine the **observed recovery suffixes**: when step *k* failed (`exit_code != 0`), what command sequence followed before the user re-entered the main path? Store as `on_error[step_k]`.

### Top 5 hypothesised patterns
1. **`lint-then-commit`** — `bun lint && bun test && git add -A && git commit -m <MSG> && git push`
2. **`mlx-bench-run`** — `python scripts/run_bench.py --model <X> && python scripts/plot_bench.py && open figures/<latest>.png` (this repo)
3. **`czech-correction-loop`** — detected via a token signature where the user pastes a Czech phrase, the agent emits a correction, the user replies with one of {`ne`, `spis`, `lepsi`} and the agent retries; encode as `correct_czech(text, max_rounds=3)`.
4. **`gepa-optimize-then-eval`** — `bun run trace-gepa/optimize.ts && bun run trace-gepa/eval.ts --split val && jq '.score' results/*.json`
5. **`worktree-spinup`** — `git worktree add ../<repo>-<feat> -b <feat> && cd ../<repo>-<feat> && bun install && code .`

### MCP tool spec (per pattern)
Each pattern compiles to one tool with:
- **`input_schema`** (zod): the variables that varied across observed runs (e.g. `commit_msg: string`, `model: enum`).
- **`output_schema`**: stdout tail, exit code, and a structured `step_results: Array<{step, ok, ms}>`.
- **`error_recovery`**: the mined `on_error[k]` map; tool retries the suffix once before surfacing failure.

### Server scaffold
- `mcp-servers/observed-patterns/` — TypeScript, Effect-based (`@effect/platform` + `@modelcontextprotocol/sdk`), zod schemas, one file per tool under `src/tools/`.
- Single binary; registered in `~/.claude.json` under `mcpServers.observed-patterns`.

## 3. Implementation Steps (design only)
1. Pattern miner script (`trace-gepa/mine_patterns.py`) → outputs `patterns.json`.
2. Codegen step: `patterns.json` → TS tool stubs (input schema inferred from arg-slot variability).
3. Hand-tune the top 5 (recovery logic, prompts shown to user on ambiguous inputs).
4. Wire into Claude Code via per-project `.mcp.json`; A/B against the action-selection-prompt baseline by replaying held-out sessions.

## 4. Effort Estimate
**3–5 days**: 1 day mining + codegen, 2 days server scaffold and 5 hand-tuned tools, 0.5 day wiring/A-B.

## 5. ROI & Honest Critique
**Win vs. plain CLI scripts:** an MCP tool carries a typed schema the agent reads, a recovery branch the agent does not have to re-invent, and a name the agent surfaces in completions — a shell script is invisible until the agent thinks to run it. Tools also compose: the agent can call `lint_then_commit` *as one step* inside a larger plan, instead of budgeting 5 turns of orchestration.

**Honest critique:** (a) frequency ≠ utility — the most-repeated chain may be repeated *because* it is annoying, not because it is good; mine for *successful* chains, not just frequent ones. (b) Over-fitting risk: a tool baked to this user's exact 5-step ritual fossilises a workflow that should evolve; mitigate by re-mining monthly and decaying old patterns. (c) Discoverability tax: the agent must *learn* the new tools exist — this circles back to the action-selection prompt the prior workstream optimised, so the two efforts are complementary, not substitutes.
