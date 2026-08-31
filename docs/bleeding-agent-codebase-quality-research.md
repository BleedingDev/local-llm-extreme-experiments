# BleedingAgent Codebase Quality Research

Date: 2026-05-04
Repo: `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx`
Branch: `master`

## Scope

This research covers the local repository as it exists now: the original local-LLM benchmark workspace, the untracked TypeScript BleedingAgent runtime, the ACP backend, the optimizer and replay harness, the `trace-gepa` Python experiment, benchmark scratch artifacts, and recent Claude Code co-authored commits.

The goal is to identify the work required to reach a maintainable, fully functional ACP coding-agent backend whose main differentiator is self-evaluation and self-optimization per model and per codebase.

## Executive Summary

The repository currently contains a promising BleedingAgent implementation, but its ownership boundary is unclear. The production runtime is mostly untracked, while older model-benchmark code is tracked, and the Claude Code `trace-gepa` commits are tracked but partly broken. This means the codebase can pass `bun test tests` while still failing `npm run typecheck` and `trace-gepa` pytest.

The strongest implemented areas are ACP session behavior, YOLO/Safe policy, edit strategy routing, replay capture schemas, optimizer registry/promotion gates, MCP runtime-tool normalization, and telemetry. The weakest areas are repo hygiene, reproducibility, trace-gepa isolation, Python test health, source adapter integration gaps, and proof that the self-optimization loop improves real coding outcomes rather than only synthetic fixtures.

The right next move is not to add more features. First stabilize the repository boundary, make the verification gates trustworthy, quarantine experimental artifacts, then close the real replay -> GEPA -> promotion -> rollback loop with measured evidence.

## Repository Shape

- Runtime package config exists in [package.json](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/package.json:1), but `git status` reports it as untracked.
- TypeScript compiler config exists in [tsconfig.json](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/tsconfig.json:1) and [tsconfig.test.json](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/tsconfig.test.json:1), but `tsconfig.json` is also untracked.
- Main CLI entrypoint is [src/index.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/index.ts:1).
- ACP backend facade is [src/acp-agent.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp-agent.ts:1).
- Modular ACP runtime lives under [src/acp](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/session.ts:1).
- Optimizer core lives under [src/optimizer](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/registry.ts:1).
- Replay/eval harness lives under [src/replay](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/replay/capture.ts:1), [src/eval-harness](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/eval-harness/types.ts:1), and tests.
- Trace analysis/indexing lives in [src/telemetry.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/telemetry.ts:1), [src/trace-store.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/trace-store.ts:1), and [src/trace-analysis.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/trace-analysis.ts:1).
- Python trace-GEPA experiment lives in [trace-gepa](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/README.md:1).
- Large benchmark scratch data lives under `bench/`, currently untracked and around 1.4 GB.

## Entrypoints

CLI commands in [src/index.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/index.ts:17):

- `bag init`
- `bag doctor`
- `bag interview`
- `bag prd`
- `bag dag`
- `bag run`
- `bag optimize`
- `bag self-optimize`
- `bag apply-optimization`
- `bag acp`
- `bag acp-settings`
- `bag metrics`
- `bag ax-smoke`

ACP server starts through `bag acp`, which calls `startAcpServer` from [src/acp-agent.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp-agent.ts:1). New ACP sessions default to `auto` mode and default YOLO when `policy.requirePermissions` is false in [src/acp/session.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/session.ts:35).

The production coding path is [src/acp/coding-runner.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/coding-runner.ts:1). It selects files, reads through ACP, chooses an edit strategy, applies edits through ACP writes, runs verification commands, records traces, captures replay data, self-evaluates, and triggers background optimization inspection.

The older autonomous tool loop is [src/autonomous-coding-turn.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/autonomous-coding-turn.ts:1). It is still used by SDK/RPC and adaptive tool-use runners, but it is more CLI-shell oriented and should be treated as a compatibility/runtime lane, not the primary ACP edit path.

## Control Flow

### ACP Prompt Routing

[src/acp-agent.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp-agent.ts:314) handles prompt input. Slash commands are handled first. In `chat` mode the agent returns a capability surface. In `auto` mode the LLM-based router in [src/acp/prompt-router.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/prompt-router.ts:25) decides between `chat`, `plan`, and `run` semantically, without language keyword rules. Temporary `run` or `plan` mode restores `auto` afterward through [src/acp/session.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/session.ts:87).

### ACP Coding Run

[src/acp/coding-runner.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/coding-runner.ts:144) performs a full run:

1. Load knowledge and scout candidate files.
2. Build repo context.
3. Select files to read/create.
4. Resolve live edit strategy from optimizer policy.
5. Generate patch.
6. Preview and write edits through ACP.
7. Fallback to another edit strategy on parse/apply/stale/protected-path failures.
8. Check post-apply consistency by re-reading edited files.
9. Run verification commands.
10. Repair up to two rounds.
11. Roll back when verification or consistency remains broken.
12. Persist traces, edit lifecycle attempts, replay capture, self-evaluation, optimization report, and manifest.

The post-apply consistency check is implemented in [src/acp/edit-lifecycle.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/edit-lifecycle.ts:134) and explicitly detects the "edit applied, then file is inconsistent" class.

### Edit Strategy Routing

Canonical edit definitions are in [src/edit-strategy/taxonomy.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/edit-strategy/taxonomy.ts:1). Supported deterministic apply families include whole-file, exact replace, unified diff, apply patch, and hash range in [src/edit-strategy/apply-layer.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/edit-strategy/apply-layer.ts:1). Routing is evidence-driven in [src/optimizer/edit-policy-router.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/edit-policy-router.ts:1), with task-shape constraints and historical metrics, not global model-specific hardcoding.

### Telemetry and Self-Optimization

[src/telemetry.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/telemetry.ts:84) writes JSONL events, metrics, and OpenInference/HALO-style spans. It records steps, LLM calls, tool calls, and detailed edit attempt attributes including strategy IDs, model profile, codebase profile, policy, hashes, phases, post-apply consistency, rollback, repair, fallback, and token/cost dimensions.

[src/self-optimize.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/self-optimize.ts:1) currently analyzes metrics and traces to propose safe config/tool guidance changes. [src/optimizer/gepa-runner.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/gepa-runner.ts:1) implements a bounded GEPA candidate loop. [src/optimizer/gepa-operations.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/gepa-operations.ts:1) defines readiness gates for real replay evidence and auto-promotion safety.

## Validation Results

Commands run locally:

- `bun test tests`: 545 pass, 0 fail.
- `npm run typecheck`: fails.
- `python -m pytest -q agent_opt tests` inside `trace-gepa`: fails during collection.

Current TS blocker:

- [src/sdk/agent-session.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/sdk/agent-session.ts:131) has `traceEntryToAgentEvent(...): AgentEvent` without an exhaustive/default return.

Current Python blocker:

- [trace-gepa/agent_opt/rag/test_rag.py](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/agent_opt/rag/test_rag.py:17) imports `build_query_text`.
- [trace-gepa/agent_opt/rag/embed.py](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/agent_opt/rag/embed.py:42) has `_record_text`, but no public `build_query_text`.
- The same test expects `embeddings.npz`, while [trace-gepa/agent_opt/rag/embed.py](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/agent_opt/rag/embed.py:138) writes `tfidf_matrix.npz`.

## High-Risk Areas

### Repository Ownership Is Ambiguous

Only 7 files under `src` and 2 under `tests` are tracked by git, while the current working tree contains 128 `src` files and 88 `tests` files. This is the highest structural risk because the codebase can appear functional locally while being unreproducible from git.

### Experiment Runtime Boundary Is Too Porous

[.mcp.json](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.mcp.json:1) hardcodes absolute machine-local paths into `.venv-gepa` and `trace-gepa`. [src/trace-rag-shim.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/trace-rag-shim.ts:17) uses `__dirname` in a project configured as ESM. [src/optimized-prompt-loader.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimized-prompt-loader.ts:1) introduces default-on prompt artifact loading. These pieces should be opt-in or registry-controlled until the runtime/product boundary is stable.

### Source Adapter Integration Has a Gap

The CC v2 adapter exists in [src/source-adapters/cc-session-v2.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/source-adapters/cc-session-v2.ts:1), and boundary detection includes it in [src/source-adapters/boundary.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/source-adapters/boundary.ts:8). However the generic canonicalization switch in [src/source-adapters/canonical.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/source-adapters/canonical.ts:137) does not route `cc-session-jsonl-v2` to `canonicalizeCcSessionV2`. This means CC detection and CC canonicalization are split, and tests do not currently cover the generic path.

### MCP Is Strongly Modeled But Not Fully Live-Loop Integrated

MCP runtime tools have normalization, policy, permission, error taxonomy, result bounds, and optimizer feedback in [src/mcp/runtime-tools.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/mcp/runtime-tools.ts:1) and [src/acp/mcp-bridge.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/acp/mcp-bridge.ts:1). The main ACP coding runner does not yet expose ACP-attached MCP tools as model-selectable tools during the same file-edit loop. This is a product gap, not just a test gap.

### Benchmarks And Scratch Artifacts Are Too Large For Mainline Hygiene

Measured sizes:

- `bench`: 1.4 GB
- `trace-gepa`: 1.3 GB
- `.venv-gepa`: 1.0 GB
- repository total: 4.5 GB

The repo needs explicit local-only cleanup/ignore rules for generated benchmark jobs, copied runtime workspaces, `.claude/worktrees`, virtualenvs, and bulky trace outputs.

## Hypotheses

### CONFIRMED: Runtime tests are green but typecheck is not.

Evidence: `bun test tests` passes 545 tests, while `npm run typecheck` fails on [src/sdk/agent-session.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/sdk/agent-session.ts:131).

### CONFIRMED: trace-gepa is not currently green.

Evidence: pytest fails importing `build_query_text` from [trace-gepa/agent_opt/rag/embed.py](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/agent_opt/rag/embed.py:1), while [trace-gepa/agent_opt/rag/test_rag.py](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/agent_opt/rag/test_rag.py:17) requires it.

### CONFIRMED: The current local runtime is not reproducible from tracked git state.

Evidence: `git ls-files src | wc -l` reports 7 tracked `src` files while `find src -type f | wc -l` reports 128 files.

### CONFIRMED: The core design already tracks many optimization dimensions needed for per-model and per-codebase tuning.

Evidence: [src/telemetry.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/telemetry.ts:84), [src/optimizer/session-pin.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/session-pin.ts:1), [src/optimizer/policy-resolver.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/policy-resolver.ts:1), and [src/optimizer/edit-policy-router.ts](/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/src/optimizer/edit-policy-router.ts:1).

### UNRESOLVED: The current local BAG runtime is competitive with Claude Code on real coding tasks.

The repository has many harnesses and docs, but the currently verified tests are mostly deterministic/unit/fixture-level. We still need a release proof run against real coding tasks with explicit scorecards.

## Plan Implications

The plan should start with stabilization and reproducibility. Only after the repo is green and source ownership is clear should we deepen MCP live-loop, replay corpus, and autonomous GEPA operations. The project goal remains an ACP coding-agent backend plus self-evolving harness, not another CLI-first coding agent.

