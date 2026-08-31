# BleedingAgent Ownership Manifest

Generated for execution graph `bleeding-agent-quality-execution-v1`.

## Purpose

This manifest defines the current repository source-of-truth boundary before the quality stabilization work continues. The immediate problem is not missing code; it is that the product code, optimizer harness, local benchmark workspaces, generated traces, and dependency caches are mixed in one dirty worktree.

The goal is to make BleedingAgent reproducible as an ACP coding-agent backend with a measured self-evaluation and self-optimization harness. ACP clients such as Glass or Zed are consumers, not product-specific runtime anchors.

## Current Inventory

Observed from read-only local commands:

- Repository size: about `4.5G`.
- Tracked files: `336`.
- Tracked `src/**` files: `7`.
- Local `src/**` files: `128`.
- Tracked modified files: `12`, with no staged files.
- Untracked entries/files: `3,653`.
- Ignored files: `115,217`.
- Dirty status by top-level path: `tests` 85, `src` 69, `docs` 51, `trace-gepa` 47, `scripts` 32, plus package/config files.
- Large local areas: `bench` about `1.4G`, `trace-gepa` about `1.3G`, `.venv-gepa` about `1.0G`, `.venv` about `603M`, `node_modules` about `212M`.
- Product package surface exists locally: `package.json`, `package-lock.json`, `rspack.config.ts`, `tsconfig.json`, `tsconfig.test.json`.
- `.gitignore` already ignores core dependency/build outputs such as `.venv/`, `node_modules/`, `dist/`, `artifacts/`, `vendor/`, `.bag/`, and selected trace-gepa generated datasets/indexes.

## Source-Of-Truth Classes

### Product Runtime

These paths are the core BleedingAgent product surface and should be treated as source-of-truth candidates:

- `package.json`
- `package-lock.json`
- `rspack.config.ts`
- `tsconfig.json`
- `tsconfig.test.json`
- `src/index.ts`
- `src/acp-agent.ts`
- `src/acp/**`
- `src/sdk/**`
- `src/config.ts`
- `src/workspace.ts`
- `src/workspace-paths.ts`
- `src/types.ts`
- `src/artifacts.ts`
- `src/telemetry.ts`
- `src/metrics.ts`
- `src/llm.ts`
- `src/llm-pi-ai.ts`

Ownership: production runtime lane. These files should be tracked or deliberately excluded before release proof. Changes here need typecheck and Bun tests.

### Coding-Agent Behavior Modules

These are product code when wired into ACP runtime or coding flow:

- `src/autonomous-coding-turn.ts`
- `src/autonomous-tools.ts`
- `src/dag.ts`
- `src/dag-tool-loop.ts`
- `src/interview.ts`
- `src/instruction-summarizer.ts`
- `src/instruction-verifier.ts`
- `src/pipeline.ts`
- `src/prd.ts`
- `src/pre-submit-self-check.ts`
- `src/task-shape-router.ts`
- `src/verifier-signature-library.ts`
- `src/scratch-hygiene.ts`
- `src/codebase-index/**`

Ownership: product behavior lane. Downstream work should keep these separate from benchmark-only harnesses.

Conflict risk: `src/autonomous-coding-turn.ts` is tracked and has a large local behavioral delta. Treat it as a single-owner file during implementation waves.

### Edit Strategy And Routing

These are core to the self-optimizing coding-agent objective:

- `src/edit-strategy/**`
- `src/edit-strategies/**`
- `src/acp/edit-lifecycle.ts`
- `src/acp/edit-routing.ts`
- `src/acp/edit-telemetry.ts`
- edit-related tests under `tests/edit-*` and `tests/acp-coding-edit-routing.test.ts`

Ownership: edit optimization lane. Do not collapse this area to one hardcoded edit strategy. Strategy selection must remain measurable per model, codebase, task shape, and trace evidence.

Conflict risk: singular `src/edit-strategy/**` is the canonical contract/apply/eval surface. Plural `src/edit-strategies/registry.ts` is an autonomous runtime dispatcher. Do not merge those concepts casually.

### Optimizer, GEPA, Replay, And Evals

These are product-adjacent harness code and can become production source-of-truth if tests and boundaries are clean:

- `src/optimizer/**`
- `src/eval-harness/**`
- `src/replay/**`
- `src/source-adapters/**`
- `src/prompts/**`
- `src/knowledge/**`
- `src/trace-analysis.ts`
- `src/trace-store.ts`
- `src/codex-trace-distilled.ts`
- matching tests under `tests/optimizer-*`, `tests/eval-harness-*`, `tests/replay-*`, `tests/source-adapters-*`, `tests/knowledge-*`, and prompt tests.

Ownership: optimizer/replay harness lane. This code must preserve redaction, train/dev/hidden-holdout split discipline, and promotion/rollback boundaries. Optimizers may propose prompt/tool/edit/verification policies, not silently rewrite runtime source.

### MCP Runtime

These are source-of-truth candidates for MCP integration:

- `src/mcp/**`
- `src/acp/mcp-bridge.ts`
- `tests/mcp-runtime-tools.test.ts`
- MCP-related ACP tests.

Ownership: live MCP loop lane. MCP must be exposed through canonical contracts, permission/side-effect policy, bounded results, stable errors, and telemetry.

### Tests

The local `tests/**` tree is broad and appears to be product-quality intent rather than scratch:

- ACP tests cover surface, tool boundary, maintenance, slash routing, planning, path policy, and coding edit routing.
- Optimizer tests cover GEPA operations, promotion, registry, materialization, policy resolution, validator, tool rendering, and evidence.
- Replay/eval/source-adapter tests cover core self-improvement plumbing.

Ownership: source-of-truth candidate, but currently mostly untracked. Tests should be tracked in the same stabilization slice as the source files they protect.

### Documentation And Reports

Documentation should be split into:

- Product docs: `README.md`, `docs/bleeding-agent.md`, operator runbooks, ACP interop docs, release readiness docs.
- Research reports: model benchmark reports, competitive comparisons, edit strategy research, GEPA/trace mining reports.
- Historical local reports: one-off benchmark notes and exploratory docs.

Ownership: docs lane. Product docs should be tracked when they describe supported behavior. Research reports can be kept if they serve as evidence, but should not become runtime prerequisites.

## Experiment And Quarantine Classes

### Trace-GEPA Python Tree

`trace-gepa/**` is mixed:

- Python optimizer experiments and tests: `trace-gepa/agent_opt/**`, `trace-gepa/tests/**`, `trace-gepa/bench/**`.
- Generated datasets: `trace-gepa/data/*.jsonl`, `trace-gepa/data/sanitised/**`.
- Derived RAG/model artifacts: `trace-gepa/artifacts/rag_index*`, `trace-gepa/artifacts/*.pkl`.
- Proposal and research docs: `trace-gepa/proposals/**`, top-level markdown reports.

Ownership: experimental optimizer research unless explicitly promoted. The TypeScript product may depend on exported artifacts or adapters only through a documented, opt-in boundary. Failing trace-gepa pytest should not block product release unless that boundary is declared in scope.

### Benchmark Harnesses

`bench/**` is benchmark/research infrastructure, not product runtime:

- `bench/jobs/**` contains run workspaces, logs, configs, and result files.
- `bench/.venv/**` is a rebuildable dependency environment.
- `bench/vendor/**` contains external benchmark dependencies.
- `bench/bag-runtime/**` is a copied runtime snapshot and should not become source-of-truth.

Ownership: benchmark evidence lane. Preserve selected summaries/results if reports cite them; otherwise treat as local or regenerable.

### Dependency And Build Caches

These are not source-of-truth:

- `.venv-gepa/**`
- `.venv/**`
- `bench/.venv/**`
- `node_modules/**`
- `dist/**`
- `.pytest_cache/**`
- `__pycache__/**`
- `.bag/**`
- `artifacts/**`

Ownership: local cache/build output. Safe to ignore; deletion/compression should be handled separately from this plan and only when the user asks for cleanup.

### Local-Only Config And Secrets

These must stay local and out of source-of-truth:

- `.env`
- `.env.*`
- `bag.config.json`
- `bench/bag-runtime/.env`
- machine-local MCP config such as `.mcp.json` if it contains absolute local paths.

Ownership: local operator config. Provide examples instead of committing machine paths.

## High-Risk Paths

- `.mcp.json`: currently tracked and likely machine-local; should become example/local override if it contains absolute paths.
- `package-lock.json`: untracked but important for reproducible `node_modules`.
- `tsconfig.json`: untracked product config, required for green gates.
- `src/sdk/agent-session.ts`: known typecheck blocker from prior research.
- `src/trace-rag-shim.ts`: ESM path handling risk and trace-RAG boundary issue.
- `src/optimized-prompt-loader.ts`: default-on optimized prompt behavior risk.
- `src/source-adapters/canonical.ts` and `src/source-adapters/boundary.ts`: adapter routing risk for `cc-session-jsonl-v2`.
- `src/acp-agent.ts`: high fan-in ACP facade; avoid concurrent edits unless explicitly serialized.
- `src/acp/coding-runner.ts`: crosses edit routing, file IO, terminal verification, replay capture, telemetry, and optimizer triggers.
- `src/optimizer/session-pin.ts`: session-time policy pin boundary between runtime and optimizer artifacts.
- `trace-gepa/data/**`: provenance-sensitive; large and mixed raw/sanitised datasets.
- `trace-gepa/artifacts/anomaly_*.pkl`: derived model artifacts that are untracked and not clearly ignored.
- `bench/jobs/**`: benchmark workspaces and logs; useful evidence but not product source.
- `bench/bag-runtime/**`: copied product runtime; should not fork source-of-truth.

## Cleanup And Ignore Recommendations

Do not delete anything as part of this ownership lane. The next cleanup pass should be evidence-preserving and reversible.

Recommended ignore/quarantine candidates:

- Add or confirm ignore rules for `trace-gepa/artifacts/*.pkl` if anomaly models are derived.
- Keep ignoring `trace-gepa/data/*.jsonl` and `trace-gepa/data/sanitised/**`, but preserve small split manifests and generation summaries if they are needed to reproduce evals.
- Keep `bench/jobs/**`, `bench/.venv/**`, `bench/vendor/**`, `bench/aider_polyglot/results/**`, and `bench/bag-runtime/**` out of product source-of-truth.
- Keep `node_modules/**`, `.venv*/**`, `dist/**`, `.bag/**`, `artifacts/**`, `.pytest_cache/**`, and `__pycache__/**` ignored.
- Convert machine-local `.mcp.json` into `.mcp.example.json` plus ignored local config if downstream quarantine confirms absolute local paths.

## Merge Readiness Criteria

Before claiming product readiness:

1. Product runtime and tests are tracked or explicitly excluded.
2. `package-lock.json`, `tsconfig.json`, `tsconfig.test.json`, and `rspack.config.ts` ownership is decided.
3. `npm run typecheck` is green for the product scope.
4. `bun test tests` is green for tracked product tests.
5. Trace-GEPA pytest is either repaired for declared scope or quarantined as experiment-only.
6. `.mcp.json` and trace-RAG behavior are local/opt-in rather than hidden production requirements.
7. Benchmark and trace artifacts are documented as evidence, not runtime dependencies.

## Current Decision

Treat `src/**`, `tests/**`, `package*.json`, `tsconfig*.json`, `rspack.config.ts`, and selected product docs as source-of-truth candidates for BleedingAgent. Treat `bench/**`, dependency environments, run artifacts, and most `trace-gepa/data` and `trace-gepa/artifacts` outputs as benchmark/experiment artifacts unless a downstream lane explicitly promotes a narrow part through a documented boundary.

Use ACP runtime as the production source of truth for live behavior: `src/acp-agent.ts` plus `src/acp/**`. Use optimizer/GEPA as the source of truth for policy artifacts, candidate generation, active pointers, checkpoints, and promotion gates. GEPA lanes should not rewrite runtime source; they should write/promote optimizer artifacts pinned into new ACP sessions through `src/optimizer/session-pin.ts`.
