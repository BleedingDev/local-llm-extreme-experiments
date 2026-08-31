# BleedingAgent Packaging Inventory

Date: 2026-05-04
Graph: `bleeding-agent-evidence-flywheel-v1`
Plan lane: `repo-packaging-repro` / `packaging-inventory`

## Summary

Wave 1 found a clear packaging problem: the BleedingAgent product source, self-evaluation harness, local benchmark evidence, generated trace artifacts, and personal runtime config are currently mixed in one dirty working tree. The next lanes can proceed only if we treat this repo as two boundaries:

- Product source of truth: TS runtime, ACP backend, optimizer/replay/eval harness, tests, package config, shared examples, and selected product docs.
- Local or research evidence: `.bag`, `bench`, `trace-gepa` raw data/indexes, Python envs, generated artifacts, vendor checkouts, logs, model benchmark outputs, and local MCP config.

No Wave 1 scout found repo-local model weights that should be preserved as source. The largest disk consumers are reinstallable or generated: `bench` (`1.4G`), `trace-gepa` (`1.3G`), `.venv-gepa` (`1.0G`), `.venv` (`603M`), `node_modules` (`212M`), `artifacts` (`62M`), and `vendor` (`43M`).

## Required Product Boundary

The clean checkout must intentionally include these files or groups:

- Package/build config: `package.json`, `package-lock.json`, `tsconfig.json`, `tsconfig.test.json`, `rspack.config.ts`, `.mcp.example.json`, `requirements.txt`.
- Core TS runtime: `src/index.ts`, `src/config.ts`, `src/types.ts`, `src/artifacts.ts`, `src/telemetry.ts`, `src/metrics.ts`, `src/llm*.ts`, `src/auth/**`, `src/workspace*.ts`, `src/pipeline.ts`, `src/interview.ts`, `src/prd.ts`, `src/dag.ts`, `src/parallel-orchestration.ts`.
- Coding-agent behavior: `src/autonomous-coding-turn.ts`, `src/autonomous-tools.ts`, `src/dag-tool-loop.ts`, `src/instruction-*.ts`, `src/pre-submit-self-check.ts`, `src/scratch-hygiene.ts`, `src/task-shape-router.ts`, `src/verifier-signature-library.ts`, `src/codebase-index/**`, `src/audit/**`.
- ACP runtime: `src/acp-agent.ts`, `src/acp/**`.
- MCP runtime boundary: `src/mcp/runtime-tools.ts` and `src/acp/mcp-bridge.ts`.
- Self-evolving harness: `src/optimizer/**`, `src/replay/**`, `src/eval-harness/**`, `src/edit-strategy/**`, `src/edit-strategies/**`, `src/source-adapters/**`, `src/knowledge/**`, `src/prompts/**`, `src/trace-analysis.ts`, `src/trace-store.ts`, `src/harness-gates.ts`, `src/optimize.ts`, `src/self-optimize.ts`.
- SDK embedding surface: `src/sdk/agent-session.ts`.
- Tests that validate the product boundary: `tests/**`, with benchmark-only tests called out in docs rather than silently dropped.

## Review Before Shipping

These files are useful, but should stay explicit because they bridge to generated or personal evidence:

- `src/codex-trace-distilled.ts`: generated from personal Codex sessions; keep only with declared provenance.
- `src/trace-rag-shim.ts`: optional Python `trace-gepa` sidecar; must remain opt-in.
- `src/optimized-prompt-loader.ts`: opt-in prompt artifact bridge; must not silently replace runtime prompts.
- `src/optimizer/dataset-adapter.ts`: depends on `bench/.bag/optimizer/dataset.jsonl`.
- `src/optimizer/failure-clusters.ts`: depends on `bench/.bag/optimizer/failure-clusters.json`.
- `src/optimizer/mipro-baseline.ts`: optional DSPy/MiPRO baseline path, not the primary GEPA path.
- `bench/bag-runtime/**`: drifted runtime copy with different dependency resolutions; do not treat it as source of truth without a separate plan.

## Local-Only Or Generated Boundary

Keep these out of product package outputs and normal source review:

- Runtime traces and telemetry: `.bag/**`, `bench/.bag/**`.
- Local config and secrets: `.env`, `.env.*`, `.mcp.json`, `bag.config.json`.
- Reinstallable dependencies and envs: `node_modules/**`, `bench/bag-runtime/node_modules/**`, `.venv/**`, `.venv-gepa/**`, `bench/.venv/**`.
- Build outputs and local vendor state: `dist/**`, `vendor/**`, `bench/vendor/**`.
- Benchmark outputs: `artifacts/**`, `bench/jobs/**`, `bench/aider_polyglot/results/**`, `bench/ablation/results/**`.
- Trace-GEPA generated data: `trace-gepa/data/*.jsonl`, `trace-gepa/data/sanitised/**`, `trace-gepa/artifacts/rag_index*`, `trace-gepa/artifacts/anomaly_*.pkl`, `trace-gepa/artifacts/real_smoke/**`, Python caches.
- Local orchestration state: `.codex/plan-graphs/**`, `.cursor/*.out`, `.oracle_probe.txt`, `.claude/worktrees/**`.

## Trace-GEPA Boundary

`trace-gepa` is valuable evidence infrastructure but should not silently affect product runtime.

Ship or track deliberately:

- `trace-gepa/mcp_servers/trace_rag/**` if the Trace RAG MCP server is a product-supported optional integration.
- Small benchmark corpora under `trace-gepa/data/benchmarks/**`.
- Curated optimized prompt artifacts and metadata under `trace-gepa/artifacts/optimized-prompts/**`, excluding generated `gepa_state`.
- Curated JSON benchmark results when they are referenced by release evidence.

Do not ship as product source:

- raw or sanitized trace datasets,
- generated RAG indexes,
- anomaly pickle models,
- smoke outputs,
- local Python envs and caches.

## Reproducibility Requirements

Current clean-install blockers:

- `package.json`, `package-lock.json`, `tsconfig.json`, and `rspack.config.ts` are untracked in the current working tree.
- Root JS dependencies are npm-lock based, while tests require Bun. Node/npm/Bun versions must be documented or pinned.
- Python GEPA dependencies are environment-derived; there is no tracked `trace-gepa/requirements.txt` or Python lockfile.
- `.mcp.json` must stay ignored local config; `.mcp.example.json` is the shared template.

Minimum verification after packaging boundary edits:

- `npm run typecheck`
- `bun test tests`
- `npm run build`
- `npm run acp:verify-consumers`

Verification after boundary edits:

- `npm install --package-lock-only --ignore-scripts`: passed, `0` vulnerabilities.
- `npm run typecheck`: passed.
- `bun test tests`: passed, `557` tests across `87` files.
- `npm run build`: passed, Rspack compiled successfully.
- `npm run acp:verify-consumers`: passed against installed Glass and Zed launch targets; the `/chat Ahoj, co umis?` probe produced `0` filesystem reads, `0` filesystem writes, `0` terminal creates, and `0` permission prompts.
- `npm pack --dry-run --json`: passed with `22` entries, `221634` bytes packed, and no `.codex`, `.bag`, `bench`, `trace-gepa`, venv, vendor, local config, or generated artifact directories in the package.

## Next Lanes

After this inventory, `packaging-source-boundary` should lock package publishing boundaries and ignore rules without staging caches. `packaging-local-config` should preserve `.mcp.json` as local-only and document `.mcp.example.json`. `packaging-clean-gates` should then run the writing build and ACP verifier from the stabilized boundary.
