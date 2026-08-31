# BleedingAgent Parallel Orchestration

Date: 2026-05-01

This note documents the Parallel Orchestration lane. The goal is not to build a new CLI/TUI subagent frontend. ACP clients remain the frontend. BleedingAgent provides the measured orchestration contracts, isolation policy, merge verification, and optimizer evidence so parallelism can become codebase/model-specific over time.

## Implemented Contract

`src/parallel-orchestration.ts` defines:

- lane contracts for exploration, implementation, and verification lanes;
- side-effect policy per lane: read-only, writes allowed, or terminal allowed;
- target/read path ownership and dependency links;
- conflict detection for overlapping write targets;
- isolation selection:
  - `shared_read_only` for read-only exploration;
  - `patch_queue` for non-conflicting implementation work;
  - `temp_workspace` or `git_worktree` for conflicting implementation work;
  - `dry_run_apply_layer` for verification lanes;
- concurrency policy tied to model/server capacity, model-codebase policy, task risk, edit-conflict risk, tool failure history, and user mode;
- merge verification plans using codebase profile typecheck/test/lint commands;
- orchestration outcome conversion into optimizer evidence bundles.

## Safety Rules

- Read-only exploration lanes can share the live workspace.
- Implementation lanes cannot silently write the same file.
- Conflicting implementation lanes must be isolated before merge.
- Merge verification is required after implementation lanes finish.
- Failed merge verification requires rollback of lane outputs.
- ACP progress labels are derived from lane status so consumers can show the orchestration without owning the runtime policy.
- Lane outcomes such as drift, duplicate work, merge conflict, verifier failure, speedup, cost, latency, and cancellation become optimizer evidence.

## Evidence

Focused check:

```bash
bun test tests/parallel-orchestration.test.ts
```

Current result:

- 5 tests passed.
- 0 failures.
- 18 assertions.

## Boundary

This lane closes the product-level orchestration contract and evidence substrate. It does not yet launch real background model workers from ACP. That future runtime execution should reuse these contracts, keep ACP updates visible, and feed outcomes back into GEPA rather than becoming a separate CLI coding agent.
