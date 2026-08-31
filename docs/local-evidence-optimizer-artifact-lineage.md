# Local Evidence Optimizer Artifact Lineage

This document summarizes the lineage contract in `.bag/evidence/optimizer/artifact-lineage-contract.json`. It applies to optimizer candidate prompts, rendered tool contracts, canonical tool specs, edit policy changes, and full model/codebase policy artifacts.

## Required Lineage

Every promotable candidate must carry:

- Candidate identity: `lineageManifestId`, `candidatePatchId`, artifact kind, artifact ID, physical content path, and `sha256` content hash.
- Profile identity: exact `modelProfileId`, `codebaseProfileId`, `baselinePolicyId`, and `candidatePolicyId`; `clientProfileId` and `codebaseRootFingerprint` when available.
- Evidence refs: evidence bundle IDs plus source evidence IDs from `.bag/evidence/index.jsonl`, source paths, roles, sanitisation, retention tier, split policy, and quality status.
- Scorecard refs: scorecard IDs from `scorecard-suite.local-evidence-flywheel-v1`, JSON and Markdown paths, split label, score/status, candidate context, and critical regression veto state.
- Validation refs: schema, train/dev, visible ACP regression, benchmark regression, and holdout results tied back to the same candidate and policy IDs.
- Holdout protection: holdout status, hidden holdout separation flag, split manifest refs, and leakage check status.
- Rollback refs: rollback checkpoint path, checkpoint hash, previous pointer hash, active pointer path, rollback mode, and post-promotion monitor window.
- Promotion decision: decision ID, decision value, decision time, decider, reason, and applies-to-new-sessions-only flag.

The source code context already has compatible gate names in `src/optimizer/artifact-lineage.ts`: candidate ID, evidence bundles, scorecards, profile match, baseline/candidate policy IDs, hidden holdout separation, rollback checkpoint, promotion decision, validation uplift, train/dev uplift, hidden holdout uplift, and weak prompt artifact veto.

## Evidence Anchors

Current local anchors:

- `evidence.optimizer.candidates`: `bench/.bag/optimizer/candidates.json`; 12 rendered tool contract prompt-fragment candidates, all validation-passed in that report.
- `gepa-feedback.20260501225146`: optimizer feedback bundle with 3 iterations and 32 feedback records.
- `evidence.optimizer.optimized-prompts`: timestamped prompt optimization runs under `trace-gepa/artifacts/optimized-prompts/**`.
- `evidence.acp.replay-index`, `evidence.acp.visible-run`, and `evidence.acp.adapter-replay-export`: ACP regression evidence and visible no-write failure signals.
- `tool-routing-scorecard`, `scorecard.edit-strategy`, `recovery-failure`, and `scorecard.benchmark-results`: scorecards listed in `.bag/evidence/scorecards/index.json`.

## Operator Gotchas

Do not trust `latest` or `latest_codex` symlinks. Resolve to the physical timestamped run directory, store the physical content path, and hash the content. A symlink can be useful for browsing but cannot be the canonical lineage ref.

Do not mix model or source labels. Run names, `track` values, seed source names, and model strings such as `codex`, `gpt55`, `claude-haiku-4-5`, and `claude-opus-4-7` are not substitutes for exact `modelProfileId` and `codebaseProfileId` matches.

Do not promote benchmark-only artifacts over ACP regressions. A positive benchmark delta must still pass visible ACP validation and adapter replay regression checks. ACP no-write and wrong-tool failures are blocking.

Keep benchmark families separate. Benchmark views overlap, so compare within the same evaluator family unless a scorecard explicitly defines cross-family comparability.

Carry caveats forward. Raw-local evidence remains local until privacy review passes, recovery sanitised evidence has a one-ID parity caveat, and the action split manifest stores a single IDs bucket that needs an explicit train/dev/holdout projection policy.

## Promotion Rule

A candidate can move past draft only when its lineage manifest proves exact profile and policy identity, at least one evidence bundle, at least one scorecard, visible validation, separated holdout evidence, and a rollback checkpoint. Missing any of those should result in reject, hold, or quarantine, not promotion.
