# Live ACP Current Release Proof

- release proof: `release-proof.live-acp-evidence-readiness-v1`
- graph: `live-acp-evidence-readiness-v1`
- selection hash: `6fbc4883fa`
- generated at: `2026-05-05T08:59:11.567Z`
- proof mode: `current_graph`
- validation passed: `true`
- candidate generation: `allowed_as_scoped_dry_run`
- auto promotion: `blocked`
- promotion ready: `false`

## Validation

- planGraphSnapshot: `passed`
- evidenceIndexCommand: `passed`
- scorecardsCommand: `passed`
- optimizerGatesCommand: `passed`
- scorecardsGraphMatchesCurrent: `passed`
- optimizerGraphMatchesCurrent: `passed`
- historicalProofPreserved: `passed`
- historicalProofNotReportedAsCurrent: `passed`

## Blocking Reasons

- hidden holdout final gate is not ready for a frozen candidate
- historical release proof targets local-evidence-flywheel-v1 and is not current for live-acp-evidence-readiness-v1
- operator approval and rollback checkpoint are required
- post-promotion-monitor-window is unsatisfied

## Historical Proof

- id: `release-proof.local-evidence-flywheel-v1`
- graph: `local-evidence-flywheel-v1`
- selection hash: `06eeb209cb`
- stale for current graph: `true`

## Next Frontier

- run promotion readiness closure against current release proof
