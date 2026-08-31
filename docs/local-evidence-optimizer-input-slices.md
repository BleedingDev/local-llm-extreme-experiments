# Local Evidence Optimizer Input Slices

Generated for graph `local-evidence-flywheel-v1` on `2026-05-04T10:55:41Z`.

Machine-readable contract: `.bag/evidence/optimizer/input-slices.json`.

## Contract

These slices define what the optimizer may read for candidate generation, development evaluation, hidden holdout, post-promotion monitoring, and live rollout. Hidden holdout is sealed evaluation evidence. It must never feed candidate generation, prompt drafting, policy synthesis, retrieval, failure clustering, or training.

Every consumer must preserve source evidence IDs, split membership, candidate ID, evaluator family, and privacy tier. Any parse error, unknown schema fingerprint, split leakage, or leaked hidden holdout content blocks promotion.

## Candidate Generation

Allowed input:

- `evidence.action.sanitised-dataset-v2` with `evidence.split.action-v2`, after a deterministic projected train/dev/hidden split is defined because `splits_v2.json` has only an `ids` bucket.
- `evidence.recovery.sanitised-dataset` with `evidence.split.recovery`, using `train` and `val` only.
- `evidence.counterfactual.sanitised` for `tool_swap`, `input_fix`, `verify_first`, `abort`, and `decompose` examples keyed by `record_id`.
- Non-hidden benchmark task definitions from `evidence.benchmark.sanitised-full-tasks` and deduped non-hidden views from `evidence.benchmark.views`.
- Non-hidden local optimizer rows and failure clusters from `evidence.optimizer.dataset` and `evidence.optimizer.failure-clusters`.

Forbidden input:

- `optimizer-input.hidden-holdout`.
- Hidden task prompts, expected outputs, verifier specs, task IDs, failure summaries, score deltas, or derived clusters.
- Raw-local evidence unless privacy review explicitly promotes a sanitised mirror.
- Derived RAG indexes as canonical optimizer data, because duplicate IDs are documented.

Candidate generation may use visible terminal-bench failure clusters to propose fixes, but only when the underlying rows are not hidden holdout members.

## Development Evaluation

Allowed input:

- Visible benchmark definitions and views: `evidence.benchmark.sanitised-full-tasks`, `evidence.benchmark.views`.
- Visible ACP replay and adapter replay: `evidence.acp.replay-index`, `evidence.acp.visible-run`, `evidence.acp.adapter-replay-export`.
- Terminal-bench and optimizer result evidence: `evidence.bench.jobs`, `evidence.optimizer.dataset`, `evidence.optimizer.failure-clusters`.
- Compact edit smoke evidence: `evidence.edit.aider-polyglot-results`.

Required validation coverage:

- ACP no-write failures: coding route, `editStrategyFamily=none`, `stopReason=end_turn`, zero `fsWrite`, zero terminal creates, and zero changed files.
- Terminal-bench repeated weak tasks: `qemu-alpine-ssh`, `qemu-startup`, and `chess-best-move`.
- Terminal-bench repeated output failures: missing `/app/summary.csv` and missing `/app/report.jsonl`.
- Terminal-bench infrastructure or environment signatures: `SSH exit 255` and `HTTP 000 webserver failure`.
- Coding tasks expected to mutate files must show at least one write, a terminal verifier, or an explicit verifier-skipped justification.

Development evaluation may be replayed during iteration. Its failures can feed the next candidate only if they are not hidden holdout members.

## Hidden Holdout

Eligible source evidence:

- `evidence.recovery.sanitised-dataset` with the `test` bucket from `evidence.split.recovery`.
- `evidence.action.sanitised-dataset-v2` only after a sealed projection is created from `evidence.split.action-v2`; the current single `ids` bucket is not enough by itself.
- Sealed benchmark task or result subsets from `evidence.benchmark.sanitised-full-tasks`, `evidence.benchmark.views`, and `evidence.bench.jobs`.

Allowed consumers are only the frozen-candidate hidden-holdout evaluator, promotion gate, and redacted aggregate audit reporter.

Forbidden consumers include candidate generation, prompt patch synthesis, tool-routing policy synthesis, recovery/edit policy synthesis, failure-cluster mining, retrieval index building, training export, and iterative dev evaluation.

Hidden holdout aggregate results may be reported after candidate freeze. Failed hidden cases must not be converted into prompts, examples, failure clusters, or candidate notes in the same optimization cycle. If hidden content leaks, invalidate the holdout and reseal a replacement split.

## Post-Promotion Monitoring

Allowed input:

- ACP replay evidence: `evidence.acp.replay-index`, `evidence.acp.visible-run`, `evidence.acp.adapter-replay-export`.
- Terminal-bench result evidence: `evidence.bench.jobs`.
- Optimizer state and lineage: `evidence.optimizer.dataset`, `evidence.optimizer.failure-clusters`, `evidence.optimizer.candidates`.

Monitoring must explicitly include ACP no-write failures and terminal-bench repeated failures, not treat them as training-only examples.

Required monitoring signals:

- ACP no-write/no-terminal/no-changed-files failures, especially `end_turn` with `editStrategyFamily=none`.
- ACP write scarcity and coding tasks with no verifier or explicit verifier-skipped justification.
- Terminal-bench weak repeated tasks: `qemu-alpine-ssh`, `qemu-startup`, `chess-best-move`.
- Terminal-bench top clusters: missing `/app/summary.csv`, missing `/app/report.jsonl`, `SSH exit 255`, `HTTP 000 webserver failure`, `assert false`.
- Terminal-bench stop and setup signals: prompt timeout, internal error, setup timeout, agent timeout, and reward-file missing.

Post-promotion monitoring failures may feed a future candidate cycle only after they are captured as new evidence with source lineage, privacy tier, and non-hidden split assignment.

## Live Rollout

Allowed input:

- Frozen candidate and prompt lineage: `evidence.optimizer.candidates`, `evidence.optimizer.optimized-prompts`.
- Policy gates: `evidence.policy.schema-audit`, `evidence.policy.retention`.
- Validation and monitoring summaries from `evidence.acp.visible-run` and `evidence.bench.jobs`.

Live rollout consumes summaries and gates, not training rows. It must fail closed if schema checks fail, split leakage is detected, hidden holdout was exposed to candidate generation, ACP no-write validation is missing, terminal-bench repeated failure validation is missing, or the post-promotion monitor window is unsatisfied.

Do not infer the best prompt from `latest` symlinks. Use timestamped run metadata, candidate IDs, scorecard lineage, and monitor results.

## Caveats

- Raw-local evidence remains local-only until privacy review passes.
- The sanitised recovery corpus has equal row counts but one missing raw ID and one extra sanitised ID.
- Benchmark views overlap and must be deduped by task ID before split assignment or evaluation summaries.
- Terminal-bench rewards, real ACP replay outcomes, Aider polyglot, LiveCodeBench, SWE-Bench, METR, and ablation results are separate evaluator families.
- Visible ACP replay is a validation and monitoring baseline, not hidden holdout.

## Verification

Required checks:

```sh
jq empty .bag/evidence/optimizer/input-slices.json
test -s docs/local-evidence-optimizer-input-slices.md
```
