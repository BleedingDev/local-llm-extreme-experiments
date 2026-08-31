# Live ACP Real Agent Quality Report

Date: 2026-05-05
Graph: `live-acp-evidence-readiness-v1`
Selection hash: `6fbc4883fa`
Lane: `Live ACP Evidence 05 Real Agent Quality Evaluation`

## Executive Summary

This lane found useful quality evidence, but it does not prove real Glass/Zed consumer quality yet.

The strongest current signal is the headless/offline ACP corpus under `.bag/replay-corpus/**`. It contains a representative train/dev coding task set and trace-mined scorecards, but the observed visible run is a negative baseline: 9 visible tasks, 0 passed, 8 failed, 1 cancelled, 0 changed files, 0 write tool calls, and 0 terminal commands. The failure is therefore not "bad edit method X"; it is a route/progress failure where coding tasks reached end-turn or cancellation without mutation or verification.

Real consumer benchmarking remains blocked because `scripts/run_real_acp_corpus.ts --mode real_consumer --plan-only` reports no wired Glass/Zed real-consumer executor and refuses non-dry real-consumer mutation in this substrate.

## Quality Evaluation Matrix

The next optimizer-ready quality matrix should keep these axes first-class instead of collapsing them into one global score.

| Axis | Values already represented | Current evidence status | Optimizer use |
| --- | --- | --- | --- |
| Task shape | simple edit, greenfield, bugfix, refactor, stale context, protected path, cancellation, rollback, applied-but-broken, verifier skip, MCP failure, user correction | Pack covers 12 labels; visible evidence covers 9 train/dev labels | Per-shape routing and failure slicing |
| Project type | TypeScript fixture, greenfield fixture | Covered in task pack and headless corpus | Codebase profile specialization |
| Edit strategy family | none, shell heredoc, scripted write, string replace, structured ACP write, Claude-style Edit/Write as observed baselines | ACP visible run is `none`; other methods come from benchmark/source-adapter evidence | Do not promote a single global edit method |
| Model profile | `model.real-acp.local-headless`, benchmark/imported observed model ids | Partial | Separate model behavior from harness behavior |
| ACP client profile | `client.real-acp.headless-capable`; real Glass/Zed missing | Headless only | Do not claim consumer parity yet |
| MCP/tool use | read, write, terminal, MCP failure fixtures | Tests and replay scenarios cover this; visible run only used reads | Tool description and routing optimization |
| Verification policy | required, allowed-to-skip, must-skip, expected-to-fail-before-repair | Covered in task pack and tests | Verifier tactics and skip justification gates |
| Recovery mode | repair, rollback, cancellation, fallback, user correction | Synthetic/replay evidence exists; visible run shows no repairs or rollbacks | Recovery prompt and fallback policy optimization |
| Failure class | no-write, verifier failed, terminal failure, hallucinated path, timeout, permission/cancellation, applied-but-broken | Strong local scorecards exist | Build hard negative slices |

## Representative Coding Tasks

The representative task pack is `src/replay/real-acp-task-pack.ts`.

It defines 12 tasks:

| Task | Split role | Shape |
| --- | --- | --- |
| `real-acp.task.simple-edit-greeting` | visible train | simple edit |
| `real-acp.task.greenfield-slugify` | visible dev | greenfield workspace |
| `real-acp.task.cart-bugfix-fail-to-pass` | visible train | bugfix fail-to-pass |
| `real-acp.task.refactor-price-format` | hidden holdout | refactor |
| `real-acp.task.stale-context-anchor` | visible train | stale context |
| `real-acp.task.protected-path-doc` | visible dev | protected path |
| `real-acp.task.cancellation-mid-edit` | visible train | cancellation |
| `real-acp.task.rollback-invalid-parser` | hidden holdout | rollback |
| `real-acp.task.applied-but-broken-import` | visible train | applied-but-broken |
| `real-acp.task.verifier-skip-docs` | visible dev | verifier skip |
| `real-acp.task.mcp-tool-failure-fallback` | hidden holdout | MCP tool failure |
| `real-acp.task.user-correction-scope` | visible train | user correction |

The deterministic split policy keeps train/dev visible for optimizer input and reserves holdout for final promotion or real-consumer regression checks.

## Current Headless Quality Evidence

Primary run: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/real-acp-run.headless-visible-20260504.manifest.json`

| Metric | Value |
| --- | ---: |
| visible tasks | 9 |
| passed | 0 |
| failed | 8 |
| cancelled | 1 |
| changed files | 0 |
| write tool calls | 0 |
| terminal commands | 0 |
| read tool calls | 14 |

Status by visible task:

| Task | Split | Status | Changed files | Write tools | Terminal commands |
| --- | --- | --- | ---: | ---: | ---: |
| `real-acp.task.applied-but-broken-import` | train | failed | 0 | 0 | 0 |
| `real-acp.task.cancellation-mid-edit` | train | cancelled | 0 | 0 | 0 |
| `real-acp.task.cart-bugfix-fail-to-pass` | train | failed | 0 | 0 | 0 |
| `real-acp.task.greenfield-slugify` | dev | failed | 0 | 0 | 0 |
| `real-acp.task.protected-path-doc` | dev | failed | 0 | 0 | 0 |
| `real-acp.task.simple-edit-greeting` | train | failed | 0 | 0 | 0 |
| `real-acp.task.stale-context-anchor` | train | failed | 0 | 0 | 0 |
| `real-acp.task.user-correction-scope` | train | failed | 0 | 0 | 0 |
| `real-acp.task.verifier-skip-docs` | dev | failed | 0 | 0 | 0 |

Interpretation:

- This is a harness/routing/progress failure before meaningful edit strategy comparison.
- It should become a negative validation slice: coding task expected mutation, route reached no-write/no-terminal/no-changed-files.
- It must not be used to rank edit methods because no edit method was actually exercised.

## Tool And Trace Signals

The trace-mined ACP scorecard reports:

| Tool | Calls | Success | Failed | Failure association |
| --- | ---: | ---: | ---: | ---: |
| `acp.fs/readTextFile` | 14 | 100.0% | 0.0% | 85.7% |

This means the read tool itself is not failing. The failure is the missing transition from read activity into write, terminal verification, explicit skip, repair, or rollback.

Existing replay tests also validate scenario coverage for:

- tool-call failures
- edit failure classes
- routing scenarios
- no-write validation
- no-write slice building
- source-adapter observed baseline export

## Baseline Comparison

Comparable baselines exist only as local/imported observed evidence. They are not gold.

| Baseline family | Evidence | Observed result | Use |
| --- | --- | --- | --- |
| Real ACP visible headless | `.bag/replay-corpus/**` | 0/9 pass, 8 failed, 1 cancelled, no writes | Negative no-write/progress baseline |
| Terminal-bench BAG | `docs/local-evidence-scorecard-benchmark-results.md` | 52 jobs / 424 trials, weighted mean 0.6304 | Comparable reward loop after ACP progress is fixed |
| Best local BAG terminal-bench run | `bench/jobs/2026-05-02__07-13-33/result.json` | 10/10, mean 1.0 | High-water mark, not expected aggregate |
| Claude Code comparator | local benchmark scorecard | 1 job / 10 trials, mean 0.9 | Observed comparator, not gold |
| Aider polyglot smoke | local benchmark scorecard | 4/5 pass | Edit-method prior only |
| Source-adapter corpus | `.bag/replay-corpus/source-adapters/adapter-replay-export/**` | 50 cases, 30 train / 10 dev / 10 holdout, all tagged `observed-baseline` | Redacted Codex/Claude/BAG behavior evidence, weak oracle |
| Local evidence scorecards | `.bag/evidence/scorecards/**` | 4 scorecards validate cleanly | Stale graph id, useful for patterns only |

The current fair conclusion is: BleedingAgent's live ACP coding quality is not yet competitive as measured by real-consumer tasks, because the real-consumer path is blocked and the visible headless path did not perform edits. The harness is collecting the right kind of failure evidence, but promotion-quality claims need a fresh real-consumer run and current graph release proof.

## Blockers

Real consumer benchmarking is blocked by the runner itself:

```text
src/replay has no real ACP consumer executor factory wired for Glass or Zed
runRealAcpCorpus currently refuses non-dry real_consumer mutation in this substrate
```

Evidence promotion also remains blocked by `npm run bag -- evidence validate`:

```text
edit-policy promotion needs first-class edit attempt telemetry
hidden holdout final gate is not ready for a frozen candidate
operator approval and rollback checkpoint are required
post-promotion-monitor-window is unsatisfied
visible ACP no-write/no-terminal validation must be represented
```

## Next Optimizer Candidates

1. Add a real-consumer ACP executor adapter for Glass/Zed with isolated fixture workspaces and transcript capture.
2. Turn no-write coding failures into a first-class validation slice: expected mutation plus zero changed files, zero write tools, zero terminal, and no justified verifier skip.
3. Require coding-task progress before final response: write, terminal verifier, explicit safe skip, repair attempt, rollback, or structured impossibility reason.
4. Emit first-class edit-attempt telemetry for attempted, applied, broken-after-apply, repaired, and rolled-back states.
5. Keep edit strategy optimization empirical per model and per codebase. Do not select hash/range/diff/whole-file/apply-patch globally from research.
6. Split harness failures from model failures in scoring. The current 0/9 ACP run is primarily a harness/route/progress failure.
7. Treat observed Codex/Claude/BAG source-adapter traces as weak baselines and hard-negative examples, not as golden behavior.

## Commands Run

Successful verification:

```bash
bun test tests/replay-real-acp-task-pack.test.ts tests/replay-real-acp-headless-executor.test.ts tests/replay-real-acp-runner.test.ts tests/replay-real-acp-index.test.ts tests/replay-real-acp-scorecard.test.ts tests/replay-real-acp-trace-scorecards.test.ts tests/source-adapters-replay.test.ts
bun test tests/replay-tool-call-scenarios.test.ts tests/replay-edit-failure-scenarios.test.ts tests/replay-routing-scenarios.test.ts src/replay/no-write-validation.test.ts src/replay/no-write-slice.test.ts
npm run bag -- evidence scorecards
npm run bag -- evidence validate
npx tsx scripts/run_real_acp_corpus.ts --metadata /tmp/bag-real-acp-metadata.* --plan-only --run-id real-acp-run.plan.headless-quality-20260505 --mode headless_acp --purpose development_eval
npx tsx scripts/run_real_acp_corpus.ts --metadata /tmp/bag-real-acp-metadata.* --plan-only --run-id real-acp-run.plan.real-consumer-quality-20260505 --mode real_consumer --purpose development_eval
```

Expected non-zero command:

- The `real_consumer` plan-only command exits 1 because the real consumer substrate is intentionally blocked.

Operator mistakes during probing:

- Two initial plan-only invocations without `--metadata` failed with `--metadata is required`.
- A process-substitution attempt produced `Unexpected end of JSON input`.
- One shell wrapper used zsh's read-only `status` variable during cleanup. The corrected wrapper used `exit_code`.

These failures did not edit source, tests, or evidence artifacts.

## Plan Status Decision

Completed:

- `define-quality-eval-matrix`
- `select-representative-coding-tasks`
- `compare-against-baselines`
- `publish-quality-report-and-next-optimizations`

Pending:

- `run-agent-dogfood-benchmarks`

Reason: available headless/offline tests and scorecards were run, but fresh real Glass/Zed consumer benchmarking is blocked by missing real-consumer executor wiring.
