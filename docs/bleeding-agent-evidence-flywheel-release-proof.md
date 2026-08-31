# BleedingAgent Evidence Flywheel Release Proof

Date: 2026-05-04

## Summary

This lane closes the evidence flywheel infrastructure, not a daily-driver coding-agent quality claim.

What is now real:

- Real headless ACP corpus execution into isolated fixture workspaces.
- Split-safe redacted replay exports and root corpus index.
- Stability scorecards for applied-but-broken, wobble, protected paths, repair, rollback, and fallback.
- Trace-mined scorecards for tools, terminal arguments, transitions, and edit families.
- Optimizer artifact lineage manifests, uplift classes, blocking gates, and reports.
- GEPA scheduler dry-run decisions with rollout stages and no silent promotion.
- Package/build/test/ACP consumer launch-target gates.

What is not proven:

- Glass/Zed GUI coding mutation parity.
- Daily-driver parity with Codex CLI, Claude Code, ForgeCode, OpenCode, or Pi.
- Positive coding uplift from the current policy/model path.
- Autonomous promotion readiness.

## Command Gates

All final gates passed:

| Gate | Result |
| --- | --- |
| `npm run typecheck` | passed |
| `bun test tests` | passed: 619 tests, 99 files |
| `npm run build` | passed |
| `npm run acp:verify-consumers` | passed |
| `npm pack --dry-run --json` | passed: 22 package entries |
| Real headless ACP visible corpus | completed |
| Stability scorecard report | completed |
| Trace-mined scorecard report | completed |
| GEPA dry-run scheduler report | completed |

## Real ACP Coding Result

Run: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/real-acp-run.headless-visible-20260504.manifest.json`

| Metric | Value |
| --- | ---: |
| Visible tasks | 9 |
| Hidden holdout exposed | 0 |
| Passed | 0 |
| Failed | 8 |
| Cancelled | 1 |
| Errors | 0 |
| File writes | 0 |
| Terminal commands | 0 |

Interpretation: the current headless ACP policy can read files but did not proceed into edits or terminal verification. This is a useful optimizer corpus because it captures a concrete failure mode, but it is not a good coding result.

## Stability Scorecard

Report: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/real-acp-run.headless-visible-20260504.stability-scorecard.md`

Observed rates:

| Metric | Rate |
| --- | ---: |
| Pass | 0.0% |
| Failed | 88.9% |
| Cancelled | 11.1% |
| Applied-but-broken | 0.0% |
| Wobble | 0.0% |
| Protected path touched | 0.0% |
| Repair attempted | 0.0% |
| Rollback failed | 0.0% |

Interpretation: zero applied-broken/wobble is not a stability win; no edit was applied, so there was no applied edit to become broken.

## Trace-Mined Scorecards

Report: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/real-acp-run.headless-visible-20260504.trace-scorecards.md`

Observed trace shape:

| Signal | Value |
| --- | ---: |
| `acp.fs/readTextFile` calls | 14 |
| Terminal argument patterns | 0 |
| Transition types | 1 |
| Edit matrix rows | 9 |
| Non-`none` edit families | 0 |

Interpretation: the trace-mined scorecards make the bottleneck explicit: the current policy is read-heavy and does not reach write/verify behavior in this real headless corpus.

## Optimizer Dry Run

Report: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/gepa-dry-run-scheduler-report.json`

| Metric | Value |
| --- | --- |
| Dataset records | 85 |
| Failure runs | 48 |
| Candidate generation ready under relaxed local thresholds | true |
| Auto-promotion ready | false |
| Scheduler decision | `needs_more_evidence` |
| Actual promotion allowed | false |
| Blocking gate | `post-promotion-monitor-window` |

Interpretation: GEPA can run as an evidence-producing dry-run scheduler. It must not silently promote.

## ACP Consumer Proof

`npm run acp:verify-consumers` passed with Glass and Zed installed and a successful ACP handshake. `/chat Ahoj, co umis?` produced 0 reads, 0 writes, 0 terminal operations, and 0 permission prompts.

This proves launch-target availability and no-side-effect chat behavior. It does not prove GUI consumer coding mutation parity.

## Next Claim

The next honest claim should be:

BleedingAgent now has a measured ACP-native optimization harness that can collect real headless ACP failures, redact and index them, score tool/edit/stability behavior, block weak optimizer artifacts, and run GEPA scheduling in dry-run mode without silent promotion.

The next frontier is not more scaffolding. It is improving the live coding policy so real ACP runs move from read-only context gathering into file edits and terminal verification, then rerunning the same corpus and scorecards to measure uplift.
