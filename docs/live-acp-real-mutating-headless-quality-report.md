# Live ACP Real Mutating Headless Quality Report

Date: 2026-05-05
Graph: `blocker-closure-v1`
Selection hash: `a49f7e68fb`
Lane: `03-real-mutating-headless-quality`
Canonical epoch: `evidence-epoch.blocker-closure-v1.a49f7e68fb`

## Scope

This lane targeted one visible mutating fixture through the same ACP `/run` coding path used by sessions, while keeping all mutation inside `.bag/replay-corpus/real-acp-runs/<run-id>/`.

Selected fixture: `real-acp.task.simple-edit-greeting`.

Expected proof chain: edit generation, ACP filesystem write, terminal verifier, and scorecard classification.

## Artifacts

Run root: `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/`

| Artifact | Path |
| --- | --- |
| Metadata | `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/metadata.json` |
| Manifest | `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/real-acp-run.headless-quality-20260505.manifest.json` |
| Transcript | `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/transcripts/real-acp.task.simple-edit-greeting.json` |
| Replay export | `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/real-acp-run.headless-quality-20260505.replay-export.json` |
| Scorecard JSON | `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/real-acp-run.headless-quality-20260505.stability-scorecard.json` |
| Scorecard markdown | `.bag/replay-corpus/real-acp-runs/real-acp-run.headless-quality-20260505/real-acp-run.headless-quality-20260505.stability-scorecard.md` |
| Root index | `.bag/replay-corpus/index.jsonl` |

## Result

The live run completed artifact generation but did not prove mutating quality.

| Metric | Value |
| --- | ---: |
| tasks | 1 |
| passed | 0 |
| failed | 1 |
| changed files | 0 |
| read tool calls | 2 |
| write tool calls | 0 |
| terminal commands | 1 |
| verifier status | failed |

Failure reason:

```text
headless ACP generated no edit operations before verifier failure; stopReason=end_turn
```

Progress diagnostic:

```text
codingProgressClass=empty_edits
```

The transcript shows the runner reached the ACP session and read fixture files, then invoked the coding runner's no-edit failure verifier:

```text
BleedingAgent failed this coding turn because no edit operations were generated for a mutating run.
```

## Interpretation

This is not an edit-strategy quality pass and should not be used as positive promotion evidence. It is precise negative evidence for the live headless coding path: configured model/profile reached the ACP loop, but the coding runner produced zero edit operations for a simple mutating fixture.

The lane infrastructure is now useful because it separates:

- model/profile prerequisite blockers, recorded as `codingProgressClass=no_model`
- no-edit generation failures, recorded as `codingProgressClass=empty_edits`
- write and verifier evidence when injected or future live runs reach those phases
- stability scorecard aggregation for each produced manifest

## Downstream Decision

Lanes 04/05 can consume this as a current negative quality artifact and a working scorecard path. They should not treat real mutating headless quality as unblocked for promotion, because the required write phase did not occur.

The next blocker is the coding generation path that returned no edit operations after reading the simple fixture. Promotion remains blocked until this fixture reaches at least one ACP write and a real verifier command.
