# Local Evidence Scorecard: Edit Strategy

Generated for graph `local-evidence-flywheel-v1` on `2026-05-04T10:48:50Z`.

## Summary

Local evidence does not support one globally best edit method. It supports several scoped priors:

| Context | Model | Strongest observable signal | Outcome |
| --- | --- | --- | --- |
| `bench/jobs` terminal-bench jobs | `claude-opus-4-7` | shell heredoc, Python/scripted writes, sed/perl/script replace | 319 trials with observable writes; pattern pass rates around 69-74%, but methods co-occur and are noisy |
| Aider polyglot smoke | configured BAG runtime, exact model not in scoreboard | shell heredoc whole-file writes | 4/5 problems passed; Java failed under a Gradle class-version verification gap |
| real ACP visible replay | `configured-runtime` | no-write behavior | 0/9 passed or changed files; 0 `fsWrite`; 8 failed, 1 cancelled |
| Claude Code action dataset | Claude Code style, mixed codebases | `Edit` and `Write` tools | high action-label support, but not verifier-scored |
| Codex GPT-5.5 action dataset | Codex GPT-5.5, mixed codebases | shell command editing only | sparse/noisy write-pattern evidence; no dedicated edit tool outcome signal |

The machine-readable scorecard is `.bag/evidence/scorecards/edit-strategy.json`.

## Strategy Findings

| Strategy | Local evidence | Result | Main caveat |
| --- | ---: | --- | --- |
| Shell heredoc whole-file write | 195 terminal-bench trials, 722 matched commands | 134 passes, mean reward `0.6872`; Aider smoke 4/5 | Co-occurs with other methods; weak before/after hash and rollback attribution |
| Python/scripted fs write | 121 terminal-bench trials, 1000 matched commands | 89 passes, mean reward `0.7355` | Often generators or helper scripts, not clean edit attempts |
| String replace/in-place | 138 terminal-bench trials, 297 matched commands | 97 passes, mean reward `0.7029` | High command-level wobble; no replacement count or stale-match diagnostics |
| Structured ACP `fs_write` | 16 early terminal-bench trials, 26 writes | 3 passes, mean reward `0.1875` | Coarse whole-file transport; likely early agent quality confound |
| Claude Code `Edit` tool | 3003 action rows | 2819 good, 171 bad, 13 user-confirmed | Action labels only; no verifier reward |
| Claude Code `Write` tool | 905 action rows | 831 good, 74 bad | Action labels only; no verifier reward |
| `apply_patch` / future structured tools | docs/tests/audit | synthetic support only | No reliable local real-attempt outcome count |

## Outcome Signals

- `no-write`: strongest real ACP signal. The visible run has 9 records, 0 changed files, and 0 `fsWrite`.
- `wobble`: terminal command failures appear inside write-pattern trials: shell heredoc 125, string replace/in-place 105, Python/scripted write 72.
- `applied-but-broken`: covered by synthetic replay/audit; the real visible ACP task label exists, but no write occurred in that run.
- `stale context`: real ACP has a stale-context task label, but again no write occurred; deterministic apply tests cover stale rejection.
- `repair` and `rollback`: documented and synthetically tested; visible ACP recorded 0 attempted repairs and 0 attempted rollbacks.
- `verification gaps`: Java Aider smoke failed because Gradle reported unsupported class file major version 70; benchmark summaries do not attribute failures to syntax, behavior, timeout, or infrastructure.

## Recommendations

1. Promote no edit method until real attempts carry complete strategy id, rendered contract version, read snapshot refs, target hashes, apply/write/verify phases, repair refs, rollback refs, and artifacts.
2. Fix the no-write ACP bottleneck before ranking edit methods for real ACP coding tasks.
3. Keep shell heredoc, scripted writes, and string replace as terminal-bench baselines, but add diff capture and verifier causality.
4. Use Claude Code `Edit`/`Write` rows as priors only; require per-codebase replay/eval before policy promotion.
5. Mark `apply_patch`, hash/range, multi-exact, fenced diff, AST structured, and other future tools as gaps until real outcome counts exist.

## Verification

Expected checks:

```sh
jq empty .bag/evidence/scorecards/edit-strategy.json
test -s docs/local-evidence-scorecard-edit-strategy.md
```
