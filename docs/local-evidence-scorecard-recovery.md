# Local Evidence Scorecard: Recovery And Failure Repair

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

## Scope

This scorecard mines recovery transitions and repeated failure-to-repair patterns from local evidence. The machine-readable companion is `.bag/evidence/scorecards/recovery-failure.json`.

Direct observations come from `trace-gepa/data/dataset_recovery.jsonl`, counterfactual JSONL, `bench/jobs/**` result/exception/ACP summary files, and the visible real ACP manifest. Derived summaries come from `trace-gepa/data/dataset_recovery_summary.md`, `bench/.bag/optimizer/failure-clusters.json`, and existing real ACP scorecards.

## Headline Findings

The recovery corpus has 4,055 transition pairs: 3,520 strong, 506 weak, and 29 transient. The dominant recoverable failure is `bash_exit_nonzero` with 2,744 pairs, followed by `hallucinated_path` with 455, `bash_timeout_141` with 251, `retry_loop` with 142, and `cancelled_parallel_batch` with 105.

Most recoveries change the input, not the tool. Input changes appear in 4,026 pairs, category changes in 3,291, and tool changes in 1,378. The highest-volume transitions are `Bash -> Bash` at 2,000 pairs, `exec_command -> exec_command` at 600, `Bash -> Read` at 452, and `Bash -> Edit` at 252.

The visible real ACP run is a separate failure mode: 9 tasks, 8 failed, 1 cancelled, 0 changed files, 0 fsWrite, 0 terminal creates, and 0 repair attempts. That is not a terminal-command repair failure; it is a run-mode progress failure where coding tasks ended without write or verifier activity.

## Recovery Patterns

| Pattern | Direct count | Main recovery behavior | Optimizer lesson |
| --- | ---: | --- | --- |
| `bash_exit_nonzero` | 2,744 | Adjust bash args, inspect files, edit targeted files | Parse stderr and choose a changed hypothesis before rerun |
| `hallucinated_path` | 455 | Rewrite `exec_command`, verify or correct paths | Verify paths and required output artifacts before acting |
| `bash_timeout_141` | 251 | Shorten commands, cap output, switch to read/search | Bound command output and separate SIGPIPE from semantic failure |
| `retry_loop` | 142 | Change command shape or switch tools | Do not repeat a low-information retry without new evidence |
| `cancelled_parallel_batch` | 105 | Resume with narrower bash/read state checks | Re-establish current state after cancellation |
| `cmd_not_found_127` | 18 | Swap command head or use local executable path | Inspect scripts, venvs, and local bins after exit 127 |

The counterfactual corpus reinforces the same policy: 431 annotations include 165 tool swaps, 133 input fixes, and 120 verify-first alternatives, with mean confidence 0.76.

## Benchmark Signals

`bench/jobs` has 541 parsed `result.json` files, 32 `exception.txt` files, and 415 `bag-acp-summary.json` files. Rewarded results are 286 full reward and 161 zero reward, with mean reward 0.6398. Of the zero-reward results, 157 have no infrastructure exception attached, so they are likely verifier/task mismatches rather than setup failures.

Exception phases in `result.json` split into 13 agent execution, 8 agent setup, 6 environment setup, and 6 verifier exceptions. Setup and infrastructure failures should be routed separately from model repair training: observed patterns include Docker environment start failures, setup timeouts, package install/network failures, and missing reward files.

The derived optimizer failure clusters cover 143 failures in 26 clusters. The highest-value hard negatives are missing `/app/summary.csv` with 25 failures, missing `/app/report.jsonl` with 22, SSH exit 255 with 19, HTTP 000 webserver failure with 14, and generic assertion failure with 10.

## Recommendations

1. Add a path-grounding gate before writes and final verification. The agent should inspect task-required output paths and verify remembered paths before acting.
2. Teach nonzero-command repair as a decision tree: parse stderr, inspect affected files, run suggested fix modes, and avoid blind reruns.
3. Penalize ACP coding runs that `end_turn` with zero writes and zero verifier commands unless the task is explicitly read-only or cancelled.
4. Treat setup failures as infrastructure routing examples, not semantic task failures.
5. Turn verifier-mismatch clusters into hard negatives that require explicit output-file, service, or exact-answer planning before final response.

## Caveats

Recovery pairs are mined transition proxies, not guaranteed causal repairs. The sanitised recovery mirror has equal row count but one missing ID and one extra ID relative to raw. Bench result files, exception files, ACP summaries, and optimizer clusters have different granularities, so their counts should not be summed as independent failures.
