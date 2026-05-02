# Sub-Benchmarks Index

Source: `data/benchmark_tasks.jsonl` + `_synthetic.jsonl`
Total tasks: 124

## By category

| Category | Count | File | When to run |
| --- | ---: | --- | --- |
| tool_routing | 33 | `benchmarks/tool_routing.tasks.jsonl` | Diagnosing tool-selection regressions. |
| planning | 19 | `benchmarks/planning.tasks.jsonl` | Validating plan structure / decomposition. |
| debugging | 15 | `benchmarks/debugging.tasks.jsonl` | Stress-test reasoning over failure traces. |
| recovery | 15 | `benchmarks/recovery.tasks.jsonl` | Post-error recovery and retry logic. |
| path_grounding | 10 | `benchmarks/path_grounding.tasks.jsonl` | Filesystem-grounded path correctness. |
| command_synthesis | 12 | `benchmarks/command_synthesis.tasks.jsonl` | Shell / CLI argument synthesis. |
| edit_safety | 20 | `benchmarks/edit_safety.tasks.jsonl` | Pre-edit invariants (Read-before-Edit, etc). |

## By difficulty

| Difficulty | Count | File |
| --- | ---: | --- |
| easy | 26 | `benchmarks/by_difficulty/easy.tasks.jsonl` |
| medium | 42 | `benchmarks/by_difficulty/medium.tasks.jsonl` |
| hard | 56 | `benchmarks/by_difficulty/hard.tasks.jsonl` |

## minibench.jsonl

30 stratified tasks (seed=42). Use for cheap CI-style runs.

Composition (category x difficulty):

| category | easy | medium | hard | total |
| --- | ---: | ---: | ---: | ---: |
| tool_routing | 2 | 1 | 2 | 5 |
| planning | 2 | 1 | 1 | 4 |
| debugging | 2 | 1 | 1 | 4 |
| recovery | 0 | 2 | 2 | 4 |
| path_grounding | 0 | 0 | 4 | 4 |
| command_synthesis | 2 | 1 | 1 | 4 |
| edit_safety | 0 | 3 | 2 | 5 |

## stress.jsonl

20 hardest tasks (hard-first, then adversarial categories). Use for regression hunts.

- `task_recovery_014` (recovery/hard) tweaked bash arguments after failure
- `task_recovery_013` (recovery/hard) changed bash command head from 'cd' to 'mv'
- `task_recovery_010` (recovery/hard) switched from Read to Bash after failure
- `task_recovery_008` (recovery/hard) switched from exec_command to spawn_agent after hallucinated_path
- `task_recovery_006` (recovery/hard) changed bash command head from 'cd' to 'find'
- `task_recovery_005` (recovery/hard) switched from write_stdin to exec_command after retry_loop
- `task_recovery_004` (recovery/hard) tweaked bash arguments after failure
- `task_recovery_003` (recovery/hard) tweaked bash arguments after failure
- `task_recovery_001` (recovery/hard) tweaked bash arguments after failure
- `task_recovery_000` (recovery/hard) narrowed bash output via head
- `task_path_grounding_009` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_008` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_007` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_006` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_005` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_004` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_003` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_002` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_001` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
- `task_path_grounding_000` (path_grounding/hard) Avoid hallucinated path; verify existence before reading.
