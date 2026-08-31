# Tool Routing Scorecard

Generated for graph `local-evidence-flywheel-v1` at `2026-05-04T10:49:14Z`.

## Evidence Used

- `trace-gepa/data/sanitised/dataset_v2.jsonl`: 26,384 action/tool rows.
- `trace-gepa/data/sanitised/dataset_toolcalling.jsonl`: 4,045 focused tool-calling rows.
- `trace-gepa/data/sanitised/codex_gpt55_dataset.jsonl`: 6,820 Codex GPT-5.5 action rows.
- `trace-gepa/data/sanitised/counterfactuals.jsonl`: 431 action alternatives.
- `trace-gepa/data/sanitised/dataset_recovery.jsonl`: 4,055 failure-to-recovery pairs.
- `.bag/replay-corpus/index.jsonl` and visible ACP replay export: 9 current ACP failure cases.
- `bench/jobs/**/bag-acp-summary.json`: 415 ACP benchmark summaries.
- `bench/.bag/optimizer/*`: optimizer dataset, candidates, readiness, and failure clusters.

## Findings

The strongest routing signal is still the action/tool corpus. In `dataset_v2`, `observed_action.result_is_error` gives 22,963 successful calls and 3,421 failed calls, for an 87.03% tool-call success rate and 12.97% failure rate. The focused tool-calling slice is similar at 89.15% success. The Codex GPT-5.5 slice is cleaner at 94.77% success, but it is mostly `exec_command` and `write_stdin`, so it should not be treated as a balanced tool namespace distribution.

Terminal/process tools carry most of the observed risk. In `dataset_v2`, the identifier-derived `terminal_or_process` bucket has 15,787 calls with 3,088 errors, a 19.56% failure rate. `read_or_search` is much lower at 1.59%, `workspace_write` is 6.11%, and orchestration/plan tools are 0.35%. This taxonomy uses tool identifiers and ACP side-effect fields only; it does not inspect command text or language keywords.

Top per-tool failure rates in `dataset_v2`:

| Tool | Calls | Errors | Failure rate |
| --- | ---: | ---: | ---: |
| `write_stdin` | 427 | 109 | 25.53% |
| `Bash` | 12,730 | 2,729 | 21.44% |
| `exec_command` | 2,604 | 247 | 9.49% |
| `Write` | 975 | 74 | 7.59% |
| `Edit` | 3,070 | 173 | 5.64% |
| `Read` | 4,238 | 62 | 1.46% |
| `ToolSearch` | 831 | 0 | 0.00% |

Argument shape is usable but partial. Common parseable shapes include `Bash` with `command,description` (10,400 rows), `Read` with `file_path` (2,197) or `file_path,limit,offset` (1,924), `Edit` with `file_path,new_string,old_string,replace_all` (2,155), `exec_command` with `cmd,max_output_tokens,workdir,yield_time_ms` (2,316), and `ToolSearch` with `max_results,query` (829). Sanitised/truncated payloads mean these are lower-bound parseable counts.

Counterfactuals are directly useful for optimizer routing: `tool_swap` appears 165 times, `input_fix` 133 times, `verify_first` 120 times, `abort` 9 times, and `decompose` 4 times. The recovery corpus adds 4,055 transition pairs, mostly strong pairs (3,520), with top failed categories `bash_exit_nonzero` (2,744), `hallucinated_path` (455), `bash_timeout_141` (251), `retry_loop` (142), and `cancelled_parallel_batch` (105).

ACP replay shows an outcome-level routing failure, not a failing read tool. The visible run has 9 cases: 8 failed and 1 cancelled. All 14 recorded ACP tool calls are `acp.fs/readTextFile/succeeded/read`, yet all 9 cases have zero changed files, zero `fsWrite`, zero terminal creates, and zero terminal exits. Eight coding-mode cases stopped with `end_turn`, with `editStrategyFamily=none`.

Bench ACP summaries show high terminal usage but scarce writes. Across 415 summaries there are 12,983 terminal creates, 12,890 terminal exits, 40 `fsRead`, 26 `fsWrite`, and 0 permission events. Only 16 of 415 sessions wrote files, while 399 sessions had zero writes. There are 45 error-stop sessions: 40 prompt timeouts and 5 internal errors.

## Observability Limits

Model is only partially observable. The optimizer dataset records `claude-opus-4-7` for 85 rows, action corpora expose `src` such as `cc` and `codex`, and visible ACP replay exposes only a local model profile id.

Task category is partial. Benchmark task definitions include 33 `tool_routing` tasks (7 easy, 13 medium, 13 hard), and replay cases have task labels, but arbitrary action rows do not have normalized task categories.

Skip and block rates are not globally observable. Visible ACP replay has 1 cancelled/skipped case out of 9, bench summaries show 0 permission events, and optimizer auto-promotion is blocked by `post-promotion-monitor-window`. The full action corpus does not encode skipped opportunities.

## Optimizer Recommendations

Use tool name plus parseable argument-shape features as primary routing inputs, not command-language keyword rules.

Add a dedicated negative validation slice for no-write ACP coding failures: coding route, `editStrategyFamily=none`, `stopReason=end_turn`, zero `fsWrite`, zero terminal, and zero changed files.

Keep ACP protocol failures separate from tool-call failures. The visible ACP read calls succeeded; the failure was the transition from read-only behavior to no mutation/no verification.

Require either a write, a terminal verifier, or an explicit verifier-skipped justification for coding tasks expected to mutate files.

Do not auto-promote current routing candidates until the monitor-window gate is satisfied and no-write ACP failures are represented in validation.
