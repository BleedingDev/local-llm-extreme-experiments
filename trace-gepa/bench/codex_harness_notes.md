# Codex CLI bench harness — notes

## Binary + version

- `which codex` -> `/opt/homebrew/bin/codex`
- `which codex-native` -> `/Users/satan/.local/bin/codex-native`
- `codex --version` -> `codex-cli 0.128.0`

## Subcommand: `codex exec` IS available

`codex --help` lists `exec` as: "Run Codex non-interactively [aliases: e]".
We use it. Prompt is read from stdin via `-` argument.

## Final invocation (per task)

```
codex exec \
    --json \
    --skip-git-repo-check \
    --ephemeral \
    --ignore-rules \
    --dangerously-bypass-approvals-and-sandbox \
    -c approval_policy=never \
    -c model_reasoning_effort=<low|medium|high|xhigh> \
    -m <model> \
    -
<prompt-on-stdin>
```

### Why each flag

- `--json` — emits JSONL events to stdout (`thread.started`, `turn.started`,
  `item.completed { item: { type: "agent_message", text: "..." }}`,
  `turn.completed`). The harness scans for the last `agent_message` text.
- `--skip-git-repo-check` — we run inside `trace-gepa/`, but we don't want
  codex to bail or pollute git state.
- `--ephemeral` — does not persist a session under `~/.codex/sessions`.
- `--ignore-rules` — skip user/project `.rules` files; harness must be
  reproducible across machines.
- `--dangerously-bypass-approvals-and-sandbox` + `-c approval_policy=never` —
  needed to run unattended without TUI prompts. The agent in this benchmark
  is asked to *return JSON describing the next action*, not to actually
  execute it, so sandbox bypass is benign here.
- `-c model_reasoning_effort=...` — the CLI does NOT have a top-level
  `--reasoning` flag in 0.128.0. Reasoning effort is plumbed via the generic
  `-c key=value` config override. Verified accepted values include `low`,
  `medium`, `high`, `xhigh`.

## Reasoning flag — gotcha

The user spec mentioned `--reasoning xhigh`. That flag does NOT exist on
`codex exec` in 0.128.0. The working path is `-c model_reasoning_effort=xhigh`.
The harness exposes `--reasoning <eff>` as a *Python-side* arg and translates
it internally to the `-c` override.

## Auth

Auth is handled entirely by codex itself (`~/.codex/auth.json`). The harness
does not touch that file. Auth detection is best-effort: the probe task runs
first, and if it returns `codex_unavailable` (binary missing) or hits common
auth-failure phrases in stderr, every remaining task is marked
`error="codex_auth_failure"` (or `codex_unavailable`) with score 0.

## Output capture

Codex JSONL stream:

```
{"type":"thread.started",...}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"<final>"}}
{"type":"turn.completed","usage":{...}}
```

We take the last `item.completed` whose `item.type == "agent_message"` as
the final response. Then we try, in order:

1. `json.loads(text)` if it looks like a bare object.
2. ```json ...``` fenced block.
3. Greedy `{...}` regex over the response.
4. Raw text fallback.

## Scoring

We pass the parsed prediction directly to `bench.verifiers.verify(task, predicted)`.
**Important:** the current `benchmark_tasks.jsonl` (n=105) ships verifier
specs with `pattern_or_command` instead of the keys
`tier1_regex.verify_regex` / `verify_structural_json` actually read
(`pattern`, `schema`). Effects:

- `regex` tasks all return `signal=regex_no_pattern, score=0.0`.
- `structural_json` tasks return `score=1.0` whenever the prediction parses
  as JSON (no schema is supplied).

This is a verifier-spec/dataset issue, NOT a harness issue. The harness
faithfully calls `verify(...)` and records whatever it returns.

## Result schema

Output JSON has shape `{ "summary": ..., "config": ..., "results": [...] }`.
Per-task fields: `id`, `category`, `difficulty`, `score`, `verifier`,
`elapsed_s`, `exit_code`, `timed_out`, `stdout_chars`, `stderr_tail`,
`final_message_chars`, `parser_status`, `message_status`, `predicted_tool`,
`predicted_preview`.

## Concurrency + timeouts

- `ThreadPoolExecutor(max_workers=4)` by default. Codex CLI spins up its own
  process per call so the harness cost is mostly subprocess + network.
- Per-task subprocess `timeout=90s`. On expiry the task is scored 0 and
  flagged `timed_out: true`.

## Smoke results

- 5 tasks (limit, first 5 = all `tool_routing`), `--reasoning high`:
  `pass_rate=1.000`, parser `json_direct=5`, exit codes all 0,
  mean elapsed ~5s/task with 4 workers.

## 20-task results

- First 20 tasks (all `tool_routing`), `--reasoning xhigh`: `pass_rate=1.000`.
- Stratified 20 spanning all 7 categories, `--reasoning xhigh`:
  `pass_rate=0.850`. Per-category:

  | category          | n | pass  |
  |-------------------|---|-------|
  | tool_routing      | 3 | 1.000 |
  | command_synthesis | 3 | 0.000 |  (regex_no_pattern from verifier)
  | edit_safety       | 3 | 1.000 |
  | path_grounding    | 3 | 1.000 |
  | debugging         | 3 | 1.000 |
  | recovery          | 3 | 1.000 |
  | planning          | 2 | 1.000 |

## Models known to work via `-m`

Confirmed during smoke: `gpt-5.5`. Other config-listed candidates
(`gpt-5.5-mini`, `gpt-5.4`, `gpt-5.3-codex-spark`) are not validated by this
run — pass `--model <name>` to attempt them; failures will surface as
non-zero exit codes plus stderr tail in the output JSON.
