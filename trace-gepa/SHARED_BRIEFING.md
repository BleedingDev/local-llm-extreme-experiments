# Trace-driven GEPA — Phase 2 build briefing

All Phase-2 build agents share this briefing. Read it before starting.

## Goal
Turn the user's massive trace archives (Claude Code + Codex) into a GEPA-compatible dataset and run a closed-loop reflective optimisation against it. Final artifact: a static optimised prompt (and/or component map) that BAG (and/or Codex CLI) can load.

## Trace locations
- Claude Code sessions: `/Users/satan/.claude/projects/<encoded-path>/<sid>.jsonl` (273 sessions, 361 MB)
- Claude Code sub-agents: `/Users/satan/.claude/projects/<encoded-path>/<sid>/subagents/agent-*.jsonl`
- Codex sessions: `/Users/satan/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (14,912 sessions, 23 GB)
- Seed-session manifest (15 + 15 high-score sessions): `trace-gepa/data/seed_sessions.json`

## Schema cheat sheet
**Claude Code event types**: `user`, `assistant`, `attachment`, `system`, `permission-mode`, `last-prompt`, `ai-title`, `queue-operation`, `file-history-snapshot`. Tool calls live inside `assistant.message.content[]` as `{type:"tool_use", id, name, input}`. Tool results in subsequent `user.message.content[]` as `{type:"tool_result", tool_use_id, content, is_error}`. Sub-agents marked by `isSidechain:true`. Skill listings in `attachment.attachment.type=="skill_listing"`. Tool failures: `is_error:true` in tool_result.

**Codex event types**: `session_meta`, `response_item`, `event_msg`, `turn_context`. Tool calls: `response_item.payload.type=="function_call"` (function_name + arguments). Tool results: `response_item.payload.type=="function_call_output"` with `output` text containing `Process exited with code N`. User messages: `event_msg.payload.type=="user_message"`. Reasoning: `event_msg.payload.type=="agent_reasoning"`.

## Failure taxonomy (priors from recon)
- bash_exit_nonzero (high count, HIGH dataset readiness)
- bash_timeout_141 (parallel batch wreckage, HIGH)
- cmd_not_found_127 (typos / missing binaries, HIGH)
- cancelled_parallel_batch (one fail kills N, HIGH)
- edit_string_not_unique (MED)
- edit_file_not_read (MED)
- hallucinated_path (HIGH)
- hallucinated_skill (e.g. plan-graph, helm — HIGH, today's session)
- search_zero_results_misinterpreted (MED)
- user_correction (HIGH — user message starts with "no", "stop", "wrong", "actually", "ne ")
- subagent_terse_prompt (MED)
- skill_trigger_missed (MED)

## Stack
- **Python venv**: `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.venv-gepa/` — has dspy 3.2.0, gepa 0.0.27, anthropic, litellm, datasets, pyarrow, orjson, tqdm, rich, python-dotenv.
- **Activate**: `source /Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/.venv-gepa/bin/activate` (or use `.venv-gepa/bin/python` directly).
- **Env var quirk**: `.env` has `ANTHROPIC_AUTH_TOKEN`, NOT `ANTHROPIC_API_KEY`. Pass `api_key=os.environ["ANTHROPIC_AUTH_TOKEN"]` explicitly to dspy.LM.
- **Models**: task LM = `anthropic/claude-haiku-4-5` (cheap), reflection LM = `anthropic/claude-opus-4-7` (proposes new prompts).

## Workspace layout (canonical)
```
trace-gepa/
├── SHARED_BRIEFING.md          (this file)
├── data/
│   ├── seed_sessions.json      (✓ exists)
│   ├── cc_dataset.jsonl        (Phase 2a #1 writes)
│   ├── codex_dataset.jsonl     (Phase 2a #2 writes)
│   └── splits.json             (Phase 2a #3 writes)
├── extractors/
│   ├── extract_cc.py           (Phase 2a #1)
│   ├── extract_codex.py        (Phase 2a #2)
│   └── categorize.py           (Phase 2a #3)
├── agent_opt/                  (renamed from `gepa/` to avoid shadowing PyPI gepa)
│   ├── adapter.py              (Phase 2a #4)
│   ├── reflection.py           (Phase 2a #4)
│   └── optimize.py             (Phase 2b)
├── bench/
│   └── eval_baseline.py        (Phase 2b)
└── artifacts/
    └── optimized-prompts/      (output)
```

## Dataset record format (target — agree on this)
Each line of `cc_dataset.jsonl` / `codex_dataset.jsonl` is:
```json
{
  "id": "cc_660da9c6_evt00042",
  "src": "cc" | "codex",
  "src_path": "/abs/path/to/session.jsonl",
  "src_event_idx": 42,
  "context": {
    "user_request": "...",            // the user's most recent ask
    "recent_actions": [...],          // last K assistant actions
    "recent_tool_results": [...],     // last K tool results (truncated)
    "available_tools": [...],         // tool inventory at this point
    "available_skills": [...]          // skills inventory (CC only)
  },
  "observed_action": {
    "kind": "tool_use" | "text" | "skill",
    "name": "Bash" | "Edit" | "/dogfood" | ...,
    "input": "...",                   // tool input (redacted/truncated)
    "result_is_error": true|false,
    "result_excerpt": "..."           // first 500 chars of result
  },
  "label": "good" | "bad" | "user_corrected" | "user_confirmed",
  "failure_category": "bash_exit_nonzero" | "cmd_not_found_127" | "hallucinated_skill" | null,
  "ideal_action_hint": "...",         // optional: extracted from following user correction or successful retry
  "next_user_message": "..."          // if exists, what user said after (truncated)
}
```

Every record SHOULD set either `label!='good'` or `failure_category` so GEPA's reflective dataset has signal. Pure-success samples are valuable too but aim for ≥40% labelled-bad in the dataset.

## Constraints
- **No PII**: the user's repos may contain mentions; redact obvious tokens. Look for "sk-ant-", "hf_", "ghp_" — replace with "<REDACTED_KEY>".
- **Truncate**: per-event content trimmed to ~2 KB so dataset stays small.
- **Don't read whole 14 GB Codex sessions** — process line-by-line streaming. Stop when 200 examples extracted per session.
- **No comments unless non-obvious why** (per user CLAUDE.md).
- **No emojis** in code or docs.
- **Output**: each agent writes ONLY its own files. Do not modify others'.
