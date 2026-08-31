# Terminal-Bench 2.0 sample — tool-use architectural pivot

**Date:** 2026-05-01
**Model:** `claude-opus-4-7` (master)
**Dataset:** `terminal-bench-sample@2.0` (10 tasks)

## Headline

| Run | Architecture | Concurrency | Wall | Reward 1.0 | Mean | Notes |
|---|---|---|---|---|---|---|
| #0 | plan+patch (baseline) | 4 | ~50min | 0/10 | 0.000 | pre-greenfield-fix |
| #1 | plan+patch + greenfield core | 4 | ~50min | 1/10 | 0.100 | regex-log flipped |
| #2 | plan+patch + 4 stability + 7 review | 4 | ~50min | 1/10 | 0.100 | log-summary flipped (regex-log regressed) |
| #3 | plan+patch + ENOENT-tolerance | 4 | ~55min | 1/10 | 0.100 | log-summary flipped |
| **#4** | **autonomous tool-use loop (mini-swe-agent style)** | **4** | **58min** | **7/10** | **0.700** | **architectural pivot** |
| #5 (running) | tools mode + 2× agent timeout | 4 | – | – | – | trying to recover the 3 timeouts from #4 |
| #6 (running) | dag-tools mode + 2× agent timeout | 4 | – | – | – | tool-loop OVER lite-DAG |

## Why the pivot from 1/10 to 7/10

The plan-and-patch architecture (`bag run` → interview → PRD → DAG → ONE structured edit envelope → apply → verify → max-2 repair rounds) cannot solve TB tasks because:

1. **Rigid Zod-validated edit envelope.** When Opus proposed multi-file changes, the JSON envelope often failed schema validation; repair burned tokens without progress (`fix-code-vulnerability` pre-pivot: 38 calls / 200k tokens / 0 fsWrite).
2. **No iterative observation.** Single-pass edit — model never gets to `cat` a file, run a test, re-read updated state, and try again.
3. **Repair feedback was textual append, not structured tool feedback.** Even after fix-2 (verifier feedback structured into the LLM prompt), the model still emits ONE big edit, not iterative tool calls.

The new `/run-tools` mode (~510 LOC across `src/autonomous-coding-turn.ts`, `src/autonomous-tools.ts`, `src/llm.ts:chatTextWithTools`, and the ACP wrapper in `src/acp/tool-use-runner.ts`) is mini-swe-agent flavored:

- **Single tool: `bash(command, timeout_sec?)`**. No file/edit tools. Everything via shell.
- **Subshell-per-call** (no cwd/env persistence). Prompt teaches `cd ... && ...`.
- **Sentinel submit:** `echo BAG_TASK_COMPLETE` as the first non-blank line of stdout terminates the loop.
- **Output elision** (head 5k + tail 5k if > 10k chars).
- **Format-error recovery:** if model returns no tool_calls, append a user-role reminder.
- **Max turns:** 80, hard cap.
- **No ACP fs.read/write.** Files are created via here-docs in bash.

The model does its own investigation, edit, verify, retry as bash calls. Verifier inside the container (Harbor's, separate from BAG's internal verifier) determines reward.

## Run #4 per-task

| task | reward | bash calls | poznámka |
|---|---|---|---|
| build-cython-ext | 1.0 ✅ | many | (pre-pivot ENOENT crash) |
| chess-best-move | **1.0** ✅ | **29** | **Opus has no vision, but installed `stockfish` via apt + analyzed PNG with PIL/numpy → extracted FEN → engine search → wrote moves to /app/move.txt. Real autonomous reasoning.** |
| configure-git-webserver | 0.0 | – | AgentTimeoutError (multi-service setup) |
| fix-code-vulnerability | 1.0 ✅ | many | bottle.py CVE patch (pre-pivot: 0 fsWrite after 38 calls) |
| log-summary-date-ranges | 0.0 | – | AgentTimeoutError |
| polyglot-c-py | 0.0 | – | AgentTimeoutError (dual-language file is hard) |
| qemu-alpine-ssh | 1.0 ✅ | many | ssh into VM solved |
| qemu-startup | 1.0 ✅ | many | |
| regex-log | 1.0 ✅ | 7 | |
| sqlite-with-gcov | 1.0 ✅ | many | |

## What broke / what to fix

- **Token tracking shows 0/0/0 in trial metadata** for tools-mode runs. Cause: BagAgent.run() pulls `manifest.json`, but autonomous mode writes `autonomous-summary.json` + `autonomous-trace.json`. Cosmetic for now; the reward number is the source of truth.
- **3/10 timeouts** → run #5 doubles timeout to test recovery.
- **stopReason='end_turn' even after `echo BAG_TASK_COMPLETE`** in some traces. The sentinel detection works (we saw `submitted: true` in tool_result), but the loop's outer `stopReason` set may have a corner case. Doesn't affect reward.

## Honest contamination caveat

- Public TB 2.0 dataset on GitHub. Opus 4.7 cutoff is 2026-01; TB 2.0 was released earlier. Some task instructions may be in training data. The verifier is sealed inside the container, so we can't game it directly. But Opus may know the *shape* of the answer.
- We have no Opus-direct baseline (e.g., harbor `claude-code` agent on the same sample) to subtract from BAG's score. The 70% number is BAG-on-Opus, not BAG-vs-Opus.
- ForgeCode at 81.8% (full TB 2.0) implies that with proper tool-use, Opus-class models recall most of TB. Our 70% on the SAMPLE is consistent with that.

## What this unlocks

- **Self-evolving feedback loop:** trace JSON → eval-harness → GEPA optimizer (which Codex is actively building in `src/optimizer/gepa-*.ts`) → tune the system prompt → next iteration. A real agentic loop now exists to optimize.
- **DAG-tools mode (run #6):** combines the planning instinct (lite plan = 1-5 issues with concrete verifier commands) with the tool-use loop. Enables per-issue verifier gating, concrete success criteria, and (eventually) parallel issue execution.
- **3 modes coexist:** `/run` (Zed review workflow, edit-strategy contract), `/run-tools` (atomic autonomous), `/run-dag-tools` (multi-step autonomous with planning scaffolding).
