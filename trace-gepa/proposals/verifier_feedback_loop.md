# Verifier-Aware Self-Correction (Round-8 #GG)

## TLDR
- **Hypothesis:** Informed retry (model sees *why* its first answer scored 0) beats independent best-of-N retry on `schema_fail` failures, because most fails are near-misses (right family, wrong shape) that explicit feedback can fix in one turn.
- **Mechanism:** On verifier score 0, append a bounded feedback message naming the failure *category* and the *required action class*, then resample. Final score = max(turn1, turn2).
- **Cost:** ~30% extra LM calls (only the ~70% that fail get a retry); ~$5 for 175 tasks at Opus.
- **Win condition:** retry lift > best-of-2 independent-sample lift at equal token budget; otherwise feedback adds no signal beyond variance.

## Fair vs Cheating Boundary (CRITICAL)
- **FAIR feedback** (allowed in retry prompt):
  - Failure *category*: `schema_fail`, `regex_miss`, `wrong_tool_family`.
  - The *required tool name* (e.g. "must use `Bash`") — this is part of the task's surface contract, not gold.
  - High-level shape hint: "your output must contain a tool call" / "argument must be a shell command".
- **CHEATING feedback** (forbidden — would leak gold):
  - The full `must_include_keywords` regex list verbatim.
  - The expected argument values, paths, or exact strings.
  - Any token from the gold answer that the verifier matches against.
- **Rule of thumb:** feedback may reveal the *schema* the verifier checks, never the *content* it expects. If a human grader reading the feedback could reconstruct the gold answer, it's cheating.

## Design
1. Run task → verifier returns `{score, signal, expected_class}`.
2. If `score == 0`: build feedback per `--retry-feedback-detail`:
   - `brief`: "Previous answer scored 0 (reason: <category>). Try again."
   - `full`: "Previous answer scored 0. Failure: <category>. Expected action class: <tool_name>. Output must be a single tool call."
3. Append as user turn, resample with same params, score.
4. Emit per-task: `turn1_score`, `turn2_score`, `final = max`, `retry_used`, `delta`.

## CLI
- `--retry-on-fail [0|1]` (default 0).
- `--retry-feedback-detail {brief|full}` (default `brief`).
- `--retry-max-turns N` (default 1; reserved for future).

## Self-Critique
Strongest risk: the `full` detail level edges close to gold leakage on tasks where `tool_name` *is* effectively the answer — if so, retry lift is illusory and we're benchmarking instruction-following, not reasoning; mitigate by reporting `brief` and `full` separately so the gap reveals how much of the lift came from schema-hinting versus genuine self-correction.

**Path:** `trace-gepa/proposals/verifier_feedback_loop.md`
