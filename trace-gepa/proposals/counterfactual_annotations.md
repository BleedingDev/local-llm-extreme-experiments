# Counterfactual Annotations Corpus — Brainstorm Round-4 Member #K

## TLDR
- Raw `bad` traces tell us *what failed* (exit code 1, timeout, cancelled batch) but not *what would have worked*; preflight predicates capped at ~17% recall precisely because 80% of failures are semantic shell errors invisible to syntactic rules.
- Run a one-shot Opus annotation pass over all 426 `bad` + 5 `user_corrected` records (~431 total) producing `counterfactuals.jsonl`: `{record_id, observed_action, counterfactual_action, rationale, confidence, failure_taxonomy}`.
- This is *grounded annotation*, not synthesis: each counterfactual is conditioned on the real `context` (user_request, recent_actions, recent_tool_results, available_tools/skills) so it remains in-distribution for that session.
- Single artifact, four downstream consumers — SFT pairs, GEPA dense reward, preflight rule mining (round-2), DPO-style contrastive data — at ~$10 one-time cost.

## 1. Hypothesis
The `bad` label is a binary signal saturated by `bash_exit_nonzero` (388/431 = 90%). Training on "this command failed" teaches the agent only to avoid the literal command — useless because most failures are context-specific (wrong path, missing flag, formatter-not-installed). A counterfactual *positive example* in the same context yields ~10× the bits per record: the model learns the corrected action *and* the contrast direction. This converts a 1-bit failure label into a structured (observed, ideal, delta-rationale) triple.

## 2. Corpus Shape
```
{
  "record_id": "cc_660da9c6_evt00088",
  "observed_action": {"kind":"tool_use","name":"Bash","input":{...},"result_is_error":true},
  "counterfactual_action": {"kind":"tool_use","name":"Bash","input":{"command":"pnpm exec vp check --fix && pnpm exec vp check"}},
  "rationale": "Formatter reported issues; senior engineer would auto-fix then re-verify, not surface the error.",
  "confidence": 0.82,
  "failure_taxonomy": "missing_remediation_step",
  "delta_kind": "command_rewrite" | "tool_swap" | "abort_and_ask" | "decompose" | "no_op_was_correct"
}
```
Target: 431 rows, ~1 per `bad`/`user_corrected` record. Skip `cancelled_parallel_batch` (7) where the original action was fine — annotate as `no_op_was_correct`.

## 3. Annotation Pipeline
**Prompt template (excerpt, 200 chars):**
> "You are reviewing a Claude Code session. The agent took ACTION X in CONTEXT Y and it failed (RESULT Z). Given only what was knowable at decision time, output the JSON action a senior engineer would..."

Full template includes: serialized `context.user_request` (truncate 2k chars), last 3 `recent_actions`, last 2 `recent_tool_results`, `available_tools`, `failure_category`, `next_user_message` (held out — used only by validator), and a strict JSON schema.

- **Batching**: 20 records per Opus call via prompt-cached system preamble; ~22 calls total.
- **Dedup**: hash `(observed_action.input, failure_category)` — collapse identical failures, share one annotation across record_ids.
- **Validation pass**: second Opus call per record scores `(counterfactual ↔ next_user_message)` agreement on a 0-1 scale; flag <0.4 for human review (~30 records expected).
- **Leak guard**: `next_user_message` is withheld from the annotator, only the validator sees it — prevents label leakage into training pairs.

## 4. Downstream Use Cases
- **SFT supervision**: `(context → counterfactual_action)` becomes positive supervision; +10× signal vs filtering bad and training only on good.
- **GEPA dense reward**: distance(candidate_action, counterfactual_action) replaces binary 0/1 — gives gradient where current reward is flat.
- **Preflight rule mining (round-2)**: cluster counterfactuals by `delta_kind` to extract *transformation rules* (e.g. "append `--fix` when formatter command lacks it") — closes the 17% recall gap.
- **DPO pair-data**: `(observed_bad, counterfactual_good)` is exactly the chosen/rejected schema, free of synthetic-pair distribution shift.

## 5. Cost
431 records × ~3k input + 400 output tokens, batched ×20 with cached preamble: ~22 annotation calls + ~22 validator calls. Opus 4.7 @ $15/$75 per Mtok → **~$8** one-time, **~$12** with one re-run. Within the $5–15 envelope.

## Path
`trace-gepa/proposals/counterfactual_annotations.md` (this file); artifact lands at `trace-gepa/data/counterfactuals.jsonl`.

## Self-Critique
Opus's notion of "what a senior engineer would do" is itself a model bias — counterfactuals encode Opus's preferences, not ground truth, so any agent fine-tuned on this corpus regresses toward Opus rather than toward the actual user; mitigation is constraining counterfactuals to the 5 `user_corrected` records' style as a calibration anchor, but n=5 is thin.
