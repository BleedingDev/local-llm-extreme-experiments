# Session-Replay Oracle

## TLDR

- **Skip the verifier_spec entirely.** The user's actual next message after each agent action already encodes ground truth ("perfect, continue" = pass; "no, use X instead" = fail-with-hint). Round-9 audits found 16 + 29 verifier-DSL bugs precisely because we were synthesising a signal that already exists in the trace.
- **Bench task = (user_msg, observed_action, user_followup, label)** with NO verifier_spec. Model emits `(tool_name, brief_reason)`; an LM-judge (`claude-opus-4-7`) scores 0 / 0.5 / 1 conditioned on the real follow-up — never seeing the gold tool_name.
- **Self-calibrating QC.** Pre-run each `observed_action` through the same judge; expect score 1 on `good`/`user_confirmed` and 0 on `bad`/`user_corrected`. Divergence > 10% is a judge-prompt smell, fixed once globally instead of per-task.
- **Scales to any new trace** with zero hand-crafting; eliminates the entire DSL bug class at the cost of ~1 Opus judge call per (task, model) pair.

## Hypothesis

Real human follow-ups are stronger ground truth than synthesised verifier_specs because they encode the user's actual taste, including soft preferences the DSL can't express ("yeah but I prefer the shorter call"). Verifier_spec was always a lossy compression of this signal — replaying the original is more honest.

## Design

1. **Packaging.** For each labelled record, store: `{user_msg, observed_action, user_followup, label, traj_id}`. No `verifier_spec`. No `expected_tool`.
2. **Eval.** Model sees `user_msg` + tool catalogue, returns `(tool_name, args_sketch, reason)`.
3. **Judge prompt** (sketch — no leakage of gold action):
   > "User said: <user_msg>. Agent proposes: <predicted>. The user's actual next message in this conversation was: <user_followup>. Does the proposed action seem to be what the user wanted, judging from how they responded to whatever the agent did? Score 0 / 0.5 / 1 with one-line reason."
4. **Aggregation.** Per-model score = mean over tasks; per-category breakdowns mirror existing bench layout.

## Quality Control

- Calibration sweep: run judge over all `observed_action`s; require ≥90% concordance with `label`.
- Inter-judge agreement: sample 5% with a second judge (sonnet-4.7) — flag tasks where they disagree.
- Hold out 50 tasks for human spot-check each release.

## Use Cases

- Drop-in replacement for current scoring path; verifier_spec becomes optional metadata.
- Onboarding a new trace: label → bench task in one pass, no DSL author needed.
- Cost: ~$0.01–0.03/task/model — acceptable for a clean signal.

## Self-Critique

Trades deterministic-but-buggy DSL scoring for non-deterministic LM-judge variance; if the judge is miscalibrated on a sub-population (e.g. terse follow-ups), the bug surface moves from DSL into prompt engineering and may be harder to detect than a thrown DSL exception.

**Path:** `trace-gepa/proposals/session_replay_oracle.md`
