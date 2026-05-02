# Proposal: Trace Curriculum — Easy → Hard Ordering for Downstream Training

**Author:** Brainstorm Round-4 Member #N
**Scope:** Curriculum ordering of trace records for any downstream learner (SFT / GEPA / DPO / distillation).
**Owns:** `trace-gepa/proposals/trace_curriculum.md`

## TLDR
- Order, don't shuffle: rank trace records by a cheap heuristic difficulty score, then feed the learner in 3 stages (easy → mixed → hard/corrected) instead of the usual random shuffle.
- Difficulty is computable with zero LM calls — purely from fields we already extract (`user_request` length, `recent_actions` count, `tool`, `label`, `failure_category`, `user_corrected`).
- Each stage emits a JSONL the existing SFT/GEPA/DPO pipeline already consumes — this is a re-orderer, not a new format. Drop-in.
- Hypothesis: with a fixed step budget (e.g. 2k MLX-LoRA steps), curriculum beats random on bench score AND converges in fewer steps; downside risk is bounded because Stage-3 still contains the hard tail.

## Hypothesis
On limited budgets (which is all we have on-device), starting on confidently-good short traces lets the learner lock in tool-calling syntax and common control flow before being exposed to ambiguous/recovered examples. This should (a) raise final bench score on `benchmark_tasks.jsonl`, and (b) reach any given score in fewer steps — the part that matters for MLX-LoRA on a laptop.

## Difficulty heuristic (5-line pseudocode)
```
def difficulty(r):
    s  = 0.30 * min(len(r.context.user_request)/2000, 1.0)
    s += 0.25 * min(len(r.context.recent_actions)/20, 1.0)
    s += 0.20 * (0.0 if r.tool in TOP10_TOOLS else 1.0)
    s += 0.15 * (1.0 if r.failure_category else 0.0) + 0.10 * (1.0 if r.user_corrected else 0.0)
    return s if r.label == "good" else s + 0.20    # bad/unknown labels are harder
```

## Curriculum stages (each is a JSONL drop-in)
- `stage1_easy.jsonl`  — top-1000 lowest score, `label=good`, `failure_category=None`. Teaches base syntax + common tools.
- `stage2_mixed.jsonl` — next 1000 across the median band; mix of tools, mild failures allowed. Teaches recovery and tool diversity.
- `stage3_hard.jsonl`  — top-500 highest score: long context, ambiguous, `user_corrected=True` or `failure_category!=None`. Teaches the long tail.

Pipeline integration: trainer reads the three files in order; for GEPA, use them as successive `validationExamples` waves; for DPO, Stage-3 supplies the `rejected`-side recoveries naturally.

## Empirical test (cheap)
1. Build all three stages from `data/dataset.jsonl` + `data/dataset_recovery.jsonl`.
2. Train two MLX-LoRA runs at identical step budget: (A) curriculum order, (B) random shuffle of the union.
3. Evaluate both on `data/benchmark_tasks.jsonl`. Report bench score and steps-to-threshold.

## Effort + ROI
- Effort: ~half-day. One ranker script (`scripts/build_curriculum.py`) + a flag on the existing trainer. No new infra.
- ROI: if it adds even +2 bench points on the same budget, it's free wins for every other proposal that trains a model — it composes with behavioral_cloning, correction_ruleset, GEPA tuning, etc.

## Self-critique
The heuristic is correlational, not causal — "short request" and "common tool" may simply select a narrow distribution; if Stage-1 is too narrow the model could overfit before Stage-2 corrects it, so the random-shuffle baseline is a real risk and must be run.
