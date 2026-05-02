# Inter-Model Disagreement as Hardness Signal

**Author:** Brainstorm Round-7 Member #AA
**Date:** 2026-05-01

## TLDR
- Task hardness is multi-dimensional; a single pass-rate hides whether failure is structural, model-idiosyncratic, or prompt-pathological.
- Cross-tabulating Opus (53/175) vs GPT-5.5 (42/175) yields a 2x2 contingency per task; the off-diagonal cells (~30-40 tasks expected) are the *discriminating* benchmark.
- Off-diagonal tasks are direct routing fuel: Opus-only-pass routes to Opus, GPT-5.5-only-pass routes to GPT-5.5; both-fail tasks are quarantine candidates.
- Re-weighting bench score by `1/disagreement_rate` makes the leaderboard primarily measure the tasks that actually distinguish models.

## Hypothesis
Mean pass-rate collapses three failure modes (structural impossibility, model-specific blind spot, prompt pathology) into one number. Inter-model agreement structure recovers them: agreed-pass = easy, agreed-fail = structural-or-pathological, disagreement = model-idiosyncratic. The off-diagonal mass is the only signal that improves a router or a curriculum.

## Concrete Output
1. **Contingency table** per task: `(opus_pass, gpt55_pass) in {TT, TF, FT, FF}`.
2. **Aggregate**: counts per cell, Cohen's kappa, disagreement rate stratified by declared difficulty (easy/med/hard).
3. **Top-20 Opus-only-pass** (TF) — sorted by GPT-5.5 confidence-of-fail (longest trace, most retries) so we surface the *most surprising* GPT-5.5 misses.
4. **Top-20 GPT-5.5-only-pass** (FT) — symmetric ranking.
5. **Top-20 both-fail** (FF) — sorted by combined token spend; high spend + both-fail = pathological prompt; low spend + both-fail = genuinely impossible.

## Use Cases
- **Round-5 cost-Pareto router**: TF/FT lists are the labeled training set for a per-task router; expected uplift bounded by `|TF| + |FT|` over the better single model.
- **Bench curation**: drop FF-cheap (likely broken), invest in TF/FT (model-discriminating).
- **Re-weighted score**: `weight_i = 1 / max(disagreement_rate_bucket_i, eps)`; agreed-easy tasks contribute near zero, off-diagonal tasks dominate — closer to a true skill measure.

## Implementation
~80 LoC pure post-hoc Python: load two jsonl files, join on `task_id`, build pandas crosstab, write three ranked CSVs + one summary markdown. No new model calls. Single script: `trace-gepa/scripts/inter_model_disagreement.py`. Runtime <5s.

## Effort + ROI
- **Effort**: 1-2h (script + writeup). Zero infra, zero spend.
- **ROI**: high — unblocks router (Round 5), bench reweighting, and FF triage simultaneously from one artifact.

## Self-Critique
With only two models the 2x2 is noisy and conflates "GPT-5.5 weakness" with "single-seed variance"; needs >=3 models or seed-replicates before TF/FT lists are trusted as routing labels.

## Path
`trace-gepa/proposals/inter_model_disagreement.md`
