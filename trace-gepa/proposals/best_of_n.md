# Best-of-N Self-Consistency (Round-8 #FF)

## TLDR
- **Single-sample greedy underestimates capability.** Action-selection variance is real at T=0.7; the right tool often surfaces in 1-of-3 even when 1-of-1 misses.
- **Reuse existing `--attempts N` knob, add `--aggregation {first-pass|majority-vote|max-score}`** so the harness can collapse N independent samples into one verdict.
- **N=3 @ T=0.7 ~= 3x cost (~$15 over the full 175-task suite)** — cheap relative to a model upgrade and orthogonal to prompt edits.
- **Predict baseline 30.3% (N=1) -> 36-40% with best-of-3-by-verifier**, with `max-score` strictly dominating `majority-vote` on disjoint-tool tasks.

## Hypothesis
Greedy decoding under-samples the capability frontier. For action-selection (tool_name + args), the model's modal token is often *not* the verifier-passing one; thermal sampling at T=0.7 spreads probability mass across plausible alternatives. Aggregating N samples by an oracle-cheap verifier (or by majority vote over `tool_name`) recovers a non-trivial fraction of those near-misses without touching the policy.

## Design
1. **CLI surface.** Bench already accepts `--attempts N`; extend with `--aggregation`:
   - `first-pass`: stream samples; emit the first with `score > 0`; abort the remaining N-1 calls. Lowest expected cost (~1.6x at 30% pass rate).
   - `majority-vote`: take all N, count `tool_name` (and optionally args-hash); emit the modal sample. No verifier dependency — works in trace-gepa training loops where verdicts are slow.
   - `max-score`: run all N through the verifier, emit `argmax(score)` (ties broken by sample index). Highest lift, full Nx cost.
2. **Sampling.** Per task, fan out N=3 independent calls at `temperature=0.7`, identical prompt, distinct seeds. Reuse the existing per-attempt logging so trace-gepa can mine the *losing* samples as negatives later.
3. **Reporting.** Emit per-task `attempts_scores: [s1,s2,s3]` and `selected_idx` so we can compute the *oracle-best-of-N* upper bound for free — that's the headroom number.

## Cost
- Full suite = 175 tasks. At ~$0.03/Opus call, N=1 ~= $5; N=3 with `max-score` ~= $15. `first-pass` at 30% baseline ~= $8 expected. All tolerable for a sweep.

## Predicted A/B
- Baseline N=1: 30.3%.
- N=3 `majority-vote`: 33-35% (capped by tasks where the wrong tool is also the modal one).
- N=3 `max-score` (verifier-routed): **36-40%**, with oracle-best-of-3 likely ~45% as the ceiling.

## Self-critique
Best-of-N papers over a noisy policy rather than fixing it; if the model is *systematically* miscalibrated on a task class (e.g., always picks `read_file` when `grep` is needed), no amount of resampling helps and we'd be better off optimizing the prompt — so this should ship as a *baseline-lift* lever, not a substitute for trace-gepa's reflective edits.

**Path:** `/Users/satan/side/experiments/local-coding-benchmark/trace-gepa/proposals/best_of_n.md`

**One-sentence self-critique:** Cheap headroom probe, but it masks policy defects rather than repairing them — use it to *measure* the variance ceiling, not as the final answer.
