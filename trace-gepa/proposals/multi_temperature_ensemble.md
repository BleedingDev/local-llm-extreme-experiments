# Multi-Temperature Ensemble

**Round-10 Member #RR**

## TLDR
- Sample N=3 at engineered-diverse temperatures (0.0 greedy, 0.3 mild, 0.7 exploratory) instead of N=3 at the same T=0.7.
- Hypothesis: T=0 anchors max-confidence mode while T=0.7 covers tails — together they span more of the answer distribution than 3 stochastic draws from one mode.
- Aggregate via verifier max-score (run all 3, pick highest) or majority-vote on `tool_name`; head-to-head against best-of-3 same-T on the identical 175-task subset.
- Cost ~$15 (3 Opus calls x 175), parity with best-of-3 — pure structural win if it lifts pass-rate.

## Hypothesis
K-temperature samples are negatively correlated by construction (greedy vs. exploratory draw from different distributions), whereas K-same-T samples are i.i.d. and frequently collapse to the same mode. Engineered diversity should yield a higher "at least one correct" rate per fixed N, especially on tasks where greedy is wrong but a sampled alternative is right (or vice versa).

## Design
- **Eval-time sampler.** For each task, issue 3 calls: T=0.0, T=0.3, T=0.7 (same prompt, same seed-policy where applicable).
- **Aggregation.**
  - *max-score:* score all 3 with the existing verifier; pick the highest.
  - *majority-vote:* tally `tool_name` across the 3; tiebreak by T=0.0.
- **Baseline.** Best-of-3 at fixed T=0.7 (round-8 protocol), identical task subset and verifier.
- **Metric.** Delta pass-rate, plus per-T contribution (how often did T=0 win? T=0.7?). If T=0.7 wins >70% the ensemble degenerates to best-of-N.

## CLI
`--temperature-set 0.0,0.3,0.7` (overrides single `--temperature`); `--ensemble-aggregator {max_score,majority}`.

## Cost
3 x 175 x ~$0.03 ≈ $15 — equal to best-of-3, so the comparison is free of cost confound.

## Use Case
Settles whether diversity-via-T-variation beats diversity-via-stochastic-N. Informs all downstream ensemble work: if multi-T wins, future best-of-N defaults shift; if it loses, stochasticity at high T is the cheaper diversity source.

## Self-Critique
T=0.0 on Opus is rarely truly deterministic and the three temperatures may still collapse to the same answer on easy tasks, leaving the lift visible only on a small hard-task slice that may not reach significance at N=175.