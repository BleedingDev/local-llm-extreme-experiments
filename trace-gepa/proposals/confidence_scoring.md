# Proposal: Confidence-Scored Bench Results

**Round-9 Member #NN** — NOVEL vs. round 8 (best-of-N): best-of-N **picks** one from K; this **averages** K and reports dispersion. Different downstream use: best-of-N is for inference-time gain; confidence scoring is for honest measurement.

## TLDR

- Single-run pass-rate has variance comparable to small effect sizes; report `mean ± std` over K runs at T>0 to make small effects credible.
- `--samples K` flag (default 1, recommended 3) re-runs each task K times at `--temperature 0.7`; emit per-task `pass_rate_mean`, `pass_rate_std`, raw scores, and a bench-level CI.
- Cost is K× per-task LM cost; K=3 is the sweet spot — enough to detect API noise, cheap enough to run on every PR.
- Leaderboard column becomes `pass@1 mean ± std (n=K)`; comparisons with overlapping CIs are reported as "tied" rather than spurious wins.

## Hypothesis

Two recent regressions (-1 task on the normaliser change, -1 task on the few-shot smoke) sat well within the expected single-run variance for a 50-task suite at T=0.7. Without dispersion data, every ±1pp wiggle gets debated as if it were signal. A K-run mean with a std bar makes the noise floor explicit so reviewers can ignore sub-noise deltas.

## Concrete design

1. **CLI:** `bench run --samples K --temperature 0.7` (K defaults to 1 for backward compat).
2. **Per-task record:**
   ```json
   {"task_id":"...", "scores":[1,0,1], "pass_rate_mean":0.667, "pass_rate_std":0.471, "n":3}
   ```
3. **Aggregate:** bench mean = mean of per-task means; bench std = sqrt(mean(per-task variances) / n_tasks) (treating tasks as independent samples).
4. **Leaderboard:** render as `62.0 ± 2.1 (n=3)`; flag rows where another row's CI overlaps as "tied" in a `notes` column.
5. **High-variance task report:** any task with std > 0.4 surfaces in a `flaky_tasks.json` artefact for triage (likely under-specified prompt or pathological grader).

## Use cases

- **Real lift vs. noise:** a +2pp change with non-overlapping CIs survives; one within ±std collapses honestly.
- **Flaky task discovery:** high-std tasks are either ambiguous (rewrite) or pathological (drop).
- **Honest model comparisons:** Opus vs. Sonnet at overlapping CIs ⇒ "tied," not a fake win.

## Self-critique

K=3 gives a very rough variance estimate (chi-square with 2 dof is wide); a methodologically pure version would need K≥10, which triples-or-worse the eval bill — proposal punts this trade-off to the user via the `--samples` knob rather than fixing it.

---

**Path:** `trace-gepa/proposals/confidence_scoring.md`
**Self-critique (1 sentence):** K=3 gives a noisy variance estimate, so the std bar itself has a std bar — useful directionally, not for publication-grade significance claims.
