# Cross-Task Transfer Matrix — Brainstorm Round-7 (#CC)

## TLDR
- Build M[a][b] = P(pass b | pass a) over the joint Opus + GPT-5.5 results (175 tasks × N runs), exposing latent task families instead of treating the bench as 175 i.i.d. coin flips.
- Spectral cluster M (symmetrized) into K groups; each cluster's medoid task becomes a **bellwether** — passing it predicts the cluster, failing it condemns the cluster.
- Two immediate payoffs: a **mini-bench** (one task per cluster ≈ score the full 175 within ±X%) and a **diagnostic readout** ("model is weak on cluster 4: I/O-heavy refactors").
- Bench audit: clusters of size > 10 with intra-cluster pass-corr > 0.9 are candidates for pruning — we're paying compute for redundant signal.

## Hypothesis
Bench tasks are **not independent Bernoullis**. They share substrate skills (regex, async, file-IO, test-reading, multi-file edits). A correlation-aware view compresses the bench, surfaces structural weaknesses per model, and lets us reason about *which* tasks earn their slot.

## Concrete Output
1. **Data assembly**: load every per-task pass/fail row from `reports/` for both models, stack into a `(n_runs, 175)` boolean matrix `R`.
2. **Conditional matrix**: `M[a,b] = (R[:,a] & R[:,b]).sum() / max(R[:,a].sum(), 1)`. Apply Laplace smoothing (+1/+2) to handle tasks with ≤2 passes. Note: M is asymmetric — keep both directions.
3. **Affinity for clustering**: `A = 0.5 * (M + M.T)`, then spectral clustering (`sklearn.cluster.SpectralClustering`, `affinity='precomputed'`, K chosen by eigengap heuristic on the normalized Laplacian, expect K≈8–15).
4. **Bellwether per cluster**: medoid = task with highest mean A to its cluster-mates. Tie-break by task with closest pass-rate to cluster average (avoids picking trivially-easy/hard medoids).
5. **Artifacts**: `bench/transfer_matrix.npz` (M, A, labels, task_ids), `bench/transfer_matrix_heatmap.png` (re-ordered by cluster, block-diagonal view), `trace-gepa/proposals/transfer_matrix_clusters.md` (clusters + bellwethers + suspected shared skill, hand-labeled).
6. **Mini-bench validation**: leave-one-model-out — score on bellwethers only, regress against full-175 score, report R² and MAE.

## Implementation
~120 LoC: `numpy`, `sklearn.cluster.SpectralClustering`, `matplotlib`. Single script `scripts/build_transfer_matrix.py`. Pure post-hoc on existing run logs — no re-running tasks.

## Effort + ROI
- Effort: ~3 hours (data plumbing dominates; clustering is one call).
- ROI: **High**. Mini-bench cuts iteration cost ~10× for prompt-engineering loops; cluster diagnosis turns a scalar score into a skill profile usable by GEPA reflection.

## Self-Critique
With only 2 models (Opus + GPT-5.5) the matrix is severely under-determined — many M entries rest on <5 joint passes; clusters will be noisy and need a third+ model before bellwethers are trustworthy.

---
Path: `trace-gepa/proposals/transfer_matrix.md`

Self-critique (1 sentence): Two models is too few signals for stable 175×175 conditional probabilities — treat clusters as hypotheses pending a third model run, not ground truth.
