# Anomaly Detector Eval Report

- algo: LocalOutlierFactor(n_neighbors=10, novelty=True) [train capped @ 8000]
- vectorizer: reused `trace-gepa/artifacts/rag_index_v2/vectorizer.pkl` (no re-fit)
- train good records: 17842
- **ROC-AUC: 0.7293**
- verdict: **SHIP**

## Per-category mean anomaly score

| category | n | mean | std |
| --- | ---: | ---: | ---: |
| bash_exit_nonzero | 2725 | 0.077 | 0.238 |
| good_heldout | 4461 | 0.012 | 0.077 |
| bad | 324 | 0.007 | 0.024 |
| user_correction | 67 | 0.000 | 0.000 |
| cmd_not_found_127 | 17 | 0.000 | 0.000 |
| retry_loop | 10 | 0.000 | 0.000 |
| bash_timeout_141 | 243 | 0.000 | 0.000 |
| hallucinated_skill | 2 | 0.000 | 0.000 |
| cancelled_parallel_batch | 100 | 0.000 | 0.000 |

## Top-10 anomalies

- score=1.000 id=`cc_v2_5129dd79e9_evt00210` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn clusterer teammate"> You are `clusterer`, a teammate in the `tachiom-zig` agent team. The team is imple'
- score=1.000 id=`cc_v2_69409e9f5f_evt00148` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn indexer teammate"> You are `indexer`, a teammate in the `tachiom-zig` agent team. The team is implement'
- score=1.000 id=`cc_v2_001e761a81_evt00148` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn indexer teammate"> You are `indexer`, a teammate in the `tachiom-zig` agent team. The team is implement'
- score=1.000 id=`cc_v2_d5c00d8e05_evt00214` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn primitives-engineer teammate"> You are `primitives-engineer`, a teammate in the `tachiom-zig` agent tea'
- score=1.000 id=`cc_v2_2a25f6dd0d_evt00210` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn clusterer teammate"> You are `clusterer`, a teammate in the `tachiom-zig` agent team. The team is imple'
- score=1.000 id=`cc_v2_032681179b_evt00210` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn clusterer teammate"> You are `clusterer`, a teammate in the `tachiom-zig` agent team. The team is imple'
- score=1.000 id=`cc_v2_49a4c8c014_evt00148` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn indexer teammate"> You are `indexer`, a teammate in the `tachiom-zig` agent team. The team is implement'
- score=1.000 id=`codex_d8f76367_evt000465` label=good cat=None tool=update_plan excerpt='Read-only performance research lane. Re-check highest-ROI bottlenecks in libs/platform/provisioning/usecase tests, focusing on any modified files in git status:'
- score=1.000 id=`cc_v2_054b4fdf7e_evt00214` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn primitives-engineer teammate"> You are `primitives-engineer`, a teammate in the `tachiom-zig` agent tea'
- score=1.000 id=`cc_v2_7126e29f16_evt00214` label=bad cat=bash_exit_nonzero tool=Bash excerpt='<teammate-message teammate_id="team-lead" summary="Spawn primitives-engineer teammate"> You are `primitives-engineer`, a teammate in the `tachiom-zig` agent tea'

## Complementarity

Orthogonal to round-2 preflight (deterministic predicates: perfect precision, zero novel-coverage) and round-4 failure-classifier (supervised: bounded by labelled corpus). This anomaly detector fits the `good` manifold and flags ANY OOD record, catching novel/zero-day weirdness neither of the others can see. Production wiring: halt if (preflight fires) OR (classifier > threshold) OR (anomaly percentile > 0.95).
