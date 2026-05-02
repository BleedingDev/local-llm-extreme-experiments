# Proposal: Unsupervised Anomaly Detector ("Off-Rails" Early Warning)

**Author:** Brainstorm Round-5 Member #P
**Status:** Draft

## TLDR

- Fit an unsupervised model on `good`-only `(context, action)` embeddings; anything far from that manifold at runtime triggers a halt+ask, regardless of whether it matches a labelled `failure_category`.
- Complements (does not duplicate) Round-4's supervised failure classifier: catches **novel** failure modes the labelled corpus has never seen, including silent drift as the user's habits evolve.
- Distinct from Round-2 preflight: preflight encodes deterministic predicates ("never `rm -rf /`"); this is a learned distributional check ("the user has never done anything *like this* before").
- Ships as MCP tool `score_action_anomaly(context, action) -> {anomaly_score, percentile, halt_recommended}` with a single tunable threshold (default p95 from a held-out good split).

## 1. Hypothesis

Supervised failure prediction has a coverage ceiling: it can only flag failure modes present in the labelled `bad` corpus. **OOD detection on the `good` manifold has the inverse property** — it flags everything that doesn't look like the user, even modes nobody has labelled yet. The two are complementary: AND/OR-ing them gives both precision (known-bad patterns) and recall (novel weirdness). A trace the agent has *never seen this user do* is intrinsically suspicious.

## 2. Concrete Approach

**Embedding.** Reuse `agent_opt.rag.embed` (TF-IDF over the joined `(context, observed_action)` string). TF-IDF is already wired into the RAG path — zero new infra cost. Fall back to sentence-transformers (`all-MiniLM-L6-v2`) behind a flag if TF-IDF underperforms.

**Model.** Fit three candidates on the `good`-labelled split, pick by held-out AUC:
1. `IsolationForest(n_estimators=200, contamination='auto')` — robust default, handles sparse TF-IDF.
2. `LocalOutlierFactor(n_neighbors=20, novelty=True)` — better when the `good` manifold has tight local clusters per task-type.
3. `OneClassSVM(kernel='rbf', nu=0.05)` — slower, baseline only.

**Runtime.** Project candidate `(context, proposed_action)` into the same TF-IDF space, get raw `decision_function` score, normalize to a 0-1 percentile against the `good` training distribution, halt if `percentile > 0.95`.

**Artifacts.** Persist the fitted vectorizer + detector to `trace-gepa/artifacts/anomaly_detector.joblib`; version with the `good` corpus hash.

## 3. Eval Methodology

- **Train:** 80% of `good` records.
- **Held-out positives (should score high anomaly):** all `bad` + `user_corrected` records.
- **Held-out negatives (should score low anomaly):** 20% held-out `good` (≥100 records).
- **Primary metric:** ROC-AUC on good-vs-(bad+corrected). **Bar: AUC ≥ 0.65.**
- **Secondary:** precision@halt-rate=5% (what fraction of top-5%-anomalous actions are actually bad/corrected). Bar: ≥ 2x base rate.
- **Ablation:** report AUC for each of the three models; pick best.
- **Drift check:** retrain monthly on rolling `good`; log AUC delta to catch silent degradation.

## 4. Deployment

MCP tool surface:

```
score_action_anomaly(context: str, action: str) ->
  { anomaly_score: float,        # raw detector score
    percentile: float,           # 0-1 against training good
    halt_recommended: bool,      # percentile > threshold
    nearest_good_examples: list  # top-3 for explainability
  }
```

Wired into the agent loop pre-execution alongside the Round-4 classifier: halt if **either** fires. `nearest_good_examples` lets the user see *why* it looked weird, which dramatically improves the override UX vs. an opaque score.

## 5. Effort + ROI + Self-Critique

**Effort:** ~1.5 days. Embedding + fit is ~80 LOC (sklearn). MCP wrapper + threshold calibration is ~120 LOC. Eval harness reuses the failure-classifier split.

**ROI:** High if the `bad`/`user_corrected` set is small (which is the regime where supervised classifiers struggle). Each caught novel failure mode is one we *could not* have caught with Round-4. Even at AUC 0.65 — modest — the marginal failures caught are pure additive value over the supervised baseline.

**Self-critique:** TF-IDF on `(context, action)` strings is a weak semantic signal — surface paraphrases of "normal" actions may score anomalously, producing false-halt fatigue; the threshold must be set conservatively (p95+) and the tool needs a per-user-feedback loop ("this was fine") to retrain the `good` corpus, otherwise the user will learn to ignore halts within a week.