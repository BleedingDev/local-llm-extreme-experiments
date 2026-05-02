# Proposal L: Pre-Action Failure-Category Classifier

**Author:** Brainstorm Round-4 Member #L
**Owns:** `trace-gepa/proposals/failure_classifier.md`

---

## 1. Hypothesis

Deterministic preflight predicates (regex on path-existence, command-allowlist, AST checks) cap at ~17% recall because they cannot reason about *semantic-contextual* mismatch. Many failures are lexically innocuous but contextually doomed: a `npm test` invocation in a `pnpm`-only repo, an `Edit` whose `old_string` is duplicated downstream of a `git pull`, a `Bash` invoking a binary the project mentions in README only as a counter-example. A learned classifier consuming `(user_request, recent_actions, candidate_action.input)` should pick up these regularities — n-grams over the candidate command jointly with bag-of-tokens over the rolling context window predict failure category far better than predicates can.

## 2. Concrete Model

**Inputs (concatenated, separated by `[SEP]` sentinels):**
- `context.user_request` (truncated 512 chars)
- `recent_actions` last 5: `tool_name + serialized input` (truncated 256 each)
- `candidate.tool_name` + `candidate.input` (full)

**Featurizer:** `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=80_000)` joined with a second `TfidfVectorizer(analyzer='word', ngram_range=(1,2))` via `FeatureUnion`. Char-wb captures path/CLI fragments; word captures intent terms.

**Output classes:** `{bash_exit_nonzero, hallucinated_path, retry_loop, cancelled_parallel_batch, cmd_not_found_127, edit_string_not_unique, hallucinated_skill, none}` — 8-way softmax.

**Model v0:** `LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0, solver='saga', multi_class='multinomial')`. **v1 upgrade:** `XGBClassifier(objective='multi:softprob', max_depth=6, n_estimators=400)` over hashed features. **v2:** distill into a 22M-param MiniLM head fine-tuned on the same labels for ~3 pp macro-F1 lift.

Calibrate with `CalibratedClassifierCV(method='isotonic', cv=5)` so emitted probabilities are gate-usable.

## 3. Training Data

`dataset_v2.jsonl` (~26K labelled records). Stratified 80/10/10 split by `failure_category`, with **trace-id grouping** (`GroupShuffleSplit`) so the same session never appears in both train and test. Class imbalance handled via `class_weight='balanced'` plus `RandomOverSampler` minority floor of 800. Heldout split is also temporally-forward (last 10% by timestamp) to surface drift.

## 4. Evaluation

- Per-category precision / recall / F1.
- **Quality bar:** macro-F1 ≥ 0.40, with per-category recall ≥ 0.25 for the top-3 frequent classes (`bash_exit_nonzero`, `edit_string_not_unique`, `hallucinated_path`).
- Reliability diagram + ECE (target ECE ≤ 0.08) since downstream gate uses raw probability.
- Compare against deterministic-predicate baseline; report Δrecall at fixed precision = 0.7.

## 5. Deployment Shapes

- **CLI:** `python -m agent_opt.failure_pred --action '{...}' --context '{...}'` → `{"probs": {...}, "top": "edit_string_not_unique", "p": 0.71}`. Cold-start ~120ms via joblib mmap; warm ~3ms.
- **MCP tool:** `predict_failure(action, context)` registered alongside trace-rag; agent calls before risky `Bash`/`Edit`.
- **BAG/Codex gate:** in `dispatch.ts`, after preflight predicates pass, call classifier; if `p(failure ≠ none) > 0.55` AND top-class ∈ {`hallucinated_path`, `edit_string_not_unique`, `cmd_not_found_127`}, route through a confirm-or-revise step instead of executing.

Artifact: single ~6 MB `joblib` blob shipped with the agent; reload-on-SIGHUP for hot-swap after retrains.

## 6. Effort, ROI, Self-Critique

**Effort:** ~2 engineer-days for v0 (featurize + LR + eval harness), +1 day to wire CLI/MCP/BAG. v1 (xgb + calibration) +1 day. Total ≤ 4 days to production gate.

**ROI:** if recall lifts from 17% → 35% at precision ≥ 0.7, expected ~2.1× reduction in user-visible failures on the gated subset. Even at macro-F1 = 0.40, the *negative-class* probability is itself a useful "confidence" annotation for trace logs and downstream RAG.

**Self-critique:** TF-IDF over concatenated context will overweight session-stylistic tokens (e.g., specific repo paths) and underweight true semantic signal — risking shortcut learning that fails to generalize across users; mitigation requires user-stratified eval and feature ablations, otherwise the gate will silently degrade for new tenants.

---

## TLDR

- Train `predict_failure(context, candidate_action) -> 8-way softmax` on ~26K labelled traces using char-wb + word TF-IDF into calibrated logistic regression (v0) → xgboost (v1).
- Group-stratified split (no trace leakage) + temporal-forward heldout; quality bar macro-F1 ≥ 0.40, ECE ≤ 0.08.
- Ship as CLI, MCP tool, and BAG `dispatch.ts` post-predicate gate that routes high-risk top-classes to confirm-or-revise.
- Effort ≤ 4 days; lifts recall ~17% → ~35% at precision ≥ 0.7 — main risk is shortcut learning on repo-stylistic tokens, fixable via user-stratified ablation.

**Path:** `trace-gepa/proposals/failure_classifier.md`

**Self-critique (1 sentence):** A bag-of-n-grams classifier will catch lexical regularities preflight predicates miss but will struggle with truly novel project conventions where the failing token has never been seen in training — the gate may underperform exactly on the long-tail cases that motivated it.
