# Cost-vs-Quality Pareto Router

**Round-5 Member #Q — NOVEL angle: cost optimisation via predictive routing (not prompt/model improvement).**

## TLDR
- Opus is ~10x Haiku; observed traces show many `label=good` actions are trivially routable to Haiku — predicting WHICH unlocks 50%+ cost savings with zero quality loss.
- Train a small sklearn classifier on `(context, user_request)` features that emits `(tier, confidence)`; deploy as a BAG runtime hook before each LM call, env-overrideable.
- Gold labels via replay: re-run each `good` record on Haiku, grade pass/fail — pass = "haiku-sufficient", fail = "opus-required". Sonnet is the middle bucket.
- Hold-out eval on 30 tasks: classifier accuracy + live A/B (Opus-always vs router) comparing $ and quality delta.

## Hypothesis
A large fraction of trace records labeled `good` involved cheap, mechanical actions (file read, single-line edit, `ls`, grep). These do not need Opus. If a classifier with >85% precision on "haiku-sufficient" exists, naive routing saves >50% of inference $ at <2% quality regression.

## Training Data
- **Source**: `trace-gepa/codex_session_models.json` (already has model field per session) joined with the existing `label=good` filtered records. For Claude Code traces, walk session_meta to recover the user's active model.
- **Features** (cheap, all extractable from `(context, user_request)` pre-call):
  - `len(user_request)`, token count
  - `recent_actions` count + type-diversity (entropy over tool names in last N)
  - `available_tools` count
  - `observed_tool` (one-hot)
  - context complexity proxies: file count in cwd, depth of recent_dirs, presence of test/build keywords
- **Labels**:
  1. *Observed*: which tier the user actually ran (weak signal — biased toward whatever they had configured).
  2. *Gold via replay*: re-run each `good` record on Haiku; if action matches + grader passes, label `haiku-sufficient`; else escalate to Sonnet replay; else `opus-required`.

## Model
- `sklearn.LogisticRegression` (interpretable, calibrated probabilities) and `RandomForest` (non-linear baseline). Pick whichever has higher AUC on hold-out.
- Output: `{tier: "haiku"|"sonnet"|"opus", confidence: float}`. Threshold-tune to a target precision (e.g. 0.9 on haiku-sufficient → conservative routing).

## Deployment
- BAG runtime hook: `route_model(ctx) -> tier` invoked before LM dispatch. Env override: `BAG_FORCE_TIER=opus` for debugging.
- Confidence floor: <0.7 → fall back to default tier (no risky downroutes).

## Eval
- Hold out 30 tasks. Measure (a) classifier accuracy vs gold labels, (b) live: run Opus-always and router-predicted in parallel, compare grader-pass-rate and $ spent.
- Success: ≥40% $ saved, ≤3% pass-rate drop.

## Effort + ROI
- ~6 BAG-hours: 2 replay-grading, 2 feature pipeline + train, 1 hook integration, 1 eval. ROI: at typical 100k token/day per dev, ~$8/dev/day saved → break-even after one BAG-hour of routing.

## Self-critique
Replay-grading on Haiku is itself expensive and the gold label may drift as models update — recommend re-training quarterly and starting with a tiny seed of 200 records before scaling.
