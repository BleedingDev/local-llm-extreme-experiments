# Session Completion Predictor

## TLDR
- **What**: At turn 5 of a session, predict P(succeed | first_5_actions) using a small sklearn classifier; flag doomed sessions early so BAG can offer abort/restart.
- **Why novel**: Operates at SESSION granularity over a sliding window of K turns — distinct from round-5 anomaly detector (per-action OOD scoring) and round-4 failure classifier (per-action category labels). Predicts global trajectory outcome, not local correctness.
- **How**: Hand-crafted features (bad_count, correction_count, tool_diversity, mean tool-name perplexity from a unigram LM over all-traces, error-keyword presence in result_excerpts, mean turn latency) → LogReg + GBT ensemble; labels derived from full-session signals (ai_title generated, positive closing tokens, no abandonment) and corrections-density tail.
- **Deploy**: `bag --check-trajectory` runtime hook + MCP tool `predict_session_outcome(session_prefix)`; if P(succeed) < 0.35, surface a soft "this looks rough — abort & restart?" prompt.

## Path
`trace-gepa/proposals/session_completion_predictor.md`

## Hypothesis
Session-level outcome correlates with early-turn signals: density of `bad` records, user corrections, tool churn, and error-keyword leakage in tool results within the first K turns are strong predictors of eventual session failure. If true, an early-abort signal at turn 5 saves wasted user time and cost.

## Concrete Design
- **K = 5** (tunable 3–8 via grid search on dev split).
- **Features** (12-dim vector per session prefix):
  1. count(`bad` records) in first K
  2. count(user_corrected) in first K
  3. tool_diversity = |unique tools| / K
  4. mean tool-name perplexity (unigram LM over corpus)
  5. error-keyword hits ("Error", "failed", "ENOENT", "Traceback") in result_excerpts
  6. mean turn latency
  7. retry_ratio (same tool consecutive)
  8. avg result_excerpt length
  9. count(empty/null results)
  10. user message sentiment (VADER)
  11. count(file_not_found patterns)
  12. tool_entropy (Shannon over tool distribution)
- **Labels**: `succeeded=1` if last user message has positive tokens OR ai_title generated AND no trailing correction streak; `failed=0` if user abandoned (no closing message) OR last 3 actions all corrected.
- **Model**: GBT (sklearn `GradientBoostingClassifier`, depth=3, n_est=200); calibrated with Platt scaling.
- **Eval**: 70/30 session split, AUC ≥ 0.7, calibration ECE ≤ 0.08.

## Effort + ROI
- Effort: ~2 days (feature extractor + train + MCP wrapper).
- ROI: high — prevents 10–20% of doomed sessions from burning a full hour of user time.

## Self-critique
Label noise from heuristic outcome derivation may cap AUC near 0.7; ai_title proxy may bias toward sessions that simply ran longer rather than truly succeeded.
