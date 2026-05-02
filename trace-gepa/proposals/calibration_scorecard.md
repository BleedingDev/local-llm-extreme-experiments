# Per-Tool Calibration Scorecard

**Round-6 Brainstorm Member #X — NOVEL proposal**

## 1. Hypothesis

Aggregate accuracy (Opus 30.3% on the 175-task bench) hides a structured signal:
**model errors are not uniformly distributed across tools.** A model can be near-perfect at
`Read` and catastrophic at `Bash` argument synthesis; the two failures demand different
fixes. The failure-classifier (round 4) predicts *category* of error per task; the existing
benchmark reports per-category aggregates. Neither answers: *for tool t, how often is the
model right when it picks t, and how often does it miss t when gold expected it?* Without
that, every prompt edit and routing decision is a guess. A per-tool precision/recall scorecard
turns the 30.3% gap into a ranked, actionable list of tool-level miscalibrations.

## 2. Concrete Output

For each tool `t` over a result file `bench/results_<run_id>.json`, compute:

| Metric | Definition |
|---|---|
| `predicted_t` | count of tasks where model picked `t` as primary tool |
| `expected_t`  | count of tasks where gold trace's primary tool is `t` |
| `correct_t`   | count where predicted == gold == `t` |
| `over_pick_rate` | `(predicted_t - expected_t) / N_total` (signed) |
| `precision_t` | `correct_t / predicted_t` |
| `recall_t`    | `correct_t / expected_t` |
| `f1_t`        | harmonic mean |
| `confused_with` | top-3 tools the model substituted when it missed `t` |

Render `bench/per_tool_scorecard_<run_id>.md` — a sortable markdown table, sorted by
`|over_pick_rate|` desc so the most miscalibrated tools surface first. Example row:

```
| Tool          | Pred | Exp | Over% | P    | R    | F1   | Confused→     |
|---------------|------|-----|-------|------|------|------|---------------|
| EnterPlanMode | 38   | 9   | +16.6 | 0.18 | 0.78 | 0.29 | Read, Bash    |
| Bash          | 22   | 41  | -10.9 | 0.91 | 0.49 | 0.64 | Read, Edit    |
```

Implementation: ~80 LOC `bench/scorecard.py` reading existing `results_*.json` (predictions
already logged) plus `data/gold_traces.jsonl`. No new model calls.

## 3. Cross-Model Scorecard

Run the same scorer over each model's result file and pivot to a single matrix:

```
                Opus    GPT-5.5   Haiku-local
EnterPlanMode  +16.6%   -2.1%    +0.0%
Bash           -10.9%   +3.4%    -22.0%
Read           +1.2%    +0.8%    +5.5%
```

This makes routing decisions concrete: if Haiku-local under-picks `Bash` by 22% but Opus
only by 11%, route Bash-heavy tasks away from Haiku.

## 4. Use Cases

- **Diagnose 30.3%** — quantitatively attribute the gap (e.g. "EnterPlanMode over-pick
  alone costs 9pp; Bash under-pick costs 6pp").
- **Inform prompt edits** — add targeted negative rules ("do not pick EnterPlanMode for
  trace-step tasks") and positive few-shots for under-picked tools.
- **Inform routing** — feed `cost_pareto_router` (existing proposal) with per-tool model
  strengths instead of one global accuracy number.
- **Regression gate** — block prompt edits that improve aggregate accuracy but worsen any
  tool's F1 by >0.1 (avoids whack-a-mole).

## 5. Effort & ROI

1 engineer-day. Pure post-hoc analysis on existing artifacts — zero new eval cost. Directly
explains the 30.3% gap and produces a ranked intervention list, so ROI is bounded below by
"every subsequent prompt experiment is targeted instead of blind."

## 6. Self-Critique

- **Primary-tool reduction is lossy.** Many bench tasks need a *sequence*; collapsing to one
  tool per task hides multi-step miscalibration. Mitigation: also compute a sequence-level
  variant (per-step precision/recall) once primary-tool version proves useful.
- **Small-N noise.** Tools appearing <5 times in gold give unstable precision/recall.
  Mitigation: report Wilson 95% CI and grey out rows with `expected_t < 5`.
- **Correlation ≠ cause.** "Opus over-picks EnterPlanMode" may be the *symptom* of a deeper
  prompt issue (e.g. ambiguous instructions), not the disease. The scorecard tells you
  *where* to look, not *why* — pairs naturally with the failure-classifier (round 4) for
  causal attribution.
- **Gold-trace dependence.** If gold traces themselves are biased toward certain tools, the
  scorecard inherits that bias. Mitigation: cross-check `expected_t` distribution against an
  independent rater on a 20-task sample.

## Self-Assessment vs Constraint

NOVEL — failure-classifier predicts task→category; benchmark reports category→accuracy;
neither produces tool→(precision, recall, over-pick). This is the missing axis.
