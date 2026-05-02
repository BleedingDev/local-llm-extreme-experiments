# Proposal: Tool-Availability Sweep (Round-9 #LL)

## Hypothesis
Robustness to varying `available_tools` lists is an unmeasured capability axis distinct from raw tool-selection accuracy. Today's audit found 12 tasks where the expected `Bash` was missing from `available_tools`, yet the prompt template instructs "pick from available_tools only." We don't know whether models honour that constraint, gracefully fall back, or hallucinate the missing tool. A sweep that perturbs the tool list per task surfaces three failure modes — refusal-deficit, hallucination, and order-sensitivity — that the current single-shot eval cannot see.

## Concrete Design
For each task `t` with `available_tools = A` and expected primary action `P`:

- **Variant A (drop):** run with `A \ {P}`. Expected: model picks a documented fallback (e.g., `Read` instead of `Bash cat`) or returns a refusal/empty action with rationale.
- **Variant B (irrelevant add):** run with `A \cup {X}` where `X` is sampled from a fixed irrelevant pool (`WebSearch`, `WebFetch`, `NotebookEdit`) chosen to not match `t`'s domain. Expected: `X` is never selected.
- **Variant C (shuffle):** run with `permute(A)` using 3 fixed seeds. Expected: action distribution is invariant across seeds.

### Metrics (per model, aggregated over tasks)
- **Refusal rate (A):** fraction where output is empty / contains explicit refusal token. Ideal: high when no fallback exists; low when a clean fallback exists. Split by sub-bucket.
- **Fallback-correctness (A):** of non-refusals, fraction picking a tool in `A \ {P}` that a human-labelled key marks as acceptable.
- **Hallucination rate (B):** fraction where chosen tool ∉ `A ∪ {X}` OR equals `X` despite irrelevance. Lower is better.
- **Order sensitivity (C):** Jensen-Shannon divergence of tool-choice distribution across the 3 shuffles. Lower is better.

### Output
`bench/robustness_sweep.md` — table with rows = models, columns = (refusal_rate, fallback_correct, hallucination, order_JSD), plus per-task drill-down appendix.

### Effort
~150 LoC in `bench/sweep.py` reusing existing task loader + judge. LM cost ≈ 3× current eval (3 variants × N tasks); seed-3 shuffle reuses cached prompts where possible. Worth it: a single number per axis per model is a durable robustness signal.

## TLDR
- Perturb `available_tools` three ways (drop-P, add-irrelevant, shuffle) per task to expose refusal, hallucination, and order axes.
- Reuses existing harness; ~150 LoC plus a small acceptable-fallback key.
- Emits `bench/robustness_sweep.md` with four per-model scores.
- Directly addresses today's 12-task audit gap.

## Path
`trace-gepa/proposals/tool_availability_sweep.md`

## Self-Critique
Fallback-correctness depends on a hand-labelled "acceptable alternates" key, which is subjective and adds annotation burden — without it, drop-variant scoring collapses into refusal-rate alone and loses most of its signal.
