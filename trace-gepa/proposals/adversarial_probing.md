# Adversarial Probing — Robustness Under user_request Perturbation

**Member #QQ, Round 10**

## TLDR

- Single-prompt PASS is noisy; perturb the user_request and re-run to get a robustness-conditioned pass rate.
- 4 perturbation classes per passed task (synonym swap, restructure, adversarial padding, Czech roundtrip) — 1 attempt each.
- Headline metric: `robust_pass_rate = passed_under_all_4 / total_original_passes` — orthogonal to round-9's `tool_availability_sweep` (that varies tools; we vary the request itself).
- Cost ~$5 (4 x 82 audited passes); flags brittle wins without re-running fails.

## Hypothesis

A model that PASSes only the canonical phrasing is overfit to the eval surface. True capability survives 1-word rephrases, clause reordering, distractor padding, and translation roundtrips. The delta between `pass_rate` and `robust_pass_rate` quantifies eval brittleness — high delta means our scorecard overcounts.

## Perturbation classes

1. **Synonym swap (low):** swap 1-2 verbs via a fixed lexicon (`read->open`, `find->locate`, `create->make`, `delete->remove`, `list->enumerate`). Mechanical; deterministic.
2. **Restructure (medium):** reorder independent clauses, normalise comma density (e.g. "Read foo.py, then summarise it" -> "Summarise foo.py after reading it"). Done by a cheap Haiku rewriter pinned to "preserve every entity and constraint".
3. **Adversarial padding (high):** prepend 2-3 sentences of plausible-but-irrelevant context ("I was just chatting with Sarah about Q3 plans. Anyway —"). Tests whether the model latches on to the actual request.
4. **Czech roundtrip (high):** EN -> CS -> EN via Haiku. Preserves intent in a non-Latin-friendly source language, stresses semantic compression. (User-natural-language constraint: caller is Czech-fluent.)

## Protocol

- Apply only to the 82 audited per-cat-v2 passes.
- Per task: 4 perturbed runs, each with a fresh tool-call session, scored by the existing rubric. ANY failure => task `not_robust`.
- Emit `robust_pass_rate` plus a per-class breakdown so we can see WHICH perturbation class each model is fragile to (signal for prompt hardening).
- Store perturbed prompts in `trace-gepa/data/adversarial/<task_id>/<class>.json` for reproducibility.

## Cost

4 calls x 82 tasks ~ 328 Opus runs. At observed ~$0.015/run for short tool-use traces ~ $5. Plus ~$0.30 of Haiku for the rewriters. Cheap.

## Self-critique

The Czech roundtrip and Haiku-restructured prompts introduce rewriter-quality noise — a fail there may indict the rewriter, not the agent; mitigate by spot-checking 10 perturbed prompts manually before scoring.

---

**Path:** `trace-gepa/proposals/adversarial_probing.md`

**Self-critique (1 sentence):** Rewriter-induced noise (especially in the Czech roundtrip class) risks false `not_robust` flags, so the metric needs a manual spot-check on a 10-prompt sample before being trusted.
