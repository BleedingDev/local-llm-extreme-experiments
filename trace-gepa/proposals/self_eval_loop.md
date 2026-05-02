# Self-Eval Feedback Loop: BAG Runtime → Bench

**Author:** Brainstorm Round-7 Member #DD

## TLDR
- Bench staleness is the #1 long-term risk; a frozen 175-task suite drifts from the user's actual workload within weeks.
- Hook BAG session-end to emit `(initial_user_request, sequence_of_actions, outcome)` and pipe through extractors → categorize → sanitise → LM-judge quality gate.
- Auto-ingested candidates land in `tasks/auto-ingest/<date>/` with weekly human review; successful runs become positive cases, failed runs become negative + counterfactual seeds.
- Closes the production-eval loop: bench grows organically (175 → 500+ over months) and tracks behavioural drift the snapshot cannot.

## Hypothesis
Static benches decay. The user's real distribution shifts (new repos, new languages, new tooling) faster than maintainers can hand-author tasks. If every BAG session is a *free* labelled trace, ignoring them is wasteful. Auto-ingestion makes the bench a living mirror of the user's work.

## Pipeline
1. **BAG hook** (`session_end`): dump trace tuple `(prompt, action_seq, outcome, repo_state_hash)`.
2. **Convert** via `extractors/build_benchmark_tasks.py` style — lift initial state, gold actions, success oracle.
3. **Categorise** with `categorize.py` (already exists for the static suite — reuse).
4. **Sanitise** through `sanitise.py` (scrub paths, secrets, identifiers, names) BEFORE any storage.
5. **Quality gate**: LM-judge scores candidate on (a) reproducibility, (b) signal density, (c) novelty vs existing suite. Pass threshold → queue.
6. **Queue** to `tasks/auto-ingest/<YYYY-MM-DD>/` with provenance metadata.
7. **Weekly review**: human accepts top-N (Pareto on novelty × quality), rejects rest, merges into main suite.

## Cadence
Weekly merge. Caps growth at ~20 tasks/week to keep review load bounded.

## Privacy
Sanitise.py runs *before* the candidate ever leaves the BAG sandbox. No raw prompt/path persists in the auto-ingest queue.

## Effort + ROI
- **Effort:** ~3 days — hook, extractor adapter, judge prompt, review CLI.
- **ROI:** Bench compounds. Doubles in size in ~6 months at zero authoring cost. Catches regressions invisible to the snapshot suite.

## Self-critique
Selection bias is real — the bench will overweight tasks the user *currently* does, blinding it to under-served capability gaps; mitigate with a quota reserving review slots for hand-authored adversarial tasks.

---
**Path:** `trace-gepa/proposals/self_eval_loop.md`
**Self-critique (1-line):** Auto-ingestion risks turning the bench into a mirror of BAG's existing strengths, hiding capability gaps the user never thinks to probe.
