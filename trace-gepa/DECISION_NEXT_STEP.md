# DECISION — Next Step After Mining Proposal D

**Date:** 2026-05-01
**Author:** Deep-Exploration Team Member
**Verdict on Proposal D (MCP-from-patterns):** **NOT VIABLE.**
**Chosen pivot:** **Proposal C — Persona Fingerprint, Steps 1+2.**
**Rejected pivot:** Proposal A — Behavioral Cloning (LoRA SFT).

Full empirical evidence: `proposals/mcp_from_patterns_v2.md`.
Mining script: `extractors/mine_patterns.py`.
Mined data: `data/mined_patterns_top30.json`.

---

## 1. Verdict (one paragraph)

After running real frequent-subsequence mining on the full
30 313-record corpus, every length-≥4 pattern with support ≥10
sessions appears in **exactly one project** and **zero main user
sessions** — all 27 supporting sessions for each top pattern are
subagent spawns from a single `ir-expo` parent session. The corpus
contains only 22 main user sessions spread across 7 unrelated
projects, with 84 % of events from one repo. There is no
cross-context workflow surface to compile into MCP tools.
Brainstorm-#D's hypothesised top-5 (lint-then-commit, mlx-bench, ...)
does not appear in mining at all. Proposal D fails its own viability
gate ("≥ 5 length-4+ patterns at ≥ 10 distinct sessions, workflow-
meaningful") on both the distinct-session clause and the
meaningful clause.

## 2. A vs C — head-to-head

|                                    | A — Behavioral Cloning (LoRA SFT) | C — Persona Fingerprint (Steps 1+2) |
|---|---|---|
| Required corpus size               | borderline-tiny at 645–2 k examples (#A's own concern) | works at *any* size; smaller = sharper fingerprint |
| Mono-project bias hurts            | **yes** — LoRA will memorise `ir-expo` paths and regress on synth eval (#A flags this) | **no — bias is the product**; "user works in `ir-expo`" is a feature, not a leak |
| Effort                             | ~6 h human + ~6 h Mac GPU (1 day) | Step 1: ~1 day extraction; Step 2: ½ day prompt wiring (1.5 days total) |
| Evaluable today                    | yes — `bench/run_mlx.py` 175-task harness exists, +5 pts vs control is a clean stop-rule | partial — `persona_profile.json` is inspectable, but "feels like me" is the only hard test |
| Failure mode if it doesn't work    | sunk Mac-day; negative result is diagnostic | sunk 1.5 days; JSON artefact is still useful as corpus summary |
| Reuses today's empirical work      | no | **yes** — the n-gram counts, project priors, and tool histograms produced by `mine_patterns.py` are exactly the Step-1 inputs |
| Differentiation                    | a tuned 3 B model (other labs ship better ones every quarter) | "no one else has these traces" — uniquely user-specific |
| Risk of over-fit to surface tics   | high (#A: paths under `/Users/satan/...`) | high (#C: Czech tokens, paths) — but here over-fit *is* the goal |

Both A and C are vulnerable to the same skew (1 project / 1 user)
that killed D. The crucial difference: **for A, the skew is a bug
to mitigate; for C, the skew is the spec.** That asymmetry is decisive
given the corpus we actually have.

## 3. Choice: **Proposal C, Steps 1+2 only.**

**Rationale.**

1. **Cheapest spike that pays a tangible artefact.** Step 1 produces
   `persona_profile.json` (≤ 50 KB) and Step 2 wires it as a system-
   prompt prefix. Total ~1.5 days. No GPU time. No new infra.

2. **Reuses the work we just did.** The miner already produced tool
   histograms, project priors, and bash-verb co-occurrence counts.
   Those are 60 % of the Step-1 deliverable; we add Czech-trigger
   n-grams and error→recovery bigrams on top.

3. **Robust to the corpus shape that broke Proposal D.** The single
   dominant project is *signal* for a persona ("user spends 80 % of
   logged time in `ir-expo`"), whereas D needed the opposite shape.

4. **Preserves the option to do A later.** A persona prefix slots
   in front of *any* base model — including a future SFT-LoRA. Doing
   C first does not foreclose A; doing A first burns a Mac-day
   before we know whether a prompt prefix would have sufficed.

5. **Defer Step 3 (LoRA).** #C themselves recommend gating Step 3
   on whether Step 2 already feels right. Same here.

## 4. Concrete next actions (Wave 4)

- [ ] Owner C-Step-1: stream both datasets, extract:
      - Top-50 tools by frequency, by project.
      - Top-50 bash verbs + 2-gram bash chains.
      - Path-prefix histogram (`~/side/experiments/<repo>` distribution).
      - Czech-corrective n-grams (regex over user turns: `nene|počkej|spíš|lepší`).
      - Error → next-action bigrams (filter `result_is_error == true`).
      Emit `data/persona_profile.json`.
- [ ] Owner C-Step-2: produce a 1–2 KB system-prompt prefix that
      summarises the profile in natural language. A/B against the
      current GEPA-optimised prompt on `bench/run_mlx.py`.
- [ ] Decision gate after Step 2: if A/B shows neutral-or-better,
      consider Step 3. If worse, the persona thesis is falsified at
      this corpus scale (parallel to #A's stop-rule).

## 5. What we will NOT do

- **No** MCP server scaffolding from mined patterns. The patterns
  do not exist at meaningful diversity.
- **No** LoRA training before Step 2 has been A/B-tested.
- **No** changes to other proposals. Owners of A and C are notified
  via this doc; the call is theirs to take or refuse.
