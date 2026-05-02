# Cross-Agent Prompt Transferability Study

**Author:** Brainstorm #B  **Status:** Proposal  **Owner files:** this doc only

## 1. Hypothesis

Our GEPA-optimised prompt was tuned for **BAG**. We currently treat its lift as a property of the prompt, but it might be a property of BAG's planner/executor topology. Transferability is an *empirical* question with three plausible outcomes:

- **(a) Universal-ish** — prompt encodes generally good coding-agent behaviour (decompose, verify, cite paths) and lifts pass-rate on Aider, Cursor, Claude Code too. Strongest contribution.
- **(b) BAG-specific** — prompt exploits BAG's loop structure (e.g. assumes a re-plan step). Doesn't transfer, may even *hurt* one-shot agents. Still publishable: we've isolated which gains are scaffold-coupled vs. prompt-coupled.
- **(c) Selective transfer** — works on planner/executor agents (BAG, OpenHands) but not on edit-in-place agents (Aider). Predicts a typology of agents by prompt-receptiveness.

Outcome (b) and (c) are arguably more interesting than (a): a *negative* result on Cursor would be the first principled claim that agent-tuned prompts don't generalise across scaffolds.

## 2. Proposal

**Targets (3):**
1. **Aider** — lightest. Prompt injection via `--read SYSTEM.md` or `.aider.conf.yml` `read:`. ~1 day.
2. **Claude Code** — `CLAUDE.md` is the documented injection point; we already produce traces from it, so this is the cleanest within-distribution test. ~1 day.
3. **Cursor** — heaviest, biggest ecosystem payoff. `Rules for AI` / `.cursorrules` injection. ~2 days (more agent surface, harder to script headlessly).

(Deferring OpenHands to a v2 — its config surface is larger and it overlaps conceptually with BAG.)

**Methodology:**
- Same 175-task bench used for BAG eval.
- Two arms per agent: **seed** prompt vs. **GEPA-optimised** prompt.
- Metric: pass rate (primary), tokens-to-solution (secondary).
- Paired bootstrap CIs over tasks; report Δ pass-rate per agent.

**Confound control (critical):**
- All agents must call the **same underlying LM** (claude-opus-4-7 via API) so we isolate prompt × scaffold from LM identity.
- Same temperature, same max-turns budget, same task timeout.
- Same eval harness (existing bench grader) — agents differ only in their orchestration loop and prompt-injection point.

## 3. Implementation steps (design only)

1. Map each agent's system-prompt injection point (Aider `--read`, Cursor rules file, Claude Code `CLAUDE.md`).
2. Build thin per-agent runner: `bench/run_aider.py`, `bench/run_cursor.py`, `bench/run_cc.py`. Each takes `(task, prompt_path) -> {pass, turns, tokens}`.
3. Pin LM via each agent's API-key/model config; verify identical model id in logs.
4. Run 2 arms × 3 agents × 175 tasks = 1,050 runs. Add BAG arm (already have) for the within-scaffold reference.
5. Report a 4-row table: agent × {seed, optimised, Δ, p-value}.

## 4. Effort estimate

- Aider runner: ~1 day. Claude Code runner: ~1 day. Cursor runner: ~2 days (headless is the hard part).
- Eval cost: ~1,050 task-runs × Opus pricing. Order-of-magnitude same as one BAG eval sweep × 3.
- Analysis + writeup: 1 day.
- **Total: ~1 person-week + eval budget.**

## 5. ROI / honest critique

**Why worth doing:** the BAG-only result is a point estimate; one extra week converts it into a *transferability claim*, which is what reviewers/users will ask. Even outcome (b) is a contribution — it bounds where prompt-optimisation gains accrue.

**Honest critiques:**
- Cursor headless integration may slip and dominate the budget; descope to Aider + Claude Code if so (still a valid 2-agent study).
- "Same LM" doesn't fully control for tool definitions — each agent exposes different tools, so prompts that reference BAG-specific tool names will artificially under-perform elsewhere. **Mitigation:** also report a "neutralised" prompt variant with tool names abstracted.
- Negative result risk is real; pre-register the analysis to avoid post-hoc spin.

---

**TLDR:** test our optimised prompt on Aider, Claude Code, and Cursor, same LM, same 175 tasks; one of three outcomes, all publishable.
