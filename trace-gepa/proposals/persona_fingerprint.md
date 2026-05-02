# Proposal C — Persona Fingerprint: An Agent That Codes *Like You*

**Author:** Brainstorm Team Member #C
**Status:** Novel proposal — orthogonal to all prior waves (which targeted *generic* agent quality).

---

## 1. Hypothesis

26 GB of one user's traces is not just a training corpus — it is a **behavioural fingerprint**. A small model fit on this corpus alone will not be a better generic agent; it will be a *specific* agent: one that reaches for `bun` before `pnpm`, scaffolds Schaltwerk worktrees, hits MLX before CUDA, drops into Czech ("Nene", "počkej") to course-correct, and lays files under `/Users/satan/side/experiments/...`. Such a model can act as a **ghost-self** — predicting how *this* user would respond to any prompt. Applications: smart autocomplete that finishes your *next move* (not just your token); pair-programming with a mirror; explainability ("would I have done this?"); offline delegation ("act as me on this PR").

No one else can build this. That is the entire point.

## 2. Concrete Proposal

Build `persona_profile.json` — a dense fingerprint summary — and (optionally) a LoRA adapter that biases a base agent toward this persona.

**Signals to extract from the trace corpus:**

| Signal | Source | Form |
|---|---|---|
| Tool preference | `Bash` calls, tool histograms | Top-k commands + flag co-occurrence (e.g. `bun run` >> `pnpm run`) |
| Language code-switches | Assistant ↔ user turns | Czech corrective n-grams ("nene", "počkej", "ne tak") + context window before each |
| Path priors | File paths touched | Distribution over `~/side/experiments/*`, `~/.claude/*`, `Schaltwerk/*` |
| Failure-recovery flows | Error → next-N-actions sequences | Markov-style templates ("MLX OOM" → "reduce batch" → "rerun") |
| Active hours | Timestamps | Diurnal histogram + session-length distribution |
| Stylistic micro-habits | AST/diff features | Naming, comment frequency, tab-vs-space, import order, test-first vs test-after |

## 3. Implementation Outline

- **Step 1 — Heuristic fingerprint (no training, ~1 day):** Stream traces; extract n-grams, tool histograms, path priors, error→recovery bigrams. Emit `persona_profile.json` (≤ 50 KB). Already useful as a system-prompt prefix.
- **Step 2 — Prompt conditioning (~½ day):** Prepend the fingerprint to any base agent: `"You are mimicking user 'satan': prefers Bun, MLX over CUDA, Schaltwerk worktrees, replies in English but corrects with Czech ('nene', 'počkej'), works under /Users/satan/side/experiments/. Match tool-choice priors below: …"`. Cheap, reversible, surprisingly effective.
- **Step 3 — LoRA adapter (optional, ~3–5 days):** Fine-tune a small base (Qwen-2.5-Coder-7B or Gemma-2-9B) on a curated `dataset_corrections` subset where the user *overrode* the agent. These are the highest-signal "this is how I actually wanted it done" examples. MLX-LoRA fits on the M-series box already in use.

## 4. Privacy / Ethics

A persona model trained on personal traces is a **dossier**. It knows working hours, repo layout, native language, failure modes. Mitigations: (a) keep the artefact local-only, (b) never publish the LoRA weights, (c) redact secrets/tokens before training (the corpus *will* contain them), (d) include a kill-switch — `persona_profile.json` deletion is sufficient to disable Step 2. This must be opt-in even for the user themselves; future-them may not consent to past-them being modelled.

## 5. ROI & Honest Critique

**Why this is THE differentiated bet:** Every other proposal in this brainstorm improves a *generic* agent — work that better-funded labs will do faster. Only **you** have your traces. A persona model is the one artefact that is structurally impossible for anyone else to build. Even Step 1 alone (heuristic JSON + prompt prefix) is a high-leverage week of work.

**Honest critique:** (1) The user's habits drift; the fingerprint will stale within months and needs continuous re-fitting. (2) "Acting as the user" is a *narrower* objective than "being a good agent" — a perfect persona model can faithfully reproduce the user's *bad* habits (e.g. skipping tests). (3) Evaluation is hard: there is no held-out "ground truth user" — the only judge is the user themselves, which is noisy. (4) Risk of over-fitting to surface tics (Czech tokens, path strings) while missing the deeper reasoning style — the fingerprint may feel uncanny rather than useful.

Recommend shipping Step 1 + Step 2 first; gate Step 3 on whether Step 2 already feels like *you*.
