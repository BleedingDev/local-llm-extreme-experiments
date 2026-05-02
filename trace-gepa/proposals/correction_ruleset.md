# Correction Ruleset — Brainstorm Round-2 Member #I

## TLDR
- Most user corrections distil to a single declarative rule; aggregating them yields a high-precision, low-volume personal style guide that any agent can load at session start, no GEPA/benchmark required.
- Walk `dataset_corrections.jsonl` (the user-corrected subset, ~67 records in v2). For each (failed_action, user_correction) pair, prompt Opus once: *"Write a 1-sentence imperative rule that prevents recurrence."*
- Deduplicate + merge near-duplicates ("use pnpm" + "don't use bun" → "Use pnpm, never bun"). Emit `~/.claude/derived_rules.md` (or `trace-gepa/artifacts/derived_rules.md`) as a bullet list, each rule annotated with its source-trace ID and timestamp.
- Refresh trigger: re-run extraction whenever ≥10 new user-correction records accumulate (cheap — ~67 Opus calls one-shot, ~10 incremental).

## Hypothesised Example Rules
1. Use `pnpm` for all Node package operations; never invoke `bun` or `npm install`. *(src: trace #0142)*
2. Prefer `rg` over `grep`/`find` for codebase search unless the user specifies otherwise. *(src: trace #0188)*
3. Never run `git push --force` to `main`/`master` without explicit confirmation in the same turn. *(src: trace #0203)*
4. When editing Python, use absolute imports rooted at the package; do not introduce relative `..` imports. *(src: trace #0091, #0156)*
5. Do not create `*.md` summary/report files unless the user explicitly asks; return findings inline. *(src: trace #0044, #0211, #0233)*

## Pipeline (no code, just spec)
1. **Extract** — single Opus pass per correction, structured output `{rule, scope, source_trace}`.
2. **Dedupe** — substring + cosine on rule text (cheap embedding) → cluster.
3. **Merge** — Opus pass per cluster collapses to one canonical rule, preserves all source IDs.
4. **Sanity review** — final Opus pass on the full list checks for: contradictions, overly-specific rules (e.g. references one filename), and stale rules (superseded by later correction).
5. **Emit** — sorted by frequency desc, written atomically.

## Path
`trace-gepa/artifacts/derived_rules.md` (project-scoped) with optional symlink/copy to `~/.claude/derived_rules.md` (global). Project-scoped is safer default — global risks cross-project bleed.

## Effort & ROI
~2 hrs to implement; ~$0.50 in Opus calls per full rebuild. ROI is high *if* the agent harness actually loads the file at session start — otherwise it's a dead artifact. Pair with a CLAUDE.md `@import derived_rules.md` directive to guarantee load.

## Self-Critique
Risk: rules extracted from one-off corrections may overfit ("user was tired that day"); mitigation is a frequency threshold (rule appears ≥2× before promotion) but that delays the highest-signal corrections — a tunable knob, not a solved problem.
