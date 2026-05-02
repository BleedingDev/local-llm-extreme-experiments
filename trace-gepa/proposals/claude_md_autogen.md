# Trace-Driven CLAUDE.md Auto-Generation

**Round-2 Member #F — NOVEL angle: regenerate `~/.claude/CLAUDE.md` itself from traces.**

## TLDR

- **Hypothesis**: a nightly-regenerated CLAUDE.md grounded in `profile.json` + `recovery_top5` + skill-usage histograms outperforms the hand-written one because it tracks habit drift and cites real frequencies — and unlike the persona prefix (per-call patch), it shapes the whole session at source level.
- **Two artefacts**: global `~/.claude/CLAUDE.md` (always) + per-repo `<repo>/CLAUDE.md` (only when ≥ 50 trace records exist for that repo, so sections are statistically grounded).
- **Generator** is ~150 LOC of Python: read `profile.json`, bin verbs/paths/languages/recoveries/Czech-stop-phrases/failure-categories, render seven Markdown sections from a fixed template; preserve a `<!-- HAND-EDITED -->` block so user overrides survive regeneration.
- **Cadence**: nightly cron (via `cron`/`schedule` skill) at 03:00, diff against prior version, write `.claude/CLAUDE.md.prev` so user can `git diff` what changed about themselves.

## Sample 8-line excerpt (concrete, what the generator emits)

```
## Tools you reach for first (last 30d, n=1,284)
1. Bash:git (31%)  2. Read (24%)  3. Bash:rg (11%)  4. Bash:zig (9%)  5. Edit (8%)

## Recovery reflex
When a Bash `cat`/`head` fails or returns truncated output, switch to Read (observed 47x).

## Czech corrective phrases that mean STOP
"ne", "pockej", "spatne", "jinak" — when seen, halt current plan and re-read the last user turn.
```

## Generation algorithm

1. Load `trace-gepa/data/profile.json`.
2. Sections: (a) tool histogram top-5, (b) path histogram top-5 with repo names, (c) language signals (Zig/TS/Python file-extension counts), (d) mined workflow n-grams (e.g. `Read → Edit → Bash:zig build`), (e) recovery pairs from `recovery_top5`, (f) Czech stop-phrase list with counts, (g) failure-category antipatterns ("avoid `cat` for >200-line files: 23 truncations observed").
3. Per-repo file only if `repo_record_count[r] ≥ 50`; otherwise fold into global.
4. Stamp header: `<!-- AUTO-GENERATED $(date) from N traces. Hand edits below the marker are preserved. -->`.

## Effort + ROI

~3 hrs to write generator, ~30 min to wire cron. ROI: removes the "stale CLAUDE.md" failure mode (user changed pnpm→bun six months ago, file still says pnpm) and makes the primitive that already shapes every session empirical rather than aspirational.

## Path
`/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/proposals/claude_md_autogen.md`

## Self-critique
Auto-regeneration risks ossifying current habits into normative rules — a bad week of `cat`-overuse would be canonised into the file the next morning unless we add a confidence/recency floor.
