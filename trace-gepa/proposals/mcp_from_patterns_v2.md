# Proposal D-v2 — Empirical Verdict on MCP-from-Patterns

**Author:** Deep-Exploration Team Member
**Date:** 2026-05-01
**Status:** **NOT VIABLE — pivoting recommended.** See `DECISION_NEXT_STEP.md`.

This document closes the loop on Brainstorm Member #D's proposal
(`mcp_from_patterns.md`). #D self-flagged that the top-5 list was a
hypothesised prior. We replaced the prior with actual mining and the
hypothesis collapses on three independent metrics.

---

## 1. What we ran

`trace-gepa/extractors/mine_patterns.py` (~140 LoC, stdlib only):

1. Stream `dataset.jsonl` (3 929 records) and `dataset_v2.jsonl`
   (26 384 records).
2. Group by `src_path` (one session per source file).
3. Tokenise each `observed_action`:
   - `Bash:<verb>` (cc) and `Exec:<verb>` (codex), where verb is
     extracted after stripping leading `cd <dir> &&`.
   - Plain tool name otherwise (`Read`, `Edit`, `SendMessage`, ...).
   - Drop bookkeeping noise: `TodoWrite`, `TaskList`, `TaskGet`,
     `TaskUpdate` (so they cannot dominate by repetition).
4. Mine contiguous n-grams of length 3..10 with min-support
   = 10 distinct sessions, then greedily extend top-80 length-3 seeds.
5. Rank by information-density `support * |distinct_tokens| * sqrt(len)`.

Wallclock: **~1 s** for the full corpus on an M3 Max — far below the
10-minute cap. Output: `data/mined_patterns_top30.json`.

## 2. What we found

The corpus is far smaller than headline counts suggest:

| metric | value |
|---|---|
| total records | 30 313 |
| distinct sessions (`src_path`) | **220** |
| of which **main** (user-driven) | **22** |
| of which **subagent** spawns | **198** |
| distinct projects | **7** |
| `ir-expo` share of all events | **~84 %** |

Top mined patterns at min-support = 10:

| # | sup | len | pattern (first 5 tokens) |
|---|----|----|---|
| 1 | 27 | 10 | `Bash:git, SendMessage, Write, Bash:ls, Bash:echo, ...` |
| 2 | 27 | 10 | `Bash:git, Bash:git, SendMessage, Write, Bash:ls, ...` |
| 3 | 27 | 10 | `Bash:zig, Bash:zig, Bash:git, Bash:git, SendMessage, ...` |
| 4 | 27 | 10 | `Bash:echo, Bash:zig, Bash:grep, Read, Write, ...` |
| 5 | 27 | 10 | `Bash:zig, Bash:git, Bash:git, SendMessage, Write, ...` |
| 6 | 27 | 10 | `Bash:zig, Bash:git, Bash:git, SendMessage, Bash:ls, ...` |
| 7 | 27 | 10 | `SendMessage, Write, Bash:ls, Bash:echo, Bash:zig, ...` |
| 8 | 27 |  9 | (sliding window of #7) |
| 9 | 27 |  9 | (sliding window of #1) |
|10 | 27 |  9 | (sliding window of #2) |

(All 30 entries fit this same shape; full list in
`data/mined_patterns_top30.json`.)

## 3. Why this disproves the hypothesis

Three falsifiers, each independently fatal:

**(a) Mono-project.** For every length-≥4 pattern in the top 30, the
distinct-project count is **1** (verified by post-hoc sweep over all
220 sessions). A "personalised MCP tool" must compose across the user's
work; a tool encoding `ir-expo`'s zig-build-and-edit ritual is dead the
moment the user opens any of the other six projects.

**(b) Mono-subagent.** All 27 supporting sessions for the top patterns
are **subagent spawns** of one parent session (`02cc31af-...`). They
share support not because the user repeats this workflow, but because
one subagent template was invoked 27 times. **Main-session support for
every top pattern is 0.**

**(c) Sliding-window degeneracy.** The top-30 list is dominated by
length-8/9/10 windows over a *single* canonical zig+git+SendMessage
sequence. There is no diverse top-5 to compile into 5 tools.

Brainstorm-#D's hypothesised top-5 (lint-then-commit, mlx-bench-run,
czech-correction-loop, gepa-optimize-then-eval, worktree-spinup)
**does not appear** in mining at any length-≥4 with ≥10 sessions.

**Verdict gate:**

> "If ≥ 5 patterns of length 4+ have support ≥ 10 distinct sessions
> AND are workflow-meaningful → VIABLE."

Patterns of length ≥4 with support ≥10: **plenty by raw count** (≈30),
**zero** when "distinct" is enforced at the project level and
"workflow-meaningful" is read as the proposal intended (cross-context,
multi-tool, captures a real user habit). Gate fails on both clauses.

## 4. Why mining was always unlikely to work here

The dataset is a snapshot of one user's recent fortnight on one repo.
Pattern mining needs **N independent sessions with shared structure**;
this corpus has 22 main sessions across 7 wildly different projects
(an Expo app, an MLX experiment, a trading bot, a Zig workshop, ...).
There is no shared workflow surface area large enough to mine.
Brainstorm-#D anticipated this in its honest-critique section but
under-estimated its severity.

A future re-mining at 10× this corpus may unlock real cross-project
patterns. Today it does not.

## 5. Path forward

See `DECISION_NEXT_STEP.md` for the A-vs-C comparison and the chosen
pivot. Short version: pivot to **Proposal C — Persona Fingerprint
(Step 1 + Step 2 only)**, because it is the only proposal whose
training signal *grows* (rather than shrinks) when the corpus is
heavily skewed to one project and one user.

## 6. Files produced

- `trace-gepa/extractors/mine_patterns.py`     — the miner
- `trace-gepa/data/mined_patterns_top30.json`  — empirical evidence
- `trace-gepa/proposals/mcp_from_patterns_v2.md` — this doc
- `trace-gepa/DECISION_NEXT_STEP.md`           — pivot decision
