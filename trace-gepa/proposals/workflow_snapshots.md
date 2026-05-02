# Workflow Snapshots: Reproducible Session State for Replay

**Author:** Brainstorm Round-5 Member #S
**Date:** 2026-05-01

## TLDR

- Capture full session start-state (prompt + git HEAD + dirty diff + referenced-file hashes + env whitelist + `.claude.json` excerpt) as a single compressed archive, indexed in sqlite.
- Replay tooling (`bag-replay`) rebuilds the workspace in a tempdir and re-runs ANY agent/model/prompt against the SAME starting conditions — enabling true A/B evaluation.
- Storage is cheap: ~5-50 MB per snapshot zstd-compressed; 1 GB holds ~100 sessions, plenty for personal benchmarking.
- Distinct from MCP-extraction (tools) and trace dataset (actions); this is the missing **starting-state** layer that makes everything else controlled-experiment grade.

## Hypothesis

Most agent evaluation today re-runs prompts against drifting workspaces: yesterday's repo had different files, different uncommitted changes, different env. We can't answer "would Opus do better than Sonnet on THIS task?" because the task substrate moves under us. Reproducibility is foundational; without it GEPA optimization, regression testing, and bug repro are all noisy.

## Design

**Trigger.** Two modes:
1. Explicit `/snapshot` slash command — user marks an interesting moment.
2. Auto-capture on every Claude Code "first user message" event via a `UserPromptSubmit` hook (configured in `settings.json`). Cheap; bound by retention policy.

**Capture payload (per snapshot id `<sid>`):**
- `prompt.txt` — verbatim user request.
- `git.head` — commit SHA + branch.
- `git.diff` — `git diff HEAD` (dirty working tree) + untracked file list.
- `files.json` — for every path mentioned in prompt or referenced in first 10 tool calls: `{path, sha256, size, mode}` plus full content if <1 MB.
- `env.json` — whitelist (`PATH`, `NODE_VERSION`, `PYTHON_VERSION`, project-specific vars declared in `.bagrc`).
- `claude.excerpt.json` — model, MCP servers, permissions snapshot from `~/.claude.json`.
- `meta.json` — timestamp, cwd, claude-code version, OS.

**Storage.** `~/.claude/snapshots/<sid>.tar.zst`. Sqlite index at `~/.claude/snapshots/index.db` with columns `(sid, ts, cwd, repo, head_sha, prompt_excerpt, size_bytes, tags)` for queryable lookup (`bag-snap ls --repo foo --since 7d`).

**Replay.** `bag-replay --snapshot <sid> [--model <m>] [--prompt <p>] [--agent <a>]`:
1. Extract archive into `$(mktemp -d)/replay-<sid>`.
2. `git checkout <head_sha>`, apply `git.diff`, restore untracked files from `files.json`.
3. Export whitelisted env.
4. Launch agent with original (or overridden) prompt; capture new trace.
5. Diff new trace against original — file changes, tool calls, final assistant message — produce `replay-report.json`.

## Use Cases

- **GEPA optim eval.** Score candidate prompts against a fixed snapshot suite, not a live drifting repo.
- **BAG regression testing.** When the harness ships a new version, replay the last 50 snapshots and flag behavior deltas.
- **Bug-report reproducer.** "Agent looped on this snapshot" — ship the `.tar.zst`, maintainer replays exactly.

## Storage Budget

Typical snapshot:
- prompt + meta + index: ~5 KB
- git diff (most dirty trees): ~50 KB - 2 MB
- referenced files: ~500 KB - 20 MB (capped at 1 MB/file, ~50 files max)
- zstd level 19 compression: ~3x ratio

Median ~8 MB, p95 ~40 MB. 1 GB cap with LRU eviction holds 100+ snapshots — sufficient for personal benchmark suites.

## Effort + ROI

- **Hook + capture:** 1 day (shell script reading git + sqlite insert).
- **Replay harness:** 2 days (tempdir checkout, env injection, trace diff).
- **CLI polish + sqlite query layer:** 1 day.
- **Total:** ~4 days for a working v1.

ROI is multiplicative: every other evaluation idea (router A/B, GEPA scoring, model comparisons) becomes rigorous instead of anecdotal. The trace dataset gains ground-truth starting states.

## Self-Critique

Snapshots may leak secrets via dirty diff or untracked files; mitigate with a `.bagignore` and an opt-in scrub pass, but the failure mode is real and users will footgun themselves at least once.
