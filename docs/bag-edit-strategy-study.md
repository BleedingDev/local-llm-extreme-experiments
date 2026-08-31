# BAG edit-strategy x model stability study

**Status:** Phase 1 (scaffolding + tests) shipped. Phases 2-4 (Opus / Sonnet /
Haiku sweeps) pending — runs are gated on Anthropic API spend approval and on
the parallel harness-ablation study (worktree `agent-a4cb9362e4160c08a`)
clearing its concurrent budget. This document is the runbook + the table the
operator will fill in once the sweep completes.

---

## Hypothesis

Each model has a preferred file-editing modality. Today BAG agents edit files
exclusively via shell primitives (`cat <<'EOF'`, `sed -i`, `printf >>`). The
forensic Aider-Polyglot run showed BAG made **17 terminal_create calls and 0
`fs/write_text_file` calls** across all wins — we have never measured whether
shell-edit is actually optimal per-model. Mainstream coding agents diverge:

| Agent       | Edit modality                                    |
| ----------- | ------------------------------------------------ |
| BAG (today) | shell here-docs / sed / printf via `bash`        |
| Claude Code | structured `Edit` tool (`old_string`/`new_string`) |
| Codex       | `apply_patch` (unified diff)                     |
| Pi-mono     | `edit-diff` blocks (line-range mutations)        |
| ACP fs role | `fs/write_text_file` (full file body)            |

The hypothesis is that the ergonomics of the chosen tool interact with the
model's training distribution: a model trained heavily on diff-format outputs
may prefer `apply-patch-unified`, whereas a model that has been RL'd against
shell traces (like Claude Code's training set) may prefer
`edit-tool-stringreplace`. We need data.

---

## Method

### Strategy registry — `src/edit-strategies/registry.ts`

Five strategies, all sharing the `EditStrategy` interface. The autonomous
coding turn queries `strategy.toolDefinitions()` and adds those to the model's
tool surface alongside `bash` / `view_image` / `code_search`. When the model
calls one of the strategy's tools, the dispatcher writes to disk and emits an
`edit_dispatch` trace entry.

| Id                          | Tool exposed             | Outcome semantics                                              |
| --------------------------- | ------------------------ | -------------------------------------------------------------- |
| `shell-heredoc`             | (none — bash only)       | always reports `delegated_to_bash` (the current default)       |
| `fs-write-whole-file`       | `fs_write_text_file`     | overwrites file with full body                                 |
| `edit-tool-stringreplace`   | `edit`                   | literal `old_string` -> `new_string`; ambiguity is `match_failed` |
| `apply-patch-unified`       | `apply_patch`            | unified diff with `--- a/+++ b/@@` headers                     |
| `edit-diff-blocks`          | `edit_diff_block`        | 1-indexed inclusive line-range replace; `expected_old_block` opt-in |

Every strategy emits an `edit_dispatch` telemetry row with shape:

```json
{
  "kind": "edit_dispatch",
  "at": "2026-05-02T10:00:00.000Z",
  "strategy": "edit-tool-stringreplace",
  "tool": "edit",
  "target": "src/foo.ts",
  "outcome": "applied|match_failed|stale_context|syntax_error|permission_denied|delegated_to_bash",
  "bytes_changed": 42,
  "retries_within_strategy": 0
}
```

### Selection — `BAG_EDIT_STRATEGY` env var

`src/harness-gates.ts::loadHarnessGates()` reads the env var. Default is
`shell-heredoc` so existing call sites are byte-equivalent. Unknown values
fall back to `shell-heredoc` with a `console.warn`. The harbor adapter
`bench/bag_agent/agent.py` accepts `bag_edit_strategy` via `--agent-kwarg` and
forwards it into the container env.

This env name does NOT collide with the parallel harness-ablation study
(`BAG_GATE_*`, `BAG_TOOL_*`).

### Wiring — `src/autonomous-coding-turn.ts`

```text
gates = loadHarnessGates();
editStrategy = createEditStrategy(gates.editStrategy);
tools = [BASH, ...optional view_image/code_search, ...editStrategy.toolDefinitions()];
systemPrompt += editStrategy.systemPromptFragment();   // only when non-default
```

When the model emits a tool call whose name matches a strategy tool, the loop
dispatches via `editStrategy.dispatch(...)`. The strategy's `emit` callback
pushes the `edit_dispatch` trace entry. The model's perceived contract is
unchanged for `shell-heredoc` (default).

### Sweep — `bench/edit_strategy_study/run.sh`

5 strategies x 3 models x 10 tasks (TB-sample) = **150 trials**. Phased to
keep the spend visible:

- **Phase 1** ship the registry + tests (this commit).
- **Phase 2** Opus across all 5 strategies (5 cells, ~25 min each = ~2 h).
- **Phase 3** Sonnet across all 5 strategies (5 cells, ~2 h).
- **Phase 4** Haiku across all 5 strategies (5 cells, ~1.5 h — Haiku is faster).

Cell naming: `edit_<strategy_short>_<model_short>` so the harbor jobs
directory has stable names the aggregator can scan.

Strategy shorts: `shell` `fswrite` `strrepl` `apatch` `diffblk`.
Model shorts: `opus` `sonnet` `haiku`.

### Aggregator — `bench/edit_strategy_study/aggregate.py`

Walks `bench/jobs/edit_*` directories, reads `result.json` / `verdict.json`
for pass/fail, walks `autonomous-trace.json{,l}` for `edit_dispatch` rows.
Emits:

- `matrix.json` — 5x3 dict-of-dicts with pass counts.
- `per_cell.json` — full per-cell stats (mean turns, edit-outcome histogram, per-task wins/losses).
- `summary.txt` — human-readable matrix + per-cell breakdown.

Missing cells render as `n/a` in the matrix.

---

## Results

### Pass-count matrix (TO FILL after Phase 4)

```text
                        opus       sonnet     haiku
shell-heredoc           __ / 10    __ / 10    __ / 10
fs-write-whole-file     __ / 10    __ / 10    __ / 10
edit-tool-stringreplace __ / 10    __ / 10    __ / 10
apply-patch-unified     __ / 10    __ / 10    __ / 10
edit-diff-blocks        __ / 10    __ / 10    __ / 10
```

### Per-cell statistics (TO FILL)

| Cell                             | pass / N | mean turns | edit-error rate | applied / total dispatches |
| -------------------------------- | -------- | ---------- | --------------- | -------------------------- |
| edit_shell_opus                  | __ / 10  | __         | __ %            | __ / __                    |
| edit_fswrite_opus                | __ / 10  | __         | __ %            | __ / __                    |
| edit_strrepl_opus                | __ / 10  | __         | __ %            | __ / __                    |
| edit_apatch_opus                 | __ / 10  | __         | __ %            | __ / __                    |
| edit_diffblk_opus                | __ / 10  | __         | __ %            | __ / __                    |
| edit_shell_sonnet                | __ / 10  | __         | __ %            | __ / __                    |
| edit_fswrite_sonnet              | __ / 10  | __         | __ %            | __ / __                    |
| edit_strrepl_sonnet              | __ / 10  | __         | __ %            | __ / __                    |
| edit_apatch_sonnet               | __ / 10  | __         | __ %            | __ / __                    |
| edit_diffblk_sonnet              | __ / 10  | __         | __ %            | __ / __                    |
| edit_shell_haiku                 | __ / 10  | __         | __ %            | __ / __                    |
| edit_fswrite_haiku               | __ / 10  | __         | __ %            | __ / __                    |
| edit_strrepl_haiku               | __ / 10  | __         | __ %            | __ / __                    |
| edit_apatch_haiku                | __ / 10  | __         | __ %            | __ / __                    |
| edit_diffblk_haiku               | __ / 10  | __         | __ %            | __ / __                    |

### Per-model recommendation (TO FILL)

The decision rule is "the strategy with the highest pass count wins; on ties,
prefer lower edit-error rate; on further ties, prefer fewer mean turns".

- **Opus master role:** `<TBD>` — ship as default for opus.
- **Sonnet master role:** `<TBD>` — ship as default for sonnet.
- **Haiku master role:** `<TBD>` — ship as default for haiku.

When the table lands, `bench/bag_agent/agent.py` should set
`bag_edit_strategy` per-model based on `master_model`; until then the global
default `shell-heredoc` stands.

---

## Cost & limitations

- **N=10** per cell. Pass counts at this scale carry +/- 2 task variance per
  random seed; differences smaller than 2 should not drive the recommendation.
- Estimated tokens: TB-sample averages ~30k input + ~6k output per task on
  Opus, ~half that on Haiku. Full sweep ~150 trials -> ~7M total tokens
  worst case. At Opus pricing this is a few hundred dollars; bring receipts.
- The `apply-patch-unified` strategy uses a deliberately tolerant matcher
  (scans +/- 50 lines around the reported `oldStart`) so it doesn't
  artificially penalize models that emit slightly stale line numbers.
- `edit-diff-blocks` is the most under-spec'd of the five — the public
  Pi-mono spec is sparse. I implemented the 1-indexed-inclusive line-range
  variant with an optional `expected_old_block` stale-context guard.
- Strategies are CONCEPTUALLY borrowed from Claude Code / Codex / Pi /
  mini-swe-agent; no code is copied.
- Single seed per cell. A future iteration should bump to 3 seeds * 10 tasks
  per cell to get standard-error bars.

## How to run

```bash
# Phase 2 — Opus across all 5 strategies, sequentially
bench/edit_strategy_study/run.sh --phase opus

# Phase 3 — Sonnet
bench/edit_strategy_study/run.sh --phase sonnet

# Phase 4 — Haiku
bench/edit_strategy_study/run.sh --phase haiku

# Aggregate
python3 bench/edit_strategy_study/aggregate.py --root bench/jobs --pattern 'edit_*'

# Read the matrix
cat bench/edit_strategy_study/results/summary.txt
```

## Where the code lives

| Concern                    | Path                                                              |
| -------------------------- | ----------------------------------------------------------------- |
| Strategy registry          | `src/edit-strategies/registry.ts`                                 |
| Env-var resolution         | `src/harness-gates.ts::loadHarnessGates`                          |
| Tool dispatch              | `src/autonomous-coding-turn.ts` (look for `editStrategy`)         |
| Telemetry trace entry      | `src/autonomous-coding-turn.ts` `kind: "edit_dispatch"`           |
| Container env propagation  | `bench/bag_agent/agent.py` (`bag_edit_strategy` kwarg + env push) |
| Sweep driver               | `bench/edit_strategy_study/run.sh`                                |
| Aggregator                 | `bench/edit_strategy_study/aggregate.py`                          |
| Tests                      | `tests/edit-strategies/registry.test.ts` `selection.test.ts`      |
