# BAG harness ablation study — does the harness pay its weight per model tier?

## Hypothesis

After 22 BAG agents and ~37 benchmark runs we noticed a worrying signal: **Opus
4.7 driving Claude Code (no harness) reportedly hits 9/10 on the
terminal-bench-sample suite, while BAG with the same model averages ~7.5/10**.
If the gap is real, BAG's harness — probe extractor, self-check auditor,
workspace snapshot/restore, view_image / code_search tools, Best-of-N retry,
failure-cluster matcher — is _hurting_ a strong model rather than helping it.
Each LLM gate consumes attention budget (extra system prompt rules, extra
auditor calls that derail the master conversation), which a strong model may
already not need.

But weaker models (Sonnet 4.6, Haiku 4.6) might still benefit from the gates.
That's exactly the kind of asymmetric utility we expect from training-data-
driven scaffolding: it raises a low ceiling more than it lowers a high one.

We measure this directly.

## Study design

3 BAG modes × 3 models × 10 terminal-bench-sample tasks = 9 cells × 10 trials
= 90 total trials.

| Mode          | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `BAG-full`    | Current state. All gates ON.                                    |
| `BAG-bare`    | Gates OFF, multi-tool ON (bash + view_image + code_search).     |
| `BAG-minimal` | Single-tool only (bash). No gates, no rich tooling.             |

| Model | Anthropic id (resolved by harness driver)            |
| ----- | ---------------------------------------------------- |
| Opus  | `claude-opus-4-7`                                    |
| Sonnet| `claude-sonnet-4-6` → `claude-sonnet-4-5-20250929`   |
| Haiku | `claude-haiku-4-6`  → `claude-haiku-4-5-20251001`    |

## Implementation

### 1. Central env-gating module (`src/harness-gates.ts`)

Every gate is independently controllable via a `BAG_*=0` env var:

| Gate / tool         | Env var                       | When `=0`                                |
| ------------------- | ----------------------------- | ---------------------------------------- |
| `probeExtractor`    | `BAG_GATE_PROBE_EXTRACTOR`    | `buildVerifierFromInstruction` returns undefined |
| `selfCheck`         | `BAG_GATE_SELF_CHECK`         | `runSelfCheckGate` accepts unconditionally |
| `snapshotRestore`   | `BAG_GATE_SNAPSHOT_RESTORE`   | No find-snapshot/comm-restore around probes |
| `viewImage`         | `BAG_TOOL_VIEW_IMAGE`         | `view_image` removed from tool list      |
| `codeSearch`        | `BAG_TOOL_CODE_SEARCH`        | `code_search` removed from tool list (legacy alias `BAG_CODE_SEARCH=0` still works) |
| `retryPath`         | `BAG_GATE_RETRY`              | First verifier failure ends the attempt — no Best-of-N |
| `clusterMatcher`    | `BAG_GATE_CLUSTER_MATCHER`    | No cluster hint injection on retry       |
| (categorical) `editStrategy` | `BAG_EDIT_STRATEGY=...`         | switches the structured-edit study (separate track) |

Two presets ship for the ablation harness:
- `BAG_MODE_BARE_ENV` — every gate off, multi-tool on.
- `BAG_MODE_MINIMAL_ENV` — every gate AND non-bash tool off.

The presets are pure data (`Record<string, string>`) so the bash driver can
emit them inline. Defaults are ON, so existing call-sites that supply no
gates argument get byte-equivalent behavior to pre-ablation BAG.

### 2. Gate wiring

- `src/instruction-verifier.ts::buildVerifierFromInstruction`:
  - `probeExtractor=false` → returns `undefined` immediately. No LLM call.
  - `snapshotRestore=false` → skips `captureWorkspaceSnapshot` /
    `restoreWorkspaceFromSnapshot` calls but probes still run.
- `src/autonomous-coding-turn.ts::runAutonomousCodingTurn`:
  - `viewImage=false` / `codeSearch=false` → tool definition removed from the
    `tools` array AND from the unknown-tool-name guard.
  - `selfCheck=false` → `runSelfCheckGate` returns `true` without emitting a
    `pre_submit_self_check` trace entry.
  - `retryPath=false` → caps `totalAllowedAttempts` at 1.
  - `clusterMatcher=false` → skips the `getFailureClusters()` lookup; the
    curated `verifier-signature-library` may still fire.

All conditions are clean pass-through skips — no `if (gate.x) { complex_alt_path }`
branches.

### 3. Tests

- `tests/harness-gates.test.ts` — 9 tests covering env-var parsing, defaults,
  per-field isolation, legacy alias compat, and the `BARE`/`MINIMAL` presets.
- `tests/instruction-verifier-gating.test.ts` — 4 tests covering verifier
  return when `probeExtractor=false`, snapshot fire/skip semantics, and probe
  pass-through when only snapshot/restore is gated off.
- Full `bun test tests/`: **545 pass / 0 fail / 0 regression**. (The other
  failures in `bun test` come from `bench/vendor/polyglot-benchmark/` —
  unrelated exercise stubs that pre-date this change.)

### 4. Harness driver (`bench/ablation/run_ablation.sh`)

Drives one cell per (mode, model) pair. Sets the env preset in a subshell,
invokes:

```
harbor run \
  -d terminal-bench-sample@2.0 \
  -m <model> \
  -n 4 \
  --agent-import-path bag_agent.agent:BagAgent \
  --ak bag_mode=auto \
  --job-name ablation_<mode>_<model_short>_<UTC>
```

Phases (incremental — each is a self-contained launch, monitor with the
aggregator between phases):

| Phase | Cells                                  | Wall-clock | Token cost (rough)  |
| ----- | -------------------------------------- | ---------- | ------------------- |
| 1     | `full × Opus` (control)                | ~25 min    | ~$25                |
| 2     | `bare × Opus`, `minimal × Opus`        | ~50 min    | ~$50                |
| 3     | `full,bare,minimal × Sonnet`           | ~75 min    | ~$25                |
| 4     | `full,bare,minimal × Haiku`            | ~75 min    | ~$5                 |
|       | **Total**                              | **~3.75 h**| **~$100–$120**      |

(Token cost dominated by Phase 2 — Opus prices ~5× Sonnet, ~25× Haiku, and
Phase 1 + Phase 2 = 3 Opus cells × ~33k tokens/cell × ~$0.015 / 1k. Final
cost lands in the $100–$200 envelope assuming median 30k tokens/trial.)

### 5. Aggregator (`bench/ablation/aggregate.py`)

Reads `bench/jobs/ablation_*` directories, aggregates per cell, and prints:

1. The 3 × 3 pass-count matrix.
2. Per-task pass/fail per cell (mode-major triple per cell, e.g. `✓✗✓`).
3. Helper / hurter classification per model:
   - **HELPS** = full passes, bare AND minimal both fail.
   - **HURTS** = bare or minimal pass, full fails.
   - **NEUTRAL** = all match.
4. JSON dump under `bench/ablation/results/aggregate_<UTC>.json` plus a
   `latest.json` for tooling.

Re-runnable as cells complete; in-progress cells render `X / k (n<10)` so
you can monitor incrementally.

## Results — 3 × 3 matrix

> Status: **harness wired, tests green, runs not yet executed.** This study
> consumes ~$100–$200 of API tokens and ~3.75 h of wall clock; the launch
> decision belongs to the human operator. Run `bench/ablation/run_ablation.sh
> --phase 1` (control) when ready, then incrementally promote through phases
> 2–4. The cells below populate as `bench/ablation/aggregate.py` re-runs.

```
              claude-opus-4-7       claude-sonnet-4-6     claude-haiku-4-6
--------------------------------------------------------------------------------
BAG-full      (pending phase 1)     (pending phase 3)     (pending phase 4)
BAG-bare      (pending phase 2)     (pending phase 3)     (pending phase 4)
BAG-minimal   (pending phase 2)     (pending phase 3)     (pending phase 4)
```

## Per-gate retire / keep decision rubric

When data lands, classify each gate against the criteria below. (Pre-data this
section is the decision rubric; populate after Phase 4.)

| Gate                | Keep if …                                                     | Retire if …                                                    |
| ------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- |
| `probeExtractor`    | `full` > `bare` on Opus by ≥ 1 task                           | `bare` ≥ `full` on Opus AND ≥ on Sonnet                       |
| `selfCheck`         | Sonnet/Haiku gain ≥ 2 tasks vs `bare`                          | No model shows lift ≥ 1 task                                  |
| `snapshotRestore`   | At least one task in `bare` corrupts workspace post-probe     | Probe-corruption never observed (in which case it's pure cost) |
| `retryPath`         | At least one Sonnet/Haiku task flips ✗→✓ on retry             | Retries never flip a verdict in the corpus                    |
| `clusterMatcher`    | `retry_hint.source=cluster` is the deciding hint on a flip    | All flips trace to `library` or `none`                        |
| `viewImage`         | `chess-best-move` passes on `bare`/`full` but fails on `minimal` | (already known: this gate is load-bearing for vision tasks; keep) |
| `codeSearch`        | `bare` > `minimal` by ≥ 1 task on the same model              | `bare` and `minimal` tied on every model                      |

## Predicted outcome (to be falsified)

- **Opus**: `bare ≥ full ≥ minimal`. The harness's LLM gates derail a strong
  model more than they help. View_image is the only gate that pays its weight
  (chess fails without it).
- **Sonnet**: `full ≥ bare ≥ minimal`. Self-check + retry catch sloppy
  submissions on a mid-tier model.
- **Haiku**: `full ≥ bare > minimal`. Probe extractor + self-check rescue
  trivially-failing trials.

If the data confirms this, the per-gate decision is: **default-OFF on Opus,
default-ON on Sonnet/Haiku.** Implementation: ship the env presets as
agent-kwargs and have `bag_agent.agent:BagAgent` choose by `model_name`
prefix at trial setup. The runtime stays generic — only the harness-level
defaults select.

## Cost estimate

- Phase 1 (Opus × full, 10 trials): ~$25
- Phase 2 (Opus × bare + minimal, 20 trials): ~$50
- Phase 3 (Sonnet × 3 × 10 trials): ~$25
- Phase 4 (Haiku × 3 × 10 trials): ~$5
- **Total: ~$105–$200 in API tokens, ~3.75 h wall clock.**

The dominant variability is whether Opus on `terminal-bench-sample` averages
30k or 60k tokens/trial — historical runs (see `bench/jobs/2026-05-02__03-31-16/`)
land around 35–45k. Use `bench/ablation/aggregate.py --target-n 10` to refine
the actual number after Phase 1 completes.

## Reproduction

```sh
# Smoke-test the wiring (no API calls):
bench/ablation/run_ablation.sh --phase 1 --dry-run
bun test tests/harness-gates.test.ts tests/instruction-verifier-gating.test.ts

# Real launch — Phase 1 only (control):
bench/ablation/run_ablation.sh --phase 1

# After it finishes:
python3 bench/ablation/aggregate.py

# Promote through phases 2 → 3 → 4 once each completes.
bench/ablation/run_ablation.sh --phase 2  # ~50 min
bench/ablation/run_ablation.sh --phase 3  # ~75 min
bench/ablation/run_ablation.sh --phase 4  # ~75 min
```
