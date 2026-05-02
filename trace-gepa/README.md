# trace-gepa

Build a trace-driven GEPA optimiser using Claude Code + Codex traces. Output: an optimised system prompt that BAG (or Codex CLI) can load.

## Stack diagram

```
  ~/.claude/projects/*.jsonl ──┐
                               ├─► extractors/extract_cc.py    ──► data/cc_dataset.jsonl    ─┐
  ~/.codex/sessions/**/*.jsonl ┴─► extractors/extract_codex.py ──► data/codex_dataset.jsonl ─┤
                                                                                              ├─► extractors/categorize.py
                                                                                              │     │
                                                                                              │     ├─► data/dataset.jsonl
                                                                                              │     └─► data/splits.json
                                                                                              ▼
                                       agent_opt/adapter.py  ◄────►  agent_opt/reflection.py (reflection LM: claude-opus-4-7)
                                                  ▲                              │
                                                  │   task LM: claude-opus-4-7   │
                                                  └──────────────┬───────────────┘
                                                                 ▼
                                                       agent_opt/optimize.py
                                                                 │
                                                                 ▼
                                                artifacts/optimized-prompts/*.json
                                                                 │
                                                                 ▼
                                                  BAG runtime / Codex CLI loader
```

## Filesystem layout

```
trace-gepa/
├── SHARED_BRIEFING.md          shared schema + constraints (read first)
├── README.md                   this file
├── data/
│   ├── seed_sessions.json      30 high-score seed sessions (15 CC + 15 Codex)
│   ├── cc_dataset.jsonl        Claude Code extracted records
│   ├── codex_dataset.jsonl     Codex extracted records
│   ├── dataset.jsonl           merged + categorised
│   └── splits.json             train/val/test indices
├── extractors/
│   ├── extract_cc.py           CC session jsonl  -> cc_dataset.jsonl
│   ├── extract_codex.py        Codex rollout     -> codex_dataset.jsonl
│   └── categorize.py           merge + label + split
├── agent_opt/                  (local package; renamed from `gepa/` to avoid shadowing the PyPI `gepa` package)
│   ├── adapter.py              GEPAAdapter binding task LM to dataset record
│   ├── reflection.py           reflection LM driver (proposes new prompts)
│   ├── llm.py                  shared dspy.LM factory (reads ANTHROPIC_AUTH_TOKEN)
│   └── optimize.py             top-level GEPA optimisation loop
├── bench/
│   └── eval_baseline.py        baseline score before optimisation
├── scripts/
│   └── sanity_check.py         workspace + venv validator
├── tests/
└── artifacts/
    └── optimized-prompts/      output: best prompts / Pareto front
```

## Quickstart

Run from `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/`. All commands use the project venv at `.venv-gepa/`.

```sh
# 1. setup (one-shot venv + sanity)
.venv-gepa/bin/python trace-gepa/scripts/sanity_check.py

# 2. extract Claude Code sessions
.venv-gepa/bin/python trace-gepa/extractors/extract_cc.py \
  --seed trace-gepa/data/seed_sessions.json \
  --out  trace-gepa/data/cc_dataset.jsonl

# 3. extract Codex sessions
.venv-gepa/bin/python trace-gepa/extractors/extract_codex.py \
  --seed trace-gepa/data/seed_sessions.json \
  --out  trace-gepa/data/codex_dataset.jsonl

# 4. categorise + split
.venv-gepa/bin/python trace-gepa/extractors/categorize.py \
  --cc    trace-gepa/data/cc_dataset.jsonl \
  --codex trace-gepa/data/codex_dataset.jsonl \
  --out   trace-gepa/data/dataset.jsonl \
  --splits trace-gepa/data/splits.json

# 5. run GEPA reflective optimisation
.venv-gepa/bin/python trace-gepa/agent_opt/optimize.py \
  --dataset trace-gepa/data/dataset.jsonl \
  --splits  trace-gepa/data/splits.json \
  --out     trace-gepa/artifacts/optimized-prompts/

# 6. point BAG at the new prompt
cp trace-gepa/artifacts/optimized-prompts/best.json bag.config.json.prompts
```

## Models

- Task LM (used during evaluation): `anthropic/claude-opus-4-7`
- Reflection LM (proposes new prompts): `anthropic/claude-opus-4-7`

All LM calls (task + reflection) use `claude-opus-4-7` (Anthropic API). Cost is not a concern.

**Env var quirk**: `.env` ships `ANTHROPIC_AUTH_TOKEN`, NOT `ANTHROPIC_API_KEY`. The dspy.LM factory in `agent_opt/llm.py` reads `os.environ["ANTHROPIC_AUTH_TOKEN"]` and passes it explicitly as `api_key=`. Don't rename it.

## Dataset

For the per-record schema (id, src, context, observed_action, label, failure_category, ...) see `SHARED_BRIEFING.md`. Every record carries a label of `good` / `bad` / `user_corrected` / `user_confirmed` and an optional `failure_category` from the recon taxonomy.

### What's in the dataset

<!-- AUTO-FILLED -->

**Source counts** (total: 3929)

| source | count |
|---|---|
| cc | 948 |
| codex | 2981 |

**Label distribution**

| label | count | pct |
|---|---|---|
| good | 3348 | 85.2% |
| bad | 426 | 10.8% |
| user_confirmed | 150 | 3.8% |
| user_corrected | 5 | 0.1% |

**Top 8 failure categories** (excluding `null`)

| category | count |
|---|---|
| `bash_exit_nonzero` | 388 |
| `hallucinated_path` *(tagged by categorize.py)* | 237 |
| `retry_loop` *(tagged by categorize.py)* | 22 |
| `bash_timeout_141` | 15 |
| `cancelled_parallel_batch` | 7 |
| `user_correction` *(tagged by categorize.py)* | 5 |
| `cmd_not_found_127` | 5 |
| `hallucinated_skill` *(tagged by categorize.py)* | 2 |

**Splits**

| split | total | good | bad | user_confirmed | user_corrected |
|---|---|---|---|---|---|
| train | 2751 | 2347 | 296 | 104 | 4 |
| val | 589 | 496 | 65 | 27 | 1 |
| test | 589 | 505 | 65 | 19 | 0 |

**Date range (approx, from seed source files):** 2026-01-26 to 2026-05-01. See `data/seed_sessions.json` for per-session detail.

<!-- AUTO-FILLED -->

## Why Python + GEPA over TS + Ax

- **GEPA reference impl** is in Python (gepa 0.0.27 on PyPI) and integrates natively with dspy 3.2.0; AxGEPA is younger, fewer adapters in the wild.
- **dspy** gives us free access to LiteLLM model routing, structured-output retries, and a familiar `Module / Predict` ergonomic that maps cleanly onto our trace records.
- **The ecosystem we already lean on for dataset work** (orjson, pyarrow, datasets, rich, tqdm) is Python-first; mixing TS for orchestration here would be friction with no payoff. BAG itself stays TS — only the optimiser layer is Python.

## Known limits

- Today's seed = 30 sessions (15 CC + 15 Codex) selected by recon score; total examples expected ~2-3K after extraction.
- Per-session cap of 200 examples and per-event 2 KB truncation keep the dataset small.
- Codex archive is 23 GB; we stream line-by-line and never materialise full sessions in memory.
- No deep PII scrub yet — only obvious key prefixes (`sk-ant-`, `hf_`, `ghp_`) are redacted.
- Reflection LM cost scales with `--budget` in `agent_opt/optimize.py`; default is conservative.
- Will scale to a wider session pool (and bring sub-agent traces into a dedicated split) once the first end-to-end run validates the loop.

See `SHARED_BRIEFING.md` for the canonical schema, failure taxonomy, and per-agent ownership rules.

## Tracks

The reflective optimiser can be seeded from any of several "track" modules in `agent_opt/`. Each track ships its own task-tuned seed prompt and is selectable via `--seed-module`:

| track | seed module | intent |
|---|---|---|
| `default` | `agent_opt/seed.py` | Generic next-tool predictor seed (works on both CC and Codex traces). |
| `bag` | `agent_opt/seed_bag.py` | BAG-runtime planner seed; mirrors the runtime's tool roster + output contract. |
| `codex` | `agent_opt/seed_codex.py` | Codex-CLI tuned seed; richer Codex tool taxonomy + `function_call` discipline. |

A fourth experimental track, `hybrid`, is produced offline by `agent_opt/merge_prompts.py`, which blends two source artefacts via the reflection LM. See `trace-gepa/artifacts/optimized-prompts/hybrid_*/run_meta.json` for the merger metadata.

## Optimised prompt loading

The BAG runtime ships with `loadOptimizedPlannerPrompt()` (`src/optimized-prompt-loader.ts`), which is **default-on**: whenever an artefact exists at `<repoRoot>/artifacts/optimized-prompts/latest/best_candidate.json`, the planner uses that prompt and logs `[bag] using optimized planner prompt run=<runId>` on first use.

Resolution order:
1. `BAG_REPO_ROOT` (when set, authoritative — no fallthrough).
2. `process.cwd()`.
3. The directory two levels above the loader source file (resolves to the repo root for normal in-tree runs).

Disable mechanisms (use any one):
- `export BAG_DISABLE_OPTIMIZED_PROMPT=1` (also accepts `true`).
- Remove or rename the `artifacts/optimized-prompts/latest` symlink.

Verified by `tests/optimized-prompt-loader.test.ts` (4 cases: missing artefact, default-on, `=1` disable, `=true` disable) and the end-to-end `trace-gepa/scripts/bag_smoke_no_root.ts` smoke (no `BAG_REPO_ROOT` set; resolves through the top-level `artifacts/optimized-prompts -> trace-gepa/artifacts/optimized-prompts` symlink).

## Smoke run results

End-to-end runs of `agent_opt/optimize.py` against the GEPA loop, task LM = `claude-opus-4-7`, reflection LM = `claude-opus-4-7`. Per-run details live in `trace-gepa/artifacts/optimized-prompts/<run_id>/run_meta.json`; the canonical comparison table is auto-generated to **`trace-gepa/REPORT.md`** by `scripts/aggregate_runs.py` — refresh it whenever new runs land.

### Wave 3 results (latest, 2026-05-01)

Best-of-track val_after across the three primary seed modules:

| seed_module | run_id | budget | train | val | val_before | val_after | delta | prompt_chars |
|---|---|---|---|---|---|---|---|---|
| codex | codex_run_20260501T224340Z | 300 | 80 | 40 | 0.5750 | **0.7500** | +0.1750 | 2490 |
| bag | bag_run_20260501T224339Z | 300 | 80 | 40 | 0.6000 | 0.6875 | +0.0875 | 2572 |
| default (xl) | run_20260501T223837Z | 600 | 120 | 50 | 0.4800 | 0.6200 | +0.1400 | 2445 |
| v2 | v2_run_20260501T224342Z | 300 | 80 | 40 | 0.4750 | 0.5625 | +0.0875 | 2072 |

Held-out test bench (`bench/results_wave2_final.json`, n=60, task=`claude-haiku-4-5`) compares all 5 prompt candidates plus the seed; the `bag` candidate took the test crown at **pass_rate = 0.7667** (vs seed 0.4667), with `bag` and `codex` both hitting 0.8 on `label=bad` and 1.0 on `user_confirmed`.

### Earlier history

- **Budget = 8.** GEPA loop closed cleanly but the metric-call budget was entirely consumed by the initial baseline pass — zero reflective iterations. Plumbing-only smoke.
- **Budget = 50.** 8+ reflective iterations executed; under strict-better (`>`) acceptance no candidate beat the seed on the 3-sample minibatch. Full proposal/evaluate/score/compare/log path verified.
- **Why ties dominate at small minibatch.** Binary tool-name match × `minibatch_size = 3` lives in `{0, 1, 2, 3}`. Mitigations: bigger minibatch, richer scorer (partial credit), or a deliberately weaker seed.
- **Budget = 200 (the "big run", `run_20260501T220148Z`).** First run to actually clear the strict-better gate: val 0.5000 → 0.5667, prompt 958 → 2021 chars; bench `results_big_run.json` confirmed seed=0.467 vs optim=0.600 (delta +0.133) on the held-out test split. Superseded by Wave-3 runs above.

<!-- BIG_RUN_RESULTS -->

### Post-unify baseline eval (test_size=30, max_workers=6)

After unifying `SEED_PROMPT` into `agent_opt/seed.py` (a single source of truth shared by `optimize.py` and `bench/eval_baseline.py`), the baseline eval is a *true* seed-vs-artifact comparison. Numbers below are from before any track-specific run had landed; see the Wave 3 table above for current state.

| metric  | seed  | optim | delta  |
|---------|-------|-------|--------|
| overall | 0.467 | 0.467 | +0.000 |
| label=good (n=23) | 0.478 | 0.478 | +0.000 |
| label=bad  (n=6)  | 0.500 | 0.500 | +0.000 |
| failure=bash_exit_nonzero (n=3) | 0.333 | 0.333 | +0.000 |
| failure=hallucinated_path (n=2) | 1.000 | 1.000 | +0.000 |

Full per-example breakdown: `bench/results_post_unify.json`.
