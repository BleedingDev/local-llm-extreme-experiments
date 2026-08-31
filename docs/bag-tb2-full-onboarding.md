# BAG x Terminal-Bench 2.0 FULL — onboarding

**Status:** smoke (3 tasks) launched 2026-05-02 — see `bench/jobs/tb2_full_smoke_3/`.
Adds Terminal-Bench 2.0 full (89 tasks, 8 categories) as a *second* BAG benchmark
config alongside the existing `terminal-bench-sample@2.0` (10-task curated easy
slice). No changes to BAG `src/`; this is purely a benchmark-runner addition.

## Dataset name

| Spec | Value |
|---|---|
| Harbor name | `terminal-bench` |
| Version | `2.0` |
| Tasks | **89** |
| Source | Harbor Hub registry (default) |
| Hub URL | https://hub.harborframework.com/datasets |
| Upstream | https://github.com/laude-institute/terminal-bench-2 |

Confirmed via Harbor 0.6.3 CLI:

```bash
cd bench && . .venv/bin/activate
harbor dataset list --legacy 2>&1 | grep -E 'terminal-bench\b'
# │ terminal-bench             │ 2.0     │    89 │ Version 2.0 of Terminal-Bench, …
```

So the canonical `harbor run -d` argument is **`terminal-bench@2.0`**. This is
the same dataset Anthropic and ForgeCode quote scores against. We were running
the 10-task curated `terminal-bench-sample@2.0` slice — that sample is *not* a
proxy for full TB 2.0 (the 89 hard tasks span 8 categories: software engineering,
data science, security/CTF, system administration, scientific computing,
data extraction, model training, ML/AI). Score deltas of 6+ pp from container
RAM/CPU alone (reported by Anthropic) make pinning these settings critical.

Resolution attempts (in order, before landing on `terminal-bench@2.0`):

1. `harborframework/terminal-bench-2.0@latest` — *not needed*; the legacy
   registry name `terminal-bench@2.0` resolves directly.
2. `terminal-bench@2.0` — **WORKS**. Used.
3. `terminal-bench-core@head` — not present in registry; would only matter
   if we tracked a moving target.

Adjacent datasets in the same family (kept here so we don't conflate them
in future runs):

| Name | Version | Tasks | Notes |
|---|---|---|---|
| `terminal-bench-sample` | `2.0` | 10 | Curated easy slice. What BAG ran prior. |
| `terminal-bench` | `2.0` | **89** | **Full TB 2.0 — this onboarding.** |
| `terminal-bench-pro` | `1.0` | 200 | Pro variant (different dataset; not a superset). |

## Reproducible command

```bash
cd bench && . .venv/bin/activate
set -a && source ../.env && set +a   # ANTHROPIC_AUTH_TOKEN

PYTHONPATH=. harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path bag_agent.agent:BagAgent \
  -m claude-opus-4-7 \
  -n 4 \
  --agent-timeout-multiplier 2.0 \
  --agent-setup-timeout-multiplier 3.0 \
  --ak bag_mode=auto \
  --job-name tb2_full_$(date +%Y%m%d_%H%M%S)
```

For a smoke (subset) run, append `-l N` (alias `--n-tasks`):

```bash
PYTHONPATH=. harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path bag_agent.agent:BagAgent \
  -m claude-opus-4-7 \
  -n 4 -l 3 \
  --agent-timeout-multiplier 2.0 \
  --agent-setup-timeout-multiplier 3.0 \
  --ak bag_mode=auto \
  --job-name tb2_full_smoke_3 -y
```

Same flags BAG uses against `terminal-bench-sample@2.0` today — only `-d` changes
from `terminal-bench-sample@2.0` to `terminal-bench@2.0`. BAG agent kwargs
(`bag_mode=auto`, master/local model split) are the same so cross-run comparison
is meaningful.

## Smoke launch — 2026-05-02

| Field | Value |
|---|---|
| Job name | `tb2_full_smoke_3` |
| Job dir | `bench/jobs/tb2_full_smoke_3/` |
| Stdout/stderr log | `bench/jobs/tb2_full_smoke_3.log` |
| Tasks (first 3, alphabetical from registry shuffle) | `gpt2-codegolf`, `llm-inference-batching-scheduler`, `break-filter-js-from-html` |
| Concurrency | 4 (only 3 trials so effectively 3-way parallel) |
| Mode | `bag_mode=auto` (adaptive router; tools vs dag-tools per task shape) |
| Master model | `claude-opus-4-7` |
| Local model | `claude-haiku-4-5-20251001` (BagAgent default) |

Container images pulled from `alexgshaw/<task>:20251031` (per terminal-bench
2.0 task manifests).

## Expected runtime

Extrapolated from `terminal-bench-sample@2.0` history (this repo's own runs):

- Sample 10 tasks @ n=4 concurrency: **~30-50 min** wall (15m59s for dag-tools,
  33m53s for tools per `bag-tb-tool-use-vs-dag-tools.md`).
- Sample tasks are explicitly the **easy slice**; full TB 2.0 is harder and
  the 880s `bag_timeout_ms` ceiling will bite more often → expect each task
  to consume closer to its full timeout budget.

Per-task allowance with the multipliers above:

- `agent-timeout-multiplier=2.0` → 2× the dataset's base agent_timeout (typically
  900s base ⇒ **1800s** = 30 min per agent run).
- `agent-setup-timeout-multiplier=3.0` → 3× base setup (typically 360s ⇒ 1080s).
- Total per-task ceiling ≈ **30 + 18 = 48 min** in the worst case.

**Full 89-task wall-time estimate (n=4):**

- Pessimistic (every task hits ceiling, no concurrency overlap loss):
  `89 × 48 / 4 ≈ 1068 min ≈ 17.8 h`.
- Realistic (Sonnet/Opus on TB 2.0 averages ~12 min/task on 70%+ adapters):
  `89 × 12 / 4 ≈ 4.5 h` — closer to a single overnight run.
- Optimistic (BAG dag-tools-style fast wins like sample's 16 min/10):
  `89 × 9 / 4 ≈ 3.3 h`.

Plan for **4-5 hours per full run** as the budgetable estimate. First run will
be longer because container images are not cached locally yet
(`docker pull alexgshaw/<task>:20251031` × 89 ≈ 30-90 GB pulls depending on
which tasks include large model weights / datasets).

## Cost estimate per full run (89 tasks, BAG default cost split)

Per-task BAG token usage observed on `terminal-bench-sample@2.0` (mean across
last 4 jobs in `bench/jobs/`):

- master (Opus 4.7): ~150-300k input, ~25-50k output per task
- local (Haiku 4.5): ~30-60k input, ~10-20k output per task

At public list prices (claude-opus-4-7 ≈ $15/MTok in, $75/MTok out;
claude-haiku-4-5 ≈ $1/MTok in, $5/MTok out), a typical task lands in
**$3.50 - $7** range; tail tasks (heavy autonomous loops, regen cycles) hit
**$10-15**.

**Budget per full TB 2.0 run:** **~$400 - $700** (89 tasks × ~$5 average +
fudge for tail). Smoke (3 tasks) is **~$15-30**. Add ~$0 for compute (local
Docker on the dev box).

If costs need to be cut without changing BAG itself, lever order:

1. Set `--ak bag_mode=tools` (skips planner LLM calls). −15-25% tokens.
2. Override `bag_local_model` to keep using Haiku 4.5 (already default).
3. Reduce `n_attempts` if it was raised for stability — default is 1 already.
4. Disable telemetry only if you don't need GEPA traces (this repo wants them).

## Gotchas

### 1. Memory / CPU sensitivity (Anthropic-confirmed: 6 pp delta)

Terminal-Bench 2.0 tasks declare default container resources in `task.toml`
(`override_cpus`, `override_memory_mb`, `override_storage_mb`). The Harbor job
config exposes overrides — they are currently `null` (use task defaults), which
is the **correct** posture for headline scores. **Do not** override these from
the command line unless reproducing a specific Anthropic-reported number.

If you must pin to match a published result, also pin Docker daemon settings:

```bash
# Set Docker Desktop limits (mac) BEFORE starting the run:
# Docker → Settings → Resources → CPUs ≥ 8, Memory ≥ 16 GiB, Swap ≥ 1 GiB
# Linux: edit ~/.docker/daemon.json or systemd drop-in; restart dockerd
```

Anthropic's 6 pp finding is specifically about **insufficient host RAM** causing
swap-driven slowdowns inside test containers (verifiers time out before the
agent's solution finishes converging). On the current 16 GiB host (`docker
info`: 15.65 GiB total), full TB 2.0 with 4 concurrent containers is at the
lower edge — consider `-n 2` if RAM pressure shows up in `docker stats` during
the smoke run.

### 2. Container variability

Some TB 2.0 tasks include large model weights or datasets in the container
image. First-time pulls of `alexgshaw/<task>:20251031` images can each be
multiple GB. Job log shows:

```
Skipping image OS validation for alexgshaw/<task>:20251031: docker inspect returned 1
```

This is expected before pull completes. Pre-pulling images saves 30-60 minutes
on first full run:

```bash
# After the smoke run completes, list which images TB 2.0 needs:
docker images | grep alexgshaw
```

### 3. BAG `bag_timeout_ms` ceiling vs harbor agent timeout

`BagAgent.bag_timeout_ms` defaults to **880000 ms (≈ 14.7 min)** — INSIDE the
30-minute harbor agent timeout window above. Hard tasks where BAG needs >15
min will be cut by BAG's own timer, not harbor's. To raise:

```bash
--ak bag_mode=auto --ak bag_timeout_ms=1700000   # 28 min, leaves harbor 2 min slack
```

Recommend bumping `bag_timeout_ms` to **1700000** for the first full TB 2.0
run because the curated sample biased timeouts low.

### 4. `terminal-bench-sample` ≠ subset of `terminal-bench`

The sample is a *curated easy slice*, not a random sample of the full set, so
do **not** subtract sample scores from full scores — they're independent
benchmarks for our purposes. Track them in separate columns of any scoreboard.

### 5. Trace artefacts location

BAG's traces tarball lands under
`bench/jobs/<job-name>/<task>__<trial>/agent-logs/bag-traces.tar.gz` and is
auto-extracted to `agent-logs/bag-traces/`. Per-trial sentinel guards stale
traces from re-used containers — see `bench/bag_agent/agent.py:283-380` for
the exact lifecycle.

### 6. Don't run full 89 unless you have the budget approved

Smoke first (3 tasks, ~15 min, ~$30). Audit per-task wall and token spend.
THEN authorise the full run. The 17.8 h pessimistic ceiling means an
unattended full run on a laptop can wedge if Docker pulls fail mid-run — for
the headline number, run on a fresh-rebooted machine with all task images
pre-pulled.

## Scoreboard template

After the smoke run finishes, capture:

```bash
cd bench && python -c "
import json, glob, os
job = 'jobs/tb2_full_smoke_3'
results = []
for path in glob.glob(f'{job}/*/result.json'):
    task = os.path.basename(os.path.dirname(path))
    with open(path) as f: data = json.load(f)
    results.append((task, data.get('reward')))
for t, r in results: print(f'{t:50s}  reward={r}')
print(f'mean = {sum(r or 0 for _, r in results)/len(results):.3f}')
"
```

Compare against:

| Run | Dataset | Tasks | Mean reward | Wall | Notes |
|---|---|---|---|---|---|
| TB sample baseline | `terminal-bench-sample@2.0` | 10 | 0.700 (run #5 tools) | 33m53s | This repo's prior best |
| TB sample dag-tools | `terminal-bench-sample@2.0` | 10 | 0.600 (run #6) | 15m59s | dag-tools regression |
| TB2 full smoke | `terminal-bench@2.0` | 3 | TBD | TBD | This onboarding |
| TB2 full | `terminal-bench@2.0` | 89 | TBD (target ≥ 0.50) | ~4-5h | After smoke gate |

Public reference points:
- ForgeCode on full TB 2.0 (claude-opus): **81.8%** (cited in
  `bag-tb-sample-tool-use-pivot.md`).
- claude-code agent (anthropic harness) on full TB 2.0: **~70-75%**
  (per Anthropic's own README example).

A BAG result above ~0.55 on full TB 2.0 would already be competitive given
that BAG's adaptive router specifically targets the sample's failure modes.

## Workaround if Harbor Hub is unreachable

The dataset IS on the default registry — no workaround needed today. If the
registry endpoint goes down:

1. Pre-download via `harbor dataset download terminal-bench@2.0 --download-dir
   ~/.harbor/datasets/`.
2. Pass `-d terminal-bench@2.0 --download-dir ~/.harbor/datasets/` (or use a
   `--config` JSON pointing at a `LocalDatasetConfig`).
3. As a last resort, clone https://github.com/laude-institute/terminal-bench-2
   and write a thin Harbor adapter mapping its `tasks/` to Harbor `task.toml`
   format. The Harbor 0.6.3 source has reference adapters under
   `bench/vendor/harbor/adapters/` — `aider_polyglot` is the closest in shape.

This path is documented for completeness; not needed for the onboarding run.
