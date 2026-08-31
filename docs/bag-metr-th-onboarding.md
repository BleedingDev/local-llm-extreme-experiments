# BAG x METR Time-Horizon 1.1 onboarding

This doc captures everything needed to run BleedingAgent (BAG) against
METR's Time-Horizon 1.1 task suite — the canonical "Moore's Law for agents"
benchmark — and to read the resulting time-horizon-vs-success curve.

The adapter lives at `bench/metr_th/`. It is a generic, metadata-driven
runner: there is no task-name keyword scanning anywhere in the filter or
the executor, and BAG's `src/` is untouched.

## What Time-Horizon measures

METR's Time-Horizon metric correlates an agent's success rate against the
*human-equivalent completion time* of a task: a frontier agent in 2025
plateaus around 1-hour tasks, doubles every ~7 months, and Time-Horizon
1.1 (released 2026-01-29) extends the suite to **228 tasks** (up from
170), doubling the count of 8h+ items from 14 to 31 so the upper end of
the curve is no longer pinned by 5 long tasks.

For BAG specifically, the 1-8h regime is where the
`requiresLongWait`-aware routing in `src/task-shape-router.ts` actually
matters — short tasks finish before the qemu-style boot/build/poll
patterns trigger, and >8h tasks blow Opus 4.7's effective working set
even at 1M context. So Time-Horizon directly stresses the part of BAG
that's interesting for the optimizer DAG.

## What ships in this adapter

```
bench/metr_th/
├── __init__.py        # public API: TaskMeta, filters, summarize
├── filter.py          # coding-only subset logic (metadata-driven, audited list)
├── run.py             # python -m bench.metr_th.run — full sweep runner
└── smoke.sh           # 3-task background smoke launcher
```

The runner reuses the existing `bench/bag-runtime/` bundle and the
`scripts/bag_acp_run.ts` driver — same harness Harbor uses for
Terminal-Bench, no fork.

## Coding-only subset filter logic

The upstream `suite_manifest.yaml` enumerates 28 families × N tasks =
**186 tasks visible publicly**. The remaining ~42 of the 228-task TH1.1
suite are HCAST-internal (METR gates the full set behind
`david[at]metr.org`). The adapter handles whichever subset the
suite_manifest currently exposes.

### The three filter axes

The filter in `bench/metr_th/filter.py` consults three **METR-published**
signals — never task names:

1. **Per-family `meta.expertise`** (when the family ships a public
   `manifest.yaml`). Tags METR uses include `software_engineering`,
   `cybersecurity`, `machine_learning`, `general_reasoning`,
   `quantitative_skills`. We keep tasks whose expertise overlaps
   `CODING_EXPERTISE` (`software_engineering`, `cybersecurity`,
   `devops_sysadmin`, `devops`).
2. **Per-task `resources.gpus`** in the family manifest. Any non-zero
   value drops the task. Combined with the family-level
   `GPU_GATED_FAMILIES` blocklist (mlab, replicate_othello,
   iclr_authors, reversal_curse, improve_agent — each documented by
   METR as needing accelerator hardware to fit any model), this
   removes the entire ML-research wing.
3. **`KNOWN_REASONING_FAMILIES`** — fermi_estimate, gaia,
   hypothesis_testing, local_research, local_research_tex, crossword.
   These are pure reasoning/quiz families with no code substrate; they
   correctly fall outside Time-Horizon's coding subset and are gated
   behind an explicit `--include-reasoning` flag.

### The fallback path

Eight families ship in `suite_manifest.yaml` but have no per-family
`manifest.yaml` checked into the public repo (their Docker images are
the source of truth). For these we cannot read `meta.expertise` at
runtime, so we fall back to `KNOWN_CODING_FAMILIES` — an explicit,
auditable list of family names whose category is documented by the
upstream README itself (e.g. `sadservers` is "linux server
troubleshooting", `make_web_server` is "build a web server",
`password_check` is "cybersec password cracking"). Each entry is
justified inline in `filter.py`. **No task names are inspected; only
family names that the upstream `suite_manifest.yaml` already enumerated.**

### What the filter keeps (current snapshot)

```
$ python -m bench.metr_th.run --list-only
[metr-th] resolved 186 total tasks across 28 families
[metr-th] coding-only filter kept 73 of 186 tasks (dropped 113)
```

The 73 surviving coding tasks span 17 families, dominated by:

| family               | N | typical human time |
|----------------------|---|--------------------|
| sadservers           | 29 | ~30 min |
| make_web_server      | 9 | ~20 min |
| password_check       | 6 | ~30 min |
| clone_voice          | 5 | ~8 h |
| debug_small_libs     | 3 | ~1 h |
| esolang              | 3 | ~30 min |
| env_scientist        | 3 | ~2 h |
| multiarmed_bandit    | 3 | ~1 h |
| complex_payments     | 2 | ~3 h |
| cowthello            | 2 | ~10 h |
| symbolic_regression  | 2 | ~1 h |
| clone_game/copycat_llm_api/data_deduplication/hex_chess_website/targeted_phishing/worm | 1 each | varies (~3-6h) |

The 113 dropped tasks are mostly `fermi_estimate` (78, reasoning),
`hypothesis_testing` (11, reasoning), `mlab` (9, ML research, GPU),
`local_research*` (6, reasoning), `iclr_authors` (3, GPU/100h),
`gaia` (2, reasoning), `crossword` (1, reasoning), plus
`replicate_othello` / `reversal_curse` / `improve_agent` (1 each, GPU).

## GPU-task exclusion rationale

The Time-Horizon paper measures *agent capability* on a wall-clock
budget, not *training capacity*. Including GPU-bound ML-research tasks
(`mlab/w*d*`, `replicate_othello/lstm-chess`, `iclr_authors/poster`,
`reversal_curse/exp1`, `improve_agent/0`) on a CPU-only host yields one
of two pathologies:

- The task literally cannot complete (no CUDA → import error → 0%
  pass rate regardless of agent quality). This contaminates the
  bottom of the curve, dragging the agent's apparent time-horizon
  down without measuring anything about the agent.
- The task takes orders of magnitude longer than its human baseline
  (5h CPU vs 1h GPU human), confounding the y-axis (success-rate)
  with the x-axis (human time).

METR's own README explicitly carves out a "compute-light" subset for
this reason — the suite_manifest ships them all, but they are
expected to be filtered out for any sweep that doesn't have GPU
allocations on the runner.

## Per-task expected wall clock

Time-Horizon doesn't publish per-task `human_minutes` in the public
manifest (the field exists in METR's internal canonical form but is
elided from the open-source suite_manifest as of TH1.1). The adapter
falls back to family-typical estimates derived from the published
README and paper text — see `FAMILY_TYPICAL_MINUTES` in `run.py`.

The `--shortest N` flag uses this ranking. The smoke selects 3 tasks
from `make_web_server` (~20 min human-equivalent each), which is the
shortest published family.

For a full 73-task sweep, the projected wall-clock with `--bag-mode
dag-tools` and Opus 4.7 master / Haiku 4.5 local:

| bucket                  | n  | est. agent wall-clock per task | bucket total |
|-------------------------|----|--------------------------------|--------------|
| <=30 min human          | 47 | ~12-15 min agent (BAG cap)     | ~10-12 h     |
| 30-120 min human        | 11 | ~14 min (BAG timeout cap)      | ~2.5 h       |
| 2-6 h human             |  9 | ~14 min (capped)               | ~2 h         |
| 6-10 h human            |  6 | ~14 min (capped)               | ~1.5 h       |
| **total (sequential)**  | 73 |                                | **~16-18 h** |

With `--n-concurrent-trials 4` (Harbor convention; the metr_th runner
is currently sequential — concurrency is a follow-up), this drops to
~4-5 wall-clock hours.

### Token cost projection

Each BAG run with the standard split costs roughly:
- Opus master: ~80k in + ~20k out per 14-minute task (1M-context-aware
  summarisation keeps the prompt bounded)
- Haiku local: ~150k in + ~30k out (classifier + scout chatter)

Per-task: ~$1.40 Opus + ~$0.16 Haiku ≈ **$1.55**.

Full 73-task sweep: **~$115** at 1× attempt, **~$345** at 3×
(Time-Horizon convention is to run each task 5× for 95% CI, but
3× is sufficient to fit the curve at 80% CI).

## BAG `requiresLongWait` instrumentation hooks for telemetry

The classifier in `src/task-shape-router.ts` emits a per-run
`routing-decision.json` artifact with this shape:

```json
{
  "shape": "atomic|compositional|monolithic-complex|hard",
  "mode": "tools|dag-tools",
  "confidence": 0.0,
  "reasoning": "short why",
  "requiresLongWait": true,
  "tokens": {"in": 1234, "out": 56},
  "chosenAt": "2026-05-02T...Z",
  "task": "..."
}
```

The artifact lands at `.bag/runs/<run_id>/routing-decision.json` inside
the task container, gets tar'd into `bag-traces.tar.gz` by the
`bench/metr_th/run.py` container driver, and ends up in
`bench/jobs/<job-name>/<task_dir>/bag-out/bag-traces.tar.gz` on the
host.

To compute "how often does the classifier flag long-wait correctly?"
across a sweep:

```python
# bench/metr_th/analyze_long_wait.py — sketch
import json, tarfile
from pathlib import Path

job = Path("bench/jobs/metr_th_full_sweep_2026-05")
for traces in job.rglob("bag-out/bag-traces.tar.gz"):
    with tarfile.open(traces, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("/routing-decision.json"):
                d = json.loads(tf.extractfile(m).read())
                # cross-reference with task family — sadservers should
                # be requires_long_wait=True (qemu boot), make_web_server
                # should be False, etc.
```

For the time-horizon curve specifically, the interesting telemetry
slice is:

- **Pass rate conditional on `requiresLongWait=True`** vs
  `requiresLongWait=False` per human-minutes bucket. If the classifier
  is calibrated, long-wait-flagged tasks should be the ones where the
  `LONG_WAIT_RUNTIME_HINT` actually helps (sadservers, env_scientist,
  cowthello).
- **`mode` distribution per family**. The router's
  atomic→tools, compositional→dag-tools mapping should show
  make_web_server as predominantly tools-mode and
  complex_payments / clone_voice as dag-tools.
- **Total wall_ms vs human_minutes regression**. Time-Horizon's
  headline curve.

The runner's `summary.json` already aggregates `pass_rate` per
human-minutes bucket — extending it to slice on `requiresLongWait` is a
~10-line addition once the upstream `human_minutes` field stabilises.

## Hard constraints / context budget

- **Opus 4.7 1M context required** for the >2h-human tasks. The 200k
  base context model will blow up reading `/app` after a few iterations
  on `complex_payments` or `clone_voice`. The adapter does NOT pin the
  model — pass `--bag-master claude-opus-4-7[1m]` (or the
  long-context alias) when invoking `python -m bench.metr_th.run`.
- BAG's `--bag-timeout-ms` defaults to 880_000 (~14.7 min). Time-Horizon
  per-task budgets typically scale to *2× the human time*; for a
  matching agent budget on a 6h task, override via
  `--bag-timeout-ms 43200000` (12h). Most coding-only tasks are short
  enough that the default holds.
- METR's public Docker registry requires `secrets.env` (`docker login`
  to METR's registry mirror). The default `metr-public` image
  repository in the adapter is a placeholder — set
  `METR_TH_IMAGE_REPO=ghcr.io/metr-public` (or your private mirror)
  before launching a real sweep.

## Quick start

### Smoke (no token, no images — exercises filter+runner only)

```
bench/metr_th/smoke.sh
# logs at bench/jobs/metr_th_smoke_3/
```

### Full coding-only sweep (1× attempt)

```
export ANTHROPIC_AUTH_TOKEN=...
export METR_TH_IMAGE_REPO=ghcr.io/metr-public  # or your mirror
python -m bench.metr_th.run \
    --job-name metr_th_full_$(date +%Y-%m-%d) \
    --bag-master 'claude-opus-4-7[1m]' \
    --bag-local claude-haiku-4-5-20251001 \
    --bag-mode dag-tools
```

### Inspect filter without running anything

```
python -m bench.metr_th.run --list-only
python -m bench.metr_th.run --list-only --include-reasoning   # adds 113 tasks
```

## License

The adapter code is part of this repository (no upstream copy). The
METR public-tasks suite is **MIT-licensed** with informal upstream
requests not to publish solutions or feed task material into model
training datasets — both honoured by treating METR images as opaque
black boxes inside docker run.

## References

- METR Time-Horizon 1.1 announcement: https://metr.org/blog/2026-1-29-time-horizon-1-1/
- Cross-domain horizon paper / repo: https://github.com/METR/cross-domain-horizon
- Public-tasks suite: https://github.com/METR/public-tasks
- HCAST public subset: https://github.com/METR/hcast-public
- Eval analysis pipeline: https://github.com/METR/eval-analysis-public
- METR Task Standard: https://github.com/METR/task-standard
- Inspect AI: https://meridianlabs-ai.github.io/inspect_swe/
- Original "Measuring AI Ability to Complete Long Tasks" paper: https://arxiv.org/html/2503.14499v1
