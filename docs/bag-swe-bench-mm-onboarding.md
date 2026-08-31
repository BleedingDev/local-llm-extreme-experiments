# BAG x SWE-Bench Multimodal — Onboarding

This document describes how the **BleedingAgent (BAG)** multimodal evaluation
adapter at `bench/swe_bench_mm/run.py` exercises BAG's `view_image` tool on
the **SWE-Bench Multimodal** benchmark.

## Why this benchmark

SWE-Bench Multimodal (M. Yang et al., ICLR 2025) is the only public software
engineering benchmark that ships **image attachments** (UI screenshots, before/
after renders, mockups) inline with bug reports. Of the 102 dev-split
instances:

| Statistic                                          | Count   | Share |
| -------------------------------------------------- | ------- | ----- |
| Instances with at least one problem-statement image | 95 / 102 | 93.1% |
| Instances with patch- or test_patch-bucket images   | 13 / 102 | 12.7% |
| Instances with **no** images at all                 | 0 / 102  | 0.0%  |
| Total problem-statement images                      | 163      | -     |
| Total images across all buckets                     | 310      | -     |

(Numbers measured 2026-05-02 against `princeton-nlp/SWE-bench_Multimodal` rev
on Hugging Face.) The test split (510 instances) is even larger; the paper
states 83.5% of test-split tasks require image understanding.

For BAG specifically, this benchmark is the public yardstick for the
multimodal `view_image` tool that lives in `src/autonomous-tools.ts` —
without an end-to-end eval, that tool's contribution to task-resolution rate
is unmeasured.

The published baseline from the paper is **SWE-agent + GPT-4o = 11.5%**
resolved on the test split; that is the public ceiling we aim to clear.

## Repo layout

```
bench/swe_bench_mm/
├── run.py            # adapter (everything below)
└── (per-job output)  # bench/jobs/<job_name>/instance_<id>/
                       #   repo/    fresh checkout @ base_commit + test_patch
                       #   images/  downloaded screenshots
                       #   task.md  prompt sent to BAG
                       #   meta.json
                       #   bag-acp-summary.json
                       #   bag.log
                       #   bag.patch          <- BAG's diff vs baseline
                       #   instance_result.json
                       # bench/jobs/<job_name>/result.json     <- aggregate
```

## Dataset access (Hugging Face)

The dev split is a public, ungated HuggingFace dataset:

```
princeton-nlp/SWE-bench_Multimodal[dev]   # 102 instances
princeton-nlp/SWE-bench_Multimodal[test]  # 510 instances (test labels masked)
```

Set `HF_TOKEN` to bypass anonymous rate limits (recommended even for unauth
access). The adapter uses `datasets.load_dataset(...)` and caches the dataset
under `~/.cache/huggingface/`.

Schema (per row) used by the adapter:

| Field              | Used by adapter                                       |
| ------------------ | ----------------------------------------------------- |
| `instance_id`      | Folder name + correlation key                         |
| `repo`             | GitHub `<org>/<name>` for `git clone`                 |
| `base_commit`      | Detached checkout target                              |
| `problem_statement`| Embedded into `task.md` verbatim                      |
| `test_patch`       | Applied unstaged before BAG sees the workspace        |
| `image_assets`     | JSON-encoded `{bucket: [url|{path,url}]}`             |
| `FAIL_TO_PASS`     | Listed in the prompt; used for verification heuristics |
| `PASS_TO_PASS`     | Listed in the prompt                                  |

`image_assets` buckets are dynamic (current dev split has
`problem_statement`, `patch`, `test_patch`). Some entries are bare URL
strings (typical screenshots); others are `{path: "...", url: "..."}` dicts
where the file is a binary fixture that must land at a specific path inside
the repo so unit tests can read it. The adapter handles both shapes and
also stages the in-repo files into the working tree.

## Prerequisites

- **Python 3.12+** with `datasets` and `requests` (already in
  `bench/.venv` for this repo).
- **Node 20/22/24** with `npm` (BAG runtime ships its own `tsx`).
- **Git** with `--filter=blob:none` partial-clone support (git ≥ 2.20).
- **`ANTHROPIC_AUTH_TOKEN`** (or `ANTHROPIC_API_KEY`) — read from the
  process env or from `<repo>/.env`. BAG's `master` and `local` model roles
  both default to `claude-opus-4-7` (override via `bench/bag-runtime/bag.config.json`).

The adapter calls `npm install` automatically inside `bench/bag-runtime/`
on first invocation.

## Running locally

```bash
# Smallest sanity smoke (1 instance, BAG enabled)
./bench/.venv/bin/python bench/swe_bench_mm/run.py \
    --instance-ids chartjs__Chart.js-10301 \
    --job-name swe_bench_mm_one \
    --bag-timeout-sec 480

# Default smoke (5 instances)
./bench/.venv/bin/python bench/swe_bench_mm/run.py -l 5 \
    --job-name swe_bench_mm_smoke

# Build fixtures only (no BAG calls — useful for debugging the harness)
./bench/.venv/bin/python bench/swe_bench_mm/run.py -l 5 --skip-bag \
    --job-name swe_bench_mm_dryrun

# Full dev split (102 instances, sequential — ~few hours @ ~120 s/instance)
./bench/.venv/bin/python bench/swe_bench_mm/run.py -l 102 \
    --job-name swe_bench_mm_dev_full
```

Caches:
- `~/.cache/swe_bench_mm/repos/<owner>__<name>/` — single shared partial
  clone per upstream repo, copied per instance.
- `~/.cache/swe_bench_mm/images/<sha256>.<ext>` — image attachments.
- `~/.cache/huggingface/...` — dataset rows.

Override with `--repo-cache` / `--image-cache` or
`SWE_BENCH_MM_REPO_CACHE` / `SWE_BENCH_MM_IMAGE_CACHE`.

## What BAG sees

The adapter materialises this layout per instance:

```
bench/jobs/<job>/instance_<id>/
├── repo/                 (cwd for BAG; --workdir of bag_acp_run.ts)
│   ├── images/           (symlink -> ../images/, ignored via .git/info/exclude)
│   ├── ...repo files at base_commit + test_patch...
│   └── .git/             (real, includes a `swe-bench-mm-baseline` tag)
├── images/
│   ├── problem_statement_00.png
│   ├── problem_statement_01.png
│   └── ...
├── task.md
├── meta.json
└── bag.log
```

`task.md` lists each downloaded image with its bucket origin and the relative
path BAG should hand to `view_image`, e.g.:

> Use the `view_image` tool with the relative path inside this workspace
> (e.g. `view_image {"path":"images/problem_statement_00.png"}`) to attach a
> screenshot to the next model turn before reasoning about it.

The `view_image` tool resolves paths relative to BAG's cwd (`repo/`), and
the symlink at `repo/images/` makes the host-side `images/` directory
visible. The harness creates the symlink **after** committing the
`swe-bench-mm-baseline` tag and registers it in `.git/info/exclude`, so it
never appears in the captured diff.

### Verifying image accessibility

Each fixture's `meta.json` records the `local_path` plus a `bytes` count
per image. `instance_result.json` reports `image_count` and
`images_with_problem_statement` per instance — at the aggregate level the
job's `result.json` rolls these up to `instances_with_problem_images` and
the **`total_view_image_calls`** counter (read from BAG's
`autonomous-trace.json` per run).

### Expected `view_image` invocation rate

On the smoke run, BAG called `view_image` ≈ 4 times per instance for the
single-instance probe (`chartjs__Chart.js-10301`, 3 problem-statement
screenshots, BAG inspected each at least once and re-read one). Working
hypothesis based on the paper (83.5% of test instances require visual
information) and prompt design:

| Population                   | Expected `view_image` ≥ 1 instance rate |
| ---------------------------- | -------------------------------------- |
| Dev instances with problem images | ≥ 80% (95/102 = 93.1% have problem images, prompt explicitly directs BAG to use the tool) |
| Full dev split                    | ≥ 75%                                  |
| Full test split                   | ≈ 80% (matches paper's 83.5% requirement) |

If the smoke run shows < 50% instances exercising `view_image` despite
problem images being present, the prompt template in `build_task_prompt`
in `run.py` likely needs to be sharpened (e.g. moved earlier in the
instruction, made more imperative).

## Verification: local vs sb-cli

There are two ways to grade BAG's output:

### 1. Local best-effort verification (`--allow-local-npm`)

`run.py --allow-local-npm` runs `npm install` + `npm test` inside each
fixture and grep-counts `FAIL_TO_PASS` test names with PASS markers. This
is heuristic and **not** authoritative — it cannot reproduce repo-specific
Docker setups (DBs, browser harnesses, headless WebGL, etc.) and a missing
test framework binding will silently report 0/N. Use it for triage only.

### 2. Official upstream verification (sb-cli)

The Princeton team distributes the SWE-Bench M+L cloud verifier as
[`sb-cli`](https://github.com/princeton-nlp/sb-cli). It accepts a
predictions JSON file shaped like:

```json
{
  "instance_id": "chartjs__Chart.js-10301",
  "model_name_or_path": "bag",
  "model_patch": "<unified diff>"
}
```

To submit BAG's smoke output:

```bash
# Build a predictions.jsonl from a job
python -c '
import json, glob, pathlib
out = []
for p in glob.glob("bench/jobs/swe_bench_mm_smoke_3/instance_*/bag.patch"):
    instance_id = pathlib.Path(p).parent.name.removeprefix("instance_")
    out.append({
        "instance_id": instance_id,
        "model_name_or_path": "bag-claude-opus-4-7",
        "model_patch": pathlib.Path(p).read_text(),
    })
pathlib.Path("bag-predictions.jsonl").write_text(
    "\n".join(json.dumps(x) for x in out) + "\n"
)
'

# Then submit (test split only; dev split runs locally via Docker)
sb-cli submit swe-bench-m --predictions_path bag-predictions.jsonl \
    --run_id bag-smoke-$(date +%Y%m%d)
```

> **Test split is gated through sb-cli's cloud only.** The current adapter
> deliberately scopes to the **public dev split**; do not attempt to verify
> on the test split outside of sb-cli — its labels are masked.

For local Docker-based dev verification, follow the official guide at
https://www.swebench.com/multimodal.html (it requires repo-specific Docker
images shipped by the SWE-Bench team).

## Adapter contract: generic for multimodal SWE benchmarks

`run.py` is intentionally generic over any benchmark whose dataset rows
expose:

- `instance_id`, `repo`, `base_commit`, `problem_statement`, `test_patch`,
- `image_assets` shaped as `{bucket: [url|{path,url}]}` (any bucket names),
- `FAIL_TO_PASS` / `PASS_TO_PASS` lists.

To wire a new multimodal SWE benchmark:

1. Edit `DATASET_NAME` and `DEV_SPLIT` at the top of `run.py`.
2. (Optional) extend `parse_image_assets` if the new dataset uses a
   different image-asset schema.
3. Submit predictions through whatever harness the upstream owns.

The adapter does **not** rely on any benchmark-specific keyword logic in
the prompt — image-bucket names are surfaced verbatim to BAG via
`task.md`.

## Known gotchas

- The `--filter=blob:none` partial clone keeps the cache slim; if the
  dataset asks for a commit that's been GC'd in the cache, the adapter
  re-fetches via `git fetch origin <commit>`.
- A handful of dev instances ship binary test fixtures via the `test_patch`
  bucket (`{path, url}` shape). The adapter stages them at the in-repo
  path BEFORE applying `test_patch`, so the binary patch hunks resolve.
- `git apply --3way` falls back to `--reject` when 3-way fails; partially
  applied test diffs surface as `*.rej` files but are excluded from BAG's
  captured diff.
- The captured `bag.patch` is a diff against the harness's `swe-bench-mm-baseline`
  tag, **not** against `base_commit`, so it isolates BAG's edits from the
  staged test_patch.
