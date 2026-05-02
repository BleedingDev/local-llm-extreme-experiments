# Final verdict (with corrected verifier)

**Date:** 2026-05-02
**Bench:** `data/benchmark_tasks_full.jsonl` (175 tasks: 105 trace-derived + 70 synthetic)
**Verifier:** post-FIX1 + INV1 (DSL parser handles `pattern_or_command`, `no_repeat` matches `input_excerpt`)
**Task LM:** `claude-opus-4-7`
**Concurrency:** ThreadPool max_workers=8
**Wallclock:** ~44s per 175-task run; ~$1-2 estimate per pair

## Headline

**Seed prompt narrowly beats the GEPA-optimised prompt** under the corrected verifier: 0.303 vs 0.291 (Δ -0.012). Statistically tied at this n. The earlier reported "0.467 → 0.767 (+64% relative)" lift was an artifact of the broken verifier (which trivially passed any parseable JSON for `structural_json` tasks).

## Per-category results

| category | n | seed | optimised | Δ |
|---|---:|---:|---:|---:|
| debugging | 20 | 0.850 | 0.850 | 0.000 |
| edit_safety | 38 | 0.447 | **0.500** | **+0.053** |
| recovery | 19 | **0.421** | 0.368 | -0.053 |
| tool_routing | 39 | 0.179 | 0.179 | 0.000 |
| planning | 19 | **0.105** | 0.000 | -0.105 |
| path_grounding | 24 | **0.083** | 0.042 | -0.041 |
| command_synthesis | 16 | 0.000 | 0.000 | 0.000 |
| **overall** | **175** | **0.303** | **0.291** | **-0.012** |

## Per-difficulty

| difficulty | n | seed | optimised | Δ |
|---|---:|---:|---:|---:|
| easy | 47 | **0.319** | 0.298 | -0.021 |
| medium | 48 | 0.292 | 0.292 | 0.000 |
| hard | 80 | 0.300 | 0.287 | -0.013 |

## Why the optimised prompt regressed

GPT-5.5 xhigh consultation (`bench/specialist_consultation.md`) gave the most likely explanation:
> For single-step tool selection, extra reasoning increases action entropy. The model over-deliberates, invents contingencies, chooses a "more generally sensible" action that strict verifiers penalise.

The optimised prompt grew from 236 chars (seed) to 2544 chars with 11 explicit rules. Under the correct verifier, the extra rules push the model toward more elaborate but less verifier-aligned outputs.

## Confounds (per GPT-5.5 specialist + INV1 findings)

1. **`command_synthesis` (16 tasks): 0.000 across all candidates.** INV1 confirmed these are pathological — `available_tools` for these tasks doesn't include the canonical shell tool the verifier expects. **Recommend dropping or rebuilding these tasks.**
2. **`path_grounding` (24 tasks): max 0.083.** Same issue — synthetic tasks offer codex tools (`exec_command`, `spawn_agent`) but verifier requires `{Bash, Glob, Grep}`.
3. **`planning` (19 tasks): max 0.105.** Likely too few `available_tools` for the planner contract.

If we exclude the 3 confounded categories (59 tasks total) and recompute on the 116 clean tasks:
- seed: 0.422
- optimised: 0.431
- **Δ +0.009 — still essentially tied.**

## Recommendation

**Roll back `latest` to the seed prompt** (effectively: set `BAG_DISABLE_OPTIMIZED_PROMPT=1` or unlink the symlink). The optimised prompt is not delivering net value under correct measurement and costs ~+10× tokens per BAG planner call (2544 vs 236 chars).

Or keep it but be honest: the Wave-2/Wave-3 +64%-relative claim was inflated; the real number is ~0%.

## What's NEXT (highest-leverage moves, per GPT-5.5)

1. **Task-validity preflight** — for every task, mechanically prove that at least one action using `available_tools` can satisfy the verifier. Block eval on invalid tasks. **This is the single highest-leverage improvement.**
2. **Split confounded categories** — `command_synthesis` (shell-construction vs API-arg) and `path_grounding` (locating vs using paths) need finer typing.
3. **Don't default to xhigh reasoning** on single-step benchmarks — keep it for multi-step / planning evals.
4. **Re-optimise** with corrected verifier feedback (the GEPA loop got reflective signal from a verifier that was broken — re-run optimisation should produce a meaningfully different artefact).

## Files

- `bench/results/full_eval/opus_seed.json` — seed result, 175 tasks
- `bench/results/full_eval/opus_optimized.json` — optimised result, 175 tasks
- `bench/specialist_consultation.md` — GPT-5.5 xhigh critique
- `bench/zero_cat_investigation.md` — INV1's diagnosis of the 3 zero categories
- `data/benchmark_tasks_full.jsonl` — 175 tasks
- `bench/verifiers/tier1_regex.py` — post-FIX1, post-INV1 verifier
