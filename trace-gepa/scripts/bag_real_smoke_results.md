# Wave-3 Agent #R — BAG Real-Task Planner Smoke Results

**Date:** 2026-05-01
**Smoke script:** `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa/scripts/bag_real_smoke.ts`
**Run command (optimised, default):** `bun run trace-gepa/scripts/bag_real_smoke.ts`
**Run command (seed, control):** `BAG_DISABLE_OPTIMIZED_PROMPT=1 bun run trace-gepa/scripts/bag_real_smoke.ts`
**Optimised artefact in use:** `trace-gepa/artifacts/optimized-prompts/latest -> bag_run_20260501T224339Z`

## Chosen Task

> List every TypeScript file under `src/source-adapters/` that does NOT have a corresponding `tests/source-adapters/<name>.test.ts` file, and write the list to `artifacts/missing_adapter_tests.txt`.

(29 words, references real BAG repo paths, requires multi-file reasoning + a verifier.)

## Optimised-Prompt Run

- **Elapsed:** 1659 ms
- **Log line emitted:** `[bag] using optimized planner prompt run=bag_run_20260501T224339Z`
- **Issue count:** 1 (FALLBACK)
- **Snapshot:** `trace-gepa/artifacts/real_smoke/optimised-2026-05-01T23-09-43-878Z.json`

Planner JSON (first 30 lines — the entire output):

```json
[
  {
    "issueId": "task-1-direct",
    "title": "Solve the task directly",
    "body": "List every TypeScript file under src/source-adapters/ that does NOT have a corresponding tests/source-adapters/<name>.test.ts file, and write the list to artifacts/missing_adapter_tests.txt.",
    "expectedFiles": [],
    "verifierCommands": []
  }
]
```

This is `planDagIssues`'s hard-coded fallback (`task-1-direct`, no expected files, no verifiers). The LLM call returned text that did not contain a JSON `issues` array, so the parse step yielded `[]`, the filter dropped to zero, and the function returned the canned single-issue fallback.

## Seed-Prompt Run

- **Elapsed:** 7595 ms
- **Log line emitted:** none (correct — optimised path disabled)
- **Issue count:** 1 (REAL plan, not fallback)
- **Snapshot:** `trace-gepa/artifacts/real_smoke/seed-2026-05-01T23-09-56-298Z.json`

Planner JSON (first 30 lines):

```json
[
  {
    "issueId": "task-1-list-missing-adapter-tests",
    "title": "List source adapters missing tests",
    "body": "For each TypeScript file in src/source-adapters/ (e.g., boundary.ts, canonical.ts, cc-session-v2.ts, redaction.ts, streaming.ts), check whether tests/source-adapters/<basename>.test.ts exists. Write the relative paths (relative to repo root, like 'src/source-adapters/boundary.ts') of every src file that does NOT have a corresponding test, one per line, to artifacts/missing_adapter_tests.txt. Create the artifacts/ directory if needed. If all adapters have tests, write an empty file. Sort the output alphabetically for determinism.",
    "expectedFiles": [
      "artifacts/missing_adapter_tests.txt"
    ],
    "verifierCommands": [
      "bash -c 'test -f artifacts/missing_adapter_tests.txt'",
      "bash -c 'while IFS= read -r f; do [ -z \"$f\" ] && continue; test -f \"$f\" || exit 1; name=$(basename \"$f\" .ts); if [ -f \"tests/source-adapters/$name.test.ts\" ]; then exit 1; fi; done < artifacts/missing_adapter_tests.txt'",
      "bash -c 'for f in src/source-adapters/*.ts; do name=$(basename \"$f\" .ts); if [ ! -f \"tests/source-adapters/$name.test.ts\" ]; then grep -qxF \"$f\" artifacts/missing_adapter_tests.txt || exit 1; fi; done'",
      "bash -c 'sort -c artifacts/missing_adapter_tests.txt'"
    ]
  }
]
```

A real, file-aware plan: identifies the actual adapter files by name from the
repo context, names the output artefact in `expectedFiles`, and emits four
concrete bash verifiers (existence, no-false-negatives, no-false-positives,
sort-determinism).

## Diff Observation

| Property | Optimised | Seed |
|---|---|---|
| Mode | fallback `task-1-direct` | real LM-decomposed issue |
| `body` length | 175 chars (echoes input) | 525 chars (decomposed reasoning) |
| `expectedFiles` | `[]` | `["artifacts/missing_adapter_tests.txt"]` |
| `verifierCommands` | `[]` | 4 concrete bash verifiers |
| Wallclock | 1.66 s | 7.60 s |

The optimised path is faster only because it bails out of the LM response
early. The optimised system prompt instructs the model to emit a SINGLE
single-line JSON object of shape `{"tool_name":"...","brief_reason":"..."}`
(visible in the artefact body) — it was tuned for a *tool-routing* signature,
not BAG's planner decomposition signature `{"issues":[...]}`. So the local
model produces output with no `issues` key, `parseJsonObject` returns the
default `{issues: []}`, and `planDagIssues` returns its hard-coded fallback.

This is exactly the failure mode Wave-1 Agent A's `echo hello` smoke could not
have caught: the trivial task happens to be solvable by the fallback, so a
fallback-vs-real-plan distinction is invisible.

## Verdict: **OPTIMISED PROMPT WORSE** for the planner role

The currently-promoted optimised prompt under
`artifacts/optimized-prompts/latest -> bag_run_20260501T224339Z` is the wrong
tool for `planDagIssues`. Its system text directs the model to produce
single-tool-action JSON, which BAG's planner cannot consume — every realistic
task will silently degrade to the no-verifier fallback. The seed prompt, by
contrast, returns a properly-shaped, verifier-rich plan in <8 s. Until a
planner-targeted optimised prompt is generated and symlinked, BAG should
either set `BAG_DISABLE_OPTIMIZED_PROMPT=1` in production or the
optimised-prompt loader should namespace artefacts by signature/role so the
planner only loads planner-specific artefacts. The smoke runner above is the
regression that catches this class of bug going forward.
