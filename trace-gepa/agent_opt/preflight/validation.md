# Preflight Decision Tree — Validation Report

**Dataset:** `trace-gepa/data/dataset_v2.jsonl` — 25,724 labelled records (good=22,303, bad=3,421; user_confirmed and user_corrected excluded).
**Date:** 2026-05-01
**Predicate set:** 7 deterministic predicates from proposal J. No LM calls. Total runtime 1.5 s for full pass (~17k rec/s).

## Headline numbers

Two configurations were measured:

- **FULL** — all 7 predicates active.
- **LIVE** — drops `edit_unique` and `file_was_read`. Both predicates depend on state that cannot be reconstructed from a replayed trace (the on-disk file at trace time, and the raw session log of prior `Read`s — `recent_actions` is a summary string list, not a session log). They remain valid in production; they are just unverifiable in offline replay.

| metric | FULL | LIVE |
|---|---:|---:|
| FP rate (good blocked)  | **10.03 %** | **1.42 %** |
| Recall   (bad blocked)  | **5.14 %**  | **1.61 %** |
| Precision               | 7.29 %      | 14.82 %     |
| Targeted-cat recall     | 7.25 %      | 7.25 %      |

Quality bars from the brief: **FP < 5 %** and **recall > 50 %**.

## Per-predicate (FULL run)

| predicate         | fires | TP  | FP   | precision |
|-------------------|------:|----:|-----:|----------:|
| `cmd_exists`      |   264 |  36 |  228 | 13.64 %   |
| `path_exists`     |   105 |  17 |   88 | 16.19 %   |
| `edit_unique`     |  1922 | 110 | 1812 |  5.72 %   |
| `file_was_read`   |  1110 | 119 |  991 | 10.72 %   |
| `skill_listed`    |     2 |   2 |    0 | **100 %** |
| `parallel_safety` |     0 |   0 |    0 | n/a       |
| `cmd_args`        |     0 |   0 |    0 | n/a       |

Best precision: `skill_listed` (100 %, only 2 fires but both are real). The two `Edit`-class predicates dominate volume but are hostile to replay validation (file mutated since record).

## Per-failure-category recall (records blocked / total)

| failure_category          | caught | total |
|---------------------------|-------:|------:|
| `bash_exit_nonzero`       |     19 |  2725 |
| (unknown / no category)   |    134 |   324 |
| `bash_timeout_141`        |      6 |   243 |
| `cancelled_parallel_batch`|      0 |   100 |
| `cmd_not_found_127`       |     11 |    17 |
| `retry_loop`              |      4 |    10 |
| `hallucinated_skill`      |      2 |     2 |

`bash_exit_nonzero` (80 % of `bad`) is the recall ceiling: these are semantic shell failures (failing tests, type errors, git conflicts) that **cannot** be detected without executing the command. No deterministic predicate can move the needle on them.

## Verdict: NOT-SHIP

Both bars fail.

- FP rate of 10 % in FULL mode is a replay artifact (Edit-class predicates penalised by post-trace file state) but the dataset is what we have to validate against; we cannot claim a number we cannot reproduce.
- Recall of 5 % is structural: 80 % of `bad` is uncatchable without execution. Even if every one of our predicates fired perfectly on the remaining 20 %, recall would top out near 17 %. The 50 % bar is unreachable with the predicate family proposed.

The 5 deterministic predicates that *do* work in live mode (`cmd_exists`, `path_exists`, `skill_listed`, `parallel_safety`, `cmd_args`) form a useful sub-1 % overhead veto layer, but they should be marketed as **cheap pre-flight assist**, not a recall-50 % gate. The proposal's own failure-category taxonomy needs richer predicates (or LM-assisted predicates) to clear a 50 % bar — which contradicts the "no-LM, deterministic" thesis.

Recommendation: ship as **shadow-mode** infra (log vetoes, do not block), then mine the veto log for new predicates over a week as the proposal Day-4 plan describes. Convert to enforcing only after observed FP < 1 % on live traffic.

## Five example records (predicates firing correctly)

1. `cc_v2_651dfb1e14_evt00094` — Bash `vp --help` → blocked by `cmd_exists`: "command not in PATH: vp" (label=bad, cat=`cmd_not_found_127`). True positive.
2. `cc_cc405b87_evt00818` — Skill `plan-graph` → blocked by `skill_listed`: "skill not in inventory: plan-graph" (label=bad, cat=`hallucinated_skill`). True positive, precision 100 %.
3. `cc_cc405b87_evt00891` — Skill `helm` → blocked by `skill_listed` (label=bad, cat=`hallucinated_skill`). True positive.
4. `cc_660da9c6_evt00514` — Edit `package.json` with `old_string="  \"name\": \"nkzw-app\","` → blocked by `edit_unique`: "old_string not found in file — Edit will fail" (label=bad). True positive (the prior model edit had already renamed it).
5. `cc_660da9c6_evt00528` — Edit `repack-ios-simulator.mjs` with no prior `Read` in `recent_actions` → blocked by `file_was_read` (label=bad). True positive: the harness rejects exactly this.

## Which BAG hook would consume this

The natural integration is BAG's tool-dispatcher pre-call hook (the same point where permission prompts and rate limits are evaluated, in `src/agent.ts` / `src/dispatch.ts`). Just before the dispatcher invokes the tool, call out via the CLI shim (`src/preflight-shim.ts`, scaffolded but not wired): pass the action and a context dict (`recent_actions` window, `available_skills` from the registry, current cwd). On `passed=false`, short-circuit with a synthetic tool result (`{ is_error: true, content: blocked_by.join("\\n") }`) and let the model retry — exactly the same affordance Claude already handles for permission denials. Every veto should be appended to `~/.claude/preflight.jsonl` for offline mining of new predicates. Run shadow-mode for one week before enforcement to confirm sub-1 % live FP.
