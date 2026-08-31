# BAG x Aider Polyglot — Onboarding

This document tells a fresh operator how to run the [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) benchmark against BleedingAgent (BAG). The adapter lives at `bench/aider_polyglot/`.

## What this benchmark stresses

- **Cross-language edit fidelity.** 225 Exercism problems across C++, Go, Java, JavaScript, Python, Rust. For each problem BAG receives a stub source file plus the problem description and must implement the missing logic.
- **Two-shot self-correction.** When the unit tests fail on the first attempt, the harness re-prompts BAG with the raw test output (mirroring Aider's own `tries=2` loop). This benchmarks BAG's pre-submit self-check **and** its retry-with-context behavior.
- **Non-Python edit competence.** BAG is most heavily exercised on Python; this benchmark forces parity in five other languages.

## Hard contracts of the adapter

- **Generic.** The adapter does not contain per-problem patches or task-specific code. It treats every Exercism problem the same way: stage it, send it as one ACP task, run tests.
- **No BAG modifications.** BAG is invoked through the existing `scripts/bag_acp_run.ts` entrypoint. The adapter never edits `src/`.
- **Two-shot uses BAG's natural retry loop.** The retry is a separate ACP session whose task is the test failure output plus a fix-the-code instruction. BAG's internal self-evaluation, planning, and tool loops run unmodified inside each attempt.
- **License.** Aider Polyglot and Aider itself are Apache-2.0; the prompt strings (`INSTRUCTIONS_ADDENDUM`, `TEST_FAILURES`) are copied verbatim with attribution in `bench/aider_polyglot/prompts.py`.

## Required toolchains

The adapter shells out to language-native test runners locally on the operator machine. Install whichever toolchains correspond to the language subset you care about:

| Language    | Tool needed                              | Verify                              |
|-------------|------------------------------------------|-------------------------------------|
| Python      | `pytest` in `.venv`                      | `.venv/bin/pytest --version`        |
| Rust        | `cargo` (1.70+)                          | `cargo --version`                   |
| Go          | `go` (1.21+)                             | `go version`                        |
| JavaScript  | `node` 22 + `npm`                        | `node --version`, `npm --version`   |
| C++         | `cmake` + `make` + a working C++ toolchain | `cmake --version`, `make --version` |
| Java        | JDK 17+ and bundled gradlew              | `java --version`                    |

On macOS Apple Silicon, all of the above are typically resolvable via Homebrew or `proto`. Java tests use the per-problem `gradlew` wrapper (which downloads the right Gradle version on first run; expect 1–2 minutes of warm-up the first time the harness hits a Java problem).

The adapter also needs:

- Node + tsx (already installed at `node_modules/.bin/tsx`)
- `.env` with `ANTHROPIC_AUTH_TOKEN=…`
- `bag.config.json` at the repo root (already present)

## Reproducible smoke run

```bash
# Smoke: 5 problems, one per language (round-robin), default dag-tools mode.
.venv/bin/python bench/aider_polyglot/run.py -n 5
```

Outputs land under `bench/aider_polyglot/results/<UTC stamp>/`:

```
results/<stamp>/
  plan.json                   # selected problems + invocation args
  records.json                # per-problem records (incremental)
  scoreboard.json             # aggregate pass-rate (incremental)
  summary.txt                 # human-readable scoreboard
  workspaces/                 # the per-problem dirs BAG edits in (grouped by language)
    python/affine-cipher/
    cpp/all-your-base/
    ...
  attempts/
    python__affine-cipher__attempt1/
      task.txt                # exact prompt sent to BAG
      bag.log                 # bag_acp_run.ts stdout/stderr stream
      bag-summary.json        # ACP trajectory + token counts
      test-output.log         # raw unit-test output (cleaned of timing noise)
    python__affine-cipher__attempt2/   # only present if attempt 1 failed
      ...
```

The same scoreboard is appended after each problem completes, so a partial run (e.g. interrupted by Ctrl-C) still has a well-formed summary on disk.

## Other useful invocations

```bash
# 8 problems sampled only from python and rust:
.venv/bin/python bench/aider_polyglot/run.py -l python,rust -n 8

# Hand-pick specific problems (slugs are <language>/<problem-name>):
.venv/bin/python bench/aider_polyglot/run.py \
  --slugs python/affine-cipher,rust/bowling,go/alphametics

# Disable the second-shot retry (matches a `tries=1` Aider configuration):
.venv/bin/python bench/aider_polyglot/run.py -n 10 --no-retry

# Stage problems and dump prompt files but do NOT actually call BAG (free):
.venv/bin/python bench/aider_polyglot/run.py -n 5 --dry-run
```

## Per-problem cost estimate

Cost depends entirely on the BAG mode and master/local model split. Defaults
mirror `bench/bag_agent/agent.py`: master = `claude-opus-4-7`, local =
`claude-haiku-4-5-20251001`, `bag_mode = dag-tools`.

Empirical envelope from `bench/jobs/2026-05-02__*` polyglot trials (single
problem, two-shot):

| metric                        | typical | upper |
|-------------------------------|---------|-------|
| wall time                     | 60–180s | 600s  |
| BAG turns                     | 8–20    | 32    |
| master prompt tokens          | ~25 k   | ~80 k |
| master completion tokens      | ~3 k    | ~10 k |
| local prompt tokens           | ~40 k   | ~150 k|
| local completion tokens       | ~5 k    | ~20 k |
| dollar cost per problem (est.)| $0.10–$0.30 | ~$1.00 |

A 225-problem full sweep at the upper estimate is therefore ≲ ~$225 with the
default cost-split config; a 5-problem smoke is well under $2.

## Interpreting `summary.txt`

```
BAG x Aider Polyglot — N problems
  pass rate         : X% (P/N)         <- final pass rate (any of two shots passed)
  pass rate attempt1: Y% (Q/N)         <- pass rate WITHOUT retry (Aider 'tries=1')
  per-language: ...
```

The gap between `pass rate` and `pass rate attempt1` is the **retry-utility** for BAG: it tells you how often the second-shot loop with raw test output rescued an attempt. A large gap means BAG benefits a lot from the test-failure feedback channel; a small gap means BAG's first-shot self-check + planning is already strong.

## Known gotchas

- **First Java problem is slow.** `gradlew` downloads the toolchain. Subsequent Java problems reuse `~/.gradle/caches`.
- **JavaScript tests** rely on `npm install`; the harness will lazily install per problem unless you pre-stage a shared `node_modules` and pass `--npm-install-dir` (not yet wired through the CLI; extend `run_one_problem` if you need it).
- **C++ problems** need cmake. CMakeLists.txt comes from the upstream repo; we do not regenerate it.
- **Rust target dirs.** Each Rust problem produces a `target/` dir inside its workspace; the harness does not clean them between runs, so disk usage grows linearly. Delete `bench/aider_polyglot/results/<stamp>/workspaces/rust/*/target` if needed.
- **Determinism.** Without `--slugs`, the round-robin selection is alphabetical per language, then by language-name alpha. Re-running with the same `-n` and `-l` selects the same problems.
