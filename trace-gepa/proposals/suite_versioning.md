# Suite Versioning — Round-9 Member #KK

## TLDR
- Adopt strict semver `MAJOR.MINOR.PATCH` on `suite.json.suite_version`; every commit touching `tasks/<suite>/v1/tasks.jsonl` or `scripts/verifiers/` must bump a level and append a `CHANGELOG.md` row keyed by `suite_hash`.
- MAJOR = task add/remove (changes `task_count`, old reports incomparable); MINOR = verifier_spec change (today's audit, old scores annotated, re-run encouraged); PATCH = harness/prompt/driver fix (reports stay comparable).
- Resolve the `tasks.jsonl` vs `tasks_audited.jsonl` ambiguity by promoting audited → canonical as **v1.1.0** (MINOR), tagging git, and auto-rerunning all model conditions within 24h; legacy v1.0.0 reports are watermarked but retained for diffing.
- Enforce in `validate_benchmark.py`: refuses to build if `suite_version != git tag --points-at HEAD`; reports embed `suite_version`; the leaderboard refuses to render mixed MAJORs and visually segregates mixed MINORs.

## Hypothesis
Without a contract for *when comparability breaks*, every verifier fix forks the leaderboard silently. Formal semver makes evolution legible: contributors know whether their PR is a typo fix or a benchmark-breaking change before it lands, and consumers know whether to trust score deltas across commits.

## Rules (canonical table)
| Level | Trigger | suite_hash | Old reports |
|-------|---------|------------|-------------|
| MAJOR | task_count changes; task IDs renumbered | rebuild | incomparable, archived |
| MINOR | verifier_spec edited; rubric tightened | rebuild | annotated `legacy-verifier`, re-run within 24h |
| PATCH | harness, prompt template, driver, logging | unchanged | fully comparable |

The `suite_hash` is the SHA-256 of `tasks.jsonl || verifier_spec.json` sorted by task_id; PATCH-only changes leave it byte-identical, which is the comparability invariant.

## Enforcement
1. `scripts/validate_benchmark.py` aborts on `suite_version` mismatch with `git tag -l --points-at HEAD`.
2. CI runs `compute_suite_hash.py` and diffs against `suite.json`; mismatch fails the build.
3. Report writers stamp `suite_version` + `suite_hash` into every JSON; the leaderboard renderer groups by MAJOR and warns on MINOR drift.
4. `CHANGELOG.md` entries follow Keep-a-Changelog with a mandatory `### Suite-impact: MAJOR|MINOR|PATCH` line.

## Migration (today)
1. `git mv tasks_audited.jsonl tasks.jsonl`, recompute suite_hash.
2. Bump `suite_version` to `1.1.0`, `git tag v1.1.0`, append CHANGELOG entry listing the 16 fixed verifiers.
3. Trigger `rerun_all_conditions.py --since v1.1.0`; old v1.0.0 reports gain a `legacy=true` flag.

## Self-critique
Rigid semver risks bureaucracy — a verifier typo fix that doesn't change any pass/fail outcome still triggers MINOR; mitigation is a `--dry-run-verifiers` gate that downgrades to PATCH if the rerun produces byte-identical scores across all archived reports.
