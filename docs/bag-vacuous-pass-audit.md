# BAG Vacuous-Pass Audit

A small post-hoc telemetry layer that detects "vacuous passes" — trials where
the verifier awarded full credit (`reward == 1.0`) but the agent never
actually executed work — across harbor-style benchmark runs.

The bug being audited is **benchmark-side**, not BAG-side. The forensic
deep-dive in `docs/bag-successful-runs-deep-dive.md` showed that the
qemu-startup task on `terminal-bench-sample` passes its own
`tests/test_outputs.py::test_version` against the base-image qemu binary
regardless of whether the agent ever started qemu. When BAG times out
mid-prompt, the verifier still records `reward=1` and the trial inflates the
headline win rate. We don't change BAG's reward semantics; we add a
side-channel audit so the inflation is visible.

## Files

| Path | Purpose |
| --- | --- |
| `bench/audit/vacuous_pass.py` | Pure-Python detector + rollup module |
| `bench/audit/test_vacuous_pass.py` | Unit tests (clean win / vacuous / real loss / older-run / bare) |
| `scripts/bag_audit_run.py` | CLI: `python scripts/bag_audit_run.py bench/jobs/<job>` |
| `bench/bag_agent/agent.py` | Calls `audit_trial` + `write_audit_jsonl` at end of every BAG trial |

## Why vacuous passes happen

The harbor verifier in `terminal-bench-sample/qemu-startup` is satisfied by
the base image's preinstalled qemu — `qemu-system-x86_64 --version`
succeeds before any agent action. When the agent prompt times out
(`stopReason: error:prompt timeout after 880000ms`) BAG never finishes a
turn, never starts qemu, never writes any artefact. Harbor still runs the
verifier against whatever state the container is in, the version check
passes, reward=1 lands. Same shape would bite any task whose verifier
probes invariants of the base image rather than artefacts the agent
produced.

## How the detector works (structural-only)

For every trial directory the detector reads (and never raises on missing
files):

1. **`verifier/reward.txt`** — canonical reward, fallback to
   `result.json::verifier_result.rewards.reward`.
2. **`agent/bag-traces/.bag/runs/<latest>/autonomous-summary.json`** —
   `turnsUsed` and `toolCallsExecuted`. The presence + non-zero counts in
   this file is the canonical positive signal that BAG completed at least
   one autonomous turn.
3. **`agent/bag-acp-summary.json::stopReason`** — fallback signal for
   older runs that pre-date the trace-tarball stash. A clean stop
   (`end_turn`, `submitted`, `max_turns`) is treated as evidence of real
   work; an error stop (`startswith("error:")`, `"fatal:"`, `"abort"`) is
   treated as the vacuous shape.

Classification:

```
agent_did_real_work :=
    (turnsUsed > 0 AND toolCallsExecuted > 0)             # autonomous-summary
    OR (no autonomous-summary AND stopReason is clean)    # ACP fallback

vacuous_pass := reward == 1.0 AND NOT agent_did_real_work
```

Trials whose agent directory contains no BAG telemetry at all (e.g.
`claude-code` or other harbor agents) skip the vacuous-pass classification
unless the entire `agent/` directory is empty — the detector quiets itself
for benchmarks whose telemetry shape it cannot reason about.

The detector deliberately **never** matches task names. The qemu-startup
column of the BAG leaderboard is the hot spot today, but the same
structural shape would catch any future task whose verifier rubber-stamps
base-image state.

## Running the audit

### One-shot CLI on existing job directories

```bash
# pretty-printed
python scripts/bag_audit_run.py bench/jobs/2026-05-02__09-45-38/

# machine-readable: full rollup + per-trial audits
python scripts/bag_audit_run.py --json bench/jobs/2026-05-02__09-45-38/

# one JSON line per trial, suitable for jq
python scripts/bag_audit_run.py --jsonl bench/jobs/2026-05-02__09-45-38/

# audit every recent BAG run at once
python scripts/bag_audit_run.py bench/jobs/2026-05-02__*
```

### Live, per-trial emission

The harbor adapter at `bench/bag_agent/agent.py` now calls
`audit.vacuous_pass.write_audit_jsonl(<job_dir>, audit)` at the end of every
trial's `run()`. This appends one JSON line per trial to
`bench/jobs/<job>/audit.jsonl`. The reward field will be `null` in this
live emission because the verifier hasn't run yet at that point —
downstream consumers should treat `audit.jsonl` as "agent-side evidence
captured live" and recompute the canonical `reward` / `vacuous_pass` from
disk via `audit_trial()` after the verifier completes.

### Programmatic API

```python
from audit.vacuous_pass import audit_trial, audit_job, effective_score

# Per-trial audit
audit_trial("bench/jobs/2026-05-02__02-29-26/qemu-startup__SMinaQE")
# {"trial": "qemu-startup__SMinaQE", "reward": 1.0,
#  "agent_did_real_work": false, "vacuous_pass": true,
#  "stop_reason": "error:prompt timeout after 880000ms",
#  "evidence": {"turns_used": 0, "tool_calls": 0,
#               "trace_files_present": ["routing-decision.json"], …}}

# Whole-job rollup
effective_score("bench/jobs/2026-05-02__02-29-26")
# {"n_trials": 10, "n_real_wins": 3, "n_vacuous_passes": 1, "n_real_losses": 6,
#  "headline_score": 0.40, "effective_score": 0.30, "delta": -0.10,
#  "vacuous_trials": ["qemu-startup__SMinaQE"]}
```

## Retrospective audit of last 49 BAG runs

Detector applied to every `bench/jobs/2026-*` directory (49 jobs, 367
scored trials):

| metric | value |
| --- | --- |
| jobs scanned | 49 |
| trials scored | 367 |
| real wins | 226 |
| **vacuous passes** | **7** |
| real losses | 134 |
| headline mean reward | 0.6349 |
| **effective mean reward** | **0.6158** |
| **delta** | **-0.0191 (-1.91 %)** |

Per-family vacuous counts: **qemu-startup: 7**, all other families: 0. The
detector confirms the deep-dive's narrative — the bug is concentrated
entirely in qemu-startup — and corroborates the headline finding that ~3 %
of BAG's reported win rate on `terminal-bench-sample` is fake. The exact
trials flagged are:

```
2026-05-01__19-51-36/qemu-startup__yhuTV8u
2026-05-02__02-29-26/qemu-startup__SMinaQE
2026-05-02__04-53-37/qemu-startup__aJKLduv
2026-05-02__05-51-57/qemu-startup__TqWjYVx
2026-05-02__06-29-50/qemu-startup__fCYzB9P
2026-05-02__08-06-13/qemu-startup__CiagMUX
2026-05-02__09-24-29/qemu-startup__WSh4GkT
```

Every one stops with `stopReason: error:prompt timeout after 880000ms`,
emits no `autonomous-summary.json`, leaves only `routing-decision.json`
under `bag-traces/.bag/runs/<latest>/`, and is rubber-stamped reward=1 by
the harbor verifier.

## Recommendation: telemetry first, verifier patch second

The audit telemetry is sufficient for the immediate need — to **never
misreport** these passes as real wins. The downstream consumers
(`scripts/bag_audit_run.py`, the per-trial `audit.jsonl`, and the
`effective_score` rollup) give us a `effective_reward` we can use as the
canonical headline metric, and the `vacuous_pass` flag gives us a way to
spot regressions in BAG's qemu handling that would otherwise be invisible.
**No further change to BAG `src/` or to the harbor adapter's reward
logic is needed for this PR.**

Patching the harbor verifier is **out of scope**. It is a separate fix
that requires either (a) upstreaming a stricter `tests/test_outputs.py`
into `terminal-bench-2-0-sample` so `test_version` only passes when the
agent has actually started a fresh qemu instance reachable on
`telnet 127.0.0.1 6665`, or (b) shimming a per-trial pre-verifier check
in our harbor adapter that fails-closed when `autonomous-summary.json` is
missing on tasks whose verifier is known to be base-image-stable. Either
fix would correct the score at source; the audit telemetry would still
remain useful as a cross-check.

The recommended next step is to wire `effective_score` into the
job-level summary that `scripts/run_*` produce, so headline numbers
quoted in subsequent reports automatically use the de-vacuoused metric.
