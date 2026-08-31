---
id: long-wait-poll-pattern
status: deprecated
order: 99
incident: future — placeholder for "agent waited synchronously for a long-running command and got killed by per-call timeout".
introduced: 2026-05-02
review_by: 2026-08-02
trigger: "applies when waiting for an external service (server boot, build, network sync) longer than perCallTimeoutSec"
merged_into: not-yet-active
---
Polling pattern for long waits (placeholder; not yet active in the seed prompt).

When you must wait for an external condition that may exceed `perCallTimeoutSec`, do NOT block the bash call. Instead:
1. Kick off the long task in the background (`nohup ... &` or `setsid ... &`).
2. Poll a sentinel file or process state in subsequent bash calls (each call short).
3. Each poll call should `cat /tmp/done.flag 2>/dev/null && echo READY || echo WAITING`.
4. Bail out after a hard cap (e.g. 30 polls) and submit with a comment if the condition never met.
