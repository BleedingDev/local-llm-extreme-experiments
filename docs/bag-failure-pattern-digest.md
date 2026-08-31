# BAG failure pattern digest — read-only meta-analysis

**Scope.** All trials under `bench/jobs/` where `agent/bag-acp-summary.json` exists
(BAG-driven trials). Opus-direct trials and trials with no result.json are excluded.

**Counts.** 94 BAG trials total: **44 wins** (1.0), **50 losses** (0.0), **0 hard
exceptions**. 5 of the 50 losses are ACP `stopReason="error:Internal error"`
(BAG died before producing artifacts); the rest are clean `end_turn` losses.

**Run map (BAG-bearing).**
- runs 1–4 (`12-57-44`, `13-15-12`, `16-24-37`, `18-03-30`, `18-55-46`) → `acp-code`
  command (manifest present, npm-style tools `typecheck`, `test`, no real bash).
  These four runs alone account for **34 of the 50 losses**.
- runs 5–8 (`19-51-36`, `21-09-19`, `21-43-38`, `22-41-38`) → `acp-auto`
  routing-decision implied (no manifest, no traces dir yet).
- run 9 (`23-11-16`, `23-31-04`, `23-32-30`, `23-36-57`) → `acp-auto` with
  `agent/bag-traces/` extracted; `routing-decision.json` available.

---

## Top failure clusters (by count)

### 1. acp-code "no real bash" cluster — 28 occurrences across runs 1–4
For five tasks (build-cython-ext, qemu-alpine-ssh, qemu-startup, sqlite-with-gcov,
polyglot-c-py "no /app/polyglot") the agent in runs 1–4 only ran npm wrappers
like `npm run typecheck` / `npm test` and never executed the actual setup script
on the container. The verifier then sees a missing artifact.

- **Verifier signatures** (representative):
  - `FileNotFoundError: [Errno 2] No such file or directory: 'sqlite3'`
  - `FileNotFoundError: [Errno 2] No such file or directory: '/tmp/data.txt'`
  - `subprocess.CalledProcessError: ['sshpass', '-p', 'password123', 'ssh', ...] returned non-zero exit status 255`
  - `ModuleNotFoundError: No module named 'cinvariants'`
  - `FileNotFoundError: '/app/polyglot'`
- **Affected trials (sample):** every loss in `13-15-12`, `16-24-37`, `18-03-30`,
  `18-55-46` for those tasks (28 trials). Examples: `2026-05-01__13-15-12/qemu-startup__hMJrqGy`,
  `2026-05-01__16-24-37/sqlite-with-gcov__TYxX2Nz`, `2026-05-01__18-55-46/build-cython-ext__B9jZhFp`,
  `2026-05-01__18-03-30/qemu-alpine-ssh__VHC2d3K`.
- **Last bash calls before submit:** typically a single `npm` invocation with args
  `["run", "typecheck"]` or `["run", "test"]`, or zero terminal calls at all
  (8 trials with `terminalCreate==0`).
- **Suggested mitigation:** Already largely fixed by switching from `acp-code`
  to `acp-auto`. Lock the legacy `acp-code` mode out of terminal-bench: in
  `bag_agent.agent:BagAgent` (or wherever the mode is selected), force `acp-auto`
  for any task whose verifier touches a host filesystem path that BAG hasn't
  already produced.

### 2. polyglot-c-py: cleanup miss (`cmain` not deleted) — 4 occurrences
- **Verifier signature:** `AssertionError: Expected only main.py.c, found: ['main.py.c', 'cmain']`
- **Affected trials:** `19-51-36/polyglot-c-py__ixWQGkc`, `21-09-19/polyglot-c-py__dveNqWQ`,
  `21-43-38/polyglot-c-py__RHtUSKX`, `22-41-38/polyglot-c-py__yEFw5yw`.
- **Last bash calls before submit:** in every case BAG compiled `cmain`, sanity-tested
  Python and C output, then submitted without removing `cmain`. Example tail
  for `22-41-38/polyglot-c-py__yEFw5yw`:
  ```
  > cd /app/polyglot && gcc main.py.c -o cmain && ./cmain 10 ...
  > echo BAG_TASK_COMPLETE
  ```
- **Suggested mitigation:** The agent system prompt already mentions cleanup
  (per task notes), but BAG keeps producing the artifact in-tree. Two options:
  (1) compile to `/tmp/cmain` (the run that did this — `16-24-37/polyglot-c-py__eKu4kew`
  — also lost but for unrelated gcc error); (2) end every polyglot bash session
  with a hard-coded `find /app/polyglot -maxdepth 1 -type f ! -name 'main.py.c' -delete`.
  Add an explicit pre-submit checklist hook for the polyglot-shape: enumerate
  files in the workdir and abort submit if any non-source artifact exists.

### 3. configure-git-webserver: end-to-end never green — 5 occurrences
- **Verifier signature:** `AssertionError: Did not pass test` with verify.sh
  output `❌ TEST FAILED: Web server returned HTTP 000` (4 cases) or `HTTP 404`
  (1 case in `23-11-16/configure-git-webserver__V8v2hGJ`).
- **Affected trials:** `13-15-12/configure-git-webserver__q3u2FkN` (internal error),
  `16-24-37/configure-git-webserver__kRYkudi` (0 bash), `18-03-30/configure-git-webserver__YKNzz35`
  (3 npm-shaped calls), `18-55-46/configure-git-webserver__FAcTJRp` (0 bash),
  `23-11-16/configure-git-webserver__V8v2hGJ` (25 bash, real attempt).
- **Last bash calls before submit (run 8 case):** `chown -R user:user /git/server`,
  then `ls -la /app`, then `BAG_TASK_COMPLETE` — no `curl http://localhost/...`
  ever ran, so BAG never observed HTTP 404 itself.
- **Suggested mitigation:** Force the agent to run the **literal** verify.sh
  loop (`bash /tests/verify.sh` is in the test, but the BAG can simulate by
  `curl http://localhost/hello.html` and `git clone git@localhost:/git/server`)
  before issuing `BAG_TASK_COMPLETE`. The `acp-auto` "monolithic-complex →
  tools" mode does not gate submission on a self-verification step. Add a hard
  gate: for tasks classified `monolithic-complex` or `compositional`, require
  the agent to produce a "self-verification stdout" containing `TEST PASSED` or
  the equivalent task-specific success token.

### 4. fix-code-vulnerability: incomplete fix or wrong CWE — 6 occurrences
Two sub-clusters:
- **(a) `_hkey` ValueError-not-raised regression (4 trials, runs 1–4):**
  signature `AssertionError: ValueError not raised by append` /
  `AssertionError: _hkey ...`. BAG only modified the report, not the function;
  or modified the wrong path. Affected: `13-15-12/fix-code-vulnerability__TnaU9PM`,
  `16-24-37/fix-code-vulnerability__dNt6rs3`, `18-03-30/fix-code-vulnerability__VPso2BE`,
  `18-55-46/fix-code-vulnerability__3QTMWyL`.
- **(b) wrong CWE id (`cwe-20` instead of `cwe-93`) — 1 trial:**
  `21-43-38/fix-code-vulnerability__WSapBfm`. BAG ran `pytest -rA -q` 30 bash
  calls before submit; saw `cwe-20` in jsonl but didn't reconcile against the
  test's expected list.
- **(c) `report.jsonl` missing (1 trial, internal error):**
  `22-41-38/fix-code-vulnerability__5x4SaSt` (runaway sed/python on bottle.py
  triggered an internal error before report was written).
- **Suggested mitigation:** When the task instruction quotes `pytest -rA` BAG
  should run `pytest -rA` and parse the *full* output, then match every assertion
  failure against its current edit set. Concretely: turn the existing self-eval
  threshold (`config.policy.selfEvalThreshold = 0.78`) into a reward-aware gate
  for any task containing the substring `pytest`.

### 5. qemu-alpine-ssh: wrong-kernel host leak — 2 occurrences (run 7+)
- **Verifier signature:** `AssertionError: '6.6.4-1-lts' in '6.19.13-orbstack-...'`
- **Affected trials:** `21-43-38/qemu-alpine-ssh__ZszdmbU` (93 bash),
  `22-41-38/qemu-alpine-ssh__E9bTFFQ` (55 bash).
- **What happened:** BAG installed sshpass and connected to *port 2222 on the
  Docker host* — the orbstack-vm. It never actually booted the QEMU guest, so
  uname -r returns the host kernel.
- **Last bash:** chained `sshpass -p password123 ssh -p 2222 ... uname -r`
  succeeds (with the wrong kernel), so BAG happily submits.
- **Suggested mitigation:** Add to the agent prompt a fingerprint check: if
  task mentions Alpine and `uname -r` returns anything containing `orbstack` or
  `linuxkit`, abort submit. Generally for VM tasks, require BAG to assert the
  expected guest version string before completing.

### 6. log-summary-date-ranges: off-by-N row counts — 2 occurrences
- **Verifier signature:** `AssertionError: Expected row ['today', 'ERROR', '370'], got ['today', 'ERROR', '414']`
- **Affected trials:** `16-24-37/log-summary-date-ranges__i7G56NP`,
  `21-43-38/log-summary-date-ranges__jTut424`.
- **Cause:** the date range "today" overlaps with "last_30_days"; BAG's
  aggregator counts a row in both ranges where the test expects exclusive bins.
- **Mitigation:** prompt the auto-router to expose this as a "spec-clarity"
  warning and require a sample-driven sanity check before submit.

### 7. chess-best-move "missing move.txt" cluster — 5 occurrences (runs 1–4)
- **Verifier signature:** `FileNotFoundError: '/app/move.txt'`.
- All 5 are runs 1–4 in `acp-code` mode where the only "command" was npm
  `typecheck`/`test` or zero terminal calls. Same root cause as cluster 1.

### 8. chess-best-move "File is wrong" — 1 occurrence
- **Verifier signature:** `assert ['e2e4'] == ['e2e4', 'g2g4']`
  (test_outputs:25 `assert sorted(move) == sorted(["g2g4", "e2e4"]), "File is wrong"`).
- Trial `23-11-16/chess-best-move__3hGaYQK`. BAG found e2e4 but missed g2g4
  (both are forced-mate-in-1). Last bash before submit was a board-attacker
  enumeration python script, then submit. BAG only emitted `e2e4`.
- **Mitigation:** for any task instruction containing the phrase "If there are
  multiple … print them all", inject a system-prompt rule: "After identifying
  the first solution, exhaustively re-search for additional solutions before
  submit."

### 9. sqlite-with-gcov: gcov not enabled — 1 occurrence (run 9, classifier misroute)
- **Verifier signature:** `AssertionError: No .gcda files found, gcov instrumentation may not be enabled`
- Trial `23-36-57/sqlite-with-gcov__qqr3VjQ`. Routed `compositional → dag-tools`,
  but the steps (extract → configure → make → install) are *sequentially
  stateful* and the parallel-DAG decomposition lost the `--enable-gcov` flag in
  the configure step. See classifier section below.

---

## Submit-without-verify counter

Heuristic: for each loss with terminal commands, did BAG ever execute a command
matching a task-specific verifier-like regex (e.g. `pytest`, `curl http`,
`cat /app/move.txt`, `sshpass …`)?

- **Eligible (loss with ≥1 terminal call):** 36
- **Submitted without running anything that resembles the verifier's literal
  command:** 21/36 (≈58%).
- **Plus 14 losses with zero terminal calls** (5 internal-error, 9 clean
  end_turn). All 14 trivially count as submit-without-verify.
- **Total submit-without-verify (loss):** 35 / 50 (70%).

Examples:
- `2026-05-01__13-15-12/qemu-startup__hMJrqGy` — only ran `npm run typecheck`,
  never `cat /tmp/data.txt`.
- `2026-05-01__16-24-37/qemu-alpine-ssh__Yr4chPX` — 0 terminal calls, submit.
- `2026-05-01__18-03-30/configure-git-webserver__YKNzz35` — `setup.sh`,
  `git/server/hooks/post-receive`, `serve.sh`. Never ran `bash /tests/verify.sh`
  or `curl http://localhost/hello.html`.
- `2026-05-01__18-03-30/regex-log__LYMJbbH` — wrote regex.txt, never tested it
  against the 9 dates listed in instruction.md.

By contrast, **all 3 wins with manifests** ran the literal verifier command at
least once before submit.

---

## Per-task variance (BAG-bearing 10-task runs)

Columns are the 10 BAG-bearing runs in chronological order; values are reward
(`-` = task absent or excluded). Run identifiers shortened to MM-SS.

| Task                     | r2 13-15 | r3 16-24 | r4 18-03 | r5 18-55 | r6 19-51 | r7 21-09 | r8 21-43 | r9 22-41 | r10 23-11 | r11 23-36 | flips |
|--------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|----------:|------:|
| build-cython-ext         | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | -  | 1 |
| chess-best-move          | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 1  | 3 |
| configure-git-webserver  | 0 | 0 | 0 | 0 | - | 1 | 1 | 1 | 0 | -  | 2 |
| fix-code-vulnerability   | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1  | 3 |
| log-summary-date-ranges  | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1  | 3 |
| polyglot-c-py            | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1  | 1 |
| qemu-alpine-ssh          | 0 | 0 | 0 | 0 | - | 1 | 0 | 0 | - | -  | 2 |
| qemu-startup             | 0 | 0 | 0 | 0 | - | - | 1 | 1 | 1 | -  | 1 |
| regex-log                | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1  | 3 |
| sqlite-with-gcov         | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0  | 2 |

Most flips: chess-best-move (3), fix-code-vulnerability (3), log-summary-date-ranges (3),
regex-log (3). All four have *non-deterministic submit logic* — see clusters
above. polyglot-c-py only flipped once (0→1) but lost 4 follow-up trials due
to the cleanup miss (cluster 2).

---

## Classifier misroutes (auto mode only — run 9 routing-decision corpus)

Run 9 (`23-36-57`) is the only run with extracted `routing-decision.json`. The
classifier saw 6 tasks and chose:

| trial                                 | shape              | mode      | conf | reward | notes |
|---------------------------------------|--------------------|-----------|-----:|-------:|-------|
| chess-best-move__c9iVHxv              | monolithic-complex | tools     | 0.85 |    1.0 | OK    |
| fix-code-vulnerability__2ZAxGd8       | monolithic-complex | tools     | 0.92 |    1.0 | OK    |
| log-summary-date-ranges__CBUuy7P      | atomic             | tools     | 0.95 |    1.0 | OK    |
| polyglot-c-py__XcUEHSH                | hard               | tools     | 0.95 |    1.0 | OK    |
| regex-log__8dHfxic                    | atomic             | tools     | 0.95 |    1.0 | OK    |
| **sqlite-with-gcov__qqr3VjQ**         | **compositional**  | **dag-tools** | 0.92 | **0.0** | misroute |

**Misroute rationale (sqlite-with-gcov).** The classifier reasoned: "Task
decomposes into independent, verifiable sub-goals: (1) extract tarball, (2)
configure, (3) compile, (4) install in PATH." This reasoning is wrong — those
four steps are *sequentially stateful* (configure flags determine compile
output; the gcov verifier looks at FINAL `.gcda` files). Suggested rules for
the classifier:

- If the verifier signature is "X file does not exist after running Y", treat
  the build pipeline as `monolithic-complex`, not `compositional`.
- Don't classify Autotools/CMake/`./configure` workflows as compositional. The
  config flags propagate forward and parallelizing kills coherence.
- General heuristic: any chain of `extract → configure → compile → install` is
  monolithic. Look for keywords `./configure`, `make`, `setup.py build`,
  `cmake` in the task instruction and demote to monolithic-complex.

(All other 5 routing-decisions in run 9 returned 1.0, so for that small sample
the classifier is 5/6 = 83% correct.)

---

## Token waste (manifest-bearing trials only — runs 1–4)

37 manifests total: 34 losses, 3 wins. (Wins in runs 1–4 are rare;
`16-24-37/regex-log__gLi4pzf`, `18-03-30/log-summary-date-ranges__ugXYV5C`,
`18-55-46/log-summary-date-ranges__38BdXCV`.)

- **Loss median input tokens:** 3,699
- **Loss mean input tokens:** 25,981 (skewed heavily by 4 fix-code-vuln trials
  at 128k–200k each)
- **Win median input tokens:** 4,246
- **Win mean input tokens:** 4,070

Top losses by spend (all in fix-code-vulnerability):
- `16-24-37/fix-code-vulnerability__dNt6rs3`: 199,767 in / 7,687 out / 41 calls
- `13-15-12/fix-code-vulnerability__TnaU9PM`: 150,134 in / 7,160 out / 39 calls
- `18-55-46/fix-code-vulnerability__3QTMWyL`: 128,703 in / 6,658 out / 38 calls
- `18-03-30/fix-code-vulnerability__VPso2BE`: 128,247 in / 6,059 out / 38 calls

**Correlation:** wins clustered tightly around the median; the heavy losses on
fix-code-vulnerability spent ~30× the median and still failed. Spend was
*not* the bottleneck — the model spun on bottle.py without ever running the
provided pytest gate.

---

## Top recommendations (prioritized for run #10+)

1. **Pre-submit verifier hook (highest impact).** Force the agent to run the
   literal command quoted in `instruction.md` before allowing
   `echo BAG_TASK_COMPLETE`. The instruction-md scrape is mechanical: regex for
   `pytest …`, `curl http…`, `cat /app/…`, `sshpass …`. Add this in the
   `acp-auto` driver — wherever the `BAG_TASK_COMPLETE` sentinel is matched
   (likely in `bag_agent/agent.py` around the loop that processes terminal
   exits). Estimated impact: would have salvaged ≥21 of the 50 losses (the
   "submit-without-verify" cluster).

2. **Polyglot/cleanup pre-submit gate.** For tasks whose instruction.md says
   "the only file in /app/X must be Y", enforce a pre-submit `find` that
   asserts no extra files exist. Drop into the auto-router shape="hard"
   handler. Impact: 4 trials.

3. **Reclassify build-pipelines as monolithic-complex.** In whatever module
   computes the routing-decision (the `acp-auto` pre-flight LLM step that
   produced `routing-decision.json`), add a rule: if instruction mentions
   `./configure`, `make`, `cmake`, `setup.py build`, `gcov`, force
   `shape=monolithic-complex`. Impact: would have caught the
   `sqlite-with-gcov__qqr3VjQ` misroute.

4. **VM/host-leak fingerprint.** For any task whose verifier asserts a kernel
   string (qemu-alpine-ssh, similar), require the agent to detect and refuse
   when `uname -r` matches `orbstack|linuxkit|wsl`. Impact: 2 trials.

5. **Multi-solution prompt rule.** For any task whose instruction.md contains
   phrases like "print them all", "all winning moves", "if multiple … list",
   inject the directive: "After your first solution, run an exhaustive
   re-check for additional solutions". Impact: 1 trial (chess-best-move
   "File is wrong"), but very cheap to add.

6. **Drop `acp-code` legacy mode.** Runs 1–4 (`acp-code`) accounted for 34
   losses, mostly by never executing real bash. If `acp-code` is still
   reachable for any task type, force `acp-auto` for terminal-bench. Impact:
   the bulk of the historical loss volume — 34 trials.

7. **Self-eval loop on token budget.** The four 128k+ fix-code-vulnerability
   losses ran 38–41 LLM calls and never converged. Add a cheap budget alarm:
   after `policy.maxTurns` × 0.75, force a recap turn that re-reads
   instruction.md and the most recent verifier output. Impact: 4 high-spend
   trials.

---

## Trial inventory (BAG losses, 50 total)

```
2026-05-01__12-57-44/chess-best-move__JmxJTwC        chess-best-move
2026-05-01__13-15-12/build-cython-ext__7zhqXoU       build-cython-ext (internal err)
2026-05-01__13-15-12/chess-best-move__eh3JFgB        chess-best-move
2026-05-01__13-15-12/configure-git-webserver__q3u2FkN configure-git-webserver (internal err)
2026-05-01__13-15-12/fix-code-vulnerability__TnaU9PM fix-code-vulnerability
2026-05-01__13-15-12/log-summary-date-ranges__ZgQeAaH log-summary-date-ranges
2026-05-01__13-15-12/polyglot-c-py__pCTeFqc          polyglot-c-py
2026-05-01__13-15-12/qemu-alpine-ssh__C4cYrQA        qemu-alpine-ssh
2026-05-01__13-15-12/qemu-startup__hMJrqGy           qemu-startup
2026-05-01__13-15-12/regex-log__vk7Dp7Y              regex-log
2026-05-01__13-15-12/sqlite-with-gcov__FLHBYiX       sqlite-with-gcov
2026-05-01__16-24-37/build-cython-ext__h4zskPP       build-cython-ext (internal err)
2026-05-01__16-24-37/chess-best-move__nviQwDH        chess-best-move
2026-05-01__16-24-37/configure-git-webserver__kRYkudi configure-git-webserver
2026-05-01__16-24-37/fix-code-vulnerability__dNt6rs3 fix-code-vulnerability
2026-05-01__16-24-37/log-summary-date-ranges__i7G56NP log-summary-date-ranges
2026-05-01__16-24-37/polyglot-c-py__eKu4kew          polyglot-c-py (gcc fail)
2026-05-01__16-24-37/qemu-alpine-ssh__Yr4chPX        qemu-alpine-ssh
2026-05-01__16-24-37/qemu-startup__38gQdVU           qemu-startup
2026-05-01__16-24-37/sqlite-with-gcov__TYxX2Nz       sqlite-with-gcov
2026-05-01__18-03-30/build-cython-ext__DcMSi2H       build-cython-ext (internal err)
2026-05-01__18-03-30/chess-best-move__f4Y6BfU        chess-best-move
2026-05-01__18-03-30/configure-git-webserver__YKNzz35 configure-git-webserver
2026-05-01__18-03-30/fix-code-vulnerability__VPso2BE fix-code-vulnerability
2026-05-01__18-03-30/polyglot-c-py__qyjazbq          polyglot-c-py
2026-05-01__18-03-30/qemu-alpine-ssh__VHC2d3K        qemu-alpine-ssh
2026-05-01__18-03-30/qemu-startup__dkQiJXw           qemu-startup
2026-05-01__18-03-30/regex-log__LYMJbbH              regex-log (incomplete pattern)
2026-05-01__18-03-30/sqlite-with-gcov__7Hunac4       sqlite-with-gcov
2026-05-01__18-55-46/build-cython-ext__B9jZhFp       build-cython-ext
2026-05-01__18-55-46/chess-best-move__g3aoNbt        chess-best-move
2026-05-01__18-55-46/configure-git-webserver__FAcTJRp configure-git-webserver
2026-05-01__18-55-46/fix-code-vulnerability__3QTMWyL fix-code-vulnerability
2026-05-01__18-55-46/polyglot-c-py__CegYrmJ          polyglot-c-py
2026-05-01__18-55-46/qemu-alpine-ssh__4ApqWxs        qemu-alpine-ssh
2026-05-01__18-55-46/qemu-startup__to2WkgD           qemu-startup
2026-05-01__18-55-46/regex-log__LveEkgC              regex-log
2026-05-01__18-55-46/sqlite-with-gcov__pdJMipG       sqlite-with-gcov
2026-05-01__19-51-36/polyglot-c-py__ixWQGkc          polyglot-c-py (cmain cleanup)
2026-05-01__21-09-19/polyglot-c-py__dveNqWQ          polyglot-c-py (cmain cleanup)
2026-05-01__21-43-38/fix-code-vulnerability__WSapBfm fix-code-vulnerability (cwe-93)
2026-05-01__21-43-38/log-summary-date-ranges__jTut424 log-summary-date-ranges
2026-05-01__21-43-38/polyglot-c-py__RHtUSKX          polyglot-c-py (cmain cleanup)
2026-05-01__21-43-38/qemu-alpine-ssh__ZszdmbU        qemu-alpine-ssh (host kernel)
2026-05-01__22-41-38/fix-code-vulnerability__5x4SaSt fix-code-vulnerability (internal err)
2026-05-01__22-41-38/polyglot-c-py__yEFw5yw          polyglot-c-py (cmain cleanup)
2026-05-01__22-41-38/qemu-alpine-ssh__E9bTFFQ        qemu-alpine-ssh (host kernel)
2026-05-01__23-11-16/chess-best-move__3hGaYQK        chess-best-move (multi-move miss)
2026-05-01__23-11-16/configure-git-webserver__V8v2hGJ configure-git-webserver (HTTP 404)
2026-05-01__23-36-57/sqlite-with-gcov__qqr3VjQ       sqlite-with-gcov (classifier misroute)
```
