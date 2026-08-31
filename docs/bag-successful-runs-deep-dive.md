# BAG Successful-Runs Deep Dive

Forensic survey of every BAG win (`reward == 1.0`) on `terminal-bench-sample` from
`bench/jobs/2026-05-01__*` and `2026-05-02__*` (49 run dirs over the last 48 h).

## Inventory

| run dirs scanned | successful trials | distinct task families | mean reward across recent runs |
|---|---|---|---|
| 49 | **183** | 12 | ~7.3 wins / 9-task suite (matches reported trajectory 5-10) |

Per-family success counts:

| family | wins | turn_min | turn_med | turn_mean | turn_max | comp_tok_med | nz_exit_rate |
|---|---|---|---|---|---|---|---|
| build-cython-ext | 16 | 24 | 34 | 35.2 | 50 | 5,448 | 21.8 % |
| chess-best-move | 15 | 18 | 34 | 34.9 | 61 | 20,171 | 17.8 % |
| configure-git-webserver | 18 | 18 | 26 | 28.3 | 47 | 5,697 | 18.7 % |
| fix-code-vulnerability | 21 | 15 | 23 | 23.9 | 37 | 2,897 | 11.0 % |
| log-summary-date-ranges | 25 | 3 | 5 | 6.0 | 18 | 1,045 | 2.7 % |
| polyglot-c-py | 21 | 5 | 7 | 7.7 | 15 | 2,164 | 6.8 % |
| qemu-alpine-ssh | 10 | 35 | 55 | 55.2 | 77 | 13,772 | 26.4 % |
| qemu-startup | 13 | **0** | 36 | 34.1 | 80 | 22,672 | 25.0 % |
| regex-log | 27 | 6 | 8 | 10.6 | 62 | 4,217 | 7.1 % |
| sqlite-with-gcov | 15 | 13 | 19 | 21.0 | 42 | 2,739 | 21.1 % |
| instance_ansible / instance_element-hq | 1 each | — | — | — | — | — | — |

`nz_exit_rate` = mean fraction of bash commands that returned non-zero across that family's wins.

## The headline finding: zero-effort passes

**7 / 183 wins (3.8 %) had `turnsUsed == 0`** — the agent never ran. All 6
qemu-startup zero-turn passes ended with `stopReason == "error:prompt timeout
after 880000ms"`; the verifier's `tests/test_outputs.py::test_version` PASSED
anyway. Evidence:

- `bench/jobs/2026-05-02__02-29-26/qemu-startup__SMinaQE` — only `routing-decision.json`
  exists, no autonomous-trace.json, prompt timed out, reward = 1.
- Same pattern in `__04-53-37/qemu-startup__aJKLduv`, `__05-51-57/TqWjYVx`,
  `__06-29-50/fCYzB9P`, `__08-06-13/CiagMUX`, `__09-24-29/WSh4GkT`.
- Tested working qemu-startup wins (e.g. `__08-30-00/MTX75qb` 36 turns) emit
  the *same* `test_version` PASSED. The verifier is satisfied by base-image
  state regardless of whether the agent started qemu.

This means our qemu-startup column on the BAG leaderboard is inflated by ~46 %
(6/13 wins are vacuous). It also implies any regression we introduce in BAG's
qemu handling will be invisible: the score won't move.

## Retry / self-check rates

| metric | value |
|---|---|
| explicit `pre_submit` / `self_check` bash invocation | **0 / 176** trials with traces |
| test-script ran, failed, then re-ran and passed | 21 / 176 = **11.9 %** |
| trials with at least one Traceback in output | 35 / 176 = **19.9 %** |
| trials with same bash command failing ≥ 3× | 4 / 176 |

Retry concentration: **16/21 fail-then-pass cases are fix-code-vulnerability**,
and **5/21 are build-cython-ext**. The agent typically scopes a failing pytest
selector, fixes, retries; that's healthy. Build-cython-ext cycles
`pip install pytest … && pytest` — every win in this family contains a
Traceback (16/16) because of a flaky `test_random_curves` that is later
ignored or worked around.

The hard signal: **`pre_submit` self-check never fired in any successful trial
during this 48 h window**. Whatever gate exists is either unconfigured for
this benchmark, a no-op, or being bypassed before it runs. Wins do not
benefit from a gate.

## Tolerated suboptimality

- **Tmp-file artifacts**: 93 / 176 wins (**52.8 %**) touch `/tmp/` with cp/mv/tee.
  Heaviest in regex-log (27/27), log-summary-date-ranges (25/25),
  sqlite-with-gcov (13/15), qemu-alpine-ssh (10/10). The harbor verifier
  doesn't probe `/tmp`, so leftover scratch never costs reward, but leaves a
  brittle assumption: any verifier upgraded to clean-room would tank scores.
- **Long chess wins**: median chess win = 34 turns; min = 18; max = **61**
  (`__09-24-29/9BUsYcX`). The 61-turn run wrote the move once but spent 50+
  turns confirming with stockfish. Chess `view_image` is essentially unused:
  3/15 wins call it, all 3 fail with "not an image" then fall back to
  PIL+stockfish (PIL/Image.open hit 261/15 cmds, stockfish hit 74).
- **Multiple distinct moves written**: 4/15 chess wins (`__23-36-57/c9iVHxv`,
  `__00-40-52/Pr5LPKK`, `__07-13-33/QqD8E46`, plus visual indicators in
  another) wrote two different moves to `/app/move.txt`. The verifier accepts
  *any* valid winning move, so the agent's wobble is masked. A stricter
  "first answer must be final" verifier would flip these to losses.
- **Hit-max-turns wins**: 3 qemu-startup wins finished at `turns == 80`
  (`stopReason: max_turns`), no clean submit, reward = 1. Same vacuous
  test-version pattern.
- **High noise floor in qemu/sqlite/build**: median 21 % – 26 % of bash
  commands return non-zero; the agent absorbs failure as expected.

## Distribution / brittleness signals per family

- **chess-best-move**: 10/15 wins exceed 25 turns. Stockfish probed many
  candidate FENs; none of the wins emit a FEN string (FEN-detection regex
  matched 0× — the agent goes straight from PIL pixel sampling to
  stockfish's UCI loop). Visual prompt that "view_image is unreliable for
  /app/*.png" would save a 50-turn cliff in 3 of these.
- **build-cython-ext**: 16/16 wins contain Traceback output. 5/16 needed a
  test-rerun to land. Most retries are `pytest tests/ … --ignore`; the
  agent learns to ignore `test_random_curves.py` mid-run. No test gate
  caught the original failure before it became user-visible.
- **fix-code-vulnerability**: highest retry rate (16/21 = 76 %). Pattern is
  always `pytest -x` → fix → re-run, which is healthy but the model never
  ran a focused failing test first; it always ran the full suite, paid
  ~2-3 turns of token cost, then narrowed.
- **log-summary-date-ranges / polyglot-c-py / regex-log**: short, clean
  (median 5–8 turns, 0 % retries). These look like the model's "comfortable
  zone" — string/file munging with deterministic output.
- **qemu-alpine-ssh**: every win ≥ 35 turns; 10/10 wrote to `/tmp` and 26 %
  of bash exits non-zero. Extreme tail: `__01-23-12/Mh3pZJb` 77 turns, 16
  failed exits — a 77-turn slog that the verifier cannot distinguish from a
  35-turn elegant solution.

## Most surprising pattern

**The qemu-startup task on terminal-bench-sample passes its own verifier
without the agent doing anything.** This single mis-configured task accounts
for ~3 % of BAG's headline win rate. Every benchmark trajectory we report
has had its qemu-startup column padded by zero-effort passes.

A close second: zero `pre_submit`/`self_check` invocations across 176
successful traces — the much-discussed self-check gate either is not wired
into the autonomous loop for terminal-bench-sample or runs only on stop and
is never recorded in the trace JSON. Whatever benefit BAG draws from
self-check is invisible in win evidence.

## Headline numbers for downstream consumers

- **wins surveyed**: 183 across 49 runs
- **vacuous wins (no agent activity)**: 7 (3.8 %), six concentrated in qemu-startup
- **retry-rate among wins**: 11.9 %
- **self-check-fire rate among wins**: 0 %
- **/tmp-leak rate among wins**: 52.8 %
- **wins exceeding median family turns by ≥ 2×**: 14 / 183 (≈ 7.7 %),
  dominated by chess (5), qemu-startup (4), build-cython-ext (3)

## Recommendations (harness-level, generic)

1. **Surface vacuous-pass detection in the run summary.** A trial whose
   `autonomous-trace.json` is missing or whose `turnsUsed == 0` should be
   flagged regardless of reward. Today the BAG aggregate metric treats
   them as full credit. A simple `effective_reward = reward * (turns > 0)`
   rollup would shrink qemu-startup contribution from 13 → 7 wins and make
   the column actually move with agent quality.

2. **Capture pre_submit / self_check signal in the trace.** Currently 0 /
   176 wins show any pre-submit instrumentation. Either the gate isn't
   wired into autonomous mode for this benchmark, or it runs without
   emitting bash commands captured by the trace JSON. Add an explicit
   `kind: "self_check"` event into `autonomous-trace.json` so wins-vs-losses
   forensics can attribute lift to the gate. Without this we cannot claim
   the gate works.

3. **Treat any successful test invocation that follows a failed one as a
   "rescue retry" metric**, exposed alongside `turnsUsed`. 11.9 % of wins
   are rescue retries; concentrated in fix-code-vulnerability (76 % of its
   wins). A high rescue-retry rate is the canary for "agent runs the wrong
   broad command first" — a guidable behaviour.

4. **Add a /tmp footprint diff to the verifier or a post-trial cleanup
   audit.** 53 % of wins leak scratch files; a single verifier upgrade
   that cleans `/tmp` before harbor checks would tank scores
   indiscriminately. We need to know which wins are actually clean.

5. **Make `view_image` fail fast and loud on PNGs in /app.** All 3 chess
   wins that called view_image got `"application/octet-stream"` errors.
   The model wastes 2-5 turns iterating renames before giving up. Either
   fix the MIME sniffer to respect PNG magic bytes (it has access to the
   bytes — the trace shows `head -c 16 ... | od -c` printing the PNG
   header), or make the error explicitly say "fall back to PIL".
