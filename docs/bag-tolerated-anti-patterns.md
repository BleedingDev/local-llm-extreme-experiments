# BAG Tolerated Anti-Patterns (in winning trials)

Anti-patterns that BAG is currently *getting away with* — they don't cost
reward today but mask brittleness, inflate metrics, or will regress the
moment a verifier tightens. Each pattern is quantified across the 183 wins
mined in `docs/bag-successful-runs-deep-dive.md`.

## A1. Vacuous qemu-startup pass (zero-effort win)

- **Frequency**: 6 / 13 qemu-startup wins (46 %), 6 / 183 overall (3.3 %).
- **Symptom**: prompt timeout (`error:prompt timeout after 880000ms`), agent
  produces only a `routing-decision.json`, no `autonomous-trace.json`,
  `turnsUsed == 0`, but `tests/test_outputs.py::test_version` PASSED.
- **Evidence**:
  - `bench/jobs/2026-05-02__02-29-26/qemu-startup__SMinaQE`
  - `bench/jobs/2026-05-02__04-53-37/qemu-startup__aJKLduv`
  - `bench/jobs/2026-05-02__05-51-57/qemu-startup__TqWjYVx`
  - `bench/jobs/2026-05-02__06-29-50/qemu-startup__fCYzB9P`
  - `bench/jobs/2026-05-02__08-06-13/qemu-startup__CiagMUX`
  - `bench/jobs/2026-05-02__09-24-29/qemu-startup__WSh4GkT`
- **Hidden cost**: BAG's qemu-startup column is uninformative. Any
  improvement *or* regression in qemu handling is invisible.

## A2. Hit-max-turns wins (verifier passes after agent gives up)

- **Frequency**: 3 / 13 qemu-startup wins (23 %), 3 / 183 overall.
- **Symptom**: `stopReason == "max_turns"`, `turnsUsed == 80`, no clean
  submit, reward = 1. Same `test_version` PASSED pattern as A1.
- **Evidence**:
  - `bench/jobs/2026-05-02__03-31-16/qemu-startup__u4ydTLU` (25 nz exits)
  - `bench/jobs/2026-05-02__03-56-31/qemu-startup__tFL4gS4` (26 nz exits)
  - `bench/jobs/2026-05-02__07-13-33/qemu-startup__XNnz92r` (18 nz exits)
- **Hidden cost**: Combined with A1, **9 / 13 qemu-startup wins (69 %)**
  represent agent failure that the verifier rubber-stamped.

## A3. Excessive turn count on chess wins

- **Frequency**: 10 / 15 chess wins exceed 25 turns; median 34, max 61. The
  18-turn solver (`__01-23-12/xErvvsb`) shows the task is solvable in <20.
- **Symptom**: Heavy stockfish round-trips, repeated PIL pixel-sampling,
  multi-pass FEN extraction. PIL/Image.open invoked **261×** across the 15
  wins (avg 17/trial); stockfish 74×.
- **Evidence**:
  - `bench/jobs/2026-05-02__09-24-29/chess-best-move__9BUsYcX` (61 turns,
    final move written at call #175 of 183 events)
  - `bench/jobs/2026-05-02__09-45-38/chess-best-move__bLMZs2L` (50 turns,
    14 nz exits, 5 failed view_image fallbacks)
  - `bench/jobs/2026-05-02__07-13-33/chess-best-move__QqD8E46` (43 turns,
    wrote two different moves: e2e4 then g2g4)

## A4. Chess answer wobble (writes multiple distinct moves)

- **Frequency**: 4 / 15 chess wins wrote two different moves to
  `/app/move.txt`. Verifier accepts any winning move, so wobble is hidden.
- **Evidence**:
  - `__23-36-57/c9iVHxv`: g2g4 then e2e4
  - `__00-40-52/Pr5LPKK`: e2e4 then g2g4
  - `__07-13-33/QqD8E46`: e2e4 then g2g4
- **Hidden cost**: A stricter "answer-stability" verifier would convert
  these wins to losses. Suggests stockfish/PIL chain produces unstable
  rankings.

## A5. view_image silently broken on /app/*.png

- **Frequency**: 3 / 15 chess wins call view_image; 3 / 3 of those fail
  with `"view_image: /app/chess_board.png is application/octet-stream, not
  an image"`. Agent burns 2-5 turns on the dead-end before falling back to
  PIL.
- **Evidence**:
  - `__09-06-04/7npBfPA`: 5 view_image calls, all fail
  - `__09-24-29/9BUsYcX`: 4 view_image calls, all fail
  - `__09-45-38/bLMZs2L`: 5 view_image calls, all fail (also tried
    `cp /app/chess_board.png /tmp/board2.PNG` — same failure)
- **Hidden cost**: 2-5 wasted turns per chess attempt; would help if the
  view_image MIME-detection were fixed or the tool surfaced a clearer
  error pointing the agent at PIL.

## A6. /tmp scratch-file leaks

- **Frequency**: 93 / 176 wins (52.8 %) cp/mv/tee into `/tmp/`.
- **Top offenders**: regex-log 27/27, log-summary-date-ranges 25/25,
  qemu-alpine-ssh 10/10, sqlite-with-gcov 13/15.
- **Hidden cost**: A clean-room verifier (one that diff-checks /app or
  enumerates filesystem state) would penalise these. Today: free pass.

## A7. Test-rerun pattern hides slow first-pass

- **Frequency**: 21 / 176 wins (12 %) ran a test, failed, fixed, re-ran.
  16 of 21 are fix-code-vulnerability; 5 are build-cython-ext.
- **Symptom**: Initial run is the full suite; agent only narrows to a
  failing selector after seeing red. 2-3 wasted turns per trial.
- **Evidence**:
  - `__23-36-57/fix-code-vulnerability__2ZAxGd8`: 4 pytest invocations
    before pass.
  - `__03-31-16/build-cython-ext__o5j99u4`: same `pytest tests/` failed
    twice with `--ignore=tests/test_random_curves` before passing on
    third try.
- **Hidden cost**: Wasted budget; 100 % of build-cython-ext wins (16/16)
  produce a Traceback before passing — the gate that should have caught
  this never fired.

## A8. Self-check / pre_submit gate is silent in 100% of wins

- **Frequency**: 0 / 176 wins emit any bash command matching
  `pre_submit|self_check|complete\s*[:=]\s*true`.
- **Symptom**: Either the gate isn't wired into the headless ACP runner
  for terminal-bench-sample, runs out-of-band of the captured trace, or
  is conditionally skipped. If it's enabled, **no win has any evidence
  of it firing**.
- **Hidden cost**: We can't claim self-check helped any win. Loss
  forensics would need to confirm whether failed runs *did* fire it; if
  yes, we have an asymmetry; if no, the gate is dead code on this suite.

## A9. Repeated-failed-command stuck loops (mild)

- **Frequency**: 4 / 176 wins repeated the same failing command ≥ 3×.
- **Examples**:
  - `polyglot-c-py__Z2GgsgW`: `cat > /app/polyglot/main.py.c << 'EOF'`
    repeated 5× (typo path, agent didn't notice .py.c was wrong).
  - `build-cython-ext__QDG222Y`: same pytest selector tried 4×.
  - `chess-best-move__WoaqbJZ`: `cd /tmp && python3 <<'EOF'` 3×.
- **Hidden cost**: Suggests the agent doesn't always diff its last failed
  command before retrying.

## A10. Build-cython-ext: 100 % traceback rate among wins

- **Frequency**: 16 / 16 build-cython-ext wins contain a Traceback in
  output. Median nz_exits per win = 7. Median completion tokens = 5,448.
- **Hidden cost**: Every "win" in this family is the agent recovering
  from a deterministic error (`test_random_curves` flake). A more
  conservative verifier or a self-check that flags "test still has
  unresolved Traceback in last output" would block these.

---

## Summary table

| anti-pattern | wins affected | % of wins | severity |
|---|---|---|---|
| A1 vacuous qemu-startup | 6 | 3.3 % | high (metric inflation) |
| A2 max-turns then pass | 3 | 1.6 % | high (overlaps A1) |
| A3 excessive chess turns | 10 | 5.5 % | medium (budget) |
| A4 chess answer wobble | 4 | 2.2 % | medium (latent) |
| A5 view_image MIME bug | 3 | 1.6 % | low-medium (waste) |
| A6 /tmp leak | 93 | 52.8 % | medium (latent) |
| A7 test-rerun pattern | 21 | 12 % | low (budget) |
| A8 self-check silent | 176 | 100 % | high (signal loss) |
| A9 repeated stuck cmd | 4 | 2.3 % | low |
| A10 cython traceback | 16 | 9 % | medium |
