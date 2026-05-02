# Comparative Positioning: trace-gepa bench vs public coding benchmarks

Phase-3 cross-validation. Goal: locate our benchmark against SWE-bench, Terminal-Bench,
Aider Polyglot, and the HumanEval/MBPP legacy line. All numbers below are from public
sources (web only, no installs); see "Sources" footer.

## 1. Comparison matrix

| Benchmark | Tasks | Task shape | Verifier | Scope | Languages |
|---|---|---|---|---|---|
| **trace-gepa (ours)** | ~105 (100 trace-derived + ~5 synth seed; growing to 150) | Single-step tool-action selection from a real Codex/CC trace prefix | 3-tier composite: regex (tier-1), LLM judge (tier-2), shell exec (tier-3) | Agent action-selection / next-tool prediction inside an in-flight trajectory | Tool/CLI agnostic (Python, shell, JSON tool args) |
| SWE-bench (full) | 2,294 GitHub issues | End-to-end repo patch (often multi-file) | `pytest` pass in a per-task Docker image | Real bug fixes + small features in 12 OSS Python repos | Python (Multilingual variant adds 9 langs) |
| SWE-bench Verified / Lite / Pro | 500 / 300 / Pro variant | Same as full, smaller curated splits | Same Docker pytest harness | Human-verified solvability (Verified); harder subset (Pro) | Python |
| Terminal-Bench (Core v0.1.x / "2.0") | ~100 sandboxed terminal tasks | Multi-turn shell trajectory, agent drives a real tmux/bash session | Per-task `test.sh` scripts inside Docker | Real ops/dev work: compile code, train models, set up servers | Bash-driven, polyglot (Py/C/C++/JS/Docker) |
| Aider Polyglot | 225 Exercism problems | Whole-file edits to satisfy a hidden test suite | Language-native unit tests (`pytest`, `cargo test`, `go test`, ...) | Self-contained algorithmic exercises | C++, Go, Java, JS, Python, Rust |
| HumanEval / MBPP | 164 / 974 | Function-body completion from a docstring | `assert`-based unit tests | Toy algorithmic completion | Python only |

## 2. Where we fit

trace-gepa bench is the only public-style benchmark we are aware of that scores
**single-step agent action-selection conditioned on a real trace prefix** rather than
end-to-end task completion. SWE-bench measures "did the final patch make pytest go
green," Terminal-Bench measures "did the agent reach the end of a shell trajectory,"
and Aider/HumanEval measure "did the final code pass unit tests." We instead ask:
*given the same trajectory prefix that a production Codex/CC agent saw, does the
optimized small model pick the same next tool call (or a verifiably-equivalent one)?*
This is finer-grained, much cheaper to score (no container per task), and directly
optimizable by GEPA — but it is not a substitute for end-to-end benches. Positioning:
**complementary to SWE-bench, not competitive.** Lower task count (~150 vs 2,294) is
acceptable because each task is a tightly-scoped decision with a deterministic
verifier ladder rather than a full repo patch.

## 3. Three borrow-worthy ideas

1. **From SWE-bench: Docker-isolated pytest as a tier-4 verifier.** Today our tier-3
   is `shell` (best-effort exec on host). Adopting SWE-bench's per-task Docker image
   pattern would let us promote the ~10 trace-derived tasks that already include a
   repo snapshot to a true "did the resulting patch pass tests" verdict, closing
   the gap on the ~15% of tasks where regex+judge disagree.

2. **From Terminal-Bench: multi-turn rollout mode.** Their test harness drives a
   live tmux session and grades the final filesystem state. We could add an
   *optional* `--rollout` mode that, instead of single-step scoring, lets the model
   continue past the prefix for N steps and grades the trace tail with the same
   tier-3 shell verifier. This gives us a free "trajectory" sub-benchmark without
   re-collecting data.

3. **From Aider Polyglot: cost-and-edit-format reporting alongside pass rate.**
   Their leaderboard reports `pass%`, `$/run`, and `correct edit format %`. We
   currently only emit pass rate per tier. Adding `tokens/task`, `$/task`, and
   `tool-arg JSON validity %` to `eval_multi.py` results would make our results
   comparable across distill targets (Gemma-3 270M vs Qwen-3 4B) on the same axes
   the broader community already uses.

## Sources

- SWE-bench landing page and Verified announcement: https://www.swebench.com/ , https://www.swebench.com/SWE-bench/
- Terminal-Bench repo (Laude Institute): https://github.com/laude-institute/terminal-bench
- Aider Polyglot leaderboard + repo: https://aider.chat/docs/leaderboards/ , https://github.com/Aider-AI/polyglot-benchmark
- Epoch AI summary of Aider Polyglot: https://epoch.ai/benchmarks/aider-polyglot
- Morph "AI Coding Benchmarks 2026" overview: https://www.morphllm.com/ai-coding-benchmarks-2026
