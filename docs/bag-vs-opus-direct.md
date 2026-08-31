# BAG vs Opus-direct (claude-code) on TB 2.0 sample

**Date:** 2026-05-01
**Model:** `claude-opus-4-7`
**Dataset:** `terminal-bench-sample@2.0` (10 tasks)
**All runs:** concurrency=4, agent_timeout 1800s, agent_setup_timeout 1080s

The contamination-and-value-add audit: does BAG (our coding harness) actually improve on calling Opus 4.7 directly via Harbor's built-in `claude-code` agent?

## TL;DR

**No.** Plain Opus-direct (90%) beats every BAG mode tested (60–80%) on this 10-task sample. The model is doing the work; our orchestration layer either matches or actively hurts. BAG's measurable value-add today on TB 2.0 sample is **negative**. The only places BAG could win — multi-model cost split, self-evolving tuning, hard benchmarks where Opus alone fails — are not yet measured or built.

## Final scoreboard

```
Opus-direct (claude-code agent):   9/10   →  90%
BAG tools mode (run #5):           8/10   →  80%
BAG dag-tools mode (run #6):       6/10   →  60%
BAG auto mode (run #7):            7/10   →  70%
```

**Honest verdict:** **BAG loses to Opus-direct on TB 2.0 sample by 10-30 percentage points across all modes.** No BAG mode matches plain Opus-direct (90%).

- best BAG mode (tools) = -10pp behind Opus-direct
- adaptive BAG mode (auto) = -20pp behind Opus-direct
- worst BAG mode (dag-tools) = -30pp behind Opus-direct

## Per-task A/B

| Task | Opus-direct | BAG tools (#5) | BAG dag-tools (#6) | BAG auto (#7) |
|---|---|---|---|---|
| build-cython-ext | 1.0 | 1.0 | 1.0 | 1.0 |
| chess-best-move | 1.0 | 1.0 | 1.0 | 1.0 (auto→tools, hard) |
| configure-git-webserver | 1.0 | 1.0 | 1.0 | 1.0 |
| fix-code-vulnerability | 1.0 | 1.0 | 0.0 | **0.0** ← variance loss in #7 |
| log-summary-date-ranges | 1.0 | 1.0 | 0.0 | 1.0 |
| polyglot-c-py | 0.0 | 0.0 | 0.0 | 0.0 (universal fail — verifier rejects cleanup) |
| qemu-alpine-ssh | 1.0 | 1.0 | 0.0 | **0.0** ← classifier picked dag-tools, ordering broken |
| qemu-startup | 1.0 | RuntimeError ❌ | 1.0 | 1.0 |
| regex-log | 1.0 | 1.0 | 1.0 | 1.0 (auto→tools, atomic) |
| sqlite-with-gcov | 1.0 | 1.0 | 1.0 | 1.0 (auto→dag-tools, compositional) |

## Auto-mode classifier audit (run #7)

Adaptive router decisions per task:

| Task | Shape | Mode | Correct? |
|---|---|---|---|
| chess-best-move | hard | tools | ✅ won |
| polyglot-c-py | hard | tools | ✅ (universal fail) |
| regex-log | atomic | tools | ✅ won |
| sqlite-with-gcov | compositional | dag-tools | ✅ won |
| log-summary-date-ranges | atomic | tools | ✅ won |
| qemu-startup | atomic | tools | ✅ won |
| build-cython-ext | monolithic-complex | tools | ✅ won |
| fix-code-vulnerability | monolithic-complex | tools | ✅ correct mode, lost to variance |
| **qemu-alpine-ssh** | **compositional** | **dag-tools** | **❌ misclassified — ordering-sensitive task; tools wins, dag-tools fragments** |
| configure-git-webserver | (not captured) | (worked) | ✅ won |

**Classifier accuracy: 9/10.** One miss (qemu-alpine-ssh) cost a point. Plus run-to-run Opus variance lost fix-code-vulnerability.

The remaining gap to Opus-direct's 8-9/10 is: the variance loss + the classifier miss.

## Where BAG actively LOSES vs Opus-direct

1. **dag-tools fragmentation** — fix-code-vulnerability and log-summary-date-ranges lose in dag-tools mode (run #6) because the lite planner over-decomposes holistic tasks. Opus-direct keeps them.
2. **Auto-mode classifier mistakes** — qemu-alpine-ssh routed to dag-tools, fragmented, lost.
3. **Setup overhead** — every BAG trial does Node install + npm install + bag-runtime upload (~3-5 min). Opus-direct's claude-code agent has lighter setup.
4. **Manual-kill drag (now fixed)** — pre-`process.exit(0)` BAG processes hung after submit. Drove qemu-startup to RuntimeError in run #5. Fixed in run #6 onward.

## Where BAG matches Opus-direct

Every "easy" task (regex-log, sqlite, chess, log-summary, configure-git, build-cython, qemu-startup): BAG tools mode and Opus-direct produce identical wins. The agentic skill on this benchmark **is in the model**, not in our orchestration.

## Where BAG could win in principle (untested)

Things Opus-direct can't do that BAG could, but not measured here:
1. **Self-evolving feedback** — BAG captures traces, has GEPA optimizer scaffolding. Opus-direct is one-shot.
2. **Multi-model orchestration** — BAG has master/local roles. Configured to use Opus for both today; could route cheap subroutines to Haiku.
3. **Persistent knowledge codification** — across runs.
4. **DAG planning where it actually fits** — qemu-startup flipped from RuntimeError → 1.0 in dag-tools mode.

None of these are reflected in single-shot TB 2.0 sample reward numbers.

## Brutally honest implications

1. **BAG cannot be sold as "+X% over Opus" on TB 2.0 sample.** Best case parity, often loses. The "70% in run #7" is a -10pp regression vs Opus-direct.

2. **The architecture pivots so far have only neutralized a shortfall, not built a moat.** We started at 0% (plan+patch), pivoted to 80% (tools), tried 60% (dag-tools), landed 70% (auto). Each pivot was rational; none of them surpasses Opus-direct.

3. **The win-set difference is interesting:**
   - tools mode wins: qemu-alpine, log-summary in monolithic, fix-code (sometimes)
   - dag-tools wins: qemu-startup (when pre-`process.exit(0)`)
   - oracle adaptive ceiling: 9/10 = 90% (fix-code variance is the only blocker)
   - But oracle requires hindsight, classifier today doesn't have it.

4. **Real value-add must be measured on different axes:**
   - **Token cost per task** — does BAG burn fewer tokens for same outcome via Haiku-for-easy-stuff?
   - **Cross-run improvement** — does BAG #2 do better after seeing #1's traces?
   - **Tasks where Opus alone fails** — longer-horizon, multi-file, novel.
   - **Stability under noise** — single-shot Opus has 10-20% run-to-run variance; BAG could reduce that with structured retries.

## What this audit changes

- Stop optimizing for TB sample mean reward. We're capped near Opus-direct.
- **Wire the multi-model split** (master=Opus, local=Haiku) and measure $/task. BAG's local-role currently mirrors master — wasted opportunity.
- **Use captured traces** to drive GEPA optimizer iteration. That's the **self-evolving** lift Opus-direct cannot do.
- **Push to harder benchmarks** (full TB 2.0 89 tasks, SWE-bench Pro) where Opus alone might fail more often, surfacing BAG's architectural advantages.
- **Improve the classifier** with the run #7 mislabel (qemu-alpine-ssh: ordering-sensitive ≠ compositional) so it doesn't repeat.

## Reproduce

```bash
# Opus-direct baseline
cd bench && . .venv/bin/activate && set -a && source ../.env && set +a
PYTHONPATH=. ANTHROPIC_API_KEY="$ANTHROPIC_AUTH_TOKEN" \
  harbor run -d terminal-bench-sample@2.0 \
  -a claude-code -m claude-opus-4-7 \
  -n 4 --agent-timeout-multiplier 2.0 --agent-setup-timeout-multiplier 3.0

# BAG tools mode
PYTHONPATH=. harbor run -d terminal-bench-sample@2.0 \
  --agent-import-path bag_agent.agent:BagAgent -m claude-opus-4-7 -n 4 \
  --agent-timeout-multiplier 2.0 --agent-setup-timeout-multiplier 3.0 \
  --ak bag_mode=tools

# BAG auto mode (adaptive)
PYTHONPATH=. harbor run -d terminal-bench-sample@2.0 \
  --agent-import-path bag_agent.agent:BagAgent -m claude-opus-4-7 -n 4 \
  --agent-timeout-multiplier 2.0 --agent-setup-timeout-multiplier 3.0 \
  --ak bag_mode=auto
```
