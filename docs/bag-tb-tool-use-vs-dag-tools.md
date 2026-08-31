# Tool-use vs DAG-tools — empirical comparison on TB 2.0 sample

**Date:** 2026-05-01
**Model:** `claude-opus-4-7`
**Dataset:** `terminal-bench-sample@2.0` (10 tasks)
**Both runs:** concurrency=4, agent-timeout 1800s, agent-setup-timeout 1080s

## Headline

```
Run #5  bag_mode=tools       (naked autonomous loop)        8/10 = 0.800   wall 33m53s
Run #6  bag_mode=dag-tools   (lite plan + per-issue scoped) 6/10 = 0.600   wall 15m59s
```

DAG-tools is **20 percentage points worse on mean reward** but **half the wall time**. The wall-time win is from the process.exit fix landing in run #6 (so containers exit cleanly without manual kills). It is unrelated to the planning architecture — both modes will benefit from the fix.

The score delta IS architectural.

## Per-task A/B

| Task | #5 tools | #6 dag-tools | Δ | Pattern |
|---|---|---|---|---|
| build-cython-ext | 1.0 | 1.0 | = | Compositional (build + test) — both work |
| chess-best-move | 1.0 | 1.0 | = | Monolithic (analyze image, compute, write file) — both work |
| configure-git-webserver | 1.0 | 1.0 | = | Compositional (install + configure + start) — both work |
| **fix-code-vulnerability** | **1.0** | **0.0** | **−1** | Monolithic-complex (CVE patch across files) — DAG fragments |
| **log-summary-date-ranges** | **1.0** | **0.0** | **−1** | Monolithic (one cohesive datetime computation) — DAG fragments |
| polyglot-c-py | 0.0 | 0.0 | = | Hard (dual-language constraint) — neither works |
| **qemu-alpine-ssh** | **1.0** | **0.0** | **−1** | Compositional but with ordering subtlety — DAG misorders |
| **qemu-startup** | err | **1.0** | **+1** | Compositional — DAG decomposition helps! |
| regex-log | 1.0 | 1.0 | = | Atomic (one file, one regex) — both work |
| sqlite-with-gcov | 1.0 | 1.0 | = | Compositional (clear build steps) — both work |

**Net for DAG-tools: −3 wins, +1 win = −2 net.** Plus 1 exception in run #5 that DAG-tools recovered.

## Empirical lesson — when planning helps and when it hurts

### When DAG-tools helps

- **Truly independent sub-goals.** `qemu-startup` decomposed cleanly into "create script" + "set permissions" + "verify boot". Per-issue verifier gating caught issues early; model didn't drift.
- **Build/install pipelines** with concrete intermediate states (build-cython, sqlite, configure-git all worked equally).

### When DAG-tools hurts

- **Holistic patches.** `fix-code-vulnerability` (CVE in bottle.py) is one cohesive change spread across multiple files. The lite planner over-decomposed into "understand the vulnerability" / "write the patch" / "verify" — the model wrote a patch in step 2 that didn't reflect the global understanding from step 1, because each issue is a fresh tool-use loop without state propagation.
- **Single-concept computation.** `log-summary-date-ranges` is one logical unit (parse logs + compute date ranges + summarize). Decomposing into "parse" + "compute" + "format" forces the model to commit to a parser interface in step 1 that step 2 can't change.
- **Ordering-sensitive ops.** `qemu-alpine-ssh` involves "boot VM" → "wait for SSH" → "auth" → "verify". DAG's per-issue verifier ran the boot+SSH check too early and bailed.

### Pattern

The lite planner makes a **commitment after one LLM call** about how to decompose, and that commitment is binding for the rest of the run. For tasks where decomposition matches the natural problem structure, this is a win (issues pass verifier locally, less drift). For tasks where the natural shape is monolithic, the commitment is wrong and the model can't recover because each per-issue loop doesn't see the global context that would have come from the unified naked loop.

This is **classic plan-quality bottleneck.** Naked tool-use loop is a "no-commitment" architecture — the model can pivot mid-flight as it observes filesystem and command results.

## What this implies — adaptive routing

Both modes have non-empty win sets that the OTHER mode loses:
- DAG-tools wins: qemu-startup (run #5 lost it as exception)
- tools wins: fix-code-vulnerability, log-summary-date-ranges, qemu-alpine-ssh

**Best case adaptive (oracle router):** 9/10 = 90 % (everything either mode wins, plus polyglot still lost).

**Realistic adaptive (heuristic + 1-LLM-call classifier):**
- Atomic / monolithic / hard → naked `tools` (let model see whole problem)
- Compositional with truly independent steps → `dag-tools`
- Default to `tools` (safe baseline)

Expected gain: **+1 to +2 wins** on a 10-task sample. Will need 50-100 runs across diverse tasks to refine the classifier.

## Adaptive router — implementation plan

### Stage A — heuristic + classifier (next, ~250 LOC)

```typescript
// src/task-shape-router.ts
async function classifyTaskShape(router, task, repoContext): Promise<{
  shape: "atomic" | "compositional" | "monolithic-complex" | "hard";
  mode: "tools" | "dag-tools";
  confidence: 0-1;
  reasoning: string;
}>
```

Single LLM call, ~500 input tokens, one of 4 shape labels. Mapping:
- atomic → tools (overhead-free)
- compositional → dag-tools (planning is the win)
- monolithic-complex → tools (preserve global view)
- hard → tools (let model improvise)

### Stage B — telemetry + offline replay

Persist `(task signature, route taken, reward, tokens)` per trial. Build accumulating dataset. Codex's `src/optimizer/gepa-*.ts` can ingest these and propose new routing rules.

### Stage C — GEPA-tuned router (later)

Treat the classifier prompt as an optimizable artifact. GEPA promotes versions that improve hold-out reward. This is the **self-evolving** piece — the agent learns to route itself by accumulating empirical evidence, with promotion gates preventing regression.

## Numbers we can't yet defend

- **Sample size 1 per cell.** Single run #5 and #6 = 1 datapoint per task per mode. Stochastic Opus output means flips are real noise. Need n=3-5 per cell for confidence.
- **Contamination.** TB 2.0 dataset is public; Opus 4.7 cutoff is 2026-01. Some training overlap likely. Verifier is sealed (we can't peek at /tests during agent run), but task instructions in `instruction.md` may be in training.
- **No Opus-direct baseline.** What's BAG's lift over plain `harbor run -a claude-code -m claude-opus-4-7`? We haven't measured. The 80% on tools mode is BAG-on-Opus; the agentic-skill component vs the model component is unmeasured.

## Reproduce

```bash
cd bench && . .venv/bin/activate && set -a && source ../.env && set +a

# tools mode (naked)
PYTHONPATH=. harbor run -d terminal-bench-sample@2.0 \
  --agent-import-path bag_agent.agent:BagAgent \
  -m claude-opus-4-7 -n 4 \
  --agent-timeout-multiplier 2.0 \
  --agent-setup-timeout-multiplier 3.0 \
  --ak bag_mode=tools

# dag-tools mode (lite plan + per-issue scoped)
PYTHONPATH=. harbor run -d terminal-bench-sample@2.0 \
  --agent-import-path bag_agent.agent:BagAgent \
  -m claude-opus-4-7 -n 4 \
  --agent-timeout-multiplier 2.0 \
  --agent-setup-timeout-multiplier 3.0 \
  --ak bag_mode=dag-tools
```

Compare per-task rewards in `bench/jobs/<timestamp>/*/result.json`.
