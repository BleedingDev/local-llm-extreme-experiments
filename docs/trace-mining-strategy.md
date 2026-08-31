# Trace-Mining Strategy: Turning Our Corpus Into Continuous Capability

**Date:** 2026-05-01
**Author:** strategy reconnaissance — meta thinking, not specific mined findings
**Status:** Roadmap. Pick from this. Nothing here is binding.

This document is the answer to: *given the trace data we sit on (Codex history+SQLite, Claude Code session jsonl, BAG bench trial outcomes, autonomous-trace bundles), what continuous capabilities can we build that compound, instead of one-shot prompt fixes?*

It cites our existing findings and plans where they exist (so we don't re-invent), and proposes 12 capability ideas tiered by effort × value. Pitfalls and infra sections are intentionally pessimistic — trace mining is famous for hallucinating signal from noise.

---

## 1. The trace-mining landscape

The academic + industry techniques that compose any agent self-improvement story. Brief survey, biased toward our data shape.

### 1.1 Program-trace analysis (classical)
Static + dynamic analysis of execution traces — long history in performance engineering (DTrace, eBPF) and in software-engineering research on test-case mining (Pacheco & Ernst, 2007 — Randoop). For agent traces, the analog is parsing tool-call sequences and command outputs to surface stable subsequences. **Pros:** deterministic, no learned model required, results are explainable. **Cons:** requires schemas; our jsonl streams are heterogeneous across Codex versions and Claude Code versions. **Fit:** good for tool-call sequence statistics, log-pattern clustering, fail-tag attribution. We already do this informally in `docs/agent-trace-mining-report.md` (bash-recipe medians, WARN cluster counts).

### 1.2 Tool-call sequence mining
Sequential-pattern mining (PrefixSpan, GSP — Pei et al. 2001) over (tool, args-shape, exit-status) tuples. The OpenAI Swarm / function-call corpus papers and Anthropic's tool-use cookbook make this a first-class lens: which 3-grams of tool calls correlate with success? **Pros:** maps onto what we have (Codex tool calls in `logs_2.sqlite`, Claude Code tool blocks in jsonl). **Cons:** verb explosion — even our 80-trial bench has dozens of bash subforms (`pip install` vs `python -m pip install` vs `uv pip install`). **Fit:** strong, especially for BAG since `bag_mode=tools` produces clean tool-call streams.

### 1.3 Prompt distillation / RAG over traces
Index past task transcripts; at runtime retrieve top-k most-similar prior runs and inject 2–3 winning examples into the system prompt. Closely related to ICL retrieval (Liu et al. 2022, "What Makes Good In-Context Examples"), and to Anthropic's "few-shot exemplar" pattern in their Constitutional AI playbook. **Pros:** zero training, immediate uplift, leverages our `~/.codex/embeddings/` (already pre-built). **Cons:** prompt bloat, exemplar staleness, retrieval noise. **Fit:** the highest-leverage low-risk capability in our pile.

### 1.4 RLHF / RLAIF from traces
Use trace outcomes as reward signal for fine-tuning (DPO, RLHF, RLAIF). **Pros:** durable capability lift. **Cons:** needs a small open model in the loop, ≥10k trace pairs for stable training, and a reward model that is not itself a hallucination machine. **Fit:** marginal for now — we have ~85 trial records and ~20 Claude project sessions; under RLHF's data-hunger threshold. T4-tier research project.

### 1.5 GEPA-style optimization
GEPA (Generative-Evolutionary Prompt Adaptation, Khattab et al. 2024 — DSPy GEPA module) treats prompt fragments as evolvable units, generates variants, evaluates them on held-out tasks, and promotes winners with regression gates. **Pros:** matches our existing scaffolding — see `.codex/plans/bleeding-agent-v1-autonomous-gepa-operations.plan.md`. **Cons:** needs a real eval set; GEPA on noisy reward is dangerous. **Fit:** already planned. Trace mining feeds GEPA, doesn't replace it.

### 1.6 Behavior cloning / trace fine-tuning
Treat (task → bash_sequence) as supervised data, fine-tune a small model to emit BAG's command sequences directly. Closely related to AgentBench / SWE-Gym imitation traces and Anthropic's Computer Use distillation. **Pros:** big inference cost win if successful. **Cons:** needs hundreds of high-quality successful trajectories; ours are mixed-quality. **Fit:** T4 research project; defer.

### 1.7 Failure-cluster autodiscovery
Embed verifier outputs / error logs, cluster (HDBSCAN, BERTopic), label clusters as "common failure mode N." Origin: log-anomaly literature (DeepLog — Du et al. 2017). **Pros:** good fit for our SQLite log stream. **Cons:** clusters need human labeling to be actionable. **Fit:** strong; T2.

### 1.8 Verifier-signature / fix-pattern indexing
Each trial that recovered from an initial verifier fail produces a (complaint → fix) tuple. Build a content-addressed library; at runtime do nearest-neighbor lookup when verifier complaint text appears. Origin: program-repair literature (Prophet — Long & Rinard 2016; Codex-style learned fixers). **Pros:** highly actionable. **Cons:** small corpus → overfit risk. **Fit:** immediate value once we have ~200 fix tuples.

### 1.9 Prompt fingerprinting + cost regression
Treat the system prompt as a versioned artifact; compute a signature; replay past task corpus through prompt-version-A vs prompt-version-B; alarm on token-cost or pass-rate regression. Borrowed from CI-style guardrails (e.g., Cohere Compass, LangSmith eval). **Pros:** prevents silent regressions; lightweight. **Cons:** requires a deterministic replay harness. **Fit:** the v1-real-replay-corpus plan covers the substrate; this is one consumer.

### 1.10 Agent self-eval honesty calibration
When BAG self-scores 0.95 but historical pass-rate on similar tasks was 0.6, surface the discrepancy. Origin: calibration literature (Guo et al. 2017, "On Calibration of Modern NNs"; selective prediction — Geifman & El-Yaniv 2017). **Pros:** very cheap once historical priors exist. **Cons:** requires a "task similarity" function. **Fit:** natural extension of the embeddings index.

### 1.11 Trace-graph mining
Treat each trace as a DAG of (state, action, observation) and mine motif subgraphs that correlate with reward. Inspired by DAG-CNN literature and recent agent-graph work (Voyager — Wang et al. 2023; AutoGen agent graphs). **Pros:** captures branching behavior the linear-sequence view misses. **Cons:** infrastructure-heavy; needs solid trace normalization. **Fit:** T3+; relevant to the dag-tools mode.

### 1.12 Persona / preference mining
Detect durable user preferences from corrective messages ("don't create files when I say analyse"; the Czech-emphasis pattern). Origin: dialogue-grounding work (Zhang et al. 2018, PersonaChat) and Anthropic's "memory" feature direction. **Pros:** big DX uplift, low compute. **Cons:** privacy footgun, tendency to over-fit one user's style. **Fit:** our `bag-trace-mining-deep-dive.md` already manually mined ~10 such preferences; mechanizing is T2.

**Cited sources flagged as plausible-but-unverified:** "Pei et al. 2001 PrefixSpan", "Long & Rinard 2016 Prophet", "Du et al. 2017 DeepLog", "Khattab et al. 2024 DSPy GEPA". These are real papers I'm reasonably confident exist; if exact dates matter for an external doc, double-check. The Anthropic-internal references ("Constitutional AI playbook," "Computer Use distillation") are gestures at directions, not pinned citations.

---

## 2. Data audit

What we actually have, with honest quality grades.

### 2.1 `~/.codex/history.jsonl` — 8 MB, ~10k lines
**Format:** line-delimited JSON; one entry per command/response. Fields include free-text user messages and Codex replies.
**Density:** very high — every Codex turn across all my projects since earliest install.
**Coverage:** broad (multiple projects, multiple personalities), shallow (no tool-call structure, no rewards).
**Quality grade:** B. Great for persona/preference mining and for "what did I curse at" pattern detection. Useless for tool-sequence mining (no tool data here).
**Gaps:** no project demarcation, no outcome labels.

### 2.2 `~/.codex/logs_2.sqlite` — 406 MB, ~150k rows
**Format:** structured event log; columns include `level`, `target`, `feedback_log_body`, `thread_id`, timestamps.
**Density:** very high — TRACE/DEBUG/INFO dominate, ERROR (115) and WARN (~800) are signal.
**Coverage:** internal Codex events only, not user-visible turns.
**Quality grade:** A for system-health mining (already produced 5 actionable findings in `docs/agent-trace-mining-report.md`), C for behavioral mining (events ≠ tool-call sequences).
**Gaps:** no reward column, no link to history.jsonl entries via stable session id (thread_id is internal, not user-facing).

### 2.3 `~/.codex/archived_sessions/` — many `rollout-*.jsonl`
**Format:** rollout records, one per session, jsonl.
**Density:** moderate (~daily files since Feb 2026).
**Coverage:** older sessions; the natural cold-storage layer.
**Quality grade:** B; same shape as history.jsonl but session-bounded.
**Gaps:** mixed schemas across Codex versions.

### 2.4 `~/.codex/embeddings/` — 2.7 MB, 46 dirs, raw float32 vectors per doc-hash
**Format:** content-addressed JSON arrays (~768-dim by inspection). Two-level shard structure: `<corpus-hash>/<doc-hash>/<chunk-hash>.json`.
**Density:** sparse — looks like Codex's own RAG-over-recent-context cache.
**Quality grade:** A as a substrate (vectors already exist!), B for semantics (we don't know which corpus each shard covers).
**Gaps:** no manifest mapping hash → content. Reverse-engineering the schema is a T1 spike.

### 2.5 `~/.claude/projects/*` — 20 project dirs, jsonl per session
**Format:** Claude Code session jsonl, multi-MB per active project.
**Density:** high; each event has type (`user`, `assistant`, `tool_use`, `tool_result`, `summary`, ...) with full structured tool-call content.
**Coverage:** the *richest* dataset we have for tool-call sequence mining.
**Quality grade:** A. This is the gold layer.
**Gaps:** no explicit outcome labels (success/failure of the *task*, not just of individual tool calls).

### 2.6 `bench/jobs/<timestamp>/` — 23 job dirs, ~80 trials
**Format:** per-trial directory with `result.json`, `agent.log`, `task_log/`, `verifier_log/`. Top-level `result.json` summarizes pass/fail across the dataset.
**Density:** moderate — n=80 trials is small for ML, big for hand-analysis.
**Coverage:** TB sample (10 tasks) × multiple bag_modes × multiple runs.
**Quality grade:** A. Has reward, mode, model, wall-time, exception-type — perfect labels.
**Gaps:** small n; correlated tasks.

### 2.7 `bench/.bag/optimizer/dataset.jsonl` — 85 records
**Format:** flattened summary per trial (trial_id, reward, agent_summary, manifest, routing, verifier stdout tail).
**Density:** dense per-record but n=85.
**Quality grade:** A; this is the curated optimizer-ready slice.
**Gaps:** sample size for any statistical claim is borderline.

### 2.8 `bag-traces/` — ostensibly run-#9+ full autonomous-trace bundles (currently empty in this checkout)
**Format (planned):** `autonomous-trace.json` + telemetry per trial.
**Density (when populated):** very high — full agent thought stream + tool calls + verifier feedback.
**Quality grade:** projected A+. This is what the v1-real-replay-corpus plan unlocks.

### 2.9 What's missing across the board
- **Stable cross-source join keys.** Codex thread_id → history.jsonl entry → SQLite events: no clean join. We need a unified session-fact-table.
- **Outcome labels on Claude Code sessions.** We have rich trajectories with no terminal pass/fail.
- **Privacy redaction layer.** `~/.codex/history.jsonl` contains personal/work content. Anything we ship out of it must be redacted (the v1-replay-redaction-policy todo names this).
- **Stable embeddings of OUR concepts.** The `~/.codex/embeddings/` is Codex's own; we'd want our own index keyed by task signature + outcome.

---

## 3. Capability roadmap (12 ideas, tiered)

### Idea 1: Verifier-Signature Library ("Complaint→Fix Index")
**What it does.** Index every (verifier_complaint_text → successful_fix_diff) tuple from `bench/jobs/*/`. At runtime, when BAG's verifier surfaces a complaint, do nearest-neighbor lookup; if a strong match exists, hot-load the historical fix as a "have you tried…" exemplar in the next prompt turn.
**Inputs.** `bench/jobs/*/verifier_log/`, `result.json`, `task_log/`.
**Outputs.** A keyed JSON store + a runtime lookup hook BAG can call before retrying.
**Effort tier.** T2.
**Expected value.** Medium-high. On the 80-trial corpus, ~20% of failures look recoverable from a sibling trial. Pass-rate uplift estimate: +5pp on TB sample (closing one of the 1–3 misclassifications per run).
**Prerequisites.** Reliable extraction of `(complaint, fix)` tuples from existing logs. Corpus growth from `bag-traces/` will compound this.

### Idea 2: Few-Shot Prompt Distillation via RAG-over-Trace
**What it does.** At task start, embed the task description, retrieve top-3 similar past *winning* trial transcripts, inject 200-token compressed exemplars into BAG's system prompt. Gate by similarity threshold so off-task injection doesn't poison the prompt.
**Inputs.** All Claude Code session jsonl (winning trajectories), `bench/jobs/*` (labeled), `~/.codex/embeddings/` (substrate).
**Outputs.** A `taskShape → exemplar bundle` retrieval service plus prompt-injection hook.
**Effort tier.** T2.
**Expected value.** High. ICL with task-similar exemplars typically yields 3–8pp on coding benchmarks; we're realistically 2–5pp because our corpus is small. Token cost goes UP by ~500–1500 per call — net win only if it converts losses to wins.
**Prerequisites.** A "task signature" embedding function. The infra in §6 covers it.

### Idea 3: Token-Cost Regression Detector ("Prompt-Diff Replay Bench")
**What it does.** Whenever the BAG system prompt changes (`git diff` on the prompt files), automatically replay the trial corpus through both old and new prompt; alarm if median tokens-per-task spikes >10% or pass-rate drops on any cell.
**Inputs.** Prompt artifacts in `src/`, replay corpus, `bench/.bag/optimizer/dataset.jsonl`.
**Outputs.** A CI job that comments on PRs: "your prompt change costs +18% tokens with no measured pass-rate gain."
**Effort tier.** T2.
**Expected value.** High DX value, prevents accidental regressions. Pass-rate uplift: 0pp directly; *prevents* future losses.
**Prerequisites.** Real-replay-corpus plan partially complete.

### Idea 4: Failure-Cluster Auto-Discovery & Naming
**What it does.** Embed all verifier_complaint texts and ERROR-level log bodies; cluster (HDBSCAN); assign each cluster a hash + auto-generated label ("Cluster 12: Python venv path drift"); produce a weekly digest of top-10 clusters by frequency × recency.
**Inputs.** `~/.codex/logs_2.sqlite` ERROR/WARN bodies, `bench/jobs/*/verifier_log/`, `bag-traces/*/autonomous-trace.json`.
**Outputs.** A markdown digest + a cluster-id field appended to each replay record.
**Effort tier.** T2.
**Expected value.** Medium. Mostly an *organizational* lever — operators see what's actually failing — but a prereq for Ideas 1, 8, 11.
**Prerequisites.** None significant.

### Idea 5: Tool-Call Sequence Fitness Scoring
**What it does.** Mine n-gram patterns of tool calls (n=2,3,4) across all sessions; compute empirical win rate per n-gram. At runtime, when BAG is about to emit a tool-call subsequence with a measured low win rate (e.g., `Edit→Bash(test)` with no `Read` between them), inject a soft warning into the next-turn context.
**Inputs.** `~/.claude/projects/*` (rich tool data), `bag-traces/`.
**Outputs.** A scored n-gram lookup + a warning injector.
**Effort tier.** T3.
**Expected value.** Medium. Realistically 2–4pp uplift; the wins are usually variance-bound.
**Prerequisites.** Outcome labels on Claude Code sessions (currently missing — see §2.9).

### Idea 6: Cross-Agent A/B Replay ("Codex-trace through BAG")
**What it does.** Take a Codex trace, parse the user-visible turns into a synthetic task spec, replay through BAG's autonomous loop, A/B compare reward + tokens against Codex's actual outcome (where outcome is inferable).
**Inputs.** `~/.codex/history.jsonl`, BAG runtime.
**Outputs.** A second eval lane that's grounded in real user tasks (not synthetic TB).
**Effort tier.** T3.
**Expected value.** Strategically high — gives us a cross-agent benchmark not contaminated by TB. Pass-rate uplift: indirect.
**Prerequisites.** Outcome inference (heuristic: did the user thank, curse, or pivot afterward?).

### Idea 7: Personality / Preference Fingerprinting
**What it does.** Mine corrective user messages across `~/.codex/history.jsonl`, extract durable preferences ("WCAG AAA, not AA"; "no files when I say analyse"), persist as a `preferences.json` BAG loads at session start. Update incrementally on every new corrective turn.
**Inputs.** History jsonl, Claude Code project jsonls.
**Outputs.** A typed preference manifest + system-prompt injector.
**Effort tier.** T2.
**Expected value.** High DX uplift on long-running projects. Token savings: ~300/task by avoiding rejection cycles. Pass-rate uplift: 0pp on TB sample (preferences don't apply); large on real user work.
**Prerequisites.** Privacy redaction policy (cite v1-replay-redaction-policy todo).

### Idea 8: Metacognitive Pre-Flight ("Have I seen this before?")
**What it does.** At task receipt, BAG runs a quick lookup: "embed the task description, find top-3 similar past tasks, report their historical pass rate." If that rate is <0.5, switch to a more conservative mode (e.g., force tools mode + extra verification step). If >0.9, proceed with confidence.
**Inputs.** Trace embedding index, outcome labels.
**Outputs.** A pre-flight router decision augment.
**Effort tier.** T2 (given Idea 4 + Idea 2's substrate).
**Expected value.** Medium. The TB sample is small enough that priors are noisy, but on a 100+ task benchmark this is +3–6pp.
**Prerequisites.** Ideas 2, 4.

### Idea 9: Self-Eval Honesty Calibrator
**What it does.** When BAG self-scores its run (`autonomous-trace.json` self_eval_score), compare against the historical pass-rate of the cluster of similar past runs. If self_eval=0.95 but historical=0.60, attach a flag: `self_eval_overconfident` to the trace and surface in the operator digest.
**Inputs.** `bag-traces/*` self-eval, similarity index, outcome labels.
**Outputs.** A confidence-calibration column in the dataset; potential downstream use as a reward-shaping signal.
**Effort tier.** T2.
**Expected value.** High *strategic* value (prevents BAG from reporting fake wins to operators). Direct pass-rate uplift: 0pp.
**Prerequisites.** Idea 4 cluster structure.

### Idea 10: Cost Forecaster
**What it does.** Train (small regression model OR k-NN on labeled corpus) to predict expected tokens-to-completion given (task_description embedding, repo_signature, mode). Use the prediction to route to cheap vs expensive model.
**Inputs.** All labeled trials with token counts (most of `bench/.bag/optimizer/dataset.jsonl` has these).
**Outputs.** A prediction fn + routing hook.
**Effort tier.** T3.
**Expected value.** Big $ savings if we wire multi-model (master=Opus, local=Haiku — already named in `bag-vs-opus-direct.md`). Pass-rate: 0pp directly; cost reduction estimate 30–50% on easy tasks.
**Prerequisites.** Multi-model split in BAG runtime (separate work item).

### Idea 11: Automatic Anti-Pattern Mining ("Recipe Smell Detector")
**What it does.** Extract bash command sequences from sessions, score by win-rate. Sequences with rate <30% across ≥5 occurrences get tagged as "smell" and a brief explanation auto-generated by an LLM (e.g., "Editing without prior Read in 11/14 cases ended in user rejection — probably stale-context risk"). Inject smell warnings into the system prompt.
**Inputs.** Tool-call streams across all sources.
**Outputs.** A smell library + warning injector.
**Effort tier.** T3.
**Expected value.** Medium-high. Closely related to Idea 5 but at the *recipe* level, not n-gram. Pass-rate uplift: 2–5pp.
**Prerequisites.** Outcome labels.

### Idea 12: Behavior Cloning of Successful Sequences (T4 fantasy)
**What it does.** Fine-tune a small open model (Qwen-3.5 or similar — we already have the MLX infra in this repo) on (task_state → next_bash_command) supervised pairs from successful trajectories. Use it as a fast first-draft generator that BAG reviews-and-edits.
**Inputs.** All winning trajectories (need ~thousands).
**Outputs.** A local "draft tool-call" model.
**Effort tier.** T4.
**Expected value.** Speculative. 5–10pp pass-rate uplift IF data is sufficient and IF the overhead of two-model coordination is manageable. **Honest skepticism:** 85 trials is far below threshold; probably need to wait for corpus growth.
**Prerequisites.** ≥10× current corpus; T1–T3 ideas to extract clean training data.

### Idea 13 (bonus, weird): Trace-Diff "Sibling Twin" Replay
**What it does.** For every failed trial, find its closest *successful* sibling (same task, different run) and produce a structured diff: "Sibling did X at step 3, you did Y. Sibling won. Try X." Use this as a feedback bundle for GEPA proposer.
**Inputs.** Multi-run trial corpus.
**Outputs.** Structured fail-vs-win diff JSON; consumed by `.codex/plans/bleeding-agent-v1-autonomous-gepa-operations.plan.md`.
**Effort tier.** T3.
**Expected value.** Speculative-but-cool. The "minimal counterfactual" framing is a known win in program-repair literature. May produce a 1–3pp uplift.
**Prerequisites.** ≥3 runs per task (we don't always have this).

---

## 4. Recommended priority ordering

Reasoning: build substrates first, exploit them with cheap consumers, defer T4 fantasy until corpus is big enough.

**Tier 1 — build the substrate (week 1):**
1. **Idea 4** (Failure-cluster auto-discovery) — foundational; unblocks 1, 8, 9.
2. **Idea 7** (Persona / preference fingerprinting) — already half-done manually in `bag-trace-mining-deep-dive.md`. Mechanizing is the easy multiplier.
3. **Idea 1** (Verifier-signature library) — directly attacks the bench-trial misses we already understand.

**Tier 2 — leverage the substrate (week 2):**
4. **Idea 3** (Token-cost regression detector) — ties into the `v1-real-replay-corpus` plan; cheap once that exists.
5. **Idea 8** (Metacognitive pre-flight) — small wrapper around Idea 4 + Idea 2's index.
6. **Idea 9** (Self-eval calibrator) — cheap given Idea 4.

**Tier 3 — measurement-grade tooling (week 3–4):**
7. **Idea 2** (Few-shot RAG-over-trace) — full retrieval pipeline; the "prompt-distillation play."
8. **Idea 11** (Anti-pattern mining) — extends Idea 5's substrate.
9. **Idea 6** (Cross-agent A/B replay) — strategic eval lane.

**Tier 4 — research bets (month 2+):**
10. **Idea 5** (Tool-call n-gram fitness)
11. **Idea 10** (Cost forecaster) — needs multi-model split first.
12. **Idea 13** (Sibling-twin diff) — needs ≥3 runs per task.
13. **Idea 12** (Behavior cloning) — needs corpus to grow ≥10×.

**Why this ordering.** Ideas 1, 4, 7 are independently valuable AND act as substrates. Ideas 2, 8, 9 stack on them with low marginal cost. Ideas 5, 10, 12 are research bets that depend on data we don't fully have yet. Building the cheap, high-leverage stuff first lets us *generate* the data that makes the research bets viable.

**Cross-reference to existing plans.** Idea 3 is naturally consumed by `v1-real-replay-corpus`. Ideas 1+13 feed `v1-autonomous-gepa-operations` candidate generation. Idea 7 plus the redaction question fits cleanly into `v1-replay-redaction-policy`.

---

## 5. Anti-pattern warnings

Trace mining is a famously self-deluding discipline. Things to avoid:

### 5.1 Privacy / leakage
- `~/.codex/history.jsonl` and Claude Code jsonl contain real client work, secrets, paths. Anything that ships outside this machine — system-prompt exemplars, GEPA feedback bundles, public benchmarks — must be **redacted by default** (paths replaced, identifiers hashed, free-text scrubbed). The `v1-replay-redaction-policy` todo is non-negotiable; do *not* let preference-fingerprint outputs ship raw.
- Czech curses are signal; they are also identifying. Hash or strip before any cross-team use.
- Embedding indices leak: cosine-similar-search over redacted text can reconstruct content with surprisingly few queries. Treat the index itself as private.

### 5.2 Overfitting to a small corpus
- 80 trials × 10 tasks × stochastic Opus = noise dominates. Any "Idea X gives +5pp" claim from this dataset is unreliable until n≥500. Always show confidence intervals; never quote a point estimate without a sample size.
- The TB sample-10 is tiny. Optimizing prompts toward it risks gaming a private benchmark we then can't trust.
- Cross-task contamination: adjacent tasks in TB sample share idioms; a fix that "works" may be exploiting that, not generalizing.

### 5.3 Hallucinating insights from log noise
- ERROR clusters can be artifacts of one buggy code path firing 100× — looks like 100 incidents, is actually 1.
- Frequency ≠ severity. The 394 manifest WARN logs in `agent-trace-mining-report.md` are real but mostly cosmetic. Don't let frequency rank alone drive the priority list.
- LLM-generated cluster labels (Idea 4) lie. Always sample 5 members of a cluster manually before believing the auto-name.

### 5.4 Self-improvement loops that game their own metrics
- If GEPA optimizes against pass-rate on the trial corpus, and the trial corpus is *also* the source of fix-pattern injections (Idea 1), you get circular reinforcement. Hold-out discipline (`v1-replay-split-discipline`) is critical.
- Self-eval calibrator (Idea 9) must NOT be used as a reward signal until it's been independently validated. Otherwise BAG learns to be more confident, not more correct.

### 5.5 Operator overload
- Each new "smell warning," "preference reminder," and "metacognitive flag" injected into the prompt costs tokens AND attention. Three separate experiments adding 200 tokens each is a 600-token tax that can outweigh the wins. Build a single injection layer with priority budgeting.
- The system-prompt sprawl is real. Limit the total derived-content budget (e.g., max 800 tokens of mined content per turn).

### 5.6 Schema drift across sources
- Codex jsonl changed schema between versions. Claude Code jsonl ditto. Any pipeline that reads them must tolerate missing fields. Use Pydantic or zod with optional fields; do not assume.

### 5.7 The gold-plating trap
- Building Idea 12 (behavior cloning) before Ideas 1, 4, 7 means spending T4 effort on a corpus that's still T1-quality. Don't.

---

## 6. Infrastructure required

The minimal substrate for this to be a *continuous* capability instead of a one-shot study.

### 6.1 Daily extractor cron
- A single script that runs on a schedule (cron / launchd / GitHub-Actions-on-self-hosted), re-derives `bench/.bag/optimizer/dataset.jsonl` plus the unified session-fact-table from raw sources.
- Outputs are timestamped + checksummed for reproducibility.
- Should be incremental (resume from last-processed offset).

### 6.2 Unified session-fact-table
- One Parquet (or DuckDB) table with columns `(session_id, source, project, task_signature, started_at, n_turns, n_tool_calls, terminal_outcome, tokens_in, tokens_out, embedding_id, redaction_status)`.
- Joins across `~/.codex/history.jsonl`, `~/.codex/logs_2.sqlite`, `~/.claude/projects/*`, `bench/jobs/*`, `bag-traces/`.
- The keystone artifact for §3 — every idea reads from it.

### 6.3 Embedding index (own, not Codex's)
- Use OpenAI `text-embedding-3-small` or local equivalent (we already have MLX infra in this repo).
- Index: task descriptions, verifier complaints, command outputs, user corrections.
- Storage: a single FAISS or hnswlib file rebuilt nightly. Same shape as Codex's `~/.codex/embeddings/` but with our manifest.

### 6.4 Replay harness
- Reads a normalized replay case from the fact-table; re-runs the BAG agent loop in a sandbox; compares to baseline outcome.
- This is what `v1-replay-runner-integration` is for. Cite that plan.

### 6.5 Prompt-fingerprint registry
- For every BAG system prompt + tool descriptions, compute a stable hash; store metadata (timestamp, git sha, hash). When a hash changes, trigger Idea 3.
- Append-only; never delete.

### 6.6 Mined-pattern registry
- A versioned JSON store of the verifier-signature library (Idea 1), preference manifest (Idea 7), smell library (Idea 11), exemplar bundles (Idea 2).
- Each entry: `(id, source_traces, evidence_count, first_seen, last_seen, status: candidate|active|retired)`.
- Promotion / retirement gated by Idea 3's regression check.

### 6.7 Operator digest (weekly markdown)
- Auto-generated. Top-10 failure clusters, new preferences detected, prompt regressions averted, candidate patterns awaiting review.
- The single human-readable surface for "is the trace-mining loop healthy?"

### 6.8 Redaction layer
- Ingestion-time scrubber: regex-based path masking, name detection, hash-replace.
- Two retention modes: `local-only-raw` and `shareable-redacted`. Default to local-only.
- Required by §5.1. Cite `v1-replay-redaction-policy`.

**Build order for infra.** 6.1 + 6.2 + 6.8 first. 6.3 + 6.5 next. 6.4 + 6.6 + 6.7 follow.

---

## 7. The "if we did all 12" thought experiment

Suppose 30 days from today (2026-05-31), we shipped Tiers 1–3 (Ideas 1, 2, 3, 4, 6, 7, 8, 9, 11) plus the §6 infra. Tier 4 still on backlog.

### What BAG looks like
- **At session start:** preferences loaded, prompt fingerprint logged, similar-past-tasks lookup runs.
- **Pre-flight:** classifier consults Idea 8 (have-I-seen-this); if low historical pass rate, escalates to a more conservative mode. Top-3 winning exemplars (Idea 2) injected, capped at 600 tokens.
- **Mid-flight:** smell warnings (Idea 11) injected on suspect tool sequences; verifier-signature library (Idea 1) consulted on every verifier complaint.
- **Post-flight:** self-eval calibrated (Idea 9) against historical priors; trace persisted with cluster labels (Idea 4); preferences updated if user corrects.
- **Continuous:** weekly digest, prompt-regression CI, fact-table refresh.

### Honest expected pass rate on TB sample
- Today's best mode: 80% (tools), 70% (auto). Opus-direct: 90%.
- Lift from Tier 1+2+3 ideas combined, with realistic *not* additive multipliers (because they share variance):
  - Idea 1 alone: ~+3pp on the misclassification-cluster failures.
  - Idea 2 alone: ~+2pp on monolithic-complex tasks.
  - Idea 8 + 9: routing improvements, ~+1–2pp on auto mode.
  - Combined (with correlation discount): **+5–7pp realistic, +10pp optimistic.**
- Projected BAG-on-TB-sample pass rate after 30 days: **85–87% (auto mode)**, vs Opus-direct 90%. We close most of the gap but not all of it.
- TB sample is the wrong battleground. The bigger story is on full TB-2.0 (89 tasks) where the substrate compounds.

### Cost / token story
- Idea 2 increases per-call tokens by ~500–1500.
- Idea 10 (NOT in the 30-day plan) is what would *reduce* costs via multi-model.
- Net: per-task tokens probably +10–20% in 30 days. Pass-rate uplift must justify it, or we burn money for nothing. **This is the single biggest risk** in the plan.

### Novel capabilities BAG gains
- Self-knowledge ("I've struggled with this kind of task before").
- Persistent memory of user preferences across sessions.
- Operator-visible failure-mode trends.
- Regression-safe prompt iteration.
- Cross-agent grounding (Codex traces become BAG eval data, not just Codex's).

### What stays uncertain at day 30
- **Generalization.** Everything is mined from my (one operator's) traces. Exporting to other users requires a fresh corpus.
- **Behavior-cloning viability** (Idea 12). Won't know until corpus is 10× bigger.
- **Whether the prompt-tax outweighs the wins.** Honest answer: probably borderline. The Idea 3 regression detector is the discipline that catches the bad cases.
- **Whether the redaction is good enough** to share findings outside this machine. Adversarial review needed before any external publication.
- **Whether GEPA on this data converges.** It might propose noise as signal. Cite `v1-gepa-eval-gates`.

### One-sentence honest summary
*Trace mining is a multiplier, not a magic wand: it gives us the discipline to capture and reuse what we already learn, but it cannot substitute for the model itself doing the work. The 30-day plan above buys us roughly the gap to Opus-direct on small benchmarks, plus a real moat on long-horizon and user-personalized work where Opus alone cannot remember.*

---

## Appendix: cross-reference to existing artifacts

- `docs/agent-trace-mining-report.md` — already mined 5 actionable system-health findings from SQLite. Idea 4 mechanizes its method.
- `docs/bag-trace-mining-deep-dive.md` — already mined 10 durable user preferences and 5 cross-project anti-patterns by hand. Idea 7 mechanizes its method.
- `docs/bag-tb-tool-use-vs-dag-tools.md` — established that mode choice matters per task shape. Idea 8 mechanizes the routing decision.
- `docs/bag-vs-opus-direct.md` — established that BAG currently loses to Opus-direct; trace mining is the moat candidate.
- `.codex/plans/bleeding-agent-v1-autonomous-gepa-operations.plan.md` — consumes Ideas 1, 13 as candidate-generation inputs.
- `.codex/plans/bleeding-agent-v1-real-replay-corpus.plan.md` — IS the substrate for Idea 3, Idea 6.
- `.codex/plans/bleeding-agent-v1-runtime-orchestration.plan.md` — names the durable-state need that Idea 1+11 require.

What is *not* in those plans and is genuinely new to this document:
- Verifier-signature library (Idea 1).
- Token-cost regression detector as a *first-class CI gate* (Idea 3).
- Failure-cluster auto-naming (Idea 4).
- Cross-agent A/B replay using Codex history as the baseline (Idea 6).
- Self-eval honesty calibrator (Idea 9).
- Sibling-twin diff for GEPA feedback (Idea 13).

Pick from those when budgeting the next sprint.
