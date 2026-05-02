# Opus → Haiku Behavioural Distillation

## TLDR
- **Hypothesis:** With 1 user / 1 dominant repo, the action-policy surface is narrow enough that a 3B-class student fine-tuned on (context → opus_action) pairs hits ≥90% of seed-Opus quality at ~1/10 the cost — paying back re-labelling spend in <1 week of BAG runtime.
- **Novelty vs sibling proposals:** unlike `behavioral_cloning.md` (clones the *user*) and `cross_agent_transfer.md` (transfers across agents), this clones a *stronger model's policy* onto a *weaker model* using the user's own context distribution as the support set — a personalised task-LM, not a personalised user-LM.
- **Why now:** the trace dataset is already context-aligned to the deployment distribution; we don't need synthetic prompts, just gold relabelling. This is the cheapest distillation setup we'll ever have.
- **Kill criterion:** if distilled-Haiku ≤ seed-Haiku +5pp on held-out coding tasks, abandon — the gap is in reasoning, not surface mimicry.

## Plan
1. **Context selection (~5K).** Filter `data/dataset_v2.jsonl` to `label=good`, dedupe by context-hash, stratify by tool-call type so rare actions (e.g. `EnterWorktree`, MCP calls) aren't drowned by `Read`/`Edit`. Hold out 500 for eval.
2. **Gold relabelling.** Re-run Opus 4.7 against each context with the GEPA-optimised system prompt; capture full assistant turn (text + tool calls) as `opus_action`. Use prompt caching on the system prompt — cache hit rate should be >95%, dropping cost to ~$50.
3. **Student training.** Two tracks in parallel: (a) Anthropic fine-tuning API on Haiku if available; (b) MLX-LoRA on Qwen2.5-3B-Instruct as open proxy. Loss = next-token CE on `opus_action` only (mask context). LoRA r=32, 3 epochs, lr 1e-4.
4. **Eval.** Three conditions on held-out 500: seed-Opus, seed-Haiku, distilled-student. Metrics: (i) action-match F1 vs Opus-gold, (ii) downstream task success on a 50-task curated coding eval (independent of trace replication — guards against tic-overfitting), (iii) cost-per-successful-task.

## Eval bar
Distilled student ≥ 0.90 × seed-Opus on (ii); strictly > seed-Haiku on (ii) and (iii). If only (i) clears, we've taught surface mimicry — ship a downgrade, not an upgrade.

## Cost
~$50–75 for relabelling (5K × ~3K input tokens cached + ~500 output) + 2–4 GPU-hours LoRA. Under $150 total.

## Path
`trace-gepa/proposals/opus_haiku_distillation.md`

## Self-critique
Distillation on this user's contexts risks teaching Haiku Opus's *stylistic tics on this repo* rather than transferable judgement — the independent 50-task coding eval is the only thing standing between us and a confidently-wrong cheaper model.
