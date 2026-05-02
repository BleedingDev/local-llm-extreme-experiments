# Proposal: Behavioral Cloning via On-Device SFT (LoRA)

**Author:** Brainstorm Member #A    **Scope:** SFT on (context -> ideal action) pairs, MLX-only.

## 1. Hypothesis

GEPA optimised the *system prompt* — a fixed token preamble — over a frozen base model. That ceiling is real: a prompt cannot teach the model **new motor patterns** (tool-argument formatting idioms, when to chain `Read+Edit` vs `Write`, when to *stop* and ask, recovery behaviours after a failed `Bash`). Those patterns live in millions of weights, not in 2 KB of preamble. Behavioural cloning hypothesises that **the residual error after GEPA is a *capability* gap, not an *instruction* gap**, and is therefore addressable only by changing the weights. Concretely: if two contexts demand different tool calls but share surface form, no prompt rewording disambiguates them — only a tuned policy can. SFT also amortises: the prompt becomes shorter (cheaper inference) because behaviour is internalised.

## 2. Concrete Proposal

**Base model.** `Qwen2.5-3B-Instruct` 4-bit MLX as primary (best published tool-calling at 3B; native chatml + JSON-mode; ~2 GB on disk; fits in <8 GB unified memory during LoRA). Fallback: `Qwen2.5-1.5B-Instruct-4bit` if 3B training is too slow on M2/M3. **Reject:** Phi-3-mini (weak tool-calling, MS-flavoured chat template friction), Gemma-2-2B (no native tool schema, MLX-LM support lags).

**Training data.** Primary corpus: `dataset_toolcalling.jsonl` filtered to `label in {good, user_confirmed, user_corrected}` and `quality_score >= 0.65`. Empirical calibration note: at `>=0.7` the corpus is **empty** (0/4045) — the existing scorer caps below 0.7 — so the brief's threshold needs lowering. At `>=0.65` we get **~645 records**; augment with `dataset_corrections.jsonl` (gold) and the `recovery_action` field of `dataset_recovery.jsonl` (treat the recovery as the target, drop the failed pair). Target ~1.5–2 k chat-formatted examples — small enough that LoRA won't overfit a 3 B base in 2–3 epochs. The `ideal_action_hint` field is the supervision target when `observed_action` was bad; otherwise `observed_action` itself.

**Recipe.** mlx-lm LoRA: rank=16, alpha=32, dropout=0.05, target=`q_proj,k_proj,v_proj,o_proj`. lr=1e-4 cosine, warmup=50, batch=4 (grad-accum=4 -> eff 16), seq=4096, epochs=3. Loss: standard causal LM masked to assistant tokens only.

**DPO is a phase-2 stretch goal, not phase 1.** `dataset_recovery.jsonl` gives natural `(failed, recovery)` preference pairs — perfect for DPO once SFT has converged. Doing DPO first on a base model usually destabilises tool-calling format, so sequence matters.

## 3. Evaluation Plan

Run `bench/run_mlx.py` on the full 175-task `benchmark_tasks_full.jsonl`. Three arms, identical decoding params:

| Arm | Prompt | Weights |
|---|---|---|
| A (control) | seed prompt | base 4-bit |
| B (GEPA baseline) | GEPA-optimised prompt | base 4-bit |
| C (this proposal) | seed prompt | SFT-LoRA fused 4-bit |
| D (stretch) | GEPA prompt | SFT-LoRA fused (do they compose?) |

Headline metric: tier-1 verifier pass rate. Secondary: tool-name match, structural-JSON validity, latency (tokens/s) — SFT should *win* on latency because the tuned model needs less prompt scaffolding. **Stop-rule:** if C does not beat A by >=5 pts absolute, the SFT thesis is falsified for this data scale and we abandon.

## 4. Effort Estimate

- Data shaping into chat format: **3 h** human.
- LoRA training script (mlx-lm CLI is sufficient, no custom code): **1 h** human.
- Training: **~2–4 h GPU** on M3 Max for 3 B / 3 epochs / 2 k samples; **~45 min** for 1.5 B.
- Eval (3 arms x 175 tasks via existing harness): **~2 h GPU**.
- Analysis + writeup: **2 h** human.
- **Total: ~6 h human, ~6 h Mac GPU.** Single-day spike.

## 5. ROI vs the Other 4 Proposals

**Why fund it.** Of the five angles, this is the only one that produces a *deployable artefact* (a fused 4-bit MLX model) rather than a research signal. It also stress-tests the claim that the trace dataset has any teaching value at all — a negative result is itself diagnostic of dataset quality and de-risks every future plan that assumes the data is good. Cost is low (one Mac-day).

**Strongest critique against myself.** 645–2000 examples is **borderline-tiny** for SFT to outperform a well-GEPA'd prompt; the LoRA risks memorising idioms of `cc`/`codex` traces (e.g. specific filepaths under `/Users/satan/...`) and *regressing* on the synthetic 70-task slice of the bench. If C beats A but loses to B, we've spent a day proving GEPA was the right lever. Mitigations: hold out a synthetic-only eval slice and path-redact the training contexts — but the risk is real and I would not bet >50 % on C beating B at this data scale.

---

**Owner files:** this file only. No code, no data mutations.
