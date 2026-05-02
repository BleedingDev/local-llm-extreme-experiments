# Proposal H — User-Taste Judge: An LM-as-Judge That Grades Like *You*

**Author:** Brainstorm Round-2 Member #H
**Status:** Novel — round 1 produced a *persona generator* (acts like you); this is the dual *persona discriminator* (grades like you). Different artefact, different loss, different deployment surface.

---

## TLDR

- Public autoraters score for **average taste**; this user has **idiosyncratic taste** (pnpm-not-bun, grep-not-rg, Czech corrections, Bash-Read recovery, MLX-first). A judge fit on *their* correction signals predicts acceptance with much higher fidelity than a generic critic.
- Training data is **already in the corpus**: positive ("perfect", "thanks", "skvělé") follow-ups, `user_corrected` negatives paired with the user's gold-label criticism message, plus auto-mined R1-bad / R2-good contrastive pairs from failure-recovery traces.
- Three tiers: **Light** (Anthropic prompt-only critic seeded with persona fingerprint), **Medium** (Qwen2.5-1.5B + LoRA pairwise reward model), **Heavy** (full DPO model on winner/loser pairs).
- Drops into existing GEPA loop as `TraceAdapter._score`, into BAG runtime as a per-candidate gate, and into fine-tuning as a learned reward — one artefact, three call sites.

## Path

`trace-gepa/proposals/user_taste_judge.md`

## Training Data

| Class | Source | Volume estimate |
|---|---|---|
| Positive | next user turn ∈ {"perfect", "thanks", "skvělé", "ship it", silent-accept after N turns} | ~thousands |
| Negative + critique | records flagged `user_corrected` — the correction message itself is the gold rationale | hundreds–low thousands |
| Contrastive pairs | (R1=rejected action, R2=accepted action) auto-mined from same-session retries | thousands |

Critique text is gold: it explains *why* in the user's own words ("use pnpm", "no, read first", "počkej, ne tak"). Use it as both rationale supervision (Light/Medium) and DPO chosen/rejected signal (Heavy).

## Model Recommendation — **Medium**

LoRA-tuned Qwen2.5-1.5B as a pairwise reward model. Rationale: Light prompt-only is brittle on edge cases and re-pays inference cost forever; Heavy DPO needs more cleaned pairs than the corpus reliably yields and over-commits before we know whether judge-quality even moves the GEPA needle. Medium gives a calibrated 0–1 score, fits on the M-series box already running MLX, runs locally (no API spend in the GEPA hot loop), and the LoRA is cheap to refresh as habits drift.

## Evaluation

Hold out 100 `user_corrected` records + 100 positives. Judge passes if it (a) agrees with the user 70%+ on accept/reject and (b) its 1-line rationale lexically overlaps the gold critique (BLEU-1 ≥ 0.3) on the negatives.

## Use Cases

1. **GEPA reward** — replace `TraceAdapter._score` with judge score; re-run optimisation against personalised signal instead of generic correctness.
2. **BAG runtime ranker** — sample N candidate actions, judge each, execute the top-1; cheap personalisation without retraining the actor.
3. **Fine-tune reward** — RLHF/DPO signal for any future persona-cloned actor (pairs cleanly with Proposal C).

## Effort & ROI

~1 week Light, ~2 weeks Medium incl. eval harness. ROI is multiplicative: every downstream optimiser (GEPA, BAG, persona LoRA) currently uses a generic or hand-tuned reward; replacing it once benefits all three.

## Self-critique

A judge trained on the user's *past* corrections will faithfully re-enforce their current habits — including the bad ones — and will resist exactly the kind of agent improvement that would have *taught* them something new.
