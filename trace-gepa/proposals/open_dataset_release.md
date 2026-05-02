# Proposal: Open Release of trace-gepa Dataset + Benchmark

**Author:** Brainstorm Team Member #E
**Status:** Draft for review
**Novelty:** Earlier waves used the data internally; this turns it into a **public scientific resource**.

---

## 1. Hypothesis

Per Bench Agent #12's comparative positioning, **no public benchmark exists for "agent action selection from real trace prefixes"**. SWE-bench targets repo-level patches; AgentBench targets synthetic environments; ToolBench targets tool-call accuracy. None expose authentic, long-horizon Claude-Code-style action distributions. Releasing our **175-task benchmark + ~30K-record trace dataset** would:

- **Fill a genuine gap.** First public corpus of real coding-agent action traces with prefix-conditioned next-action labels.
- **Drive citations + adoption.** Workshop paper at ICLR/NeurIPS LM-eval tracks plausibly hits 20-50 cites in year 1; HF download counter is a public-facing signal.
- **Enable independent reproduction.** Third parties can validate or refute the user's GEPA / DSPy findings, which strengthens (or correctly weakens) the claims.

## 2. Privacy / Sanitisation Requirements (CRITICAL)

The traces contain **personally identifying material** that internal use tolerated but public release cannot:

- **Home dir paths** — `/Users/satan/...` appears in tool outputs, error stacks, and file arguments throughout. Must be globally substituted to `/Users/user_a/...` (or `/home/user_a/...`).
- **API keys** — already redacted to `<REDACTED_KEY>` at extraction time, but a second-pass regex sweep (high-entropy strings, `sk-...`, `gh[ps]_...`, AWS-style) is mandatory before release.
- **Repo + project names** — `supergemma-dflash-ddtree-mlx`, `trace-gepa`, and similar reveal the user's research direction. Map to opaque IDs (`project_001`, `project_002`).
- **Proper nouns** — collaborator names, internal tool names, Slack handles. Need NER pass + manual review of top-frequency tokens.
- **Differential-privacy thinking.** Even after string sanitisation, **stylometric and topical fingerprints remain**: the user's commit-message cadence, distinctive tool-use patterns (e.g., heavy `rg` + Read), and project-specific vocabulary (DDTree, DFlash, GEPA) could re-identify. Mitigations: (a) drop the lowest-frequency 5% of tasks where N=1 examples dominate, (b) shuffle task ordering to break temporal correlation, (c) **explicit consent statement from the user** acknowledging residual re-identification risk is the honest baseline, since formal DP is impractical for trace text.

## 3. Concrete Proposal

| Output | Location | Format |
|---|---|---|
| Dataset card | `huggingface.co/datasets/<user>/trace-gepa-bench` | HF standard + Datasheet (Gebru et al. 2018) |
| 175-task benchmark | same repo, `bench/` split | JSONL with full schema |
| Paper draft | `trace-gepa/paper/draft.tex` | 4-8 pp, ICLR/NeurIPS workshop |
| Repro code | existing harnesses + verifiers | README polish only |

**Licensing:** MIT (code), CC-BY-4.0 (data).

## 4. Implementation Steps

1. **Sanitisation pass** — ~1 day (regex sweep + NER + manual top-token review + DP-style frequency trim).
2. **Datasheet + dataset card** — ~half a day.
3. **Paper draft** — ~3-5 days (the real cost).
4. **HF Hub upload** — ~1 hour.

Total: **~5-7 user-days**, dominated by paper writing.

## 5. ROI / Honest Critique

**For:** unique public artefact, citation signal, profile boost, validates findings via reproduction.
**Against:** 5-7 days of writing time the user might prefer to spend on next experiments; residual privacy risk even post-sanitisation; obligation to maintain the dataset / answer issues. **If the user values privacy or velocity over external visibility, this is the wrong investment.** Recommend a **go/no-go decision before sanitisation begins**, since sanitisation is the only step that's hard to undo.

---

**Self-critique:** This proposal underweights the long tail of dataset-maintenance burden (issues, takedown requests, schema-versioning) that turns a "one-week project" into an indefinite obligation.

