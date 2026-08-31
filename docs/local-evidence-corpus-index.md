# Local Evidence Corpus Index

Generated for graph `local-evidence-flywheel-v1` on 2026-05-04.

## Contract

The canonical index is `.bag/evidence/index.jsonl`. Each line is one JSON object conforming to `.bag/evidence/index.schema.json` with `schemaVersion` set to `local-evidence-index.v1`.

The index has two record kinds:

- `source`: metadata for a local evidence source, derived artifact, or policy artifact.
- `slice`: a named operational view over source records.

Index records must not copy raw evidence rows. A source record may include file paths, row counts, byte sizes, schema fingerprints, label/count summaries, retention tiers, parent evidence IDs, derived-from links, and quality caveats. Consumers must read raw evidence from the referenced paths only after applying the retention and privacy policy.

## Identity And Lineage

`evidenceId` is stable within this repository and uses the `evidence.*` namespace. `sliceId` uses the `slice.*` namespace.

Lineage fields:

- `parentEvidenceIds`: direct parent corpus or source where lineage is known, such as sanitised mirrors pointing back to raw datasets.
- `derivedFrom`: upstream evidence used to build a derived artifact, such as optimizer candidates, failure clusters, RAG indexes, or schema audits.

Rows that do not expose an `id` field declare an alternate `metadata.primaryKey` when known. Examples include counterfactual rows keyed by `record_id`, replay rows keyed by `indexRecordId`, and optimizer rows keyed by job/task/trial metadata.

## Named Slices

- `slice.optimizer.train-dev-holdout-candidates`: sanitised action/recovery/counterfactual corpora, split manifests, benchmark definitions, and local optimizer dataset candidates.
- `slice.real-acp-failures`: canonical visible ACP failure index, visible run artifacts, and adapter replay export.
- `slice.terminal-bench-runs`: benchmark job result corpus plus derived optimizer dataset and failure clusters.
- `slice.edit-strategy-evidence`: edit-safety benchmark views, Aider polyglot results, and benchmark job edit/tool evidence.
- `slice.claude-codex-comparison`: Claude Code style corpus, Codex GPT-5.5 corpus, unified raw dataset, and optimized prompt lineage.
- `slice.derived-indexes`: RAG metadata indexes plus schema audit and retention policy artifacts.

## Quality Rules

Use sanitised mirrors for optimizer inputs and sharing by default. Raw-local evidence remains local-only until privacy review passes.

Use split manifests where available and keep split leakage checks blocking. The current audit reports no split bucket overlaps. The recovery sanitised mirror has equal row counts but one missing ID and one extra ID relative to raw, so consumers must keep that caveat visible.

Exclude duplicate-risk legacy datasets from canonical training manifests unless they are deterministically deduped. Derived RAG indexes are operationally useful but have duplicate IDs and are not canonical training inputs.

Do not use `latest` optimized-prompt symlinks as canonical truth. Use timestamped run metadata and scorecards.

## Verification

Required parse checks:

```sh
jq empty .bag/evidence/index.schema.json
while IFS= read -r line; do jq -e . >/dev/null <<<"$line"; done < .bag/evidence/index.jsonl
test -s docs/local-evidence-corpus-index.md
```

The index was built from `docs/local-evidence-inventory.md`, `docs/local-evidence-quality-audit.md`, `docs/local-evidence-retention-policy.md`, `.bag/evidence/schema-audit.json`, `.bag/evidence/retention-policy.json`, and direct local metadata checks with `stat`, `wc`, `find`, `ls`, and `jq`.
