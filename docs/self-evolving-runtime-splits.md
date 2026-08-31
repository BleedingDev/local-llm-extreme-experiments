# Self-Evolving Runtime Splits

`trace-gepa/data/splits_v2.json` is currently an audited ID bucket, not a
complete train/dev/hidden-holdout manifest. Optimizer consumers must project
those stable IDs through a sealed deterministic policy before using the corpus
as optimizer evidence.

`src/optimizer/split-projection.ts` provides the in-memory projection primitive:

- Stable IDs are normalized, sorted, and rejected if duplicated.
- Each ID is assigned by `sha256-threshold.v1` over the ID, projection seed,
  projection version, and optional source ID.
- The default weight is 70 train / 15 dev / 15 hidden-holdout, with explicit
  ratios supported for fixtures or future sealed manifests.
- Output carries seed, version, source ID, algorithm, ratios, split counts, and
  a stable projection ID.

Visibility is intentionally fail-closed:

- Candidate generation, prompt drafting, policy synthesis, retrieval, failure
  clustering, training, and development evaluation may read only `train` and
  `dev` labels.
- Frozen-candidate hidden-holdout evaluation may read only `hidden-holdout`.
- Promotion gates and aggregate audit/reporting consumers may read hidden labels
  only as evaluation or aggregate gate inputs, not as candidate-generation
  feedback.

The helper also exposes duplicate and split-leakage checks for small fixtures.
These are meant to mirror the local evidence caveats: split leakage blocks
promotion, hidden-holdout content must never become optimizer input, and
single-bucket action manifests require an external deterministic projection
before any holdout claim is made.
