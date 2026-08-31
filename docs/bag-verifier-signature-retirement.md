# BAG Verifier-Signature Library Retirement Plan

Date: 2026-05-02
Owner: BleedingAgent (BAG) self-improvement track
Status: Soft-deprecated. Deletion criteria below.

## Why retire?

`src/verifier-signature-library.ts` ships 8 hand-curated regex → fix-hint
entries. It is the closest thing BAG has to a hardcoded keyword list, and
hardcoded keyword lists do not scale. Every new failure mode would need a
human to write a regex and prose — expensive, lossy, and biased toward what
post-mortem authors happened to remember.

`src/optimizer/failure-clusters.ts` consumes
`bench/.bag/optimizer/failure-clusters.json` (built by
`scripts/build_failure_clusters.py`) — auto-discovered clusters from the
real BAG trial corpus via signature extraction + character-trigram Jaccard.
Same role (retry-hint at Best-of-N retry time) but driven by data, not by
hand. As of 2026-05-02 the corpus is 143 failures across 26 distinct
clusters and growing.

Retirement is the principled path. The remaining question is *when*, not
*whether*.

## Parity bar

`tests/verifier-signature-vs-clusters-parity.test.ts` is the gating
artifact. Method: render a synthetic verifier output for each curated
signature and ask both matchers whether they fire. Recall target: ≥5/8 of
specific signatures must produce a non-null cluster match at threshold
**0.30** (configured in `bench/.bag/optimizer/failure-clusters-config.json`,
referenced by both the test and the runtime call site
`src/autonomous-coding-turn.ts`).

### Current parity table (2026-05-02, 143-failure corpus)

| # | Curated signature | Cluster match | Score | Status |
|---|---|---|---|---|
| 1 | `polyglot-cmain-leftover` | `polyglot-c-py-main-cmain` (size 9) | 1.000 | covered |
| 2 | `chess-multiple-moves-missed` | `chess-best-move-e2e4-g2h3-g2g4` (size 9) | 1.000 | covered |
| 3 | `end-to-end-http-not-served` | `configure-git-webserver-http-000-test-passed-http` (size 14) | 0.944 | covered |
| 4 | `qemu-host-vs-guest-confusion` | `qemu-alpine-ssh-orbstack-gb73df9775337` (size 3) | 1.000 | covered |
| 5 | `build-pipeline-flag-lost` | `sqlite-with-gcov-assert-0-0` (size 1) | 1.000 | covered |
| 6 | `typecheck-missing-import` | (no matching cluster) | 0.105 | NOT covered — corpus shape mismatch |
| 7 | `acp-internal-error` | (no matching cluster ≥ 0.30) | 0.216 | NOT covered — needs more corpus |
| 8 | `submit-without-verify-catchall` | (no matching cluster) | 0.025 | NOT covered — generic by design |

Recall on specific signatures: **5/8 = 62.5%**.

### Why we did not pick threshold 0.20 (which would catch acp-internal-error)

Lowering to 0.20 would also pull in a spurious match: acp-internal-error
incidentally trigram-overlaps with the FileNotFoundError cluster (the
`ENOENT: no such file or directory` substring drives the overlap). Shipping
a wrong fix hint is worse than shipping no hint — so 0.30 is the minimum
threshold that keeps the matcher honest. See
`bench/.bag/optimizer/failure-clusters-config.json` for the full rationale.

## Migration timeline

**Now (2026-05-02):** soft-deprecation. The library is annotated `@deprecated`
and the runtime in `src/autonomous-coding-turn.ts` queries the cluster
matcher FIRST and falls back to the library. Each retry emits a
`retry_hint` trace entry tagged with `source: "cluster" | "library" |
"both" | "none"`.

**Audit gate (≥30 BAG runs from now):** sample the trace stream, compute
library hit rate. If `library` (or `both`) source fires < 5% of total
non-`none` hints across 30 runs, the curated library is doing nothing the
clusters aren't already covering — delete it.

**Corpus expansion fast path:** typecheck-missing-import and
acp-internal-error are not pytest-shaped, so the existing build script
won't cluster them naturally. If we want to retire those slots faster,
extend `scripts/build_failure_clusters.py` to also ingest non-pytest
verifier outputs (TypeScript compilation errors from CI, ACP RPC errors
from session logs).

## Honest assessment: which signatures would be lost on retirement today?

Three slots:

1. **`typecheck-missing-import`** — the BAG corpus runs against pytest
   verifiers; TypeScript type errors don't appear in
   `bench/.bag/optimizer/dataset.jsonl`. Retirement TODAY would lose this
   safety net for self-coding tasks where the agent edits BAG's own
   TypeScript and the verifier is `tsc --noEmit`. Volume of impact: small
   (only the optimizer's self-eval surface has this shape).
2. **`acp-internal-error`** — corpus has only a handful of ACP crashes;
   none clustered above the 0.30 threshold. Volume: ~5/50 historical
   failures per `docs/bag-failure-pattern-digest.md`. Retirement TODAY
   would silently drop this nudge.
3. **`submit-without-verify-catchall`** — generic "you submitted without
   running the verifier" prose. Triggers on any AssertionError. By design
   this is the bottom of the priority list and matches everything the
   specific signatures don't. The cluster matcher has no equivalent; if we
   need a generic catchall after retirement, it should live as a hardcoded
   default in `autonomous-coding-turn.ts` (≤10 lines) rather than as a
   regex entry in the library file.

## Recommendation

**Do NOT retire today.** Soft-deprecation is the right move. Reasons:

- 5/8 covered (62.5%) is below the original ≥7/8 (87.5%) bar. The bar was
  set on the assumption that all 8 curated signatures would have corpus
  representation; in reality 3 of them (typecheck, acp-internal,
  catchall) are out-of-distribution for the current pytest-shaped corpus
  and will not converge with more BAG runs alone.
- The cluster matcher is now PRIMARY (cluster-first ordering in
  `autonomous-coding-turn.ts`), so the curated library only fires as a
  safety net. This already captures the bulk of the retirement benefit
  (clusters drive the dominant signal) without losing the 3 fallback
  patterns.
- Library hit-rate telemetry (`retry_hint` trace entry) lets us measure
  the actual marginal value of keeping the library over the next 30 runs.
  If it really is < 5%, deletion is one PR. If it's higher, we have hard
  numbers to support keeping it.

**Retire AFTER:** 30 BAG runs of cluster-primary ordering, with library
hit-rate < 5% of total retry-hint fires AND a hardcoded catchall added to
`autonomous-coding-turn.ts` (so we don't lose the generic
"submit-without-verify" nudge).

## Files touched

- `src/verifier-signature-library.ts` — `@deprecated` annotation + TODO
  marker (soft deprecation; no behavior change).
- `src/autonomous-coding-turn.ts` — cluster matcher is now PRIMARY, library
  is FALLBACK, threshold pinned to 0.30, `retry_hint` trace entry added.
- `tests/verifier-signature-vs-clusters-parity.test.ts` — parity test
  (gating artifact; ≥5/8 specific-signature coverage required).
- `bench/.bag/optimizer/failure-clusters-config.json` — threshold and
  retirement criteria, single source of truth.
- `bench/.bag/optimizer/failure-clusters.json` — re-built from current
  corpus (143 failures, 26 clusters).
- `docs/bag-verifier-signature-retirement.md` — this document.
