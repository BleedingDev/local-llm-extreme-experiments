# Drift Detection — Temporal Delta over Trace Behaviour

**Round 6 / Member V.** Persona fingerprint and CLAUDE.md auto-gen are *snapshots*. Traces span Aug 2025 → May 2026 (9 months for Codex, ~10 for Claude). Behaviour drifts: bun → pnpm, jest → vitest, tmux-popup workflow adopted in Feb 2026, etc. A snapshot extracted in October is stale by April. Drift detection makes re-extraction *precise* instead of periodic.

## Hypothesis

Per-period tool/intent histograms diverge non-trivially across months. Stable patterns (top-1 tool = Read, recovery reflex = `git status`) persist; transient patterns (package manager choice, test runner, scratch-dir name) flip. A KL-divergence trip-wire flags exactly the months that changed and surfaces the *direction* of change — exactly the signal CLAUDE.md authoring needs.

## Approach

1. **Bucket** trace records by ISO-week (or month) using session timestamp from `data/trace_records.parquet`.
2. **Histograms per period**: tool-name distribution, intent-tag distribution (from extractor), top-N file extensions, top-N command verbs. Smooth with Laplace (alpha=1).
3. **Drift score**: symmetric KL `D(P_t || P_{t-1}) + D(P_{t-1} || P_t)` per axis. Also Jensen-Shannon for boundedness.
4. **Flag** periods where any axis exceeds 95th percentile of historical drift.
5. **Mine the why**: for each flagged period, sample 5 sessions where the top-changed tool appears, LM-summarise ("user adopted X because Y; abandoned Z").
6. **Tag stability**: tools/patterns appearing in every period for 6+ months → "stable, commit to CLAUDE.md". Tools with drift events → "volatile, omit or version".

## Outputs

- `data/drift_report.json` — per-period histograms, KL/JS scores, top-changed entries.
- `data/drift_summary.md` — timeline ("2026-02-W3: bun → pnpm; 2026-03-W2: jest dropped").
- Hook: when any period's max-axis JS > threshold, emit `triggers/reextract_persona.flag` so `agent_opt/persona/` regenerates only on real change.

## Use Cases

- **CLAUDE.md freshness** — only stable items get committed; volatile items become a "current preferences" appendix with a date.
- **Bench revalidation** — flagged periods invalidate behavioural-cloning splits trained on pre-drift data.
- **Train/eval hygiene** — split by drift boundary, not random shuffle, to measure generalisation across regime changes.

## Effort + ROI

~1.5 day: histogram + KL is ~150 LOC, summariser is one Haiku prompt per flagged period. ROI: kills wasteful nightly re-extraction; gives downstream proposals (persona, CLAUDE.md, RAG indexing) a "this is current" guarantee.

## Self-critique

KL on small per-week samples is noisy — mitigate with month buckets + bootstrap CI, but power genuinely caps at ~9 monthly bins for Codex.

---

**TLDR**
- Snapshots go stale; periodic re-extraction is wasteful.
- Bucket traces per ISO-week, KL/JS-divergence consecutive histograms, flag the spikes.
- LM-summarise *why* the spike happened; tag patterns as stable vs volatile.
- Trigger persona re-extraction only on real drift; keeps CLAUDE.md honest.

**Path:** `trace-gepa/proposals/drift_detection.md`

**Self-critique:** 9 monthly bins is statistically thin — drift-vs-noise calls will be uncertain at the boundaries.
