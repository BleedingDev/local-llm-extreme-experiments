# Local Evidence Schema And Quality Audit
Generated for graph `local-evidence-flywheel-v1` on `2026-05-04` from `docs/local-evidence-inventory.md` plus direct local checks. Outputs are paired with machine-readable audit `.bag/evidence/schema-audit.json`.
## Scope
- JSONL files checked: `48` central evidence files.
- JSON files parsed: `1127` selected split, optimizer, replay, prompt, and job result files.
- Bench job family: `{'result_json_files': 541, 'bag_acp_summary_json_files': 415, 'audit_jsonl_files': 19, 'exception_txt_files': 32}`.
- Privacy scan reports pattern names and file/line references only; it intentionally does not copy matched sensitive text.

## Priority Findings
- `critical` `privacy`: Secret-like tokens detected by pattern scan Recommendation: Do not publish or train on affected files until manually reviewed and redacted. Rotate any confirmed live credentials.
- `high` `ids`: Duplicate IDs exist inside multiple datasets Recommendation: Regenerate duplicate-risk legacy datasets or exclude them from canonical training manifests; enforce unique id per file for canonical corpora.
- `high` `raw_vs_sanitised_parity`: Raw and sanitised mirrors are not all parity-clean Recommendation: Treat only parity-clean sanitised mirrors as canonical; regenerate or explicitly document any sanitised file with row/id deltas.
- `medium` `ids`: Some JSONL families intentionally or accidentally lack row IDs Recommendation: For trainable corpora, require stable ids. For metadata/audit/event JSONL, declare alternate primary keys or exempt them explicitly.
- `medium` `privacy`: Local identifiers and contact/network patterns appear in evidence Recommendation: Prefer sanitised mirrors for sharing and optimizer inputs; add allowlist/denylist review for usernames, emails, IPs, and absolute home paths.
- `medium` `schema`: Multiple schema families require explicit ingestion contracts Recommendation: Define family-specific readers for action/tool rows, recovery rows, benchmark tasks, counterfactuals, replay indexes, optimizer rows, and RAG metadata instead of one catch-all parser.
- `info` `row_integrity`: No JSONL parse errors in audited central JSONL files Recommendation: Keep jq/python parse checks in the evidence ingestion gate.
- `info` `split_leakage`: No split bucket overlaps detected in audited split files Recommendation: Keep pairwise split disjointness checks in CI or local release scripts.

## High-Value Corpus Checks
| path | rows | parse_errors | missing_id | duplicate_extra_ids | schema | labels |
| --- | --- | --- | --- | --- | --- | --- |
| trace-gepa/data/benchmark_tasks_full.jsonl | 175 | 0 | 0 | 0 | c7903f76c69a597d | {} |
| trace-gepa/data/cc_dataset_v2_new.jsonl | 22340 | 0 | 0 | 0 | 345ca80be5d3befe | {"bad": 2963, "good": 18934, "user_confirmed": 443} |
| trace-gepa/data/codex_gpt55_dataset.jsonl | 6820 | 0 | 0 | 0 | 345ca80be5d3befe | {"bad": 357, "good": 5389, "user_confirmed": 970, "user_corrected": 104} |
| trace-gepa/data/counterfactuals.jsonl | 431 | 0 | 431 | 0 | 3f734926507d0c1a | {} |
| trace-gepa/data/dataset_recovery.jsonl | 4055 | 0 | 0 | 0 | 3dd0843cec27a48a | {} |
| trace-gepa/data/dataset_v2.jsonl | 26384 | 0 | 0 | 0 | 345ca80be5d3befe | {"bad": 3421, "good": 22303, "user_confirmed": 593, "user_corrected": 67} |
| trace-gepa/data/sanitised/benchmark_tasks_full.jsonl | 175 | 0 | 0 | 0 | c7903f76c69a597d | {} |
| trace-gepa/data/sanitised/dataset_recovery.jsonl | 4055 | 0 | 0 | 0 | 3dd0843cec27a48a | {} |
| trace-gepa/data/sanitised/dataset_v2.jsonl | 26384 | 0 | 0 | 0 | 345ca80be5d3befe | {"bad": 3421, "good": 22303, "user_confirmed": 593, "user_corrected": 67} |
| .bag/replay-corpus/index.jsonl | 9 | 0 | 9 | 0 | f7eb1b5788a58f2d | {} |
| bench/.bag/optimizer/dataset.jsonl | 85 | 0 | 85 | 0 | 2763cc753655f2f1 | {} |

## Schema Fingerprints And Families
| fingerprint | files | rows | keys | examples |
| --- | --- | --- | --- | --- |
| c7903f76c69a597d | 15 | 874 | category, difficulty, expected, human_readable_summary, id, prompt, rubric_weight, source_record_ids, verifier_kind, verifier_spec | trace-gepa/data/benchmark_tasks.jsonl, trace-gepa/data/benchmark_tasks_full.jsonl, trace-gepa/data/benchmark_tasks_synthetic.jsonl, trace-gepa/data/sanitised/benchmark_tasks.jsonl |
| 345ca80be5d3befe | 14 | 127204 | context, failure_category, id, ideal_action_hint, label, next_user_message, observed_action, src, src_event_idx, src_path | trace-gepa/data/cc_dataset.jsonl, trace-gepa/data/cc_dataset_v2_new.jsonl, trace-gepa/data/codex_dataset.jsonl, trace-gepa/data/codex_dataset_v2_new.jsonl |
| f7eb1b5788a58f2d | 6 | 22 | identities, indexRecordId, labels, reproduction, runId, runResultId, safety, schemaVersion, scores, sourceKind, sourceRefs, split, status, taskId, taskPackId, title | .bag/replay-corpus/index.jsonl, .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504/index.jsonl, .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504b/index.jsonl, .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504c/index.jsonl |
| 3f734926507d0c1a | 3 | 912 | confidence, counterfactual_action, delta_kind, observed_action, rationale, record_id | trace-gepa/data/counterfactuals.jsonl, trace-gepa/data/counterfactuals_smoke.jsonl, trace-gepa/data/sanitised/counterfactuals.jsonl |
| 3dd0843cec27a48a | 2 | 8110 | distance_events, failed_record, id, lesson, pair_strength, recovery_record, session_id, src, transformation | trace-gepa/data/dataset_recovery.jsonl, trace-gepa/data/sanitised/dataset_recovery.jsonl |
| 5816c94ea708f6fa | 2 | 38577 | failure_category, id, label, next_user_message_excerpt, observed_tool, src, src_event_idx, src_path, user_request_excerpt | trace-gepa/artifacts/rag_index/metadata.jsonl, trace-gepa/artifacts/rag_index_v2/metadata.jsonl |
| 5c71d089d9e7ec0e | 2 | 52 | ground_truth_issues, id, src, src_path, user_request | trace-gepa/data/planner_dataset.jsonl, trace-gepa/data/sanitised/planner_dataset.jsonl |
| eee9a4d36b0d6e2a | 2 | 8090 | context, failure_category, id, ideal_action_hint, label, next_user_message, observed_action, quality_score, src, src_event_idx, src_path | trace-gepa/data/dataset_toolcalling.jsonl, trace-gepa/data/sanitised/dataset_toolcalling.jsonl |
| 2763cc753655f2f1 | 1 | 85 | agent_summary, bag_mode, exception_type, instruction_text, job_id, manifest, model, reward, routing, source_paths, task_name, trial_id, verifier, wall_seconds | bench/.bag/optimizer/dataset.jsonl |
| 4a540fbd9d8d9f0b | 1 | 8264 | failure_category, id, label, observed_tool, src, src_path, user_request_excerpt | trace-gepa/artifacts/rag_index_filtered/metadata.jsonl |

Main families observed: action/tool supervision rows, recovery transition rows, benchmark task rows, counterfactual annotation rows, replay/telemetry rows, optimizer rows, benchmark audit rows, and RAG metadata rows. These should have explicit readers rather than a single permissive loader.

## Row Integrity And IDs
No JSONL parse errors were found in audited central JSONL files.
| path | duplicate_extra_ids | duplicate_unique_ids |
| --- | --- | --- |
| trace-gepa/artifacts/rag_index/metadata.jsonl | 3929 | 3844 |
| trace-gepa/artifacts/rag_index_filtered/metadata.jsonl | 3582 | 3577 |
| trace-gepa/artifacts/rag_index_v2/metadata.jsonl | 3582 | 3577 |
| trace-gepa/data/cc_dataset.jsonl | 85 | 66 |
| trace-gepa/data/dataset.jsonl | 85 | 66 |
| trace-gepa/data/sanitised/cc_dataset.jsonl | 85 | 66 |
| trace-gepa/data/sanitised/dataset.jsonl | 85 | 66 |
| trace-gepa/data/planner_dataset.jsonl | 4 | 1 |
| trace-gepa/data/sanitised/planner_dataset.jsonl | 4 | 1 |
Some audited JSONL files do not use `id` as a row key. Treat these as non-training metadata/event families unless an alternate primary key is declared.
| path | rows | missing_id | invalid_id |
| --- | --- | --- | --- |
| trace-gepa/data/counterfactuals.jsonl | 431 | 431 | 0 |
| trace-gepa/data/counterfactuals_smoke.jsonl | 50 | 50 | 0 |
| trace-gepa/data/sanitised/counterfactuals.jsonl | 431 | 431 | 0 |
| .bag/replay-corpus/index.jsonl | 9 | 9 | 0 |
| .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504/index.jsonl | 1 | 1 | 0 |
| .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504b/index.jsonl | 1 | 1 | 0 |
| .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504c/index.jsonl | 1 | 1 | 0 |
| .bag/replay-corpus/real-acp-runs/real-acp-run.headless-smoke-20260504d/index.jsonl | 1 | 1 | 0 |
| .bag/replay-corpus/real-acp-runs/real-acp-run.headless-visible-20260504/index.jsonl | 9 | 9 | 0 |
| bench/.bag/optimizer/dataset.jsonl | 85 | 85 | 0 |

## Raw Vs Sanitised Parity
| pair | status | row_delta | missing_ids | extra_ids | content_equal |
| --- | --- | --- | --- | --- | --- |
| trace-gepa/data/benchmark_tasks.jsonl -> trace-gepa/data/sanitised/benchmark_tasks.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/benchmark_tasks_full.jsonl -> trace-gepa/data/sanitised/benchmark_tasks_full.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/benchmark_tasks_synthetic.jsonl -> trace-gepa/data/sanitised/benchmark_tasks_synthetic.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/cc_dataset.jsonl -> trace-gepa/data/sanitised/cc_dataset.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/cc_dataset_v2_new.jsonl -> trace-gepa/data/sanitised/cc_dataset_v2_new.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/codex_dataset.jsonl -> trace-gepa/data/sanitised/codex_dataset.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/codex_dataset_v2_new.jsonl -> trace-gepa/data/sanitised/codex_dataset_v2_new.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/codex_gpt55_dataset.jsonl -> trace-gepa/data/sanitised/codex_gpt55_dataset.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/counterfactuals.jsonl -> trace-gepa/data/sanitised/counterfactuals.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/dataset.jsonl -> trace-gepa/data/sanitised/dataset.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/dataset_recovery.jsonl -> trace-gepa/data/sanitised/dataset_recovery.jsonl | checked | 0 | 1 | 1 | False |
| trace-gepa/data/dataset_toolcalling.jsonl -> trace-gepa/data/sanitised/dataset_toolcalling.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/dataset_v2.jsonl -> trace-gepa/data/sanitised/dataset_v2.jsonl | checked | 0 | 0 | 0 | False |
| trace-gepa/data/planner_dataset.jsonl -> trace-gepa/data/sanitised/planner_dataset.jsonl | checked | 0 | 0 | 0 | False |
Content hashes are expected to differ when redaction changes payloads; parity success is row/id/schema preservation, not byte equality.

## Split Leakage
| path | parse_ok | bucket_counts | duplicate_entries_by_bucket | leakage_pairs |
| --- | --- | --- | --- | --- |
| trace-gepa/data/splits.json | True | {"ids": 3929} | {"ids": 85} | [] |
| trace-gepa/data/splits_recovery.json | True | {"test": 407, "train": 3241, "val": 407} | {"test": 0, "train": 0, "val": 0} | [] |
| trace-gepa/data/splits_toolcalling.json | True | {"splits": 4045} | {"splits": 0} | [] |
| trace-gepa/data/splits_v2.json | True | {"ids": 26384} | {"ids": 0} | [] |

## Privacy Red Flags
Pattern hit counts: `{"bearer_token": 36, "email": 40542, "home_path": 193102, "ipv4": 7918}`.
Secret-like token patterns were detected. The JSON audit lists file/line references only; inspect locally before any release.

## Inventory Claim Checks
| path | inventory_rows | observed_rows | matches |
| --- | --- | --- | --- |
| trace-gepa/data/dataset_v2.jsonl | 26384 | 26384 | True |
| trace-gepa/data/sanitised/dataset_v2.jsonl | 26384 | 26384 | True |
| trace-gepa/data/cc_dataset_v2_new.jsonl | 22340 | 22340 | True |
| trace-gepa/data/codex_gpt55_dataset.jsonl | 6820 | 6820 | True |
| trace-gepa/data/dataset_recovery.jsonl | 4055 | 4055 | True |
| trace-gepa/data/counterfactuals.jsonl | 431 | 431 | True |
| trace-gepa/data/benchmark_tasks_full.jsonl | 175 | 175 | True |

## Prioritized Remediation
1. `splits`: Keep pairwise split disjointness checks as a blocking gate; regenerate any split file with non-zero overlap before optimizer/eval use. Blocking condition: Any leakage_pairs entry has overlap_count > 0.
2. `canonical_ids`: Use `trace-gepa/data/sanitised/dataset_v2.jsonl`, `trace-gepa/data/sanitised/dataset_recovery.jsonl`, and sanitised benchmark files as canonical only when raw-vs-sanitised row/id parity remains clean. Blocking condition: Row delta, missing IDs, extra IDs, or parse errors in parity report.
3. `legacy_duplicates`: Exclude duplicate-risk legacy datasets from training manifests unless deduped with deterministic id remapping and lineage notes. Blocking condition: duplicate_id_extra_rows > 0 for a candidate input file.
4. `privacy`: Default optimizer/sharing to sanitised mirrors. Manually inspect files with secret-like or local-identity pattern hits before release. Blocking condition: Any secret-like privacy hit; soft identity hits require review for external sharing.
5. `schema_contracts`: Publish explicit ingestion contracts per schema family with required keys and allowed optional keys. Blocking condition: A new input file has an unknown schema_fingerprint or unhandled top-level shape.
6. `row_integrity`: Add JSONL parse, required-key, stable-id, duplicate-id, and label-domain assertions to local ingestion scripts. Blocking condition: Parse errors, missing required keys, empty IDs, duplicate IDs, or unexpected labels.

## Commands Run
- `pwd && rg --files docs .bag/evidence | sort`
- `git status --short`
- `sed -n '1,240p' docs/local-evidence-inventory.md`
- `find . -maxdepth 4 ... evidence file discovery`
- `python3 structured schema/quality audit generator`
- `jq . .bag/evidence/schema-audit.json >/dev/null`
- `test -s docs/local-evidence-quality-audit.md`
