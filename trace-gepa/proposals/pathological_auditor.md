# Pathological Task Auditor

## TLDR

- **Persistent 0% categories are a diagnostic signal, not a capability claim**: command_synthesis scoring 0/16 across Opus, GPT-5.5, and per-cat v2 means at least one of {verifier broken, prompt under-specified, task genuinely impossible} is true — and the bench cannot distinguish which without an audit.
- **Floor-based flagging is cheap**: compute `model_floor(t) = max(pass_rate(t, run_id) for run_id in reports/)`; any task with `model_floor == 0` across `>=3` independent run conditions is flagged. Empirically ~5–10% of any bench is pathological for boring reasons; surfacing them denoises every downstream comparison.
- **Hand-crafted oracle answers separate verifier-broken from genuinely-hard**: for each flagged task, write an "ideal" answer by hand, run it through the verifier, and branch on the result. Verifier rejects ideal -> `FIX_VERIFIER`. Verifier accepts ideal but no model produces it -> either `FIX_PROMPT` (if prompt is missing critical constraint) or `KEEP_AS_HARD` (if prompt is fair).
- **Output is a per-task dossier plus a roll-up**: `tasks/audit/pathological_<date>/<task_id>.md` per task (definition + 3 sample model outputs + verifier_spec + ideal answer + verifier trace), and `audit_report.md` recommending `FIX_VERIFIER | FIX_PROMPT | KEEP_AS_HARD | REMOVE` per task. Budget ~3h per 16 tasks.

## Pipeline

1. Scan `reports/*/per_task.json` -> aggregate `model_floor` per task_id.
2. Filter to `model_floor == 0` with `>=3` distinct run_ids covering `>=2` model families.
3. For each flagged task `t`, emit `tasks/audit/pathological_<date>/<t>.md` containing: task definition, verifier_spec, 3 representative failed model outputs (one per run_id), and a blank `ideal_answer:` field.
4. Auditor (human or second agent) fills `ideal_answer`; harness re-runs verifier against it and records pass/fail in `verifier_trace:`.
5. Decision rule: ideal fails -> `FIX_VERIFIER`; ideal passes and a fix to the prompt makes >=1 model pass on a re-run -> `FIX_PROMPT`; ideal passes but prompt is fair and unchanged -> `KEEP_AS_HARD`; task is malformed at the data layer -> `REMOVE`.
6. Roll up into `audit_report.md` with counts per recommendation and links to dossiers.

## Path

`trace-gepa/proposals/pathological_auditor.md`

## Self-critique

The "ideal answer" is a single auditor's subjective interpretation of an under-specified spec, so it must be blind-cross-checked by a second agent before any `KEEP_AS_HARD` verdict is trusted — otherwise we just relabel verifier bugs as task difficulty.
