# Proposal: Output Normaliser Between LM and Verifier

**Round-8 Member #HH** — small infra fix, no retraining, no re-prompting.

## TLDR

- Models wrap JSON in ```json fences, prepend "Here is the JSON:", append commentary, and emit smart quotes/BOM despite explicit instructions; bench's `tier1_regex` only catches a fraction of this long tail.
- Insert `normalise(raw_lm_output: str) -> str` in `dispatch_call` BEFORE the verifier ever sees the string — strip fences (3/4/6 backtick variants), preamble phrases, trailing commentary after last balanced `}`, smart quotes → ASCII, BOM, leading/trailing whitespace.
- Gated by `--normalise [0|1]` (default `1`); every action logged to the record's `diagnostic.normalise_actions` field for audit and ablation. Verifier-port found 73% orchestration noise; expecting comparable pp lift here purely from format hygiene.
- ~50 LoC + golden-file tests; re-run baseline + per-cat v2 with normaliser on, measure delta on top of existing numbers — pure win unless self-critique below bites.

## Hypothesis

Verifier-port telemetry showed 73% of "wrong" outputs were format/orchestration noise, not reasoning failures. Output-side noise (fences, preamble, trailing prose) is structurally identical: the model knew the answer, the wrapper killed the parse. Strip the wrapper deterministically → recover pp without touching weights or prompts.

## Design

```
def normalise(raw: str) -> tuple[str, list[str]]:
    actions = []
    s = raw.lstrip("﻿").strip()                    # BOM + ws
    s = s.translate(SMART_QUOTE_MAP)                     # ""'' → "'
    s = FENCE_RE.sub(lambda m: m.group("body"), s)       # ```json...```, ```...```, ``````
    s = PREAMBLE_RE.sub("", s)                           # "Here is the JSON:" / "Here's my answer:" / "Sure, " / "Output:"
    s = strip_trailing_after_last_brace(s)               # only if {...} parses
    return s, actions
```

Called once in `dispatch_call` between raw LM return and verifier dispatch. Original raw kept on record for debugging. CLI: `--normalise 0` disables for raw inspection.

## Eval

Re-run `baseline` + `per-cat v2` suites with `--normalise 1`. Report Δpp per category; expect biggest lift on json/structured-output cats, near-zero on free-form. Cheap: no model calls change.

## Audit

Each normalisation action (`fence_stripped`, `preamble_stripped`, `smart_quotes`, `trailing_prose`, `bom`) appended to `record.diagnostic.normalise_actions`. Per-task visibility; aggregate counts in summary.

## Self-critique

Risk: normaliser "fixes" output that was substantively wrong (e.g. trims a malformed brace into a valid-but-incorrect answer), inflating scores while hiding real model failures — mitigate by logging every action and running a `--normalise 0` control on the same seeds to bound the masking effect.

**Path:** `trace-gepa/proposals/output_normaliser.md`
