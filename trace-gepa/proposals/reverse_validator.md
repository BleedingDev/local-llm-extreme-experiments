# Reverse-Direction Verifier Validator (Round-9, Member #JJ)

## TLDR
- Every `verifier_spec` implicitly defines an "ideal answer"; if a synthesized ideal cannot pass it, the verifier (not the model) is broken.
- Pre-flight pass synthesizes an ideal model output from `expected.primary_action.tool_name` + first `must_include_keywords_in_reason`, runs it through the live `bench.verifiers.verify`, and asserts `score > 0`.
- Wired into `validate_benchmark.py` as a CI gate: any `tasks.jsonl` change with reverse-failures blocks merge unless waived in `tasks/<id>/known_pathologies.yaml`.
- Would have caught all 16 command_synthesis bugs from today's audit (regex-anchored-on-wrong-haystack, `available_tools` missing the expected tool) at suite-build time, before any model ran.

## Hypothesis
A verifier_spec that rejects its own intended answer is a definitionally broken spec. The author's intent is encoded twice — once in the verifier, once in `expected.primary_action` — and a divergence means one is wrong. The cheapest place to detect this is offline, deterministically, at suite-build time.

## Design (~80 LoC)

```python
# bench/reverse_validator.py
def synthesize_ideal_answer(task: dict) -> dict:
    exp = task["expected"]["primary_action"]
    kw = (task["expected"].get("must_include_keywords_in_reason") or [""])[0]
    return {
        "tool_name": exp["tool_name"],
        "tool_args": exp.get("tool_args", {}),
        "reason": f"{kw}: invoking {exp['tool_name']} as required.",
    }

def reverse_validate(tasks: list[dict]) -> list[dict]:
    from bench.verifiers import verify
    fails = []
    for t in tasks:
        ideal = synthesize_ideal_answer(t)
        result = verify(t, ideal)
        if result.score <= 0:
            fails.append({"task_id": t["id"], "ideal": ideal, "verdict": result})
    return fails
```

Integration: `validate_benchmark.py` calls `reverse_validate(load_tasks())`; non-empty list → exit 1. Waivers live in `tasks/<id>/known_pathologies.yaml` with key `reverse_validator_waived: <reason>` and require human sign-off in PR.

## Use Cases
1. Catch today's bug class permanently (16 unreachable-spec tasks).
2. Regression-protect new task contributions.
3. Forces verifier authors to make the "happy path" explicit and reachable.

## Effort
~80 LoC, ~10 min. Reuses the existing verifier suite end-to-end — no new evaluation logic, no fixtures, no stubs.

## Self-Critique
Synthesis is heuristic: complex multi-step verifiers, fuzzy-match scorers, or verifiers reading external state (tool catalogs, file fixtures) may legitimately reject the synthesized stub even when the spec is correct — so we tag synth confidence (`high` for single-tool exact-match, `low` for regex/multi-keyword/external-state) and only hard-fail on `high`-confidence rejections, surfacing `low` ones as warnings.

Path: `trace-gepa/proposals/reverse_validator.md`

One-line self-critique: ideal-answer synthesis is heuristic and will produce false-positive verifier-bug reports for verifiers whose "correct" output depends on state outside `expected.primary_action`, so we gate hard-fails on synthesis confidence.
