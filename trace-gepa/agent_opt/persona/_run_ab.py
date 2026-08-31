"""One-shot persona-prefix A/B test driver.

Run: python -m agent_opt.persona._run_ab
Stratifies 30 tasks from data/benchmark_tasks_full.jsonl, runs both arms
(seed-only, seed+persona-prefix), writes ab_test_results.md.
LM budget: 60 calls (30 tasks * 2 arms). Threadpool 8 wide.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_opt import llm as _llm  # noqa: E402
from agent_opt.persona.prefix import build_persona_prefix, inject_persona  # noqa: E402
from agent_opt.seed import SEED_PROMPT  # noqa: E402
from bench.run_anthropic import _build_user_prompt, _parse_json  # noqa: E402

try:
    from bench.verifiers import verify as verify_fn  # type: ignore
except Exception:
    from bench.run_anthropic import _fallback_verify as verify_fn  # type: ignore

TASKS = ROOT / "data" / "benchmark_tasks_full.jsonl"
PROFILE = ROOT / "agent_opt" / "persona" / "profile.json"
OUT_MD = ROOT / "agent_opt" / "persona" / "ab_test_results.md"
MODEL = "claude-opus-4-7"
N_TASKS = 30


def stratified_sample(tasks: list[dict], n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_cat[t.get("category") or "?"].append(t)
    total = sum(len(v) for v in by_cat.values())
    picks: list[dict] = []
    for c in sorted(by_cat):
        rng.shuffle(by_cat[c])
        share = max(1, round(n * len(by_cat[c]) / total))
        picks.extend(by_cat[c][:share])
    rng.shuffle(picks)
    if len(picks) > n:
        picks = picks[:n]
    while len(picks) < n:
        pool = [t for t in tasks if t not in picks]
        if not pool:
            break
        picks.append(rng.choice(pool))
    return picks


def run_one(task: dict, system_prompt: str) -> dict:
    user_prompt = _build_user_prompt(task)
    raw, parsed, err = "", None, None
    t0 = time.time()
    try:
        raw = _llm.chat(messages=[{"role": "user", "content": user_prompt}], model=MODEL,
                        max_tokens=512, temperature=0.0, system=system_prompt)
    except Exception as e:
        err = f"llm_error: {e}"
    if raw and err is None:
        parsed = _parse_json(raw)
    verdict = {"score": 0.0, "signal": "no_verdict"}
    if err is None:
        try:
            verdict = verify_fn(task, parsed if parsed is not None else raw)
        except Exception as e:
            err = f"verify_error: {e}"
    return {"id": task.get("id"), "category": task.get("category"),
            "score": 0.0 if err else float(verdict.get("score") or 0.0),
            "signal": verdict.get("signal"), "predicted": parsed,
            "latency_ms": int((time.time() - t0) * 1000), "error": err}


def aggregate(rows: list[dict]) -> dict:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["score"])
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0  # noqa: E731
    return {"overall": mean([r["score"] for r in rows]), "n": len(rows),
            "n_errors": sum(1 for r in rows if r.get("error")),
            "per_category": {k: {"n": len(v), "mean": mean(v)} for k, v in sorted(by_cat.items())}}


def render_md(seed_agg: dict, persona_agg: dict, prefix_text: str, n_changed: int, examples: list[dict]) -> str:
    cats = sorted(set(seed_agg["per_category"]) | set(persona_agg["per_category"]))
    delta = persona_agg["overall"] - seed_agg["overall"]
    lines = [
        "# Persona Prefix A/B Test", "",
        f"Model: `{MODEL}`. Tasks: {seed_agg['n']} stratified across 7 categories.",
        f"LM calls: {seed_agg['n'] + persona_agg['n']} (cap 60). "
        f"Errors: seed={seed_agg['n_errors']}, persona={persona_agg['n_errors']}.", "",
        "## Overall", "",
        "| Arm | Mean score | Changed-vs-seed |", "|---|---|---|",
        f"| seed-only | {seed_agg['overall']:.3f} | - |",
        f"| seed+persona | {persona_agg['overall']:.3f} | {n_changed}/{seed_agg['n']} tasks |", "",
        "## Per-category", "",
        "| Category | n | seed | seed+persona | delta |", "|---|---|---|---|---|",
    ]
    for c in cats:
        s = seed_agg["per_category"].get(c, {"n": 0, "mean": 0.0})
        p = persona_agg["per_category"].get(c, {"n": 0, "mean": 0.0})
        lines.append(f"| {c} | {s['n']} | {s['mean']:.3f} | {p['mean']:.3f} | {p['mean']-s['mean']:+.3f} |")
    lines += ["", f"## Persona prefix ({len(prefix_text)} chars)", "", "```", prefix_text, "```", "",
              "## Sample diverging predictions (first 5)", ""]
    for ex in examples[:5]:
        lines.append(f"- **{ex['id']}** ({ex['category']}): seed=`{ex['seed_pred']}` vs "
                     f"persona=`{ex['persona_pred']}` (seed_score={ex['seed_score']:.1f}, "
                     f"persona_score={ex['persona_score']:.1f})")
    if n_changed >= 5:
        verdict = (f"Persona prefix MEASURABLY shifts behavior: {n_changed}/{seed_agg['n']} tasks "
                   f"produce a different prediction than seed-only. Overall mean delta = {delta:+.3f}.")
    else:
        verdict = (f"Persona prefix barely shifts behavior ({n_changed}/{seed_agg['n']} divergent). "
                   "At a 30-task sample this is consistent with the prefix being too similar to the seed.")
    if n_changed >= 5 and delta >= -0.05:
        rec = ("PROCEED-WITH-CAUTION. Prefix demonstrably moves outputs without collapsing accuracy. "
               "A LoRA fine-tune is worth a Mac-day to internalise the persona signal that survives "
               "prompt clipping.")
    elif n_changed >= 5 and delta < -0.05:
        rec = ("TWEAK. Prefix shifts behavior but mean score regressed. Iterate on prefix wording "
               "(de-emphasise distracting trivia, keep tool-priority + Czech-correction signals) "
               "before committing to a LoRA.")
    else:
        rec = ("ABANDON-FOR-NOW. Prefix barely changes outputs at 30 tasks; LoRA would amplify a "
               "signal that already isn't moving the needle. Reconsider only if a larger A/B "
               "(>=100 tasks) shows divergence.")
    lines += ["", "## Verdict", "", verdict, "", "## Step 3 (LoRA) recommendation", "", rec]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not PROFILE.exists():
        print("run agent_opt.persona.fingerprint first", file=sys.stderr)
        return 2
    profile = json.loads(PROFILE.read_text())
    prefix_text = build_persona_prefix(profile)
    seed_system = SEED_PROMPT
    persona_system = inject_persona(SEED_PROMPT, profile)

    tasks = [json.loads(l) for l in TASKS.open() if l.strip()]
    sample = stratified_sample(tasks, N_TASKS)
    cat_dist = {c: sum(1 for t in sample if t["category"] == c)
                for c in sorted({t["category"] for t in sample})}
    print(f"sampled {len(sample)} tasks; categories: {cat_dist}")

    def run_arm(label: str, system: str) -> list[dict]:
        rows: list[dict] = [None] * len(sample)  # type: ignore
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(run_one, t, system): i for i, t in enumerate(sample)}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    rows[i] = f.result()
                except Exception as e:
                    rows[i] = {"id": sample[i].get("id"), "category": sample[i].get("category"),
                               "score": 0.0, "error": f"future: {e}", "predicted": None,
                               "signal": "future_error", "latency_ms": 0}
        print(f"  {label}: mean={sum(r['score'] for r in rows)/len(rows):.3f} "
              f"errors={sum(1 for r in rows if r.get('error'))}")
        return rows

    print("running seed arm...")
    seed_rows = run_arm("seed", seed_system)
    print("running persona arm...")
    persona_rows = run_arm("persona", persona_system)

    examples = []
    for s, p in zip(seed_rows, persona_rows):
        sp = (s.get("predicted") or {}).get("tool_name", "") if isinstance(s.get("predicted"), dict) else ""
        pp = (p.get("predicted") or {}).get("tool_name", "") if isinstance(p.get("predicted"), dict) else ""
        if sp != pp:
            examples.append({"id": s["id"], "category": s["category"],
                             "seed_pred": sp, "persona_pred": pp,
                             "seed_score": s["score"], "persona_score": p["score"]})
    n_changed = len(examples)
    seed_agg, persona_agg = aggregate(seed_rows), aggregate(persona_rows)
    OUT_MD.write_text(render_md(seed_agg, persona_agg, prefix_text, n_changed, examples))
    OUT_MD.with_suffix(".raw.json").write_text(json.dumps(
        {"seed": seed_rows, "persona": persona_rows, "seed_agg": seed_agg,
         "persona_agg": persona_agg, "diverging_examples": examples,
         "prefix_chars": len(prefix_text)}, indent=2))
    print(f"wrote {OUT_MD}")
    print(f"seed mean: {seed_agg['overall']:.3f}  persona mean: {persona_agg['overall']:.3f}  "
          f"diverged: {n_changed}/{seed_agg['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
