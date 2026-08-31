"""Persona-prefix A/B test driver, v2 (60 tasks, seed=42, claude-opus-4-7).

Run: python -m agent_opt.persona._run_ab_v2
Stratifies 60 tasks from data/benchmark_tasks_full.jsonl, runs both arms
(seed-only, seed+persona-prefix), writes ab_test_results_v2.md and .raw.json.
LM budget: 120 calls (60 tasks * 2 arms). Threadpool 8 wide.
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
OUT_MD = ROOT / "agent_opt" / "persona" / "ab_test_results_v2.md"
OUT_RAW = ROOT / "agent_opt" / "persona" / "ab_test_results_v2.raw.json"
MODEL = "claude-opus-4-7"
N_TASKS = 60
SAMPLE_SEED = 42


def stratified_sample(tasks: list[dict], n: int, seed: int = SAMPLE_SEED) -> list[dict]:
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


def render_md(seed_agg: dict, persona_agg: dict, prefix_text: str,
              n_changed: int, examples: list[dict], verdict_label: str,
              rec_text: str) -> str:
    cats = sorted(set(seed_agg["per_category"]) | set(persona_agg["per_category"]))
    delta = persona_agg["overall"] - seed_agg["overall"]
    n = seed_agg["n"]
    div_rate = (n_changed / n) if n else 0.0
    lines = [
        "# Persona Prefix A/B Test (v2)", "",
        f"Model: `{MODEL}`. Tasks: {n} stratified (seed={SAMPLE_SEED}).",
        f"LM calls: {seed_agg['n'] + persona_agg['n']} (cap 120). "
        f"Errors: seed={seed_agg['n_errors']}, persona={persona_agg['n_errors']}.", "",
        "## Overall", "",
        "| Arm | Mean score | Divergence count |", "|---|---|---|",
        f"| seed-only | {seed_agg['overall']:.3f} | - |",
        f"| seed+persona | {persona_agg['overall']:.3f} | {n_changed}/{n} ({div_rate:.1%}) |",
        f"| delta | {delta:+.3f} | - |", "",
        "## Per-category", "",
        "| Category | n | seed | seed+persona | delta |", "|---|---|---|---|---|",
    ]
    for c in cats:
        s = seed_agg["per_category"].get(c, {"n": 0, "mean": 0.0})
        p = persona_agg["per_category"].get(c, {"n": 0, "mean": 0.0})
        lines.append(f"| {c} | {s['n']} | {s['mean']:.3f} | {p['mean']:.3f} | {p['mean']-s['mean']:+.3f} |")
    lines += ["", f"## Persona prefix ({len(prefix_text)} chars)", "", "```", prefix_text, "```", "",
              "## Sample diverging predictions (first 8)", ""]
    for ex in examples[:8]:
        lines.append(f"- **{ex['id']}** ({ex['category']}): seed=`{ex['seed_pred']}` vs "
                     f"persona=`{ex['persona_pred']}` (seed_score={ex['seed_score']:.1f}, "
                     f"persona_score={ex['persona_score']:.1f})")
    lines += ["", "## Verdict", "", f"**{verdict_label}**", "",
              f"Divergence rate: {div_rate:.1%} ({n_changed}/{n}). Mean delta: {delta:+.3f}.", "",
              rec_text]
    return "\n".join(lines) + "\n"


def decide_verdict(n_changed: int, n: int, delta: float) -> tuple[str, str]:
    """Apply v2 verdict logic from the persona iteration spec."""
    div_rate = (n_changed / n) if n else 0.0
    if div_rate > 0.25 and delta >= -0.02:
        return ("PROCEED",
                "Divergence > 25% AND persona arm score is within 0.02 of seed: "
                "the prefix has measurable purchase without collapsing accuracy. "
                "Recommend PROCEED to a Mac-day LoRA fine-tune to internalise "
                "the persona signal.")
    if div_rate > 0.15 and delta < 0:
        return ("TWEAK",
                "Divergence > 15% but persona arm scores worse than seed. "
                "The prefix is reaching the model but the wording is net-negative. "
                "Recommend continuing to TWEAK the prefix (de-emphasise distracting "
                "trivia, sharpen tool-priority + Czech-correction signals) before "
                "committing to LoRA.")
    if div_rate < 0.10:
        return ("ABANDON",
                "Divergence < 10%: the prefix has effectively no purchase on the "
                "model's outputs. A LoRA would amplify a signal that is barely "
                "moving the needle. ABANDON this iteration; reconsider only with "
                "a redesigned prefix or a different injection point.")
    # Middle band (10% <= div <= 25%, or div > 25% with too-large regression).
    if div_rate > 0.25 and delta < -0.02:
        return ("TWEAK",
                f"Divergence {div_rate:.1%} with mean regression of {delta:+.3f} "
                "below the -0.02 floor. Behavior is shifting but accuracy is "
                "regressing - TWEAK prefix wording before LoRA.")
    return ("TWEAK",
            f"Divergence {div_rate:.1%} sits in the inconclusive 10-25% band. "
            "Keep iterating prefix; a LoRA is premature.")


def main() -> int:
    if not PROFILE.exists():
        print("run agent_opt.persona.fingerprint first", file=sys.stderr)
        return 2
    profile = json.loads(PROFILE.read_text())
    prefix_text = build_persona_prefix(profile)
    seed_system = SEED_PROMPT
    persona_system = inject_persona(SEED_PROMPT, profile)
    print(f"prefix: {len(prefix_text)} chars")

    tasks = [json.loads(l) for l in TASKS.open() if l.strip()]
    sample = stratified_sample(tasks, N_TASKS, seed=SAMPLE_SEED)
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
    delta = persona_agg["overall"] - seed_agg["overall"]
    verdict_label, rec_text = decide_verdict(n_changed, seed_agg["n"], delta)

    OUT_MD.write_text(render_md(seed_agg, persona_agg, prefix_text,
                                n_changed, examples, verdict_label, rec_text))
    OUT_RAW.write_text(json.dumps(
        {"seed": seed_rows, "persona": persona_rows, "seed_agg": seed_agg,
         "persona_agg": persona_agg, "diverging_examples": examples,
         "prefix_chars": len(prefix_text), "verdict": verdict_label,
         "sample_seed": SAMPLE_SEED, "n_tasks": N_TASKS, "model": MODEL}, indent=2))
    print(f"wrote {OUT_MD}")
    print(f"seed mean: {seed_agg['overall']:.3f}  persona mean: {persona_agg['overall']:.3f}  "
          f"diverged: {n_changed}/{seed_agg['n']}  verdict: {verdict_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
