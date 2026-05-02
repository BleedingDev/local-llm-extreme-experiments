from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt.seed import SEED_PROMPT  # noqa: E402


def _load_test_records(dataset_path: Path, splits_path: Path) -> list[dict]:
    test_ids = set(json.loads(splits_path.read_text())["ids"]["test"])
    seen: set[str] = set()
    out: list[dict] = []
    for line in dataset_path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = rec.get("id")
        if rid in test_ids and rid not in seen:
            seen.add(rid)
            out.append(rec)
    return out


def _stratified_sample(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n >= len(records):
        return list(records)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = str(r.get("failure_category"))
        buckets[key].append(r)
    total = len(records)
    quotas: dict[str, int] = {}
    for k, items in buckets.items():
        quotas[k] = max(1, round(n * len(items) / total))
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=lambda k: quotas[k])
        quotas[biggest] -= 1
    while sum(quotas.values()) < n:
        biggest = max(buckets, key=lambda k: len(buckets[k]) - quotas[k])
        quotas[biggest] += 1
    out: list[dict] = []
    for k, q in quotas.items():
        items = list(buckets[k])
        rng.shuffle(items)
        out.extend(items[:q])
    rng.shuffle(out)
    return out


def _run_eval(adapter, batch: list[dict], system_prompt: str, max_workers: int) -> tuple[list[float], list[dict | None]]:
    scores: list[float | None] = [None] * len(batch)
    outputs: list[dict | None] = [None] * len(batch)

    def _one(i: int):
        single = adapter.evaluate([batch[i]], candidate={"system": system_prompt}, capture_traces=False)
        scores[i] = float(single.scores[0]) if single.scores else 0.0
        outputs[i] = single.outputs[0] if single.outputs else None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_one, range(len(batch))))
    return [s or 0.0 for s in scores], outputs


def _breakdown(records: list[dict], scores: list[float], key: str) -> dict[str, dict]:
    by: dict[str, list[float]] = defaultdict(list)
    for r, s in zip(records, scores):
        by[str(r.get(key))].append(s)
    return {k: {"n": len(v), "pass_rate": (sum(v) / len(v)) if v else 0.0} for k, v in by.items()}


def _print_table(summary: dict) -> None:
    print(f"\n{'metric':<28}{'seed':>14}{'optim':>14}{'delta':>10}")
    print("-" * 66)
    pa, pb = summary["pass_rate_a"], summary["pass_rate_b"]
    print(f"{'overall':<28}{pa:>14.3f}{pb:>14.3f}{(pb - pa):>+10.3f}")
    for label, ka, kb in [("by failure_category", "by_category_a", "by_category_b"),
                          ("by label", "by_label_a", "by_label_b")]:
        print(f"\n{label}:")
        ma, mb = summary[ka], summary[kb]
        for k in sorted(set(ma) | set(mb)):
            a = ma.get(k, {"n": 0, "pass_rate": 0.0})
            b = mb.get(k, {"n": 0, "pass_rate": 0.0})
            print(f"  {k:<26}n={a['n']:<4}{a['pass_rate']:>10.3f}{b['pass_rate']:>14.3f}{(b['pass_rate']-a['pass_rate']):>+10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_ROOT / "data" / "dataset.jsonl"))
    ap.add_argument("--splits", default=str(_ROOT / "data" / "splits.json"))
    ap.add_argument("--prompt-a-text", default=None)
    ap.add_argument("--prompt-b-text", default=None)
    ap.add_argument("--prompt-b-file", default=str(_ROOT / "artifacts" / "optimized-prompts" / "latest" / "best_candidate.system.md"))
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--task-model", default="claude-opus-4-7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    prompt_a = args.prompt_a_text or SEED_PROMPT
    if args.prompt_b_text:
        prompt_b = args.prompt_b_text
        prompt_b_src = "inline"
    else:
        pb_path = Path(args.prompt_b_file)
        if not pb_path.exists():
            print(f"no optimised prompt yet — run optimize.py first (looked at {pb_path})", file=sys.stderr)
            return 1
        prompt_b = pb_path.read_text()
        prompt_b_src = str(pb_path)

    from agent_opt.adapter import TraceAdapter
    rng = random.Random(args.seed)
    all_records = _load_test_records(Path(args.dataset), Path(args.splits))
    batch = _stratified_sample(all_records, args.test_size, rng)
    print(f"loaded {len(all_records)} test records; sampling {len(batch)} (seed={args.seed})")
    print(f"prompt_a: SEED ({len(prompt_a)} chars)")
    print(f"prompt_b: {prompt_b_src} ({len(prompt_b)} chars)")
    print(f"task_model={args.task_model} max_workers={args.max_workers}")
    print(f"category mix: {dict(Counter(str(r.get('failure_category')) for r in batch))}")

    adapter = TraceAdapter(task_model=args.task_model)
    t0 = time.time()
    scores_a, out_a = _run_eval(adapter, batch, prompt_a, args.max_workers)
    t1 = time.time()
    print(f"prompt_a done in {t1 - t0:.1f}s — pass_rate={sum(scores_a)/len(scores_a):.3f}")
    scores_b, out_b = _run_eval(adapter, batch, prompt_b, args.max_workers)
    t2 = time.time()
    print(f"prompt_b done in {t2 - t1:.1f}s — pass_rate={sum(scores_b)/len(scores_b):.3f}")

    pa = sum(scores_a) / len(scores_a)
    pb = sum(scores_b) / len(scores_b)
    summary = {
        "pass_rate_a": pa,
        "pass_rate_b": pb,
        "delta": pb - pa,
        "n": len(batch),
        "task_model": args.task_model,
        "seed": args.seed,
        "prompt_a_chars": len(prompt_a),
        "prompt_b_chars": len(prompt_b),
        "prompt_b_source": prompt_b_src,
        "by_category_a": _breakdown(batch, scores_a, "failure_category"),
        "by_category_b": _breakdown(batch, scores_b, "failure_category"),
        "by_label_a": _breakdown(batch, scores_a, "label"),
        "by_label_b": _breakdown(batch, scores_b, "label"),
        "elapsed_a_s": t1 - t0,
        "elapsed_b_s": t2 - t1,
    }
    def _pred(o): return o.get("tool_name") if isinstance(o, dict) else None
    per_example = [
        {"id": r.get("id"), "label": r.get("label"), "category": r.get("failure_category"),
         "score_a": sa, "score_b": sb, "predicted_a": _pred(oa), "predicted_b": _pred(ob),
         "observed": (r.get("observed_action") or {}).get("name")}
        for r, sa, sb, oa, ob in zip(batch, scores_a, scores_b, out_a, out_b)
    ]
    out_path = Path(args.output) if args.output else (_HERE / f"results_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_example": per_example}, indent=2))
    print(f"wrote {out_path}")
    _print_table(summary)

    if pb < pa - 0.10:
        print("REGRESSION")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
