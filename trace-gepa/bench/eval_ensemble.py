"""Evaluate a test-time ensemble of optimised prompts vs. each candidate alone.

For every test record we (a) score each candidate independently and (b) score
the ensemble's chosen output. Prints a single table comparing all rows.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.eval_baseline import _breakdown, _load_test_records, _stratified_sample  # noqa: E402
from bench.eval_multi import _parse_candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_ROOT / "data" / "dataset.jsonl"))
    ap.add_argument("--splits", default=str(_ROOT / "data" / "splits.json"))
    ap.add_argument("--test-size", type=int, default=60)
    ap.add_argument("--task-model", default="claude-opus-4-7")
    ap.add_argument("--judge-model", default="claude-opus-4-7")
    ap.add_argument("--aggregator", choices=("judge", "vote"), default="judge")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="space-separated name:source pairs (module:VAR or path/glob)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    candidates: list[tuple[str, str, str]] = []
    for raw in args.candidates:
        parsed = _parse_candidate(raw)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        print("no usable candidates — aborting", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    all_records = _load_test_records(Path(args.dataset), Path(args.splits))
    batch = _stratified_sample(all_records, args.test_size, rng)

    n_cand = len(candidates)
    n_rec = len(batch)
    judge_calls = n_rec if args.aggregator == "judge" else 0
    expected_calls = n_cand * n_rec + judge_calls
    print(f"loaded {len(all_records)} test records; sampling {n_rec} (seed={args.seed})")
    print(f"candidates ({n_cand}):")
    for name, src, text in candidates:
        print(f"  - {name:<10} chars={len(text):>6}  src={src}")
    print(f"category mix: {dict(Counter(str(r.get('failure_category')) for r in batch))}")
    print(f"task_model={args.task_model} judge_model={args.judge_model} aggregator={args.aggregator}")
    print(f"expected LM calls: {n_cand} cand x {n_rec} rec + {judge_calls} judge = {expected_calls}")

    if not args.yes and sys.stdin.isatty():
        try:
            ans = input("proceed? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("aborted")
            return 130

    from agent_opt.adapter import TraceAdapter
    from agent_opt.ensemble import EnsemblePredictor

    adapter = TraceAdapter(task_model=args.task_model)
    ensemble = EnsemblePredictor(
        prompts=[(n, t) for n, _s, t in candidates],
        task_model=args.task_model,
        judge_model=args.judge_model,
        aggregator=args.aggregator,
    )

    cand_scores: list[list[float]] = [[0.0] * n_rec for _ in range(n_cand)]
    cand_outs: list[list[dict | None]] = [[None] * n_rec for _ in range(n_cand)]
    ens_scores: list[float] = [0.0] * n_rec
    ens_chosen: list[str] = [""] * n_rec
    ens_outs: list[dict | None] = [None] * n_rec

    t0 = time.time()

    def _cand_task(args_: tuple[int, int]) -> None:
        ci, ri = args_
        eb = adapter.evaluate([batch[ri]], candidate={"system": candidates[ci][2]}, capture_traces=False)
        cand_scores[ci][ri] = float(eb.scores[0]) if eb.scores else 0.0
        cand_outs[ci][ri] = eb.outputs[0] if eb.outputs else None

    cand_jobs = [(ci, ri) for ci in range(n_cand) for ri in range(n_rec)]
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(_cand_task, cand_jobs))
    t1 = time.time()
    print(f"per-candidate scoring done in {t1-t0:.1f}s ({len(cand_jobs)} calls)")

    def _ens_task(ri: int) -> None:
        res = ensemble.predict(batch[ri])
        ens_chosen[ri] = res["chosen_name"]
        ens_outs[ri] = res["chosen_output"]
        ens_scores[ri] = adapter._score(batch[ri], res["chosen_output"], raw="")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        list(ex.map(_ens_task, range(n_rec)))
    t2 = time.time()
    print(f"ensemble scoring done in {t2-t1:.1f}s ({n_rec} records, ~{n_cand+judge_calls//max(1,n_rec)} calls each)")

    rows: list[dict] = []
    ens_by_lab = _breakdown(batch, ens_scores, "label")
    rows.append({
        "name": "ensemble",
        "overall": sum(ens_scores) / n_rec,
        "bad": ens_by_lab.get("bad", {}).get("pass_rate", 0.0),
        "good": ens_by_lab.get("good", {}).get("pass_rate", 0.0),
        "uconf": ens_by_lab.get("user_confirmed", {}).get("pass_rate", 0.0),
    })
    cand_rows: dict[str, dict] = {}
    for ci, (name, _src, _t) in enumerate(candidates):
        s = cand_scores[ci]
        by_lab = _breakdown(batch, s, "label")
        cand_rows[name] = {
            "name": name,
            "overall": sum(s) / n_rec,
            "bad": by_lab.get("bad", {}).get("pass_rate", 0.0),
            "good": by_lab.get("good", {}).get("pass_rate", 0.0),
            "uconf": by_lab.get("user_confirmed", {}).get("pass_rate", 0.0),
        }
        rows.append(cand_rows[name])

    print(f"\n{'candidate':<12}{'overall':>10}{'bad':>8}{'good':>8}{'uconf':>8}")
    print("-" * 46)
    for r in rows:
        print(f"{r['name']:<12}{r['overall']:>10.3f}{r['bad']:>8.3f}{r['good']:>8.3f}{r['uconf']:>8.3f}")

    best_alone = max((r["overall"] for r in rows[1:]), default=0.0)
    ens_overall = rows[0]["overall"]
    if ens_overall > best_alone:
        print(f"\nensemble dominates: +{ens_overall - best_alone:.3f} over best individual ({best_alone:.3f})")
    else:
        print(f"\nensemble lags best individual by {best_alone - ens_overall:.3f} ({ens_overall:.3f} vs {best_alone:.3f})")

    chosen_counts = Counter(ens_chosen)
    print(f"ensemble chose: {dict(chosen_counts)}")

    summary = {
        "task_model": args.task_model,
        "judge_model": args.judge_model,
        "aggregator": args.aggregator,
        "seed": args.seed,
        "n": n_rec,
        "expected_lm_calls": expected_calls,
        "ensemble": rows[0],
        "candidates": cand_rows,
        "chosen_counts": dict(chosen_counts),
        "category_mix": dict(Counter(str(r.get("failure_category")) for r in batch)),
    }

    def _pred(o):
        return o.get("tool_name") if isinstance(o, dict) else None

    per_example = []
    for ri, rec in enumerate(batch):
        row: dict = {
            "id": rec.get("id"),
            "label": rec.get("label"),
            "category": rec.get("failure_category"),
            "observed": (rec.get("observed_action") or {}).get("name"),
            "ensemble_score": ens_scores[ri],
            "ensemble_chosen": ens_chosen[ri],
            "ensemble_pred": _pred(ens_outs[ri]),
        }
        for ci, (name, _src, _t) in enumerate(candidates):
            row[f"score_{name}"] = cand_scores[ci][ri]
            row[f"pred_{name}"] = _pred(cand_outs[ci][ri])
        per_example.append(row)

    out_path = Path(args.output) if args.output else (_HERE / f"results_ensemble_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_example": per_example}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
