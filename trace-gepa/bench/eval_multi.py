"""Multi-target benchmark: compare several candidate prompts on one shared test sample.

Reuses sampling + scoring helpers from `bench.eval_baseline` so the test split
matches what eval_baseline produces for `--seed 42`.
"""
from __future__ import annotations

import argparse
import glob
import importlib
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

from bench.eval_baseline import _load_test_records, _stratified_sample, _breakdown  # noqa: E402


def _resolve_module_var(spec: str) -> str | None:
    """spec is 'module.path:VAR'."""
    mod_name, _, var = spec.partition(":")
    if not mod_name or not var:
        return None
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"  module import failed for {mod_name}: {e}", file=sys.stderr)
        return None
    val = getattr(mod, var, None)
    if not isinstance(val, str):
        print(f"  {mod_name}.{var} is not a string", file=sys.stderr)
        return None
    return val


def _resolve_path(spec: str) -> Path | None:
    """Resolve a path or glob; if glob, pick most recent match."""
    p = Path(spec)
    if not p.is_absolute():
        cand_root = _ROOT.parent / spec
        if cand_root.exists() or any(c in spec for c in "*?["):
            p = cand_root
        else:
            p = (_ROOT / spec) if (_ROOT / spec).exists() else cand_root
    if any(c in str(p) for c in "*?["):
        matches = [Path(m) for m in glob.glob(str(p))]
        matches = [m for m in matches if m.exists()]
        if not matches:
            return None
        matches.sort(key=lambda m: m.stat().st_mtime, reverse=True)
        return matches[0]
    return p if p.exists() else None


def _parse_candidate(arg: str) -> tuple[str, str, str] | None:
    """Parse `name:source` -> (name, source_label, prompt_text). Source can be
    `module:VAR` (two colons total) or a path/glob.
    """
    name, _, src = arg.partition(":")
    if not name or not src:
        print(f"skipping malformed candidate '{arg}'", file=sys.stderr)
        return None
    # Heuristic: module form has another colon AND no path separator before the second colon.
    if ":" in src and "/" not in src.split(":", 1)[0] and not src.endswith(".md"):
        text = _resolve_module_var(src)
        if text is None:
            print(f"WARN: skipping '{name}' — could not resolve module spec '{src}'", file=sys.stderr)
            return None
        return name, f"module:{src}", text
    # path / glob form
    resolved = _resolve_path(src)
    if resolved is None:
        print(f"WARN: skipping '{name}' — no file matches '{src}'", file=sys.stderr)
        return None
    try:
        text = resolved.read_text()
    except Exception as e:
        print(f"WARN: skipping '{name}' — read failed for {resolved}: {e}", file=sys.stderr)
        return None
    return name, str(resolved), text


def _score_one(adapter, record: dict, system_prompt: str) -> tuple[float, dict | None]:
    eb = adapter.evaluate([record], candidate={"system": system_prompt}, capture_traces=False)
    score = float(eb.scores[0]) if eb.scores else 0.0
    out = eb.outputs[0] if eb.outputs else None
    return score, out


def _run_all(adapter, batch: list[dict], candidates: list[tuple[str, str, str]],
             max_workers: int) -> dict[str, dict]:
    """One global ThreadPoolExecutor; submit (cand_idx, rec_idx) tasks."""
    n_cands = len(candidates)
    n_recs = len(batch)
    scores = [[0.0] * n_recs for _ in range(n_cands)]
    outputs: list[list[dict | None]] = [[None] * n_recs for _ in range(n_cands)]

    def _task(args: tuple[int, int]):
        ci, ri = args
        s, o = _score_one(adapter, batch[ri], candidates[ci][2])
        scores[ci][ri] = s
        outputs[ci][ri] = o

    jobs = [(ci, ri) for ci in range(n_cands) for ri in range(n_recs)]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(_task, jobs):
            done += 1
            if done % max(1, len(jobs) // 10) == 0:
                pct = 100.0 * done / len(jobs)
                print(f"  progress: {done}/{len(jobs)} ({pct:.0f}%) elapsed={time.time()-t0:.1f}s")
    elapsed = time.time() - t0
    result: dict[str, dict] = {}
    for ci, (name, src, text) in enumerate(candidates):
        s = scores[ci]
        result[name] = {
            "source": src,
            "chars": len(text),
            "scores": s,
            "outputs": outputs[ci],
            "pass_rate": (sum(s) / len(s)) if s else 0.0,
        }
    print(f"all candidates done in {elapsed:.1f}s ({len(jobs)} LM calls total)")
    return result


def _print_ranked_table(per_cand: dict[str, dict], by_label: dict[str, dict[str, dict]]) -> None:
    rows = []
    for name, d in per_cand.items():
        labels = by_label.get(name, {})
        row = {
            "name": name,
            "chars": d["chars"],
            "overall": d["pass_rate"],
            "bad_pass": labels.get("bad", {}).get("pass_rate", 0.0),
            "good_pass": labels.get("good", {}).get("pass_rate", 0.0),
            "user_conf_pass": labels.get("user_confirmed", {}).get("pass_rate", 0.0),
            "notes": "(long prompt, high latency)" if d["chars"] > 8000 else "",
        }
        rows.append(row)
    rows.sort(key=lambda r: r["overall"], reverse=True)
    print(f"\n{'rank':<5}{'name':<10}{'chars':>7}  {'overall':>8}  {'bad':>6}  {'good':>6}  {'uconf':>6}  notes")
    print("-" * 70)
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['name']:<10}{r['chars']:>7}  {r['overall']:>8.3f}  "
              f"{r['bad_pass']:>6.3f}  {r['good_pass']:>6.3f}  {r['user_conf_pass']:>6.3f}  {r['notes']}")


def _print_category_breakdown(per_cand: dict[str, dict],
                              by_cat: dict[str, dict[str, dict]],
                              top_k: int = 5) -> None:
    # find top categories by count using any candidate's view (all share batch)
    any_name = next(iter(by_cat))
    cats_sorted = sorted(by_cat[any_name].items(), key=lambda kv: kv[1]["n"], reverse=True)
    top_cats = [k for k, _ in cats_sorted[:top_k]]
    print(f"\nper-category pass_rate (top {len(top_cats)} of {len(cats_sorted)}):")
    header = f"  {'category':<26}{'n':>4}  " + "  ".join(f"{n:>10}" for n in per_cand)
    print(header)
    for cat in top_cats:
        n = by_cat[any_name][cat]["n"]
        cells = "  ".join(f"{by_cat[name].get(cat, {'pass_rate': 0.0})['pass_rate']:>10.3f}"
                          for name in per_cand)
        print(f"  {cat:<26}{n:>4}  {cells}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(_ROOT / "data" / "dataset.jsonl"))
    ap.add_argument("--splits", default=str(_ROOT / "data" / "splits.json"))
    ap.add_argument("--test-size", type=int, default=50)
    ap.add_argument("--task-model", default="claude-opus-4-7")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="space-separated name:source pairs")
    ap.add_argument("--yes", action="store_true",
                    help="skip cost-confirmation prompt (non-interactive)")
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
    n_calls = len(candidates) * len(batch)
    print(f"loaded {len(all_records)} test records; sampling {len(batch)} (seed={args.seed})")
    print(f"candidates: {len(candidates)}")
    for name, src, text in candidates:
        print(f"  - {name:<10} chars={len(text):>6}  src={src}")
    print(f"category mix: {dict(Counter(str(r.get('failure_category')) for r in batch))}")
    print(f"task_model={args.task_model} max_workers={args.max_workers}")
    print(f"expected LM calls: {len(candidates)} candidates x {len(batch)} records = {n_calls}")

    if not args.yes and sys.stdin.isatty():
        try:
            ans = input("proceed? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("aborted")
            return 130

    from agent_opt.adapter import TraceAdapter
    adapter = TraceAdapter(task_model=args.task_model)

    per_cand = _run_all(adapter, batch, candidates, args.max_workers)

    by_cat = {n: _breakdown(batch, per_cand[n]["scores"], "failure_category") for n in per_cand}
    by_lab = {n: _breakdown(batch, per_cand[n]["scores"], "label") for n in per_cand}

    requested = {raw.split(":", 1)[0] for raw in args.candidates}
    loaded = set(per_cand.keys())
    skipped = sorted(requested - loaded)

    summary = {
        "task_model": args.task_model,
        "seed": args.seed,
        "n": len(batch),
        "skipped_candidates": skipped,
        "category_mix": dict(Counter(str(r.get("failure_category")) for r in batch)),
        "candidates": {
            name: {
                "source": d["source"],
                "chars": d["chars"],
                "pass_rate": d["pass_rate"],
                "by_category": by_cat[name],
                "by_label": by_lab[name],
            } for name, d in per_cand.items()
        },
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
        }
        for name, d in per_cand.items():
            row[f"score_{name}"] = d["scores"][ri]
            row[f"pred_{name}"] = _pred(d["outputs"][ri])
        per_example.append(row)

    out_path = Path(args.output) if args.output else (_HERE / f"results_multi_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_example": per_example}, indent=2))
    print(f"\nwrote {out_path}")
    if skipped:
        print(f"skipped (source missing): {', '.join(skipped)}")

    _print_ranked_table(per_cand, by_lab)
    _print_category_breakdown(per_cand, by_cat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
