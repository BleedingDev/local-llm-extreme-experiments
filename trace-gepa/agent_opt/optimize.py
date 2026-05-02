from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure trace-gepa/ is on sys.path so `import agent_opt.*` resolves.
_HERE = Path(__file__).resolve().parent
_PKG_PARENT = _HERE.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

import gepa as gepa_pkg  # noqa: E402

from agent_opt import llm as _llm  # noqa: E402
from agent_opt.adapter import TraceAdapter  # noqa: E402
from agent_opt.reflection import REFLECTION_PROMPT_TEMPLATE  # noqa: E402
from agent_opt.seed import SEED_PROMPT as _DEFAULT_SEED_PROMPT  # noqa: E402

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
DATA = ROOT / "data" / "dataset.jsonl"
SPLITS = ROOT / "data" / "splits.json"
ART = ROOT / "artifacts" / "optimized-prompts"


def load_dataset_records(ids: list[str]) -> list[dict]:
    wanted = set(ids)
    by_id: dict[str, dict] = {}
    with DATA.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid in wanted:
                by_id[rid] = rec
                if len(by_id) == len(wanted):
                    break
    return [by_id[i] for i in ids if i in by_id]


def stratified_sample(records: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n >= len(records):
        return list(records)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cat = r.get("failure_category") or f"label:{r.get('label') or 'unknown'}"
        by_cat[cat].append(r)
    cats = sorted(by_cat.keys())
    for c in cats:
        rng.shuffle(by_cat[c])
    quotas: dict[str, int] = {c: 0 for c in cats}
    total = len(records)
    remaining = n
    for c in cats:
        share = round(n * len(by_cat[c]) / total)
        share = max(1, min(share, len(by_cat[c])))
        quotas[c] = share
    while sum(quotas.values()) > n:
        c = max(cats, key=lambda x: quotas[x])
        if quotas[c] <= 1:
            break
        quotas[c] -= 1
    while sum(quotas.values()) < n:
        c = min(cats, key=lambda x: quotas[x] / max(1, len(by_cat[x])))
        if quotas[c] >= len(by_cat[c]):
            cats_remaining = [x for x in cats if quotas[x] < len(by_cat[x])]
            if not cats_remaining:
                break
            c = cats_remaining[0]
        quotas[c] += 1
    out: list[dict] = []
    for c in cats:
        out.extend(by_cat[c][: quotas[c]])
    rng.shuffle(out)
    return out[:n]


def build_reflection_lm(model: str):
    def reflect(prompt: str) -> str:
        return _llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=4096,
            temperature=1.0,
        )
    return reflect


def evaluate_candidate(adapter: TraceAdapter, candidate: dict, valset: list[dict]) -> float:
    eb = adapter.evaluate(valset, candidate, capture_traces=False)
    if not eb.scores:
        return 0.0
    return sum(eb.scores) / len(eb.scores)


def replace_latest_symlink(run_dir: Path) -> None:
    latest = ART / "latest"
    with contextlib.suppress(FileNotFoundError):
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    latest.symlink_to(run_dir.name)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            with contextlib.suppress(Exception):
                st.flush()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--train-size", type=int, default=80)
    p.add_argument("--val-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-model", default="claude-opus-4-7")
    p.add_argument("--reflection-model", default="claude-opus-4-7")
    p.add_argument("--seed-module", choices=["default", "bag", "codex"], default="default",
                   help="Which seed prompt to optimise (default=canonical action-selector, bag=BAG planner, codex=Codex CLI BASE_INSTRUCTIONS — covers gpt-5.4/5.5/5.4-mini/5.3-codex-spark which all use prompt.md verbatim).")
    p.add_argument("--run-name", default=None,
                   help="Optional run label; produces artifacts/optimized-prompts/<NAME>_run_<TS>/.")
    args = p.parse_args()

    if args.seed_module == "bag":
        from agent_opt.seed_bag import SEED_PROMPT_BAG as SEED_PROMPT  # noqa: F811
    elif args.seed_module == "codex":
        from agent_opt.seed_codex import SEED_PROMPT_CODEX as SEED_PROMPT  # noqa: F811
    else:
        SEED_PROMPT = _DEFAULT_SEED_PROMPT

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_subdir = f"{args.run_name}_run_{ts}" if args.run_name else f"run_{ts}"
    run_dir = ART / run_subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    log_f = log_path.open("w")
    tee_out = _Tee(sys.__stdout__, log_f)
    tee_err = _Tee(sys.__stderr__, log_f)
    sys.stdout = tee_out
    sys.stderr = tee_err

    t0 = time.time()
    val_before: float | None = None
    val_after: float | None = None
    err: str | None = None

    try:
        print(f"[optimize] run_dir={run_dir}")
        print(f"[optimize] args={vars(args)}")

        with SPLITS.open() as f:
            splits = json.load(f)
        train_ids = splits["ids"]["train"]
        val_ids = splits["ids"]["val"]

        print(f"[optimize] loading {len(train_ids)} train + {len(val_ids)} val records")
        train_all = load_dataset_records(train_ids)
        val_all = load_dataset_records(val_ids)
        print(f"[optimize] loaded train={len(train_all)} val={len(val_all)}")

        rng = random.Random(args.seed)
        trainset = stratified_sample(train_all, args.train_size, rng)
        valset = stratified_sample(val_all, args.val_size, rng)

        def cat_dist(rs):
            d: dict[str, int] = defaultdict(int)
            for r in rs:
                d[str(r.get("failure_category"))] += 1
            return dict(d)

        print(f"[optimize] train sampled={len(trainset)} categories={cat_dist(trainset)}")
        print(f"[optimize] val sampled={len(valset)} categories={cat_dist(valset)}")

        adapter = TraceAdapter(task_model=args.task_model)
        seed_candidate = {"system": SEED_PROMPT}

        print("[optimize] computing baseline val score on seed prompt")
        val_before = evaluate_candidate(adapter, seed_candidate, valset)
        print(f"[optimize] val_score_before={val_before:.4f}")

        reflection_lm = build_reflection_lm(args.reflection_model)

        opt_kwargs = dict(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            skip_perfect_score=True,
            perfect_score=1.0,
            reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
            max_metric_calls=args.budget,
            seed=args.seed,
            raise_on_exception=True,
            run_dir=str(run_dir / "gepa_state"),
            display_progress_bar=False,
            cache_evaluation=True,
        )

        print(f"[optimize] calling gepa.optimize with budget={args.budget}")
        result = gepa_pkg.optimize(**opt_kwargs)
        elapsed_opt = time.time() - t0
        print(f"[optimize] gepa.optimize returned in {elapsed_opt:.1f}s")

        best = dict(result.best_candidate)
        try:
            val_after = float(result.val_aggregate_scores[result.best_idx])
        except Exception:
            val_after = evaluate_candidate(adapter, best, valset)
        print(f"[optimize] best_idx={result.best_idx} val_score_after={val_after:.4f}")
        print(f"[optimize] num_candidates={result.num_candidates} total_metric_calls={result.total_metric_calls}")

        (run_dir / "best_candidate.json").write_text(json.dumps(best, indent=2))
        (run_dir / "best_candidate.system.md").write_text(best.get("system", ""))

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("[optimize] EXCEPTION:\n" + err)
    finally:
        elapsed = time.time() - t0
        meta = {
            "timestamp": ts,
            "elapsed_seconds": round(elapsed, 2),
            "budget": args.budget,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "seed": args.seed,
            "task_model": args.task_model,
            "reflection_model": args.reflection_model,
            "val_score_before": val_before,
            "val_score_after": val_after,
            "delta": (None if (val_before is None or val_after is None) else round(val_after - val_before, 4)),
            "error": err,
            "run_dir": str(run_dir),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        with contextlib.suppress(Exception):
            replace_latest_symlink(run_dir)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_f.close()

    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
