from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_PARENT = _HERE.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

import gepa as gepa_pkg  # noqa: E402

from agent_opt import llm as _llm  # noqa: E402
from agent_opt.adapter import TraceAdapter  # noqa: E402
from agent_opt.codex_dataset import codex_only  # noqa: E402
from agent_opt.optimize import (  # noqa: E402
    _Tee,
    build_reflection_lm,
    evaluate_candidate,
    load_dataset_records,
    stratified_sample,
)
from agent_opt.reflection import REFLECTION_PROMPT_TEMPLATE  # noqa: E402

try:
    from agent_opt.seed_codex import SEED_PROMPT_CODEX as SEED_PROMPT  # type: ignore
    _SEED_SOURCE = "agent_opt.seed_codex.SEED_PROMPT_CODEX"
except Exception:
    from agent_opt.seed import SEED_PROMPT  # noqa: F401
    _SEED_SOURCE = "agent_opt.seed.SEED_PROMPT (fallback)"

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
SPLITS = ROOT / "data" / "splits.json"
ART = ROOT / "artifacts" / "optimized-prompts"


def replace_latest_codex_symlink(run_dir: Path) -> None:
    latest = ART / "latest_codex"
    with contextlib.suppress(FileNotFoundError):
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    latest.symlink_to(run_dir.name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--train-size", type=int, default=80)
    p.add_argument("--val-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-model", default="claude-opus-4-7")
    p.add_argument("--reflection-model", default="claude-opus-4-7")
    args = p.parse_args()

    print(f"[optimize_codex] seed prompt source: {_SEED_SOURCE}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ART / f"codex_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    log_f = log_path.open("w")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    t0 = time.time()
    val_before: float | None = None
    val_after: float | None = None
    err: str | None = None

    try:
        print(f"[optimize_codex] run_dir={run_dir}")
        print(f"[optimize_codex] args={vars(args)}")
        print(f"[optimize_codex] seed prompt source: {_SEED_SOURCE}")

        with SPLITS.open() as f:
            splits = json.load(f)
        train_ids = splits["ids"]["train"]
        val_ids = splits["ids"]["val"]

        train_all = codex_only(load_dataset_records(train_ids))
        val_all = codex_only(load_dataset_records(val_ids))
        print(f"[optimize_codex] codex-only train={len(train_all)} val={len(val_all)}")

        rng = random.Random(args.seed)
        trainset = stratified_sample(train_all, args.train_size, rng)
        valset = stratified_sample(val_all, args.val_size, rng)

        def cat_dist(rs):
            d: dict[str, int] = defaultdict(int)
            for r in rs:
                d[str(r.get("failure_category"))] += 1
            return dict(d)

        print(f"[optimize_codex] train sampled={len(trainset)} categories={cat_dist(trainset)}")
        print(f"[optimize_codex] val sampled={len(valset)} categories={cat_dist(valset)}")

        adapter = TraceAdapter(task_model=args.task_model)
        seed_candidate = {"system": SEED_PROMPT}

        print("[optimize_codex] computing baseline val score on seed prompt")
        val_before = evaluate_candidate(adapter, seed_candidate, valset)
        print(f"[optimize_codex] val_score_before={val_before:.4f}")

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

        print(f"[optimize_codex] calling gepa.optimize with budget={args.budget}")
        result = gepa_pkg.optimize(**opt_kwargs)
        print(f"[optimize_codex] gepa.optimize returned in {time.time() - t0:.1f}s")

        best = dict(result.best_candidate)
        try:
            val_after = float(result.val_aggregate_scores[result.best_idx])
        except Exception:
            val_after = evaluate_candidate(adapter, best, valset)
        print(f"[optimize_codex] best_idx={result.best_idx} val_score_after={val_after:.4f}")

        (run_dir / "best_candidate.json").write_text(json.dumps(best, indent=2))
        (run_dir / "best_candidate.system.md").write_text(best.get("system", ""))

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("[optimize_codex] EXCEPTION:\n" + err)
    finally:
        meta = {
            "track": "codex",
            "timestamp": ts,
            "elapsed_seconds": round(time.time() - t0, 2),
            "budget": args.budget,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "seed": args.seed,
            "task_model": args.task_model,
            "reflection_model": args.reflection_model,
            "seed_source": _SEED_SOURCE,
            "val_score_before": val_before,
            "val_score_after": val_after,
            "delta": (None if (val_before is None or val_after is None) else round(val_after - val_before, 4)),
            "error": err,
            "run_dir": str(run_dir),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        with contextlib.suppress(Exception):
            replace_latest_codex_symlink(run_dir)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_f.close()

    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
