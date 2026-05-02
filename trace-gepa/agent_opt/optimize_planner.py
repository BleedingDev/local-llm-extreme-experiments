"""GEPA optimisation entry point for BAG's planner step.

Mirrors `optimize.py` but uses `PlannerAdapter` and `planner_dataset.jsonl`.
Maintains a separate `latest_planner` symlink under
artifacts/optimized-prompts/ to avoid stomping on the action-selector's `latest`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_PARENT = _HERE.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

import gepa as gepa_pkg  # noqa: E402

from agent_opt import llm as _llm  # noqa: E402
from agent_opt.planner_adapter import PlannerAdapter  # noqa: E402
from agent_opt.reflection import REFLECTION_PROMPT_TEMPLATE  # noqa: E402
from agent_opt.seed_planner import SEED_PROMPT_PLANNER  # noqa: E402

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
DATA = ROOT / "data" / "planner_dataset.jsonl"
ART = ROOT / "artifacts" / "optimized-prompts"


def _load() -> list[dict]:
    if not DATA.exists():
        return []
    out = []
    with DATA.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _split(records: list[dict], frac: float, rng: random.Random) -> tuple[list, list]:
    rs = list(records); rng.shuffle(rs)
    n = max(1, int(len(rs) * frac))
    return rs[:n], rs[n:]


def _reflection_lm(model: str):
    def reflect(prompt: str) -> str:
        return _llm.chat(messages=[{"role": "user", "content": prompt}], model=model, max_tokens=4096, temperature=1.0)
    return reflect


def _eval(adapter: PlannerAdapter, candidate: dict, valset: list[dict]) -> float:
    eb = adapter.evaluate(valset, candidate, capture_traces=False)
    return (sum(eb.scores) / len(eb.scores)) if eb.scores else 0.0


def _replace_symlink(run_dir: Path) -> None:
    latest = ART / "latest_planner"
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
                st.write(s); st.flush()
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
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-model", default="claude-opus-4-7")
    p.add_argument("--reflection-model", default="claude-opus-4-7")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ART / f"planner_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "log.txt").open("w")
    sys.stdout = _Tee(sys.__stdout__, log_f); sys.stderr = _Tee(sys.__stderr__, log_f)

    t0 = time.time(); val_before = val_after = None; err: str | None = None
    try:
        print(f"[optimize_planner] run_dir={run_dir} args={vars(args)}")
        records = _load()
        if not records:
            raise RuntimeError(f"No records in {DATA}; run extract_planner_dataset.py first.")
        rng = random.Random(args.seed)
        trainset, valset = _split(records, args.train_frac, rng)
        if not valset:
            valset = trainset[-1:]; trainset = trainset[:-1] or trainset
        print(f"[optimize_planner] train={len(trainset)} val={len(valset)}")

        adapter = PlannerAdapter(task_model=args.task_model)
        seed = {"system": SEED_PROMPT_PLANNER}
        val_before = _eval(adapter, seed, valset)
        print(f"[optimize_planner] val_score_before={val_before:.4f}")

        result = gepa_pkg.optimize(
            seed_candidate=seed, trainset=trainset, valset=valset, adapter=adapter,
            reflection_lm=_reflection_lm(args.reflection_model),
            candidate_selection_strategy="pareto", skip_perfect_score=True, perfect_score=1.0,
            reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
            max_metric_calls=args.budget, seed=args.seed, raise_on_exception=True,
            run_dir=str(run_dir / "gepa_state"), display_progress_bar=False, cache_evaluation=True,
        )
        best = dict(result.best_candidate)
        try:
            val_after = float(result.val_aggregate_scores[result.best_idx])
        except Exception:
            val_after = _eval(adapter, best, valset)
        print(f"[optimize_planner] best_idx={result.best_idx} val_score_after={val_after:.4f}")
        (run_dir / "best_candidate.json").write_text(json.dumps(best, indent=2))
        (run_dir / "best_candidate.system.md").write_text(best.get("system", ""))
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("[optimize_planner] EXCEPTION:\n" + err)
    finally:
        meta = {"timestamp": ts, "elapsed_seconds": round(time.time() - t0, 2),
                "budget": args.budget, "train_frac": args.train_frac, "seed": args.seed,
                "task_model": args.task_model, "reflection_model": args.reflection_model,
                "val_score_before": val_before, "val_score_after": val_after,
                "delta": (None if (val_before is None or val_after is None) else round(val_after - val_before, 4)),
                "error": err, "run_dir": str(run_dir)}
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        with contextlib.suppress(Exception):
            _replace_symlink(run_dir)
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__; log_f.close()
    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
