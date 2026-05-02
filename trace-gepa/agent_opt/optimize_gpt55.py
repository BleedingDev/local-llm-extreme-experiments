"""GPT-5.5 targeted GEPA optimisation.

Re-extracts records from gpt-5.5 codex sessions (none made it into dataset.jsonl),
splits into train/val, runs GEPA with seed prompt = SEED_PROMPT_CODEX. Falls back
to gpt-5.5 + gpt-5.4 combined if too few gpt-5.5 actionable records.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import tempfile
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
import orjson  # noqa: E402

from agent_opt.adapter import TraceAdapter  # noqa: E402
from agent_opt.optimize import _Tee, build_reflection_lm, evaluate_candidate, stratified_sample  # noqa: E402
from agent_opt.reflection import REFLECTION_PROMPT_TEMPLATE  # noqa: E402
from agent_opt.seed_codex import SEED_PROMPT_CODEX as SEED_PROMPT  # noqa: E402
from extractors import extract_codex as _ec  # noqa: E402
from extractors.categorize import recategorize  # noqa: E402

ROOT = Path("/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/trace-gepa")
MODELS = ROOT / "data" / "codex_session_models.json"
OUT_DATASET = ROOT / "data" / "codex_gpt55_dataset.jsonl"
ART = ROOT / "artifacts" / "optimized-prompts"
_orig_extract = _ec.extract_session


def _coerce(payload: dict) -> None:
    out = payload.get("output")
    if isinstance(out, list):
        parts = [
            "<image_omitted>" if isinstance(b, dict) and b.get("type") == "input_image"
            else (str(b.get("text") or "") if isinstance(b, dict) else str(b))
            for b in out
        ]
        payload["output"] = "\n".join(parts)
    elif out is None:
        payload["output"] = ""
    elif not isinstance(out, str):
        payload["output"] = str(out)


def extract_session(path: str, src_idx: int) -> list[dict]:
    safe = []
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                evt = orjson.loads(line)
            except Exception:
                continue
            payload = evt.get("payload") or {}
            if evt.get("type") == "response_item" and payload.get("type") == "function_call_output":
                _coerce(payload)
                evt["payload"] = payload
            safe.append(orjson.dumps(evt))
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", prefix="codex_safe_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"\n".join(safe))
        recs = _orig_extract(tmp, src_idx)
    finally:
        with contextlib.suppress(Exception):
            Path(tmp).unlink()
    sid = Path(path).stem.split("-")[-1][:8]
    for r in recs:
        r["src_path"] = path
        r["id"] = f"codex_{sid}_evt{r.get('src_event_idx', 0):06d}"
    return recs


def collect(max_sessions: int, fallback: bool) -> tuple[list[dict], dict]:
    models = json.loads(MODELS.read_text())
    targets = {"gpt-5.5"} | ({"gpt-5.4"} if fallback else set())
    paths = sorted(
        [p for p, m in models.items() if m in targets and Path(p).exists()],
        key=lambda p: -Path(p).stat().st_size,
    )[:max_sessions]
    print(f"[gpt55] extracting from {len(paths)} sessions (targets={sorted(targets)})")
    out: list[dict] = []
    for idx, p in enumerate(paths):
        out.extend(extract_session(p, idx))
    print(f"[gpt55] raw records: {len(out)}")
    recategorize(out)
    return out, {"sessions": len(paths), "records": len(out), "targets": sorted(targets)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--train-size", type=int, default=80)
    p.add_argument("--val-size", type=int, default=40)
    p.add_argument("--max-sessions", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-model", default="claude-opus-4-7")
    p.add_argument("--reflection-model", default="claude-opus-4-7")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ART / f"gpt55_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = (run_dir / "log.txt").open("w")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    t0 = time.time()
    val_before: float | None = None
    val_after: float | None = None
    err: str | None = None
    fallback_used = False
    extract_meta: dict = {}

    try:
        print(f"[gpt55] run_dir={run_dir}")
        print(f"[gpt55] args={vars(args)}")
        records, extract_meta = collect(args.max_sessions, fallback=False)
        actionable = [r for r in records if r.get("label") != "good"]
        print(f"[gpt55] gpt-5.5 actionable: {len(actionable)}")
        if len(actionable) < 60:
            print(f"[gpt55] FALLBACK: combining gpt-5.5 + gpt-5.4 (actionable={len(actionable)})")
            fallback_used = True
            records, extract_meta = collect(args.max_sessions, fallback=True)

        with OUT_DATASET.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[gpt55] wrote {len(records)} records to {OUT_DATASET}")

        rng = random.Random(args.seed)
        rng.shuffle(records)
        n = len(records)
        n_val = min(args.val_size * 2, max(args.val_size, n // 4))
        valpool, trainpool = records[:n_val], records[n_val:]
        trainset = stratified_sample(trainpool, min(args.train_size, len(trainpool)), rng)
        valset = stratified_sample(valpool, min(args.val_size, len(valpool)), rng)

        def cat(rs):
            d: dict[str, int] = defaultdict(int)
            for r in rs:
                d[str(r.get("failure_category"))] += 1
            return dict(d)

        print(f"[gpt55] train={len(trainset)} val={len(valset)}")
        print(f"[gpt55] train cat={cat(trainset)}")
        print(f"[gpt55] val cat={cat(valset)}")

        adapter = TraceAdapter(task_model=args.task_model)
        seed_candidate = {"system": SEED_PROMPT}
        print("[gpt55] baseline val on seed prompt")
        val_before = evaluate_candidate(adapter, seed_candidate, valset)
        print(f"[gpt55] val_before={val_before:.4f}")

        result = gepa_pkg.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=build_reflection_lm(args.reflection_model),
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
        print(f"[gpt55] gepa returned in {time.time() - t0:.1f}s")
        best = dict(result.best_candidate)
        try:
            val_after = float(result.val_aggregate_scores[result.best_idx])
        except Exception:
            val_after = evaluate_candidate(adapter, best, valset)
        print(f"[gpt55] best_idx={result.best_idx} val_after={val_after:.4f}")
        (run_dir / "best_candidate.json").write_text(json.dumps(best, indent=2))
        (run_dir / "best_candidate.system.md").write_text(best.get("system", ""))
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("[gpt55] EXCEPTION:\n" + err)
    finally:
        meta = {
            "track": "gpt55",
            "timestamp": ts,
            "elapsed_seconds": round(time.time() - t0, 2),
            "fallback_used": fallback_used,
            "extract_meta": extract_meta,
            "args": vars(args),
            "val_score_before": val_before,
            "val_score_after": val_after,
            "delta": (None if (val_before is None or val_after is None) else round(val_after - val_before, 4)),
            "error": err,
            "run_dir": str(run_dir),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_f.close()

    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
