"""POC runner for the session-replay oracle (50 records, serial).

Run as: python -m agent_opt.oracle._run_poc
Writes: artifacts/oracle_poc_results.json (or stdout dump).
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_opt.oracle.session_judge import (  # noqa: E402
    score_action_via_followup,
    score_observed,
    synthesize_ideal_answer,
)

DATASET = _ROOT / "data" / "dataset.jsonl"
OUT = _ROOT / "artifacts" / "oracle_poc_results.json"
TARGET = {"good": 25, "bad": 15, "user_confirmed": 5, "user_corrected": 5}
LABEL_TO_BINARY = {"good": 1, "user_confirmed": 1, "bad": 0, "user_corrected": 0}


def _stratified_sample(rng: random.Random) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with DATASET.open() as f:
        for line in f:
            r = json.loads(line)
            if not r.get("next_user_message"):
                continue
            lbl = r.get("label")
            if lbl in TARGET:
                buckets[lbl].append(r)
    sampled: list[dict] = []
    for lbl, n in TARGET.items():
        pool = buckets[lbl]
        # Stratify by failure_category as best-effort.
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_cat[r.get("failure_category") or "_none"].append(r)
        for cat in by_cat:
            rng.shuffle(by_cat[cat])
        # Round-robin pull across categories.
        cats = list(by_cat.keys())
        i = 0
        chosen: list[dict] = []
        while len(chosen) < n and any(by_cat[c] for c in cats):
            cat = cats[i % len(cats)]
            if by_cat[cat]:
                chosen.append(by_cat[cat].pop())
            i += 1
        sampled.extend(chosen)
    return sampled


def _weighted_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Macro-weighted F1 over the two binary classes."""
    cls = sorted(set(y_true))
    f1s = []
    weights = []
    for c in cls:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        weights.append(y_true.count(c))
    total = sum(weights) or 1
    return sum(f * w for f, w in zip(f1s, weights)) / total


def main() -> None:
    rng = random.Random(2026)
    sample = _stratified_sample(rng)
    print(f"sampled {len(sample)} records: {Counter(r['label'] for r in sample)}", flush=True)
    cat_counts = Counter(r.get("failure_category") for r in sample)
    print(f"by category: {dict(cat_counts)}", flush=True)

    results: list[dict] = []
    t0 = time.time()
    for i, rec in enumerate(sample):
        # Pass 1: judge on the original observed_action (calibration).
        cal = score_observed(rec)
        # Pass 2: judge on the synthesised "ideal" action.
        ideal = synthesize_ideal_answer(rec)
        syn = score_action_via_followup(rec, ideal)
        results.append(
            {
                "id": rec["id"],
                "label": rec["label"],
                "failure_category": rec.get("failure_category"),
                "calib_score": cal["score"],
                "calib_raw": cal["raw"][:80],
                "calib_latency": round(cal.get("latency", 0.0), 2),
                "synth_score": syn["score"],
                "synth_raw": syn["raw"][:80],
                "ideal_tool": ideal["tool_name"],
            }
        )
        elapsed = time.time() - t0
        print(
            f"[{i+1}/{len(sample)}] {rec['id']} label={rec['label']} "
            f"calib={cal['score']} synth={syn['score']} ({elapsed:.1f}s)",
            flush=True,
        )

    # Calibration F1.
    pairs = [(r, LABEL_TO_BINARY[r["label"]]) for r in results if r["calib_score"] is not None]
    y_true = [p[1] for p in pairs]
    y_pred = [1 if p[0]["calib_score"] >= 0.5 else 0 for p in pairs]
    f1 = _weighted_f1(y_true, y_pred) if pairs else 0.0

    # Per-label distributions.
    by_label: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r["calib_score"] is not None:
            by_label[r["label"]].append(r["calib_score"])
    label_stats = {
        lbl: {
            "n": len(v),
            "mean": round(statistics.mean(v), 3) if v else None,
            "std": round(statistics.pstdev(v), 3) if len(v) > 1 else 0.0,
        }
        for lbl, v in by_label.items()
    }
    synth_hist = Counter(r["synth_score"] for r in results)

    summary = {
        "n_total": len(results),
        "n_scored": len(pairs),
        "calibration_weighted_f1": round(f1, 3),
        "label_score_stats": label_stats,
        "synth_hist": {str(k): v for k, v in synth_hist.items()},
        "category_counts": dict(cat_counts),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
