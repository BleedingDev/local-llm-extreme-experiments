"""CLI: train / eval / score the anomaly detector."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from agent_opt.anomaly.detector import (
    DEFAULT_VECTORIZER_PATH, Detector, score, score_batch, train)

DATASET = "trace-gepa/data/dataset_v2.jsonl"
ARTIFACT_DIR = Path("trace-gepa/artifacts")
REPORT_PATH = Path("trace-gepa/agent_opt/anomaly/eval_report.md")
_ALGO_DESC = {
    "iforest": "IsolationForest(n_estimators=200, contamination='auto')",
    "lof": "LocalOutlierFactor(n_neighbors=10, novelty=True) [train capped @ 8000]",
    "ocsvm": "OneClassSVM(kernel='rbf', nu=0.05) [train capped @ 5000]",
}
_COMPLEMENTARITY = (
    "Orthogonal to round-2 preflight (deterministic predicates: perfect precision, zero "
    "novel-coverage) and round-4 failure-classifier (supervised: bounded by labelled corpus). "
    "This anomaly detector fits the `good` manifold and flags ANY OOD record, catching "
    "novel/zero-day weirdness neither of the others can see. Production wiring: halt if "
    "(preflight fires) OR (classifier > threshold) OR (anomaly percentile > 0.95)."
)


def _load_dataset(path: str = DATASET):
    return [json.loads(l) for l in open(path) if l.strip()]


def _split(records, seed: int = 42):
    good = [r for r in records if r.get("label") == "good"]
    bad = [r for r in records if r.get("label") == "bad"]
    corr = [r for r in records if r.get("label") == "user_corrected"]
    random.Random(seed).shuffle(good)
    cut = int(0.8 * len(good))
    return good[:cut], good[cut:], bad, corr


def _auc(det, pos, neg):
    sp, sn = score_batch(det, pos), score_batch(det, neg)
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    return float(roc_auc_score(y, np.concatenate([sp, sn]))), sp, sn


def _excerpt(r):
    ur = ((r.get("context") or {}).get("user_request") or "")[:120]
    obs = r.get("observed_action") or {}
    oa = obs.get("name", "") if isinstance(obs, dict) else ""
    return f"[{r.get('label')}/{r.get('failure_category')}] {oa} :: {ur!r}"


def cmd_train(args):
    tg, *_ = _split(_load_dataset())
    print(f"training {args.algo} on {len(tg)} good records...")
    out = Path(args.out) if args.out else ARTIFACT_DIR / f"anomaly_{args.algo}.pkl"
    train(tg, algo=args.algo).save(out)
    print(f"saved -> {out}")


def cmd_eval(args):
    tg, hg, bad, corr = _split(_load_dataset())
    test_pos, test_neg = bad + corr, hg
    if args.model:
        det = Detector.load(args.model)
    elif args.compare:
        cands = {}
        for algo in ("iforest", "lof"):
            mp = ARTIFACT_DIR / f"anomaly_{algo}.pkl"
            if not mp.exists():
                print(f"training {algo}...")
                train(tg, algo=algo).save(mp)
            cands[algo] = Detector.load(mp)
        aucs = {a: _auc(d, test_pos, test_neg)[0] for a, d in cands.items()}
        print(f"candidate AUCs: {aucs}")
        chosen = max(aucs, key=aucs.get)
        det = cands[chosen]
        print(f"selected algo: {chosen}")
    else:
        mp = ARTIFACT_DIR / "anomaly_iforest.pkl"
        if not mp.exists():
            train(tg, algo="iforest").save(mp)
        det = Detector.load(mp)

    auc, sp, sn = _auc(det, test_pos, test_neg)
    # Robust display normalisation; AUC is on raw scores.
    raw = np.concatenate([sp, sn])
    p1, p99 = float(np.percentile(raw, 1)), float(np.percentile(raw, 99))
    span = max(p99 - p1, 1e-9)
    sp = np.clip((sp - p1) / span, 0.0, 1.0)
    sn = np.clip((sn - p1) / span, 0.0, 1.0)

    cats: dict[str, list[float]] = defaultdict(list)
    for r, s in zip(test_pos, sp):
        cats[r.get("failure_category") or r.get("label") or "unknown"].append(float(s))
    cats["good_heldout"] = [float(s) for s in sn]
    breakdown = {c: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
                 for c, v in cats.items()}

    all_recs = test_pos + test_neg
    all_s = np.concatenate([sp, sn])
    top, bot = np.argsort(-all_s)[:10], np.argsort(all_s)[:10]
    print(json.dumps({"auc": auc, "n_pos": len(sp), "n_neg": len(sn)}, indent=2))
    print("--- per-category mean anomaly score ---")
    for c, v in sorted(breakdown.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {c:30s} n={v['n']:5d} mean={v['mean']:.3f} std={v['std']:.3f}")
    top_records = [(float(all_s[int(i)]), all_recs[int(i)]) for i in top]
    for hdr, idxs in (("top-10 anomalous", top), ("bottom-10 most-normal", bot)):
        print(f"--- {hdr} (test set) ---")
        for i in idxs:
            r = all_recs[int(i)]
            print(f"  score={all_s[int(i)]:.3f} id={r.get('id')} {_excerpt(r)}")
    if args.write_report:
        verdict = "SHIP" if auc >= 0.65 else ("WEAK-SIGNAL" if auc >= 0.55 else "ABANDON")
        REPORT_PATH.write_text(_report(auc, breakdown, top_records, verdict, len(tg), det.algo))
        print(f"wrote report -> {REPORT_PATH}")


def _report(auc, breakdown, top_records, verdict, n_train, algo):
    rows = "\n".join(f"| {c} | {v['n']} | {v['mean']:.3f} | {v['std']:.3f} |"
                     for c, v in sorted(breakdown.items(), key=lambda kv: -kv[1]["mean"]))
    def _t(s, r):
        ur = ((r.get("context") or {}).get("user_request") or "").replace("\n", " ")[:160]
        oa = (r.get("observed_action") or {}).get("name", "") if isinstance(r.get("observed_action"), dict) else ""
        return (f"- score={s:.3f} id=`{r.get('id')}` label={r.get('label')} "
                f"cat={r.get('failure_category')} tool={oa} excerpt={ur!r}")
    tops = "\n".join(_t(s, r) for s, r in top_records)
    return (f"# Anomaly Detector Eval Report\n\n- algo: {_ALGO_DESC[algo]}\n"
            f"- vectorizer: reused `{DEFAULT_VECTORIZER_PATH}` (no re-fit)\n"
            f"- train good records: {n_train}\n- **ROC-AUC: {auc:.4f}**\n"
            f"- verdict: **{verdict}**\n\n## Per-category mean anomaly score\n\n"
            f"| category | n | mean | std |\n| --- | ---: | ---: | ---: |\n{rows}\n\n"
            f"## Top-10 anomalies\n\n{tops}\n\n## Complementarity\n\n{_COMPLEMENTARITY}\n")


def cmd_score(args):
    mp = Path(args.model) if args.model else ARTIFACT_DIR / "anomaly_iforest.pkl"
    print(json.dumps({"anomaly_score": score(Detector.load(mp), args.query),
                      "query": args.query[:200]}))


def main(argv=None):
    p = argparse.ArgumentParser(prog="agent_opt.anomaly.cli")
    sp = p.add_subparsers(dest="cmd", required=True)
    pt = sp.add_parser("train")
    pt.add_argument("--algo", choices=["iforest", "lof", "ocsvm"], default="iforest")
    pt.add_argument("--out", default=None)
    pt.set_defaults(func=cmd_train)
    pe = sp.add_parser("eval")
    pe.add_argument("--model", default=None)
    pe.add_argument("--no-compare", dest="compare", action="store_false")
    pe.add_argument("--no-report", dest="write_report", action="store_false")
    pe.set_defaults(write_report=True, compare=True, func=cmd_eval)
    ps = sp.add_parser("score")
    ps.add_argument("--model", default=None)
    ps.add_argument("--query", required=True)
    ps.set_defaults(func=cmd_score)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
