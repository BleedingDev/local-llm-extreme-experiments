"""TF-IDF index builder for trace records.

Reads JSONL trace datasets, drops orchestration boilerplate by default,
augments each doc with observed action + corrective signals, fits a
TfidfVectorizer, and writes:
  - tfidf_matrix.npz   (sparse CSR, [N, V])
  - vectorizer.pkl     (sklearn TfidfVectorizer, fitted)
  - metadata.jsonl     (one JSON line per record, in same order)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


_ORCH_PREFIXES = (
    "<teammate-message",
    "<task-notification",
    "<system-reminder",
    "<command-name>",
    "<command-message>",
)


def _action_to_text(act) -> str:
    if not isinstance(act, dict):
        return str(act)[:200]
    name = act.get("name") or act.get("kind") or ""
    inp = act.get("input") or act.get("arguments")
    if isinstance(inp, str):
        return f"{name}({inp[:240]})"
    if isinstance(inp, dict):
        return f"{name}({json.dumps(inp)[:240]})"
    return name or json.dumps(act)[:200]


def _record_text(rec: dict, last_k: int = 3) -> str:
    ctx = rec.get("context") or {}
    user_request = (ctx.get("user_request") or "")[:1500]
    actions = ctx.get("recent_actions") or []
    actions_text = " | ".join(_action_to_text(a) for a in actions[-last_k:])
    obs = rec.get("observed_action") or {}
    obs_text = _action_to_text(obs) if obs else ""
    next_user = (rec.get("next_user_message") or "")[:600]
    cat = rec.get("failure_category") or ""
    label = rec.get("label") or ""
    parts = [
        user_request,
        actions_text,
        obs_text,
        f"next_user: {next_user}" if next_user else "",
        f"failure_category: {cat}" if cat else "",
        f"label: {label}" if label else "",
    ]
    return "\n".join(p for p in parts if p)


def _is_orchestration(rec: dict) -> bool:
    ur = ((rec.get("context") or {}).get("user_request") or "").lstrip()
    if not ur:
        return True
    return any(ur.startswith(p) for p in _ORCH_PREFIXES)


def _iter_records(paths: list[str], limit: int | None, drop_orchestration: bool) -> Iterator[dict]:
    n = 0
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if drop_orchestration and _is_orchestration(rec):
                    continue
                yield rec
                n += 1
                if limit is not None and n >= limit:
                    return


def build_index(
    dataset_paths: list[str],
    output_dir: str,
    limit: int | None = None,
    drop_orchestration: bool = True,
    min_df: int = 2,
    max_df: float = 0.9,
    ngram_range: tuple = (1, 2),
    max_features: int = 50000,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs: list[str] = []
    metas: list[dict] = []
    for rec in _iter_records(dataset_paths, limit, drop_orchestration):
        docs.append(_record_text(rec))
        metas.append(
            {
                "id": rec.get("id"),
                "src": rec.get("src"),
                "src_path": rec.get("src_path"),
                "src_event_idx": rec.get("src_event_idx"),
                "label": rec.get("label"),
                "failure_category": rec.get("failure_category"),
                "observed_tool": (rec.get("observed_action") or {}).get("name"),
                "user_request_excerpt": (((rec.get("context") or {}).get("user_request")) or "")[:200],
                "next_user_message_excerpt": (rec.get("next_user_message") or "")[:200],
            }
        )

    if not docs:
        raise RuntimeError("no records found")

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(docs)
    matrix = sparse.csr_matrix(matrix)

    sparse.save_npz(out / "tfidf_matrix.npz", matrix)
    with (out / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vec, f)
    with (out / "metadata.jsonl").open("w") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return {
        "n_records": matrix.shape[0],
        "vocab_size": matrix.shape[1],
        "nnz": int(matrix.nnz),
        "drop_orchestration": drop_orchestration,
        "output_dir": str(out),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output", default="trace-gepa/artifacts/rag_index")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keep-orchestration", action="store_true")
    args = p.parse_args()
    info = build_index(args.datasets, args.output, limit=args.limit, drop_orchestration=not args.keep_orchestration)
    print(json.dumps(info, indent=2))
