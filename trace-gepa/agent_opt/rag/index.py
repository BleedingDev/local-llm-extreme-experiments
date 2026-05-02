"""TraceIndex: cosine-similarity retrieval + MMR diversity reranker."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


class TraceIndex:
    def __init__(self, index_dir: str):
        d = Path(index_dir)
        self.matrix = sparse.load_npz(d / "tfidf_matrix.npz").tocsr()
        with (d / "vectorizer.pkl").open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.metadata: list[dict] = []
        with (d / "metadata.jsonl").open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        self._matrix_norm = normalize(self.matrix, norm="l2", axis=1)

    def query(
        self,
        text: str,
        k: int = 8,
        mmr_lambda: float = 0.7,
        candidate_pool: int = 64,
        dedupe_by: str | None = "src_path",
    ) -> list[dict]:
        q = normalize(self.vectorizer.transform([text]), norm="l2", axis=1)
        scores = (self._matrix_norm @ q.T).toarray().ravel()
        if candidate_pool >= len(scores):
            cand_idx = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, candidate_pool)[:candidate_pool]
            cand_idx = top[np.argsort(-scores[top])]

        # MMR + optional path-dedup
        chosen: list[int] = []
        chosen_paths: set[str] = set()
        cand = list(cand_idx)
        sub = self._matrix_norm[cand]
        sub_dense_sims = (sub @ sub.T).toarray()
        idx_to_pos = {idx: pos for pos, idx in enumerate(cand)}

        while cand and len(chosen) < k:
            best_score = -1e9
            best_pos = -1
            for pos, idx in enumerate(cand):
                meta = self.metadata[idx]
                if dedupe_by and (sp := meta.get(dedupe_by)) and sp in chosen_paths:
                    continue
                rel = float(scores[idx])
                if not chosen:
                    mmr = rel
                else:
                    chosen_pos = [idx_to_pos[c] for c in chosen]
                    div = float(sub_dense_sims[idx_to_pos[idx], chosen_pos].max())
                    mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * div
                if mmr > best_score:
                    best_score = mmr
                    best_pos = pos
            if best_pos < 0:
                # all remaining candidates dedup-blocked; fall back to ignoring dedup
                break
            picked = cand.pop(best_pos)
            chosen.append(picked)
            sp = self.metadata[picked].get(dedupe_by) if dedupe_by else None
            if sp:
                chosen_paths.add(sp)

        results = []
        for rank, idx in enumerate(chosen):
            meta = dict(self.metadata[idx])
            meta["similarity"] = float(scores[idx])
            meta["rank"] = rank + 1
            results.append(meta)
        return results
