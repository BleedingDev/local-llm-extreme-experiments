"""Unsupervised anomaly detector. Reuses the pre-fitted TF-IDF vectorizer at
``trace-gepa/artifacts/rag_index_v2/vectorizer.pkl`` (no re-fit) and fits a
novelty model (IsolationForest / LOF / OneClassSVM) on `label=good` records.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from agent_opt.rag.embed import _record_text

DEFAULT_VECTORIZER_PATH = "trace-gepa/artifacts/rag_index_v2/vectorizer.pkl"


@dataclass
class Detector:
    model: Any
    vectorizer: Any
    algo: str
    score_min: float
    score_max: float

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "Detector":
        with open(path, "rb") as f:
            return pickle.load(f)

    def _raw_score(self, X) -> np.ndarray:
        # decision_function: higher == more normal -> negate for "anomalous".
        return -self.model.decision_function(X)


def _subsample(X, cap, seed):
    if X.shape[0] <= cap:
        return X
    idx = np.random.default_rng(seed).choice(X.shape[0], size=cap, replace=False)
    return X[idx]


def train(records, algo: str = "iforest", vectorizer_path=DEFAULT_VECTORIZER_PATH,
          vectorizer=None, seed: int = 42) -> Detector:
    """Fit a novelty detector on records (assumed all label=good)."""
    if algo not in {"iforest", "lof", "ocsvm"}:
        raise ValueError(f"unknown algo: {algo}")
    if not records:
        raise ValueError("no records to train on")
    if vectorizer is None:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    X = vectorizer.transform([_record_text(r) for r in records])
    if algo == "iforest":
        model = IsolationForest(n_estimators=200, contamination="auto",
                                random_state=seed, n_jobs=-1).fit(X)
    elif algo == "lof":
        # k=10 won the held-out sweep on dataset_v2 (AUC 0.73 vs 0.65 at k=20).
        # LOF stores all training points; cap to keep pickle + scoring tractable.
        model = LocalOutlierFactor(n_neighbors=10, novelty=True, n_jobs=-1)
        model.fit(_subsample(X, 8000, seed))
    else:  # ocsvm — O(n^2), so cap aggressively.
        model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        model.fit(_subsample(X, 5000, seed))
    raw = -model.decision_function(X)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return Detector(model=model, vectorizer=vectorizer, algo=algo,
                    score_min=lo, score_max=hi)


def score(detector: Detector, record) -> float:
    """Return anomaly score in [0,1] (higher = more anomalous)."""
    text = record if isinstance(record, str) else _record_text(record)
    raw = float(detector._raw_score(detector.vectorizer.transform([text]))[0])
    norm = (raw - detector.score_min) / (detector.score_max - detector.score_min)
    return float(max(0.0, min(1.0, norm)))


def score_batch(detector: Detector, records) -> np.ndarray:
    """Vectorised raw scoring (un-clamped); higher = more anomalous."""
    return detector._raw_score(detector.vectorizer.transform(
        [_record_text(r) for r in records]))
