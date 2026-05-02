"""Unit tests for the anomaly detector."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_opt.anomaly.detector import Detector, score, score_batch, train


def _rec(text: str) -> dict:
    return {"id": text[:8], "label": "good",
            "context": {"user_request": text, "recent_actions": []},
            "observed_action": {"name": "Bash", "input": {"command": "ls"}}}


@pytest.fixture(scope="module")
def vectorizer():
    corpus = [
        "list files in directory ls", "read python file open",
        "edit configuration file vim", "run unit tests pytest",
        "git status check repo", "build the project with cargo",
        "deploy the application to production", "review changes in pull request",
        "format code with prettier", "install dependencies via npm",
        "quantum chromodynamics tensor decomposition spectroscopy",
        "zzzzzzz qqqqqq xxxxxx vvvvvv kkkkk wwwww",
    ] * 5
    v = TfidfVectorizer(min_df=1, ngram_range=(1, 2)); v.fit(corpus); return v


@pytest.fixture(scope="module")
def trained(vectorizer):
    rs = [_rec("list files in directory ls"), _rec("read python file open"),
          _rec("edit configuration file vim"), _rec("run unit tests pytest"),
          _rec("git status check repo")] * 8
    return train(rs, algo="iforest", vectorizer=vectorizer)


def test_train_returns_detector(trained):
    assert isinstance(trained, Detector)
    assert trained.algo == "iforest"
    assert trained.score_max > trained.score_min


def test_score_in_unit_interval(trained):
    s = score(trained, _rec("list files in directory ls"))
    assert isinstance(s, float) and 0.0 <= s <= 1.0


def test_ood_scores_higher_than_id(vectorizer):
    rs = ([_rec("list files in directory ls")] * 8 + [_rec("git status check repo")] * 8
          + [_rec("run unit tests pytest")] * 8 + [_rec("read python file open")] * 8
          + [_rec("edit configuration file vim")] * 8)
    det = train(rs, algo="lof", vectorizer=vectorizer)
    id_s = score_batch(det, [_rec("list files in directory ls"),
                             _rec("git status check repo"),
                             _rec("run unit tests pytest")])
    ood = score_batch(det, [_rec("quantum chromodynamics tensor decomposition spectroscopy"),
                            _rec("zzzzzzz qqqqqq xxxxxx vvvvvv kkkkk wwwww")])
    assert float(np.mean(ood)) > float(np.mean(id_s))


def test_pickle_roundtrip(trained, tmp_path):
    p = tmp_path / "det.pkl"; trained.save(p)
    loaded = Detector.load(p)
    assert loaded.algo == trained.algo and loaded.score_min == trained.score_min
    r = _rec("git status check repo")
    assert abs(score(trained, r) - score(loaded, r)) < 1e-9


def test_score_accepts_string_query(trained):
    s = score(trained, "git status check repo")
    assert isinstance(s, float) and 0.0 <= s <= 1.0


def test_train_rejects_unknown_algo(vectorizer):
    with pytest.raises(ValueError):
        train([_rec("hi")], algo="bogus", vectorizer=vectorizer)


def test_train_rejects_empty(vectorizer):
    with pytest.raises(ValueError):
        train([], algo="iforest", vectorizer=vectorizer)


def test_lof_algo_works(vectorizer):
    rs = [_rec(f"text sample number {i} ls files") for i in range(40)]
    det = train(rs, algo="lof", vectorizer=vectorizer)
    assert det.algo == "lof"
    assert 0.0 <= score(det, "text sample number 1 ls files") <= 1.0
