"""Unit tests for per-tool calibration scorecard."""
from __future__ import annotations

import json
from pathlib import Path

from agent_opt.calibration.scorecard import (
    compute_cross_model,
    compute_scorecard,
    render_cross_md,
    render_single_md,
)


def _write_tasks(p: Path, tasks: list[tuple[str, str]]) -> None:
    lines = []
    for tid, tool in tasks:
        lines.append(json.dumps({
            "id": tid,
            "expected": {"primary_action": {"tool_name": tool}},
        }))
    p.write_text("\n".join(lines))


def _write_results(p: Path, run_id: str, rows: list[tuple[str, str, float]]) -> None:
    lines = []
    for tid, predicted, score in rows:
        lines.append(json.dumps({
            "run": {"run_id": run_id},
            "task": {"id": tid},
            "record": {"check_score": score, "ok": bool(score > 0)},
            "code": json.dumps({"tool_name": predicted}),
        }))
    p.write_text("\n".join(lines))


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    tasks = tmp_path / "tasks.jsonl"
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    # 3 tasks: t0->Read, t1->Bash, t2->Edit
    _write_tasks(tasks, [("t0", "Read"), ("t1", "Bash"), ("t2", "Edit")])
    # model A: picks Read, Read, Edit -> Read 1/2 correct (t0), Bash missed, Edit correct
    _write_results(a, "model_a", [
        ("t0", "Read", 1.0),
        ("t1", "Read", 0.0),
        ("t2", "Edit", 1.0),
    ])
    # model B: picks Read, Bash, Read -> Read missed t2 (predicted Read but gold Edit -> wrong)
    _write_results(b, "model_b", [
        ("t0", "Read", 1.0),
        ("t1", "Bash", 1.0),
        ("t2", "Read", 0.0),
    ])
    return tasks, a, b


def test_basic_counts(tmp_path: Path) -> None:
    tasks, a, _ = _setup(tmp_path)
    sc = compute_scorecard(a, tasks)
    assert sc["n_total"] == 3
    assert sc["n_correct"] == 2  # Read t0 + Edit t2
    pt = sc["per_tool"]
    assert pt["Read"]["predicted_count"] == 2
    assert pt["Read"]["expected_count"] == 1
    assert pt["Read"]["correct_picks"] == 1
    assert pt["Bash"]["predicted_count"] == 0
    assert pt["Bash"]["expected_count"] == 1
    assert pt["Edit"]["predicted_count"] == 1
    assert pt["Edit"]["correct_picks"] == 1


def test_precision_recall_f1_and_overpick(tmp_path: Path) -> None:
    tasks, a, _ = _setup(tmp_path)
    sc = compute_scorecard(a, tasks)
    pt = sc["per_tool"]
    # Read: predicted 2, expected 1, correct 1 -> P=0.5, R=1.0, F1=2/3
    assert abs(pt["Read"]["precision"] - 0.5) < 1e-9
    assert abs(pt["Read"]["recall"] - 1.0) < 1e-9
    assert abs(pt["Read"]["f1"] - (2 / 3)) < 1e-9
    # over-pick Read: (2-1)/3 = +0.3333
    assert abs(pt["Read"]["over_pick_rate"] - (1 / 3)) < 1e-9
    # Bash: P=None (no predictions), R=0.0
    assert pt["Bash"]["precision"] is None
    assert pt["Bash"]["recall"] == 0.0
    # Edit: P=1, R=1, F1=1
    assert abs(pt["Edit"]["f1"] - 1.0) < 1e-9


def test_confusion_matrix(tmp_path: Path) -> None:
    tasks, a, _ = _setup(tmp_path)
    sc = compute_scorecard(a, tasks)
    # Bash was missed; model A picked Read instead -> Bash confused-with Read
    confused = dict(sc["per_tool"]["Bash"]["confused_with"])
    assert confused.get("Read") == 1


def test_cross_model_pivot_and_strengths(tmp_path: Path) -> None:
    tasks, a, b = _setup(tmp_path)
    cx = compute_cross_model([a, b], tasks)
    assert set(cx["models"].keys()) == {"model_a", "model_b"}
    assert "Bash" in cx["pivot"]
    # model_b got Bash right (P=R=F1=1.0); model_a never predicted Bash so F1 is None
    assert cx["pivot"]["Bash"]["model_b"]["f1"] == 1.0
    assert cx["pivot"]["Bash"]["model_a"]["f1"] is None
    # strengths: model_b leads on Bash by >= 0.2
    bash_strength = [s for s in cx["strengths"]["model_b"] if s["tool"] == "Bash"]
    assert bash_strength and bash_strength[0]["f1"] - bash_strength[0]["runner_up_f1"] >= 0.2


def test_renderers_emit_markdown(tmp_path: Path) -> None:
    tasks, a, b = _setup(tmp_path)
    sc = compute_scorecard(a, tasks)
    md1 = render_single_md(sc, "title-a")
    assert "title-a" in md1 and "Read" in md1 and "F1" in md1
    cx = compute_cross_model([a, b], tasks)
    md2 = render_cross_md(cx, "cross-title")
    assert "cross-title" in md2 and "model_a" in md2 and "model_b" in md2
