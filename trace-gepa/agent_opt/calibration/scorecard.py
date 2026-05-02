"""Per-tool calibration scorecard (proposal: calibration_scorecard.md)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _gold_map(tasks_jsonl: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in _load_jsonl(tasks_jsonl):
        tid, tool = t.get("id"), (t.get("expected") or {}).get("primary_action", {}).get("tool_name")
        if tid and tool:
            out[tid] = tool
    return out


def _predicted_tool(row: dict[str, Any]) -> str | None:
    code = row.get("code") or row.get("content")
    if not code:
        return None
    try:
        return json.loads(code).get("tool_name")
    except (ValueError, TypeError):
        return None


def _score(row: dict[str, Any]) -> float:
    rec = row.get("record") or {}
    s = rec.get("check_score")
    return float(s) if s is not None else (1.0 if rec.get("ok") else 0.0)


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def compute_scorecard(results_jsonl: Path, tasks_jsonl: Path) -> dict[str, Any]:
    """Compute per-tool precision/recall/over-pick stats for one model run."""
    gold = _gold_map(Path(tasks_jsonl))
    rows = _load_jsonl(Path(results_jsonl))
    pred_count: Counter[str] = Counter()
    exp_count: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    n_total = n_correct = 0
    for row in rows:
        tid = (row.get("task") or {}).get("id")
        if not tid or tid not in gold:
            continue
        n_total += 1
        g, p, s = gold[tid], _predicted_tool(row), _score(row)
        exp_count[g] += 1
        if p:
            pred_count[p] += 1
        if p == g and s > 0:
            correct[g] += 1
            n_correct += 1
        elif p and p != g:
            confusions[g][p] += 1  # gold tool missed -> what model picked
    per_tool: dict[str, dict[str, Any]] = {}
    for t in sorted(set(pred_count) | set(exp_count)):
        pc, ec, c = pred_count.get(t, 0), exp_count.get(t, 0), correct.get(t, 0)
        precision = (c / pc) if pc else None
        recall = (c / ec) if ec else None
        f1 = _f1(precision or 0.0, recall or 0.0) if (precision is not None and recall is not None) else None
        per_tool[t] = {
            "predicted_count": pc, "expected_count": ec, "correct_picks": c,
            "over_pick_rate": ((pc - ec) / n_total) if n_total else 0.0,
            "precision": precision, "recall": recall, "f1": f1,
            "confused_with": confusions[t].most_common(3),
        }
    return {
        "n_total": n_total, "n_correct": n_correct,
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "per_tool": per_tool,
    }


def compute_cross_model(results_paths: list[Path], tasks_jsonl: Path) -> dict[str, Any]:
    """Compute per-model scorecards plus a tool×model pivot and strengths."""
    models: dict[str, dict[str, Any]] = {}
    for p in results_paths:
        sc = compute_scorecard(Path(p), Path(tasks_jsonl))
        rows = _load_jsonl(Path(p))
        name = ((rows[0] if rows else {}).get("run") or {}).get("run_id") or Path(p).stem
        models[name] = sc
    pivot: dict[str, dict[str, dict[str, Any]]] = {}
    for t in sorted({tt for sc in models.values() for tt in sc["per_tool"]}):
        pivot[t] = {
            name: {k: sc["per_tool"].get(t, {}).get(k, (0 if k.endswith("_count") else (0.0 if k == "over_pick_rate" else None)))
                   for k in ("over_pick_rate", "precision", "recall", "f1", "predicted_count", "expected_count")}
            for name, sc in models.items()
        }
    strengths: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for t, by_model in pivot.items():
        scored = sorted(((n, by_model[n].get("f1") or 0.0) for n in models), key=lambda kv: kv[1], reverse=True)
        if len(scored) >= 2 and (scored[0][1] - scored[1][1]) >= 0.2:
            strengths[scored[0][0]].append({"tool": t, "f1": scored[0][1], "runner_up_f1": scored[1][1]})
    return {"models": models, "pivot": pivot, "strengths": strengths}


# ---------- markdown rendering ----------

def _fmt_pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:+.1f}%"


def _fmt_num(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def render_single_md(scorecard: dict[str, Any], title: str) -> str:
    L = [
        f"# {title}", "",
        f"**N tasks:** {scorecard['n_total']}  |  **Correct:** {scorecard['n_correct']}  |  **Accuracy:** {scorecard['accuracy'] * 100:.1f}%", "",
        "Note: scorecard accuracy may differ from the bench headline pass rate. A row counts as correct only when "
        "`predicted_tool == gold.primary_action.tool_name AND check_score > 0`; the bench verifier can be permissive "
        "(accept a tool set or `must_avoid` rules), which the scorecard does not credit.", "",
        "Sorted by |over_pick_rate| desc. Rows with expected_count < 5 are low-N (noisy).", "",
        "| Tool | Pred | Exp | Over% | Precision | Recall | F1 | Confused-with (top3) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tool, m in sorted(scorecard["per_tool"].items(), key=lambda kv: abs(kv[1]["over_pick_rate"]), reverse=True):
        conf = ", ".join(f"{c}({n})" for c, n in m["confused_with"]) or "-"
        low_n = "*" if m["expected_count"] < 5 else ""
        L.append(
            f"| {tool}{low_n} | {m['predicted_count']} | {m['expected_count']} | "
            f"{_fmt_pct(m['over_pick_rate'])} | {_fmt_num(m['precision'])} | "
            f"{_fmt_num(m['recall'])} | {_fmt_num(m['f1'])} | {conf} |"
        )
    L += ["", "`*` = expected_count < 5 (low-N, treat with caution)."]
    return "\n".join(L) + "\n"


def render_cross_md(cross: dict[str, Any], title: str) -> str:
    models = list(cross["models"].keys())
    L = [f"# {title}", "", "Models compared: " + ", ".join(f"`{m}`" for m in models), ""]
    for m in models:
        sc = cross["models"][m]
        L.append(f"- `{m}`: accuracy {sc['accuracy'] * 100:.1f}% ({sc['n_correct']}/{sc['n_total']})")
    items = sorted(
        cross["pivot"].items(),
        key=lambda kv: max(abs(kv[1][m]["over_pick_rate"]) for m in models),
        reverse=True,
    )
    L += ["", "## Over-pick rate by tool x model (signed)", ""]
    L.append("| " + " | ".join(["Tool", "Exp"] + models) + " |")
    L.append("|" + "|".join(["---"] * (len(models) + 2)) + "|")
    for tool, by_model in items:
        any_exp = max(by_model[m]["expected_count"] for m in models)
        L.append("| " + " | ".join([tool, str(any_exp)] + [_fmt_pct(by_model[m]["over_pick_rate"]) for m in models]) + " |")
    L += ["", "## F1 by tool x model", ""]
    L.append("| " + " | ".join(["Tool"] + models) + " |")
    L.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for tool, by_model in items:
        L.append("| " + " | ".join([tool] + [_fmt_num(by_model[m]["f1"]) for m in models]) + " |")
    L += ["", "## Model strengths (F1 lead >= 0.20 vs runner-up)", ""]
    for name, lst in cross["strengths"].items():
        if not lst:
            L.append(f"- `{name}`: none")
        else:
            parts = ", ".join(f"{r['tool']} ({r['f1']:.2f} vs {r['runner_up_f1']:.2f})" for r in lst)
            L.append(f"- `{name}`: {parts}")
    return "\n".join(L) + "\n"
